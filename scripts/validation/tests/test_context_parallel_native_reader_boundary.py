"""Exercise actual official-backend methods with disjoint logical/device rows."""
import ast
import __future__
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
RELATIVE = 'python/sglang/srt/layers/attention/dsa_backend.py'


def patched_tree(relative):
    source_root = os.environ.get('HISPARSE_NATIVE_SOURCE_ROOT')
    if not source_root:
        pytest.skip('Set HISPARSE_NATIVE_SOURCE_ROOT to the exact patched official source')
    raw = (Path(source_root)/relative).read_bytes()
    patch = (ROOT/'patches/sglang.patch').read_text()
    section = patch.split('diff --git a/'+relative+' b/'+relative+'\n',1)[1]
    index = next(line for line in section.splitlines() if line.startswith('index '))
    expected = index.split()[1].split('..')[1]
    actual = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest()
    assert actual == expected
    return ast.parse(raw)


@pytest.fixture
def backend_methods():
    cls = next(n for n in patched_tree(RELATIVE).body if isinstance(n, ast.ClassDef)
               and n.name == 'DeepseekSparseAttnBackend')
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in ('forward_decode', 'forward_extend', '_forward_flashmla_kv',
                            'set_dsa_prefill_impl')]
    namespace = dict(torch=torch, FlowKVSharedIndexAdapter=object,
        concat_mla_absorb_q_general=lambda nope, rope: torch.cat((nope, rope), dim=-1))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), RELATIVE, 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return SimpleNamespace(**{n.name: namespace[n.name] for n in nodes})


@pytest.mark.parametrize('window', [None, 0, 8192])
@pytest.mark.parametrize('dcp', [1, 2])
def test_bounded_prefill_avoids_native_flat_mha_reader(backend_methods, window, dcp):
    pool_tree = patched_tree('python/sglang/srt/mem_cache/flowkv_plain_pool.py')
    pool_cls = next(n for n in pool_tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'FlowKVPlainTokenToKVPool')
    capability = next(n for n in pool_cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'supports_mha_one_shot')
    namespace = {}
    exec(compile(ast.Module(body=[capability], type_ignores=[]), 'pool_capability', 'exec'), namespace)
    if window is None:
        pool = SimpleNamespace(dtype=torch.float8_e4m3fn)
    else:
        pool = type('Pool', (), {'supports_mha_one_shot': namespace['supports_mha_one_shot']})()
        pool.flowkv_layout = SimpleNamespace(prefill_window_rows=window)
        pool.dtype = torch.float8_e4m3fn
    backend = SimpleNamespace(dcp_size=dcp, token_to_kv_pool=pool,
        _is_in_breakable_cuda_graph=lambda: False, _is_in_tc_piecewise_cuda_graph=lambda: False,
        _get_device_sm=lambda: 90, _is_blackwell=lambda: False,
        hisparse_coordinator=None, enable_auto_select_prefill_impl=False)
    backend_cls = next(n for n in patched_tree(RELATIVE).body if isinstance(n, ast.ClassDef)
                       and n.name == 'DeepseekSparseAttnBackend')
    init = next(n for n in backend_cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    assignment = next(n for n in init.body if isinstance(n, ast.AnnAssign)
                      and isinstance(n.target, ast.Attribute) and n.target.attr == 'supports_mha_one_shot')
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'backend_capability', 'exec'), {'self': backend})
    method = backend_methods.set_dsa_prefill_impl
    method.__globals__.update(_IS_GFX95=False, is_dsa_enable_prefill_cp=lambda: False,
        envs=SimpleNamespace(SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=SimpleNamespace(get=lambda: 65536)))
    batch = SimpleNamespace(seq_lens_cpu=torch.tensor([128]), get_max_chunk_capacity=lambda: 8192,
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True))
    method(backend, batch)
    assert backend.use_mha is (dcp == 1 and window in (None, 0))


@pytest.mark.parametrize('position,owner', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('verify', [False, True])
def test_global_selection_is_remapped_before_writer_and_native_reader(backend_methods, position, owner, verify):
    calls = []
    logical = torch.tensor([[900000, 900008, -1]], dtype=torch.int32)
    physical = torch.tensor([[12, 13, -1]], dtype=torch.int32)

    def remap(**kw):
        assert kw['sparse_indices'] is logical and kw['decode_like']
        calls.append('remap')
        return physical

    pool = SimpleNamespace(flowkv_position_sharded=position, _ulacache_native_tp_cache=owner,
        flowkv_plain_compact_prefill=False, flowkv_after_attention=lambda layer: calls.append("retire"),
        native_sparse_cache_adapter=None,
        flowkv_prepare_packed_attention=remap,
        set_mla_kv_buffer=lambda *a: calls.append('write'),
        get_key_buffer=lambda layer: 'storage')
    layer = SimpleNamespace(is_cross_attention=False, layer_id=2, tp_q_head_num=2,
                            v_head_dim=512, head_dim=576, scaling=.1)
    metadata = SimpleNamespace()
    batch = SimpleNamespace(out_cache_loc=torch.tensor([900016]),
        forward_mode=SimpleNamespace(is_target_verify=lambda: verify))

    def reader(**kw):
        assert kw['page_table_1'] is (physical if position or owner else logical)
        assert kw['kv_cache'] == 'storage'
        calls.append('read')
        return 'native-output'

    backend = SimpleNamespace(token_to_kv_pool=pool, forward_metadata=metadata,
        dsa_decode_impl='flashmla_kv', hisparse_coordinator=None, use_fused_topk=True,
        _resolve_kpool_tail_backend=lambda indices, impl: impl,
        _check_kpool_tail_backend=lambda *a: None,
        _pad_topk_indices=lambda rows, queries: rows,
        _get_fused_topk_page_table=lambda rows: rows,
        _forward_flashmla_kv=reader)
    backend.forward_decode = lambda *a, **kw: backend_methods.forward_decode(backend, *a, **kw)
    method = backend_methods.forward_extend if verify else backend_methods.forward_decode
    output = method(backend, torch.zeros((1, 2, 576)), object(), object(),
                    layer, batch, topk_indices=logical)
    assert output == 'native-output'
    assert calls == ((['remap'] if position or owner else []) + ['write', 'read'] + (['retire'] if owner else []))
    assert logical.tolist() == [[900000, 900008, -1]]


def test_partial_reader_retires_after_native_attention_and_zero_owner_fixup(backend_methods, monkeypatch):
    calls = []
    native = ModuleType('sgl_kernel.flash_mla')

    def attention(**kw):
        assert kw['indices'].tolist() == [[[12, 13, -1]]]
        calls.append('attention')
        return torch.zeros((1, 1, 2, 512)), torch.zeros((1, 2, 1))

    native.flash_mla_with_kvcache = attention
    fixup = ModuleType('sglang.kernels.ops.attention.fixup_zero_kv')
    fixup.fixup_zero_kv_rows = lambda *a: calls.append('fixup')
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, fixup.__name__, fixup)
    pool = SimpleNamespace(flowkv_position_sharded=True,
        flowkv_local_attention_counts=lambda *a: torch.tensor([2]),
        flowkv_after_attention=lambda layer: calls.append('retire'))
    backend = SimpleNamespace(token_to_kv_pool=pool, physical_page_size=64,
        kv_cache_dim=656, dsa_index_topk=3, dsa_kv_cache_store_fp8=True,
        flashmla_kv_num_q_heads=2, dcp_size=8, flashmla_kv_format='native',
        get_device_int32_arange=lambda n: torch.arange(n, dtype=torch.int32))
    metadata = SimpleNamespace(dsa_cache_seqlens_int32=torch.tensor([3]),
        flashmla_metadata=SimpleNamespace(flashmla_metadata=object(), num_splits=object()))
    result = backend_methods._forward_flashmla_kv(backend, torch.zeros((1, 2, 576)),
        torch.zeros((64, 656), dtype=torch.uint8), 512, .1,
        SimpleNamespace(tp_q_head_num=2, head_dim=576, layer_id=2), metadata,
        torch.tensor([[12, 13, -1]], dtype=torch.int32))
    assert isinstance(result, tuple) and calls == ['attention', 'fixup', 'retire']
