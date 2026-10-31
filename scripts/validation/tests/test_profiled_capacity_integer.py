"""A fractional HiSparse byte coefficient must still produce integer slots."""
import ast
import __future__
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
RELATIVE = 'python/sglang/srt/model_executor/pool_configurator.py'


@pytest.fixture
def native_methods():
    source = os.environ.get('HISPARSE_NATIVE_SOURCE_ROOT')
    if not source:
        pytest.skip('Set HISPARSE_NATIVE_SOURCE_ROOT to the exact patched official source')
    raw = (Path(source) / RELATIVE).read_bytes()
    section = (ROOT / 'patches/sglang.patch').read_text().split(
        'diff --git a/' + RELATIVE + ' b/' + RELATIVE + '\n', 1)[1]
    expected = next(line for line in section.splitlines() if line.startswith('index ')).split()[1].split('..')[1]
    assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\x00' + raw).hexdigest() == expected
    cls = next(node for node in ast.parse(raw).body
               if isinstance(node, ast.ClassDef) and node.name == 'DefaultPoolConfigurator')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('calculate_pool_sizes', 'calculate_pool_sizes_from_max_tokens',
                                 '_compute_dsa_indexer_cell_size')]
    namespace = dict(MemoryPoolConfig=lambda **kwargs: SimpleNamespace(**kwargs))
    exec(compile(ast.Module(body=methods, type_ignores=[]), RELATIVE, 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return SimpleNamespace(context=namespace, **{node.name: namespace[node.name] for node in methods})


@pytest.mark.parametrize('coefficient', [341, 341.25, 0])
def test_profile_floor_never_exceeds_budget_and_returns_integer_slots(native_methods, coefficient):
    budget, overhead, page = 183720301, 1024, 64
    profiler = SimpleNamespace(_fixed_overhead_bytes=overhead, _cell_size=coefficient,
                               _zero_kv_max_tokens=538399)
    result = native_methods.calculate_pool_sizes(profiler, budget, page)
    slots = result.max_total_num_tokens
    assert type(slots) is int and slots % page == 0
    if coefficient:
        assert slots * coefficient <= budget - overhead < (slots + page) * coefficient
    else:
        assert slots == 538368


def test_capped_integral_float_is_normalized_without_growing_capacity(native_methods):
    result = native_methods.calculate_pool_sizes_from_max_tokens(None, 538368.0, 64)
    assert type(result.max_total_num_tokens) is int
    assert result.max_total_num_tokens == 538368


@pytest.mark.parametrize('hisparse,elide,draft,layers,expected_layers', [
    (True, True, False, 78, 21), (True, False, False, 78, 78),
    (True, True, True, 1, 1), (False, True, False, 78, 21)])
def test_index_sizing_uses_the_same_layer_ownership_as_pool_construction(
        native_methods, monkeypatch, hisparse, elide, draft, layers, expected_layers):
    # Actual GLM-5.3 target owners from all-rank startup inventories.
    owners = {0, 1, 2, *range(6, 75, 4)}
    assert len(owners) == 21
    ratio = 3.6723992106005077 if hisparse else 1
    configurator = ModuleType('sglang.srt.mem_cache.kv_cache_configurator')
    configurator._should_elide_dsa_index_k = lambda **kwargs: elide and not kwargs['is_draft_worker']
    sparsity = ModuleType('sglang.srt.mem_cache.sparsity')
    sparsity.parse_hisparse_config = lambda: SimpleNamespace(host_to_device_ratio=ratio)
    cp = ModuleType('sglang.srt.layers.cp.utils')
    cp.get_glm_dsa_cp_layer_shard_info = lambda kvc: (None, 1)
    cp.get_layer_shard_range = lambda *args: (_ for _ in ()).throw(AssertionError('Not layer sharding'))
    for module in (configurator, sparsity, cp):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    native_methods.context.update(_is_npu=False,
        get_dsa_index_head_dim=lambda hf: 128, get_dsa_index_kpool=lambda hf: 1,
        ceil_div=lambda value, divisor: (value + divisor - 1) // divisor,
        torch=SimpleNamespace(_utils=SimpleNamespace(_element_size=lambda dtype: 1)),
        DSATokenToKVPool=SimpleNamespace(quant_block_size=128, index_k_with_scale_buffer_dtype=object()),
        get_memory=lambda: SimpleNamespace(enable_hisparse=hisparse),
        dsa_layer_skips_topk=lambda hf, layer: layer not in owners,
        _get_dsa_cache_layer_ids=lambda kvc, count: list(range(count)))
    kvc = SimpleNamespace(model_config=SimpleNamespace(hf_config=object()), is_draft_worker=draft,
                          server_args=SimpleNamespace(enable_hisparse=hisparse))
    size = native_methods._compute_dsa_indexer_cell_size(None, kvc=kvc, num_layers=layers,
                                                       allocate_all_layers=draft)
    assert size == int(132 * expected_layers * ratio)
