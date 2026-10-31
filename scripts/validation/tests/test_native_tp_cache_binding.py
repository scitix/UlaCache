"""Exercise native TP ownership and writer boundaries without CUDA allocation."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("allocator_page,dcp", [(64, False), (512, False), (512, True)])
def test_prefix_tree_uses_the_allocator_ownership_boundary(allocator_page, dcp):
    """A 64-token kernel page may belong to a 512-token shared allocation."""
    import ast

    root = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if root is None:
        pytest.skip("Set HISPARSE_NATIVE_SOURCE_ROOT to the patched official checkout")
    path = Path(root) / "python/sglang/srt/mem_cache/kv_cache_builder.py"
    build = next(node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "build_kv_cache")
    parallel = NS(dcp_enabled=dcp, attn_cp_group=NS(cpu_group=None),
                  attn_tp_group=NS(cpu_group=None), pp_group=NS(cpu_group=None),
                  pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1)
    memory = NS(disable_radix_cache=False, radix_eviction_policy="lru",
                radix_eviction_policy_config=None, enable_session_radix_cache=False,
                enable_mamba_extra_buffer=False, enable_mamba_extra_buffer_lazy=False)
    allocator = NS(page_size=allocator_page)
    worker = NS(is_hybrid_swa=False, model_runner=NS(token_to_kv_pool=object(),
                model_config=object(), mtp_draft_device_pools=(), mtp_draft_mamba_pools=()),
                get_memory_pool=lambda: (object(), allocator))
    captured = {}

    class AtTreeConstruction(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise AtTreeConstruction

    namespace = dict(get_parallel=lambda: parallel,
        get_resolved_model_impl=lambda _: "native", ModelImpl=NS(TRANSFORMERS="other"),
        uses_ssm_state=lambda _: False, is_deepseek_dsa=lambda _: True,
        resolve_decode_retraction_backup=lambda **_: "cpu_tensor", get_memory=lambda: memory,
        get_disagg=lambda: NS(disaggregation_decode_enable_radix_cache=False),
        get_schedule=lambda: NS(chunked_prefill_size=8192),
        get_dp_tp_group=lambda: NS(cpu_group=None),
        get_exec=lambda: NS(mamba=NS(enable_mamba_extra_buffer=False,
                                   enable_mamba_extra_buffer_lazy=False)),
        CacheInitParams=capture)
    source = "from __future__ import annotations\n" + ast.unparse(build)
    exec(compile(source, str(path), "exec"), namespace)
    with pytest.raises(AtTreeConstruction):
        namespace[build.name](server_args=object(), model_config=NS(hf_config=None,
            is_multimodal=False), tp_worker=worker, page_size=64,
            spec_algorithm=NS(is_eagle=lambda: True), enable_metrics=False,
            enable_kv_cache_events=False, enable_hierarchical_cache=False)
    assert captured["page_size"] == allocator_page


@pytest.fixture
def binding(monkeypatch):
    root = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if root is None:
        pytest.skip("Set HISPARSE_NATIVE_SOURCE_ROOT to the patched official checkout")
    path = Path(root) / "python/sglang/srt/ulacache/native_tp_cache.py"
    if not path.is_file():
        pytest.fail("The installed SGLang patch lacks the native TP cache binding")

    class NativePool:
        def _create_buffers(self):
            self.events.append("allocate")

        def _clear_buffers(self):
            self.events.append("release")

        def finish_flowkv_forward(self):
            self.events.append("finish")

    class PageAllocator:
        def __init__(self, size, page_size, dtype, device, pool, need_sort):
            self._kvcache = pool
            self.clear()

        def clear(self):
            self._kvcache.events.append("pages_clear")

    host_constructor = Mock()
    modules = {
        "sglang.srt.mem_cache.allocator.paged": dict(
            PagedTokenToKVPoolAllocator=PageAllocator
        ),
        "sglang.srt.mem_cache.allocator.flowkv_host": dict(
            FlowKVHostTokenToKVPoolAllocator=host_constructor
        ),
        "sglang.srt.mem_cache.flowkv_plain_pool": dict(
            FlowKVPlainTokenToKVPool=NativePool
        ),
        "sglang.srt.mem_cache.flowkv_record_storage": dict(
            global_ids=lambda pool, ids: ids + 100
        ),
        "sglang.srt.mem_cache.memory_pool": dict(
            unwrap_write_loc=lambda value: (value, None, None)
        ),
        "sglang.srt.ulacache.v0522_owner_tp": dict(
            configuration=lambda: NS(native_kv_format="scaled656")
        ),
    }
    for name, contents in modules.items():
        module = ModuleType(name)
        module.__dict__.update(contents)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_native_tp_cache_binding_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.host_constructor = host_constructor
    return module


def pool(binding):
    value = binding.NativeTensorParallelCache()
    value.events = []
    value.start_layer = 10
    value.flowkv_host = None
    value.ulacache = Mock()
    return value


def test_full_history_pipeline_uses_native_forward_boundaries(binding):
    value = pool(binding)
    history = object()
    backend = NS(plan=object(), begin=Mock(), prepare=Mock(),
                 after_attention=Mock(), finish=Mock())
    value._tp_full_prefill = backend
    value.flowkv_prepare_extend(history)
    value.flowkv_arena = NS(invalidate=Mock())
    mode = NS(is_decode_or_idle=lambda: False, is_target_verify=lambda: False)
    ids = torch.tensor([512, 513])
    batch = NS(forward_mode=mode, out_cache_loc=ids)
    value.flowkv_before_layer_prepare(10, batch)
    value.flowkv_before_layer_prepare(11, batch)
    backend.begin.assert_called_once_with(batch, native_history=history)
    assert [c.args[0] for c in backend.prepare.call_args_list] == [0, 1]
    value.flowkv_after_attention(11)
    backend.after_attention.assert_called_once_with(1)
    value.ulacache.finish_layer_attention.assert_not_called()
    value.finish_flowkv_forward()
    backend.finish.assert_called_once_with()
    assert value.events == ['finish']
    assert value._tp_native_prefill_history is None


def test_constructor_and_teardown_have_one_cache_owner(binding, monkeypatch):
    value = pool(binding)
    monkeypatch.setattr(
        binding,
        "adopt_native_pool",
        lambda pool, **kwargs: pool.events.append(("adopt", kwargs)),
    )
    value.ulacache.close.side_effect = lambda: value.events.append("close")
    value._create_buffers()
    value._clear_buffers()
    assert value.events == [
        "allocate",
        ("adopt", {"native_kv_format": "scaled656"}),
        "close",
        "release",
    ]


def test_native_allocator_uses_public_lifetime_once(binding):
    value = pool(binding)
    value.dtype, value.device, value.allocation_page_size = torch.uint8, "cpu", 64
    value.flowkv_host = None
    value.ulacache.clear_registered_records.side_effect = lambda: value.events.append(
        "cache_clear"
    )
    allocator = value.create_logical_allocator(1024, need_sort=True)
    assert value.events == ["cache_clear", "pages_clear"]
    value.events.clear()
    allocator.clear()
    assert value.events == ["cache_clear", "pages_clear"]

    value.flowkv_host = NS(row_batch_limit=1024)
    value.create_logical_allocator(1024, need_sort=True)
    authority = binding.host_constructor.call_args.kwargs["host_tier"]
    ids = torch.tensor([64, 65])
    authority.allocate(ids)
    authority.release(ids)
    authority.clear()
    value.ulacache.register_logical_records.assert_called_once_with(ids)
    value.ulacache.free.assert_called_once_with(ids)
    assert authority.row_batch_limit == 1024
    assert value.ulacache.clear_registered_records.call_count == 3


def test_shared_selection_is_borrowed_and_reader_retires(binding):
    value = pool(binding)
    value.flowkv_layout = NS(topk=4)
    value.flowkv_prefetch = True
    value.flowkv_arena = NS(
        valid=torch.tensor([2]),
        invalidate=Mock(),
        kernel=lambda: NS(
            globalize_current=lambda ids, dst, base: dst.copy_(ids + base)
        ),
    )
    value._flowkv_global_current = torch.empty(2, dtype=torch.int64)
    value._flowkv_owner_start = 100
    value.flowkv_step_publication = NS(begin=Mock())
    mode = NS(is_decode_or_idle=lambda: True, is_target_verify=lambda: False)
    ids = torch.tensor([8, 9])
    batch = NS(forward_mode=mode, out_cache_loc=ids)
    value.flowkv_before_layer_prepare(10, batch)
    value.ulacache.begin_model_forward.assert_called_once_with()
    value.ulacache.prepare_layer_execution.assert_called_once_with(0)
    assert value._flowkv_global_current.tolist() == [108, 109]
    selected = torch.arange(8).reshape(2, 1, 4)
    value.flowkv_on_index_ready(10, selected, batch)
    layer, plan = value.ulacache.prepare_sparse_attention.call_args.args
    assert (
        layer == 0
        and plan.selected_logical_record_ids.data_ptr() == selected.data_ptr()
    )
    assert plan.current_logical_record_ids is ids
    slots = torch.arange(8).reshape(2, 4)
    value.ulacache.access.return_value = NS(slot_ids=slots)
    result = value.flowkv_prepare_packed_attention(
        layer_id=10, sparse_indices=selected, packed_out_loc=ids, decode_like=True
    )
    assert result is slots
    value.flowkv_after_attention(10)
    value.ulacache.finish_layer_attention.assert_called_once_with(0)


def test_native_quantizer_precedes_public_write(binding, monkeypatch):
    value = pool(binding)
    value.dsa_kv_cache_store_fp8 = True
    value._check_physical_write_loc = Mock()
    quantize = Mock(
        return_value=(
            torch.tensor([11], dtype=torch.uint8),
            torch.tensor([17], dtype=torch.uint8),
        )
    )
    name = "sglang.kernels.ops.attention.dsa.quant_k_cache"
    module = ModuleType(name)
    module.quantize_k_cache_separate = quantize
    monkeypatch.setitem(sys.modules, name, module)
    ids = torch.tensor([64])
    inputs = (torch.ones(1, dtype=torch.bfloat16), torch.zeros(1, dtype=torch.bfloat16))
    value.set_mla_kv_buffer(NS(layer_id=10), ids, *inputs)
    quantize.assert_called_once_with(*inputs)
    actual_ids, layer, encoded = value.ulacache.write.call_args.args
    assert actual_ids is ids and layer == 0
    assert encoded is quantize.return_value or all(
        a is b for a, b in zip(encoded, quantize.return_value)
    )
    value._check_physical_write_loc.assert_called_once()
