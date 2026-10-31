"""Public lifecycle over the real TP coordinator; CPU-only dispatch contracts."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ulacache import (
    CacheExecutionOptions, KeyValueRecordPlane, KeyValueStorageLayout,
    LayerReuseSpecification, MemoryTierCapacity, PrefetchPolicy,
    SparseAccessSpecification, SparseAttentionAccessPlan,
    TensorParallelCacheBindings, UnifiedCacheSpecification, init,
)
from ulacache.cache.layers import LayerGroupPlan
from ulacache.cache.prefetch import SharedIndexLifecycle


def adopted_cache(monkeypatch, *, allocator=None):
    groups = LayerGroupPlan.from_layer_anchors((0, 0))
    storage = torch.zeros((2, 64, 656), dtype=torch.uint8)
    arena = SimpleNamespace(storage=storage, valid=torch.zeros(1, dtype=torch.int32),
        groups=(SimpleNamespace(remap=torch.tensor([[1, 3]], dtype=torch.int32)),)*2)
    layout = SimpleNamespace(layers=2, layer_groups=groups, row_bytes=656,
        page_size=64, max_query_tokens=2, topk=2, max_staging_tokens=4)
    host = SimpleNamespace(host_slots=128, allocate=Mock(), release=Mock(), clear=Mock())
    pool = SimpleNamespace(flowkv_layout=layout, flowkv_arena=arena,
        flowkv_host=host, flowkv_request=None, flowkv_shards=storage,
        _flowkv_storage_size=8, _flowkv_namespaces=1,
        _flowkv_current_native=torch.tensor([5], dtype=torch.int32),
        flowkv_step_publication=SimpleNamespace(publish=Mock()))
    lifecycle = SharedIndexLifecycle(groups, pool)
    pool.flowkv_shared_index = lifecycle
    calls = []

    def plan(layer, ids):
        group = groups.group(layer)
        assert layer == groups.anchors[group] and group not in lifecycle._planned
        lifecycle._planned.add(group)
        lifecycle._keepalive[group] = (ids, None, pool._flowkv_current_native)
        calls.append(('plan', layer))

    monkeypatch.setattr(lifecycle, '_plan_index', plan)
    monkeypatch.setattr(lifecycle, '_pull_history', lambda layer: calls.append(('pull', layer)))
    spec = UnifiedCacheSpecification(
        KeyValueStorageLayout('history', (KeyValueRecordPlane('kv', 'mla.scaled656.fp8.v1', (528, 128)),), 64),
        LayerReuseSpecification.from_shared_sparse_selection((0, 0)),
        SparseAccessSpecification(2, 2), MemoryTierCapacity(64, 128, 4))
    policy = TensorParallelCacheBindings(pool, logical_slot_allocator=allocator,
        execution_options=CacheExecutionOptions(record_storage_backend='tensor_parallel',
            prefetch=PrefetchPolicy(False)))
    return init(spec, policy), pool, spec, policy, calls


def test_adopts_one_pool_and_preserves_one_reader_lifetime(monkeypatch):
    allocator = Mock()
    ids = torch.tensor([4, 6], dtype=torch.int32)
    allocator.alloc.return_value = ids
    cache, pool, _, _, calls = adopted_cache(monkeypatch, allocator=allocator)
    assert cache._execution_engine.pool is pool
    lifecycle = pool.flowkv_shared_index
    assert cache.alloc(2) is ids
    allocator.alloc.assert_called_once_with(2)
    pool.flowkv_host.allocate.assert_called_once_with(ids)
    selected = ids.reshape(1, 2)
    cache.begin_model_forward()
    cache.prepare_sparse_attention(0, SparseAttentionAccessPlan(selected, pool._flowkv_current_native))
    for layer in (0, 1):
        view = cache.access(selected, layer)
        assert view.gpu_buffer.data_ptr() == pool.flowkv_arena.storage[layer].data_ptr()
        assert view.slot_ids.tolist() == [[1, 3]]
        with pytest.raises(RuntimeError, match='Finish every TP reader'):
            cache.free(ids)
        cache.finish_layer_attention(layer)
    assert calls == [('plan', 0), ('pull', 0), ('pull', 1)]
    assert pool.flowkv_shared_index is lifecycle
    pool.flowkv_step_publication.publish.assert_called_once_with()
    cache.free(ids)
    pool.flowkv_host.release.assert_called_once_with(ids)
    allocator.free.assert_called_once_with(ids)
    cache.close()
    assert pool.flowkv_shards is not None and pool.flowkv_host is not None


def test_native_allocator_hooks_register_without_another_authority(monkeypatch):
    cache, pool, _, _, _ = adopted_cache(monkeypatch)
    ids = torch.tensor([7, 8], dtype=torch.int32)
    with pytest.raises(RuntimeError, match='logical_slot_allocator'):
        cache.alloc(2)
    cache.register_logical_records(ids)
    cache.free(ids)
    pool.flowkv_host.allocate.assert_called_once_with(ids)
    pool.flowkv_host.release.assert_called_once_with(ids)


def test_rejects_new_prefetch_policy_without_rebuilding_native_pool(monkeypatch):
    _, _, spec, policy, _ = adopted_cache(monkeypatch)
    options = replace(policy.execution_options, prefetch=PrefetchPolicy(True))
    with pytest.raises(ValueError, match='lowered before native pool'):
        init(spec, replace(policy, execution_options=options))


@pytest.mark.parametrize('completion', ['group', 'layer'])
def test_adoption_preserves_existing_completion_and_does_not_create_another_stream(monkeypatch, completion):
    cache, pool, spec, bindings, _ = adopted_cache(monkeypatch)
    cache.close()
    pool.device = 'cpu'
    stream = Mock()
    constructor = Mock(return_value=stream)
    monkeypatch.setattr(torch.cuda, 'Stream', constructor)
    monkeypatch.setattr(torch.cuda, 'Event', Mock(side_effect=lambda: object()))
    lifecycle = SharedIndexLifecycle(pool.flowkv_layout.layer_groups, pool,
        prefetch=PrefetchPolicy(True, completion))
    pool.flowkv_shared_index = lifecycle
    other = 'layer' if completion == 'group' else 'group'
    wrong = replace(bindings.execution_options, prefetch=PrefetchPolicy(True, other))
    with pytest.raises(ValueError, match='completion must match'):
        init(spec, replace(bindings, execution_options=wrong))
    options = replace(bindings.execution_options, prefetch=PrefetchPolicy(True, completion))
    adopted = init(spec, replace(bindings, execution_options=options))
    assert adopted._execution_engine.pool.flowkv_shared_index is lifecycle
    assert lifecycle._stream is stream
    constructor.assert_called_once_with(device='cpu')
    adopted.close()


def test_prefix_workspace_is_distinct_from_owner_history_capacity(monkeypatch):
    cache, pool, spec, policy, _ = adopted_cache(monkeypatch)
    cache.close()
    pool.flowkv_layout.prefill_window_rows = 65536
    capacity = replace(spec.capacity_plan, maximum_prefill_union_record_count=65536,
        prefill_workspace_record_capacity=65536)
    # A bounded fixture can have small owner history and a full native window.
    second = init(replace(spec, capacity_plan=capacity), policy)
    assert second._execution_engine.pool is pool
    with pytest.raises(ValueError, match='declared workspace'):
        replace(capacity, maximum_prefill_union_record_count=65537)
    with pytest.raises(ValueError, match='actual per-owner'):
        init(replace(spec, capacity_plan=replace(capacity, maximum_prefill_union_record_count=32768,
            prefill_workspace_record_capacity=32768)), policy)


def test_pool_cannot_acquire_another_public_allocation_handle(monkeypatch):
    cache, pool, spec, policy, _ = adopted_cache(monkeypatch)
    with pytest.raises(RuntimeError, match='already has a public cache handle'):
        init(spec, policy)
    cache.close()
    assert not hasattr(pool, '_ulacache_public_engine')
    assert init(spec, policy)._execution_engine.pool is pool


def test_public_writer_keeps_encoded_chunks_without_repacking(monkeypatch):
    cache, pool, _, _, _ = adopted_cache(monkeypatch)
    from ulacache.storage import writers as tp_writer

    writer = Mock()
    monkeypatch.setattr(tp_writer, 'write_native_record_segments', writer)
    # Native quantization supplies complete bytes, including scales and RoPE.
    nope = torch.arange(3*528, dtype=torch.int32).to(torch.uint8).reshape(3, 528)
    rope = torch.arange(3*128, dtype=torch.int32).to(torch.uint8).reshape(3, 128)
    ids = torch.tensor([4, 5, 6], dtype=torch.int32)
    for start, stop in ((0, 2), (2, 3)):
        cache.write(ids[start:stop], 0, (nope[start:stop], rope[start:stop]))
    assert writer.call_count == 2
    for call, offset in zip(writer.call_args_list, (0, 2)):
        actual_pool, layer, actual_ids, segments = call.args
        assert actual_pool is pool and layer == 0
        assert actual_ids.data_ptr() == ids[offset:].data_ptr()
        assert segments[0].data_ptr() == nope[offset:].data_ptr()
        assert segments[1].data_ptr() == rope[offset:].data_ptr()


def test_follower_cannot_replace_the_producer_selection(monkeypatch):
    cache, pool, _, _, _ = adopted_cache(monkeypatch)
    selected = torch.tensor([[4, 6]], dtype=torch.int32)
    cache.begin_model_forward()
    cache.prepare_sparse_attention(0, SparseAttentionAccessPlan(selected, pool._flowkv_current_native))
    cache.access(selected, 0)
    cache.finish_layer_attention(0)
    with pytest.raises(ValueError, match='producer selection tensor'):
        cache.access(selected.clone(), 1)
    with pytest.raises(RuntimeError, match='Finish every TP reader'):
        cache.close()
    cache.access(selected, 1)
    cache.finish_layer_attention(1)
