"""Public facade contracts with CPU tensors; no CUDA execution is claimed."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from ulacache import (
    CacheExecutionOptions, ForwardExecutionContext, LayerReuseSpecification,
    LocalCacheBindings, RecordOwnerBindings, ResidentAttentionInputs, ResidentView,
    SparseAccessSpecification, SparseAttentionAccessPlan, UlaCache, init,
)
from test_standalone_contracts import create_cache_specification


def make_cache(*, allocator=None, owner=False):
    spec = replace(create_cache_specification(), sparse_access=SparseAccessSpecification(2, 3))
    engine = Mock()
    engine.active = torch.zeros(1, dtype=torch.int32)
    engine.gpu = tuple(torch.zeros((6, 8), dtype=torch.uint8) for _ in range(4 if owner else 3))
    engine.attention_rows.side_effect = lambda layer, queries: torch.tensor([[1, 4, 2]], dtype=torch.int32)[:queries]
    engine.device_storage_inventory.return_value = {7: 100}
    engine.memory_inventory.return_value = dict(gpu_bytes=100, gpu_kv_bytes=80, gpu_metadata_bytes=20)
    if owner:
        spec = replace(spec, layer_reuse=LayerReuseSpecification(
            (0, 1, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)))
        policy = RecordOwnerBindings('cuda:0', object(), Mock(), Mock(), Mock(),
            256, 64, True, logical_slot_allocator=allocator)
        factory = 'ulacache.cache.create.create_record_owner_execution_engine'
    else:
        policy = LocalCacheBindings('cuda:0', Mock(), Mock(),
            CacheExecutionOptions(record_storage_backend='native_paged'), logical_slot_allocator=allocator)
        factory = 'ulacache.cache.create.create_rank_local_execution_engine'
    with patch(factory, return_value=engine):
        cache = init(spec, policy)
    return cache, engine


def begin(cache):
    cache.begin_model_forward(ForwardExecutionContext('decode', 1, 1))


def selection():
    return torch.tensor([[4, 6, 9]], dtype=torch.int32)


def test_figure_mapping_and_one_allocation_authority():
    events = []
    ids = torch.tensor([4, 6, 9], dtype=torch.int32)
    allocator = Mock()
    allocator.alloc.side_effect = lambda n: events.append(('alloc', n)) or ids
    allocator.free.side_effect = lambda ids: events.append('recycle')
    cache, engine = make_cache(allocator=allocator)
    engine.allocate_unbound.side_effect = lambda *args, **kwargs: events.append('register')
    engine.release.side_effect = lambda ids: events.append('release')
    assert isinstance(cache, UlaCache)
    assert cache.alloc(3) is ids
    assert events == [('alloc', 3), 'register']
    engine.allocate_unbound.assert_called_once_with(ids, prefer_gpu=True)
    begin(cache)
    selected = ids.reshape(1, 3)
    for layer in (0, 1):
        engine.gpu[layer][[1, 4, 2], 0] = torch.tensor([4, 6, 9], dtype=torch.uint8)
        view = cache.access(selected, layer)
        assert isinstance(view, ResidentView)
        assert view.gpu_buffer is engine.gpu[layer]
        assert view.slot_ids.tolist() == [[1, 4, 2]]
        assert view.gpu_buffer[view.slot_ids.long(), 0].tolist() == [[4, 6, 9]]
        with pytest.raises(RuntimeError, match='Finish every access reader'):
            cache.free(ids)
        cache.finish_layer_attention(layer)
    assert engine.plan.call_count == 1
    cache.free(ids)
    assert events == [('alloc', 3), 'register', 'release', 'recycle']
    with pytest.raises(RuntimeError, match='begin_model_forward'):
        cache.access(selected, 0)


def test_prepared_decode_does_not_replan_or_allocate_read_only_scratch():
    cache, engine = make_cache()
    selected, current = selection(), torch.tensor([10], dtype=torch.int32)
    begin(cache)
    cache.prepare_sparse_attention(0, SparseAttentionAccessPlan(selected, current))
    cache.write(current, 0, (object(), object()))
    view = cache.access(selected, 0)
    cache.finish_layer_attention(0)
    cache.access(selected.view_as(selected), 1)
    engine.plan.assert_called_once_with(0, selected, current)
    assert view.slot_ids is not selected
    assert cache._no_current_records is None
    assert engine.write.call_count == 1


def test_contiguous_backend_keeps_its_existing_256_aligned_selection_contract():
    spec = create_cache_specification()
    engine = Mock(active=torch.zeros(1, dtype=torch.int32), gpu=(object(), object(), object()))
    selected = torch.full((1, 256), -1, dtype=torch.int32)
    selected[0, :3] = torch.tensor([4, 6, 9], dtype=torch.int32)
    engine.attention_rows.return_value = torch.zeros_like(selected)
    with patch('ulacache.cache.create.create_rank_local_execution_engine', return_value=engine):
        cache = init(spec, LocalCacheBindings('cuda:0', Mock(), Mock()))
    begin(cache)
    assert cache.access(selected, 0).slot_ids.shape == (1, 256)
    cache.finish_layer_attention(0)
    cache.access(selected, 1)
    cache.finish_layer_attention(1)
    assert engine.plan.call_count == 1
    assert engine.plan.call_args.args[1] is selected


def test_owner_banks_with_shared_topk_keep_independent_placement_plans():
    cache, engine = make_cache(owner=True)
    selected = selection()
    begin(cache)
    for layer in range(4):
        cache.access(selected, layer)
        cache.finish_layer_attention(layer)
    assert [call.args[0] for call in engine.plan.call_args_list] == [0, 1]
    cache.free(selected.flatten())
    assert engine.release.call_count == 1


def test_prepared_selection_metadata_does_not_extend_native_tensor_lifetime():
    import gc
    import weakref

    cache, engine = make_cache()
    selected = selection()
    reference = weakref.ref(selected)
    begin(cache)
    cache.prepare_sparse_attention(0, SparseAttentionAccessPlan(selected, selected[:, 0].contiguous()))
    engine.reset_mock()  # Remove the mock call recorder's independent strong refs.
    del selected
    gc.collect()
    assert reference() is None


def test_read_only_scratch_is_reused_and_included_in_memory_inventory():
    cache, engine = make_cache()
    selected = selection()
    for _ in range(2):
        begin(cache)
        cache.access(selected, 0)
        scratch = cache._no_current_records
        assert torch.equal(engine.plan.call_args.args[2], torch.tensor([-1], dtype=torch.int32))
        cache.finish_layer_attention(0)
        cache.access(selected, 1)
        cache.finish_layer_attention(1)
    assert cache._no_current_records is scratch
    storage = scratch.untyped_storage()
    assert cache.get_device_storage_inventory() == {7: 100, storage.data_ptr(): storage.nbytes()}
    assert cache.get_memory_inventory()['gpu_metadata_bytes'] == 20 + storage.nbytes()


def test_access_never_reads_ids_back_to_the_host():
    cache, _ = make_cache()
    selected = selection()
    begin(cache)
    with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')):
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')):
            with patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host IDs')):
                cache.access(selected, 0)
                cache.access(selected, 1)


@pytest.mark.parametrize('n', [-1, True, 1.5, '3'])
def test_invalid_allocation_count_never_calls_the_allocator(n):
    allocator = Mock()
    cache, _ = make_cache(allocator=allocator)
    with pytest.raises(ValueError, match='nonnegative integer'):
        cache.alloc(n)
    allocator.alloc.assert_not_called()


def test_allocation_requires_a_binding_and_preserves_capacity_failure():
    cache, engine = make_cache()
    with pytest.raises(RuntimeError, match='logical_slot_allocator'):
        cache.alloc(3)
    allocator = Mock()
    allocator.alloc.return_value = None
    cache, engine = make_cache(allocator=allocator)
    with pytest.raises(MemoryError):
        cache.alloc(3)
    engine.allocate_unbound.assert_not_called()


@pytest.mark.parametrize('operation', ['free', 'alloc', 'clear_registered_records', 'begin_model_forward'])
def test_mutation_cannot_recycle_a_shared_selection_before_its_last_reader(operation):
    allocator = Mock()
    cache, engine = make_cache(allocator=allocator)
    begin(cache)
    selected = selection()
    cache.access(selected, 0)
    cache.finish_layer_attention(0)
    args = {'free': (selected.flatten(),), 'alloc': (3,)}.get(operation, ())
    with pytest.raises(RuntimeError, match='Finish every access reader'):
        getattr(cache, operation)(*args)
    allocator.alloc.assert_not_called()
    allocator.free.assert_not_called()
    engine.release.assert_not_called()


def test_release_failure_does_not_recycle_ids():
    allocator = Mock()
    cache, engine = make_cache(allocator=allocator)
    engine.release.side_effect = RuntimeError('reader still active')
    with pytest.raises(RuntimeError, match='reader still active'):
        cache.free(selection().flatten())
    allocator.free.assert_not_called()


def test_access_rejects_unprepared_followers_changed_ids_and_finished_layers():
    cache, engine = make_cache()
    selected = selection()
    with pytest.raises(RuntimeError, match='begin_model_forward'):
        cache.access(selected, 0)
    begin(cache)
    with pytest.raises(RuntimeError, match='producer'):
        cache.access(selected, 1)
    cache.access(selected, 0)
    with pytest.raises(ValueError, match='prepared selection'):
        cache.access(selected.clone(), 1)
    cache.finish_layer_attention(0)
    with pytest.raises(RuntimeError, match='finished access layer'):
        cache.access(selected, 0)
    assert engine.plan.call_count == 1


@pytest.mark.parametrize('layer, shape', [(-1, (1, 3)), (3, (1, 3)), (True, (1, 3)),
    (0, (3,)), (0, (1, 2)), (0, (3, 3))])
def test_access_rejects_wrong_layer_or_input_bounds_before_planning(layer, shape):
    cache, engine = make_cache()
    begin(cache)
    with pytest.raises(ValueError):
        cache.access(torch.zeros(shape, dtype=torch.int32), layer)
    engine.plan.assert_not_called()


def test_invalid_record_error_is_propagated_without_registration():
    cache, engine = make_cache()
    begin(cache)
    engine.plan.side_effect = RuntimeError('unregistered history ID')
    with pytest.raises(RuntimeError, match='unregistered history ID'):
        cache.access(selection(), 0)
    engine.allocate_unbound.assert_not_called()
    engine.attention_rows.assert_not_called()


def test_read_only_first_use_during_graph_capture_fails_before_allocating(monkeypatch):
    cache, engine = make_cache()
    begin(cache)
    ids = SimpleNamespace(shape=(1, 3), is_cuda=True, new_full=Mock())
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='Warm up'):
        cache.access(ids, 0)
    ids.new_full.assert_not_called()
    engine.plan.assert_not_called()


def test_multi_plane_view_preserves_native_layout_and_rejects_a_single_buffer():
    keys, values, slots = object(), object(), object()
    view = ResidentView((keys, values), slots, 64, 'segment_major')
    assert ResidentView is ResidentAttentionInputs
    assert view.slot_ids is slots and view.record_planes == (keys, values)
    with pytest.raises(ValueError, match='record_planes'):
        _ = view.gpu_buffer


@pytest.mark.parametrize('operation, args', [('alloc', (3,)), ('write', (None, 0, ())),
    ('access', (None, 0)), ('free', (None,))])
def test_public_operations_reject_a_closed_cache(operation, args):
    cache, _ = make_cache()
    cache.close()
    with pytest.raises(RuntimeError, match='closed'):
        getattr(cache, operation)(*args)
