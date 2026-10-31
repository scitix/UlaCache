"""Native write publication and reader ordering without CUDA execution."""

from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
import torch

from ulacache import KeyValueRecordPlane, KeyValueStorageLayout, SparseAttentionAccessPlan
from test_logical_memory_api import begin, make_cache, selection


def test_alloc_write_access_preserves_native_bytes_and_logical_order():
    ids = selection().flatten()
    allocator = Mock(alloc=Mock(return_value=ids))
    cache, backend = make_cache(allocator=allocator)
    cache.alloc(3)
    # Include all byte patterns, preserving native payload and scale bits.
    segments = (torch.arange(3 * 528).to(torch.uint8).reshape(3, 528),
        torch.arange(3 * 128).to(torch.uint8).reshape(3, 128))
    expected = torch.cat(segments, dim=1)
    backend.gpu[0].resize_(6, 656)
    physical = torch.tensor([1, 4, 2])

    def publish(layer, logical_ids, *native_segments):
        assert logical_ids is ids
        assert all(actual is original for actual, original in zip(native_segments, segments))
        backend.gpu[layer][physical] = torch.cat(native_segments, dim=1)

    backend.write.side_effect = publish
    with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host ID copy')):
        cache.write(ids, 0, segments)
        backend.plan.assert_not_called()
        begin(cache)
        view = cache.access(ids.reshape(1, 3), 0)
    assert torch.equal(view.gpu_buffer[view.slot_ids.long()], expected.unsqueeze(0))
    cache.finish_layer_attention(0)
    cache.access(ids.reshape(1, 3), 1)
    cache.finish_layer_attention(1)
    cache.free(ids)
    allocator.alloc.assert_called_once_with(3)
    allocator.free.assert_called_once_with(ids)
    assert backend.allocate_unbound.call_count == 1


@pytest.mark.parametrize('plane_count', [1, 2])
def test_decode_packed_write_keeps_source_positions_and_plan_publish_read_order(plane_count):
    cache, backend = make_cache()
    if plane_count == 2:
        cache.cache_specification = replace(cache.cache_specification,
            storage_layout=KeyValueStorageLayout('split', (
                KeyValueRecordPlane('k', 'native.k', (8,)),
                KeyValueRecordPlane('v', 'native.v', (8,)),
            ), records_per_page=64))
    ids = torch.tensor([10], dtype=torch.int32)
    source_positions = torch.tensor([7], dtype=torch.int64)
    pages = tuple(torch.arange(64).to(torch.uint8) for _ in range(plane_count))
    pages = pages[0] if plane_count == 1 else pages
    events = []
    backend.plan.side_effect = lambda *_: events.append('plan')

    def publish(layer, logical_ids, source_pages, indices):
        assert layer == 0 and logical_ids is ids
        assert source_pages is pages and indices is source_positions
        events.append('publish')

    backend.write_native_paged_records.side_effect = publish
    backend.attention_rows.side_effect = lambda *_: (
        events.append('read') or torch.tensor([[1, 4, 2]], dtype=torch.int32))
    selected = selection()
    begin(cache)
    cache.prepare_sparse_attention(0, SparseAttentionAccessPlan(selected, ids))
    cache.write(ids, 0, pages, source_indices=source_positions)
    cache.access(selected, 0)
    assert events == ['plan', 'publish', 'read']
    backend.write.assert_not_called()
    backend.allocate_unbound.assert_not_called()


@pytest.mark.parametrize('ids, layer, values, source_indices', [
    (torch.tensor([[4]], dtype=torch.int32), 0, (object(), object()), None),
    (torch.tensor([4], dtype=torch.int32), -1, (object(), object()), None),
    (torch.tensor([4], dtype=torch.int32), True, (object(), object()), None),
    (torch.tensor([4], dtype=torch.int32), 3, (object(), object()), None),
    (torch.arange(513), 0, (object(), object()), None),
    (torch.tensor([4], dtype=torch.int32), 0, (object(),), None),
    (torch.tensor([4], dtype=torch.int32), 0, object(), torch.tensor([1, 2])),
    (torch.tensor([4], dtype=torch.int32), 0, object(), torch.tensor([[1]])),
])
def test_invalid_publication_is_rejected_before_any_native_write(ids, layer, values, source_indices):
    cache, backend = make_cache()
    with pytest.raises(ValueError):
        cache.write(ids, layer, values, source_indices=source_indices)
    backend.write.assert_not_called()
    backend.write_native_paged_records.assert_not_called()
    backend.allocate_unbound.assert_not_called()


def test_write_failure_preserves_logical_allocation_for_retry():
    allocator = Mock()
    cache, backend = make_cache(allocator=allocator)
    ids, segments = selection().flatten(), (object(), object())
    backend.write.side_effect = RuntimeError('publication failed')
    with pytest.raises(RuntimeError, match='publication failed'):
        cache.write(ids, 0, segments)
    backend.release.assert_not_called()
    allocator.free.assert_not_called()
    backend.write.side_effect = None
    cache.write(ids, 0, segments)
    assert backend.write.call_count == 2
