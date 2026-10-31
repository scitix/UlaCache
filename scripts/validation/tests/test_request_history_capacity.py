"""Request-based history budgets stay independent of speculative query bounds."""
import pytest

from ulacache.storage.capacity import request_history_capacity
from ulacache.storage.workspace import ResidentQueryLayout


def make_layout(width, ratio=3):
    queries = 64 * width
    return ResidentQueryLayout(layers=78, groups=21, max_batch=64, pool_rows=1048576,
        row_bytes=656, disaggregation_mode='null', topk=2048,
        max_query_tokens=1 << (queries - 1).bit_length(), max_resident_queries=queries,
        history_slots=request_history_capacity(64, 2048, ratio), page_size=64,
        native_segment_bytes=(656,),
        layer_anchors=(0, 1) + tuple(2 + 4 * ((layer - 2) // 4) for layer in range(2, 78)))


def test_same_request_budget_off_and_two_drafts():
    ordinary, speculative = make_layout(1), make_layout(3)
    assert ordinary.capacity == speculative.capacity == 64 * 3 * 2048
    assert ordinary.resident_query_capacity == 64
    assert speculative.resident_query_capacity == 192
    assert speculative.max_query_tokens == 256
    assert speculative.local_kv_buffer_bytes - ordinary.local_kv_buffer_bytes == 128 * 78 * 656


def test_execution_viability_never_silently_increases_requested_capacity():
    assert request_history_capacity(64, 2048, 2) == 262144
    with pytest.raises(ValueError, match='largest TopK batch'):
        make_layout(3, ratio=2)


@pytest.mark.parametrize('ratio', [1.1, 0.5, 'nan', 'inf'])
def test_nonrepresentable_request_capacity_is_rejected_without_rounding(ratio):
    with pytest.raises(ValueError):
        request_history_capacity(64, 2048, ratio)
