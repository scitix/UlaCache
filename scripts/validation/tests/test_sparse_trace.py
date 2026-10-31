import pytest
from ulacache.telemetry.sparse_trace import SparseTraceGeometry, historical_lookup_reference


def test_mtp_union_excludes_all_current_positions_per_request():
    selected = [[10, 11, 12, 21, -1], [10, 12, 21, 22, 0], [10, 13, 22, 23, -1],
                [10, 21, 10, -1, 0]]
    lookups = historical_lookup_reference(selected, [4, 4, 4, 7], [21, 22, 23, 40])
    assert lookups == frozenset({(4, 10), (4, 11), (4, 12), (4, 13), (7, 10), (7, 21)})


def test_padding_query_cannot_create_lookup_or_remove_another_request_history():
    assert historical_lookup_reference([[7, 9], [7, 9]], [2, -1], [20, 7]) == frozenset({(2, 7), (2, 9)})


def test_empty_trace_query_set():
    assert historical_lookup_reference([], [], []) == frozenset()


def test_trace_requires_aligned_query_metadata():
    with pytest.raises(ValueError):
        historical_lookup_reference([[1]], [], [2])


@pytest.mark.parametrize('changes', [dict(frames=65), dict(query_rows=0), dict(topk=True), dict(groups=2**31)])
def test_trace_rejects_invalid_or_incomplete_geometry(changes):
    values = dict(groups=21, query_rows=48, topk=2048)
    values.update(changes)
    with pytest.raises(ValueError):
        SparseTraceGeometry(**values)
