import pytest

from ulacache.telemetry.tp_trace_coverage import validate_fused_begin_rotation


def test_fused_begin_accepts_current_members_and_cold_pin_removal():
    rows = [[4, 1, 1, 7, 4, 2, 10, 16],
            [5, 2, 2, 0, 5, 2, 16, 15],
            [6, 0, 3, 4, 6, 1, 15, 19]]
    value = validate_fused_begin_rotation(rows, 3, [2, 2, 2])
    assert value["expired_members"] == 11
    assert value["expired_headers"] == 3
    assert value["full_rotation"]
    assert not value["pure_expiry_time_measured"]


@pytest.mark.parametrize("replacement", [
    [5, 1, 1, 7, 5, 2, 10, 16],  # wrong physical bucket
    [4, 1, 3, 7, 4, 2, 10, 16],  # a nonempty bucket is too young
    [4, 1, 1, 7, 4, 3, 10, 16],  # more current members than valid queries
    [4, 1, 1, 7, 4, 2, 10, 18],  # excess unexplained cold insertion
    [4, 1, 1, 7, 4, 2, 10, 7],   # excess unexplained cold removal
])
def test_fused_begin_rejects_inconsistent_native_headers(replacement):
    with pytest.raises(ValueError):
        validate_fused_begin_rotation([replacement,
            [5, 2, 2, 0, 5, 2, 16, 15],
            [6, 0, 3, 4, 6, 1, 15, 19]], 3, [2, 2, 2])


def test_fused_begin_requires_complete_group_rotation():
    with pytest.raises(ValueError, match="full bucket rotation"):
        validate_fused_begin_rotation([[4, 1, 1, 7, 4, 2, 10, 16]], 3, [2])
