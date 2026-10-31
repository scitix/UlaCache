import pytest
from ulacache.telemetry.trace_coverage import validate_bucket_rotation


def native_rows():
    return [[10, 1, 7, 17, 10, 0, 30, 47],
            [11, 2, 8, 5, 11, 0, 47, 52],
            [12, 0, 9, 0, 12, 0, 52, 52]]


def test_rotation_uses_group_clock_and_real_expired_members():
    result = validate_bucket_rotation(native_rows(), 3)
    assert result['full_rotation']
    assert result['expired_headers'] == 3
    assert result['expired_members'] == 22
    assert result['nonempty_expirations'] == 2


def test_global_forward_count_cannot_replace_full_rotation():
    with pytest.raises(ValueError, match='full native bucket rotation'):
        validate_bucket_rotation(native_rows()[:2], 3)


@pytest.mark.parametrize('column,value', [(0, 13), (1, 0), (4, 99), (5, 1), (7, 100)])
def test_rotation_rejects_missing_clock_or_bad_expiry_evidence(column, value):
    rows = native_rows()
    rows[1][column] = value
    with pytest.raises(ValueError):
        validate_bucket_rotation(rows, 3)


def test_empty_expiry_still_records_a_real_rotation():
    rows = [[e, e % 3, -1, 0, e, 0, 0, 0] for e in range(1, 4)]
    assert validate_bucket_rotation(rows, 3)['expired_members'] == 0
