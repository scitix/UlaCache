import pytest

from ulacache.transfer.receive_window import (
    incremental_receive_state_pages, incremental_state_indices,
)


@pytest.mark.parametrize('prefix,received', [(0, [3, 0, 7, 1]), (128, [3, 0]), (256, [])])
def test_native_sender_slices_only_once_and_reaches_common_receive_lease(prefix, received):
    canonical, source = [8, 19, 11, 30], [40, 42, 41, 49]
    mapped = incremental_receive_state_pages(canonical, received, prefix_tokens=prefix, page_size=64)
    src, dst = incremental_state_indices(source, mapped, prefix, 64)
    assert tuple(src) == tuple(source[prefix//64:])
    assert tuple(dst) == tuple(received)
    assert tuple(mapped[:prefix//64]) == tuple(canonical[:prefix//64])
    assert canonical == [8, 19, 11, 30]


@pytest.mark.parametrize('canonical,received,prefix,page_size', [
    ([8, 9], [0, 1], 0, 0), ([8, 9], [0, 1], -64, 64),
    ([8, 9], [0, 1], 32, 64), ([8, 9], [0, 1], True, 64),
    ([8, 9], [], 192, 64), ([8, 9], [0], 0, 64),
    ([0], [0], 0, 64), ([8, 8], [0, 1], 0, 64),
    ([8, 9], [0, 0], 0, 64), ([8], [-1], 0, 64), ([8], [True], 0, 64),
])
def test_invalid_native_state_or_receive_lease_geometry_fails_closed(canonical, received, prefix, page_size):
    with pytest.raises(ValueError):
        incremental_receive_state_pages(canonical, received, prefix_tokens=prefix, page_size=page_size)
