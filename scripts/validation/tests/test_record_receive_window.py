"""Portable receive ownership and incremental Index protocol controls."""
import pytest

from ulacache.transfer.receive_window import (
    ReceivePageAllocator, incremental_state_indices,
)


def test_failed_reservation_is_atomic_and_live_pages_do_not_overlap():
    pool = ReceivePageAllocator(5)
    first = pool.reserve("first", 3)
    with pytest.raises(BufferError):
        pool.reserve("oversized", 3)
    assert pool.available_pages == 2
    second = pool.reserve("second", 2)
    assert not set(first.pages) & set(second.pages)
    assert pool.available_pages == 0


def test_remote_writes_must_drain_and_stale_generations_cannot_free_reused_pages():
    pool = ReceivePageAllocator(2)
    old = pool.reserve("room", 2)
    with pytest.raises(RuntimeError):
        pool.release(old, transport_quiesced=False)
    assert pool.available_pages == 0
    pool.release(old, transport_quiesced=True)
    new = pool.reserve("room", 2)
    assert new.generation != old.generation
    with pytest.raises(ValueError):
        pool.release(old, transport_quiesced=True)
    assert pool.available_pages == 0
    pool.release(new, transport_quiesced=True)
    assert pool.available_pages == 2


def test_duplicate_room_and_zero_page_handoff():
    pool = ReceivePageAllocator(2)
    lease = pool.reserve("complete-prefix", 0)
    assert lease.pages == () and pool.available_pages == 2
    with pytest.raises(ValueError):
        pool.reserve("complete-prefix", 1)
    pool.release(lease, transport_quiesced=True)


@pytest.mark.parametrize("prefix,expected", [(0, 4), (64, 3), (192, 1), (256, 0)])
def test_incremental_index_preserves_native_positional_page_mapping(prefix, expected):
    source, destination = incremental_state_indices([9, 4, 18, 7], [2, 31, 5, 6], prefix, 64)
    assert len(source) == len(destination) == expected
    assert list(zip(source, destination)) == list(zip([9, 4, 18, 7], [2, 31, 5, 6]))[prefix//64:]


@pytest.mark.parametrize("source,destination,prefix,page_size", [
    ([1], [], 0, 64), ([1], [2], 65, 64), ([1], [2], 128, 64),
    ([1], [2], -1, 64), ([1], [2], 0, 0), ([1], [2], True, 64),
])
def test_invalid_incremental_state_metadata_fails_closed(source, destination, prefix, page_size):
    with pytest.raises(ValueError):
        incremental_state_indices(source, destination, prefix, page_size)
@pytest.mark.parametrize('pages,count,expected', [(0, 0, 0), (8, 509, 509),
    (7, 445, 445), (1, 1, 1), (1, None, 64), (8, 512, 512)])
def test_native_receive_excludes_unallocated_final_page_tail(pages, count, expected):
    from ulacache.transfer.receive_window import native_receive_record_count
    assert native_receive_record_count(pages, 64, count) == expected


@pytest.mark.parametrize('pages,count', [(8, 448), (7, 449), (1, 0), (0, 1), (1, True)])
def test_native_receive_rejects_inconsistent_valid_record_counts(pages, count):
    from ulacache.transfer.receive_window import native_receive_record_count
    with pytest.raises(ValueError):
        native_receive_record_count(pages, 64, count)


@pytest.mark.parametrize('allocation,transfer,expected', [
    (64,64,(2,5)), (512,64,tuple(range(16,24))+tuple(range(40,48))),
    (256,64,tuple(range(8,12))+tuple(range(20,24))),
])
def test_allocation_retirement_expands_every_native_transfer_page(allocation, transfer, expected):
    from ulacache.transfer.receive_window import transfer_pages_from_retired_allocation_pages
    assert transfer_pages_from_retired_allocation_pages([2,5],
        allocation_page_size=allocation, transfer_page_size=transfer) == expected


@pytest.mark.parametrize('pages,allocation,transfer', [
    ([0],512,64), ([-1],512,64), ([True],512,64), ([2,2],512,64),
    ([2],0,64), ([2],512,0), ([2],64,512), ([2],513,64),
    ([2],True,1), ([2**28],512,64),
])
def test_invalid_allocation_retirement_metadata_is_rejected(pages, allocation, transfer):
    from ulacache.transfer.receive_window import transfer_pages_from_retired_allocation_pages
    with pytest.raises(ValueError):
        transfer_pages_from_retired_allocation_pages(pages,
            allocation_page_size=allocation, transfer_page_size=transfer)
