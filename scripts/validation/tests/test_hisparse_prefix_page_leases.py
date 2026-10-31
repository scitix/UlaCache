import pytest

from ulacache.cache.prefill.leases import PrefillPageLeases


def test_shared_prefix_keeps_physical_page_until_last_reader():
    leases = PrefillPageLeases()
    leases.acquire('a', {1: 7, 2: 8})
    leases.acquire('b', {1: 7, 3: 9})
    assert leases.release('a') == {2: 8}
    assert leases.pages[1].references == 1
    assert leases.release('b') == {1: 7, 3: 9}
    assert not leases.pages and not leases.requests


def test_chunk_extension_and_retry_do_not_double_pin_prefix():
    leases = PrefillPageLeases()
    assert leases.acquire('a', {1: 7}) == [1]
    assert leases.acquire('a', {1: 7, 2: 8}) == [2]
    assert leases.pages[1].references == 1
    assert leases.release('a') == {1: 7, 2: 8}
    assert leases.release('a') == {}


def test_failed_prefill_allocation_rolls_back_only_new_borrowed_pages():
    leases = PrefillPageLeases()
    leases.acquire('producer', {1: 7})
    leases.acquire('consumer', {1: 7})
    newly_acquired = leases.acquire('consumer', {1: 7, 2: 8})
    assert leases.release('consumer', newly_acquired) == {2: 8}
    assert leases.pages[1].references == 2
    assert leases.release('consumer') == {}
    assert leases.release('producer') == {1: 7}


def test_eviction_cannot_reclaim_a_live_shared_prefix():
    leases = PrefillPageLeases()
    leases.acquire('a', {1: 7})
    with pytest.raises(ValueError, match='active prefill readers'):
        leases.assert_reclaimable([1, 2])
    leases.release('a')
    leases.assert_reclaimable([1, 2])
    leases.acquire('new-prefix', {1: 23})
    assert leases.release('new-prefix') == {1: 23}


def test_bad_update_and_foreign_release_leave_ownership_unchanged():
    leases = PrefillPageLeases()
    leases.acquire('a', {1: 7})
    with pytest.raises(ValueError, match='changed physical ownership'):
        leases.acquire('b', {2: 8, 1: 9})
    assert 'b' not in leases.requests and 2 not in leases.pages
    with pytest.raises(ValueError, match="another reader"):
        leases.release('a', [2])
    assert leases.pages[1].references == 1
