"""Real byte-copy ownership tests on CPU; native CUDA/RDMA gates are separate."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from ulacache.transfer.receive_window import PinnedRecordReceiveWindow


@pytest.fixture
def window(monkeypatch):
    # CPU execution of the publication algorithm needs no CUDA allocator or
    # stream. Only pinned allocation and final device synchronization are stubbed.
    original_empty = torch.empty

    def empty(*args, **kwargs):
        kwargs.pop('pin_memory', None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, 'empty', empty)
    monkeypatch.setattr(torch.cuda, 'current_stream',
        lambda device: SimpleNamespace(synchronize=lambda: None))

    def host_storage(*, layers, slots, row_bytes, device):
        data = torch.zeros((layers, slots, row_bytes), dtype=torch.uint8)
        return SimpleNamespace(storage=data, nbytes=data.nbytes)

    gpu = torch.full((2, 8, 3), 91, dtype=torch.uint8)
    host = host_storage(layers=2, slots=8, row_bytes=3, device='cpu')
    host.storage.fill_(92)
    # Incoming logical rows 8,9 map to CPU slots 0,1, and 4 maps to GPU slot 2.
    directory = torch.full((1, 16), -1, dtype=torch.int64)
    directory[0, 8:10] = torch.tensor([-2, -3])
    directory[0, 4] = 2
    engine = SimpleNamespace(gpu=gpu, host=host, directory=directory,
        row_bytes=3, gpu_slots=8, host_slots=8, _drain_transfers=lambda: None,
        groups=SimpleNamespace(layer_count=2, group=lambda layer: 0))
    cache = SimpleNamespace(_execution_engine=engine,
        cache_specification=SimpleNamespace(storage_layout=SimpleNamespace(records_per_page=2)))
    return PinnedRecordReceiveWindow(cache, record_capacity=8,
        allocate_host_record_storage=host_storage, device='cpu', copy_chunk_records=2)


def test_target_and_sidecar_share_staging_pages_without_touching_live_destinations(window):
    destination = torch.full((16, 1, 5), 77, dtype=torch.uint8)
    infos = window.add_sidecar_record_banks('draft', [destination], page_size=2)
    ptrs, sizes, strides = infos()
    group = window._sidecars['draft']
    assert ptrs == [group['storage'].storage[0].data_ptr()]
    assert ptrs != [destination.data_ptr()] and sizes == [40] and strides == [10]
    gpu_before, host_before, draft_before = window.engine.gpu.clone(), window.engine.host.storage.clone(), destination.clone()
    # Nonmonotonic pages and an unallocated final tail exercise real native ids.
    window.reserve('room', torch.tensor([4, 2]), record_count=3)
    assert window.transfer_pages('room') == (0, 1)
    window.storage.storage[:, :3] = torch.arange(18, dtype=torch.uint8).reshape(2, 3, 3)
    group['storage'].storage[:, :3] = torch.arange(15, dtype=torch.uint8).reshape(1, 3, 5)
    assert torch.equal(destination, draft_before)
    assert torch.equal(window.engine.gpu, gpu_before)
    assert torch.equal(window.engine.host.storage, host_before)
    assert window.publish('room') == 3
    logical = torch.tensor([8, 9, 4])
    assert torch.equal(destination[logical].flatten(1), group['storage'].storage[0, :3])
    untouched = torch.ones(16, dtype=torch.bool)
    untouched[logical] = False
    assert torch.equal(destination[untouched], draft_before[untouched])
    assert torch.equal(destination[5], draft_before[5])
    for layer in range(2):
        assert torch.equal(window.engine.host.storage[layer, :2], window.storage.storage[layer, :2])
        assert torch.equal(window.engine.gpu[layer, 2], window.storage.storage[layer, 2])
        assert torch.equal(window.engine.gpu[layer, :2], gpu_before[layer, :2])
    assert window.available_records == 8


def test_aborted_sidecar_lease_is_not_reused_until_all_logical_pages_retire(window):
    destination = torch.full((16, 5), 77, dtype=torch.uint8)
    window.add_sidecar_record_banks('draft', [destination], page_size=2)
    first = window.reserve('aborted', torch.tensor([4, 2]), record_count=3)
    window.logical_pages_freed([4])
    assert window.available_records == 4
    other = window.reserve('other', torch.tensor([5]), record_count=1)
    assert not set(first.pages) & set(other.pages)
    window.logical_pages_freed([2])
    assert window.available_records == 6
    assert torch.all(destination == 77)


def test_receive_bank_registration_and_bounds_fail_before_leasing(window):
    destination = torch.full((8, 5), 77, dtype=torch.uint8)
    getter = window.add_sidecar_record_banks('draft', [destination], page_size=2)
    with pytest.raises(ValueError, match='capacity'):
        window.reserve('out-of-range', torch.tensor([4, 1]), record_count=3)
    assert window.available_records == 8 and not window._pending
    getter()
    with pytest.raises(RuntimeError, match='registration'):
        window.add_sidecar_record_banks('late', [destination], page_size=2)


def test_inventory_includes_sidecar_host_staging_and_device_scratch(window):
    before = window.memory_inventory()
    window.add_sidecar_record_banks('draft', [torch.zeros((16, 1, 5), dtype=torch.uint8)], page_size=2)
    after = window.memory_inventory()
    assert after['cpu_receive_bytes'] - before['cpu_receive_bytes'] == 8 * 5
    assert after['cpu_copy_scratch_bytes'] - before['cpu_copy_scratch_bytes'] == 2 * 5
    assert after['gpu_receive_metadata_bytes'] - before['gpu_receive_metadata_bytes'] == 2 * 5
    assert after['sidecar_bank_count'] == 1


def test_publication_resolves_locations_after_reservation(window):
    window.reserve('relocated', torch.tensor([4, 2]), record_count=3)
    incoming = torch.arange(18, dtype=torch.uint8).reshape(2, 3, 3)
    window.storage.storage[:, :3].copy_(incoming)
    # Eviction/promotion may change destinations while RDMA is still in flight.
    window.engine.directory[0, torch.tensor([8, 9, 4])] = torch.tensor([5, 6, -5])
    assert window.publish('relocated') == 3
    assert torch.equal(window.engine.gpu[:, 5:7], incoming[:, :2])
    assert torch.equal(window.engine.host.storage[:, 3], incoming[:, 2])
    assert torch.all(window.engine.gpu[:, 2] == 91)
    assert torch.all(window.engine.host.storage[:, :2] == 92)


def test_failed_visibility_boundary_retains_the_receive_lease(window, monkeypatch):
    window.reserve('unfinished', torch.tensor([4]), record_count=2)
    window.storage.storage[:, :2].fill_(17)

    def failed_sync():
        raise RuntimeError('visibility boundary not completed')

    monkeypatch.setattr(torch.cuda, 'current_stream',
        lambda device: SimpleNamespace(synchronize=failed_sync))
    with pytest.raises(RuntimeError, match='visibility boundary'):
        window.publish('unfinished')
    assert 'unfinished' in window._pending
    assert window.available_records == 6
