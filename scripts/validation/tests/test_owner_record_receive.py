"""CPU tensor publication for owner-local target KV and replicated draft KV."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from ulacache.transfer.owner_receive_window import OwnerStripedRecordReceiveWindow


@pytest.fixture
def make_window(monkeypatch):
    original_empty = torch.empty

    def empty(*args, **kwargs):
        kwargs.pop('pin_memory', None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, 'empty', empty)

    def create(rank):
        events = []
        monkeypatch.setattr(torch.cuda, 'current_stream',
            lambda device: SimpleNamespace(synchronize=lambda: events.append('sync')))

        def host_storage(*, layers, slots, row_bytes, device):
            data = torch.full((layers, slots, row_bytes), 91, dtype=torch.uint8)
            return SimpleNamespace(storage=data, nbytes=data.nbytes)

        gpu = torch.full((3, 4, 5), 90, dtype=torch.uint8)
        host = host_storage(layers=3, slots=4, row_bytes=5, device='cpu')
        directory = torch.full((2, 8), -1, dtype=torch.int64)
        directory[:, 2:4] = torch.tensor([[0, -2], [-3, 2]])
        engine = SimpleNamespace(gpu=gpu, host=host, directory=directory,
            gpu_slots=4, host_slots=4, row_bytes=5,
            groups=SimpleNamespace(layer_count=3, group=lambda layer: min(layer, 1)),
            _drain_transfers=lambda: events.append('drain'))
        cache = SimpleNamespace(_execution_engine=engine,
            cache_specification=SimpleNamespace(storage_layout=SimpleNamespace(records_per_page=2)))
        window = OwnerStripedRecordReceiveWindow(cache, degree=8, rank=rank,
            record_capacity=12, allocate_host_record_storage=host_storage,
            device='cpu', copy_chunk_records=2)
        return window, events

    return create


@pytest.mark.parametrize('rank', range(8))
def test_all_owners_publish_only_their_target_rows_but_every_draft_row(make_window, rank):
    window, events = make_window(rank)
    draft = torch.full((64, 1, 7), 92, dtype=torch.uint8)
    window.add_sidecar_record_banks('draft', [draft], page_size=2)
    logical = [16, 17, 26, 27, 18, 19, 30, 31, 22]
    expected_gpu, expected_host, expected_draft = (
        window.engine.gpu.clone(), window.engine.host.storage.clone(), draft.clone())
    window.reserve('room', torch.tensor([8, 13, 9, 15, 11]), record_count=9)
    assert window.transfer_pages('room') == (0, 1, 2, 3, 4)
    for layer in range(3):
        payload = torch.arange(45, dtype=torch.uint8).reshape(9, 5) + layer*73
        window.storage.storage[layer, :9] = payload
        for position, row in enumerate(logical):
            if row % 8 != rank:
                continue
            location = window.engine.directory[min(layer, 1), row // 8].item()
            if location >= 0:
                expected_gpu[layer, location] = payload[position]
            else:
                expected_host[layer, -location-2] = payload[position]
    sidecar = window._sidecars['draft']
    sidecar['storage'].storage[0, :9] = torch.arange(63, dtype=torch.uint8).reshape(9, 7)
    expected_draft[logical, 0] = sidecar['storage'].storage[0, :9]
    assert torch.all(window.engine.gpu == 90) and torch.all(draft == 92)
    assert window.publish('room') == 9
    assert torch.equal(window.engine.gpu, expected_gpu)
    assert torch.equal(window.engine.host.storage, expected_host)
    assert torch.equal(draft, expected_draft)
    assert torch.all(draft[23] == 92)
    assert window.available_records == 12 and events == ['drain', 'sync']
    assert window.memory_inventory()['target_owner_rank'] == rank


def test_owner_publication_uses_residency_after_inflight_moves(make_window):
    window, events = make_window(0)
    window.reserve('room', torch.tensor([8]), record_count=1)
    old_gpu = window.engine.gpu.clone()
    window.engine.directory[:, 2] = torch.tensor([-2, -3])
    window.storage.storage[:, 0] = torch.arange(15, dtype=torch.uint8).reshape(3, 5)
    window.publish('room')
    assert torch.equal(window.engine.gpu, old_gpu)
    assert torch.equal(window.engine.host.storage[0, 0], window.storage.storage[0, 0])
    assert torch.equal(window.engine.host.storage[1:, 1], window.storage.storage[1:, 0])
    assert events == ['drain', 'sync']


def test_owner_capacity_fails_before_receive_lease_and_abort_keeps_live_pages(make_window):
    window, _ = make_window(0)
    with pytest.raises(ValueError, match='capacity'):
        window.reserve('invalid', torch.tensor([32]), record_count=1)
    with pytest.raises(ValueError, match='reserved'):
        window.reserve('padding', torch.tensor([7]), record_count=1)
    assert window.available_records == 12
    lease = window.reserve('aborted', torch.tensor([8, 9]), record_count=3)
    window.logical_pages_freed([8])
    assert window.available_records == 8
    second = window.reserve('other', torch.tensor([10]), record_count=1)
    assert not set(lease.pages) & set(second.pages)
    window.logical_pages_freed([9])
    assert window.available_records == 10
