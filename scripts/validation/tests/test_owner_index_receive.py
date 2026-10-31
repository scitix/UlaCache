"""CPU integration of target/draft KV and Index under a native receive lease."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from ulacache.transfer.index_pages import (
    PAGE_BYTES, plan_index_record_copies, publish_index_record_bytes,
)
from ulacache.transfer.owner_index_receive import OwnerIndexRecordReceiveWindow


@pytest.fixture
def make_index_window(monkeypatch):
    original_empty = torch.empty

    def empty(*args, **kwargs):
        kwargs.pop('pin_memory', None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, 'empty', empty)

    def make(rank=0):
        events = []
        monkeypatch.setattr(torch.cuda, 'current_stream',
            lambda device: SimpleNamespace(synchronize=lambda: events.append('sync')))

        def host_storage(*, layers, slots, row_bytes, device):
            data = torch.full((layers, slots, row_bytes), 91, dtype=torch.uint8)
            return SimpleNamespace(storage=data, nbytes=data.nbytes)

        gpu = torch.full((3, 128, 5), 90, dtype=torch.uint8)
        directory = torch.full((2, 256), -1, dtype=torch.int64)
        directory[:, 64:128] = torch.arange(64, 128)
        directory[:, 128:] = -torch.arange(128)-2
        directory[1, 64:128] = -torch.arange(64)-2
        directory[1, 128:192] = torch.arange(64, 128)
        directory[1, 192:] = -torch.arange(64)-66
        engine = SimpleNamespace(gpu=gpu,
            host=host_storage(layers=3, slots=128, row_bytes=5, device='cpu'),
            directory=directory, gpu_slots=128, host_slots=128, row_bytes=5,
            groups=SimpleNamespace(layer_count=3, group=lambda layer: min(layer, 1)),
            _drain_transfers=lambda: events.append('drain'))
        cache = SimpleNamespace(_execution_engine=engine,
            cache_specification=SimpleNamespace(storage_layout=SimpleNamespace(records_per_page=64)))
        window = OwnerIndexRecordReceiveWindow(cache, degree=8, rank=rank,
            record_capacity=640, allocate_host_record_storage=host_storage,
            device='cpu', copy_chunk_records=17)
        return window, events

    return make


@pytest.mark.parametrize('rank', range(8))
def test_21_target_index_banks_replicated_draft_and_kv_share_final_lease(make_index_window, monkeypatch, rank):
    window, events = make_index_window(rank)
    indexes = [torch.full((4, PAGE_BYTES), 0xDB, dtype=torch.uint8) for _ in range(21)]
    draft_index = torch.full((32, PAGE_BYTES), 0xA3, dtype=torch.uint8)
    draft_kv = torch.full((2048, 7), 0x92, dtype=torch.uint8)
    target_infos = window.add_index_page_banks('target_index', indexes)
    draft_infos = window.add_index_page_banks('draft_index', [draft_index], replicated=True)
    window.add_sidecar_record_banks('draft_kv', [draft_kv], page_size=64)
    assert target_infos()[2] == [PAGE_BYTES] * 21
    assert draft_infos()[2] == [PAGE_BYTES]
    pages = [8, 19, 11, 26, 9, 24, 18, 30]
    logical = [page * 64 + offset for page in pages for offset in range(64)][:509]
    window.reserve('room', torch.tensor(pages), record_count=509)
    item = window._pending['room']
    received = item['source_rows'].tolist()
    expected_gpu, expected_host = window.engine.gpu.clone(), window.engine.host.storage.clone()
    for layer in range(3):
        payload = torch.arange(509*5).reshape(509, 5).add(layer*73).remainder(256).to(torch.uint8)
        window.storage.storage[layer, received] = payload
        for position, row in enumerate(logical):
            if row % 8 != rank:
                continue
            location = window.engine.directory[min(layer, 1), row//8].item()
            if location >= 0:
                expected_gpu[layer, location] = payload[position]
            else:
                expected_host[layer, -location-2] = payload[position]
    draft_payload = torch.arange(509*7).reshape(509, 7).remainder(256).to(torch.uint8)
    window._sidecars['draft_kv']['storage'].storage[0, received] = draft_payload
    expected_draft = draft_kv.clone()
    expected_draft[logical] = draft_payload
    expected_index = []
    for name, banks, degree, owner in [
            ('target_index', indexes, 8, rank), ('draft_index', [draft_index], 1, 0)]:
        group = window._index_banks[name]
        plan = plan_index_record_copies(logical, received, degree=degree, rank=owner)
        for number, bank in enumerate(banks):
            source = group['storage'].storage[number]
            source.copy_(torch.arange(source.numel()).reshape(source.shape).add(number*17).remainder(256).to(torch.uint8))
            expected = bytearray(bank.reshape(-1).tolist())
            publish_index_record_bytes(bytes(source.reshape(-1).tolist()), expected, plan,
                transport_quiesced=True)
            expected_index.append((bank, expected))
    assert all(torch.all(bank == 0xDB) for bank in indexes)
    assert torch.all(draft_index == 0xA3) and torch.all(draft_kv == 0x92)

    def sync():
        # Neither target/draft KV nor either Index group may retire this lease
        # before the last canonical write and the native current-stream fence.
        assert window.available_records == 128
        for bank, expected in expected_index:
            assert bytes(bank.reshape(-1).tolist()) == expected
        events.append('sync')

    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: SimpleNamespace(synchronize=sync))
    assert window.publish('room') == 509
    assert torch.equal(window.engine.gpu, expected_gpu)
    assert torch.equal(window.engine.host.storage, expected_host)
    assert torch.equal(draft_kv, expected_draft)
    assert window.available_records == 640 and events == ['drain', 'sync']
    inventory = window.memory_inventory()
    assert inventory['index_receive_bank_count'] == 22
    assert inventory['cpu_receive_bytes'] == 3*640*5 + 640*7 + 22*10*PAGE_BYTES


def test_index_capacity_and_duplicate_banks_fail_before_lease_or_registration(make_index_window):
    window, _ = make_index_window()
    bank = torch.zeros((1, PAGE_BYTES), dtype=torch.uint8)
    with pytest.raises(ValueError, match='overlap'):
        window.add_index_page_banks('duplicate', [bank, bank])
    window.add_index_page_banks('small', [bank])
    with pytest.raises(ValueError, match='exceeds'):
        window.reserve('overflow', torch.tensor([8]), record_count=1)
    assert window.available_records == 640
    with pytest.raises(ValueError, match='unique'):
        window.add_sidecar_record_banks('small', [torch.zeros((2048, 7), dtype=torch.uint8)], page_size=64)
    window.contiguous_buffer_infos()
    with pytest.raises(RuntimeError, match='registration'):
        window.add_index_page_banks('late', [bank])


def test_aborted_index_receive_retains_common_lease_and_preserves_canonical_bytes(make_index_window):
    window, events = make_index_window()
    bank = torch.full((4, PAGE_BYTES), 0xDB, dtype=torch.uint8)
    window.add_index_page_banks('index', [bank])
    lease = window.reserve('abort', torch.tensor([8, 9]), record_count=125)
    window._index_banks['index']['storage'].storage.fill_(0xF7)
    window.logical_pages_freed([8])
    assert window.available_records == 512
    other = window.reserve('other', torch.tensor([10]), record_count=64)
    assert not set(lease.pages) & set(other.pages)
    window.logical_pages_freed([9])
    assert window.available_records == 576
    assert torch.all(bank == 0xDB) and events == []
