"""Real registered-memory byte oracles for local and owner-striped receive."""
import bisect
import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('width', [576, 656])
@pytest.mark.parametrize('degree,rank', [(1, 0), (2, 0), (2, 1), (8, 0), (8, 7)])
def test_mapped_publication_preserves_owned_rows_sidecars_and_unallocated_tail(width, degree, rank):
    from ulacache.transfer.receive_window import PinnedRecordReceiveWindow
    from ulacache.transfer.owner_receive_window import OwnerStripedRecordReceiveWindow

    device = torch.device('cuda', torch.cuda.current_device())
    torch.empty(1, device=device)
    libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/libcudart.so' in line}
    assert len(libraries) == 1
    runtime = ctypes.CDLL(next(iter(libraries)))
    get_pointer = runtime.cudaHostGetDevicePointer
    get_pointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    get_pointer.restype = ctypes.c_int

    def host_storage(*, layers, slots, row_bytes, device):
        tensor = torch.empty((layers, slots, row_bytes), dtype=torch.uint8, pin_memory=True)
        pointer = ctypes.c_void_p()
        assert get_pointer(ctypes.byref(pointer), ctypes.c_void_p(tensor.data_ptr()), 0) == 0
        return SimpleNamespace(storage=tensor, nbytes=tensor.nbytes,
            ptrs=torch.tensor([pointer.value], dtype=torch.uint64, device=device))

    slots, layers, count = 1024, 3, 131
    gpu = torch.full((layers, slots, width), 91, dtype=torch.uint8, device=device)
    host = host_storage(layers=layers, slots=slots, row_bytes=width, device=device)
    host.storage.fill_(92)
    directory = torch.full((2, 2 * slots), -1, dtype=torch.int64, device=device)
    groups = SimpleNamespace(layer_count=layers, group=lambda layer: bisect.bisect_right([0, 1], layer)-1)
    engine = SimpleNamespace(gpu=gpu, host=host, directory=directory, groups=groups,
        gpu_slots=slots, host_slots=slots, row_bytes=width,
        _drain_transfers=lambda: torch.cuda.synchronize(device))
    cache = SimpleNamespace(_execution_engine=engine,
        cache_specification=SimpleNamespace(storage_layout=SimpleNamespace(records_per_page=64)))
    kwargs = dict(record_capacity=512, allocate_host_record_storage=host_storage,
                  device=device, copy_chunk_records=65)
    window = (PinnedRecordReceiveWindow(cache, **kwargs) if degree == 1 else
              OwnerStripedRecordReceiveWindow(cache, degree=degree, rank=rank, **kwargs))
    assert window._publication_layer_groups is not None
    sidecars = [torch.full((degree * 2 * slots, 17), 77, dtype=torch.uint8, device=device)
                for _ in range(2)]
    window.add_sidecar_record_banks('draft', sidecars, page_size=64)
    first, blocker = window.pages.reserve('first', 2), window.pages.reserve('blocker', 1)
    window.pages.release(first, transport_quiesced=True)
    pages = torch.tensor([5 * degree, 2 * degree, 8 * degree], device=device)
    window.reserve('room', pages, record_count=count)
    window.pages.release(blocker, transport_quiesced=True)
    item = window._pending['room']
    assert item['lease'].pages == (0, 1, 3)
    logical = item['rows'].cpu()
    owned = (logical % degree == rank).nonzero().flatten()
    target = logical[owned] // degree
    source_rows = item['source_rows']
    generator = torch.Generator().manual_seed(109 + degree + rank + width)
    incoming = torch.randint(0, 256, (layers, count, width), dtype=torch.uint8, generator=generator)
    for layer in range(layers):
        window.storage.storage[layer].index_copy_(0, source_rows, incoming[layer])
    draft = window._sidecars['draft']
    expected_drafts = []
    for bank, destination in enumerate(sidecars):
        payload = torch.randint(0, 256, (count, 17), dtype=torch.uint8, generator=generator)
        draft['storage'].storage[bank].index_copy_(0, source_rows, payload)
        expected = destination.cpu()
        expected.index_copy_(0, logical, payload)
        expected_drafts.append(expected)
    expected_gpu, expected_host = gpu.cpu(), host.storage.clone()
    # Resolve a changed mixed residency after reservation, independently per group.
    locations = []
    for group in range(2):
        positions = torch.arange(target.numel())
        hot = (positions + group) % 2 == 0
        location = torch.where(hot, positions, -positions-2)
        directory[group, target.to(device)] = location.to(device)
        locations.append((positions, hot))
    for layer in range(layers):
        positions, hot = locations[groups.group(layer)]
        payload = incoming[layer].index_select(0, owned)
        expected_gpu[layer].index_copy_(0, positions[hot], payload[hot])
        expected_host[layer].index_copy_(0, positions[~hot], payload[~hot])
    assert torch.all(gpu == 91) and torch.all(host.storage == 92)
    assert window.publish('room') == count
    assert torch.equal(gpu.cpu(), expected_gpu)
    assert torch.equal(host.storage, expected_host)
    assert all(torch.equal(actual.cpu(), expected) for actual, expected in zip(sidecars, expected_drafts))
    assert not window._pending and window.available_records == 512


def test_receive_count_variation_reuses_one_compiled_program():
    from ulacache.transfer.kernels.receive_publication import _publish_records

    device = torch.device('cuda', torch.cuda.current_device())
    width, layers, slots = 656, 2, 512
    # Reuse the registered pointer provider exercised by the ownership cases.
    tensor = torch.empty((layers, slots, width), dtype=torch.uint8, pin_memory=True)
    target = torch.empty_like(tensor, pin_memory=True)
    libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/libcudart.so' in line}
    assert len(libraries) == 1
    runtime = ctypes.CDLL(next(iter(libraries)))
    get_pointer = runtime.cudaHostGetDevicePointer
    get_pointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    get_pointer.restype = ctypes.c_int

    def pointer_table(value):
        pointer = ctypes.c_void_p()
        assert get_pointer(ctypes.byref(pointer), ctypes.c_void_p(value.data_ptr()), 0) == 0
        return torch.tensor([pointer.value], dtype=torch.uint64, device=device)

    source_ptrs, host_ptrs = pointer_table(tensor), pointer_table(target)
    gpu = torch.empty((layers, slots, width), dtype=torch.uint8, device=device)
    groups = torch.tensor([0, 1], dtype=torch.int32, device=device)
    programs = set()
    for count in [1, 17, 65, 131, 385]:
        rows = torch.arange(count, dtype=torch.int64, device=device)
        location = torch.where(rows % 2 == 0, rows, -rows - 2)
        locations = torch.stack([location, location])
        payload = torch.randint(0, 256, (layers, count, width), dtype=torch.uint8)
        tensor[:, :count].copy_(payload)
        compiled = _publish_records[(count, layers)](rows, locations, groups,
            source_ptrs, host_ptrs, gpu, count, tensor.stride(0), target.stride(0),
            gpu.stride(0), width, 1024, num_warps=4)
        torch.cuda.synchronize(device)
        programs.add(compiled.hash)
        hot = torch.arange(count) % 2 == 0
        assert torch.equal(gpu[:, :count].cpu()[:, hot], payload[:, hot])
        assert torch.equal(target[:, :count][:, ~hot], payload[:, ~hot])
    assert len(programs) == 1, 'Receive lengths must not create separately compiled kernels'
