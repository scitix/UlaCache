"""CPU byte oracles for cached routing of quiescent snapshot records."""

import pytest
import importlib.util

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                reason="Snapshot byte oracles require optional Torch")

from ulacache.tools import SnapshotLayout, SnapshotRequest
from ulacache.tools.snapshot.cuda import PoolSnapshotBindings
from ulacache.tools.snapshot.cuda import FastCudaRequestSnapshot


def test_group_routes_preserve_gpu_and_cpu_bytes_across_shared_layers():
    import torch
    layout = SnapshotLayout("routing", 1, 3, 5, (0, 2), (2, 1), 4)
    gpu = torch.zeros((3, 8, 5), dtype=torch.uint8)
    cpu = torch.zeros_like(gpu)
    index = {anchor: torch.zeros((4, 12), dtype=torch.uint8) for anchor in (0, 2)}
    directories = torch.full((2, 8), -1, dtype=torch.int64)
    directories[0, 1:7] = torch.tensor([3, -2, 0, -4, 5, -7])
    directories[1, 1:7] = torch.tensor([-3, 2, -2, 0, -5, 7])
    backend = FastCudaRequestSnapshot(PoolSnapshotBindings(layout, 0, gpu, index,
        (0, 0, 1), cpu_mla=cpu, directories=directories, quiesce=lambda: None))
    request = SnapshotRequest("r", tuple(range(6)), (42,))
    rows = backend._owned(request, [1, 2, 3, 4, 5, 6])
    for layer in range(3):
        data = torch.arange(30, dtype=torch.uint8).reshape(6, 5) + layer * 31
        for start in (0, 3):
            backend._mla_write(layer, rows[start:start + 3], data[start:start + 3])
            assert torch.equal(backend._mla_read(layer, rows[start:start + 3]), data[start:start + 3])
        for offset, location in enumerate(directories[(0, 0, 1)[layer], 1:7].tolist()):
            storage, slot = (gpu, location) if location >= 0 else (cpu, -location - 2)
            assert torch.equal(storage[layer, slot], data[offset])
    assert len(backend._group_locations) == 2
    assert len(backend._chunk_routes) == 4
