"""Packed transport group geometry and native compiler ABI regressions."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import ulacache.flashag.packed as flash_ag_packed


@pytest.mark.parametrize("world", [4, 8])
def test_compiler_binds_the_native_namespace_and_group_specialization(monkeypatch, world):
    compiler = Mock()
    monkeypatch.setattr(flash_ag_packed, "load_jit", compiler)
    # Use the uncached request to keep tests independent of prior bindings.
    flash_ag_packed._module.__wrapped__(world)
    assert compiler.call_args.kwargs["cuda_wrappers"] == [
        ("gather", f"udcache::index::packed_all_gather<{world}>"),
        ("resident_grid_limit", f"udcache::index::packed_resident_blocks<{world}>"),
    ]


@pytest.mark.parametrize("world", [1, 2, 16])
def test_unsupported_world_does_not_reach_compiler(monkeypatch, world):
    compiler = Mock()
    monkeypatch.setattr(flash_ag_packed, "load_jit", compiler)
    with pytest.raises(ValueError, match="four or eight"):
        flash_ag_packed._module.__wrapped__(world)
    compiler.assert_not_called()


class Device:
    def __init__(self, value):
        if isinstance(value, Device):
            self.type, self.index = value.type, value.index
        else:
            self.type, self.index = "cuda", 0

    def __eq__(self, other):
        return (self.type, self.index) == (other.type, other.index)


class Tensor:
    def __init__(self, shape, dtype, device):
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.ndim = len(self.shape)
        self.dtype, self.device = dtype, device

    def numel(self):
        result = 1
        for extent in self.shape:
            result *= extent
        return result

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return id(self) * 16


@pytest.fixture
def provider(monkeypatch):
    """Expose startup allocations without importing Torch or opening CUDA IPC."""
    module = Mock()
    module.resident_grid_limit.return_value = 512
    monkeypatch.setattr(flash_ag_packed, "_module", lambda world: module)
    allocations = []

    def allocate(shape, *, dtype, device):
        tensor = Tensor(shape, dtype, device)
        allocations.append(tensor)
        return tensor

    cuda = SimpleNamespace(is_current_stream_capturing=lambda: False,
        get_device_capability=lambda device: (10, 3))
    torch = SimpleNamespace(device=Device, cuda=cuda, float32="fp32",
        uint32="u32", int32="i32", int64="i64", zeros=allocate, empty=allocate, arange=allocate,
        empty_like=lambda tensor: allocate(tensor.shape, dtype=tensor.dtype, device=tensor.device))
    dist = Mock()

    def gather(output, value, *, group):
        assert len(output) == dist.get_world_size.return_value
        output[:] = [value] * len(output)

    dist.all_gather_object.side_effect = gather
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    ipc = SimpleNamespace(FlowKVPeerShards=lambda slab, group, **kwargs: SimpleNamespace(ptrs=object()))
    monkeypatch.setitem(sys.modules, "ulacache.transfer.ipc", ipc)
    for name in ("PROBES", "FORCE_ALGO", "PROBE_GRID", "PROBE_GRIDS"):
        monkeypatch.delenv("SGLANG_FLASH_AG_" + name, raising=False)
    return SimpleNamespace(module=module, torch=torch, dist=dist, allocations=allocations)


@pytest.mark.parametrize("world,columns", [(4, 33536), (8, 16896)])
@pytest.mark.parametrize("mode", ["dense", "compact"])
def test_full_group_buffers_and_active_gather(provider, monkeypatch, world, columns, mode):
    monkeypatch.setenv("UDCACHE_INDEX_SCORE_LAYOUT", mode)
    provider.dist.get_world_size.return_value = world
    provider.dist.get_rank.return_value = 0
    ops = Mock()
    ops.target.return_value = "b300"
    ops.family_of.side_effect = lambda algo, blocks, threads: blocks
    ops.make_layout.return_value = SimpleNamespace(total=256, bases=lambda family: (0, 1, 1))
    group = SimpleNamespace(world_size=world, rank_in_group=0, cpu_group=object())
    owner = flash_ag_packed.PackedFlashAGIndexOwner(group, max_queries=24,
        max_page_width=2084, device="cuda:0", transport_ops=ops, ipc_manager_factory=object)
    assert owner.local_columns == columns
    assert owner.packed_workspace.numel() == 24 * columns * world
    assert owner.gathered_workspace.numel() == 24 * columns * world
    layout_world, ceilings = ops.make_layout.call_args.args
    assert layout_world == world
    assert max(ceilings.values()) == owner.maximum_gather_bytes
    assert owner.unpacks_scores == (mode == "dense")

    local = Tensor((24, columns), "fp32", owner.device)
    output = Tensor((world * 24, columns), "fp32", owner.device)
    output.data_ptr = owner.gathered_workspace.data_ptr
    prefix = Tensor((25,), "i32", owner.device)
    packs = Tensor((1,), "i64", owner.device)
    owner.all_gather_compact(output, local, prefix, packs)
    provider.module.gather.assert_called_once()
    output.shape = ((world + 1) * 24, columns)
    with pytest.raises(ValueError, match="stable aligned"):
        owner.all_gather_compact(output, local, prefix, packs)


def test_hopper_is_still_outside_the_validated_packed_transport(provider):
    provider.dist.get_world_size.return_value = 8
    provider.dist.get_rank.return_value = 0
    provider.torch.cuda.get_device_capability = lambda device: (9, 0)
    group = SimpleNamespace(world_size=8, rank_in_group=0, cpu_group=object())
    with pytest.raises(ValueError, match="B300"):
        flash_ag_packed.PackedFlashAGIndexOwner(group, max_queries=24,
            max_page_width=2084, device="cuda:0", transport_ops=Mock(), ipc_manager_factory=object)
    assert not provider.allocations
