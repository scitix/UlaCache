"""Real CPU allocations expose pending-current aliasing without CUDA kernels."""
from types import SimpleNamespace
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec('torch') is None,
    reason='Real workspace storage checks require optional Torch')

from ulacache.cache.local_cache import UnifiedLocalResidency
from ulacache.cache.layers import LayerGroupPlan


def make_cache(enabled, *, full_workspace=False):
    import torch
    def host_factory(*, layers, slots, row_bytes, device):
        storage = torch.empty((layers, slots, row_bytes), dtype=torch.uint8)
        return SimpleNamespace(storage=storage, nbytes=storage.nbytes,
            ptrs=torch.tensor([storage.data_ptr()], dtype=torch.uint64))

    owners = tuple(range(21)) + (20,) * 57 if full_workspace else (0, 0, 2, 3)
    return UnifiedLocalResidency(groups=LayerGroupPlan.from_layer_anchors(owners),
        gpu_slots=64, host_slots=128, row_bytes=8, device='cpu',
        max_queries=33 if full_workspace else 3, topk=2048 if full_workspace else 4,
        max_write_rows=8192 if full_workspace else 17,
        prefill_union_rows=8192 if full_workspace else 32,
        host_pool_factory=host_factory, parallel_prepare=True, parallel_current=enabled)


def workspace_bytes(cache):
    allocations = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                   for work in cache.workspaces for tensor in work if tensor.numel()}
    return sum(allocations.values())


def test_actual_constructor_keeps_every_groups_pending_currents_independent():
    import torch
    cache = make_cache(True)
    groups = cache.groups.group_count
    for field, column in [('new_rows', 9), ('promote_rows', 10)]:
        arrays = [getattr(work, field) for work in cache.workspaces]
        assert len({array.data_ptr() for array in arrays}) == groups
        assert cache.frames[:, column].tolist() == [array.data_ptr() for array in arrays]
        # Domains can have distinct CPU-current subsets and insertion orders.
        for group, array in enumerate(arrays):
            array[:3] = torch.tensor([17 + group, 3 + group, 9 + group])
        pending = [array[:3].clone() for array in arrays]
        # Earlier selection appends may replace its whole union. Every later
        # group's already prepared current entries must remain unchanged.
        arrays[0].fill_(91)
        for group in range(1, groups):
            assert torch.equal(arrays[group][:3], pending[group])
    assert len({work.plan.data_ptr() for work in cache.workspaces}) == 1
    assert len({work.candidate_slots.data_ptr() for work in cache.workspaces}) == 1


def test_disabled_path_keeps_existing_serial_storage_and_zero_extra_bytes():
    cache = make_cache(False)
    for field in ('new_rows', 'promote_rows', 'plan', 'candidate_slots'):
        assert len({getattr(work, field).data_ptr() for work in cache.workspaces}) == 1
    assert cache.current_preparation_scratch_bytes == 0


def test_glm_query_workspace_extra_bytes_match_real_unique_storage():
    serial, parallel = make_cache(False, full_workspace=True), make_cache(True, full_workspace=True)
    assert parallel.workspaces[0].new_rows.numel() == 67617
    assert parallel.groups.group_count == 21
    assert parallel.current_preparation_scratch_bytes == 10818720
    assert workspace_bytes(parallel) - workspace_bytes(serial) == 10818720


def test_contiguous_frames_match_the_actual_shared_cuda_abi():
    from pathlib import Path
    import re

    root = Path(__file__).resolve().parents[3]
    header = (root / 'ulacache/csrc/cache/residency_fused.cuh').read_text()
    fields = re.search(r'enum FrameField\s*\{([^}]+)\}', header).group(1)
    fields = [field.strip() for field in fields.split(',') if field.strip()]
    assert fields.pop() == 'FFrameWords'
    cache = make_cache(True)
    assert cache.frames.shape == (cache.groups.group_count, len(fields))
    assert cache.frames[:, fields.index('FDeferredEpoch')].tolist() == [0] * cache.groups.group_count
    assert cache.frames[:, fields.index('FPeers')].tolist() == [1] * cache.groups.group_count
    for group, row in enumerate(cache.frames):
        assert row[:17].tolist() == [tensor.data_ptr() for tensor in
                                    (*cache.states[group], *cache.workspaces[group])]


def test_receive_free_stack_ordering_accepts_the_real_constructor_frame_width():
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / 'ulacache/cache/kernels/local_cache.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'order_free_cpu_receive_tail')
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))
            return launch

    namespace = {'_order_free_cpu_receive_tail': Kernel()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    cache = make_cache(False)
    frames = SimpleNamespace(is_cuda=True, is_contiguous=lambda: True,
                             ndim=cache.frames.ndim, shape=cache.frames.shape)
    assert namespace[function.name](frames, 128) is True
    assert launches[0][0] == (cache.groups.group_count,)
    assert launches[0][1][1] == cache.frames.shape[1]
    frames.shape = (cache.groups.group_count, cache.frames.shape[1] - 1)
    with pytest.raises(ValueError, match='frames'):
        namespace[function.name](frames, 128)
