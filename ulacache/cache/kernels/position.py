"""One stable owner-local selection transform per shared-Index anchor."""

import triton
import triton.language as tl


@triton.jit
def _selection(
    Selected,
    Local,
    Counts,
    Active,
    K: tl.constexpr,
    Stride: tl.constexpr,
    Rank: tl.constexpr,
    Degree: tl.constexpr,
    Capacity: tl.constexpr,
    Block: tl.constexpr,
    HasActive: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, Block)
    slot = tl.load(Selected + row * Stride + col, col < K, other=-1).to(tl.int64)
    active = True
    if HasActive:
        active = row < tl.load(Active)
    owned = (
        active & (col < K) & (slot > 0) & (slot % Degree == Rank) & (slot // Degree > 0)
    )
    local = slot // Degree
    tl.device_assert(
        (~owned) | (local < Capacity), "CP selection exceeds its owner capacity"
    )
    count = tl.sum(owned.to(tl.int32))
    destination = tl.cumsum(owned.to(tl.int32)) - 1
    # Prefix and padding writes are disjoint, so no clearing launch or
    # intra-block write-after-write barrier is necessary.
    tl.store(Local + row * K + col, -1, (col < K) & (col >= count))
    tl.store(Local + row * K + destination, local.to(tl.int32), owned)
    tl.store(Counts + row, count)


def localize_selection(selected, output, counts, *, layout, active_queries=None):
    import torch

    if (
        selected.ndim != 2
        or selected.dtype not in (torch.int32, torch.int64)
        or not selected.is_cuda
        or selected.stride(1) != 1
        or output.shape != selected.shape
        or not output.is_contiguous()
        or output.dtype != torch.int32
        or counts.dtype != torch.int32
        or counts.shape != (selected.shape[0],)
        or any(t.device != selected.device for t in (output, counts))
    ):
        raise ValueError(
            "CP selection requires native CUDA rows and fixed local int32 workspaces"
        )
    if selected.shape[0]:
        _selection[(selected.shape[0],)](
            selected,
            output,
            counts,
            active_queries,
            selected.shape[1],
            selected.stride(0),
            layout.rank,
            layout.degree,
            layout.local_records,
            triton.next_power_of_2(selected.shape[1]),
            active_queries is not None,
        )
