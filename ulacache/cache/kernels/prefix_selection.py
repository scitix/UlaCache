"""Exact prefill membership compaction without a CPU-visible output length."""

import triton
import triton.language as tl


@triton.jit
def _union(
    Selected,
    Marks,
    Output,
    Count,
    N,
    Logical: tl.constexpr,
    Capacity: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(Selected + i, i < N, other=-1).to(tl.int32)
    tl.device_assert(
        (i >= N) | (row <= 0) | (row < Logical),
        "Prefill identity exceeds logical capacity",
    )
    valid = (i < N) & (row > 0) & (row < Logical)
    # The atomic old value assigns each identity to exactly one producer,
    # including repeats inside a CTA and repeats across different queries.
    prior = tl.atomic_or(Marks + tl.where(valid, row, 0), 1, mask=valid, sem="relaxed")
    won = valid & (prior == 0)
    n = tl.sum(won.to(tl.int32), 0)
    base = tl.atomic_add(Count, n, sem="relaxed")
    rank = tl.cumsum(won.to(tl.int32), 0) - 1
    tl.device_assert(
        base + n <= Capacity, "Prefill union exceeds startup residency bound"
    )
    tl.store(Output + base + rank, row, won & (base + rank < Capacity))


def compact_prefill_union(selected, marks, output, count):
    """Return padded unique positive IDs and a device-only count.

    Membership is exact; output order is unspecified. The caller serializes
    scratch reuse with native attention on the same stream. No .item(),
    boolean-index output allocation, sort, or device-to-host copy is used.
    """
    if not selected.is_contiguous() or selected.ndim != 2:
        raise ValueError("Native prefill TopK must be contiguous [query, selected]")
    marks.zero_()
    output.fill_(-1)
    count.zero_()
    if selected.numel():
        _union[(triton.cdiv(selected.numel(), 256),)](
            selected,
            marks,
            output,
            count,
            selected.numel(),
            marks.numel(),
            output.numel(),
            256,
        )
    return output.view(1, -1), count
