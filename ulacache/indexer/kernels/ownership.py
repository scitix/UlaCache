"""Native Index owner write mapping and independent-consumer score restore."""

import triton
import triton.language as tl


@triton.jit
def _map_index_write_records(
    Logical,
    Local,
    Count: tl.constexpr,
    Capacity: tl.constexpr,
    Reserved: tl.constexpr,
    Rank: tl.constexpr,
    Degree: tl.constexpr,
    Block: tl.constexpr,
):
    offset = tl.arange(0, Block)
    record = tl.load(Logical + offset, offset < Count, -1).to(tl.int64)
    live = (offset < Count) & (record >= Reserved)
    tl.device_assert(
        (~live) | (record < Capacity), "Index write exceeds the logical namespace"
    )
    first_guard = tl.min(
        tl.where((offset < Count) & (record == 0), offset, Count), axis=0
    )
    guard = (record == 0) & (offset == first_guard)
    owned = guard | (live & (record % Degree == Rank) & (record < Capacity))
    tl.store(Local + offset, tl.where(owned, record // Degree, -1), offset < Count)


def map_index_write_records(logical_records, local_records, *, ownership, owner_rank):
    import torch

    ownership.validate_rank(owner_rank)
    if (
        logical_records.device.type != "cuda"
        or local_records.device != logical_records.device
        or logical_records.dtype not in (torch.int32, torch.int64)
        or local_records.dtype != torch.int64
        or logical_records.shape != local_records.shape
        or not logical_records.is_contiguous()
        or not local_records.is_contiguous()
    ):
        raise ValueError(
            "Native Index mapping needs matching contiguous CUDA integer records"
        )
    count = logical_records.numel()
    if count:
        _map_index_write_records[(1,)](
            logical_records,
            local_records,
            count,
            ownership.logical_record_capacity,
            ownership.reserved_record_count,
            owner_rank,
            ownership.owner_count,
            triton.next_power_of_2(count),
        )


@triton.jit
def _restore_consumer_scores(
    Gathered,
    Lengths,
    Output,
    Rows: tl.constexpr,
    Width: tl.constexpr,
    LocalWidth: tl.constexpr,
    OutputStride: tl.constexpr,
    Degree: tl.constexpr,
    Consumer: tl.constexpr,
    Block: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * Block + tl.arange(0, Block)
    length = tl.load(Lengths + row)
    tl.device_assert(
        (length >= 0) & (length <= Width), "Index visibility exceeds its score bound"
    )
    visible = (column < Width) & (column < length)
    owner = column % Degree
    owner_column = column // Degree
    consumer_query = Consumer * Rows + row
    source = (
        owner.to(tl.int64) * Degree * Rows + consumer_query
    ) * LocalWidth + owner_column
    score = tl.load(Gathered + source, visible, -float("inf"))
    tl.store(
        Output + row.to(tl.int64) * OutputStride + column, score, column < OutputStride
    )


def restore_consumer_scores(
    gathered_scores, lengths, output, *, logical_width, consumer_rank
):
    """Restore [owner, consumer*query, local-column] to native logical rows.

    Every consumer retains its own queries and receives its own score rows.
    Token stripes restore original column order before the unchanged native
    top-k. Device lengths mask stale capacity tails on every Graph replay.
    Output rows may include the native candidate kernel's alignment padding.
    """
    import torch

    if gathered_scores.ndim != 3 or output.ndim != 2:
        raise ValueError("Declare rank-major scores and two-dimensional native output")
    degree, consumer_queries, local_width = gathered_scores.shape
    rows = output.shape[0]
    if (
        degree < 2
        or degree & (degree - 1)
        or consumer_queries != degree * rows
        or type(consumer_rank) is not int
        or not 0 <= consumer_rank < degree
        or type(logical_width) is not int
        or not 0 < logical_width <= degree * local_width
        or output.shape[1] < logical_width
        or lengths.shape != (rows,)
        or lengths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Independent consumer Index score geometry disagrees")
    if (
        gathered_scores.device.type != "cuda"
        or output.device != gathered_scores.device
        or lengths.device != gathered_scores.device
        or gathered_scores.dtype != torch.float32
        or output.dtype != torch.float32
        or not gathered_scores.is_contiguous()
        or not output.is_contiguous()
        or not lengths.is_contiguous()
    ):
        raise ValueError(
            "Index restore requires contiguous CUDA FP32 scores and device lengths"
        )
    if rows:
        _restore_consumer_scores[(rows, triton.cdiv(output.shape[1], 256))](
            gathered_scores,
            lengths,
            output,
            rows,
            logical_width,
            local_width,
            output.shape[1],
            degree,
            consumer_rank,
            256,
        )
