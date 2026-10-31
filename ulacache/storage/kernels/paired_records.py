"""Resident K/V address gathering; native attention owns all computation."""

import triton
import triton.language as tl


@triton.jit
def compose_record_ids_kernel(
    table,
    requests,
    positions,
    lengths,
    output,
    WIDTH: tl.constexpr,
    CONTEXT: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(requests + row).to(tl.int64)
    length = tl.load(lengths + row)
    position = tl.load(
        positions + row * POSITION_STRIDE + column, column < WIDTH, other=-1
    )
    valid = (position >= 0) & (position < length) & (position < CONTEXT)
    record = tl.load(
        table + request * REQUEST_STRIDE + tl.maximum(position, 0),
        valid & (column < WIDTH),
        other=-1,
    )
    tl.store(output + row * WIDTH + column, record.to(tl.int32), column < WIDTH)


@triton.jit
def gather_pairs_kernel(
    key,
    value,
    records,
    positions,
    lengths,
    offsets,
    out_key,
    out_value,
    SELECTED: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    VALUE_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    columns: tl.constexpr,
    PADDED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    column = tile * 16 + tl.arange(0, 16)
    feature = tl.arange(0, BLOCK_D)
    start = tl.load(offsets + row)
    span = tl.load(offsets + row + 1) - start
    length = tl.load(lengths + row)
    position = tl.load(
        positions + row * POSITION_STRIDE + column, column < SELECTED, other=-1
    )
    slot = tl.load(records + row * SELECTED + column, column < SELECTED, other=-1)
    native_valid = (column < span) & (position >= 0) & (position < length)
    valid = native_valid & (slot >= 0)
    source_key = slot.to(tl.int64)[:, None] * KEY_STRIDE + head * DIM + feature[None, :]
    source_value = (
        slot.to(tl.int64)[:, None] * VALUE_STRIDE + head * DIM + feature[None, :]
    )
    destination = (
        (start + column).to(tl.int64)[:, None] * HEADS * DIM
        + head * DIM
        + feature[None, :]
    )
    load_mask = valid[:, None] & (feature[None, :] < DIM)
    if PADDED:
        store_mask = (column < columns)[:, None] & (feature[None, :] < DIM)
    else:
        store_mask = native_valid[:, None] & (feature[None, :] < DIM)
    tl.store(
        out_key + destination, tl.load(key + source_key, load_mask, other=0), store_mask
    )
    tl.store(
        out_value + destination,
        tl.load(value + source_value, load_mask, other=0),
        store_mask,
    )
