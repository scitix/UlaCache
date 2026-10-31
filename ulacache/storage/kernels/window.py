# SPDX-License-Identifier: Apache-2.0
"""Original bounded native-page gather/remap kernels."""

import triton
import triton.language as tl


@triton.jit
def _build_map(
    DIRECTORY, KEYS, VALUES, N: tl.constexpr, SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(DIRECTORY + row, row < N, other=-1)
    valid = (row < N) & (value >= 0)
    slot = (value.to(tl.uint32) * 2654435761) & (SIZE - 1)
    address = tl.where(valid, slot, SIZE)
    empty = tl.full((BLOCK,), -1, tl.int32)
    previous = tl.atomic_cas(
        KEYS + address, empty, tl.where(valid, value, -2), sem="relaxed"
    )
    pending = valid & (previous != -1) & (previous != value)
    while tl.sum(pending.to(tl.int32), 0) > 0:
        slot = tl.where(pending, (slot + 1) & (SIZE - 1), slot)
        address = tl.where(pending, slot, SIZE)
        previous = tl.atomic_cas(
            KEYS + address, empty, tl.where(pending, value, -2), sem="relaxed"
        )
        pending = pending & (previous != -1) & (previous != value)
    # Stable sorting puts current records after a duplicate historical row.
    # Choose the last representative, so its local staging data wins.
    tl.atomic_max(VALUES + slot, row, valid, sem="relaxed")


@triton.jit
def _map_selected(
    SOURCE,
    LENGTHS,
    KEYS,
    VALUES,
    OUT,
    SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    value = tl.load(SOURCE + query * WIDTH + column, column < WIDTH, other=-1)
    valid = (column < WIDTH) & (column < tl.load(LENGTHS + query)) & (value >= 0)
    slot = (value.to(tl.uint32) * 2654435761) & (SIZE - 1)
    key = tl.load(KEYS + slot, valid, other=-1)
    pending = valid & (key != -1) & (key != value)
    while tl.sum(pending.to(tl.int32), 0) > 0:
        slot = tl.where(pending, (slot + 1) & (SIZE - 1), slot)
        key = tl.load(KEYS + slot, valid, other=-1)
        pending = pending & (key != -1) & (key != value)
    position = tl.load(VALUES + slot, valid & (key == value), other=-1)
    tl.store(
        OUT + query * WIDTH + column,
        tl.where(valid & (key == value), position, -1),
        column < WIDTH,
    )


@triton.jit
def _gather_window(
    DIRECTORY,
    FRESH,
    STAGING,
    PEERS,
    OUT,
    FIRST: tl.constexpr,
    COUNT: tl.constexpr,
    WINDOW: tl.constexpr,
    BASE: tl.constexpr,
    DEGREE: tl.constexpr,
    LAYER_BYTES: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    word = tl.arange(0, 128)
    active = (row < WINDOW) & (FIRST + row < COUNT)
    native = tl.load(DIRECTORY + FIRST + row, active, other=0).to(tl.int64)
    fresh = tl.load(FRESH + FIRST + row, active, other=-1).to(tl.int64)
    logical = tl.maximum(native, 0) + BASE
    owner, local = logical % DEGREE, logical // DEGREE
    peer = tl.load(PEERS + owner, active, other=0)
    page = tl.maximum(fresh, 0) // 64
    offset = tl.where(
        word < 72,
        tl.maximum(fresh, 0)[:, None] % 64 * 576 + word[None, :] * 8,
        64 * 576 + tl.maximum(fresh, 0)[:, None] % 64 * 8,
    )
    staging_address = STAGING.to(tl.uint64) + page[:, None] * PAGE_BYTES + offset
    peer_address = (
        peer[:, None] + LAYER_BYTES + local[:, None] * 584 + word[None, :] * 8
    )
    address = tl.where(fresh[:, None] >= 0, staging_address, peer_address)
    value = tl.load(
        address.to(tl.pointer_type(tl.uint64)),
        active[:, None] & (native[:, None] > 0) & (word[None, :] < 73),
        other=0,
    )
    target = row[:, None] // 64 * PAGE_BYTES + tl.where(
        word[None, :] < 72,
        row[:, None] % 64 * 576 + word[None, :] * 8,
        64 * 576 + row[:, None] % 64 * 8,
    )
    tl.store(
        (OUT.to(tl.uint64) + target).to(tl.pointer_type(tl.uint64)),
        value,
        (row[:, None] < WINDOW) & (word[None, :] < 73),
    )
