"""Region address expansion and remapping; native scoring and masks stay intact."""

import triton
import triton.language as tl


@triton.jit
def expand_regions_kernel(
    packed,
    counts,
    active,
    output,
    packed_stride: tl.constexpr,
    TOKENS: tl.constexpr,
    REGIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    region = column // TOKENS
    entry = tl.load(packed + row * packed_stride + region, region < REGIONS, other=-1)
    count = tl.load(counts + row)
    valid = (row < tl.load(active)) & (region < count) & (entry >= 0)
    start = (entry >> 32) * TOKENS
    record = tl.where(valid, start + column % TOKENS, -TOKENS + column % TOKENS)
    tl.store(
        output + row * (REGIONS * TOKENS) + column,
        record.to(tl.int32),
        column < REGIONS * TOKENS,
    )


@triton.jit
def remap_regions_kernel(
    packed,
    records,
    output,
    packed_stride: tl.constexpr,
    TOKENS: tl.constexpr,
    REGIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    entry = tl.load(packed + row * packed_stride + column, column < REGIONS, other=-1)
    first = tl.load(
        records + row * (REGIONS * TOKENS) + column * TOKENS, column < REGIONS, other=-1
    ).to(tl.int64)
    remapped = ((first // TOKENS) << 32) | (entry & 0xFFFFFFFF)
    tl.store(
        output + row * REGIONS + column,
        tl.where((entry >= 0) & (first >= 0), remapped, -1),
        column < REGIONS,
    )
