# SPDX-License-Identifier: Apache-2.0
"""Gather complete compressed prefix rows without copying request IDs to CPU."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_prefix(
    TABLE,
    REQS,
    LENGTHS,
    OFFSETS,
    OUT,
    STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    tile = tl.program_id(1)
    pos = tile * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(LENGTHS + req) // RATIO
    index = tl.load(REQS + req)
    begin = tl.load(OFFSETS + req)
    row = tl.load(
        TABLE + index * STRIDE + (pos + 1) * RATIO - 1, mask=pos < count, other=0
    )
    tl.store(OUT + begin + pos, row // RATIO, mask=pos < count)


def gather_prefix_cache_indices(
    table, reqs, lengths, lengths_cpu, *, compression_ratio
):
    counts = [int(n) // compression_ratio for n in lengths_cpu]
    if len(counts) != reqs.numel():
        raise ValueError("Prefix lengths must match the request table indices")
    out = torch.empty(sum(counts), dtype=torch.int32, device=table.device)
    if out.numel():
        offsets = [0]
        for n in counts[:-1]:
            offsets.append(offsets[-1] + n)
        starts = torch.tensor(offsets, dtype=torch.int32, device=table.device)
        _gather_prefix[(len(counts), triton.cdiv(max(counts), 256))](
            table, reqs, lengths, starts, out, table.stride(0), compression_ratio, 256
        )
    return out
