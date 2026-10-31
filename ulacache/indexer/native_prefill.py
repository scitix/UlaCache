# SPDX-License-Identifier: Apache-2.0
"""Read owner-sharded native Index pages into the existing ragged prefill ABI."""

import torch
import triton
import triton.language as tl


# Layer offsets and page-table geometry vary across prefill chunks.
@triton.jit(do_not_specialize=["ANCHOR_BYTES", "PAGE_STRIDE", "PAGE_WIDTH"])
def _gather_owner_index(
    PEERS,
    PAGES,
    LENGTHS,
    K,
    SCALE,
    DEGREE: tl.constexpr,
    ANCHOR_BYTES,
    NATIVE_ROWS: tl.constexpr,
    PAGE_STRIDE,
    PAGE_WIDTH,
    LENGTH_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGE_MAP=None,
    REQUEST_MAP: tl.constexpr = False,
    GPU_PAGES: tl.constexpr = 0,
):
    query = tl.program_id(0)
    first = tl.program_id(1) * BLOCK
    length = tl.load(LENGTHS + query)
    if first >= length:
        return
    tokens = first + tl.arange(0, BLOCK)
    active = tokens < length
    previous = tl.arange(0, LENGTH_POW2)
    output_start = tl.sum(tl.load(LENGTHS + previous, previous < query, other=0))
    page = tl.load(
        PAGES + query * PAGE_STRIDE + tokens // 64,
        active & (tokens // 64 < PAGE_WIDTH),
        other=0,
    ).to(tl.int64)
    native = page * 64 + tokens % 64
    valid = active & (native > 0) & (native < NATIVE_ROWS)
    owner = native % DEGREE
    local = native // DEGREE
    if REQUEST_MAP:
        physical = tl.load(PAGE_MAP + local // 64, valid, other=0)
        tl.device_assert(
            (~valid) | ((physical >= 0) & (physical < GPU_PAGES)),
            "request prefill Index page is not GPU resident",
        )
        local = physical.to(tl.int64) * 64 + local % 64
    base = tl.load(PEERS + owner, valid, other=0)
    base += ANCHOR_BYTES.to(tl.int64) + (local // 64) * 8448
    key_offsets = (local % 64)[:, None] * 128 + tl.arange(0, 128)[None, :]
    key_ptr = (base[:, None] + key_offsets).to(tl.pointer_type(tl.uint8))
    keys = tl.load(key_ptr, valid[:, None], other=0, cache_modifier=".cg")
    output = output_start + tokens
    tl.store(
        K + output[:, None] * 128 + tl.arange(0, 128)[None, :], keys, active[:, None]
    )
    scale_offsets = 8192 + (local % 64)[:, None] * 4 + tl.arange(0, 4)[None, :]
    scale_ptr = (base[:, None] + scale_offsets).to(tl.pointer_type(tl.uint8))
    scales = tl.load(scale_ptr, valid[:, None], other=0, cache_modifier=".cg")
    tl.store(
        SCALE + output[:, None] * 4 + tl.arange(0, 4)[None, :], scales, active[:, None]
    )


def gather_owner_index(
    peers: torch.Tensor,
    page_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    degree: int,
    anchor: int,
    owner_pages: int,
    native_rows: int,
    seq_len_sum: int,
    max_seq_len: int,
    page_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call after all owners' prefill stores have completed their NCCL ready task.

    Page indices retain the native 64-token physical page unit. No Index keys
    are persistently replicated; only these original ragged outputs are dense.
    """
    if (
        peers.shape != (degree,)
        or peers.dtype != torch.uint64
        or peers.device.type != "cuda"
        or page_indices.ndim != 2
        or seq_lens.ndim != 1
        or page_indices.shape[0] != seq_lens.numel()
        or page_indices.dtype not in (torch.int32, torch.int64)
        or seq_lens.dtype not in (torch.int32, torch.int64)
        or page_indices.stride(1) != 1
        or not seq_lens.is_contiguous()
        or page_indices.device != peers.device
        or seq_lens.device != peers.device
        or degree < 2
        or degree & (degree - 1)
        or owner_pages <= 0
        or native_rows <= 0
        or anchor < 0
        or seq_len_sum < 0
        or max_seq_len < 0
        or page_indices.shape[1] * 64 < max_seq_len
    ):
        raise ValueError("Unsupported owner Index ragged prefill metadata")
    if page_map is not None and (
        page_map.dtype != torch.int32
        or page_map.device != peers.device
        or page_map.ndim != 1
        or not page_map.is_contiguous()
        or page_map.numel() * 64 * degree < native_rows
    ):
        raise ValueError("Request prefill requires the mirrored logical owner-page map")
    keys = torch.empty((seq_len_sum, 128), dtype=torch.uint8, device=peers.device)
    scales = torch.empty((seq_len_sum, 4), dtype=torch.uint8, device=peers.device)
    if max_seq_len and seq_lens.numel():
        _gather_owner_index[(seq_lens.numel(), triton.cdiv(max_seq_len, 32))](
            peers,
            page_indices,
            seq_lens,
            keys,
            scales,
            degree,
            anchor * owner_pages * 8448,
            native_rows,
            page_indices.stride(0),
            page_indices.shape[1],
            triton.next_power_of_2(seq_lens.numel()),
            32,
            page_map,
            page_map is not None,
            owner_pages,
        )
    return keys, scales
