"""Read-only canonical GPU/CPU records into the shared prefill workspaces.

The directory and both authoritative tiers are inputs. These kernels never
promote records, select victims, update LRU state, or publish new ownership.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _history(
    Rows,
    Fresh,
    Directory,
    GPU,
    HostPtrs,
    Out,
    Count,
    HostSlots: tl.constexpr,
    Layer: tl.constexpr,
    RowBytes: tl.constexpr,
    BlockRows: tl.constexpr,
    Words: tl.constexpr,
    OwnerCount: tl.constexpr = 1,
    OwnerRank: tl.constexpr = 0,
):
    first = tl.program_id(0) * BlockRows
    word = tl.arange(0, Words)
    host = tl.load(HostPtrs).to(tl.uint64)
    while first < Count:
        index = first + tl.arange(0, BlockRows)
        row = tl.load(Rows + index, index < Count, other=0).to(tl.int64)
        fresh = tl.load(Fresh + index, index < Count, other=-1)
        local = row // OwnerCount
        live = (
            (index < Count)
            & (row > 0)
            & (local > 0)
            & (fresh < 0)
            & (row % OwnerCount == OwnerRank)
        )
        location = tl.load(Directory + local, live, other=-1)
        tl.device_assert(
            (~live) | (location != -1), "Unallocated prefill history identity"
        )
        gpu_address = GPU.to(tl.uint64) + tl.maximum(location, 0) * RowBytes
        host_address = (
            host + (Layer * HostSlots + tl.maximum(-location - 2, 0)) * RowBytes
        )
        source = tl.where(location >= 0, gpu_address, host_address)
        address = source[:, None] + word[None, :] * 8
        value = tl.load(
            address.to(tl.pointer_type(tl.uint64)),
            live[:, None] & (word[None, :] < RowBytes // 8),
            other=0,
        )
        target = Out.to(tl.uint64) + index[:, None] * RowBytes + word[None, :] * 8
        write = (index < Count) if OwnerCount > 1 else live
        tl.store(
            target.to(tl.pointer_type(tl.uint64)),
            value,
            write[:, None] & (word[None, :] < RowBytes // 8),
        )
        first += tl.num_programs(0) * BlockRows


@triton.jit(do_not_specialize=["Count", "CurrentCount"])
def _current(
    Rows,
    Fresh,
    CurrentRows,
    Nope,
    Rope,
    Out,
    Count,
    CurrentCount,
    NopeStride: tl.constexpr,
    RopeStride: tl.constexpr,
    NopeWords: tl.constexpr,
    RowBytes: tl.constexpr,
    BlockRows: tl.constexpr,
    Words: tl.constexpr,
):
    first = tl.program_id(0) * BlockRows
    word = tl.arange(0, Words)
    while first < Count:
        index = first + tl.arange(0, BlockRows)
        row = tl.load(Rows + index, index < Count, other=0)
        fresh = tl.load(Fresh + index, index < Count, other=-1).to(tl.int64)
        live = (index < Count) & (row > 0) & (fresh >= 0)
        tl.device_assert(
            (~live) | (fresh < CurrentCount), "Prefill current position outside writer"
        )
        writer_row = tl.load(
            CurrentRows + fresh, live & (fresh < CurrentCount), other=0
        )
        tl.device_assert(
            (~live) | (writer_row == row), "Prefill writer identity differs"
        )
        naddr = Nope.to(tl.uint64) + fresh[:, None] * NopeStride + word[None, :] * 8
        raddr = (
            Rope.to(tl.uint64)
            + fresh[:, None] * RopeStride
            + (word[None, :] - NopeWords) * 8
        )
        source = tl.where(word[None, :] < NopeWords, naddr, raddr)
        value = tl.load(
            source.to(tl.pointer_type(tl.uint64)),
            live[:, None] & (word[None, :] < RowBytes // 8),
            other=0,
        )
        target = Out.to(tl.uint64) + index[:, None] * RowBytes + word[None, :] * 8
        tl.store(
            target.to(tl.pointer_type(tl.uint64)),
            value,
            live[:, None] & (word[None, :] < RowBytes // 8),
        )
        first += tl.num_programs(0) * BlockRows


def _inputs(rows, fresh, window, row_bytes):
    import torch

    if row_bytes not in (576, 656):
        raise ValueError("Bind native raw576 or scaled656 complete records")
    if (
        not rows.is_cuda
        or rows.dtype != torch.int32
        or rows.ndim != 1
        or not rows.is_contiguous()
        or fresh.dtype != torch.int32
        or fresh.shape != rows.shape
        or not fresh.is_contiguous()
        or window.dtype != torch.uint8
        or not window.is_contiguous()
        or window.numel() % row_bytes
        or window.numel() < rows.numel() * row_bytes
        or fresh.device != rows.device
        or window.device != rows.device
    ):
        raise ValueError(
            "CUDA compact rows/fresh map and a complete raw window are required"
        )


def gather_history(
    rows,
    fresh,
    directory,
    gpu_layer,
    host_ptrs,
    window,
    *,
    layer,
    host_slots,
    row_bytes,
    launch=(64, 256),
    owner_count=1,
    owner_rank=0,
):
    """Copy only existing history, leaving fresh positions for the writer."""
    import torch

    _inputs(rows, fresh, window, row_bytes)
    if (
        type(owner_count) is not int
        or owner_count < 1
        or owner_count & (owner_count - 1)
        or type(owner_rank) is not int
        or not 0 <= owner_rank < owner_count
    ):
        raise ValueError("Use a power-of-two position owner count and its local rank")
    if (
        directory.dtype != torch.int64
        or directory.ndim != 1
        or directory.device != rows.device
        or gpu_layer.device != rows.device
        or gpu_layer.dtype != torch.uint8
        or not gpu_layer.is_contiguous()
        or gpu_layer.ndim != 2
        or gpu_layer.shape[1] != row_bytes
        or host_ptrs.device != rows.device
        or host_ptrs.dtype not in (torch.int64, torch.uint64)
        or layer < 0
        or host_slots <= 0
    ):
        raise ValueError("Canonical layer storage and directory differ")
    blocks, threads = launch
    if blocks <= 0 or threads not in (128, 256, 512, 1024):
        raise ValueError("Use ordinary positive blocks and supported CUDA block size")
    if rows.numel():
        _history[(min(blocks, triton.cdiv(rows.numel(), threads // 32)),)](
            rows,
            fresh,
            directory,
            gpu_layer,
            host_ptrs,
            window,
            rows.numel(),
            host_slots,
            layer,
            row_bytes,
            threads // 32,
            triton.next_power_of_2(row_bytes // 8),
            owner_count,
            owner_rank,
            num_warps=threads // 32,
        )


def install_current(
    rows, fresh, current_rows, nope, rope, window, *, row_bytes, launch=(64, 256)
):
    """Install the native writer's encoded segments; never read unwritten KV."""
    import torch

    _inputs(rows, fresh, window, row_bytes)
    allowed = (torch.uint8, torch.float8_e4m3fn) if row_bytes == 576 else (torch.uint8,)
    n, r = nope.reshape(-1, nope.shape[-1]), rope.reshape(-1, rope.shape[-1])
    widths = (512, 64) if row_bytes == 576 else (528, 128)
    if (
        nope.dtype not in allowed
        or rope.dtype != nope.dtype
        or (n.shape[1], r.shape[1]) != widths
        or n.shape[0] != current_rows.numel()
        or r.shape[0] != current_rows.numel()
        or n.stride(1) != 1
        or r.stride(1) != 1
        or current_rows.ndim != 1
        or not current_rows.is_contiguous()
        or current_rows.dtype not in (torch.int32, torch.int64)
        or any(t.device != rows.device for t in (current_rows, nope, rope))
    ):
        raise ValueError(
            "Current positions and complete native quantized writer segments differ"
        )
    blocks, threads = launch
    if blocks <= 0 or threads not in (128, 256, 512, 1024):
        raise ValueError("Use ordinary positive blocks and supported CUDA block size")
    if rows.numel():
        _current[(min(blocks, triton.cdiv(rows.numel(), threads // 32)),)](
            rows,
            fresh,
            current_rows,
            n.view(torch.uint8),
            r.view(torch.uint8),
            window,
            rows.numel(),
            current_rows.numel(),
            n.stride(0),
            r.stride(0),
            widths[0] // 8,
            row_bytes,
            threads // 32,
            triton.next_power_of_2(row_bytes // 8),
            num_warps=threads // 32,
        )
