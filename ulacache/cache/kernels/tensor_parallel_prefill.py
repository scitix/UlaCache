"""Owner-only GPU/mapped-CPU reads; no migration, LRU or remote CPU pointers."""

import triton
import triton.language as tl


@triton.jit
def _page_offset(
    row, byte, Page: tl.constexpr, PageBytes: tl.constexpr, Segments: tl.constexpr
):
    offset = row // Page * PageBytes
    row_prefix = 0
    page_prefix = 0
    within = tl.full((row.shape[0], byte.shape[1]), 0, tl.int64)
    for segment in tl.static_range(len(Segments)):
        width = Segments[segment]
        here = (byte >= row_prefix) & (byte < row_prefix + width)
        within = tl.where(
            here, page_prefix + row % Page * width + byte - row_prefix, within
        )
        row_prefix += width
        page_prefix += Page * width
    return offset + within


@triton.jit
def _owner_history(
    Rows,
    Gids,
    Fresh,
    Directory,
    GPU,
    HostPtrs,
    Out,
    Count,
    Rank: tl.constexpr,
    Logical: tl.constexpr,
    GPUCap: tl.constexpr,
    HostCap: tl.constexpr,
    Layer: tl.constexpr,
    Bytes: tl.constexpr,
    Page: tl.constexpr,
    PageBytes: tl.constexpr,
    Segments: tl.constexpr,
    BlockRows: tl.constexpr,
    Words: tl.constexpr,
):
    word = tl.arange(0, Words)
    first = tl.program_id(0) * BlockRows
    host = tl.load(HostPtrs).to(tl.uint64)
    while first < Count:
        index = first + tl.arange(0, BlockRows)
        native = tl.load(Rows + index, index < Count, other=0).to(tl.int64)
        gid = tl.load(Gids + index, index < Count, other=-1).to(tl.int64)
        fresh = tl.load(Fresh + index, index < Count, other=-1)
        owner, local = gid // Logical, gid % Logical
        live = (index < Count) & (native > 0) & (owner == Rank) & (fresh < 0)
        # local0 on a nonzero owner is valid. Only the native zero sentinel is invalid.
        loc = tl.load(Directory + local, live, other=-1)
        cpu_slot = -loc - 2
        valid = ((loc >= 0) & (loc < GPUCap)) | ((loc <= -2) & (cpu_slot < HostCap))
        tl.device_assert(
            (~live) | valid, "Unallocated or out-of-bounds TP prefix identity"
        )
        gpu = GPU.to(tl.uint64) + tl.maximum(loc, 0) * Bytes
        cpu = host + (Layer * HostCap + tl.maximum(cpu_slot, 0)) * Bytes
        source = tl.where(loc >= 0, gpu, cpu)
        value = tl.load(
            (source[:, None] + word[None, :] * 8).to(tl.pointer_type(tl.uint64)),
            live[:, None] & valid[:, None] & (word[None, :] < Bytes // 8),
            other=0,
        )
        target = Out.to(tl.uint64) + _page_offset(
            index[:, None], word[None, :] * 8, Page, PageBytes, Segments
        )
        tl.store(
            target.to(tl.pointer_type(tl.uint64)),
            value,
            live[:, None] & (word[None, :] < Bytes // 8),
        )
        first += tl.num_programs(0) * BlockRows


@triton.jit(do_not_specialize=["Count", "CurrentCount"])
def _staged_current(
    Rows,
    Fresh,
    CurrentRows,
    Staging,
    Out,
    Count,
    CurrentCount,
    Bytes: tl.constexpr,
    Page: tl.constexpr,
    PageBytes: tl.constexpr,
    Segments: tl.constexpr,
    BlockRows: tl.constexpr,
    Words: tl.constexpr,
):
    word = tl.arange(0, Words)
    first = tl.program_id(0) * BlockRows
    while first < Count:
        index = first + tl.arange(0, BlockRows)
        native = tl.load(Rows + index, index < Count, other=0)
        fresh = tl.load(Fresh + index, index < Count, other=-1).to(tl.int64)
        live = (index < Count) & (native > 0) & (fresh >= 0)
        tl.device_assert(
            (~live) | (fresh < CurrentCount), "Current TP row outside native writer"
        )
        writer = tl.load(CurrentRows + fresh, live & (fresh < CurrentCount), other=0)
        tl.device_assert(
            (~live) | (writer == native), "Current TP native identity differs"
        )
        source = Staging.to(tl.uint64) + _page_offset(
            fresh[:, None], word[None, :] * 8, Page, PageBytes, Segments
        )
        value = tl.load(
            source.to(tl.pointer_type(tl.uint64)),
            live[:, None] & (word[None, :] < Bytes // 8),
            other=0,
        )
        target = Out.to(tl.uint64) + _page_offset(
            index[:, None], word[None, :] * 8, Page, PageBytes, Segments
        )
        tl.store(
            target.to(tl.pointer_type(tl.uint64)),
            value,
            live[:, None] & (word[None, :] < Bytes // 8),
        )
        first += tl.num_programs(0) * BlockRows


def _inputs(rows, fresh, window, native_page):
    import torch

    if (
        not rows.is_cuda
        or rows.dtype != torch.int32
        or rows.ndim != 1
        or not rows.is_contiguous()
        or fresh.dtype != torch.int32
        or fresh.shape != rows.shape
        or fresh.device != rows.device
        or not fresh.is_contiguous()
        or window.dtype != torch.uint8
        or window.device != rows.device
        or not window.is_contiguous()
        or native_page.row_bytes != 656
        or native_page.page_size != 64
    ):
        raise ValueError(
            "Use native page64 full656 CUDA rows and a separate encoded window"
        )
    needed = (
        triton.cdiv(rows.numel(), native_page.page_size) * native_page.bytes_per_page
    )
    if window.nbytes < needed:
        raise ValueError("TP prefill window cannot hold its complete native pages")
    return needed


def gather_owner_history(
    rows,
    gids,
    fresh,
    directory,
    gpu_layer,
    host_ptrs,
    window,
    *,
    rank,
    logical_slots,
    host_slots,
    layer,
    native_page,
):
    import torch

    _inputs(rows, fresh, window, native_page)
    if (
        gids.shape != rows.shape
        or gids.dtype != torch.int32
        or gids.device != rows.device
        or not gids.is_contiguous()
        or directory.dtype != torch.int64
        or directory.shape != (logical_slots,)
        or directory.device != rows.device
        or gpu_layer.dtype != torch.uint8
        or not gpu_layer.is_contiguous()
        or gpu_layer.shape[1:] != (656,)
        or gpu_layer.device != rows.device
        or host_ptrs.dtype not in (torch.int64, torch.uint64)
        or host_ptrs.device != rows.device
        or not 0 <= rank < 8
        or host_slots < 1
        or layer < 0
    ):
        raise ValueError("Canonical owner storage/identity geometry differs")
    if rows.numel():
        _owner_history[(min(64, triton.cdiv(rows.numel(), 8)),)](
            rows,
            gids,
            fresh,
            directory,
            gpu_layer,
            host_ptrs,
            window,
            rows.numel(),
            rank,
            logical_slots,
            gpu_layer.shape[0],
            host_slots,
            layer,
            656,
            native_page.page_size,
            native_page.bytes_per_page,
            native_page.segment_bytes,
            8,
            128,
            num_warps=8,
        )


def install_staged_current(rows, fresh, current_rows, staging, window, *, native_page):
    import torch

    _inputs(rows, fresh, window, native_page)
    if (
        current_rows.ndim != 1
        or not current_rows.is_contiguous()
        or current_rows.dtype not in (torch.int32, torch.int64)
        or current_rows.device != rows.device
        or staging.device != rows.device
        or staging.dtype != torch.uint8
        or not staging.is_contiguous()
        or staging.nbytes
        < triton.cdiv(current_rows.numel(), 64) * native_page.bytes_per_page
    ):
        raise ValueError("TP native staged writer and row identities differ")
    if rows.numel():
        _staged_current[(min(64, triton.cdiv(rows.numel(), 8)),)](
            rows,
            fresh,
            current_rows,
            staging,
            window,
            rows.numel(),
            current_rows.numel(),
            656,
            native_page.page_size,
            native_page.bytes_per_page,
            native_page.segment_bytes,
            8,
            128,
            num_warps=8,
        )
