"""Native-ID translation and complete-record writes for independent DPA ranks."""

from __future__ import annotations

import triton
import triton.language as tl
from ulacache.compiler import cache_once, load_jit


@cache_once
def _allocation_module():
    return load_jit(
        "unified_local_allocation",
        cuda_files=["csrc/cache/local_cache.cuh"],
        cuda_wrappers=[
            ("register", "sglang::unified_local_register_unbound"),
            ("expire", "sglang::unified_local_expire_groups"),
            ("register_receive", "sglang::unified_local_register_receive"),
            ("receive_locations", "sglang::unified_local_receive_locations"),
            ("rebind_cpu_receive", "sglang::unified_local_rebind_cpu_receive"),
        ],
    )


def register_unbound(rows, frames, *, prefer_gpu=True):
    """Rows must be unique within the native allocation result."""
    _allocation_module().register(rows, frames, prefer_gpu)


def _receive_devices(rows, frames):
    if (
        not rows.is_cuda
        or rows.device != frames.device
        or rows.ndim != 1
        or not rows.is_contiguous()
        or not frames.is_contiguous()
    ):
        raise ValueError(
            "Receive rows and residency frames must be contiguous on one CUDA device"
        )


def register_receive_rows(rows, frames, *, prefer_gpu=False):
    """Allocate ordered tiles in canonical storage for coalesced direct RDMA."""
    _receive_devices(rows, frames)
    _allocation_module().register_receive(rows, frames, prefer_gpu)


@triton.jit
def _order_free_cpu_receive_tail(
    Frames, FrameWords: tl.constexpr, Window: tl.constexpr
):
    frame = Frames + tl.program_id(0) * FrameWords
    free = tl.load(frame + 7).to(tl.pointer_type(tl.int32))
    counts = tl.load(frame + 8).to(tl.pointer_type(tl.int32))
    count = tl.load(counts + 1)
    start = tl.maximum(count - Window, 0)
    offset = start + tl.arange(0, Window)
    values = tl.load(free + offset, offset < count, other=-2147483648)
    ordered = tl.sort(values, descending=True)
    tl.store(free + offset, ordered, offset < count)


def order_free_cpu_receive_tail(frames, records):
    """Prefer dense unused slots below recently released fragments.

    The caller drains residency work and serializes this with allocation on
    the owning stream. Only a bounded tail of the CPU free stack is permuted;
    its slot set/count and every live directory, tag and payload stay intact.
    Allocation still accepts scattered free slots when no extent is available.
    """
    if records <= 0 or records > 16384:
        return False
    if (
        not frames.is_cuda
        or not frames.is_contiguous()
        or frames.ndim != 2
        or frames.shape[1] != 31
    ):
        raise ValueError(
            "Receive free-stack ordering requires native contiguous frames"
        )
    window = 16384 if records <= 4096 else 32768
    _order_free_cpu_receive_tail[(frames.shape[0],)](
        frames, frames.shape[1], window, num_warps=8
    )
    return True


def acquire_receive_locations(rows, frames):
    """Snapshot physical slots and retain GPU-slot leases across remote writes.

    Rows must be newly allocated, unpublished and held by native logical-page
    ownership. The caller first drains earlier residency moves. CPU rows cannot
    enter attention before publication and remain owned until remote completion.
    """
    import torch

    _receive_devices(rows, frames)
    locations = torch.empty(
        (frames.shape[0], rows.numel()), dtype=torch.int64, device=frames.device
    )
    _allocation_module().receive_locations(rows, frames, locations, True)
    return locations


def release_receive_locations(rows, frames, locations, *, transport_quiesced):
    """Retire advertised destinations only after completion or drained abort."""
    import torch

    if transport_quiesced is not True:
        raise ValueError("Outstanding remote writes must retire before their slots")
    _receive_devices(rows, frames)
    if (
        locations.device != frames.device
        or locations.dtype != torch.int64
        or locations.shape != (frames.shape[0], rows.numel())
        or not locations.is_contiguous()
    ):
        raise ValueError("Receive location snapshot differs from its owning allocation")
    _allocation_module().receive_locations(rows, frames, locations, False)


def rebind_unpublished_cpu_receive(rows, frames, locations):
    """Apply a permutation of this new receive's CPU slots before publication.

    GPU destinations/leases and the allocated CPU slot set must be unchanged.
    No KV bytes, free-stack entries, epochs or existing-record bindings move.
    """
    import torch

    _receive_devices(rows, frames)
    if (
        locations.device != frames.device
        or locations.dtype != torch.int64
        or locations.shape != (frames.shape[0], rows.numel())
        or not locations.is_contiguous()
    ):
        raise ValueError("Receive binding metadata differs from its owning allocation")
    _allocation_module().rebind_cpu_receive(rows, frames, locations)


def expire_recent_groups(frames):
    """Retire independent bucket headers without advancing canonical epochs."""
    _allocation_module().expire(frames)


@triton.jit
def _remap(
    S,
    Active,
    Directory,
    Out,
    Stats,
    N: tl.constexpr,
    Width: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    real = tl.load(Active)
    row = tl.load(S + i, i < N, other=-1).to(tl.int32)
    valid = (i < N) & (i // Width < real) & (row > 0)
    location = tl.load(Directory + row, valid, other=-1)
    if COUNT:
        tl.atomic_add(Stats, tl.sum(valid.to(tl.int64)))
        tl.atomic_add(Stats + 1, tl.sum((valid & (location >= 0)).to(tl.int64)))
        tl.atomic_add(Stats + 2, tl.sum((valid & (location <= -2)).to(tl.int64)))
    else:
        tl.device_assert((~valid) | (location >= 0), "Selected KV was not promoted")
        tl.store(Out + i, tl.where(valid, location, -1).to(tl.int32), i < N)


def count_residency(selected, active, directory, stats):
    if selected.numel():
        _remap[(triton.cdiv(selected.numel(), 256),)](
            selected,
            active,
            directory,
            selected,
            stats,
            selected.numel(),
            selected.shape[1],
            True,
            256,
        )


def remap_resident_rows(selected, active, directory, output):
    if selected.numel():
        _remap[(triton.cdiv(selected.numel(), 256),)](
            selected,
            active,
            directory,
            output,
            active,
            selected.numel(),
            selected.shape[1],
            False,
            256,
        )


@triton.jit
def _remap_prefill(Selected, Directory, Output, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(Selected + i, i < N, other=-1).to(tl.int32)
    valid = (i < N) & (row > 0)
    location = tl.load(Directory + row, valid, other=-1)
    tl.device_assert((~valid) | (location >= 0), "Selected KV was not promoted")
    tl.store(Output + i, tl.where(valid, location, -1).to(tl.int32), i < N)


def remap_prefill_rows(selected, directory, output):
    """Every eager prefill query is real; avoid allocating/copying a scalar."""
    if selected.numel():
        _remap_prefill[(triton.cdiv(selected.numel(), 256),)](
            selected, directory, output, selected.numel(), 256
        )


@triton.jit
def _write(
    Rows,
    Nope,
    Rope,
    Directory,
    GPU,
    HostPtrs,
    NS: tl.constexpr,
    RS: tl.constexpr,
    NW: tl.constexpr,
    RW: tl.constexpr,
    Layer: tl.constexpr,
    HostSlots: tl.constexpr,
    RowBytes: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0)
    row = tl.load(Rows + q).to(tl.int32)
    if row > 0:
        location = tl.load(Directory + row)
        tl.device_assert(location != -1, "Writing an unallocated KV identity")
        cols = tl.arange(0, BLOCK)
        nope = tl.load(Nope + q * NS + cols, cols < NW, other=0)
        rope = tl.load(
            Rope + q * RS + cols - NW, (cols >= NW) & (cols < RowBytes), other=0
        )
        data = tl.where(cols < NW, nope, rope)
        if location >= 0:
            target = GPU + location * RowBytes
        else:
            host = tl.load(HostPtrs).to(tl.pointer_type(tl.uint8))
            target = host + (Layer * HostSlots - location - 2) * RowBytes
        tl.store(target + cols, data, cols < RowBytes)


def write_native_records(
    rows, nope, rope, directory, gpu, host_ptrs, layer, host_slots, row_bytes
):
    import torch

    if row_bytes == 576:
        allowed = (torch.float8_e4m3fn, torch.uint8)
    elif row_bytes == 656:
        allowed = (torch.uint8,)
    else:
        raise ValueError("Direct native writes support raw576 or scaled656 records")
    if nope.dtype not in allowed or rope.dtype != nope.dtype:
        raise ValueError("Direct writes require native quantized byte segments")
    n, r = nope.reshape(-1, nope.shape[-1]), rope.reshape(-1, rope.shape[-1])
    if (
        n.shape[0] != rows.numel()
        or r.shape[0] != rows.numel()
        or n.shape[1] + r.shape[1] != row_bytes
    ):
        raise ValueError("Native split records and physical write rows disagree")
    if n.stride(1) != 1 or r.stride(1) != 1:
        raise ValueError(
            "Native FP8 record segments must be contiguous along their width"
        )
    if (n.shape[1], r.shape[1]) != ((512, 64) if row_bytes == 576 else (528, 128)):
        raise ValueError("Native record segments do not match the declared FP8 layout")
    if rows.numel():
        _write[(rows.numel(),)](
            rows,
            n.view(torch.uint8),
            r.view(torch.uint8),
            directory,
            gpu,
            host_ptrs,
            n.stride(0),
            r.stride(0),
            n.shape[1],
            r.shape[1],
            layer,
            host_slots,
            row_bytes,
            triton.next_power_of_2(row_bytes),
        )


@cache_once
def _prefill_offload_module():
    return load_jit(
        "unified_local_prefill_offload",
        cuda_files=["csrc/cache/prefix.cuh"],
        cuda_wrappers=[("offload", "sglang::unified_local_offload_prefill")],
    )


def offload_prefill_rows(rows, frames, gpu, host_ptrs, first_layer, layers):
    """Return new prefill identities to CPU after the group reader retires."""
    _prefill_offload_module().offload(rows, frames, gpu, host_ptrs, first_layer, layers)
