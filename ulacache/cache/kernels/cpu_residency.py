"""Owner-local, exclusive GPU/CPU residency for complete FlowKV records.

Logical rows and their GPU Index rows never move.  A directory entry is a GPU
physical row (>= 0), a CPU row (-row-2), or unallocated (-1).  Inputs use the
existing rank-major global logical IDs, not physical pool addresses.

One state/workspace belongs to one shared-Index layer group.  Callers serialize
its prepare/exchange/commit sequence, publish completion before remote VA reads,
and release leases after those reads complete.  Nothing here synchronizes with
the CPU or allocates during CUDA graph replay.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from ulacache.compiler import cache_once, load_jit
from ulacache.cache.directory import (
    HostTierState,
    create_host_tier_state as create_host_tier_state,
    clear_host_tier_state as clear_host_tier_state,
)


class HostTierWorkspace(NamedTuple):
    new_rows: torch.Tensor
    promote_rows: torch.Tensor
    plan: torch.Tensor
    counters: torch.Tensor
    histogram: torch.Tensor
    selection_prefix: torch.Tensor
    candidate_slots: torch.Tensor
    leased_slots: torch.Tensor


@cache_once
def _host_module():
    return load_jit(
        "flowkv_host_tier",
        cuda_files=["csrc/cache/residency_fused.cuh"],
        cuda_wrappers=[
            ("register", "sglang::flowkv_host_register"),
            ("register_groups", "sglang::flowkv_host_register_groups"),
            ("register_native_groups", "sglang::flowkv_host_register_native_groups"),
            ("prepare", "sglang::flowkv_host_prepare"),
            ("exchange", "sglang::flowkv_host_exchange"),
            ("commit", "sglang::flowkv_host_commit"),
            ("release_leases", "sglang::flowkv_host_release_leases"),
            ("release_rows", "sglang::flowkv_host_release_rows"),
            ("release_rows_retired", "sglang::flowkv_host_release_rows_retired"),
            ("drain_readers", "sglang::flowkv_host_drain_readers"),
            ("clear_ring", "sglang::flowkv_host_clear_ring"),
        ],
    )


def create_host_tier_workspace(
    capacity: int, device: torch.device | str
) -> HostTierWorkspace:
    """Capacity is maximum selected misses plus current tokens for this graph."""
    if capacity < 1:
        raise ValueError("Host tier workspace capacity must be positive")
    return HostTierWorkspace(
        torch.empty(capacity, dtype=torch.int32, device=device),
        torch.empty(capacity, dtype=torch.int32, device=device),
        torch.empty((5, capacity), dtype=torch.int32, device=device),
        torch.zeros(8, dtype=torch.int32, device=device),
        # Bound to a retained PresenceRing.view before any operation/capture.
        torch.empty(0, dtype=torch.int64, device=device),
        torch.zeros(3, dtype=torch.int64, device=device),
        torch.empty(capacity, dtype=torch.int32, device=device),
        torch.empty(capacity, dtype=torch.int32, device=device),
    )


def _rows(rows: torch.Tensor) -> None:
    if rows.dtype not in (torch.int32, torch.int64) or not rows.is_contiguous():
        raise ValueError("Global logical rows must be contiguous int32 or int64")


def register_host_tier_rows(
    rows: torch.Tensor,
    owner: int,
    state: HostTierState,
    *,
    ring,
    prefer_gpu: bool = True,
) -> None:
    """Bind new prefill rows to free slots, without copying or evicting records.

    The ordinary record writer subsequently writes directly to their GPU/CPU
    location.  Register once per group before any layer writes its records.
    """
    _rows(rows)
    if rows.ndim != 1:
        raise ValueError("Registered logical rows must be one-dimensional")
    _host_module().register(rows, owner, prefer_gpu, ring.view, *state)


def register_host_tier_groups(
    rows: torch.Tensor,
    frames: torch.Tensor,
    *,
    prefer_gpu: bool = True,
) -> None:
    """Register the same logical rows in all independent shared-Index groups."""
    _rows(rows)
    if rows.ndim != 1:
        raise ValueError("Registered logical rows must be one-dimensional")
    if frames.dtype != torch.int64 or frames.ndim != 2 or not frames.is_contiguous():
        raise ValueError("Group frames must be contiguous int64[group, fields]")
    _host_module().register_groups(rows, frames, prefer_gpu)


def register_native_host_tier_groups(
    rows: torch.Tensor,
    frames: torch.Tensor,
    base: int,
    *,
    prefer_gpu: bool = True,
) -> None:
    """Translate native token slots and register every group in one launch."""
    _rows(rows)
    if rows.ndim != 1:
        raise ValueError("Registered native rows must be one-dimensional")
    if frames.dtype != torch.int64 or frames.ndim != 2 or not frames.is_contiguous():
        raise ValueError("Group frames must be contiguous int64[group, fields]")
    _host_module().register_native_groups(rows, frames, base, prefer_gpu)


def prepare_host_tier(
    selected: torch.Tensor,
    current: torch.Tensor,
    num_real_queries: torch.Tensor,
    owner: int,
    state: HostTierState,
    workspace: HostTierWorkspace,
    *,
    selected_counts: torch.Tensor | None = None,
    parallel: bool = False,
    parallel_victims: bool = False,
    current_already_prepared: bool = False,
) -> None:
    """Pin required rows and plan strictly oldest GPU victims entirely on GPU.

    ``selected`` is [query, selected_capacity].  To use the existing local-buffer
    miss plan, pass its per-query counts: already cached KV need not be promoted.
    Current rows may be unallocated; selected history must already be registered.
    Recent buckets retain epoch order; the unified cold bucket has arbitrary ties.
    CPU/GPU-full promotion
    swaps into the selected CPU slot, requiring no additional KV staging pool.
    """
    _rows(selected)
    _rows(current)
    from ulacache.compiler import recency_policy

    words = 12 if recency_policy() == "naive" else 11
    if workspace.histogram.dtype != torch.int64 or workspace.histogram.shape != (
        1,
        words,
    ):
        raise ValueError(
            "Bind the retained metadata view for the process recency policy"
        )
    if selected.ndim != 2 or current.ndim != 1:
        raise ValueError("selected must be [query, topk], current must be [query]")
    if current.numel() != selected.size(0):
        raise ValueError("Current and selected query counts must match")
    required = selected.numel() + current.numel()
    if workspace.new_rows.numel() < required:
        raise ValueError("Host tier workspace cannot hold the selected/current union")
    if num_real_queries.dtype != torch.int32 or num_real_queries.numel() != 1:
        raise ValueError("num_real_queries must be an int32 device scalar")
    if selected_counts is not None and (
        selected_counts.dtype != torch.int32
        or selected_counts.shape != (selected.size(0),)
        or not selected_counts.is_contiguous()
    ):
        raise ValueError("selected_counts must be contiguous int32[query]")
    if parallel_victims and not parallel:
        raise ValueError("Parallel victim selection requires parallel preparation")
    if current_already_prepared:
        if not parallel:
            raise ValueError("Batched current preparation requires parallel pin/finish")
        from ulacache.cache.kernels.batched_prepare import module

        prepare = module(parallel_victims).prepare
    elif parallel:
        from ulacache.cache.kernels.residency_parallel import prepare_function

        prepare = prepare_function(parallel_victims)
    else:
        prepare = _host_module().prepare
    prepare(
        selected,
        current,
        num_real_queries,
        selected_counts if selected_counts is not None else num_real_queries,
        selected_counts is not None,
        owner,
        *state,
        *workspace,
    )


def validate_exchange_launch(launch: tuple[int, int]) -> tuple[int, int]:
    """Validate a startup-only CUDA grid without allocating device metadata."""
    if not isinstance(launch, tuple) or len(launch) != 2:
        raise ValueError("Exchange launch requires a (blocks, threads) tuple")
    blocks, threads = launch
    if (
        type(blocks) is not int
        or type(threads) is not int
        or not 0 < blocks < 2**31
        or not 0 < threads <= 1024
        or threads % 32
    ):
        raise ValueError(
            "Exchange needs positive blocks and 32..1024 warp-aligned threads"
        )
    return blocks, threads


@cache_once
def _exchange_module(launch: tuple[int, int]):
    blocks, threads = validate_exchange_launch(launch)
    if launch == (4, 1024):
        return _host_module()
    return load_jit(
        "flowkv_host_exchange_launch_v1",
        str(blocks),
        str(threads),
        cuda_files=["csrc/cache/residency_fused.cuh"],
        cuda_wrappers=[("exchange", "sglang::flowkv_host_exchange")],
        extra_cuda_cflags=[
            f"-DUDCACHE_HOST_EXCHANGE_BLOCKS={blocks}",
            f"-DUDCACHE_HOST_EXCHANGE_THREADS={threads}",
        ],
    )


def exchange_host_tier(
    host_cache: torch.Tensor,
    gpu_cache: torch.Tensor,
    workspace: HostTierWorkspace,
    item_size_bytes: int,
    *,
    host_stride_bytes: int | None = None,
    gpu_stride_bytes: int | None = None,
    launch: tuple[int, int] = (4, 1024),
) -> None:
    """Apply one plan to a layer or a contiguous shared-Index layer group.

    Buffers are linear [slots, record] or [layers, slots, record].  One group IO
    launch covers every member before residency bindings are committed.
    """
    if not host_cache.is_pinned() or not gpu_cache.is_cuda:
        raise ValueError("Host tier exchange requires pinned CPU and CUDA record pools")
    if host_cache.ndim not in (2, 3) or gpu_cache.ndim != host_cache.ndim:
        raise ValueError("Host/GPU records must both be two- or three-dimensional")
    layers = host_cache.size(0) if host_cache.ndim == 3 else 1
    if gpu_cache.ndim == 3 and gpu_cache.size(0) != layers:
        raise ValueError("Host/GPU shared-Index groups must have the same layer count")
    host_stride_bytes = (
        host_cache.stride(-2) * host_cache.element_size()
        if host_stride_bytes is None
        else host_stride_bytes
    )
    gpu_stride_bytes = (
        gpu_cache.stride(-2) * gpu_cache.element_size()
        if gpu_stride_bytes is None
        else gpu_stride_bytes
    )
    if (
        item_size_bytes <= 0
        or item_size_bytes % 8
        or host_stride_bytes < item_size_bytes
        or gpu_stride_bytes < item_size_bytes
        or host_stride_bytes % 8
        or gpu_stride_bytes % 8
    ):
        raise ValueError("Complete linear records require aligned eight-byte geometry")
    _exchange_module(launch).exchange(
        host_cache,
        gpu_cache,
        workspace.plan,
        workspace.counters,
        item_size_bytes,
        host_stride_bytes,
        gpu_stride_bytes,
        layers,
        host_cache.stride(0) * host_cache.element_size() if host_cache.ndim == 3 else 0,
        gpu_cache.stride(0) * gpu_cache.element_size() if gpu_cache.ndim == 3 else 0,
    )


def commit_host_tier(state: HostTierState, workspace: HostTierWorkspace) -> None:
    """Commit bindings after every layer in this group finished its exchange."""
    _host_module().commit(*state, *workspace)


def release_host_tier_leases(
    state: HostTierState, workspace: HostTierWorkspace
) -> None:
    """Release VA-read protection after all local/peer pulls using this plan."""
    _host_module().release_leases(
        state.leases, workspace.leased_slots, workspace.counters
    )


def release_host_tier_rows(
    rows: torch.Tensor, owner: int, state: HostTierState, *, ring
) -> None:
    """Free dead logical rows selected by the existing prefix-tree leaf eviction."""
    _rows(rows)
    if rows.ndim != 1:
        raise ValueError("Released logical rows must be one-dimensional")
    _host_module().release_rows(rows, owner, ring.view, *state)


def release_retired_host_tier_rows(
    rows, owner, state, ready_peers, ready_offset, *, ring
):
    _rows(rows)
    _host_module().release_rows_retired(
        rows, owner, ready_peers, ready_offset, ring.view, *state
    )


def clear_host_tier_ring(ring) -> None:
    """Clear membership after readers retire, preserving the group epoch."""
    _host_module().clear_ring(ring.view)


def drain_host_tier_readers(state, ready_peers, ready_offset):
    _host_module().drain_readers(ready_peers, ready_offset, *state)
