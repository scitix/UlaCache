"""GPU directory metadata mapping logical records to exclusive CPU/GPU slots."""

from __future__ import annotations

from typing import NamedTuple
import torch


class HostTierState(NamedTuple):
    directory: torch.Tensor
    gpu_tags: torch.Tensor
    host_tags: torch.Tensor
    access_time: torch.Tensor
    leases: torch.Tensor
    epoch: torch.Tensor
    gpu_free: torch.Tensor
    host_free: torch.Tensor
    free_counts: torch.Tensor


def create_host_tier_state(
    gpu_slots: int,
    host_slots: int,
    device: torch.device | str,
    *,
    directory: torch.Tensor | None = None,
    access_time: torch.Tensor | None = None,
) -> HostTierState:
    """Allocate only GPU metadata; the host record pool is supplied separately."""
    if gpu_slots < 1 or host_slots < 1:
        raise ValueError("CPU FlowKV requires positive GPU and CPU capacities")
    logical_slots = gpu_slots + host_slots
    if logical_slots > torch.iinfo(torch.int32).max:
        raise ValueError("Owner-local logical rows must fit int32")
    if directory is None:
        directory = torch.full((logical_slots,), -1, dtype=torch.int64, device=device)
    elif (
        directory.shape != (logical_slots,)
        or directory.dtype != torch.int64
        or not directory.is_contiguous()
    ):
        raise ValueError("directory must be contiguous int64[gpu_slots + host_slots]")
    else:
        directory.fill_(-1)
    if access_time is None:
        access_time = torch.zeros(logical_slots, dtype=torch.int64, device=device)
    if (
        access_time.shape != (logical_slots,)
        or access_time.dtype != torch.int64
        or not access_time.is_contiguous()
    ):
        raise ValueError(
            "FlowKV access-time view must be contiguous int64 logical rows"
        )
    return HostTierState(
        directory,
        torch.full((gpu_slots,), -1, dtype=torch.int32, device=device),
        torch.full((host_slots,), -1, dtype=torch.int32, device=device),
        access_time,
        torch.zeros(gpu_slots, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.arange(gpu_slots, dtype=torch.int32, device=device),
        torch.arange(host_slots, dtype=torch.int32, device=device),
        torch.tensor([gpu_slots, host_slots], dtype=torch.int32, device=device),
    )


def clear_host_tier_state(state: HostTierState) -> None:
    """Reset a drained group without replacing IPC-exported directory storage."""
    state.directory.fill_(-1)
    state.gpu_tags.fill_(-1)
    state.host_tags.fill_(-1)
    state.access_time.zero_()
    state.leases.zero_()
    # Published group epochs are monotonic for the IPC lifetime.
    torch.arange(state.gpu_free.numel(), out=state.gpu_free)
    torch.arange(state.host_free.numel(), out=state.host_free)
    state.free_counts[0].fill_(state.gpu_free.numel())
    state.free_counts[1].fill_(state.host_free.numel())
