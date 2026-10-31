# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC mappings for GPU storage exported by cache owners on one host."""

from __future__ import annotations

import socket

import torch
import torch.distributed as dist


class FlowKVPeerShards:
    """Keep peer mappings alive while kernels read each owner's GPU storage.

    The allocation is an ordinary CUDA tensor so PD transfer engines can
    register its local shard.  Peer mappings are set up once, before any CUDA
    graph capture, using the IPC manager already used by SGLang all-reduce.
    """

    def __init__(
        self,
        storage: torch.Tensor,
        cpu_group: dist.ProcessGroup,
        *,
        ipc_manager_factory,
    ) -> None:
        if storage.device.type != "cuda" or not storage.is_contiguous():
            raise ValueError("FlowKV peer storage must be a contiguous CUDA tensor")

        self.storage = storage
        self._ipc_manager = ipc_manager_factory()
        rank = dist.get_rank(cpu_group)
        world_size = dist.get_world_size(cpu_group)
        hosts = [None] * world_size
        dist.all_gather_object(hosts, socket.gethostname(), group=cpu_group)
        if len(set(hosts)) != 1:
            raise ValueError("FlowKV attention TP peers must reside on one host")

        # cudaIpcGetMemHandle requires the allocation to exist on every peer
        # before another rank can open it.  This is startup-only work.
        torch.cuda.synchronize(storage.device)
        try:
            handles = self._ipc_manager.batch_get_handles([storage.data_ptr()])
        except RuntimeError as exc:
            raise RuntimeError(
                "FlowKV TP peer mapping requires cudaMalloc-backed KV storage"
            ) from exc
        local = [(list(handle), int(offset)) for handle, offset in handles]
        gathered: list[list[tuple[list[int], int]] | None] = [None] * world_size
        dist.all_gather_object(gathered, local, group=cpu_group)

        peer_ptrs = []
        for peer_rank, remote in enumerate(gathered):
            if remote is None or len(remote) != 1:
                raise RuntimeError("FlowKV TP peer did not publish its KV shard")
            if peer_rank == rank:
                peer_ptrs.append(storage.data_ptr())
            else:
                try:
                    peer_ptrs.append(
                        next(iter(self._ipc_manager.batch_open_handles(remote)))
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "FlowKV could not open a local TP KV shard through CUDA IPC"
                    ) from exc

        self.ptrs = torch.tensor(peer_ptrs, dtype=torch.uint64, device=storage.device)
        torch.cuda.synchronize(storage.device)
        dist.barrier(group=cpu_group)

    @property
    def device_ptr(self) -> int:
        return self.ptrs.data_ptr()

    def close(self) -> None:
        torch.cuda.synchronize(self.storage.device)
        self._ipc_manager.destroy()


class FlowKVStepPublication:
    """GPU epochs for serial ordinary-TP decode forwards.

    All peers execute the same begin/publish sequence, including padded idle.
    Flags and arena.step must never be reset separately. close() is collective
    after all forwards/graphs drain, before destroying the startup CPU group.
    No CPU operation runs per decode.
    """

    def __init__(
        self, arena, module, cpu_group, *, ipc_manager_factory, host_groups=None
    ):
        self.arena, self.module, self.cpu_group = arena, module, cpu_group
        self.host_groups = host_groups
        self.ready = torch.zeros_like(arena.step)
        self.peers = FlowKVPeerShards(
            self.ready, cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        # Startup-only validation of the actual imported GPU addresses.
        for pointer in self.peers.ptrs.cpu().tolist():
            module.validate_step_peer(self.ready, int(pointer))
        self._closed = False

    @property
    def nbytes(self):
        return self.ready.nbytes + self.peers.ptrs.nbytes

    def begin(self, current, valid):
        arena = self.arena
        self.module.begin_published(
            current,
            valid,
            arena.layout.groups,
            arena.layout.pool_rows,
            arena.step,
            arena.active,
            arena.current_keys,
            arena.current_values,
            arena.cursors,
            arena.totals,
            arena.hints,
            arena.counts,
            arena.histograms,
            self.peers.ptrs,
            *((self.host_groups,) if self.host_groups is not None else ()),
        )

    def publish(self):
        self.module.publish_step(
            self.arena.step,
            self.ready,
            *((self.host_groups,) if self.host_groups is not None else ()),
        )

    def close(self):
        if self._closed:
            return
        torch.cuda.synchronize(self.ready.device)
        # Drain all consumers before any IPC mapping is removed; close all
        # imports before a rank can release its exported flag allocation.
        dist.barrier(group=self.cpu_group)
        self.peers.close()
        dist.barrier(group=self.cpu_group)
        self._closed = True
