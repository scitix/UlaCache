"""Device-length FP32 Index gather over existing same-host CUDA IPC mappings.

The native scorer and compact TopK reader remain unchanged. GPU prefixes
select the payload on every replay; no host length read or NCCL count change
is required. Exported staging is shared serially by all Index groups.
"""

from contextlib import contextmanager

from ulacache.compiler import cache_once, load_jit


@cache_once
def _module(world):
    if world != 4:
        raise ValueError(
            "The experimental B300 compact IPC transport requires four ranks"
        )
    return load_jit(
        "udcache_index_ipc_compact_v2",
        str(world),
        cuda_files=["csrc/indexer/ipc_compact.cuh"],
        cuda_wrappers=[("gather", f"udcache::index::compact_ipc_gather<{world}>")],
    )


class IPCCompactIndexOwner:
    def __init__(
        self, serving_group, *, max_queries, max_page_width, device, ipc_manager_factory
    ):
        import torch
        import torch.distributed as dist
        from ulacache.transfer.ipc import FlowKVPeerShards

        for value in (max_queries, max_page_width):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("Index bounds must be positive integers")
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            raise ValueError("Construct the CUDA IPC Index owner before Graph capture")
        self.world_size = serving_group.world_size
        self.rank_in_group = serving_group.rank_in_group
        self.cpu_group = serving_group.cpu_group
        if (
            self.world_size != 4
            or dist.get_world_size(self.cpu_group) != 4
            or dist.get_rank(self.cpu_group) != self.rank_in_group
            or torch.cuda.get_device_capability(self.device) != (10, 3)
        ):
            raise ValueError(
                "Compact IPC Index is explicitly scoped to four B300 ranks"
            )
        fingerprint = (max_queries, max_page_width, self.world_size)
        configurations = [None] * self.world_size
        dist.all_gather_object(configurations, fingerprint, group=self.cpu_group)
        if any(x != fingerprint for x in configurations):
            raise ValueError("Compact IPC Index bounds differ across ranks")
        self.max_queries, self.max_page_width = max_queries, max_page_width
        self.local_columns = ((max_page_width + 3) // 4 + 3) // 4 * 256
        local_elements = max_queries * self.local_columns
        if local_elements * self.world_size >= 1 << 31:
            raise ValueError("Compact IPC prefix offsets must fit signed int32")
        self.maximum_gather_bytes = self.world_size * local_elements * 4
        self.gathered_workspace = torch.empty(
            self.world_size * local_elements, dtype=torch.float32, device=self.device
        )
        self.staging = torch.empty(
            local_elements, dtype=torch.float32, device=self.device
        )
        # Epoch, publication, reader retirement, pack/pull CTA counters,
        # declared dynamic payload. Keep epochs monotonic across Graph replay.
        self.flags = torch.zeros(8, dtype=torch.int64, device=self.device)
        self.staging_peers = FlowKVPeerShards(
            self.staging, self.cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        self.flag_peers = FlowKVPeerShards(
            self.flags, self.cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        self.module = _module(self.world_size)
        self.compact_scores = True
        self._closed = False

    def validate_bounds(self, max_queries, max_page_width):
        if self._closed or (max_queries, max_page_width) != (
            self.max_queries,
            self.max_page_width,
        ):
            raise ValueError(
                "Shared compact IPC owner requires identical native Graph bounds"
            )

    @contextmanager
    def graph_capture(self):
        if self._closed:
            raise RuntimeError("Compact IPC owner is closed")
        yield

    def all_gather_compact(self, output, local, prefix, packs):
        import torch

        if (
            self._closed
            or local.ndim != 2
            or output.ndim != 2
            or local.dtype != torch.float32
            or output.dtype != torch.float32
            or local.device != self.device
            or output.device != self.device
            or not local.is_contiguous()
            or not output.is_contiguous()
            or output.shape != (self.world_size * local.shape[0], local.shape[1])
            or output.numel() * 4 > self.maximum_gather_bytes
            or output.data_ptr() != self.gathered_workspace.data_ptr()
            or prefix.shape != (local.shape[0] + 1,)
            or prefix.dtype != torch.int32
            or packs.shape != (1,)
            or packs.dtype != torch.int64
            or prefix.device != self.device
            or packs.device != self.device
            or not prefix.is_contiguous()
            or not packs.is_contiguous()
            or local.shape[1] % 4
            or local.data_ptr() % 16
            or output.data_ptr() % 16
        ):
            raise ValueError(
                "Compact IPC requires aligned stable FP32 workspaces and GPU descriptors"
            )
        self.module.gather(
            local,
            output,
            prefix,
            packs,
            self.staging,
            self.flags,
            self.staging_peers.ptrs,
            self.flag_peers.ptrs,
            self.rank_in_group,
        )

    def close(self):
        if self._closed:
            return
        import torch
        import torch.distributed as dist

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot close the Index owner during Graph capture")
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu_group)
        self.staging_peers.close()
        self.flag_peers.close()
        dist.barrier(group=self.cpu_group)
        self._closed = True
