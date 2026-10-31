"""Explicit B300 TP4/TP8 pack/FlashAG/unpack using ordinary CUDA IPC slabs.

No multicast object is required. GPU descriptors select the live payload on
each replay. All Index groups reuse one owner and one transfer workspace.
"""

import os

from ulacache.compiler import cache_once, load_jit
from . import IndexTransport


@cache_once
def _module(world):
    if world not in (4, 8):
        raise ValueError("Packed unicast FlashAG requires four or eight B300 ranks")
    return load_jit(
        "udcache_b300_packed_flash_ag_v4",
        str(world),
        cuda_files=["csrc/flashag/b300/packed_adapter.cuh"],
        cuda_wrappers=[
            ("gather", f"udcache::index::packed_all_gather<{world}>"),
            ("resident_grid_limit", f"udcache::index::packed_resident_blocks<{world}>"),
        ],
    )


class PackedFlashAGIndexOwner(IndexTransport):
    def __init__(
        self,
        serving_group,
        *,
        max_queries,
        max_page_width,
        device,
        transport_ops,
        ipc_manager_factory,
        compact_scores=None,
    ):
        import torch
        import torch.distributed as dist
        from ulacache.transfer.ipc import FlowKVPeerShards

        for value in (max_queries, max_page_width):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("Index bounds must be positive integers")
        self.device = torch.device(device)
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            raise ValueError("Construct the CUDA Index owner before Graph capture")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.world_size = serving_group.world_size
        self.rank_in_group = serving_group.rank_in_group
        self.cpu_group = serving_group.cpu_group
        if (
            self.world_size not in (4, 8)
            or dist.get_world_size(self.cpu_group) != self.world_size
            or dist.get_rank(self.cpu_group) != self.rank_in_group
            or torch.cuda.get_device_capability(self.device) != (10, 3)
            or transport_ops.target() != "b300"
        ):
            raise ValueError(
                "Packed unicast FlashAG requires four or eight same-host B300 ranks"
            )
        mode = os.environ.get("UDCACHE_INDEX_SCORE_LAYOUT", "dense")
        if mode not in ("dense", "compact"):
            raise ValueError("UDCACHE_INDEX_SCORE_LAYOUT must be dense or compact")
        if compact_scores is not None:
            if type(compact_scores) is not bool:
                raise TypeError("Compact score selection must be boolean")
            mode = "compact" if compact_scores else "dense"
        if any(
            os.environ.get("SGLANG_FLASH_AG_" + suffix)
            for suffix in ("PROBES", "FORCE_ALGO", "PROBE_GRID", "PROBE_GRIDS")
        ):
            raise ValueError(
                "Packed Index FlashAG does not accept vendor probe overrides"
            )
        fingerprint = (
            max_queries,
            max_page_width,
            mode,
            self.world_size,
            "D31-unicast-v2-query-tiles",
        )
        peers = [None] * self.world_size
        dist.all_gather_object(peers, fingerprint, group=self.cpu_group)
        if any(x != fingerprint for x in peers):
            raise ValueError("Packed FlashAG configuration differs across ranks")
        self.max_queries, self.max_page_width = max_queries, max_page_width
        self.local_columns = (
            ((max_page_width + self.world_size - 1) // self.world_size + 3) // 4 * 256
        )
        elements = max_queries * self.local_columns
        if elements * self.world_size >= 1 << 31:
            raise ValueError("GPU score prefixes must fit signed int32")
        self.maximum_gather_bytes = elements * self.world_size * 4
        self.compact_scores = True  # Compute the existing GPU payload descriptors.
        self.unpacks_scores = mode == "dense"
        # Original FlashAG D31 families and fixed per-family slot geometry.
        # The small grid prices actual low-BS payloads instead of padded capacity.
        self._ops = transport_ops
        self.module = _module(self.world_size)
        self.resident_grid_limit = self.module.resident_grid_limit()
        if self.resident_grid_limit < 128:
            raise RuntimeError(
                "FlashAG does not have a co-resident spinning grid on this device"
            )
        limits = [None] * self.world_size
        dist.all_gather_object(limits, self.resident_grid_limit, group=self.cpu_group)
        if any(limit != self.resident_grid_limit for limit in limits):
            raise RuntimeError("FlashAG kernel residency differs across ranks")
        self.maximum_grid = 512 if self.resident_grid_limit >= 512 else 128
        self._families = {
            blocks: transport_ops.family_of(830, blocks, 256)
            for blocks in {128, self.maximum_grid}
        }
        ceilings = {
            family: (
                max_queries if blocks == self.maximum_grid else min(max_queries, 4)
            )
            * self.local_columns
            * self.world_size
            * 4
            for blocks, family in self._families.items()
        }
        self.layout = transport_ops.make_layout(self.world_size, ceilings)
        slab_bytes = (self.layout.total + 255) // 256 * 256
        self.slab = torch.zeros(
            slab_bytes // 4, dtype=torch.float32, device=self.device
        )
        self.slab_peers = FlowKVPeerShards(
            self.slab, self.cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        self.staging = torch.empty(elements, dtype=torch.float32, device=self.device)
        self.packed_workspace = torch.empty(
            elements * self.world_size, dtype=torch.float32, device=self.device
        )
        self.gathered_workspace = (
            torch.empty_like(self.packed_workspace)
            if self.unpacks_scores
            else self.packed_workspace
        )
        self._counters = {
            blocks: torch.zeros(blocks * 8, dtype=torch.uint32, device=self.device)
            for blocks in self._families
        }
        self._closed = False
        self._initialize_dense_gather()

    def all_gather_compact(self, output, local, prefix, packs):
        self._gather(output, local, prefix, packs, unpack=self.unpacks_scores)

    def _gather(self, output, local, prefix, packs, *, unpack):
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
                "Packed FlashAG requires stable aligned FP32 buffers and GPU descriptors"
            )
        blocks = 128 if local.shape[0] <= 4 else self.maximum_grid
        region, stride, size = self.layout.bases(self._families[blocks])
        self.module.gather(
            local,
            output,
            prefix,
            packs,
            self.staging,
            self.packed_workspace,
            self.slab_peers.ptrs,
            self.slab,
            self._counters[blocks],
            self.rank_in_group,
            region,
            stride,
            size,
            blocks,
            unpack,
        )

    def close(self):
        if self._closed:
            return
        import torch
        import torch.distributed as dist

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot close an Index owner during Graph capture")
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu_group)
        self.slab_peers.close()
        dist.barrier(group=self.cpu_group)
        self._release_dense_gather()
        self._closed = True
