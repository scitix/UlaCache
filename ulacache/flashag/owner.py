"""Private FP32 Index all-gather over the pinned FlashAG implementation.

Importing this module does not import FlashAG or allocate CUDA resources.
Construction is collective and must precede CUDA Graph capture. Serving TP
communication remains owned by the original GroupCoordinator.
"""

from __future__ import annotations

import os
from . import IndexTransport


class FlashAGIndexOwner(IndexTransport):
    def __init__(
        self,
        serving_group,
        *,
        max_queries,
        max_page_width,
        device,
        transport_ops,
        compact_scores=None,
    ):
        import torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        for value in (max_queries, max_page_width):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("Index communication bounds must be positive integers")
        self.device = torch.device(device)
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Construct the CUDA Index owner before Graph capture")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.world_size = serving_group.world_size
        self.rank_in_group = serving_group.rank_in_group
        self.unique_name = serving_group.unique_name + "/index-flash-ag"
        self.cpu_group = serving_group.cpu_group
        if (
            dist.get_world_size(self.cpu_group) != self.world_size
            or dist.get_rank(self.cpu_group) != self.rank_in_group
        ):
            raise ValueError(
                "Private Index rank order disagrees with the serving group"
            )
        if self.world_size < 2 or self.world_size & (self.world_size - 1):
            raise ValueError(
                "CP Index communication requires a supported power-of-two group"
            )
        self.max_queries = max_queries
        self.max_page_width = max_page_width
        self.local_columns = (
            ((max_page_width + self.world_size - 1) // self.world_size + 3) // 4 * 256
        )
        self.maximum_gather_bytes = (
            self.world_size * max_queries * self.local_columns * 4
        )
        int64_max = (1 << 63) - 1
        if self.maximum_gather_bytes > int64_max:
            raise ValueError("Native Index workspace exceeds the int64 payload ABI")
        for suffix in ("PROBES", "FORCE_ALGO", "PROBE_GRID", "PROBE_GRIDS"):
            if os.environ.get("SGLANG_FLASH_AG_" + suffix):
                raise ValueError(
                    "Index owner does not admit FlashAG algorithm/grid probes"
                )
        declared = os.environ.get("SGLANG_FLASH_AG_PROBE_MAX_KB", "")
        if declared and (not declared.isdecimal() or int(declared) < 1):
            raise ValueError("FlashAG declared capacity must be a positive integer KiB")
        if declared and int(declared) > int64_max // 1024:
            raise ValueError("FlashAG declared capacity exceeds the int64 payload ABI")
        if declared and int(declared) * 1024 < self.maximum_gather_bytes:
            raise ValueError(
                "FlashAG declared capacity does not cover native Graph geometry"
            )

        ops = transport_ops

        if ops.target() is None or self.world_size not in ops.SUPPORTED_WORLD_SIZES:
            raise RuntimeError(
                "FlashAG has no policy for this CUDA architecture/Index group"
            )
        self._ops = ops
        compact_default = (
            "1" if ops.target() == "h200" and self.world_size == 8 else "0"
        )
        compact = os.environ.get(
            "UDCACHE_INDEX_COMPACT_SCORES",
            os.environ.get("SGLANG_DSA_CP_COMPACT_SCORES", compact_default),
        )
        if (
            "UDCACHE_INDEX_COMPACT_SCORES" in os.environ
            and "SGLANG_DSA_CP_COMPACT_SCORES" in os.environ
            and os.environ["UDCACHE_INDEX_COMPACT_SCORES"]
            != os.environ["SGLANG_DSA_CP_COMPACT_SCORES"]
        ):
            raise ValueError("Conflicting UDCache and legacy compact score settings")
        if compact not in ("0", "1"):
            raise ValueError("UDCACHE_INDEX_COMPACT_SCORES must be 0 or 1")
        if compact_scores is not None and type(compact_scores) is not bool:
            raise TypeError("Compact score selection must be boolean")
        self.compact_scores = (
            compact == "1" if compact_scores is None else compact_scores
        )
        self.unpacks_scores = False
        if getattr(ops, "compact_only", False) and not self.compact_scores:
            raise NotImplementedError(
                "This portable H200 binding exposes compact FP32 scores only"
            )
        if self.compact_scores and (ops.target() != "h200" or self.world_size != 8):
            raise ValueError(
                "Compact CP scores require the validated H200 eight-rank FlashAG transport"
            )
        table = ops.POLICY_BY_WORLD[self.world_size]
        # Same top-band extension as vendor PROBE_MAX_KB, private to this owner.
        # The algorithm/grid and device protocol are unchanged; no global table
        # or process environment is rewritten for another communicator.
        self.maximum_payload_bytes = max(
            table[-1].max_bytes,
            (self.maximum_gather_bytes + 1023) // 1024 * 1024,
            int(declared) * 1024 if declared else 0,
        )
        self._table = table[:-1] + (
            table[-1]._replace(max_bytes=self.maximum_payload_bytes),
        )
        fingerprint = (
            ops.target(),
            self.world_size,
            max_queries,
            max_page_width,
            self.maximum_payload_bytes,
            tuple(self._table),
            tuple(
                sorted(
                    (key, value)
                    for key, value in os.environ.items()
                    if key.startswith("SGLANG_FLASH_AG")
                )
            ),
            self.compact_scores,
        )
        configurations = [None] * self.world_size
        dist.all_gather_object(configurations, fingerprint, group=self.cpu_group)
        if any(configuration != fingerprint for configuration in configurations):
            raise RuntimeError("Index FlashAG configuration differs across ranks")
        ceilings = {}
        for band in self._table:
            family = ops.family_of(band.algo, band.num_blocks, band.num_threads)
            ceilings[family] = max(ceilings.get(family, 0), band.max_bytes)
        self.layout = ops.make_layout(self.world_size, ceilings)
        slab_bytes = (self.layout.total + (2 << 20) - 1) // (2 << 20) * (2 << 20)
        if slab_bytes > int64_max:
            raise ValueError("FlashAG slab exceeds the int64 allocation ABI")
        self.slab = symm_mem.empty(
            slab_bytes // 4, device=self.device, dtype=torch.float32
        )
        self.slab.zero_()
        self._handle = symm_mem.rendezvous(self.slab, self.cpu_group.group_name)
        if int(self._handle.multicast_ptr) == 0:
            raise RuntimeError("Index FlashAG requires an NVLink multicast mapping")
        if self.compact_scores:
            from ulacache.flashag.compact import compile_compact

            compile_compact(self.world_size)
        else:
            ops.compile_all_gather(torch.float32, self.world_size)
        torch.cuda.synchronize(self.device)
        self._handle.barrier()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu_group)
        if hasattr(self._handle, "get_buffer_ptrs"):
            peers = self._handle.get_buffer_ptrs()
        else:
            peers = self._handle.buffer_ptrs
        self._peer_table = torch.tensor(
            [int(peers[rank]) for rank in range(self.world_size)],
            dtype=torch.int64,
            device=self.device,
        )
        self._counters = {
            family: torch.zeros(
                max(1, threads // 32), dtype=torch.uint32, device=self.device
            )
            for family, threads in ops.counter_slots(self.world_size).items()
        }
        self.gathered_workspace = torch.empty(
            self.maximum_gather_bytes // 4,
            dtype=torch.float32,
            device=self.device,
        )
        self._closed = False
        self._initialize_dense_gather()

    def all_gather_compact(self, output, local, prefix, packs):
        if self._closed:
            raise RuntimeError("Index owner is closed")
        from ulacache.flashag.compact import all_gather

        all_gather(self, output, local, prefix, packs)

    def close(self):
        """All ranks call after releasing captured Graphs; no per-step barrier."""
        if self._closed:
            return
        import torch
        import torch.distributed as dist

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot close an Index owner during Graph capture")
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu_group)
        self.gathered_workspace = None
        self._counters.clear()
        self._peer_table = None
        self.slab = None
        self._handle = None
        self._release_dense_gather()
        self._closed = True
