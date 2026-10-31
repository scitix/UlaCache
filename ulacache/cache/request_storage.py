# SPDX-License-Identifier: Apache-2.0
"""Whole-request physical residency using existing FlowKV byte transfers.

Admission, native logical allocation and prefix locks retain their original
owners. The request route has no per-Index-group State/ring/manager. Runtime
addressing/allocation hooks must be installed before this route can execute.
"""

from __future__ import annotations

from ulacache.cache.request_pages import FlowKVRequestPagePool
from ulacache.transfer.page_exchange import (
    NativeExchangeBoundaryExecutor,
    BoundaryCompletion as BoundaryCompletion,
)


def request_workspace_rows(layout, peers):
    return max(layout.page_size, layout.max_query_tokens * (peers * layout.topk + 1))


class LocalPageMapPublisher:
    """A local mirror of the identical native TP superpage schedule.

    No remote directory read or boundary collective is used. Native Unified
    relative recency with a stable node-ID tie, identical boundary input order
    and deterministic physical min heaps must produce the same map on peers.
    """

    def __init__(self, owner_rank, protocol, device):
        import torch

        self.owner_rank = owner_rank
        self.protocol = protocol
        self.host_map = torch.tensor(
            protocol.map_row_snapshot(), dtype=torch.int32, device="cpu"
        )
        self.device_map = self.host_map.to(device)

    def publish(self, owner_rank, updates):
        if owner_rank != self.owner_rank:
            raise ValueError("publisher belongs to another owner")
        if self.protocol.pending_native_binding_count:
            raise RuntimeError(
                "settle every pending native binding before boundary publication"
            )
        for logical, physical in updates:
            self.host_map[logical] = physical
        # Boundary-only publication; consumer model stream follows this copy.
        # Fresh allocation's GPU binding is a separate new-page-only entry.
        self.device_map.copy_(self.host_map, non_blocking=False)

    def record_completed_bindings(self, updates):
        """Settle CPU metadata after the original allocator's GPU scatter.

        This does not copy or inspect the GPU table. A later boundary publish
        must retain fresh bindings already written by the allocator.
        """
        for logical, physical in updates:
            self.host_map[logical] = physical


class FlowKVRequestTier:
    """Physical all-layer MLA/Index storage beside the native logical pool."""

    def __init__(self, pool, host_rows, *, host_pool_factory, host_memory_budget_bytes):
        self.pool = pool
        self.physical_gpu_rows = pool.flowkv_shards.shape[1]
        self.physical_cpu_rows = host_rows
        self.logical_owner_rows = pool._flowkv_global_rows // pool._flowkv_storage_size
        # Native DSA sets skip_topk_layers after _create_buffers returns. The
        # early record layout already contains the same local anchor plan.
        self.anchors = tuple(pool.flowkv_layout.layer_groups.anchors)
        self.protocol = FlowKVRequestPagePool(
            owner_count=pool._flowkv_storage_size,
            owner_rank=pool.flowkv_rank,
            gpu_token_rows=self.physical_gpu_rows,
            cpu_token_rows=host_rows,
            logical_owner_token_rows=self.logical_owner_rows,
            layer_num=pool.layer_num,
            row_bytes=pool.flowkv_layout.row_bytes,
            index_anchor_layers=self.anchors,
            reserved_gpu_pages=(0,),
            sentinel_gpu_page=0,
        )
        self.gpu_index_shape = self.protocol.plane_shapes()["index_gpu"]
        self.gpu_page_count = self.protocol.gpu_page_count
        self.cpu_page_count = self.protocol.cpu_page_count
        self.logical_page_count = self.protocol.logical_owner_pages
        self.allocation_page_size = self.protocol.allocation_page_size
        self.gpu_usable_pages = self.protocol.gpu_usable_pages
        self.cpu_usable_pages = self.protocol.cpu_usable_pages
        workspace_rows = request_workspace_rows(
            pool.flowkv_layout, pool._flowkv_storage_size
        )
        host_bytes = (
            pool.layer_num * host_rows * pool.flowkv_layout.row_bytes
            + len(self.anchors) * (host_rows // 64) * 8448
            + 20 * workspace_rows
            + 4 * self.logical_page_count
            + 256
        )
        if host_bytes > host_memory_budget_bytes(host_bytes):
            raise ValueError(
                "complete request MLA and Index planes exceed the native host-memory budget"
            )
        self.host = host_pool_factory(
            layers=pool.layer_num,
            slots=host_rows,
            row_bytes=pool.flowkv_layout.row_bytes,
            device=pool.device,
        )
        try:
            self.index_host = host_pool_factory(
                layers=len(self.anchors),
                slots=host_rows // 64,
                row_bytes=8448,
                device=pool.device,
            )
        except Exception:
            self.host.close()
            raise
        self.publisher = LocalPageMapPublisher(
            pool.flowkv_rank, self.protocol, pool.device
        )
        self.page_map = self.publisher.device_map
        self.executor = NativeExchangeBoundaryExecutor(
            mla_cpu=self.host.storage,
            mla_gpu=pool.flowkv_shards,
            index_cpu=self.index_host.storage,
            device=pool.device,
            capacity=workspace_rows,
            readers_retired=self._readers_retired,
        )
        self._runtime = None

    def bind_runtime(self, runtime):
        """Install real mapped address/new-page/retirement entries at startup.

        The runtime object owns CUDA/JIT ABI glue. It must certify the exact
        map and physical shapes, and implement the original-reader boundary.
        This is a capability binding, not a per-token protection flag.
        """
        if runtime is None:
            raise ValueError(
                "request physical-address and retirement runtime is required"
            )
        runtime.validate(self)
        self._runtime = runtime

    def require_runtime(self):
        if self._runtime is None:
            raise NotImplementedError(
                "request policy physical addressing, new-page allocation and reader-retirement hooks are not connected"
            )
        return self._runtime

    def bind_index_storage(self, storage):
        if tuple(storage.shape) != self.gpu_index_shape:
            raise ValueError(
                "request physical Index shape disagrees with the sized pool"
            )
        self.executor.bind_index(storage)

    def _readers_retired(self):
        return self.require_runtime().readers_retired(self)

    def check_gpu_capacity(self, *args, **kwargs):
        return self.protocol.check_gpu_capacity(*args, **kwargs)

    def location(self, logical_page):
        return self.protocol.location(logical_page)

    def reserve_request(self, *args, **kwargs):
        return self.protocol.reserve_request(*args, **kwargs)

    def ensure_resident(self, tx):
        self.require_runtime()
        return self.protocol.ensure_resident(tx, self.executor)

    def commit(self, tx, completion):
        # submit enqueues the two planes; native admission can only publish
        # their new location after this real boundary event has completed.
        completion.wait()
        return self.protocol.commit(tx, completion, self.publisher)

    def consume_future_pages(self, reservation, count):
        return self.protocol.consume_future_pages(reservation, count)

    def return_consumed_pages(self, reservation, physical_pages):
        return self.protocol.return_consumed_pages(reservation, physical_pages)

    def record_consumed_pages(self, logical_pages, physical_pages):
        logical_pages, physical_pages = tuple(logical_pages), tuple(physical_pages)
        self.protocol.record_consumed_pages(logical_pages, physical_pages)
        self.publisher.record_completed_bindings(
            tuple(zip(logical_pages, physical_pages))
        )

    def attach_native_pages(self, logical_pages, **kwargs):
        self.require_runtime()
        return self.protocol.attach_native_pages(
            logical_pages, publisher=self.publisher, **kwargs
        )

    def retire_native_pages(self, logical_pages, *, readers_retired):
        self.require_runtime()
        return self.protocol.retire_native_pages(
            logical_pages, readers_retired=readers_retired, publisher=self.publisher
        )

    def reservation_remaining_pages(self, reservation):
        return self.protocol.reservation_remaining_pages(reservation)

    def release_reservation(self, reservation):
        return self.protocol.release_reservation(reservation)

    def cancel(self, tx):
        return self.protocol.cancel(tx)

    def clear(self):
        self.executor.stream.synchronize()
        self.protocol.clear(readers_retired=self._readers_retired())
        self.publisher.publish(
            self.pool.flowkv_rank, tuple(enumerate(self.protocol.map_row_snapshot()))
        )

    @property
    def gpu_free_pages(self):
        return self.protocol.gpu_free_pages

    @property
    def metadata_bytes(self):
        return (
            self.page_map.nbytes
            + self.executor.plan.nbytes
            + self.executor.counters.nbytes
            + self.host.ptrs.nbytes
            + self.index_host.ptrs.nbytes
        )

    @property
    def host_payload_bytes(self):
        return self.host.nbytes + self.index_host.nbytes

    @property
    def host_metadata_bytes(self):
        return (
            self.publisher.host_map.nbytes
            + self.executor.host_plan.nbytes
            + self.executor.page_offsets.nbytes
        )

    def capacity_audit(self):
        return {
            "policy": "request",
            "owner_rank": self.pool.flowkv_rank,
            "logical_owner_rows": self.logical_owner_rows,
            "physical_gpu_rows": self.physical_gpu_rows,
            "physical_cpu_rows": self.physical_cpu_rows,
            "gpu_usable_pages": self.protocol.gpu_usable_pages,
            "cpu_usable_pages": self.protocol.cpu_usable_pages,
            "map_shape": tuple(self.page_map.shape),
            "plane_shapes": self.protocol.plane_shapes(),
            "metadata_bytes": self.metadata_bytes,
            "host_payload_bytes": self.host_payload_bytes,
            "host_metadata_bytes": self.host_metadata_bytes,
            "runtime_connected": self._runtime is not None,
        }

    def close(self):
        # The native pool close drains StepPublication/forwards first.
        self.executor.close()
        self.index_host.close()
        self.host.close()
        del self.page_map, self.publisher
