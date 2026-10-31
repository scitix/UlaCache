"""Host-tier residency and its shared transfer workspace."""

from __future__ import annotations

import torch
from ulacache.transfer.ipc import FlowKVPeerShards


class FlowKVHostTier:
    """Stable Index identities with per-group GPU/CPU residency.

    Allocation and release consume native interleaved locations. Transport
    uses the existing rank-major global IDs. Neither path moves Index keys.
    """

    def __init__(
        self,
        pool,
        host_slots,
        cpu_group,
        *,
        host_pool_factory,
        ipc_manager_factory,
        workspace_factory=None,
        exchange_launch=(4, 1024),
        parallel_finish=False,
        parallel_victims=False,
    ):
        from ulacache.cache.kernels.cpu_residency import (
            create_host_tier_state,
            create_host_tier_workspace,
            validate_exchange_launch,
        )

        self.exchange_launch = validate_exchange_launch(exchange_launch)
        if type(parallel_finish) is not bool or type(parallel_victims) is not bool:
            raise ValueError("Host parallel finish and victim options must be boolean")
        if parallel_victims and not parallel_finish:
            raise ValueError("Parallel host victims require parallel host finish")
        self.parallel_finish = parallel_finish
        self.parallel_victims = parallel_victims
        self.pool = pool
        self._reserved_native_rows = None
        self.groups = pool.flowkv_layout.layer_groups
        self.logical_slots = pool._flowkv_global_rows // pool._flowkv_storage_size
        self.host_slots = host_slots
        self.gpu_slots = pool.flowkv_shards.shape[1]
        if self.logical_slots != self.gpu_slots + self.host_slots:
            raise ValueError(
                "FlowKV logical rows must equal GPU plus CPU physical rows"
            )
        self.host = host_pool_factory(
            layers=pool.flowkv_layout.layers,
            slots=host_slots,
            row_bytes=pool.flowkv_layout.row_bytes,
            device=pool.device,
        )
        self.directory = torch.full(
            (self.groups.group_count, self.logical_slots),
            -1,
            dtype=torch.int64,
            device=pool.device,
        )
        self.access_time = torch.zeros(
            (self.groups.group_count, self.logical_slots),
            dtype=torch.int64,
            device=pool.device,
        )
        self.states = tuple(
            create_host_tier_state(
                self.gpu_slots,
                self.host_slots,
                pool.device,
                directory=self.directory[group],
                access_time=self.access_time[group],
            )
            for group in range(self.groups.group_count)
        )
        queries, topk = pool.flowkv_layout.max_query_tokens, pool.flowkv_layout.topk
        # Only the actual miss prefix is written. Existing anchor/follower
        # dependencies retire these physical rows before the next plan.
        self.physical_rows = torch.empty(
            (self.groups.group_count, queries, topk),
            dtype=torch.int32,
            device=pool.device,
        )
        capacity = queries * (pool._flowkv_storage_size * topk + 1)
        self.row_batch_limit = max(capacity, pool.allocation_page_size)
        self.workspaces = (
            tuple(
                create_host_tier_workspace(capacity, pool.device) for _ in self.states
            )
            if workspace_factory is None
            else workspace_factory(
                capacity,
                len(self.states),
                self.gpu_slots,
                pool.device,
            )
        )
        from ulacache.eviction.lru import create_presence_ring

        self.rings = tuple(
            create_presence_ring(state, workspace)
            for state, workspace in zip(self.states, self.workspaces)
        )
        # The former histogram word carries the owner-local Group88 view.
        # All backing tensors remain retained for every captured parent/child.
        self.workspaces = tuple(
            workspace._replace(histogram=ring.view)
            for workspace, ring in zip(self.workspaces, self.rings)
        )
        self.ready = torch.zeros(
            (self.groups.group_count, 2),
            dtype=torch.int64,
            device=pool.device,
        )
        self.miss_ready = torch.zeros(
            self.groups.group_count,
            dtype=torch.int64,
            device=pool.device,
        )
        from ulacache.compiler import recency_policy

        options = getattr(getattr(pool, "udcache_services", None), "config", None)
        self.deferred_epochs = (
            torch.zeros(self.groups.group_count, dtype=torch.int64, device=pool.device)
            if bool(getattr(options, "prefetch", False)) and recency_policy() != "naive"
            else None
        )

        self.directory_peers = FlowKVPeerShards(
            self.directory, cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        self.ready_peers = FlowKVPeerShards(
            self.ready, cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        self.access_time_peers = FlowKVPeerShards(
            self.access_time, cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        self.pull_rows_peers = FlowKVPeerShards(
            pool.flowkv_arena.pull_rows,
            cpu_group,
            ipc_manager_factory=ipc_manager_factory,
        )
        self.pull_counts_peers = FlowKVPeerShards(
            pool.flowkv_arena.counts, cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        self.miss_ready_peers = FlowKVPeerShards(
            self.miss_ready, cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        # Host virtual addresses belong to this process. Peers only import
        # CUDA-resident directories and GPU KV; CPU IO stays on the owner.
        host_pointer = int(self.host.ptrs[0].item())
        self.descriptors = tuple(
            torch.tensor(
                [
                    self.directory_peers.device_ptr,
                    host_pointer,
                    self.ready_peers.device_ptr,
                    self.logical_slots,
                    self.gpu_slots,
                    self.host_slots,
                    self.groups.group(layer),
                    self.states[self.groups.group(layer)].epoch.data_ptr(),
                    self.access_time_peers.device_ptr,
                    self.physical_rows[self.groups.group(layer)].data_ptr(),
                    self.rings[self.groups.group(layer)].view.data_ptr(),
                ],
                dtype=torch.int64,
                device=pool.device,
            )
            for layer in range(self.groups.layer_count)
        )
        from ulacache.transfer.kernels.device_graph import device_graph_module

        if len(self.states) > 120:
            raise ValueError(
                "CPU tier worst-case dispatches exceed the CUDA limit of 120 fire-and-forget children per parent execution"
            )
        cold = device_graph_module(
            exchange_blocks=self.exchange_launch[0],
            exchange_threads=self.exchange_launch[1],
            parallel_victims=parallel_victims,
        )
        self._cold_module = cold
        self.cold_execs = []
        for group, (state, workspace) in enumerate(zip(self.states, self.workspaces)):
            anchor = self.groups.anchors[group]
            followers = self.groups.followers[group]
            end = followers[-1] + 1 if followers else anchor + 1
            self.cold_execs.append(
                cold.create(
                    self.host.storage[anchor:end],
                    pool.flowkv_shards[anchor:end],
                    self.ready[group, 0:1],
                    *state,
                    *workspace,
                )
            )
        # The first 17 fields retain the State9/workspace ABI. Remaining fields
        # reference the original local-buffer miss plan; no TopK copy exists.
        self.fused_metadata = torch.tensor(
            [
                [tensor.data_ptr() for tensor in (*state, *workspace)]
                + [
                    self.pull_rows_peers.device_ptr,  # per-peer [group, query, miss]
                    self.pull_counts_peers.device_ptr,  # per-peer [group, query]
                    self.miss_ready_peers.device_ptr,
                    group,
                    self.ready[group, 0:1].data_ptr(),
                    self.gpu_slots,
                    self.host_slots,
                    workspace.new_rows.numel(),
                    pool.flowkv_rank,
                    pool._flowkv_storage_size,
                    queries,
                    topk,
                    self.cold_execs[group],
                    self.deferred_epochs[group : group + 1].data_ptr()
                    if self.deferred_epochs is not None
                    else 0,
                ]
                for group, (state, workspace) in enumerate(
                    zip(self.states, self.workspaces)
                )
            ],
            dtype=torch.int64,
            device=pool.device,
        )

    def _global_ids(self, native_rows):
        # Reuse the existing native/global translation kernel, including its
        # zero sentinel. Avoid a chain of tiny tensor arithmetic launches and
        # keep eager prefill's temporary bytes within its explicit budget.
        rows = torch.empty(
            native_rows.numel(), dtype=torch.int32, device=self.pool.device
        )
        self.pool.flowkv_arena.kernel().globalize_current(
            native_rows,
            rows,
            self.pool._flowkv_owner_start,
        )
        return rows

    def allocate(self, native_rows, *, prefer_gpu=True):
        from ulacache.cache.kernels.cpu_residency import register_native_host_tier_groups

        register_native_host_tier_groups(
            native_rows,
            self.fused_metadata,
            self.pool._flowkv_owner_start,
            prefer_gpu=prefer_gpu,
        )

    def reserve_native_padding(self, native_rows):
        """Bind framework-reserved page-zero aliases before graph warmup.

        These identities are outside real request admission. Native linear
        verification writes nonzero padding slots in the reserved first page.
        Preserve those bindings whenever the original allocator is cleared.
        """
        if self._reserved_native_rows is not None:
            raise ValueError("Native padding aliases are configured once")
        self._reserved_native_rows = native_rows
        self.allocate(native_rows)

    def release(self, native_rows):
        from ulacache.cache.kernels.cpu_residency import release_retired_host_tier_rows

        for chunk in native_rows.split(self.row_batch_limit):
            rows = self._global_ids(chunk)
            for group, state in enumerate(self.states):
                release_retired_host_tier_rows(
                    rows,
                    self.pool.flowkv_rank,
                    state,
                    self.ready_peers.ptrs,
                    group * 2 + 1,
                    ring=self.rings[group],
                )
            # Hints must be invalidated before a logical row is reused.
            self.pool.flowkv_arena.invalidate(rows)
            del rows

    def clear(self):
        from ulacache.cache.kernels.cpu_residency import (
            clear_host_tier_ring,
            clear_host_tier_state,
            drain_host_tier_readers,
        )
        from ulacache.cache.tp_cache import join_local_residency_clear

        join_local_residency_clear(self.pool)

        for group, state in enumerate(self.states):
            drain_host_tier_readers(state, self.ready_peers.ptrs, group * 2 + 1)
            clear_host_tier_ring(self.rings[group])
            clear_host_tier_state(state)
        # Preserve monotonic group/step flags across cache flush. A peer may
        # still be acquiring the old epoch while this rank finishes reset.
        self.pool.flowkv_arena.clear_local_residency(readers_retired=True)
        if self._reserved_native_rows is not None:
            self.allocate(self._reserved_native_rows)

    def drain_readers(self):
        """Retire native peer readers before host publication between forwards.

        This drains cache readers, not Mooncake remote writers. Native transport
        completion or acknowledged deferred retirement must establish the latter.
        Host memory may be mutated immediately after this method returns.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Host publication cannot drain readers during CUDA Graph capture"
            )
        lifecycle = getattr(self.pool, "flowkv_shared_index", None)
        if (
            lifecycle is None
            or lifecycle.pool is not self.pool
            or lifecycle.groups is not self.groups
        ):
            raise ValueError(
                "Host publication requires its actual native shared Index lifecycle"
            )
        lifecycle.join_idle_prefetch()
        from ulacache.cache.kernels.cpu_residency import drain_host_tier_readers

        for group, state in enumerate(self.states):
            drain_host_tier_readers(state, self.ready_peers.ptrs, group * 2 + 1)
        # Device-side epoch waits must complete before CPU destination writes.
        torch.cuda.current_stream(self.pool.device).synchronize()

    def _prepare(self, layer, selected, current, active):
        from ulacache.cache.kernels.cpu_residency import (
            commit_host_tier,
            exchange_host_tier,
            prepare_host_tier,
        )

        pool = self.pool
        group = self.groups.group(layer)
        state, workspace = self.states[group], self.workspaces[group]
        anchor = self.groups.anchors[group]
        followers = self.groups.followers[group]
        end = followers[-1] + 1 if followers else anchor + 1
        prepare_host_tier(
            selected,
            current,
            active,
            pool.flowkv_rank,
            state,
            workspace,
        )
        exchange_host_tier(
            self.host.storage[anchor:end],
            pool.flowkv_shards[anchor:end],
            workspace,
            item_size_bytes=pool.flowkv_layout.row_bytes,
        )
        commit_host_tier(state, workspace)
        # Publish owner completion only after every member's bytes and the
        # directory are ready. Remote VA reads acquire the same group epoch.
        pool.flowkv_arena.kernel().host_publish(
            self.descriptors[layer],
            self.ready,
            pool.flowkv_rank,
        )

    def prepare_group(self, layer, *, queries=0):
        group = self.groups.group(layer)
        if self.groups.anchors[group] != layer:
            raise RuntimeError(
                "CPU decode residency must be planned at an Index anchor"
            )
        anchor = self.groups.anchors[group]
        followers = self.groups.followers[group]
        end = followers[-1] + 1 if followers else anchor + 1
        # The existing miss allocator plans cold IO. A capture-only helper
        # launches that preplanned child; eager uses the same host launch.
        if type(queries) is not int or not 0 <= queries < 2**31:
            raise ValueError("Host finish queries must be a nonnegative int32 count")
        # Match common.cuh's static Q/recency predicate. At Q1..7 its last
        # producer CTA already finishes metadata; do not reserve it twice.
        # Larger shapes retain DPA's already-retired-producer finish kernels.
        parallel = self.parallel_finish and queries >= 8
        from ulacache.compiler import recency_policy

        prepared = 0 < queries < 8 and recency_policy() != "naive"
        finish = (
            self._cold_module.finish_prepared
            if prepared
            else self._cold_module.finish_parallel
            if parallel
            else self._cold_module.finish
        )
        finish(
            self.pool.flowkv_shards[anchor:end],
            self.cold_execs[group],
            self.ready[group, 0:1],
            self.ready_peers.ptrs,
            group * 2,
            *((queries,) if parallel else ()),
            *self.states[group],
            *self.workspaces[group],
        )

    def finish_deferred_group(self, layer, *, queries):
        """Schedule only the captured early GPU-only route after payload publication."""
        if self.deferred_epochs is None or not 0 < queries < 8:
            return False
        group = self.groups.group(layer)
        if self.groups.anchors[group] != layer:
            raise RuntimeError("Deferred residency belongs to an Index anchor")
        self.pool.flowkv_arena.kernel().finish_deferred_group(
            self.fused_metadata[group : group + 1],
            self.pool.flowkv_arena.step,
            self.pool.flowkv_arena.active,
            queries,
        )
        return True

    @property
    def prefill_chunk_rows(self):
        capacity = self.workspaces[0].new_rows.numel()
        return min(self.gpu_slots, capacity // 2)

    def prepare_prefill_group(self, layer, native_rows):
        if native_rows.numel() > self.prefill_chunk_rows:
            raise ValueError(
                "CPU prefill history chunk exceeds its GPU residency bound"
            )
        rows = self._global_ids(native_rows)
        current = torch.full_like(rows, -1)
        active = torch.full(
            (1,), rows.numel(), dtype=torch.int32, device=self.pool.device
        )
        self._prepare(layer, rows.view(-1, 1), current, active)

    @property
    def compact_prefill_chunk_rows(self):
        """Bound complete Plain unions using one absent-current workspace row."""
        capacity = self.workspaces[0].new_rows.numel()
        return min(self.gpu_slots, capacity - 1)

    def prepare_compact_prefill_group(self, layer, native_rows):
        if native_rows.numel() > self.compact_prefill_chunk_rows:
            raise ValueError(
                "CPU compact prefill union exceeds its GPU residency bound"
            )
        rows = self._global_ids(native_rows)
        current = torch.full((1,), -1, dtype=torch.int32, device=self.pool.device)
        active = torch.ones((1,), dtype=torch.int32, device=self.pool.device)
        self._prepare(layer, rows.view(1, -1), current, active)

    def finish_prefill_group(self, layer):
        # All peer VA reads must finish before another chunk can evict these
        # physical slots. This device barrier runs only at prefill chunks.
        self.pool.flowkv_arena.kernel().host_prefill_read_done(
            self.descriptors[layer],
            self.ready,
            self.pool.flowkv_rank,
        )
        # Existing prefill read-done barrier retires the epoch; no lease list.

    @property
    def metadata_bytes(self):
        tensors = [
            self.directory,
            self.access_time,
            self.ready,
            self.host.ptrs,
            self.miss_ready,
            self.fused_metadata,
            self.physical_rows,
        ]
        if self.deferred_epochs is not None:
            tensors.append(self.deferred_epochs)
        tensors += [
            self.directory_peers.ptrs,
            self.access_time_peers.ptrs,
            self.ready_peers.ptrs,
            self.pull_rows_peers.ptrs,
            self.pull_counts_peers.ptrs,
            self.miss_ready_peers.ptrs,
        ]
        tensors += list(self.descriptors)
        for state in self.states:
            tensors += [
                tensor
                for tensor in state
                if tensor is not state.directory and tensor is not state.access_time
            ]
        for workspace in self.workspaces:
            # Group88 belongs to the ring and is counted there, not twice.
            tensors += [
                tensor for tensor in workspace if tensor is not workspace.histogram
            ]
        return sum(
            tensor.nbytes for tensor in tensors if tensor.device.type == "cuda"
        ) + sum(ring.metadata_bytes for ring in self.rings)

    def get_host_storage_inventory(self):
        """Describe retained host backing without copies or kernels."""
        storage = self.host.storage.untyped_storage()
        return {int(storage.data_ptr()): int(storage.nbytes())}

    def get_memory_inventory(self):
        """Expose payload bytes separately from the full storage-union budget."""
        return {"cpu_kv_bytes": int(self.host.nbytes)}

    def capacity_audit(self):
        from ulacache.compiler import recency_policy

        return {
            "gpu_slots_per_rank": self.gpu_slots,
            "cpu_slots_per_rank": self.host_slots,
            "logical_slots_per_rank": self.logical_slots,
            "index_slots_per_rank": self.logical_slots,
            "groups": self.groups.group_count,
            "host_storage_bytes": self.host.nbytes,
            "gpu_metadata_bytes": self.metadata_bytes,
            "residency_policy": "index",
            "replacement_policy": recency_policy(),
            "total_residency_buckets": 0
            if recency_policy() == "naive"
            else self.rings[0].recent_count + 1,
            "ring_metadata_bytes": sum(ring.metadata_bytes for ring in self.rings),
            "index_identity_moves_with_kv": False,
            "device_graph_exchange_launch": list(self.exchange_launch),
            "parallel_metadata_finish": self.parallel_finish,
            "parallel_victim_selection": self.parallel_victims,
            "parallel_finish_min_queries": 8,
            "fused_prepare_finish_max_queries": 7 if recency_policy() != "naive" else 0,
            "fused_prepare_finish_completion": "last_producer_CTA_acq_rel",
            "deferred_gpu_bookkeeping_configured": self.deferred_epochs is not None,
            "deferred_route_storage_bytes": 0
            if self.deferred_epochs is None
            else self.deferred_epochs.nbytes,
            "deferred_bookkeeping_retirement": "ordinary_prefetch_join_before_step_publication",
        }

    def close(self):
        # Parent host-Graph completion includes every fire-and-forget child.
        # Synchronize before destroying any graph that holds pool pointers.
        torch.cuda.synchronize(self.pool.device)
        for executable in self.cold_execs:
            self._cold_module.destroy(self.pool.flowkv_shards, executable)
        self.directory_peers.close()
        self.ready_peers.close()
        self.access_time_peers.close()
        self.pull_rows_peers.close()
        self.pull_counts_peers.close()
        self.miss_ready_peers.close()
        self.host.close()


def create_host_tier_workspaces(capacity, groups, gpu_slots, device):
    """Share transfer plans while keeping parallel expiry scratch independent.

    The existing device Graph finish waits for exchange/commit publication.
    Decode groups and prefill chunks retain that same-stream ordering.
    Begin-forward can expire every group's bucket concurrently, so counters,
    prefixes and bounded expiry lists never alias another group's storage.
    """
    from ulacache.storage.capacity import shared_host_workspace_bytes
    from ulacache.cache.kernels.cpu_residency import HostTierWorkspace

    shared_host_workspace_bytes(capacity, groups, gpu_slots)
    options = {"dtype": torch.int32, "device": device}
    new_rows = torch.empty(capacity, **options)
    promote_rows = torch.empty(capacity, **options)
    plan = torch.empty((5, capacity), **options)
    candidates = torch.empty(capacity, **options)
    histogram = torch.empty(0, dtype=torch.int64, device=device)
    expiry_capacity = min(capacity, gpu_slots)
    return tuple(
        HostTierWorkspace(
            new_rows,
            promote_rows,
            plan,
            torch.zeros(8, **options),
            histogram,
            torch.zeros(3, dtype=torch.int64, device=device),
            candidates,
            torch.empty(expiry_capacity, **options),
        )
        for _ in range(groups)
    )


class SharedWorkspaceHostTier(FlowKVHostTier):
    """Keep serialized transfer plans shared and parallel expiry state private."""

    def __init__(
        self,
        pool,
        host_slots,
        cpu_group,
        *,
        host_pool_factory,
        ipc_manager_factory,
        exchange_launch=(4, 1024),
        parallel_finish=False,
        parallel_victims=False,
    ):
        super().__init__(
            pool,
            host_slots,
            cpu_group,
            host_pool_factory=host_pool_factory,
            ipc_manager_factory=ipc_manager_factory,
            exchange_launch=exchange_launch,
            parallel_finish=parallel_finish,
            parallel_victims=parallel_victims,
            workspace_factory=create_host_tier_workspaces,
        )

    @property
    def metadata_bytes(self):
        # The preserved native inventory counts each workspace tensor view.
        # Replace only that term with the actual shared backing allocations.
        tensors = [
            tensor
            for work in self.workspaces
            for tensor in work
            if tensor is not work.histogram
            and tensor.device.type == "cuda"
            and tensor.numel()
        ]
        allocations = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in tensors
        }
        duplicate_bytes = sum(tensor.nbytes for tensor in tensors) - sum(
            allocations.values()
        )
        return super().metadata_bytes - duplicate_bytes

    def capacity_audit(self):
        value = super().capacity_audit()
        value.update(
            transfer_workspace_shared_across_groups=True,
            transfer_workspace_capacity=self.workspaces[0].new_rows.numel(),
            expiry_workspace_capacity_per_group=self.workspaces[0].leased_slots.numel(),
        )
        return value
