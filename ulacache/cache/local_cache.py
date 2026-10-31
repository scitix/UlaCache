"""Rank-independent GPU/CPU KV residency with direct attention reads.

There is one GPU record bank, no query-local KV cache, and no peer transport.
Native Index identities remain logical. Each shared-Index group reuses the
existing exclusive-residency planner and one transfer plan for all its layers.
"""

from __future__ import annotations


class UnifiedLocalResidency:
    def __init__(
        self,
        *,
        groups,
        gpu_slots,
        host_slots,
        row_bytes,
        max_queries,
        topk,
        max_write_rows,
        device,
        host_pool_factory,
        prefill_union_rows=0,
        parallel_expiry=True,
        parallel_prepare=False,
        async_followers=False,
        parallel_current=False,
        parallel_victims=False,
        exchange_launch=(4, 1024),
        follower_exchange_launch=None,
        follower_copy_mode="group",
        prefill_exchange_launch=None,
        prefill_union_mode="sort",
    ):
        import torch
        from ulacache.cache.kernels.cpu_residency import (
            create_host_tier_state,
            validate_exchange_launch,
        )
        from ulacache.eviction.lru import create_presence_ring
        from ulacache.cache.host import create_host_tier_workspaces

        if min(gpu_slots, host_slots, row_bytes, max_queries, topk, max_write_rows) < 1:
            raise ValueError("Unified local residency requires positive capacities")
        if row_bytes % 8 or gpu_slots % 64:
            raise ValueError(
                "Native paged attention needs aligned records and 64-row GPU pages"
            )
        self.groups, self.row_bytes = groups, row_bytes
        self.parallel_expiry = parallel_expiry
        self.parallel_prepare = parallel_prepare
        if parallel_current and not parallel_prepare:
            raise ValueError("Parallel current preparation requires parallel residency")
        self.parallel_current = parallel_current
        self._forward_current_records = None
        if parallel_victims and not parallel_prepare:
            raise ValueError("Parallel victim selection requires parallel preparation")
        self.parallel_victims = parallel_victims
        self.exchange_launch = validate_exchange_launch(exchange_launch)
        if prefill_union_mode not in ("sort", "device"):
            raise ValueError("Prefill union mode must be sort or device")
        self.prefill_union_mode = prefill_union_mode
        self.prefill_exchange_launch = validate_exchange_launch(
            exchange_launch
            if prefill_exchange_launch is None
            else prefill_exchange_launch
        )
        self.follower_exchange_launch = validate_exchange_launch(
            exchange_launch
            if follower_exchange_launch is None
            else follower_exchange_launch
        )
        self.async_followers = async_followers
        if follower_copy_mode not in ("group", "layer"):
            raise ValueError("Follower copy mode must be group or layer")
        if follower_copy_mode == "layer" and not async_followers:
            raise ValueError("Layer follower copies require asynchronous followers")
        self.follower_copy_mode = follower_copy_mode
        self._pending_groups, self._joined_groups = set(), set()
        self._joined_layers = set()
        if async_followers:
            self._transfer_stream = torch.cuda.Stream(device=device)
            self._transfer_start = tuple(
                torch.cuda.Event() if f else None for f in groups.followers
            )
            if follower_copy_mode == "layer":
                self._transfer_layer_done = {
                    layer: torch.cuda.Event()
                    for followers in groups.followers
                    for layer in followers
                }
                self._transfer_done = tuple(
                    self._transfer_layer_done[f[-1]] if f else None
                    for f in groups.followers
                )
            else:
                self._transfer_done = tuple(
                    torch.cuda.Event() if f else None for f in groups.followers
                )
        self.gpu_slots, self.host_slots = gpu_slots, host_slots
        self.logical_slots = gpu_slots + host_slots
        self.max_queries, self.topk = max_queries, topk
        self.gpu = torch.zeros(
            (groups.layer_count, gpu_slots, row_bytes), dtype=torch.uint8, device=device
        )
        self.host = host_pool_factory(
            layers=groups.layer_count,
            slots=host_slots,
            row_bytes=row_bytes,
            device=device,
        )
        self.directory = torch.empty(
            (groups.group_count, self.logical_slots), dtype=torch.int64, device=device
        )
        self.access_time = torch.zeros_like(self.directory)
        self.states = tuple(
            create_host_tier_state(
                gpu_slots,
                host_slots,
                device,
                directory=self.directory[g],
                access_time=self.access_time[g],
            )
            for g in range(groups.group_count)
        )
        capacity = max(max_queries * (topk + 1), max_write_rows, prefill_union_rows + 1)
        workspaces = create_host_tier_workspaces(
            capacity, groups.group_count, gpu_slots, device
        )
        self.current_preparation_scratch_bytes = 0
        if parallel_current:
            # Current pinning runs across all groups before their selections.
            # Its new/promote entries must survive later groups' current pinning
            # and earlier groups' selected-row appends. Match the native-paged
            # ownership rule; exchange plans and candidates stay serialized.
            workspaces = tuple(
                work
                if group == 0
                else work._replace(
                    new_rows=torch.empty_like(work.new_rows),
                    promote_rows=torch.empty_like(work.promote_rows),
                )
                for group, work in enumerate(workspaces)
            )
            self.current_preparation_scratch_bytes = (
                8 * capacity * (groups.group_count - 1)
            )
        self.rings = tuple(
            create_presence_ring(state, work)
            for state, work in zip(self.states, workspaces)
        )
        self.workspaces = tuple(
            w._replace(histogram=r.view) for w, r in zip(workspaces, self.rings)
        )
        self.remap = torch.empty(
            (groups.group_count, max_queries, topk), dtype=torch.int32, device=device
        )
        self.active = torch.zeros(1, dtype=torch.int32, device=device)
        self.prefill_active = torch.ones_like(self.active)
        self.prefill_current = torch.zeros(1, dtype=torch.int32, device=device)
        # Optional device counters: selected IDs, initial GPU hits, CPU misses.
        # Updating these counters does not copy values to the host per step.
        self.stats = torch.zeros(
            (groups.group_count, 3), dtype=torch.int64, device=device
        )
        self.collect_stats = False
        # Preserve the native registration frame ABI. The peer/child fields are
        # unused by rank-local registration, and never form an IPC namespace.
        # The shared TP deferred-route pointer is explicitly null for DPA.
        self.frames = torch.tensor(
            [
                [t.data_ptr() for t in (*state, *work)]
                + [
                    0,
                    0,
                    0,
                    g,
                    0,
                    gpu_slots,
                    host_slots,
                    capacity,
                    0,
                    1,
                    max_queries,
                    topk,
                    0,
                    0,
                ]
                for g, (state, work) in enumerate(zip(self.states, self.workspaces))
            ],
            dtype=torch.int64,
            device=device,
        )
        if prefill_union_mode == "device":
            union_capacity = min(gpu_slots, capacity - 1)
            self._prefill_marks = torch.zeros(
                self.logical_slots, dtype=torch.int32, device=device
            )
            self._prefill_unique = torch.empty(
                union_capacity, dtype=torch.int32, device=device
            )
            self._prefill_unique_count = torch.zeros(
                1, dtype=torch.int32, device=device
            )
        self._planned = set()

    def allocate(self, rows):
        from ulacache.cache.kernels.cpu_residency import register_native_host_tier_groups

        register_native_host_tier_groups(rows, self.frames, 0)

    def allocate_unbound(self, rows, *, prefer_gpu=True):
        """Bind unique native outputs, including the unwritten tail of a page."""
        from ulacache.cache.kernels.local_cache import register_unbound

        register_unbound(rows, self.frames, prefer_gpu=prefer_gpu)

    def allocate_receive_rows(self, rows):
        """Bind new unpublished native rows in ordered canonical extents."""
        from ulacache.cache.kernels.local_cache import (
            register_receive_rows,
            order_free_cpu_receive_tail,
        )

        self._drain_transfers()
        order_free_cpu_receive_tail(self.frames, rows.numel())
        register_receive_rows(rows, self.frames, prefer_gpu=False)

    def release(self, rows):
        from ulacache.cache.kernels.cpu_residency import release_host_tier_rows

        self._drain_transfers()
        for state, ring in zip(self.states, self.rings):
            release_host_tier_rows(rows, 0, state, ring=ring)

    def clear(self):
        from ulacache.cache.kernels.cpu_residency import (
            clear_host_tier_ring,
            clear_host_tier_state,
        )

        self._drain_transfers()
        for state, ring in zip(self.states, self.rings):
            clear_host_tier_ring(ring)
            clear_host_tier_state(state)
        self._planned.clear()
        self._forward_current_records = None
        self.stats.zero_()

    def begin_forward(self):
        self._drain_transfers()
        self._pending_groups.clear()
        self._joined_groups.clear()
        self._joined_layers.clear()
        self._planned.clear()
        self._forward_current_records = None
        if self.parallel_expiry:
            from ulacache.cache.kernels.local_cache import expire_recent_groups

            # Expiry scratch is group-private. Original per-group prepare
            # retains epoch increments, pinning, and the shared transfer plan.
            expire_recent_groups(self.frames)

    def prepare_forward_current_records(self, current):
        """Advance independent groups using their original current transaction.

        Every selection producer must supply this same tensor. Current payloads
        retain their original per-layer writer and exchange ordering; this call
        changes metadata only and performs no record copy or mapping commit.
        """
        if not self.parallel_current:
            return False
        if self._planned or self._forward_current_records is not None:
            raise RuntimeError("Bind current records once before any group selection")
        import torch

        if (
            current.ndim != 1
            or current.dtype not in (torch.int32, torch.int64)
            or current.device != self.gpu.device
            or not current.is_contiguous()
            or current.numel() > self.max_queries
        ):
            raise ValueError(
                "Bind contiguous current IDs within the startup query bound"
            )
        self._drain_transfers()
        from ulacache.cache.kernels.batched_prepare import begin_native_domains

        begin_native_domains(
            current, self.active, self.frames, parallel_victims=self.parallel_victims
        )
        self._forward_current_records = current
        return True

    def finish_forward(self):
        """Order a native prefix publication after outstanding follower writes.

        This adds only existing event dependencies to the current stream. It
        neither synchronizes the host nor copies a second prefix payload.
        """
        self._drain_transfers()

    def plan(self, layer, selected, current):
        from ulacache.cache.kernels.cpu_residency import (
            prepare_host_tier,
            exchange_host_tier,
            commit_host_tier,
        )
        from ulacache.cache.kernels.local_cache import count_residency, remap_resident_rows

        self._drain_transfers()
        group = self.groups.group(layer)
        if self.groups.anchors[group] != layer or group in self._planned:
            raise RuntimeError("Plan exactly once at each native shared-Index anchor")
        if selected.shape[1] != self.topk or selected.shape[0] > self.max_queries:
            raise ValueError(
                "Selected native Index exceeds the rank-local startup bound"
            )
        state, work = self.states[group], self.workspaces[group]
        common = self._forward_current_records
        prepared = common is not None
        if prepared and (
            current.data_ptr() != common.data_ptr()
            or current.shape != common.shape
            or current.dtype != common.dtype
        ):
            raise ValueError(
                "Every selection must use the explicitly bound current-ID tensor"
            )
        if self.collect_stats:
            count_residency(selected, self.active, state.directory, self.stats[group])
        prepare_host_tier(
            selected,
            current,
            self.active,
            0,
            state,
            work,
            parallel=self.parallel_prepare,
            parallel_victims=self.parallel_victims,
            current_already_prepared=prepared,
        )
        anchor = self.groups.anchors[group]
        end = (
            self.groups.followers[group][-1] + 1
            if self.groups.followers[group]
            else anchor + 1
        )
        copy_end = anchor + 1 if self.async_followers else end
        exchange_host_tier(
            self.host.storage[anchor:copy_end],
            self.gpu[anchor:copy_end],
            work,
            self.row_bytes,
            launch=self.exchange_launch,
        )
        commit_host_tier(state, work)
        target = self.remap[group, : selected.shape[0]]
        remap_resident_rows(selected, self.active, state.directory, target)
        self._planned.add(group)
        if self.async_followers and self.groups.followers[group]:
            import torch

            stream = torch.cuda.current_stream(self.gpu.device)
            self._transfer_start[group].record(stream)
            self._transfer_stream.wait_event(self._transfer_start[group])
            with torch.cuda.stream(self._transfer_stream):
                if self.follower_copy_mode == "layer":
                    for follower in self.groups.followers[group]:
                        exchange_host_tier(
                            self.host.storage[follower : follower + 1],
                            self.gpu[follower : follower + 1],
                            work,
                            self.row_bytes,
                            launch=self.follower_exchange_launch,
                        )
                        self._transfer_layer_done[follower].record(
                            self._transfer_stream
                        )
                else:
                    exchange_host_tier(
                        self.host.storage[anchor + 1 : end],
                        self.gpu[anchor + 1 : end],
                        work,
                        self.row_bytes,
                        launch=self.follower_exchange_launch,
                    )
                    self._transfer_done[group].record(self._transfer_stream)
            self._pending_groups.add(group)
        return target

    def before_layer_prepare(self, layer):
        """Finish follower victim copies before its Q preparation or KV writer."""
        group = self.groups.group(layer)
        if (
            group in self._pending_groups
            and group not in self._joined_groups
            and layer != self.groups.anchors[group]
        ):
            import torch

            stream = torch.cuda.current_stream(self.gpu.device)
            if self.follower_copy_mode == "layer":
                if layer not in self._joined_layers:
                    stream.wait_event(self._transfer_layer_done[layer])
                    self._joined_layers.add(layer)
                if layer == self.groups.followers[group][-1]:
                    self._joined_groups.add(group)
            else:
                stream.wait_event(self._transfer_done[group])
                self._joined_groups.add(group)

    def _drain_transfers(self):
        # A group's final event covers every follower, including partial-layer
        # joins and early exit. Retire all copies before reusing its plan.
        if self._pending_groups - self._joined_groups:
            import torch

            stream = torch.cuda.current_stream(self.gpu.device)
            for group in self._pending_groups - self._joined_groups:
                stream.wait_event(self._transfer_done[group])
                self._joined_groups.add(group)

    def attention_rows(self, layer, query_count):
        group = self.groups.group(layer)
        if group not in self._planned:
            raise RuntimeError(
                "Native Index must prepare direct GPU reads before attention"
            )
        self.before_layer_prepare(layer)
        return self.remap[group, :query_count]

    def plan_prefill(self, layer, selected):
        """Eager native TopK union, using the same GPU bank without KV scratch.

        New rows may be promoted before their native writer runs. Their old
        bytes are unused and are overwritten by that writer in every member.
        """
        import torch
        from ulacache.cache.kernels.cpu_residency import (
            prepare_host_tier,
            exchange_host_tier,
            commit_host_tier,
        )
        from ulacache.cache.kernels.local_cache import remap_resident_rows

        if self._forward_current_records is not None:
            raise RuntimeError("Prefill cannot reuse a decode current transaction")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Rank-local prefill union requires eager execution")
        group = self.groups.group(layer)
        state, work = self.states[group], self.workspaces[group]
        if group not in self._planned:
            if layer != self.groups.anchors[group]:
                raise RuntimeError("Prefill shared Index must start at its anchor")
            counts = None
            if self.prefill_union_mode == "device":
                from ulacache.cache.kernels.prefix_selection import compact_prefill_union

                needed, counts = compact_prefill_union(
                    selected,
                    self._prefill_marks,
                    self._prefill_unique,
                    self._prefill_unique_count,
                )
            else:
                needed = torch.unique(
                    selected[selected > 0].int(), sorted=True
                ).reshape(1, -1)
            if (
                needed.numel() + 1 > work.new_rows.numel()
                or needed.numel() > self.gpu_slots
            ):
                raise ValueError(
                    "Complete prefill TopK union exceeds its startup residency bound"
                )
            device_union = self.prefill_union_mode == "device"
            prepare_host_tier(
                needed,
                self.prefill_current,
                self.prefill_active,
                0,
                state,
                work,
                selected_counts=counts,
                parallel=device_union and self.parallel_prepare,
                parallel_victims=device_union and self.parallel_victims,
            )
            end = (
                self.groups.followers[group][-1] + 1
                if self.groups.followers[group]
                else layer + 1
            )
            exchange_host_tier(
                self.host.storage[layer:end],
                self.gpu[layer:end],
                work,
                self.row_bytes,
                launch=self.prefill_exchange_launch,
            )
            commit_host_tier(state, work)
            self._planned.add(group)
        output = torch.empty_like(selected, dtype=torch.int32)
        if self.prefill_union_mode == "device":
            from ulacache.cache.kernels.local_cache import remap_prefill_rows

            remap_prefill_rows(selected, state.directory, output)
        else:
            active = torch.tensor(
                [selected.shape[0]], dtype=torch.int32, device=selected.device
            )
            remap_resident_rows(selected, active, state.directory, output)
        return output

    def after_attention(self, layer):
        from ulacache.cache.kernels.cpu_residency import release_host_tier_leases

        group = self.groups.group(layer)
        self.before_layer_prepare(layer)
        last = (
            self.groups.followers[group][-1]
            if self.groups.followers[group]
            else self.groups.anchors[group]
        )
        if layer == last:
            release_host_tier_leases(self.states[group], self.workspaces[group])

    def write(self, layer, rows, nope, rope):
        from ulacache.cache.kernels.local_cache import write_native_records

        self.before_layer_prepare(layer)
        group = self.groups.group(layer)
        write_native_records(
            rows,
            nope,
            rope,
            self.directory[group],
            self.gpu[layer],
            self.host.ptrs,
            layer,
            self.host_slots,
            self.row_bytes,
        )

    def write_native_paged_records(self, *_native_arguments):
        raise NotImplementedError(
            "Packed page sources require the native_paged storage backend"
        )

    def prepare_prefill_record_chunk(self, *_native_arguments):
        raise NotImplementedError(
            "Bounded reconstruction chunks require the native_paged storage backend"
        )

    def finish_prefill_record_chunk(self, *_native_arguments):
        raise NotImplementedError(
            "Bounded reconstruction chunks require the native_paged storage backend"
        )

    def offload_prefill(self, group, rows):
        """Keep new prefill KV in CPU slots outside its attention lifetime."""
        from ulacache.cache.kernels.local_cache import offload_prefill_rows

        self._drain_transfers()
        layers = (self.groups.anchors[group], *self.groups.followers[group])
        if layers != tuple(range(layers[0], layers[-1] + 1)):
            raise ValueError(
                "Prefill offload requires a contiguous native shared-Index group"
            )
        offload_prefill_rows(
            rows,
            self.frames[group : group + 1],
            self.gpu,
            self.host.ptrs,
            layers[0],
            len(layers),
        )

    def device_storage_inventory(self):
        """Backing addresses for an adapter's exact native allocation audit."""
        tensors = [
            self.gpu,
            self.directory,
            self.access_time,
            self.remap,
            self.active,
            self.prefill_active,
            self.prefill_current,
            self.stats,
            self.frames,
            self.host.ptrs,
        ]
        tensors += [t for s in self.states for t in s]
        if self.prefill_union_mode == "device":
            tensors += [
                self._prefill_marks,
                self._prefill_unique,
                self._prefill_unique_count,
            ]
        tensors += [t for w in self.workspaces for t in w]
        for ring in self.rings:
            tensors += [
                ring.bits,
                ring.summary,
                ring.bucket_epochs,
                ring.gpu_bucket,
                ring.bucket_views,
                ring.view,
            ]
            if ring.naive_histogram is not None:
                tensors.append(ring.naive_histogram)
        return {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for t in tensors
            if t.device.type == "cuda" and t.numel()
        }

    def host_storage_inventory(self):
        storage = self.host.storage.untyped_storage()
        return {storage.data_ptr(): storage.nbytes()} if storage.nbytes() else {}

    def memory_inventory(self):
        """Unique GPU storage bytes, including reused plan backing allocations."""
        storage = self.device_storage_inventory()
        return dict(
            gpu_bytes=sum(storage.values()),
            gpu_kv_bytes=self.gpu.nbytes,
            gpu_metadata_bytes=sum(storage.values()) - self.gpu.nbytes,
            cpu_kv_bytes=self.host.nbytes,
            gpu_slots=self.gpu_slots,
            cpu_slots=self.host_slots,
            logical_slots=self.logical_slots,
            local_buffer_bytes=0,
            cache_peer_count=1,
            parallel_current_scratch_bytes=self.current_preparation_scratch_bytes,
        )

    def close(self):
        """Drain readers before the native allocator unregisters host storage."""
        if getattr(self, "_closed", False):
            return
        import torch

        self._drain_transfers()
        torch.cuda.synchronize(self.gpu.device)
        self.host.close()
        self._closed = True
