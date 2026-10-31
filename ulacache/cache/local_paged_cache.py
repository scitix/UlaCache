"""Exclusive residency for native pages with independent selection producers.

Physical KV producers retain separate payload banks. One residency directory
coordinates banks that share a placement plan. A later selection replaces that
directory's leases after its previous readers retire. Model math stays native.
"""


class NativePagedResidency:
    def __init__(self, cache_specification, cache_bindings, execution_plan):
        import torch
        from ulacache.cache.kernels.cpu_residency import create_host_tier_state
        from ulacache.eviction.lru import create_presence_ring
        from ulacache.cache.host import create_host_tier_workspaces
        from ulacache.storage.buffers import RecordStorage

        options = cache_bindings.execution_options
        if options.copy_follower_records_asynchronously:
            if len(cache_specification.storage_layout.record_planes) != 1:
                raise NotImplementedError(
                    "Gate native multi-plane follower IO separately"
                )
            if options.follower_copy_mode != "layer":
                raise ValueError(
                    "This opt-in native-paged candidate requires per-layer follower events"
                )
            follower_launch = options.follower_exchange_launch_dimensions
            if (
                follower_launch is not None
                and follower_launch != options.exchange_launch_dimensions
            ):
                raise NotImplementedError(
                    "Bind equal anchor/follower launch geometry in this candidate"
                )
        if (
            options.select_victims_in_parallel
            and not options.prepare_residency_in_parallel
        ):
            raise ValueError("Parallel victim selection requires parallel preparation")
        if options.prefill_union_method not in ("sort", "device"):
            raise ValueError("Prefill union method must be sort or device")
        self.execution_plan = execution_plan
        self.cache_specification = cache_specification
        self.execution_options = options
        self.device = cache_bindings.device
        capacity = cache_specification.capacity_plan
        access = cache_specification.sparse_access
        self.maximum_query_count = access.maximum_padded_query_count
        self.maximum_selection_width = access.maximum_selected_record_count
        self.device_record_capacity = capacity.device_record_capacity
        domain_count = len(execution_plan.residency_producer_layers)
        workspace_capacity = max(
            self.maximum_query_count * (self.maximum_selection_width + 1),
            capacity.maximum_write_record_count,
            capacity.maximum_prefill_union_record_count + 1,
            capacity.maximum_residency_input_record_count + 1,
        )
        self.residency_states = tuple(
            create_host_tier_state(
                capacity.device_record_capacity,
                capacity.host_record_capacity,
                self.device,
            )
            for _ in range(domain_count)
        )
        workspaces = create_host_tier_workspaces(
            workspace_capacity,
            domain_count,
            capacity.device_record_capacity,
            self.device,
        )
        self.current_preparation_scratch_bytes = 0
        if options.prepare_current_records_in_parallel:
            # begin_current appends CPU-current promotions concurrently. Only
            # new/promote lists need private full-union capacity; transfer plans
            # and victim candidates remain serialized at each selection.
            workspaces = tuple(
                workspace
                if domain == 0
                else workspace._replace(
                    new_rows=torch.empty_like(workspace.new_rows),
                    promote_rows=torch.empty_like(workspace.promote_rows),
                )
                for domain, workspace in enumerate(workspaces)
            )
            self.current_preparation_scratch_bytes = (
                8 * workspace_capacity * (domain_count - 1)
            )
        self.presence_rings = tuple(
            create_presence_ring(state, workspace)
            for state, workspace in zip(self.residency_states, workspaces)
        )
        self.transfer_workspaces = tuple(
            workspace._replace(histogram=ring.view)
            for workspace, ring in zip(workspaces, self.presence_rings)
        )
        self.remapped_selections = torch.empty(
            (
                len(execution_plan.selection_producer_layers),
                self.maximum_query_count,
                self.maximum_selection_width,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.active_query_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.prefill_active_query_count = torch.ones_like(self.active_query_count)
        self.prefill_current_record = torch.zeros_like(self.active_query_count)
        # Native registration/batched-current retains the shared frame ABI;
        # its final TP-only deferred-route pointer is explicitly null.
        self.registration_frames = torch.tensor(
            [
                [tensor.data_ptr() for tensor in (*state, *workspace)]
                + [
                    0,
                    0,
                    0,
                    domain,
                    0,
                    capacity.device_record_capacity,
                    capacity.host_record_capacity,
                    workspace_capacity,
                    0,
                    1,
                    self.maximum_query_count,
                    self.maximum_selection_width,
                    0,
                    0,
                ]
                for domain, (state, workspace) in enumerate(
                    zip(self.residency_states, self.transfer_workspaces)
                )
            ],
            dtype=torch.int64,
            device=self.device,
        )
        if options.prefill_union_method == "device":
            union_capacity = min(
                capacity.device_record_capacity, workspace_capacity - 1
            )
            self.prefill_union_marks = torch.zeros(
                capacity.logical_record_capacity, dtype=torch.int32, device=self.device
            )
            self.prefill_unique_records = torch.empty(
                union_capacity, dtype=torch.int32, device=self.device
            )
            self.prefill_unique_record_count = torch.zeros_like(self.active_query_count)
        self._planned_selections = set()
        self._forward_current_records = None
        self._live_residency_domains = set()
        # Allocate all metadata before acquiring the native pinned host resource.
        self.record_storage = RecordStorage(
            cache_specification, cache_bindings, execution_plan
        )
        self.device_record_planes_by_layer = tuple(
            self.record_storage.get_device_record_planes(layer)
            for layer in range(cache_specification.layer_reuse.layer_count)
        )
        self.device_record_banks_by_layer = tuple(
            planes[0] for planes in self.device_record_planes_by_layer
        )
        self._pending_transfers = set()
        self._joined_banks = set()
        self._async_banks = {}
        self._anchor_banks = {}
        self._transfer_start = {}
        self._transfer_done = {}
        self._bank_ready = {}
        if options.copy_follower_records_asynchronously:
            # This first native-paged async path is deliberately bound to
            # HY4's independent KV writer at every layer. Reindex-only
            # producers, aliased KV and interleaved consumer intervals
            # require separate native writer/reader race gates.
            layer_count = cache_specification.layer_reuse.layer_count
            if execution_plan.storage_producer_layers != tuple(range(layer_count)):
                raise NotImplementedError("Gate native async KV aliases separately")
            self._transfer_stream = torch.cuda.Stream(device=self.device)
            for domain in range(domain_count):
                producers = [
                    layer
                    for layer in execution_plan.selection_producer_layers
                    if execution_plan.residency_domain_indices_by_layer[layer] == domain
                ]
                if len(producers) != 1:
                    raise NotImplementedError(
                        "Async candidate requires one native selection per residency domain"
                    )
                anchor = producers[0]
                if anchor not in execution_plan.storage_producer_layers:
                    raise NotImplementedError(
                        "Async selection anchor must own its native KV writer"
                    )
                banks = tuple(
                    bank
                    for start, stop in execution_plan.storage_bank_ranges_by_residency_domain[
                        domain
                    ]
                    for bank in range(start, stop)
                )
                anchor_bank = execution_plan.storage_bank_indices_by_layer[anchor]
                if not banks or anchor_bank != banks[0]:
                    raise NotImplementedError(
                        "Async selection must precede every follower writer"
                    )
                selection = execution_plan.selection_indices_by_layer[anchor]
                last = execution_plan.last_consumer_layers_by_selection[selection]
                if any(
                    execution_plan.storage_producer_layers[bank] > last
                    for bank in banks
                ):
                    raise NotImplementedError(
                        "Async follower lifetime exceeds its native selection"
                    )
                self._anchor_banks[domain] = anchor_bank
                followers = tuple(bank for bank in banks if bank != anchor_bank)
                self._async_banks[domain] = followers
                if followers:
                    self._transfer_start[domain] = torch.cuda.Event()
                    for bank in followers:
                        self._bank_ready[bank] = torch.cuda.Event()
                    self._transfer_done[domain] = self._bank_ready[followers[-1]]
        self._closed = False

    @property
    def gpu(self):
        """Borrowed per-layer views; aliases retain the same physical storage."""
        if len(self.cache_specification.storage_layout.record_planes) > 1:
            raise ValueError(
                "Use get_device_record_planes for a cache with separate native planes"
            )
        return self.device_record_banks_by_layer

    @property
    def active(self):
        return self.active_query_count

    def allocate_unbound(self, logical_record_ids, *, prefer_gpu=True):
        from ulacache.cache.kernels.local_cache import register_unbound

        register_unbound(
            logical_record_ids, self.registration_frames, prefer_gpu=prefer_gpu
        )

    def allocate_receive_rows(self, logical_record_ids):
        """Bind unpublished destinations using the common receive allocator."""
        from ulacache.cache.kernels.local_cache import (
            order_free_cpu_receive_tail,
            register_receive_rows,
        )

        self._require_retired_readers()
        order_free_cpu_receive_tail(
            self.registration_frames, logical_record_ids.numel()
        )
        register_receive_rows(
            logical_record_ids, self.registration_frames, prefer_gpu=False
        )

    def release(self, logical_record_ids):
        from ulacache.cache.kernels.cpu_residency import release_host_tier_rows

        self._require_retired_readers()
        for state, ring in zip(self.residency_states, self.presence_rings):
            release_host_tier_rows(logical_record_ids, 0, state, ring=ring)

    def clear(self):
        from ulacache.cache.kernels.cpu_residency import (
            clear_host_tier_ring,
            clear_host_tier_state,
        )

        self._require_retired_readers()
        for state, ring in zip(self.residency_states, self.presence_rings):
            clear_host_tier_ring(ring)
            clear_host_tier_state(state)
        self._planned_selections.clear()

    def _drain_transfers(self):
        if self._pending_transfers:
            import torch

            stream = torch.cuda.current_stream(self.device)
            for domain in self._pending_transfers:
                stream.wait_event(self._transfer_done[domain])
            self._pending_transfers.clear()

    def _require_retired_readers(self):
        self._drain_transfers()
        if self._live_residency_domains:
            raise RuntimeError(
                "Finish every record reader before changing cache lifetime"
            )

    def begin_forward(self):
        self._require_retired_readers()
        self._planned_selections.clear()
        self._forward_current_records = None
        # Native prepare advances and expires the domain epoch at every selection,
        # including successive reindex layers within this same model forward.

    def prepare_forward_current_records(self, current_logical_record_ids):
        """Plan original current transactions once over independent domains.

        The native binding explicitly promises this same tensor for every KV
        producer. Reused residency domains and writer-free selection anchors
        retain the ordinary transaction. Check the promise at every selection.
        """
        import torch

        self._require_retired_readers()
        plan = self.execution_plan
        domains = [
            plan.residency_domain_indices_by_layer[layer]
            for layer in plan.selection_producer_layers
        ]
        if (
            not self.execution_options.prepare_current_records_in_parallel
            or not self.execution_options.prepare_residency_in_parallel
            or sorted(domains) != list(range(len(self.residency_states)))
            or not set(plan.selection_producer_layers)
            <= set(plan.storage_producer_layers)
        ):
            return False
        if (
            current_logical_record_ids.ndim != 1
            or current_logical_record_ids.dtype not in (torch.int32, torch.int64)
            or not current_logical_record_ids.is_contiguous()
            or current_logical_record_ids.numel() > self.maximum_query_count
        ):
            raise ValueError(
                "Bind contiguous native current IDs within the startup query bound"
            )
        from ulacache.cache.kernels.batched_prepare import begin_native_domains

        begin_native_domains(
            current_logical_record_ids,
            self.active_query_count,
            self.registration_frames,
            parallel_victims=self.execution_options.select_victims_in_parallel,
        )
        self._forward_current_records = current_logical_record_ids
        return True

    def prepare_next_residency_epochs(self):
        """Use one existing CTA per domain to retire aged bucket headers.

        This is the original contiguous backend's expiry operation, with the
        same State9/Work8 frame ABI. It does not advance canonical time, pin
        current rows, alter victim selection or publish payloads. All readers
        from the preceding forward must already be retired on this stream.
        """
        self._require_retired_readers()
        if self.execution_options.expire_residency_in_parallel:
            from ulacache.cache.kernels.local_cache import expire_recent_groups

            expire_recent_groups(self.registration_frames)

    def before_layer_prepare(self, layer_index):
        """Join a follower bank before either its native write or attention."""
        domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
        bank = self.execution_plan.storage_bank_indices_by_layer[layer_index]
        if (
            domain in self._pending_transfers
            and bank in self._bank_ready
            and bank not in self._joined_banks
        ):
            import torch

            torch.cuda.current_stream(self.device).wait_event(self._bank_ready[bank])
            self._joined_banks.add(bank)

    def _start_selection(self, layer_index):
        # Residency domains alias the transfer-plan buffer. Finish every
        # outstanding reader before any next selection overwrites that plan.
        self._drain_transfers()
        self._joined_banks.clear()
        selection = self.execution_plan.selection_indices_by_layer[layer_index]
        domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
        if (
            self.execution_plan.selection_producer_layers[selection] != layer_index
            or selection in self._planned_selections
        ):
            raise RuntimeError("Plan exactly once at each native selection producer")
        if domain in self._live_residency_domains:
            raise RuntimeError(
                "The previous selection on this payload domain still has readers"
            )
        return selection, domain

    def _exchange_and_commit(self, domain, *, selection_layer=None):
        from ulacache.cache.kernels.cpu_residency import commit_host_tier

        followers = (
            self._async_banks.get(domain, ()) if selection_layer is not None else ()
        )
        workspace = self.transfer_workspaces[domain]
        if followers:
            self.record_storage.exchange_record_bank(
                self._anchor_banks[domain], workspace
            )
        else:
            self.record_storage.exchange_residency_plan(domain, workspace)
        commit_host_tier(self.residency_states[domain], workspace)
        self._live_residency_domains.add(domain)
        if followers:
            import torch

            stream = torch.cuda.current_stream(self.device)
            self._transfer_start[domain].record(stream)
            self._transfer_stream.wait_event(self._transfer_start[domain])
            with torch.cuda.stream(self._transfer_stream):
                for bank in followers:
                    self.record_storage.exchange_record_bank(bank, workspace)
                    self._bank_ready[bank].record(self._transfer_stream)
            self._pending_transfers.add(domain)

    def plan(self, layer_index, selected_logical_records, current_write_records):
        """A reindex-only producer supplies negative current-write sentinels.

        Current records skip fetching their old bytes and must be freshly
        written after this call, before their first attention consumer.
        """
        from ulacache.cache.kernels.cpu_residency import prepare_host_tier
        from ulacache.cache.kernels.local_cache import remap_resident_rows

        selection, domain = self._start_selection(layer_index)
        if (
            selected_logical_records.shape[1] != self.maximum_selection_width
            or selected_logical_records.shape[0] > self.maximum_query_count
        ):
            raise ValueError(
                "Native selection exceeds its startup query or width bound"
            )
        state, workspace = (
            self.residency_states[domain],
            self.transfer_workspaces[domain],
        )
        prepared = self._forward_current_records is not None
        if prepared:
            common = self._forward_current_records
            if (
                current_write_records.data_ptr() != common.data_ptr()
                or current_write_records.shape != common.shape
                or current_write_records.dtype != common.dtype
            ):
                raise ValueError(
                    "Every selection must use the explicitly bound native current-ID tensor"
                )
        prepare_host_tier(
            selected_logical_records,
            current_write_records,
            self.active_query_count,
            0,
            state,
            workspace,
            parallel=self.execution_options.prepare_residency_in_parallel,
            parallel_victims=self.execution_options.select_victims_in_parallel,
            current_already_prepared=prepared,
        )
        self._exchange_and_commit(domain, selection_layer=layer_index)
        self._planned_selections.add(selection)
        output = self.remapped_selections[
            selection, : selected_logical_records.shape[0]
        ]
        remap_resident_rows(
            selected_logical_records, self.active_query_count, state.directory, output
        )
        return output

    def attention_rows(self, layer_index, query_count):
        self.before_layer_prepare(layer_index)
        selection = self.execution_plan.selection_indices_by_layer[layer_index]
        if selection not in self._planned_selections:
            raise RuntimeError(
                "Prepare the native selection before reading its resident records"
            )
        return self.remapped_selections[selection, :query_count]

    def prepare_record_residency(
        self,
        residency_domain_index,
        owner_local_record_ids,
        *,
        current_already_prepared=False,
    ):
        """Lease a caller-supplied record union using the existing native planner.

        Inputs can repeat and contain negative padding. Every positive ID is
        already registered in this owner's local directory. Native preparation
        deduplicates them by its existing access epoch, avoiding a logical-pool
        mark-array reset. The startup input bound includes all padded consumers,
        separately from the number of unique records that can fit on the GPU.
        Callers serialize remote reader retirement before replacing a domain.
        This method neither gathers peers nor publishes distributed completion.
        The native planner sees a single local owner (index zero); the caller
        has already resolved the actual distributed storage owner.
        """
        from ulacache.cache.kernels.cpu_residency import prepare_host_tier

        self._drain_transfers()
        domain = residency_domain_index
        if domain in self._live_residency_domains:
            raise RuntimeError(
                "Finish the current record union before replacing its residency"
            )
        if current_already_prepared and self._forward_current_records is None:
            raise ValueError(
                "Bind native current records before reusing their domain preparation"
            )
        prepare_host_tier(
            owner_local_record_ids.reshape(1, -1),
            self.prefill_current_record,
            self.prefill_active_query_count,
            0,
            self.residency_states[domain],
            self.transfer_workspaces[domain],
            parallel=self.execution_options.prepare_residency_in_parallel,
            parallel_victims=self.execution_options.select_victims_in_parallel,
            current_already_prepared=current_already_prepared,
        )
        self._exchange_and_commit(domain)

    def finish_record_residency(self, residency_domain_index):
        """Finish a record-union plan before its caller retires GPU readers.

        These native plans reset LeaseCount to zero and protect selected slots
        with epoch-local ring membership. They never populate a lease list.
        The owner caller still publishes real reader retirement on the current
        CUDA stream before the next domain preparation can advance its epoch.
        """
        self._live_residency_domains.remove(residency_domain_index)

    def plan_prefill(self, layer_index, selected_logical_records):
        """Eager union; preserve bytes written before planning by CPU-first prefill."""
        import torch
        from ulacache.cache.kernels.local_cache import remap_prefill_rows

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native prefill union requires eager execution")
        selection = self.execution_plan.selection_indices_by_layer[layer_index]
        domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
        state = self.residency_states[domain]
        if selection not in self._planned_selections:
            self._start_selection(layer_index)
            self._prepare_prefill_residency_records(domain, selected_logical_records)
            self._planned_selections.add(selection)
        output = torch.empty_like(selected_logical_records, dtype=torch.int32)
        remap_prefill_rows(selected_logical_records, state.directory, output)
        return output

    def _prepare_prefill_residency_records(self, domain, selected_logical_records):
        self._drain_transfers()
        import torch
        from ulacache.cache.kernels.cpu_residency import prepare_host_tier

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native prefill union requires eager execution")
        if domain in self._live_residency_domains:
            raise RuntimeError(
                "Finish the current domain reader before replacing its prefill records"
            )
        state, workspace = (
            self.residency_states[domain],
            self.transfer_workspaces[domain],
        )
        counts = None
        device_union = self.execution_options.prefill_union_method == "device"
        if device_union:
            from ulacache.cache.kernels.prefix_selection import compact_prefill_union

            needed, counts = compact_prefill_union(
                selected_logical_records.reshape(1, -1),
                self.prefill_union_marks,
                self.prefill_unique_records,
                self.prefill_unique_record_count,
            )
        else:
            needed = torch.unique(
                selected_logical_records[selected_logical_records > 0].int(),
                sorted=True,
            ).reshape(1, -1)
        if (
            needed.numel() + 1 > workspace.new_rows.numel()
            or needed.numel() > self.device_record_capacity
        ):
            raise ValueError("Complete native prefill union exceeds its startup bound")
        prepare_host_tier(
            needed,
            self.prefill_current_record,
            self.prefill_active_query_count,
            0,
            state,
            workspace,
            selected_counts=counts,
            parallel=device_union
            and self.execution_options.prepare_residency_in_parallel,
            parallel_victims=device_union
            and self.execution_options.select_victims_in_parallel,
        )
        self._exchange_and_commit(domain)

    def prepare_prefill_record_chunk(self, layer_index, logical_record_ids):
        """Borrow one resident chunk until its native reconstruction read finishes."""
        import torch
        from ulacache.cache.kernels.local_cache import remap_prefill_rows

        domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
        self._prepare_prefill_residency_records(domain, logical_record_ids)
        output = torch.empty_like(logical_record_ids, dtype=torch.int32)
        remap_prefill_rows(
            logical_record_ids, self.residency_states[domain].directory, output
        )
        return output

    def finish_prefill_record_chunk(self, layer_index):
        self._release_residency_readers(
            self.execution_plan.residency_domain_indices_by_layer[layer_index]
        )

    def _release_residency_readers(self, domain):
        self._drain_transfers()
        from ulacache.cache.kernels.cpu_residency import release_host_tier_leases

        release_host_tier_leases(
            self.residency_states[domain], self.transfer_workspaces[domain]
        )
        self._live_residency_domains.remove(domain)

    def after_attention(self, layer_index):
        selection = self.execution_plan.selection_indices_by_layer[layer_index]
        if (
            layer_index
            == self.execution_plan.last_consumer_layers_by_selection[selection]
        ):
            domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
            self._release_residency_readers(domain)

    def write(self, layer_index, logical_record_ids, *record_segments):
        self.before_layer_prepare(layer_index)
        domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
        self.record_storage.write_native_record_segments(
            layer_index,
            logical_record_ids,
            record_segments,
            self.residency_states[domain].directory,
        )

    @property
    def supports_local_native_current_publication(self):
        return (
            getattr(self.record_storage, "fuse_native_record_publication", False)
            is True
        )

    def write_native_paged_records(
        self,
        layer_index,
        logical_record_ids,
        source_record_pages,
        source_record_indices,
        *,
        local_device_after_commit=False,
    ):
        self.before_layer_prepare(layer_index)
        domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
        arguments = (
            layer_index,
            logical_record_ids,
            source_record_pages,
            source_record_indices,
            self.residency_states[domain].directory,
        )
        if local_device_after_commit:
            self.record_storage.write_native_record_source(
                *arguments, local_device_after_commit=True
            )
        else:
            self.record_storage.write_native_record_source(*arguments)

    def offload_prefill(self, bank_index, logical_record_ids):
        raise NotImplementedError(
            "Paged records currently retain their final prefill residency"
        )

    def device_storage_inventory(self):
        tensors = [
            self.remapped_selections,
            self.active_query_count,
            self.prefill_active_query_count,
            self.prefill_current_record,
            self.registration_frames,
        ]
        tensors += [tensor for state in self.residency_states for tensor in state]
        tensors += [
            tensor for workspace in self.transfer_workspaces for tensor in workspace
        ]
        for ring in self.presence_rings:
            tensors.extend(
                (
                    ring.bits,
                    ring.summary,
                    ring.bucket_epochs,
                    ring.gpu_bucket,
                    ring.bucket_views,
                    ring.view,
                )
            )
            if ring.naive_histogram is not None:
                tensors.append(ring.naive_histogram)
        if self.execution_options.prefill_union_method == "device":
            tensors.extend(
                (
                    self.prefill_union_marks,
                    self.prefill_unique_records,
                    self.prefill_unique_record_count,
                )
            )
        inventory = self.record_storage.device_storage_inventory()
        inventory.update(
            {
                tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                for tensor in tensors
                if tensor.numel()
            }
        )
        return inventory

    def host_storage_inventory(self):
        return self.record_storage.host_storage_inventory()

    def memory_inventory(self):
        inventory = self.record_storage.memory_inventory()
        inventory["gpu_bytes"] = sum(self.device_storage_inventory().values())
        inventory["gpu_metadata_bytes"] = (
            inventory["gpu_bytes"] - inventory["gpu_kv_bytes"]
        )
        inventory["residency_directory_count"] = len(self.residency_states)
        inventory["selection_producer_count"] = len(
            self.execution_plan.selection_producer_layers
        )
        inventory["parallel_current_scratch_bytes"] = (
            self.current_preparation_scratch_bytes
        )
        inventory["async_follower_io"] = (
            self.execution_options.copy_follower_records_asynchronously
        )
        inventory["follower_copy_mode"] = self.execution_options.follower_copy_mode
        inventory["exchange_launch_dimensions"] = list(
            self.execution_options.exchange_launch_dimensions
        )
        inventory["async_follower_banks_by_domain"] = {
            str(domain): list(banks) for domain, banks in self._async_banks.items()
        }
        return inventory

    def close(self):
        if self._closed:
            return
        self._drain_transfers()
        self.record_storage.close()
        self._closed = True
