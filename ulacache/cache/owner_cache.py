"""Bank-local native residency with disjoint owners and local attention rows."""

from dataclasses import replace


class RecordOwnerPagedResidency:
    """Compose existing placement, publication and opaque native record IO.

    Native writers supply complete record replicas within the owner group.
    Logical IDs belong to one shared allocator namespace. Each cold record is
    stored on one owner; local attention banks contain temporary copies and do
    not add logical history. Native Index and attention math stay outside.
    """

    def __init__(self, cache_specification, cache_bindings, execution_plan):
        import torch
        import torch.distributed as distributed
        from ulacache.compiler import configure_compiler
        from ulacache.bindings import LocalCacheBindings
        from ulacache.transfer.ipc import FlowKVPeerShards
        from ulacache.transfer.gpu_peer import NativePagedPeerRecordTransfer
        from ulacache.cache.reader_lifetime import DistributedResidencyPublication
        from ulacache.cache.local_paged_cache import NativePagedResidency
        from ulacache.cache.planning import create_paged_attention_workspace_plan

        self.cache_specification = cache_specification
        self.execution_plan = execution_plan
        self.device = cache_bindings.device
        self.owner_process_group = cache_bindings.owner_process_group
        self.owner_count = distributed.get_world_size(self.owner_process_group)
        self.owner_rank = distributed.get_rank(self.owner_process_group)
        self.native_selections_replicated = cache_bindings.native_selections_replicated
        self.selection_input_source_count = (
            1 if self.native_selections_replicated else self.owner_count
        )
        self.maximum_prefill_record_count = cache_bindings.maximum_prefill_record_count
        self.allocation_records_per_page = cache_bindings.allocation_records_per_page
        capacity = cache_specification.capacity_plan
        access = cache_specification.sparse_access
        layout = cache_specification.storage_layout
        self.owner_logical_capacity = capacity.logical_record_capacity
        self.maximum_query_count = access.maximum_padded_query_count
        self.selection_width = access.maximum_selected_record_count
        self.records_per_page = layout.records_per_page
        if not cache_bindings.native_record_writes_replicated:
            raise NotImplementedError(
                "Record owners currently require complete native writer replicas"
            )
        if (
            self.allocation_records_per_page < self.owner_count
            or self.allocation_records_per_page % self.owner_count
        ):
            raise NotImplementedError(
                "Native allocation pages must cover complete record-owner stripes"
            )
        if (
            capacity.reserved_logical_record_count * self.owner_count
            != self.allocation_records_per_page
        ):
            raise ValueError(
                "Per-owner reservations must compose the actual shared allocation guard"
            )
        if capacity.device_record_capacity < capacity.reserved_logical_record_count:
            raise ValueError(
                "Each owner must hold its native allocation padding on the GPU"
            )
        if (
            self.owner_logical_capacity * self.owner_count >= 2**31
            or self.owner_logical_capacity
            * self.owner_count
            % self.allocation_records_per_page
        ):
            raise ValueError(
                "The shared logical namespace must be int32 and native-allocation aligned"
            )
        if not 0 < self.maximum_prefill_record_count <= capacity.device_record_capacity:
            raise ValueError(
                "Bound native reconstruction chunks by owner GPU record capacity"
            )
        if cache_bindings.execution_options.record_storage_backend != "native_paged":
            raise ValueError("Record-owner services require native-paged storage")
        agreement = (
            cache_specification,
            self.allocation_records_per_page,
            self.maximum_prefill_record_count,
            cache_bindings.execution_options,
            self.native_selections_replicated,
        )
        peer_agreements = [None] * self.owner_count
        distributed.all_gather_object(
            peer_agreements, agreement, group=self.owner_process_group
        )
        if any(peer != agreement for peer in peer_agreements):
            raise ValueError(
                "Record owners must agree on native storage and execution geometry"
            )
        configure_compiler(cache_bindings.compile_device_module)

        exported_capacity = max(
            self.maximum_query_count * (self.selection_width + 1),
            self.maximum_prefill_record_count,
        )
        attention_capacity = max(
            self.maximum_query_count * self.selection_width,
            self.maximum_prefill_record_count,
        )
        attention_pages = (
            attention_capacity + self.records_per_page - 1
        ) // self.records_per_page
        bank_count = len(execution_plan.storage_producer_layers)
        self.attention_workspace_plan = create_paged_attention_workspace_plan(
            execution_plan,
            reuse_retired_banks=cache_bindings.execution_options.reuse_attention_workspaces,
        )
        options = dict(dtype=torch.int32, device=self.device)
        self.exported_records = torch.empty(exported_capacity, **options)
        self.owner_input_records = torch.empty(
            self.selection_input_source_count * exported_capacity, **options
        )
        self.owner_write_records = torch.empty(
            capacity.maximum_write_record_count, **options
        )
        self.source_owner_ranks = torch.empty(attention_capacity, **options)
        self.source_record_indices = torch.empty(attention_capacity, **options)
        self.prefill_record_indices = torch.empty(
            self.maximum_prefill_record_count, **options
        )
        self.remapped_selections = torch.empty(
            (
                len(execution_plan.selection_producer_layers),
                self.maximum_query_count,
                self.selection_width,
            ),
            **options,
        )
        self.no_current_records = torch.full(
            (self.maximum_query_count,), -1, dtype=torch.int64, device=self.device
        )
        self.attention_storage = tuple(
            torch.zeros(
                (
                    self.attention_workspace_plan.workspace_count,
                    attention_pages,
                    plane.bytes_per_page(self.records_per_page),
                ),
                dtype=torch.uint8,
                device=self.device,
            )
            for plane in layout.record_planes
        )
        self.record_transfers = tuple(
            NativePagedPeerRecordTransfer(plane, self.records_per_page)
            for plane in layout.record_planes
        )
        self.publication = DistributedResidencyPublication(
            bank_count,
            self.device,
            self.owner_process_group,
            ipc_manager_factory=cache_bindings.ipc_manager_factory,
        )
        owner_specification = replace(
            cache_specification,
            capacity_plan=replace(
                capacity,
                maximum_residency_input_record_count=max(
                    capacity.maximum_residency_input_record_count,
                    self.selection_input_source_count * exported_capacity,
                ),
            ),
        )
        local_services = LocalCacheBindings(
            self.device,
            cache_bindings.allocate_host_record_storage,
            cache_bindings.compile_device_module,
            cache_bindings.execution_options,
        )
        self.owner_residency = NativePagedResidency(
            owner_specification, local_services, execution_plan
        )
        storage = self.owner_residency.record_storage
        self.plane_storages = tuple(storage.plane_storage)
        self.payload_peers = tuple(
            FlowKVPeerShards(
                plane.device_record_pages,
                self.owner_process_group,
                ipc_manager_factory=cache_bindings.ipc_manager_factory,
            )
            for plane in self.plane_storages
        )
        self.directory_peers = tuple(
            FlowKVPeerShards(
                state.directory,
                self.owner_process_group,
                ipc_manager_factory=cache_bindings.ipc_manager_factory,
            )
            for state in self.owner_residency.residency_states
        )
        self.consumer_peers = FlowKVPeerShards(
            self.exported_records,
            self.owner_process_group,
            ipc_manager_factory=cache_bindings.ipc_manager_factory,
        )
        self.device_record_planes_by_layer = tuple(
            tuple(
                plane[
                    self.attention_workspace_plan.workspace_indices_by_storage_bank[
                        bank
                    ]
                ]
                for plane in self.attention_storage
            )
            for bank in execution_plan.storage_bank_indices_by_layer
        )
        self._selection_records = {}
        self._materialized_selections = set()
        self._prefill_remapped_selections = {}
        self._forward_current_records = None
        self._step_reader_retirement_layer = execution_plan.step_reader_retirement_layer
        self._completed_step_selections = set()
        self._batch_reader_retirement = self._step_reader_retirement_layer is not None
        self._communication_closed = False
        self._closed = False
        padding = torch.arange(
            1, self.allocation_records_per_page, dtype=torch.int64, device=self.device
        )
        self.allocate_unbound(padding)

    @property
    def active(self):
        return self.owner_residency.active_query_count

    @property
    def gpu(self):
        if len(self.attention_storage) != 1:
            raise ValueError("Use per-plane views for separate native record planes")
        return tuple(planes[0] for planes in self.device_record_planes_by_layer)

    def _map_owner_records(self, logical_record_ids):
        import torch
        from ulacache.cache.kernels.owner_lookup import map_owner_records

        count = logical_record_ids.numel()
        output = (
            self.owner_write_records[:count]
            if count <= self.owner_write_records.numel()
            else torch.empty(count, dtype=torch.int32, device=self.device)
        )
        map_owner_records(
            logical_record_ids,
            output,
            owner_rank=self.owner_rank,
            owner_count=self.owner_count,
            owner_logical_capacity=self.owner_logical_capacity,
        )
        return output

    def allocate_unbound(self, logical_record_ids, *, prefer_gpu=True):
        self.owner_residency.allocate_unbound(
            self._map_owner_records(logical_record_ids), prefer_gpu=prefer_gpu
        )

    def release(self, logical_record_ids):
        self.owner_residency._require_retired_readers()
        self.publication.wait_for_all_readers()
        self.owner_residency.release(self._map_owner_records(logical_record_ids))

    def clear(self):
        import torch

        self.owner_residency._require_retired_readers()
        self.publication.wait_for_all_readers()
        self.owner_residency.clear()
        padding = torch.arange(
            1, self.allocation_records_per_page, dtype=torch.int64, device=self.device
        )
        self.allocate_unbound(padding)
        self._selection_records.clear()
        self._materialized_selections.clear()
        self._prefill_remapped_selections.clear()
        self._forward_current_records = None
        self._completed_step_selections.clear()
        self._batch_reader_retirement = self._step_reader_retirement_layer is not None

    def begin_forward(self):
        self.owner_residency.begin_forward()
        self.publication.wait_for_all_readers()
        self._selection_records.clear()
        self._materialized_selections.clear()
        self._prefill_remapped_selections.clear()
        self._forward_current_records = None
        self._completed_step_selections.clear()
        self._batch_reader_retirement = self._step_reader_retirement_layer is not None

    def prepare_forward_current_records(self, current_logical_record_ids):
        """Reuse the shared native transaction on this owner's local IDs."""
        import torch

        if (
            current_logical_record_ids.ndim != 1
            or current_logical_record_ids.dtype not in (torch.int32, torch.int64)
            or not current_logical_record_ids.is_contiguous()
            or current_logical_record_ids.numel() > self.maximum_query_count
        ):
            raise ValueError(
                "Bind contiguous native current IDs within the startup query bound"
            )
        if self._selection_records:
            raise RuntimeError(
                "Prepare shared current records before the first owner selection"
            )
        mapped = self._map_owner_records(current_logical_record_ids)
        if not self.owner_residency.prepare_forward_current_records(mapped):
            return False
        self._forward_current_records = current_logical_record_ids
        return True

    def prepare_next_residency_epochs(self):
        self.owner_residency.prepare_next_residency_epochs()

    def before_layer_prepare(self, layer_index):
        """All native writes and record reads use the current CUDA stream."""

    def _prepare_exported_records(
        self, domain, records_per_consumer, *, current_already_prepared=False
    ):
        from ulacache.cache.kernels.owner_lookup import (
            collect_owner_records,
            map_owner_records,
        )

        self.publication.begin_selection(domain)
        needed = self.owner_input_records[
            : self.selection_input_source_count * records_per_consumer
        ]
        if self.native_selections_replicated:
            map_owner_records(
                self.exported_records[:records_per_consumer],
                needed,
                owner_rank=self.owner_rank,
                owner_count=self.owner_count,
                owner_logical_capacity=self.owner_logical_capacity,
            )
        else:
            collect_owner_records(
                self.consumer_peers.ptrs,
                needed,
                owner_rank=self.owner_rank,
                owner_count=self.owner_count,
                records_per_consumer=records_per_consumer,
                owner_logical_capacity=self.owner_logical_capacity,
            )
        self.owner_residency.prepare_record_residency(
            domain, needed, current_already_prepared=current_already_prepared
        )

    def plan(self, layer_index, selected_logical_records, current_write_records):
        prepared = self._forward_current_records is not None
        if prepared:
            common = self._forward_current_records
            if (
                current_write_records.data_ptr() != common.data_ptr()
                or current_write_records.shape != common.shape
                or current_write_records.dtype != common.dtype
            ):
                raise ValueError(
                    "Every owner selection must use the bound native current-ID tensor"
                )
        from ulacache.cache.kernels.owner_lookup import export_consumer_records

        selection = self.execution_plan.selection_indices_by_layer[layer_index]
        if (
            layer_index != self.execution_plan.selection_producer_layers[selection]
            or selection in self._selection_records
        ):
            raise RuntimeError(
                "Plan each bank-local selection once at its first consumer"
            )
        exported = self.exported_records[
            : self.maximum_query_count * (self.selection_width + 1)
        ].view(self.maximum_query_count, self.selection_width + 1)
        export_consumer_records(
            selected_logical_records, current_write_records, self.active, exported
        )
        self._selection_records[selection] = selected_logical_records
        domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
        self._prepare_exported_records(
            domain,
            selected_logical_records.shape[0] * (self.selection_width + 1),
            current_already_prepared=prepared,
        )

    def write(self, layer_index, logical_record_ids, *record_segments):
        self.owner_residency.write(
            layer_index, self._map_owner_records(logical_record_ids), *record_segments
        )

    def write_native_paged_records(
        self, layer_index, logical_record_ids, source_pages, source_indices
    ):
        self.owner_residency.write_native_paged_records(
            layer_index,
            self._map_owner_records(logical_record_ids),
            source_pages,
            source_indices,
        )

    def _gather_records(
        self, layer_index, selected_records, active_queries, output_indices
    ):
        from ulacache.cache.kernels.owner_lookup import read_owner_locations

        domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
        bank = self.execution_plan.storage_bank_indices_by_layer[layer_index]
        workspace_index = (
            self.attention_workspace_plan.workspace_indices_by_storage_bank[bank]
        )
        self.publication.publish_residency(domain)
        count = selected_records.numel()
        owners, locations = (
            self.source_owner_ranks[:count],
            self.source_record_indices[:count],
        )
        read_owner_locations(
            selected_records,
            active_queries,
            self.directory_peers[domain].ptrs,
            owners,
            locations,
            output_indices,
            owner_count=self.owner_count,
            owner_logical_capacity=self.owner_logical_capacity,
        )
        for transfer, peers, storage, attention in zip(
            self.record_transfers,
            self.payload_peers,
            self.plane_storages,
            self.attention_storage,
        ):
            transfer.gather(
                peers.ptrs,
                owners,
                locations,
                attention[workspace_index],
                output_indices.reshape(-1),
                source_bank_byte_offset=bank * storage.device_record_pages.stride(0),
            )

    def attention_rows(self, layer_index, query_count):
        selection = self.execution_plan.selection_indices_by_layer[layer_index]
        output = self.remapped_selections[selection, :query_count]
        if selection not in self._materialized_selections:
            selected = self._selection_records[selection]
            self._gather_records(layer_index, selected, self.active, output)
            self._materialized_selections.add(selection)
        return output

    def after_attention(self, layer_index):
        selection = self.execution_plan.selection_indices_by_layer[layer_index]
        if (
            layer_index
            == self.execution_plan.last_consumer_layers_by_selection[selection]
        ):
            if not self._batch_reader_retirement:
                self.finish_prefill_record_chunk(layer_index)
                return
            if selection in self._completed_step_selections:
                raise RuntimeError("Finish each owner selection exactly once")
            self._completed_step_selections.add(selection)
            if layer_index == self._step_reader_retirement_layer:
                count = len(self.execution_plan.selection_producer_layers)
                if self._completed_step_selections != set(range(count)):
                    raise RuntimeError(
                        "Finish every owner selection before step retirement"
                    )
                # Independent directories remain live until all local consumers
                # have been queued. Keep reclamation checks effective meanwhile.
                domains = range(len(self.owner_residency.residency_states))
                if self.owner_residency._live_residency_domains != set(domains):
                    raise RuntimeError(
                        "Step retirement requires every owner directory to be live"
                    )
                for domain in domains:
                    self.owner_residency.finish_record_residency(domain)
                self.publication.finish_all_readers()

    def plan_prefill(self, layer_index, selected_logical_records):
        # Eager prefix reconstruction may revisit a directory in bounded chunks.
        # Its existing per-chunk publication remains independent of decode batching.
        self._batch_reader_retirement = False
        selection = self.execution_plan.selection_indices_by_layer[layer_index]
        if selected_logical_records.shape[0] <= self.maximum_query_count:
            if selection not in self._selection_records:
                self.plan(
                    layer_index,
                    selected_logical_records,
                    self.no_current_records[: selected_logical_records.shape[0]],
                )
            return self.attention_rows(layer_index, selected_logical_records.shape[0])
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native owner prefill union requires eager execution")
        if selection not in self._selection_records:
            if self.execution_plan.selection_producer_layers[selection] != layer_index:
                raise RuntimeError(
                    "Plan each bank-local selection once at its first consumer"
                )
            # Prefill can have many more query rows than the decode Graph
            # bound. Gather each referenced native record once, then restore
            # the original query/selection order without changing reader math.
            records, inverse = torch.unique(
                selected_logical_records.reshape(-1), sorted=True, return_inverse=True
            )
            if records.numel() > self.maximum_prefill_record_count:
                raise ValueError(
                    "Native prefill selected union exceeds its startup record bound"
                )
            # Every consumer publishes the same startup-bounded prefix, even
            # when independent native selections have different union lengths.
            # A local unique count cannot describe the other consumers' IPC
            # buffers. Negative padding retains their complete live unions.
            bound = min(
                self.maximum_prefill_record_count, selected_logical_records.numel()
            )
            resident = self.prepare_prefill_record_chunk(
                layer_index, records, consumer_record_bound=bound
            )
            self._prefill_remapped_selections[selection] = resident.index_select(
                0, inverse
            ).reshape(selected_logical_records.shape)
            self._selection_records[selection] = selected_logical_records
        return self._prefill_remapped_selections[selection]

    def prepare_prefill_record_chunk(
        self, layer_index, logical_record_ids, *, consumer_record_bound=None
    ):
        import torch

        count = logical_record_ids.numel()
        bound = count if consumer_record_bound is None else consumer_record_bound
        if not 0 <= count <= bound <= self.maximum_prefill_record_count:
            raise ValueError("Native reconstruction exceeds its startup chunk bound")
        if bound > count:
            self.exported_records[count:bound].fill_(-1)
        self.exported_records[:count].copy_(
            torch.where(logical_record_ids > 0, logical_record_ids, -1).int()
        )
        domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
        self._prepare_exported_records(domain, bound)
        output = self.prefill_record_indices[:count]
        self._gather_records(
            layer_index,
            logical_record_ids.reshape(1, -1),
            self.owner_residency.prefill_active_query_count,
            output,
        )
        return output

    def finish_prefill_record_chunk(self, layer_index):
        domain = self.execution_plan.residency_domain_indices_by_layer[layer_index]
        self.owner_residency.finish_record_residency(domain)
        self.publication.finish_readers(domain)

    def offload_prefill(self, bank_index, logical_record_ids):
        raise NotImplementedError(
            "Owner paged storage retains its final prefill residency"
        )

    def device_storage_inventory(self):
        tensors = [
            self.exported_records,
            self.owner_input_records,
            self.owner_write_records,
            self.source_owner_ranks,
            self.source_record_indices,
            self.prefill_record_indices,
            self.remapped_selections,
            self.no_current_records,
            *self.attention_storage,
        ]
        tensors += [
            peer.ptrs
            for peer in (
                *self.payload_peers,
                *self.directory_peers,
                self.consumer_peers,
            )
        ]
        tensors += list(self._selection_records.values())
        inventory = self.owner_residency.device_storage_inventory()
        inventory.update(self.publication.get_device_storage_inventory())
        inventory.update(
            {
                tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                for tensor in tensors
                if tensor.numel()
            }
        )
        return inventory

    def host_storage_inventory(self):
        return self.owner_residency.host_storage_inventory()

    def memory_inventory(self):
        import torch.distributed as distributed

        value = self.owner_residency.memory_inventory()
        attention_bytes = sum(plane.nbytes for plane in self.attention_storage)
        value.update(
            gpu_cold_key_value_bytes=value["gpu_kv_bytes"],
            gpu_resident_attention_bytes=attention_bytes,
            resident_attention_workspace_count=self.attention_workspace_plan.workspace_count,
            gpu_kv_bytes=value["gpu_kv_bytes"] + attention_bytes,
            gpu_bytes=sum(self.device_storage_inventory().values()),
            record_distribution="owner_striped_complete_records",
            storage_owner_count=self.owner_count,
            storage_owner_rank=self.owner_rank,
            storage_owner_process_ranks=(
                distributed.get_process_group_ranks(self.owner_process_group)
                if distributed.is_initialized()
                else None
            ),
            gpu_cold_records_per_owner=self.cache_specification.capacity_plan.device_record_capacity,
            cpu_cold_records_per_owner=self.cache_specification.capacity_plan.host_record_capacity,
            global_logical_record_capacity=self.owner_count
            * self.owner_logical_capacity,
            global_usable_logical_record_capacity=self.owner_count
            * self.owner_logical_capacity
            - self.allocation_records_per_page,
        )
        value["gpu_metadata_bytes"] = value["gpu_bytes"] - value["gpu_kv_bytes"]
        return value

    def close(self):
        import torch
        import torch.distributed as distributed

        if self._closed:
            return
        if not self._communication_closed:
            self.owner_residency._require_retired_readers()
            self.publication.wait_for_all_readers()
            torch.cuda.synchronize(self.device)
            distributed.barrier(group=self.owner_process_group)
            for peer in (
                *self.payload_peers,
                *self.directory_peers,
                self.consumer_peers,
            ):
                peer.close()
            distributed.barrier(group=self.owner_process_group)
            self.publication.close()
            self._communication_closed = True
        self.owner_residency.close()
        self._closed = True
