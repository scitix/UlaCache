"""Shared execution lifecycle for framework-owned sparse cache readers/writers."""

from ulacache.cache.access import SparseAttentionAccessPlan


class NativeSparseCacheExecution:
    """Coordinate one homogeneous cache tier without model or framework imports.

    Native bindings supply canonical logical record IDs, packed writer output,
    and a reader that reconstructs one resident chunk into native output rows.
    Quantization, attention math and recurrent state remain framework-owned.
    """

    def __init__(
        self,
        memory_cache,
        execution_plan,
        writer_staging,
        cache_specification,
        device,
        *,
        maximum_prefill_record_count=None,
        prefill_buffers=None,
        local_current_publication=False,
    ):
        import torch

        self.memory_cache = memory_cache
        if type(local_current_publication) is not bool:
            raise ValueError("Local current publication must be explicitly boolean")
        self.local_current_publication = local_current_publication
        self.execution_plan = execution_plan
        self.writer_staging = writer_staging
        self.device = device
        self.records_per_page = cache_specification.storage_layout.records_per_page
        self.prefill_chunk_capacity = (
            cache_specification.capacity_plan.device_record_capacity
            if maximum_prefill_record_count is None
            else maximum_prefill_record_count
        )
        query_capacity = cache_specification.sparse_access.maximum_padded_query_count
        self.current_write_records = torch.full(
            (query_capacity,), -1, dtype=torch.int64, device=device
        )
        self.no_current_write_records = torch.full_like(self.current_write_records, -1)
        self.reserved_logical_records = torch.arange(
            1, self.records_per_page, dtype=torch.int32, device=device
        )
        self.memory_cache.register_logical_records(self.reserved_logical_records)
        self.decode_like = False
        self.prefill_buffers = prefill_buffers
        self._resident_prefill_layers = set()
        self._committed_current_publications = {}

    @staticmethod
    def _current_view_identity(records):
        return (records.data_ptr(), records.shape, records.dtype, records.device)

    def begin_forward(
        self,
        *,
        decode_like,
        active_query_counts,
        attention_data_parallel_rank,
        query_count,
        publish_active_query_count=True,
        current_logical_record_ids=None,
        prefill_request_query_counts=None,
    ):
        """Reset each execution; publish query counts outside recorded forwards.

        Native Graph warmup/capture repeats one prepared batch. Those executions
        reset host bookkeeping without recording a padded query-count overwrite.
        """
        self.decode_like = decode_like
        buffers = getattr(self, "prefill_buffers", None)
        if buffers is not None:
            buffers.begin_forward(None if decode_like else prefill_request_query_counts)
        self._resident_prefill_layers.clear()
        self._committed_current_publications = {}
        if publish_active_query_count:
            active_count = self.memory_cache.active_query_count_tensor
            if active_query_counts is None:
                active_count.fill_(query_count)
            else:
                rank = attention_data_parallel_rank
                active_count.copy_(active_query_counts[rank : rank + 1])
        self.memory_cache.begin_model_forward()
        # Framework replay setup and recorded forward setup may both call this
        # method. Publish counts here, but emit device lifetime operations only
        # at the first native reader. Eager and recorded forwards then each
        # advance canonical epochs exactly once, even with two setup hooks.
        self._residency_preparation_pending = decode_like
        self._pending_forward_current_records = current_logical_record_ids

    def _prepare_forward_residency_once(self, current_write_records=None):
        if not getattr(self, "_residency_preparation_pending", False):
            return
        current = self._pending_forward_current_records
        # Eager DP padding can replace or reshape the batch's writer locations.
        # Check the explicit common-current promise before any device mutation;
        # an unbound first producer retains the ordinary per-layer transaction.
        bound = (
            current is not None
            and current_write_records is not None
            and current.data_ptr() == current_write_records.data_ptr()
            and current.shape == current_write_records.shape
            and current.dtype == current_write_records.dtype
        )
        prepare_current = getattr(
            self.memory_cache, "prepare_forward_current_records", None
        )
        prepared = bound and prepare_current is not None and prepare_current(current)
        if not prepared:
            prepare_expiry = getattr(
                self.memory_cache, "prepare_next_residency_epochs", None
            )
            if prepare_expiry is not None:
                prepare_expiry()
        self._residency_preparation_pending = False

    def publish_after_native_write(self, cache_layer):
        if not self.decode_like:
            self.writer_staging.publish_staged_records(self.memory_cache, cache_layer)

    def prepare_attention(self, cache_layer, logical_record_ids):
        shape = logical_record_ids.shape
        selected = logical_record_ids.flatten(1).int().contiguous()
        if self.decode_like:
            pending = self.writer_staging.pending_logical_record_ids(cache_layer)
            selection = self.execution_plan.selection_indices_by_layer[cache_layer]
            if self.execution_plan.selection_producer_layers[selection] == cache_layer:
                current = self.no_current_write_records[: shape[0]]
                if pending is not None:
                    if pending.numel() == shape[0]:
                        current = pending
                    else:
                        self.current_write_records.fill_(-1)
                        self.current_write_records[: pending.numel()].copy_(pending)
                        current = self.current_write_records[: shape[0]]
                self._prepare_forward_residency_once(current)
                self.memory_cache.prepare_sparse_attention(
                    cache_layer, SparseAttentionAccessPlan(selected, current)
                )
                # The existing transaction has now committed every active current
                # to this GPU. Followers qualify only for that exact ID view.
                if (
                    getattr(self, "local_current_publication", False)
                    and pending is not None
                    and pending.numel() <= shape[0]
                    and getattr(
                        self.memory_cache,
                        "supports_local_native_current_publication",
                        False,
                    )
                    is True
                ):
                    self._committed_current_publications[selection] = (
                        self._current_view_identity(pending)
                    )
            # A selection follower can still own an independent KV payload.
            if pending is not None:
                committed = getattr(self, "_committed_current_publications", {}).get(
                    selection
                )
                local_device = (
                    committed is not None
                    and committed == self._current_view_identity(pending)
                )
                if local_device:
                    self.writer_staging.publish_staged_records(
                        self.memory_cache, cache_layer, local_device_after_commit=True
                    )
                else:
                    self.writer_staging.publish_staged_records(
                        self.memory_cache, cache_layer
                    )
            inputs = self.memory_cache.get_resident_attention_inputs(
                cache_layer, shape[0]
            )
        else:
            buffers = getattr(self, "prefill_buffers", None)
            if buffers is not None:
                return buffers.select(cache_layer, selected).reshape(shape)
            inputs = self.memory_cache.prepare_prefill_attention_inputs(
                cache_layer, selected
            )
            self._resident_prefill_layers.add(cache_layer)
        return inputs.device_record_indices.reshape(shape)

    def reconstruct_prefill_records(
        self, cache_layer, logical_record_ids, output, read_native_records
    ):
        """Reconstruct one tensor or several native planes with the same record order."""
        for start in range(0, logical_record_ids.numel(), self.prefill_chunk_capacity):
            stop = min(start + self.prefill_chunk_capacity, logical_record_ids.numel())
            inputs = self.memory_cache.prepare_prefill_record_chunk(
                cache_layer, logical_record_ids[start:stop]
            )
            chunk_output = (
                tuple(plane[start:stop] for plane in output)
                if isinstance(output, tuple)
                else output[start:stop]
            )
            read_native_records(
                inputs.record_planes, inputs.device_record_indices, chunk_output
            )
            self.memory_cache.finish_prefill_record_chunk(cache_layer)

    def finish_attention(self, cache_layer):
        if self.decode_like or cache_layer in self._resident_prefill_layers:
            self.memory_cache.finish_layer_attention(cache_layer)
            self._resident_prefill_layers.discard(cache_layer)
            if self.decode_like:
                last_consumers = getattr(
                    self.execution_plan, "last_consumer_layers_by_selection", ()
                )
                selection = self.execution_plan.selection_indices_by_layer[cache_layer]
                if last_consumers and last_consumers[selection] == cache_layer:
                    self._committed_current_publications.pop(selection, None)

    def _logical_records_for_pages(self, page_ids):
        import torch

        return (
            (
                page_ids[:, None] * self.records_per_page
                + torch.arange(self.records_per_page, device=self.device)
            )
            .reshape(-1)
            .int()
        )

    def register_full_pages(self, page_ids, *, prefer_device):
        # Native causal masking discards unwritten page tails after reconstruction.
        self.memory_cache.register_logical_records(
            self._logical_records_for_pages(page_ids), prefer_device=prefer_device
        )

    def release_full_pages(self, page_ids):
        self.memory_cache.release_logical_records(
            self._logical_records_for_pages(page_ids)
        )

    def clear_records(self):
        self._committed_current_publications.clear()
        self.memory_cache.clear_registered_records()
        self.memory_cache.register_logical_records(self.reserved_logical_records)

    def get_device_storage_inventory(self):
        inventory = self.memory_cache.get_device_storage_inventory()
        inventory.update(self.writer_staging.get_device_storage_inventory())
        for tensor in (
            self.current_write_records,
            self.no_current_write_records,
            self.reserved_logical_records,
        ):
            inventory[tensor.untyped_storage().data_ptr()] = (
                tensor.untyped_storage().nbytes()
            )
        if self.prefill_buffers is not None:
            inventory.update(self.prefill_buffers.inventory())
        return inventory
