"""Native sparse callbacks over the existing ordinary-TP cache implementation.

The ordinary TP arena, PB-LRU planner, miss-only pull, host tier, native writer,
and step publication retain their original implementations. This adapter binds
native byte pages and callback order; it defines no replacement algorithm.
"""

from dataclasses import replace
from types import SimpleNamespace


class NativeSharedTensorParallelExecution:
    accepts_encoded_native_records = True

    def __init__(self, cache, pool, staging, specification, native_width, dtype):
        self.memory_cache, self.pool, self.writer_staging = cache, pool, staging
        self.specification, self.native_width, self.dtype = (
            specification,
            native_width,
            dtype,
        )
        self.records_per_page = specification.storage_layout.records_per_page
        self.prefill_buffers = None
        self.decode_like = False
        self.current = None
        self._begun = False
        self._padded = {}
        self._written_views = {}

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
        if self.pool.flowkv_shared_index.in_forward:
            raise RuntimeError("The previous native TP reader has not retired")
        self.decode_like, self.current = decode_like, current_logical_record_ids
        self._begun = False
        self._padded.clear()
        self._written_views.clear()
        self.pool._flowkv_mode = "decode" if decode_like else "prefill"
        if publish_active_query_count:
            valid = self.pool.flowkv_arena.valid
            if active_query_counts is None:
                valid.fill_(query_count)
            else:
                rank = attention_data_parallel_rank
                valid.copy_(active_query_counts[rank : rank + 1])
        self.prefill_buffers.begin_forward(
            None if decode_like else prefill_request_query_counts
        )

    def _begin_decode(self, pending):
        pool = self.pool
        if self.current is None:
            self.current = pending
        if (
            self.current.data_ptr() != pending.data_ptr()
            or self.current.shape != pending.shape
            or self.current.dtype != pending.dtype
        ):
            raise ValueError(
                "Every native TP writer must use the bound current-ID tensor"
            )
        if self._begun:
            return
        pool._flowkv_current_native = self.current
        arena = pool.flowkv_arena
        current = pool._flowkv_global_current[: self.current.numel()]
        arena.kernel().globalize_current(
            self.current, current, pool._flowkv_owner_start
        )
        pool.flowkv_step_publication.begin(current, arena.valid)
        self.memory_cache.begin_model_forward()
        self._begun = True

    def _publish(self, layer):
        """Borrow the original native writer's encoded output, with no recast."""
        bank = self.writer_staging.execution_plan.storage_bank_indices_by_layer[layer]
        pending = self.writer_staging._pending_writes[bank]
        count = pending.logical_record_ids.numel()
        pages = self.writer_staging._record_planes_for_bank(bank)
        # NativeRecordWriteStaging reserves ordinal zero and writes1..count.
        values = pages.view(self.dtype).reshape(
            -1, 1, self.specification.storage_layout.bytes_per_record // 2
        )[1 : count + 1]
        self.memory_cache.write(pending.logical_record_ids, layer, (values,))
        del self.writer_staging._pending_writes[bank]

    def publish_after_native_write(self, cache_layer):
        if not self.decode_like:
            self._publish(cache_layer)

    def write_encoded_records(self, cache_layer, ids, values):
        """Use the original TP writer directly for declared complete native bytes."""
        if self.writer_staging.pending_logical_record_ids(cache_layer) is not None:
            return False
        if self.decode_like:
            group = self.pool.flowkv_layout.layer_groups.group(cache_layer)
            if group not in self._padded:
                return False
            if cache_layer in self._written_views:
                raise RuntimeError(
                    "Consume the native TP attention before another write"
                )
            self._begin_decode(ids)
            self.memory_cache.prepare_layer_execution(cache_layer)
            self._written_views[cache_layer] = self.memory_cache.access(
                self._padded[group], cache_layer
            )
        self.memory_cache.write(ids, cache_layer, (values,))
        return True

    def prefetch_attention(self, cache_layer, logical_record_ids):
        import torch
        from ulacache.cache.access import SparseAttentionAccessPlan

        if not self.decode_like:
            return
        selected = logical_record_ids.flatten(1).int().contiguous()
        if selected.shape[1] != self.native_width:
            raise ValueError("Native sparse selection width changed after startup")
        if self.current is None:
            raise RuntimeError("Bind current IDs before prefetching native TP history")
        self._begin_decode(self.current)
        groups = self.pool.flowkv_layout.layer_groups
        group = groups.group(cache_layer)
        self.memory_cache.prepare_layer_execution(cache_layer)
        if cache_layer == groups.anchors[group] and group not in self._padded:
            width = self.pool.flowkv_layout.topk
            if selected.shape[1] != width:
                selected = torch.nn.functional.pad(
                    selected, (0, width - selected.shape[1]), value=-1
                )
            self._padded[group] = selected
            self.memory_cache.prepare_sparse_attention(
                cache_layer, SparseAttentionAccessPlan(selected, self.current)
            )

    def prepare_attention(self, cache_layer, logical_record_ids):
        shape = logical_record_ids.shape
        if not self.decode_like:
            selected = logical_record_ids.flatten(1).int().contiguous()
            return self.prefill_buffers.select(cache_layer, selected).reshape(shape)
        if cache_layer in self._written_views:
            return (
                self._written_views[cache_layer]
                .slot_ids[:, : self.native_width]
                .reshape(shape)
            )
        pending = self.writer_staging.pending_logical_record_ids(cache_layer)
        if pending is None:
            raise RuntimeError("The native KV writer must run before TP attention")
        self._begin_decode(pending)
        self.prefetch_attention(cache_layer, logical_record_ids)
        group = self.pool.flowkv_layout.layer_groups.group(cache_layer)
        selected = self._padded[group]
        view = self.memory_cache.access(selected, cache_layer)
        # The original writer follows the selected history completion event.
        self._publish(cache_layer)
        # The native TileLang tail pad already materializes a contiguous table.
        return view.slot_ids[:, : self.native_width].reshape(shape)

    def reconstruct_prefill_records(
        self, cache_layer, logical_record_ids, output, read_native_records
    ):
        def read(planes, indices):
            read_native_records(planes, indices, output)
            return output

        return self.prefill_buffers.read_records(cache_layer, logical_record_ids, read)

    def finish_attention(self, cache_layer):
        if self.decode_like:
            self.memory_cache.finish_layer_attention(cache_layer)
            self._written_views.pop(cache_layer, None)

    def finish_forward(self):
        # Normal eager/captured decode retires at its last sparse layer.
        # A partial failed decode must drain its own stream before teardown.
        if self.pool.flowkv_shared_index.in_forward:
            self.pool.flowkv_shared_index.abort_forward()

    def _rows(self, pages):
        import torch

        return (
            (
                pages[:, None] * self.records_per_page
                + torch.arange(self.records_per_page, device=pages.device)
            )
            .reshape(-1)
            .int()
        )

    def register_full_pages(self, page_ids, *, prefer_device):
        self.memory_cache.register_logical_records(
            self._rows(page_ids), prefer_device=prefer_device
        )

    def release_full_pages(self, page_ids):
        self.memory_cache.release_logical_records(self._rows(page_ids))

    def clear_records(self):
        self.memory_cache.clear_registered_records()

    def get_device_storage_inventory(self):
        value = self.memory_cache.get_device_storage_inventory()
        value.update(self.writer_staging.get_device_storage_inventory())
        value.update(self.prefill_buffers.inventory())
        return value


def create_native_shared_tp_execution(
    specification,
    staging,
    *,
    device,
    dtype,
    group,
    degree,
    compiler,
    host_factory,
    ipc_factory,
    memory_region,
    prefill_records,
    prefill_query_chunk=256,
    prefetch=True,
    resident_ratio=3,
    low_query_pull_blocks=None,
):
    import torch
    from ulacache import UnifiedMemoryCache
    from ulacache.bindings import BackendServices, TensorParallelCacheBindings
    from ulacache.cache_config import CacheOptions, CacheExecutionOptions, PrefetchPolicy
    from ulacache.cache.prefetch import SharedIndexLifecycle
    from ulacache.cache.prefill.buffers import RequestPrefillBuffers
    from ulacache.cache.planning import create_owner_paged_execution_plan
    from ulacache.cache.tp_cache import configure_pool
    from ulacache.compiler import configure_compiler
    from ulacache.storage.workspace import PlainPrefillLayout
    from ulacache.storage.layout import history_capacity
    from ulacache.transfer.cpu_gpu import NativePagedRecordTransfer

    planes = specification.storage_layout.record_planes
    if (
        len(planes) != 1
        or planes[0].native_encoding != "mla.bfloat16.v1"
        or planes[0].bytes_per_segment != (1024,)
        or dtype != torch.bfloat16
    ):
        raise NotImplementedError(
            "Native shared TP currently binds complete BF16 NoPE512 records"
        )
    if degree != 8 or prefill_records < 1:
        raise NotImplementedError(
            "Bind eight native owners and explicit independent prefill windows"
        )
    sharing = specification.layer_reuse
    if sharing.key_value_storage_owners != tuple(range(sharing.layer_count)):
        raise NotImplementedError(
            "Native shared TP requires independent KV payload banks"
        )
    bounds, capacity = specification.sparse_access, specification.capacity_plan
    query_bound = 1 << (bounds.maximum_padded_query_count - 1).bit_length()
    width = (bounds.maximum_selected_record_count + 255) // 256 * 256
    page = specification.storage_layout.records_per_page
    if capacity.device_record_capacity % page or capacity.host_record_capacity % page:
        raise ValueError(
            "Native shared TP canonical tiers must contain complete native pages"
        )
    scratch = (
        (max(width, capacity.maximum_write_record_count) + page - 1) // page * page
    )
    layout = PlainPrefillLayout(
        layers=sharing.layer_count,
        groups=len(set(sharing.sparse_selection_owners)),
        max_batch=bounds.maximum_padded_query_count,
        max_query_tokens=query_bound,
        pool_rows=degree * capacity.logical_record_capacity,
        row_bytes=1024,
        disaggregation_mode="null",
        topk=width,
        page_size=page,
        history_slots=history_capacity(
            bounds.maximum_padded_query_count, width, resident_ratio, page
        ),
        native_segment_bytes=(1024,),
        layer_anchors=sharing.sparse_selection_owners,
        max_staging_tokens=capacity.maximum_write_record_count,
        prefill_window_rows=scratch,
        max_resident_queries=bounds.maximum_padded_query_count,
    )
    pool = SimpleNamespace(
        device=device,
        dtype=dtype,
        start_layer=0,
        allocation_page_size=page,
        flowkv_layout=layout,
        flowkv_plain_compact_prefill=True,
        supports_native_placement_preferences=True,
        _flowkv_current_native=None,
        _flowkv_current_rows=None,
        _flowkv_prefill_layer=None,
        _flowkv_decode_layer=None,
        _flowkv_mode="prefill",
    )
    # Match the native pool's byte offset into each peer's contiguous banks.
    pool._flowkv_peer_offset = (
        lambda layer: layer * pool.flowkv_shards.shape[1] * layout.row_bytes
    )
    policy = PrefetchPolicy(enabled=True, completion="layer")
    services = BackendServices(
        degree,
        group,
        1,
        CacheOptions(
            cpu_offload=True,
            cpu_slots=capacity.host_record_capacity,
            prefetch=policy,
            local_eviction_policy="age_lru",
            host_parallel_finish=True,
            host_parallel_victims=True,
            low_query_pull_blocks=low_query_pull_blocks,
        ),
        memory_region,
        ipc_factory,
        host_factory,
        lambda requested: requested,
    )
    configure_compiler(compiler)
    configure_pool(pool, services, step_publication=True)
    pool.flowkv_shared_index = SharedIndexLifecycle(
        layout.layer_groups, pool, prefetch=policy
    )
    public_spec = replace(
        specification,
        sparse_access=replace(
            bounds, maximum_padded_query_count=query_bound, maximum_selected_units=width
        ),
        capacity_plan=replace(
            capacity,
            prefill_workspace_record_capacity=scratch,
            maximum_prefill_union_record_count=scratch,
        ),
    )
    cache = UnifiedMemoryCache(
        public_spec,
        TensorParallelCacheBindings(
            pool,
            execution_options=CacheExecutionOptions(
                record_storage_backend="tensor_parallel", prefetch=policy
            ),
            owns_storage=True,
        ),
    )
    execution = NativeSharedTensorParallelExecution(
        cache, pool, staging, public_spec, bounds.maximum_selected_record_count, dtype
    )
    # The existing read-only prefill reader borrows the same canonical owner
    # directories and native page views. It never plans decode replacement.
    host = pool.flowkv_host
    plane_storage = SimpleNamespace(
        device_record_pages=pool.flowkv_shards.view(
            sharing.layer_count, capacity.device_record_capacity // page, page * 1024
        ),
        host_record_pool=SimpleNamespace(
            storage=host.host.storage.view(
                sharing.layer_count, capacity.host_record_capacity // page, page * 1024
            )
        ),
        record_transfer=NativePagedRecordTransfer(planes[0], page),
    )
    storage = SimpleNamespace(plane_storage=(plane_storage,))
    reader_backend = SimpleNamespace(
        active=pool.flowkv_arena.valid,
        owner_count=degree,
        owner_rank=pool.flowkv_rank,
        owner_process_group=group.cpu_group,
        execution_plan=create_owner_paged_execution_plan(specification),
        owner_residency=SimpleNamespace(
            record_storage=storage, residency_states=host.states
        ),
    )
    execution.prefill_buffers = RequestPrefillBuffers(
        reader_backend,
        specification,
        prefill_records,
        device_group=group.device_group,
        ipc_manager_factory=ipc_factory,
        query_chunk=prefill_query_chunk,
        prefetch=prefetch,
    )
    return execution
