"""Logical KV memory API over the existing native cache execution paths."""

from ulacache.cache.access import (
    ResidentAttentionInputs as ResidentAttentionInputs,
    ResidentView,
    SparseAttentionAccessPlan,
)
from ulacache.cache.specification import UnifiedCacheSpecification
from ulacache.cache.access import ForwardExecutionContext
from ulacache.bindings import (
    LocalCacheBindings,
    RecordOwnerBindings,
    TensorParallelCacheBindings,
)

__all__ = ["init", "UlaCache", "UnifiedMemoryCache", "ResidentView"]


def init(
    spec: UnifiedCacheSpecification,
    policy: LocalCacheBindings | RecordOwnerBindings | TensorParallelCacheBindings,
):
    """Bind resources, execution policy and an optional native logical allocator."""
    return UlaCache(spec, policy)


class UnifiedMemoryCache:
    """Own one sparse cache tier; retain native allocation and attention math.

    Logical IDs are registered by the framework allocator. Forward inputs are
    its contiguous CUDA int32 tensors, with the native query ordering and causal
    metadata. The existing backend checks its tensor and workspace contracts.
    Calls use the current stream and are serialized with captured Graph replay.
    Close runs after the caller retires Graphs and borrowed storage views.
    """

    def __init__(
        self,
        cache_specification: UnifiedCacheSpecification,
        cache_bindings: LocalCacheBindings
        | RecordOwnerBindings
        | TensorParallelCacheBindings,
    ):
        from ulacache.cache.create import create_execution_engine

        backend = cache_bindings.execution_options.record_storage_backend
        self.cache_specification = cache_specification
        self._execution_engine = create_execution_engine(
            cache_specification, cache_bindings
        )
        if (
            backend == "native_paged"
            and len(cache_specification.storage_layout.record_planes) > 1
        ):
            self._device_record_planes_by_layer = (
                self._execution_engine.device_record_planes_by_layer
            )
        else:
            self._device_record_planes_by_layer = tuple(
                (self._execution_engine.gpu[layer_index],)
                for layer_index in range(cache_specification.layer_reuse.layer_count)
            )
        from ulacache.cache.runtime import AccessEngine, SlotManager

        if isinstance(cache_bindings, TensorParallelCacheBindings):
            self._access_engine = self._execution_engine.create_access_engine(
                self._device_record_planes_by_layer
            )
        else:
            self._access_engine = AccessEngine(
                cache_specification,
                cache_bindings,
                self._execution_engine,
                self._device_record_planes_by_layer,
            )
        self._slot_manager = SlotManager(
            self._execution_engine,
            cache_bindings.logical_slot_allocator,
            self._access_engine,
        )

    def alloc(self, n):
        """Allocate logical IDs through the bound authority, then register storage.

        This is an admission operation outside Graph capture, not a KV write.
        Existing integrations may keep their native allocator and registration
        hooks instead of binding an allocator here; no second ID pool is created.
        """
        if self._execution_engine is None:
            raise RuntimeError("The cache is closed")
        return self._slot_manager.alloc(n)

    def write(self, ids, layer, values, *, source_indices=None):
        """Publish native KV bytes into already allocated logical records.

        ``values`` is a tuple of native segments in specification order (plane,
        then segment). With ``source_indices``, it is native packed pages, or
        one page tensor per plane. Source indices are physical record positions
        in those pages. A ``FileRecordSource`` instead supplies complete native
        rows from a mounted file, including GPFS. That path is a bounded,
        blocking restore outside model forwards and CUDA Graphs; it retains
        logical IDs and writes through the existing physical storage routing.
        Quantization and RoPE remain with the native writer.
        This queues the existing write on the current stream; it neither
        allocates IDs nor plans residency. Publish before the matching reader,
        and after planning when that plan reserves fresh-write destinations.
        """
        if self._execution_engine is None:
            raise RuntimeError("The cache is closed")
        if (
            type(layer) is not int
            or not 0 <= layer < self.cache_specification.layer_reuse.layer_count
        ):
            raise ValueError("Layer outside the configured cache")
        if len(ids.shape) != 1:
            raise ValueError("Write IDs must be a one-dimensional logical-ID tensor")
        count = ids.numel()
        from ulacache.transfer.file_io import FileRecordSource

        if isinstance(values, FileRecordSource):
            if source_indices is not None:
                raise ValueError(
                    "File record sources already describe their complete source range"
                )
            if count != values.record_count:
                raise ValueError("Supply one logical write ID per file source record")
            from ulacache.storage.layout import native_mla_encodings_match

            planes = self.cache_specification.storage_layout.record_planes
            if (
                len(planes) != 1
                or not native_mla_encodings_match(
                    planes[0].native_encoding, values.record_plane.native_encoding
                )
                or planes[0].bytes_per_record != values.row_bytes
            ):
                raise ValueError(
                    "File source must match the complete native cache record encoding"
                )
            restore = getattr(self._execution_engine, "write_file_records", None)
            if restore is None:
                raise NotImplementedError(
                    "This backend has no validated native file writer"
                )
            self._access_engine.require_idle("restoring file records")
            return restore(layer, ids, values)
        if count > self.cache_specification.capacity_plan.maximum_write_record_count:
            raise ValueError("Native writer exceeds its startup record capacity")
        if source_indices is None:
            planes = self.cache_specification.storage_layout.record_planes
            if not isinstance(values, (tuple, list)) or len(values) != sum(
                len(plane.bytes_per_segment) for plane in planes
            ):
                raise ValueError(
                    "Native writer must supply every declared plane segment in a tuple or list"
                )
            self.write_native_record_segments(layer, ids, values)
        else:
            if len(source_indices.shape) != 1 or source_indices.numel() != count:
                raise ValueError("Supply one source record index per logical write ID")
            self.write_native_paged_records(layer, ids, values, source_indices)

    def access(self, ids, layer) -> ResidentView:
        """Acquire selected records using existing planning and reader lifetimes.

        ``ids`` is the canonical padded [queries, width] logical-ID tensor. Call
        begin_model_forward first. Native writers still publish KV explicitly.
        A prepared selection is reused without another residency transaction;
        otherwise this call plans read-only access with no current-write IDs.
        Advanced bindings prepare with current IDs before publishing writes.
        Inputs remain unchanged through all consumers of their selection.
        """
        if self._execution_engine is None:
            raise RuntimeError("The cache is closed")
        return self._access_engine.access(ids, layer)

    def free(self, ids):
        """Release retired records, then return IDs to the bound logical allocator.

        The serving adapter must finish native readers and retire request/prefix
        references first. Without a bound allocator this releases cache storage;
        the external allocator retains responsibility for recycling its IDs.
        """
        if self._execution_engine is None:
            raise RuntimeError("The cache is closed")
        self._slot_manager.free(ids)

    def register_logical_records(self, logical_record_ids, *, prefer_device=True):
        self._slot_manager.register(logical_record_ids, prefer_device=prefer_device)

    def register_receive_records(self, logical_record_ids):
        """Allocate canonical destinations before publishing remote addresses."""
        self._execution_engine.allocate_receive_rows(logical_record_ids)

    def release_logical_records(self, logical_record_ids):
        """Release only records retired by the native allocator or prefix tree."""
        self._slot_manager.release(logical_record_ids)

    def clear_registered_records(self):
        self._slot_manager.clear()

    def begin_model_forward(
        self, execution_context: ForwardExecutionContext | None = None
    ):
        """A native metadata hook may already have updated active_query_count_tensor."""
        self._access_engine.begin_forward(execution_context)

    def prepare_forward_current_records(self, current_logical_record_ids):
        """Optional same-current binding; unsupported reuse keeps serial begin."""
        prepare = getattr(
            self._execution_engine, "prepare_forward_current_records", None
        )
        return (
            bool(prepare(current_logical_record_ids)) if prepare is not None else False
        )

    def prepare_next_residency_epochs(self):
        """Retire independent recent buckets before a native decode forward.

        Canonical epochs and current-record pinning remain in each original
        per-selection transaction. Backends without this optional preparation
        retain their existing lifecycle.
        """
        prepare = getattr(self._execution_engine, "prepare_next_residency_epochs", None)
        if prepare is not None:
            prepare()

    def prepare_layer_execution(self, layer_index):
        self._execution_engine.before_layer_prepare(layer_index)

    def write_native_record_segments(
        self, layer_index, logical_record_ids, record_segments
    ):
        """Native segments, ordered by plane and then segment, including scales."""
        self._execution_engine.write(layer_index, logical_record_ids, *record_segments)

    @property
    def supports_local_native_current_publication(self):
        return getattr(
            self._execution_engine, "supports_local_native_current_publication", False
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
        """Copy native packed pages, or a tuple of pages for separate planes."""
        arguments = (
            layer_index,
            logical_record_ids,
            source_record_pages,
            source_record_indices,
        )
        if local_device_after_commit:
            if not self.supports_local_native_current_publication:
                raise ValueError(
                    "Local current publication requires native paged storage"
                )
            self._execution_engine.write_native_paged_records(
                *arguments, local_device_after_commit=True
            )
        else:
            self._execution_engine.write_native_paged_records(*arguments)

    def prepare_sparse_attention(
        self, layer_index, access_plan: SparseAttentionAccessPlan
    ):
        """Plan once at each layer that produces a new native selection result."""
        return self._access_engine.prepare_selection(layer_index, access_plan)

    def get_resident_attention_inputs(self, layer_index, query_count):
        return self._access_engine.resident_inputs(layer_index, query_count)

    def prepare_prefill_attention_inputs(
        self, layer_index, selected_logical_record_ids
    ):
        """Use the existing eager prefill union and native causal selection."""
        device_record_indices = self._execution_engine.plan_prefill(
            layer_index, selected_logical_record_ids
        )
        return self._create_attention_inputs(layer_index, device_record_indices)

    def _create_attention_inputs(self, layer_index, device_record_indices):
        return self._access_engine.create_view(layer_index, device_record_indices)

    def prepare_prefill_record_chunk(self, layer_index, logical_record_ids):
        """Borrow a bounded chunk for native reconstruction into its own workspace."""
        device_record_indices = self._execution_engine.prepare_prefill_record_chunk(
            layer_index, logical_record_ids
        )
        return self._create_attention_inputs(layer_index, device_record_indices)

    def finish_prefill_record_chunk(self, layer_index):
        """Release the chunk after its native reader is queued on the current stream."""
        self._execution_engine.finish_prefill_record_chunk(layer_index)

    def prepare_native_prefill_read_plan(self, selected_logical_record_ids):
        """Prepare the existing bounded TP prefix windows without reading KV."""
        prepare = getattr(
            self._execution_engine, "prepare_native_prefill_read_plan", None
        )
        if prepare is None:
            raise NotImplementedError(
                "This backend uses its resident prefill chunk interface"
            )
        return prepare(selected_logical_record_ids)

    def read_native_prefill_records(
        self,
        layer_index,
        selected_logical_record_ids,
        native_reader,
        *,
        output,
        read_plan=None,
    ):
        """Read bounded prefix windows with the adapter's original native math.

        The native reader receives borrowed pages and remapped IDs. The cache
        retains its existing transfers and staging; no full-prefix copy or
        replacement attention implementation is introduced.
        """
        read = getattr(self._execution_engine, "read_native_prefill_records", None)
        if read is None:
            raise NotImplementedError(
                "This backend uses its resident prefill chunk interface"
            )
        return read(
            layer_index,
            selected_logical_record_ids,
            native_reader,
            output=output,
            read_plan=read_plan,
        )

    def attend_native_prefill_records(
        self, layer_index, query, selected_logical_record_ids, lengths, native_attention
    ):
        """Supply complete query windows to the original native prefill reader.

        Selection and causal lengths stay paired with each query. The callback
        retains native attention and sink semantics; the cache supplies bounded
        resident windows using its existing prefill schedule.
        """
        consume = getattr(self._execution_engine, "attend_native_prefill_records", None)
        if consume is None:
            raise NotImplementedError(
                "This backend uses its resident prefill chunk interface"
            )
        return consume(
            layer_index, query, selected_logical_record_ids, lengths, native_attention
        )

    def finish_layer_attention(self, layer_index):
        self._access_engine.finish_layer(layer_index)

    @property
    def _no_current_records(self):
        return self._access_engine.no_current_records

    def finish_model_forward(self):
        """Retire asynchronous record writes before native prefix publication.

        A framework's radix tree keeps the same logical IDs across residency
        changes. The contiguous rank-local backend makes its pending writes
        visible to the next current-stream reader at this lifecycle boundary.
        """
        finish = getattr(self._execution_engine, "finish_forward", None)
        if finish is None:
            raise NotImplementedError(
                "This record backend has no prefix-publication boundary"
            )
        finish()

    def offload_prefill_records(self, residency_group_index, logical_record_ids):
        self._execution_engine.offload_prefill(
            residency_group_index, logical_record_ids
        )

    def get_memory_inventory(self):
        inventory = self._execution_engine.memory_inventory()
        if self._no_current_records is not None:
            inventory["gpu_bytes"] = sum(self.get_device_storage_inventory().values())
            inventory["gpu_metadata_bytes"] = (
                inventory["gpu_bytes"] - inventory["gpu_kv_bytes"]
            )
        return inventory

    def get_device_storage_inventory(self):
        inventory = self._execution_engine.device_storage_inventory()
        if self._no_current_records is not None:
            storage = self._no_current_records.untyped_storage()
            inventory[storage.data_ptr()] = storage.nbytes()
        return inventory

    def get_host_storage_inventory(self):
        """Count pinned backing allocations once, including separate native planes."""
        return self._execution_engine.host_storage_inventory()

    def get_device_record_planes(self, layer_index):
        """Borrow all native planes at startup; per-plane geometry stays in the specification."""
        return self._device_record_planes_by_layer[layer_index]

    @property
    def device_record_storage(self):
        """Single-plane compatibility view; separate K/V uses get_device_record_planes."""
        return self._execution_engine.gpu

    @property
    def active_query_count_tensor(self):
        """Persistent scalar populated by the framework metadata hook."""
        return self._execution_engine.active

    def close(self):
        if self._execution_engine is not None:
            self._execution_engine.close()
            self._execution_engine = None
            self._device_record_planes_by_layer = ()
            self._access_engine.detach()
            self._slot_manager.detach()

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()


UlaCache = UnifiedMemoryCache
