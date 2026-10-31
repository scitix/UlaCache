"""Public cache operations over the existing ordinary-TP storage and streams.

The native adapter has already bound the pool, raw writer, Index selection and
current-record publication. Adopting it preserves its single directory,
SharedIndexLifecycle, PB replacement state and existing transfer kernels.
"""

from types import ModuleType

from ulacache.cache.access import ResidentAttentionInputs, SparseAttentionAccessPlan


class NativeTensorParallelExecution:
    def __init__(self, specification, bindings):
        pool = bindings.pool
        layout = pool.flowkv_layout
        lifecycle = pool.flowkv_shared_index
        sharing = specification.layer_reuse
        owners = tuple(
            layout.layer_groups.anchors[layout.layer_groups.group(layer)]
            for layer in range(layout.layers)
        )
        if (
            pool._flowkv_storage_size < 2
            or pool._flowkv_namespaces != 1
            or pool.flowkv_request is not None
        ):
            raise NotImplementedError(
                "Bind the existing shared ordinary-TP Index residency path"
            )
        if lifecycle.pool is not pool or lifecycle.groups is not layout.layer_groups:
            raise ValueError("Bind the actual pool and shared Index lifecycle")
        if lifecycle.in_forward:
            raise RuntimeError("Adopt a TP pool before beginning a forward")
        if (
            sharing.layer_count != layout.layers
            or sharing.key_value_storage_owners != tuple(range(layout.layers))
            or sharing.index_storage_owners != owners
            or sharing.sparse_selection_owners != owners
            or sharing.residency_plan_anchors != owners
        ):
            raise ValueError(
                "The public sharing declaration must match the native TP groups"
            )
        planes = specification.storage_layout.record_planes
        if (
            len(planes) != 1
            or planes[0].bytes_per_record != layout.row_bytes
            or planes[0].page_memory_order != "record_major"
            or specification.storage_layout.records_per_page != layout.page_size
        ):
            raise ValueError("Bind the existing contiguous complete native TP records")
        access = specification.sparse_access
        if (
            access.maximum_padded_query_count != layout.max_query_tokens
            or access.maximum_selected_record_count != layout.topk
            or access.selection_identifier_space != "logical"
        ):
            raise ValueError(
                "The public selection bounds must match the native TP pool"
            )
        capacity = specification.capacity_plan
        host_slots = pool.flowkv_host.host_slots if pool.flowkv_host is not None else 0
        if (
            capacity.device_record_capacity != pool.flowkv_shards.shape[1]
            or capacity.host_record_capacity != host_slots
            or capacity.maximum_write_record_count != layout.max_staging_tokens
            or capacity.prefill_workspace_record_capacity
            != getattr(layout, "prefill_window_rows", 0)
        ):
            raise ValueError("Bind the actual per-owner GPU/CPU and writer capacities")
        options = bindings.execution_options
        if options.record_storage_backend != "tensor_parallel":
            raise ValueError("Ordinary TP requires its existing execution backend")
        if options.prefetch is not None:
            if options.prefetch.enabled != lifecycle.prefetch:
                raise ValueError(
                    "Prefetch must be lowered before native pool construction"
                )
            if (
                options.prefetch.enabled
                and options.follower_copy_mode != lifecycle.completion
            ):
                raise ValueError(
                    "Prefetch completion must match the native pool lifecycle"
                )
        if getattr(pool, "_ulacache_public_engine", None) is not None:
            raise RuntimeError("The TP pool already has a public cache handle")
        self.pool = pool
        self.specification = specification
        self.owns_storage = bindings.owns_storage
        self.gpu = tuple(
            pool.flowkv_arena.storage[layer] for layer in range(layout.layers)
        )
        self.active = pool.flowkv_arena.valid
        pool._ulacache_public_engine = self

    def create_access_engine(self, record_planes):
        return TensorParallelAccessEngine(self, record_planes)

    def allocate_unbound(self, ids, *, prefer_gpu=True):
        # Native TP registration retains its original initial-placement policy.
        # Changing that policy cannot be implemented by silently replacing IDs.
        if not prefer_gpu and not getattr(
            self.pool, "supports_native_placement_preferences", False
        ):
            raise NotImplementedError(
                "Ordinary TP retains native registration placement"
            )
        if self.pool.flowkv_host is not None:
            if prefer_gpu:
                self.pool.flowkv_host.allocate(ids)
            else:
                self.pool.flowkv_host.allocate(ids, prefer_gpu=False)

    def release(self, ids):
        if self.pool.flowkv_host is not None:
            self.pool.flowkv_host.release(ids)

    def clear(self):
        if self.pool.flowkv_host is not None:
            self.pool.flowkv_host.clear()
        else:
            from ulacache.cache.tp_cache import join_local_residency_clear

            join_local_residency_clear(self.pool)
            self.pool.flowkv_arena.clear_local_residency(readers_retired=True)

    def before_layer_prepare(self, layer):
        if self.pool.flowkv_shared_index.prefetch:
            self.pool.flowkv_shared_index.before_layer_prepare(layer)

    def prepare_forward_current_records(self, current):
        # The adapter already globalizes/publishes these same native IDs. Do
        # not emit that publication twice through the public facade.
        existing = self.pool._flowkv_current_native
        if existing is None or existing.data_ptr() != current.data_ptr():
            raise ValueError(
                "Publish the native current-write IDs before preparing access"
            )
        return True

    def write(self, layer, ids, *segments):
        from ulacache.storage.writers import write_native_record_segments

        if len(segments) == 1:
            # A complete NoPE record has no declared empty RoPE segment.
            # Keep the two-tensor native writer ABI inside this backend.
            segments = (segments[0], segments[0][..., :0])
        write_native_record_segments(self.pool, layer, ids, segments)

    def write_file_records(self, layer, ids, source):
        """Restore complete rows to their existing owner-local GPU/CPU slots.

        IDs use the same native interleaved namespace as alloc/write/access.
        Each owner consumes only its rows; a snapshot owner file therefore
        pairs with that stripe's global IDs, never a second local ID allocator.
        The caller keeps allocations and request publication unchanged until
        every owner's complete restore succeeds.
        """
        import torch
        from ulacache.cache.tp_cache import global_ids
        from ulacache.transfer.file_io import QuiescentRecordIO

        pool = self.pool
        if pool.flowkv_shared_index.in_forward:
            raise RuntimeError("Finish every TP reader before restoring file records")
        device = pool.flowkv_shards.device
        if (
            ids.dtype not in (torch.int32, torch.int64)
            or not ids.is_contiguous()
            or ids.device != device
        ):
            raise ValueError("File write IDs must be contiguous native device integers")
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("File writes must be outside CUDA Graph capture/replay")
        # Read metadata only. Payload buffers remain bounded by the existing
        # writer capacity, regardless of total file or request size.
        native_ids = ids.detach().to(device="cpu", dtype=torch.int64)
        if native_ids.numel() and (
            bool(torch.any(native_ids <= 0))
            or bool(torch.any(native_ids >= pool._flowkv_global_rows))
            or native_ids.unique().numel() != native_ids.numel()
        ):
            raise ValueError("File writes require unique allocated native logical IDs")
        degree = pool._flowkv_storage_size
        positions = torch.where(native_ids.remainder(degree) == pool.flowkv_rank)[0]
        owned_rows = (
            native_ids.index_select(0, positions)
            .div(degree, rounding_mode="floor")
            .tolist()
        )
        host = pool.flowkv_host

        def quiesce():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def invalidate():
            # Invalidate even on partial failure: an old resident copy must
            # never survive a write to its backing record.
            pool.flowkv_arena.invalidate(global_ids(pool, ids))
            pool._flowkv_prefill_layer = None

        chunk_rows = min(
            self.specification.capacity_plan.maximum_write_record_count,
            max(1, 4 * 1024 * 1024 // source.row_bytes),
        )
        with (
            source.open() as reader,
            QuiescentRecordIO(
                gpu_records=pool.flowkv_shards,
                cpu_records=host.host.storage if host is not None else None,
                layer_groups=tuple(
                    pool.flowkv_layout.layer_groups.group(i)
                    for i in range(pool.flowkv_layout.layers)
                ),
                owner_rows=owned_rows,
                directories=host.directory if host is not None else None,
                page_rows=pool.flowkv_layout.page_size,
                quiesce=quiesce,
                invalidate=invalidate,
            ) as writer,
        ):
            buffer = torch.empty(
                (min(chunk_rows, max(1, source.record_count)), source.row_bytes),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=device.type == "cuda",
            )
            owned_start = 0
            while (chunk := reader.readinto(buffer)) is not None:
                start, stop = chunk.start, chunk.start + chunk.data.shape[0]
                selected = positions[(positions >= start) & (positions < stop)] - start
                try:
                    if selected.numel():
                        data = (
                            chunk.data
                            if selected.numel() == chunk.data.shape[0]
                            else chunk.data.index_select(0, selected)
                        )
                        # The reused writer completes CPU-buffer consumption
                        # before returning; the bounded lease is then reusable.
                        writer.write(layer, owned_start, data)
                        owned_start += selected.numel()
                finally:
                    chunk.release()

    def prepare_native_prefill_read_plan(self, ids):
        from ulacache.cache.prefill.window import prepare_compact_prefill_record_read_plan

        return prepare_compact_prefill_record_read_plan(self.pool, ids)

    def read_native_prefill_records(
        self, layer, ids, native_reader, *, output, read_plan=None
    ):
        from ulacache.cache.prefill.window import run_compact_prefill_records

        return run_compact_prefill_records(
            self.pool,
            layer + self.pool.start_layer,
            ids,
            native_reader,
            output_buffer=output,
            read_plan=read_plan,
        )

    def attend_native_prefill_records(
        self, layer, query, ids, lengths, native_attention
    ):
        backend = getattr(self.pool, "_tp_full_prefill", None)
        if backend is not None and backend.plan is not None:
            window = backend.read(layer)
            result = native_attention(
                query, backend.plan.map_native_rows(ids), window, lengths
            )
            backend.after_attention(layer)
            return result
        from ulacache.cache.prefill.window import run_compact_prefill_attention

        return run_compact_prefill_attention(
            self.pool,
            layer + self.pool.start_layer,
            query,
            ids,
            lengths,
            native_attention,
        )

    @staticmethod
    def _storage_inventory(root, device_type):
        """Count backing storage once without inspecting tensor values."""
        seen, result = set(), {}

        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            storage_method = getattr(value, "untyped_storage", None)
            device = getattr(value, "device", None)
            if (
                callable(storage_method)
                and getattr(device, "type", None) == device_type
            ):
                storage = storage_method()
                if storage.nbytes():
                    result[storage.data_ptr()] = storage.nbytes()
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)
            elif (
                not callable(value)
                and not isinstance(value, ModuleType)
                and hasattr(value, "__dict__")
            ):
                for item in vars(value).values():
                    visit(item)

        visit(root)
        return result

    def device_storage_inventory(self):
        return self._storage_inventory(self.pool, "cuda")

    def host_storage_inventory(self):
        return self._storage_inventory(self.pool, "cpu")

    def memory_inventory(self):
        gpu_bytes = sum(self.device_storage_inventory().values())
        host_bytes = sum(self.host_storage_inventory().values())
        gpu_kv = self.pool.flowkv_shards.nbytes + sum(
            value.nbytes for value in self.gpu
        )
        host = self.pool.flowkv_host
        host_kv = host.host.nbytes if host is not None else 0
        return dict(
            gpu_bytes=gpu_bytes,
            gpu_kv_bytes=gpu_kv,
            gpu_metadata_bytes=gpu_bytes - gpu_kv,
            host_bytes=host_bytes,
            host_kv_bytes=host_kv,
            cpu_kv_bytes=host_kv,
            host_metadata_bytes=host_bytes - host_kv,
            gpu_cold_key_value_bytes=self.pool.flowkv_shards.nbytes,
            storage_owner_count=self.pool._flowkv_storage_size,
            storage_owner_rank=self.pool.flowkv_rank,
        )

    def close(self):
        if self.pool is not None and self.pool.flowkv_shared_index.in_forward:
            raise RuntimeError("Finish every TP reader before closing the cache")
        if self.pool is not None and self.owns_storage:
            from ulacache.cache.pool_lifecycle import close_pool

            close_pool(self.pool, step_publication=True)
        if (
            self.pool is not None
            and getattr(self.pool, "_ulacache_public_engine", None) is self
        ):
            del self.pool._ulacache_public_engine
        self.pool = None
        self.gpu = ()
        self.active = None


class TensorParallelAccessEngine:
    """Use the original lifecycle as the only forward/reader state machine."""

    no_current_records = None

    def __init__(self, backend, record_planes):
        self.backend = backend
        self.record_planes = record_planes

    def require_idle(self, operation):
        if self.backend.pool.flowkv_shared_index.in_forward:
            raise RuntimeError("Finish every TP reader before " + operation)

    def reset(self):
        self.require_idle("resetting the cache")

    def begin_forward(self, context=None):
        self.require_idle("beginning another forward")
        if context is not None:
            if isinstance(context.active_query_count, int):
                self.backend.active.fill_(context.active_query_count)
            else:
                self.backend.active.copy_(context.active_query_count)
        self.backend.pool.flowkv_shared_index.begin_forward()

    def prepare_selection(self, layer, plan):
        pool = self.backend.pool
        current = pool._flowkv_current_native
        if (
            current is None
            or current.data_ptr() != plan.current_logical_record_ids.data_ptr()
        ):
            raise ValueError(
                "Access planning must use the published native current-write IDs"
            )
        pool.flowkv_shared_index.on_index_ready(layer, plan.selected_logical_record_ids)

    @staticmethod
    def _signature(ids):
        return (
            ids.data_ptr(),
            tuple(ids.shape),
            tuple(ids.stride()),
            ids.dtype,
            ids.device,
        )

    def access(self, ids, layer):
        pool = self.backend.pool
        layout, lifecycle = pool.flowkv_layout, pool.flowkv_shared_index
        if type(layer) is not int or not 0 <= layer < layout.layers:
            raise ValueError("Layer outside the configured TP cache")
        if not lifecycle.in_forward:
            raise RuntimeError("Call begin_model_forward before access")
        if layer in lifecycle._consumed:
            raise RuntimeError(
                "Begin a new forward before reusing a finished access layer"
            )
        if (
            len(ids.shape) != 2
            or ids.shape[1] != layout.topk
            or ids.shape[0] > layout.max_query_tokens
        ):
            raise ValueError(
                "Access IDs exceed the configured native TP selection bounds"
            )
        group = lifecycle.groups.group(layer)
        if group not in lifecycle._planned:
            if layer != lifecycle.groups.anchors[group]:
                raise RuntimeError("Access the selection producer before its followers")
            current = pool._flowkv_current_native
            if current is None:
                raise RuntimeError(
                    "Bind the native current-write boundary before TP access"
                )
            self.prepare_selection(layer, SparseAttentionAccessPlan(ids, current))
        if self._signature(ids) != self._signature(lifecycle._keepalive[group][0]):
            raise ValueError("A TP group must reuse its producer selection tensor")
        lifecycle.before_history_consume(layer)
        pool._flowkv_mode, pool._flowkv_decode_layer = "decode", layer
        return self.resident_inputs(layer, ids.shape[0])

    def resident_inputs(self, layer, query_count):
        return self.create_view(
            layer, self.backend.pool.flowkv_arena.groups[layer].remap[:query_count]
        )

    def create_view(self, layer, rows):
        layout = self.backend.specification.storage_layout
        return ResidentAttentionInputs(
            self.record_planes[layer],
            rows,
            layout.records_per_page,
            layout.record_planes[0].page_memory_order,
        )

    def finish_layer(self, layer):
        lifecycle = self.backend.pool.flowkv_shared_index
        if not lifecycle.in_forward:
            raise RuntimeError("The TP forward has already finished")
        lifecycle.after_attention(layer)
        if layer + 1 == lifecycle.groups.layer_count:
            lifecycle.finish_forward()

    def detach(self):
        self.backend = None
        self.record_planes = ()


def adopt_native_pool(pool, *, native_kv_format):
    """Adopt actual capacities and shared selections without another allocation."""
    from ulacache import (
        CacheExecutionOptions,
        KeyValueRecordPlane,
        KeyValueStorageLayout,
        LayerReuseSpecification,
        MemoryTierCapacity,
        PrefetchPolicy,
        SparseAccessSpecification,
        TensorParallelCacheBindings,
        UnifiedCacheSpecification,
        init,
    )
    from ulacache.storage.layout import NativeMLARecord

    if getattr(pool, "ulacache", None) is not None:
        raise RuntimeError("The native TP pool already has a public cache handle")
    layout, groups = pool.flowkv_layout, pool.flowkv_layout.layer_groups
    owners = tuple(
        groups.anchors[groups.group(layer)] for layer in range(layout.layers)
    )
    record = NativeMLARecord(native_kv_format)
    if record.row_bytes != layout.row_bytes:
        raise ValueError(
            "The public native encoding differs from the allocated TP pool"
        )
    host_slots = pool.flowkv_host.host_slots if pool.flowkv_host is not None else 0
    allocation_page = int(pool.allocation_page_size)
    if allocation_page % pool._flowkv_storage_size:
        raise ValueError(
            "Native allocator pages must reserve equal rows on every owner"
        )
    spec = UnifiedCacheSpecification(
        storage_layout=KeyValueStorageLayout(
            "native-mla-records",
            (KeyValueRecordPlane("mla", record.native_encoding, record.segment_bytes),),
            records_per_page=layout.page_size,
        ),
        layer_reuse=LayerReuseSpecification.from_shared_sparse_selection(owners),
        sparse_access=SparseAccessSpecification(layout.max_query_tokens, layout.topk),
        capacity_plan=MemoryTierCapacity(
            device_record_capacity=int(pool.flowkv_shards.shape[1]),
            host_record_capacity=int(host_slots),
            maximum_write_record_count=layout.max_staging_tokens,
            reserved_logical_record_count=allocation_page // pool._flowkv_storage_size,
            maximum_prefill_union_record_count=layout.prefill_window_rows,
            prefill_workspace_record_capacity=layout.prefill_window_rows,
            maximum_residency_input_record_count=layout.max_query_tokens
            * (layout.topk + 1),
        ),
    )
    policy = TensorParallelCacheBindings(
        pool=pool,
        owns_storage=False,
        logical_slot_allocator=None,
        execution_options=CacheExecutionOptions(
            record_storage_backend="tensor_parallel",
            prefetch=PrefetchPolicy(
                enabled=pool.flowkv_shared_index.prefetch,
                completion=pool.flowkv_shared_index.completion,
            ),
        ),
    )
    pool.ulacache = init(spec, policy)


def canonical_selection(ids, width):
    """Keep native QxK and Qx1xK selections as the same borrowed 2D view."""
    shape = tuple(ids.shape)
    if len(shape) == 2 and shape[1] == width:
        return ids
    if len(shape) == 3 and shape[1:] == (1, width):
        view = ids.view(shape[0], width)
        if view.data_ptr() != ids.data_ptr():
            raise ValueError(
                "Canonical selection must retain the native backing pointer"
            )
        return view
    raise ValueError("Native TP selections require QxK or Qx1xK IDs")
