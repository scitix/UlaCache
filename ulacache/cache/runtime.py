"""Paper runtime components over shared native pools and directory state.

SlotManager owns logical lifetime. AccessEngine owns selection planning and
borrowed views. Both use the same backend; they do not duplicate directories,
payload storage, PB-LRU state, transfer workspaces or prefetch streams.
"""

from ulacache.bindings import RecordOwnerBindings
from ulacache.cache.access import ResidentView, SparseAttentionAccessPlan


class SlotManager:
    """ID allocation/recycling and lifetime updates to the native directory."""

    def __init__(self, backend, logical_allocator, access_engine):
        self.backend = backend
        self.logical_allocator = logical_allocator
        self.access_engine = access_engine

    def alloc(self, n):
        self.access_engine.require_idle("allocating logical slots")
        if type(n) is not int or n < 0:
            raise ValueError("The logical slot count must be a nonnegative integer")
        if self.logical_allocator is None:
            raise RuntimeError("Bind logical_slot_allocator to use alloc(n)")
        ids = self.logical_allocator.alloc(n)
        if ids is None:
            raise MemoryError("The logical slot allocator has insufficient capacity")
        self.register(ids)
        return ids

    def register(self, ids, *, prefer_device=True):
        self.backend.allocate_unbound(ids, prefer_gpu=prefer_device)

    def release(self, ids):
        self.access_engine.require_idle("releasing logical slots")
        self.backend.release(ids)

    def free(self, ids):
        self.access_engine.require_idle("freeing logical slots")
        self.release(ids)
        self.access_engine.reset()
        if self.logical_allocator is not None:
            self.logical_allocator.free(ids)

    def clear(self):
        self.access_engine.require_idle("clearing logical slots")
        self.backend.clear()
        self.access_engine.reset()

    def detach(self):
        self.backend = self.logical_allocator = self.access_engine = None


class AccessEngine:
    """Frozen layer sharing, existing residency/prefetch and borrowed GPU views."""

    def __init__(self, specification, bindings, backend, record_planes):
        self.specification = specification
        self.layer_sharing = specification.layer_reuse
        self.execution_options = bindings.execution_options
        self.backend = backend
        self.record_planes = record_planes
        self.producers = self.layer_sharing.sparse_selection_owners
        if isinstance(bindings, RecordOwnerBindings):
            from ulacache.cache.planning import create_owner_paged_execution_plan

            plan = create_owner_paged_execution_plan(specification)
            self.producers = tuple(
                plan.selection_producer_layers[index]
                for index in plan.selection_indices_by_layer
            )
        self.no_current_records = None
        self.plans = {}
        self.live_producers = set()
        self.finished_layers = set()
        self.forward_started = False

    def reset(self):
        self.plans.clear()
        self.forward_started = False
        self.live_producers.clear()
        self.finished_layers.clear()

    def require_idle(self, operation):
        if self.live_producers:
            raise RuntimeError(f"Finish every access reader before {operation}")

    def begin_forward(self, context=None):
        self.require_idle("beginning another forward")
        if context is not None:
            active = context.active_query_count
            if isinstance(active, int):
                self.backend.active.fill_(active)
            else:
                self.backend.active.copy_(active)
        self.backend.begin_forward()
        self.reset()
        self.forward_started = True

    @staticmethod
    def tensor_signature(ids):
        # Metadata only; never extend the lifetime of temporary native TopK.
        return (
            ids.data_ptr(),
            tuple(ids.shape),
            tuple(ids.stride()),
            ids.dtype,
            ids.device,
        )

    def prepare_selection(self, layer, plan):
        result = self.backend.plan(
            layer, plan.selected_logical_record_ids, plan.current_logical_record_ids
        )
        self.plans[layer] = self.tensor_signature(plan.selected_logical_record_ids)
        return result

    def access(self, ids, layer):
        if not self.forward_started:
            raise RuntimeError("Call begin_model_forward before access")
        if type(layer) is not int or not 0 <= layer < len(self.producers):
            raise ValueError("Layer outside the configured cache")
        if layer in self.finished_layers:
            raise RuntimeError(
                "Begin a new forward before reusing a finished access layer"
            )
        bounds = self.specification.sparse_access
        if (
            len(ids.shape) != 2
            or ids.shape[1] != bounds.maximum_selected_record_count
            or ids.shape[0] > bounds.maximum_padded_query_count
        ):
            raise ValueError(
                "Access IDs must have the configured padded query/selection shape"
            )
        producer = self.producers[layer]
        prepared = self.plans.get(producer)
        if prepared is None:
            if layer != producer:
                raise RuntimeError("Access the selection producer before its followers")
            if self.no_current_records is None:
                if ids.is_cuda:
                    import torch

                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "Warm up read-only access before Graph capture"
                        )
                self.no_current_records = ids.new_full(
                    (bounds.maximum_padded_query_count,), -1
                )
            self.prepare_selection(
                layer,
                SparseAttentionAccessPlan(ids, self.no_current_records[: ids.shape[0]]),
            )
        elif self.tensor_signature(ids) != prepared:
            raise ValueError(
                "Access must reuse the prepared selection tensors within a forward"
            )
        self.live_producers.add(producer)
        return self.resident_inputs(layer, ids.shape[0])

    def resident_inputs(self, layer, query_count):
        return self.create_view(layer, self.backend.attention_rows(layer, query_count))

    def create_view(self, layer, slot_ids):
        layout = self.specification.storage_layout
        return ResidentView(
            self.record_planes[layer],
            slot_ids,
            layout.records_per_page,
            layout.record_planes[0].page_memory_order,
        )

    def finish_layer(self, layer):
        self.backend.after_attention(layer)
        if self.live_producers:
            producer = self.producers[layer]
            self.finished_layers.add(layer)
            if producer not in self.producers[layer + 1 :]:
                self.live_producers.discard(producer)

    def detach(self):
        self.reset()
        self.backend = None
        self.record_planes = ()
        self.no_current_records = None
