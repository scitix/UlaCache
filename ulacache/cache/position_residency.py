"""Local CPU/GPU residency under the native round-robin slot namespace.

There is no peer KV export, replicated attention bank, or new communicator.
The caller merges native partial attention outputs through its existing DCP
backend. Layer sharing reuses selections, never another layer's KV payload.
"""

from ulacache.storage.layout import PositionShardLayout


class PositionShardedResidency:
    def __init__(self, local_engine, layout: PositionShardLayout):
        import torch

        if (local_engine.gpu_slots, local_engine.host_slots) != (
            layout.device_records,
            layout.host_records,
        ):
            raise ValueError(
                "Position ownership must cover the actual exclusive CPU/GPU tiers"
            )
        self.local_engine = local_engine
        self.layout = layout
        self.groups = local_engine.groups
        self.gpu = local_engine.gpu
        self.host = local_engine.host
        self.directory = local_engine.directory
        self.gpu_slots, self.host_slots = layout.device_records, layout.host_records
        self.logical_slots = layout.global_records
        self.row_bytes = local_engine.row_bytes
        self.active = local_engine.active
        self.max_queries, self.topk = local_engine.max_queries, local_engine.topk
        self.prefill_exchange_launch = local_engine.prefill_exchange_launch
        options = dict(device=self.gpu.device, dtype=torch.int32)
        self.selected = torch.empty(
            (self.groups.group_count, self.max_queries, self.topk), **options
        )
        self.selected_counts = torch.empty(
            (self.groups.group_count, self.max_queries), **options
        )
        self.current = torch.empty(self.max_queries, **options)
        self._current_signature = None
        # Registration/release are eager allocator operations; decode uses
        # the preallocated selection/current buffers and allocates nothing.
        self.write_rows = torch.empty(
            max(self.max_queries, self.max_queries * self.topk), **options
        )

    def _map(self, rows, output):
        from ulacache.cache.kernels.owner_lookup import map_owner_records

        map_owner_records(
            rows,
            output,
            owner_rank=self.layout.rank,
            owner_count=self.layout.degree,
            owner_logical_capacity=self.layout.local_records,
        )
        return output

    def _allocator_rows(self, rows):
        import torch

        if rows.numel() > self.write_rows.numel():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Allocator mapping cannot grow during a captured forward"
                )
            return self._map(
                rows, torch.empty(rows.shape, device=rows.device, dtype=torch.int32)
            )
        return self._map(rows, self.write_rows[: rows.numel()].view(rows.shape))

    def allocate_unbound(self, rows, *, prefer_gpu=True):
        self.local_engine.allocate_unbound(
            self._allocator_rows(rows), prefer_gpu=prefer_gpu
        )

    def release(self, rows):
        self.local_engine.release(self._allocator_rows(rows))

    def clear(self):
        self.local_engine.clear()

    def begin_forward(self):
        self._current_signature = None
        self.local_engine.begin_forward()

    def prepare_forward_current_records(self, rows):
        return self.local_engine.prepare_forward_current_records(
            self._map_current(rows)
        )

    @staticmethod
    def _signature(rows):
        return rows.data_ptr(), rows.numel(), rows.dtype

    def _map_current(self, rows):
        signature = self._signature(rows)
        if self._current_signature != signature:
            self._map(rows, self.current[: rows.numel()])
            self._current_signature = signature
        return self.current[: rows.numel()]

    def begin_prefill(self, rows):
        self._current_signature = None
        # Prefill chunks can exceed the decode Graph query bound. They use
        # the eager writer workspace instead of the captured current buffer.
        if rows.numel() <= self.max_queries:
            self._map_current(rows)

    def plan(self, layer, selected, current):
        group = self.groups.group(layer)
        if selected.shape[0] > self.max_queries or selected.shape[1] != self.topk:
            raise ValueError("CP selection exceeds the fixed native query/TopK bounds")
        if current.numel() != selected.shape[0]:
            raise ValueError(
                "Current records must preserve the native padded query order"
            )
        from .kernels.position import localize_selection

        local = self.selected[group, : selected.shape[0]]
        localize_selection(
            selected,
            local,
            self.selected_counts[group, : selected.shape[0]],
            layout=self.layout,
            active_queries=self.active,
        )
        current = self._map_current(current)
        self.local_engine.plan(layer, local, current)

    def before_layer_prepare(self, layer):
        self.local_engine.before_layer_prepare(layer)

    def attention_rows(self, layer, query_count):
        # These are already LOCAL PHYSICAL rows. A framework must not apply
        # slot % CP / slot // CP to this result for a second time.
        return self.local_engine.attention_rows(layer, query_count)

    def write(self, layer, rows, nope, rope):
        mapped = (
            self._map_current(rows)
            if rows.numel() <= self.max_queries
            else self._allocator_rows(rows)
        )
        self.local_engine.write(layer, mapped, nope, rope)

    def after_attention(self, layer):
        self.local_engine.after_attention(layer)

    def finish_forward(self):
        self.local_engine.finish_forward()
        self._current_signature = None

    def _drain_transfers(self):
        self.local_engine._drain_transfers()

    def device_storage_inventory(self):
        inventory = dict(self.local_engine.device_storage_inventory())
        for tensor in (
            self.selected,
            self.selected_counts,
            self.current,
            self.write_rows,
        ):
            storage = tensor.untyped_storage()
            inventory[storage.data_ptr()] = storage.nbytes()
        return inventory

    def memory_inventory(self):
        value = dict(self.local_engine.memory_inventory())
        extra = sum(
            x.nbytes
            for x in (
                self.selected,
                self.selected_counts,
                self.current,
                self.write_rows,
            )
        )
        value["gpu_bytes"] += extra
        value["gpu_metadata_bytes"] += extra
        value.update(
            global_logical_record_capacity=self.layout.global_records,
            global_usable_logical_record_capacity=self.layout.usable_records,
            storage_owner_rank=self.layout.rank,
            storage_owner_count=self.layout.degree,
            record_distribution="position_modulo_cp",
            peer_KV_read_bytes=0,
            replicated_decode_attention_payload_bytes=0,
            additional_mapping_bytes=extra,
        )
        return value

    def host_storage_inventory(self):
        return self.local_engine.host_storage_inventory()

    def allocate_receive_rows(self, rows):
        raise NotImplementedError(
            "Position CP remote receive needs an owner-aware transport binding"
        )

    def plan_prefill(self, layer, rows):
        raise NotImplementedError(
            "Position CP prefill uses the shared two-window materializer"
        )

    def close(self):
        self.local_engine.close()
        self.gpu = self.host = self.directory = None
        self.selected = self.selected_counts = self.current = self.write_rows = None
