"""Staging and publication of already encoded native records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StagedNativeRecordWrite:
    logical_record_ids: object
    source_record_indices: object


class NativeRecordWriteStaging:
    """Bounded native writer output, with one bank per KV producer.

    ``write_native_record_pages`` is the original native writer bound to its
    layout. It receives native input values, destination pages and ordinal
    destination indices; optional arguments such as RoPE tables pass through.
    All staging, planning and publication use the same current CUDA stream.
    """

    def __init__(
        self, cache_specification, execution_plan, device, write_native_record_pages
    ):
        import torch

        self.execution_plan = execution_plan
        self.write_native_record_pages = write_native_record_pages
        self.maximum_write_record_count = (
            cache_specification.capacity_plan.maximum_write_record_count
        )
        layout = cache_specification.storage_layout
        page_count = (
            self.maximum_write_record_count + layout.records_per_page
        ) // layout.records_per_page
        self.native_record_plane_pages = tuple(
            torch.empty(
                (
                    len(execution_plan.storage_producer_layers),
                    page_count,
                    plane.bytes_per_page(layout.records_per_page),
                ),
                dtype=torch.uint8,
                device=device,
            )
            for plane in layout.record_planes
        )
        self.native_record_pages = self.native_record_plane_pages[0]
        # Native fused writers reserve index zero as their padding destination.
        self.source_record_indices = torch.arange(
            1, self.maximum_write_record_count + 1, dtype=torch.int64, device=device
        )
        self._pending_writes = {}

    def stage_native_records(
        self,
        cache_layer_index,
        logical_record_ids,
        native_values,
        **native_writer_arguments,
    ):
        pages, source_indices = self.reserve_native_write(
            cache_layer_index, logical_record_ids
        )
        self.write_native_record_pages(
            native_values, pages, source_indices, **native_writer_arguments
        )

    def reserve_native_write(self, cache_layer_index, logical_record_ids):
        """Also lets an original fused compressor write directly into these pages."""
        bank = self.execution_plan.storage_bank_indices_by_layer[cache_layer_index]
        if bank in self._pending_writes:
            raise RuntimeError(
                "Publish the previous native write before reusing its staging pages"
            )
        count = logical_record_ids.numel()
        if count > self.maximum_write_record_count:
            raise ValueError(
                "Native quantizer output exceeds its startup staging capacity"
            )
        source_indices = self.source_record_indices[:count]
        self._pending_writes[bank] = StagedNativeRecordWrite(
            logical_record_ids, source_indices
        )
        pages = self._record_planes_for_bank(bank)
        return pages, source_indices

    def _record_planes_for_bank(self, bank):
        if len(self.native_record_plane_pages) == 1:
            return self.native_record_pages[bank]
        return tuple(pages[bank] for pages in self.native_record_plane_pages)

    def pending_logical_record_ids(self, cache_layer_index):
        bank = self.execution_plan.storage_bank_indices_by_layer[cache_layer_index]
        pending = self._pending_writes.get(bank)
        return pending.logical_record_ids if pending is not None else None

    def publish_staged_records(
        self, memory_cache, cache_layer_index, *, local_device_after_commit=False
    ):
        """After decode planning, or before eager CPU-first prefill planning."""
        bank = self.execution_plan.storage_bank_indices_by_layer[cache_layer_index]
        pending = self._pending_writes[bank]
        arguments = (
            cache_layer_index,
            pending.logical_record_ids,
            self._record_planes_for_bank(bank),
            pending.source_record_indices,
        )
        if local_device_after_commit:
            memory_cache.write_native_paged_records(
                *arguments, local_device_after_commit=True
            )
        else:
            memory_cache.write_native_paged_records(*arguments)
        del self._pending_writes[bank]

    def get_device_storage_inventory(self):
        return {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in (*self.native_record_plane_pages, self.source_record_indices)
            if tensor.numel()
        }


def write_native_record_segments(pool, local_layer, logical_ids, segments):
    """Write one native batch after the adapter's quantize/cast operation.

    ``segments`` contains the original latent and RoPE tensors, each shaped
    ``[rows, width]`` or ``[rows, 1, width]``. Scaled FP8 MLA supplies uint8
    regions528/128; ordinary FP8 and BF16 MLA retain their original typed
    regions. All bytes must cover the pool's single contiguous native record.
    Page-segmented or independently quantized formats require another writer.

    Forward preparation, native allocation and physical-location validation
    remain caller responsibilities. This function preserves the native mode
    and chunk bounds and updates current-row/prefill state only after writing.
    """
    import torch

    layout = pool.flowkv_layout
    if type(local_layer) is not int or not 0 <= local_layer < layout.layers:
        raise ValueError("Native MLA writer layer is outside the local pool")
    if layout.native_segment_bytes != (layout.row_bytes,):
        raise NotImplementedError(
            "Ordinary TP writes require one contiguous native record"
        )
    if len(segments) != 2:
        raise ValueError("Native MLA writing requires latent and RoPE segments")
    latent, rope = segments
    if (
        logical_ids.ndim != 1
        or logical_ids.dtype not in (torch.int32, torch.int64)
        or not logical_ids.is_contiguous()
    ):
        raise ValueError("Native MLA write IDs must be contiguous int32 or int64")
    count = logical_ids.numel()
    if count > layout.max_staging_tokens:
        raise ValueError("Native MLA write exceeds its configured chunk capacity")
    if count > pool.flowkv_staging_target_rows.numel():
        raise ValueError("Native MLA write exceeds its allocated staging rows")
    element_bytes = latent.element_size()
    if latent.dtype not in (torch.uint8, pool.dtype) or element_bytes not in (1, 2):
        raise ValueError(
            "Native MLA segments must already have the pool's encoded dtype"
        )
    for segment in segments:
        if (
            segment.ndim not in (2, 3)
            or (segment.ndim == 3 and segment.shape[1] != 1)
            or segment.shape[0] != count
            or segment.device != logical_ids.device
            or segment.shape[-1] * element_bytes % 8
        ):
            raise ValueError(
                "Native MLA segment shape, device or byte alignment disagrees"
            )
    if rope.shape[-1] and rope.dtype != latent.dtype:
        raise ValueError("Native MLA segments must have the same encoded dtype")
    if (latent.shape[-1] + rope.shape[-1]) * element_bytes != layout.row_bytes:
        raise ValueError("Native MLA segments do not cover the complete record")

    arena, group = pool.flowkv_arena, pool.flowkv_arena.groups[local_layer]
    decode = pool._flowkv_mode == "decode"
    if not decode and pool._flowkv_mode != "prefill":
        raise RuntimeError("Native MLA writer requires a forward lifecycle")
    compact_prefill = not decode and pool.flowkv_plain_compact_prefill
    resident = (
        arena.storage[local_layer]
        if decode
        else arena.staging
        if compact_prefill
        else arena.prefill
    )
    if resident.device != logical_ids.device:
        raise ValueError("Native MLA writer inputs must use the resident device")
    current_rows = pool.flowkv_staging_target_rows[:count]
    writer = (
        arena.kernel().write_decode
        if decode
        else arena.kernel().write_prefill_staged
        if compact_prefill
        else arena.kernel().write_prefill
    )
    host_args = (
        (pool.flowkv_host.descriptors[local_layer],)
        if pool.flowkv_host is not None
        else ()
    )
    if pool.flowkv_request is not None:
        host_args = (pool.flowkv_request.page_map,)
    writer(
        latent,
        rope,
        logical_ids,
        pool.flowkv_peers.device_ptr,
        resident,
        current_rows,
        pool._flowkv_owner_start,
        pool._flowkv_peer_offset(local_layer),
        pool.flowkv_rank,
        pool._flowkv_namespaces > 1,
        layout.capacity,
        group.cursors,
        group.totals,
        arena.active,
        *host_args,
    )
    pool._flowkv_current_rows = current_rows
    if compact_prefill:
        invalidated = getattr(pool, "_ulacache_prefill_invalidated_rows", None)
        already_invalidated = (
            invalidated is not None
            and invalidated.device == logical_ids.device
            and invalidated.dtype == logical_ids.dtype
            and invalidated.numel() == count
            and invalidated.data_ptr() == logical_ids.data_ptr()
        )
        if not already_invalidated:
            from ulacache.cache.prefill.window import invalidate_current_prefill_rows

            invalidate_current_prefill_rows(pool, pool.start_layer + local_layer)
    pool._flowkv_prefill_layer = None
