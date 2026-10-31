"""Bounded ordinary-TP prefill reconstruction over physical native TopK rows.

Framework adapters supply the native attention callback after TopK has been
translated to physical native row IDs. Decode and target verification retain
their existing cache path. This eager prefill path is not graph captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompactPrefillRecordWindow:
    native_record_ids: Any
    output_record_indices: Any
    staging_record_indices: Any


@dataclass(frozen=True)
class CompactPrefillRecordReadPlan:
    """Reuse metadata while history order and current write IDs stay unchanged.

    The plan owns index tensors only. Each layer still prepares its own
    residency and reads its own payload before the next window can replace it.
    """

    windows: tuple[CompactPrefillRecordWindow, ...]


def _prepare_compact_prefill_record_window(pool, native_rows, row_bound):
    import torch

    lengths = torch.full(
        (1,), native_rows.numel(), dtype=torch.int32, device=native_rows.device
    )
    needed, remapped, fresh = compact_topk_rows(
        native_rows.reshape(1, -1), lengths, pool._flowkv_current_rows
    )
    joined = _shared_cpu_union(pool, needed, remapped, fresh, row_bound)
    if joined is None:
        raise ValueError("Dense TP prefix union exceeds its bounded window")
    needed, remapped, fresh = joined
    if not bool((remapped >= 0).all().item()):
        raise ValueError("Dense prefix rows must be valid physical native slots")
    return CompactPrefillRecordWindow(needed, remapped.reshape(-1), fresh)


def _compact_prefill_record_read_inputs(pool, native_rows):
    import torch

    layout = pool.flowkv_layout
    if (
        pool._flowkv_mode != "prefill"
        or not layout.prefill_window_rows
        or layout.compression_ratio != 1
    ):
        raise RuntimeError("Dense compact reads require bounded Plain prefill")
    if pool._flowkv_namespaces != 1 or pool.flowkv_request is not None:
        raise NotImplementedError("Dense compact reads require shared ordinary TP")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Dense compact reads require eager prefill")
    rows = native_rows.reshape(-1).to(dtype=torch.int32).contiguous()
    if not rows.numel():
        raise ValueError("Dense compact prefix reads require nonempty native rows")
    row_bound = layout.prefill_window_rows
    if pool.flowkv_host is not None:
        row_bound = min(row_bound, pool.flowkv_host.compact_prefill_chunk_rows)
    return rows, row_bound


def prepare_compact_prefill_record_read_plan(pool, native_rows):
    """Prepare the shared history metadata once for a homogeneous native tier."""
    rows, row_bound = _compact_prefill_record_read_inputs(pool, native_rows)
    windows = []
    for part in rows.split(row_bound):
        window = _prepare_compact_prefill_record_window(pool, part, row_bound)
        # Own compact allocations rather than retaining a collective/sort
        # workspace whose backing storage may exceed its visible tensor.
        windows.append(
            CompactPrefillRecordWindow(
                window.native_record_ids.clone(),
                window.output_record_indices.clone(),
                window.staging_record_indices.clone(),
            )
        )
        del window
    return CompactPrefillRecordReadPlan(tuple(windows))


def _local_layer(pool, layer_id):
    return layer_id - getattr(pool, "start_layer", 0)


def invalidate_current_prefill_rows(pool, layer_id):
    """Drop only aliases of newly written rows from this layer's cache group."""
    from ulacache.cache.tp_cache import global_ids

    local = _local_layer(pool, layer_id)
    arena = pool.flowkv_arena
    arena.kernel().invalidate(
        global_ids(pool, pool._flowkv_current_rows),
        1,
        pool._flowkv_global_rows,
        arena.groups[local].hints,
    )


def compact_topk_rows(native_rows, seq_lens, current_rows):
    """Return sorted complete union, order-preserving remap, staging positions.

    Native zero and negative rows are sentinels. Compact row zero is valid.
    The fresh representative wins when current and historical slots coincide.
    """
    import torch

    width = native_rows.shape[1]
    live = (native_rows > 0) & (
        torch.arange(width, device=native_rows.device)[None, :]
        < seq_lens.reshape(-1, 1)
    )
    needed = torch.unique(native_rows[live].to(torch.int32), sorted=True)
    remapped = torch.searchsorted(needed, native_rows.contiguous(), out_int32=True)
    remapped = torch.where(live, remapped, -1).to(torch.int32).contiguous()
    if current_rows.numel():
        sorted_current, original_positions = torch.sort(current_rows)
        positions = torch.searchsorted(sorted_current, needed)
        safe = positions.clamp(max=sorted_current.numel() - 1)
        fresh = (
            torch.where(
                (positions < sorted_current.numel()) & (sorted_current[safe] == needed),
                original_positions[safe],
                -1,
            )
            .to(torch.int32)
            .contiguous()
        )
    else:
        fresh = torch.full_like(needed, -1)
    return needed, remapped, fresh


def _shared_cpu_union(pool, needed, remapped, fresh, row_bound):
    """Use one owner residency plan and split decision across TP readers."""
    import torch
    import torch.distributed as dist

    degree = pool._flowkv_storage_size
    if pool.flowkv_host is None or degree == 1:
        return needed, remapped, fresh
    group = getattr(pool.udcache_services.group, "device_group", None)
    if group is None:
        raise RuntimeError("CPU compact prefill requires a bound TP device group")
    largest = torch.tensor([needed.numel()], dtype=torch.int64, device=needed.device)
    dist.all_reduce(largest, op=dist.ReduceOp.MAX, group=group)
    width = int(largest.item())
    if width > row_bound:
        return None
    if not width:
        return needed, remapped, fresh
    send = torch.full((width,), -1, dtype=torch.int32, device=needed.device)
    send[: needed.numel()] = needed
    received = torch.empty(degree * width, dtype=torch.int32, device=needed.device)
    dist.all_gather_into_tensor(received, send, group=group)
    shared = torch.unique(received[received > 0], sorted=True)
    if shared.numel() > row_bound:
        return None
    mapping = torch.searchsorted(shared, needed, out_int32=True)
    safe = remapped.clamp(min=0)
    if needed.numel():
        remapped = torch.where(remapped >= 0, mapping[safe], -1).contiguous()
    current = pool._flowkv_current_rows
    if current.numel():
        sorted_current, positions = torch.sort(current)
        at = torch.searchsorted(sorted_current, shared)
        safe = at.clamp(max=sorted_current.numel() - 1)
        fresh = (
            torch.where(
                (at < sorted_current.numel()) & (sorted_current[safe] == shared),
                positions[safe],
                -1,
            )
            .to(torch.int32)
            .contiguous()
        )
    else:
        fresh = torch.full_like(shared, -1)
    return shared, remapped, fresh


def run_compact_prefill_records(
    pool, layer_id, native_rows, consume, *, output_buffer=None, read_plan=None
):
    """Materialize an ordered native dense-prefix read through bounded windows.

    ``consume(paged_uint8_window, remapped_native_rows)`` returns an independent
    tensor, such as native dequantized latent KV. When the native reader supplies
    ``output_buffer``, call ``consume(window, indices, destination_slice)`` so
    it can write directly into that buffer. Other callers retain concatenated output.
    The adapter retains its original conversion/up-projection/attention.
    """
    import torch

    rows, row_bound = _compact_prefill_record_read_inputs(pool, native_rows)
    arena = pool.flowkv_arena
    host = pool.flowkv_host
    local = _local_layer(pool, layer_id)
    outputs = []
    output_offset = 0
    windows = (
        read_plan.windows
        if read_plan is not None
        else (
            _prepare_compact_prefill_record_window(pool, part, row_bound)
            for part in rows.split(row_bound)
        )
    )
    for window in windows:
        needed = window.native_record_ids
        remapped = window.output_record_indices
        fresh = window.staging_record_indices
        if host is not None:
            host.prepare_compact_prefill_group(local, needed)
        arena.kernel().fill_compact(
            pool.flowkv_peers.device_ptr,
            needed,
            arena.staging,
            fresh,
            arena.prefill,
            pool._flowkv_peer_offset(local),
            pool._flowkv_storage_size,
            pool.flowkv_rank,
            pool._flowkv_owner_start,
            False,
            *((host.descriptors[local],) if host is not None else ()),
        )
        if host is not None:
            host.finish_prefill_group(local)
        if output_buffer is None:
            outputs.append(consume(arena.prefill, remapped))
        else:
            consume(
                arena.prefill,
                remapped,
                output_buffer.narrow(0, output_offset, remapped.numel()),
            )
            output_offset += remapped.numel()
    if output_buffer is not None:
        return output_buffer
    return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]


def run_compact_prefill_attention(pool, layer_id, query, native_rows, seq_lens, attend):
    """Call native attention on complete query groups sharing a fixed KV window.

    ``attend(query_slice, compact_indices, paged_uint8_window, lengths_slice)``
    must construct native attention metadata for that slice and return a tensor
    whose leading dimension is its query count. The window has the original
    page geometry; the callback owns the model's typed/view conversion.
    """
    import torch

    layout, arena = pool.flowkv_layout, pool.flowkv_arena
    if not layout.prefill_window_rows or layout.compression_ratio != 1:
        raise ValueError("Compact Plain prefill was not budgeted at startup")
    if pool._flowkv_mode != "prefill":
        raise RuntimeError("Compact prefill cannot consume decode or verify state")
    if pool._flowkv_namespaces != 1 or pool.flowkv_request is not None:
        raise NotImplementedError("Compact prefill requires shared ordinary TP")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Compact prefill union/remap requires eager execution")
    if (
        native_rows.ndim != 2
        or native_rows.dtype != torch.int32
        or native_rows.device != query.device
        or seq_lens.numel() != native_rows.shape[0]
        or query.shape[0] != native_rows.shape[0]
    ):
        raise ValueError("Native attention query, physical TopK and lengths disagree")
    if native_rows.shape[1] != layout.topk:
        raise ValueError("Native TopK width disagrees with the bounded window budget")
    if not query.shape[0]:
        return query
    local = _local_layer(pool, layer_id)
    host = pool.flowkv_host
    row_bound = layout.prefill_window_rows
    if host is not None:
        row_bound = min(row_bound, host.compact_prefill_chunk_rows)
    if row_bound < layout.topk:
        raise ValueError("The GPU residency workspace cannot hold one complete TopK")
    group_queries = min(
        layout.max_staging_tokens, 8 * layout.prefill_window_rows // layout.topk
    )
    if query.shape[0] > layout.max_staging_tokens:
        raise ValueError(
            "Prefill query count exceeds its startup staging/workspace bound"
        )

    def groups(start, stop):
        lengths = seq_lens.reshape(-1)[start:stop].contiguous()
        needed, remapped, fresh = compact_topk_rows(
            native_rows[start:stop],
            lengths,
            pool._flowkv_current_rows,
        )
        joined = _shared_cpu_union(pool, needed, remapped, fresh, row_bound)
        if joined is None or needed.numel() > row_bound:
            if stop - start == 1:
                raise RuntimeError("A complete TopK union exceeded its reserved window")
            middle = (start + stop) // 2
            del needed, remapped, fresh, lengths
            yield from groups(start, middle)
            yield from groups(middle, stop)
        else:
            needed, remapped, fresh = joined
            yield start, stop, needed, remapped, fresh, lengths

    outputs = []
    for start in range(0, query.shape[0], group_queries):
        stop = min(start + group_queries, query.shape[0])
        for first, last, needed, remapped, fresh, lengths in groups(start, stop):
            if host is not None and needed.numel():
                host.prepare_compact_prefill_group(local, needed)
            arena.kernel().fill_compact(
                pool.flowkv_peers.device_ptr,
                needed,
                arena.staging,
                fresh,
                arena.prefill,
                pool._flowkv_peer_offset(local),
                pool._flowkv_storage_size,
                pool.flowkv_rank,
                pool._flowkv_owner_start,
                False,
                *((host.descriptors[local],) if host is not None else ()),
            )
            if host is not None and needed.numel():
                host.finish_prefill_group(local)
            outputs.append(attend(query[first:last], remapped, arena.prefill, lengths))
    return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
