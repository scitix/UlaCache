"""Interleaved complete records behind native request/page allocation.

Storage identities use owner namespaces and contiguous physical rank intervals.
Native page offsets select the token owner. The explicit native page descriptor
controls resident views; every cold slot contains one complete contiguous row.
"""

from dataclasses import replace

import torch


def join_local_residency_clear(pool):
    """Order an idle TP clear after its actual prefetch stream."""
    if (
        torch.device(pool.device).type == "cuda"
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError("TP residency cannot be cleared during CUDA Graph capture")
    lifecycle = getattr(pool, "flowkv_shared_index", None)
    if (
        lifecycle is None
        or lifecycle.pool is not pool
        or lifecycle.groups is not pool.flowkv_layout.layer_groups
    ):
        raise ValueError("Clear requires the actual native shared Index lifecycle")
    lifecycle.join_idle_prefetch()


def configure_pool(pool, services, *, step_publication=False):
    from ulacache.storage.kernels.records import flowkv_record_module
    from ulacache.cache.attention_buffers import FlowKVC4Arena
    from ulacache.transfer.ipc import FlowKVPeerShards

    native = pool.flowkv_layout
    memory = services.config
    compact_plain = bool(native.prefill_window_rows and native.compression_ratio == 1)
    if compact_plain:
        from ulacache.storage.workspace import PlainPrefillLayout

        if not isinstance(native, PlainPrefillLayout):
            raise ValueError(
                "Compact Plain prefill requires its explicit window/staging memory descriptor"
            )
    if getattr(memory, "low_query_pull_blocks", None) is not None and not compact_plain:
        raise NotImplementedError(
            "Low-query pull tuning requires native Plain prefetch storage"
        )
    record_factory = flowkv_record_module
    uses_policy_module = memory.local_eviction_policy != "circular" or compact_plain
    if uses_policy_module:
        from ulacache.eviction.kernels.local_lru import flowkv_record_policy_module

        record_factory = flowkv_record_policy_module
    pool.udcache_services = services
    degree, group, namespaces = services.degree, services.group, services.namespaces
    if step_publication and (namespaces != 1 or native.compression_ratio != 1):
        raise NotImplementedError(
            "Step publication requires one shared ordinary TP namespace"
        )
    if compact_plain and (
        namespaces != 1 or (memory.cpu_offload and memory.cpu_policy == "request")
    ):
        raise NotImplementedError(
            "Compact Plain prefill currently requires shared ordinary TP"
        )
    if memory.cpu_offload and not step_publication:
        raise NotImplementedError("The CPU tier requires ordinary MLA shared-global TP")
    global_rows = namespaces * native.pool_rows
    if native.pool_rows % degree or native.page_size % degree or global_rows >= 2**31:
        raise ValueError(
            "Record geometry needs divisible local capacity and int32 global IDs"
        )
    pool._flowkv_record_storage = True
    pool._flowkv_chunked_prefill = bool(native.prefill_window_rows)
    pool._flowkv_storage_size, pool._flowkv_namespaces = degree, namespaces
    pool.flowkv_rank = group.rank_in_group
    pool._flowkv_owner_start = (
        pool.flowkv_rank * native.pool_rows if namespaces > 1 else 0
    )
    pool._flowkv_global_rows = global_rows
    from ulacache.storage.layout import InterleavedRecordLayout

    pool.flowkv_slot_directory = InterleavedRecordLayout(
        degree, global_rows // degree, native.native_page, layers=native.layers
    )
    pool.kv_cache_total_dim = native.row_bytes
    pool.bytes_per_page_padded = native.storage_bytes_per_page
    # Preserve native-slot prefill scratch addressing. The CPU tier's expanded
    # logical row count is included in that scratch's explicit GPU budget.
    arena_layout = replace(native, pool_rows=global_rows, disaggregation_mode="decode")
    logical_per_rank = global_rows // degree
    if memory.cpu_offload:
        from ulacache.storage.capacity import flowkv_cpu_pool_geometry

        gpu_per_rank, cpu_slots = flowkv_cpu_pool_geometry(
            logical_per_rank,
            cpu_slots=memory.cpu_slots,
            cpu_ratio=memory.cpu_ratio,
            page_size=native.page_size,
        )
        if gpu_per_rank + cpu_slots != logical_per_rank:
            raise ValueError("FlowKV CPU ratio must match the sized logical pool")
    else:
        gpu_per_rank, cpu_slots = logical_per_rank, 0
    if gpu_per_rank < 1:
        raise ValueError("FlowKV CPU tier needs a nonempty GPU global pool")
    with services.memory_region():
        pool.flowkv_shards = torch.zeros(
            (native.layers, gpu_per_rank, native.row_bytes),
            dtype=torch.uint8,
            device=pool.device,
        )
        pool.kv_buffer = list(pool.flowkv_shards.unbind(0))
        pool.flowkv_arena = FlowKVC4Arena(
            arena_layout,
            device=pool.flowkv_shards.device,
            bytes_per_page=pool.bytes_per_page_padded,
            layer_groups=native.layer_groups,
        )
        pool.flowkv_arena.prefill = torch.zeros(
            (
                native.prefill_buffer_bytes // pool.bytes_per_page_padded,
                pool.bytes_per_page_padded,
            ),
            dtype=torch.uint8,
            device=pool.device,
        )
        if pool._flowkv_chunked_prefill:
            if compact_plain:
                pool.flowkv_arena.staging = torch.zeros(
                    (
                        native.staging_buffer_bytes // pool.bytes_per_page_padded,
                        pool.bytes_per_page_padded,
                    ),
                    dtype=torch.uint8,
                    device=pool.device,
                )
            elif services.window_initializer is None:
                raise ValueError(
                    "Bounded prefill requires an explicit native window adapter"
                )
            if services.window_initializer is not None:
                services.window_initializer(pool)
    pool.flowkv_peers = FlowKVPeerShards(
        pool.flowkv_shards,
        group.cpu_group,
        ipc_manager_factory=services.ipc_manager_factory,
    )
    pool.flowkv_host = None
    pool.flowkv_request = None
    request_policy = cpu_slots and memory.cpu_policy == "request"
    if request_policy:
        from ulacache.cache.request_storage import FlowKVRequestTier

        pool.flowkv_request = FlowKVRequestTier(
            pool,
            cpu_slots,
            host_pool_factory=services.host_pool_factory,
            host_memory_budget_bytes=services.host_memory_budget_bytes,
        )
    elif cpu_slots:
        from ulacache.cache.host import (
            SharedWorkspaceHostTier as FlowKVHostTier,
        )

        pool.flowkv_host = FlowKVHostTier(
            pool,
            cpu_slots,
            group.cpu_group,
            host_pool_factory=services.host_pool_factory,
            ipc_manager_factory=services.ipc_manager_factory,
            exchange_launch=(memory.host_exchange_blocks, memory.host_exchange_threads),
            parallel_finish=memory.host_parallel_finish,
            parallel_victims=memory.host_parallel_victims,
        )
    module = record_factory(
        topk=native.topk,
        max_queries=native.max_query_tokens,
        page_size=native.page_size,
        bytes_per_page=pool.bytes_per_page_padded,
        peers=degree,
        slots_per_rank=global_rows // degree,
        row_bytes=native.row_bytes,
        native_segment_bytes=native.native_segment_bytes,
        **(
            {
                "local_eviction_policy": memory.local_eviction_policy,
                "lru_histogram_blocks": memory.local_lru_histogram_blocks,
            }
            if uses_policy_module
            else {}
        ),
        **({"compact_prefill": True} if compact_plain else {}),
        step_publication=step_publication,
        host_tier=bool(cpu_slots) and not request_policy,
        **(
            {"request_tier": True, "gpu_slots_per_rank": gpu_per_rank}
            if request_policy
            else {}
        ),
    )
    pool.flowkv_arena._module = module
    if pool.flowkv_host is not None:
        pool.flowkv_host.reserve_native_padding(
            torch.arange(
                1, pool.allocation_page_size, dtype=torch.int32, device=pool.device
            )
        )
    if cpu_slots and not request_policy:
        # The imported completion flags use the same system-scope atomics as
        # ordinary TP step publication. Validate real mappings at startup.
        for ready, peers in (
            (pool.flowkv_host.ready, pool.flowkv_host.ready_peers),
            (pool.flowkv_host.miss_ready, pool.flowkv_host.miss_ready_peers),
        ):
            for pointer in peers.ptrs.cpu().tolist():
                module.validate_step_peer(ready, int(pointer))
    if step_publication:
        from ulacache.transfer.ipc import FlowKVStepPublication

        pool.flowkv_step_publication = FlowKVStepPublication(
            pool.flowkv_arena,
            pool.flowkv_arena.kernel(),
            group.cpu_group,
            ipc_manager_factory=services.ipc_manager_factory,
            host_groups=pool.flowkv_host.fused_metadata
            if pool.flowkv_host is not None
            else None,
        )
    if request_policy:
        from ulacache.cache.request_access import RequestAddressRuntime

        pool.flowkv_request.bind_runtime(RequestAddressRuntime(pool))
    packed_rows = native.pool_rows if native.compression_ratio > 1 else 0
    pool.flowkv_staging_positions = torch.arange(
        packed_rows, dtype=torch.int64, device=pool.device
    )
    pool.flowkv_staging_rows = torch.arange(
        packed_rows, dtype=torch.int32, device=pool.device
    )
    pool.flowkv_staging_target_rows = torch.empty(
        packed_rows or native.max_staging_tokens, dtype=torch.int32, device=pool.device
    )
    pool._flowkv_global_indices = torch.empty(
        (native.max_query_tokens, native.topk), dtype=torch.int32, device=pool.device
    )
    pool._flowkv_global_current = torch.empty(
        native.max_query_tokens, dtype=torch.int32, device=pool.device
    )
    device = torch.cuda.get_device_properties(pool.flowkv_shards.device)
    pool._flowkv_pull_grid_budget = (
        device.multi_processor_count * device.max_threads_per_multi_processor // 256
    )
    pool.flowkv_prefetch = bool(memory.prefetch)
    # The existing framework factory passes the enabled flag and this same pool.
    # Retain the policy here so it cannot discard an explicit completion mode.
    pool.flowkv_prefetch_policy = memory.prefetch
    pool.flowkv_fused_topk = memory.fused_topk
    pool._flowkv_lookup_layer = None
    pool._flowkv_lookup_descriptors = []
    if pool.flowkv_fused_topk:
        for layer in range(native.layers):
            state = pool.flowkv_arena.groups[layer]
            tensors = (
                pool.flowkv_arena.step,
                pool.flowkv_arena.active,
                pool.flowkv_arena.current_keys,
                pool.flowkv_arena.current_values,
                state.tags,
                state.pins,
                state.hints,
                state.remap,
                state.pull_rows,
                state.pull_counts,
                state.totals,
                pool._flowkv_global_indices,
            )
            descriptor = [tensor.data_ptr() for tensor in tensors] + [
                native.capacity,
                pool.flowkv_arena.current_keys.numel(),
                native.topk,
                global_rows,
                pool._flowkv_owner_start,
                global_rows // degree,
                degree,
                state.cursors.data_ptr(),
            ]
            pool._flowkv_lookup_descriptors.append(
                torch.tensor(descriptor, dtype=torch.int64, device=pool.device)
            )


def global_ids(pool, rows):
    logical = rows + pool._flowkv_owner_start
    degree = pool._flowkv_storage_size
    slots = pool._flowkv_global_rows // degree
    return (
        torch.where(rows > 0, (logical % degree) * slots + logical // degree, rows)
        .to(torch.int32)
        .contiguous()
    )


def get_key_buffer(pool, layer):
    if pool._flowkv_mode == "decode" and pool._flowkv_decode_layer == layer:
        return pool.flowkv_arena.storage[layer]
    if pool._flowkv_mode != "prefill":
        raise RuntimeError("Record KV read requires decode or prefill context")
    if pool._flowkv_chunked_prefill:
        raise RuntimeError(
            "Bounded prefill KV must be consumed through its remapped attention window"
        )
    if pool._flowkv_prefill_layer != layer:
        arena, layout = pool.flowkv_arena, pool.flowkv_layout
        arena.kernel().fill_native(
            pool.flowkv_peers.device_ptr,
            pool._flowkv_history_rows,
            arena.staging,
            pool._flowkv_current_rows,
            arena.prefill,
            pool._flowkv_peer_offset(layer),
            pool._flowkv_storage_size,
            pool.flowkv_rank,
            pool._flowkv_owner_start,
            pool._flowkv_namespaces > 1,
            *(
                (pool.flowkv_request.page_map,)
                if pool.flowkv_request is not None
                else ()
            ),
        )
        history, current = pool._flowkv_history_rows, pool._flowkv_current_rows
        if history.numel() + current.numel() <= layout.capacity:
            state = arena.groups[layer]
            arena.kernel().seed_native(
                arena.prefill,
                history,
                current,
                arena.storage[layer],
                state.tags,
                state.pins,
                state.hints,
                state.cursors,
                pool._flowkv_owner_start,
            )
        pool._flowkv_prefill_layer = layer
    return pool.flowkv_arena.prefill


def begin_decode(pool, current):
    arena = pool.flowkv_arena
    cur = pool._flowkv_global_current[: current.numel()]
    if pool.flowkv_fused_topk:
        arena.kernel().begin_native(
            current.contiguous(),
            cur,
            pool._flowkv_owner_start,
            arena.layout.groups,
            arena.layout.pool_rows,
            arena.step,
            arena.active,
            arena.current_keys,
            arena.current_values,
            arena.cursors,
            arena.totals,
            arena.hints,
            arena.counts,
            arena.histograms,
        )
        return
    arena.kernel().globalize_current(
        current.contiguous(), cur, pool._flowkv_owner_start
    )
    arena.begin(cur)


def prepare_topk_lookup(pool, layer, current):
    if not pool.flowkv_fused_topk:
        return None
    if pool._flowkv_lookup_layer is not None:
        raise RuntimeError("FlowKV TopK lookup must be consumed before the next Index")
    if layer == 0:
        begin_decode(pool, current)
    pool._flowkv_lookup_layer = layer
    return pool._flowkv_lookup_descriptors[layer]


def plan_decode(pool, layer, indices, current, *, begin=True):
    layout, arena = pool.flowkv_layout, pool.flowkv_arena
    if (
        indices.ndim != 2
        or indices.shape[1] != layout.topk
        or indices.shape[0] != current.numel()
    ):
        raise ValueError("Record KV TopK and current query shapes disagree")
    indices = indices.to(torch.int32).contiguous()
    rows = indices.shape[0]
    if rows > layout.max_query_tokens:
        raise ValueError("Record KV queries exceed the configured allocation bound")
    host = pool.flowkv_host
    host_group = layout.layer_groups.group(layer) if host is not None else 0
    idx = pool._flowkv_global_indices[:rows]
    module = arena.kernel()
    if begin and layer == 0 and not pool.flowkv_fused_topk:
        begin_decode(pool, current)
    group = arena.groups[layer]
    if rows == 0:
        return rows
    if pool.flowkv_fused_topk and pool._flowkv_lookup_layer != layer:
        raise RuntimeError("FlowKV allocation has no matching fused TopK lookup")
    operation = module.allocate if pool.flowkv_fused_topk else module.plan_native
    inputs = (
        (idx,) if pool.flowkv_fused_topk else (indices, idx, pool._flowkv_owner_start)
    )
    operation(
        *inputs,
        arena.step,
        arena.active,
        arena.current_keys,
        arena.current_values,
        group.tags,
        group.pins,
        group.hints,
        group.remap,
        group.pull_rows,
        group.pull_slots,
        group.pull_counts,
        group.histogram,
        group.cursors,
        group.totals,
        group.errors,
        layout.capacity,
        pool._flowkv_global_rows,
        *(
            (
                host.miss_ready[host_group : host_group + 1],
                host.fused_metadata[host_group],
            )
            if host is not None
            else ()
        ),
    )
    pool._flowkv_lookup_layer = None
    return rows


def pull_records(pool, layer, rows, operation, blocks):
    layout, arena = pool.flowkv_layout, pool.flowkv_arena
    group = arena.groups[layer]
    host_args = (
        (pool.flowkv_host.descriptors[layer],) if pool.flowkv_host is not None else ()
    )
    if pool.flowkv_request is not None:
        host_args = (pool.flowkv_request.page_map,)
    operation(
        pool.flowkv_peers.device_ptr,
        group.pull_rows,
        group.pull_slots,
        group.pull_counts,
        pool._flowkv_global_indices[:rows],
        group.hints,
        group.remap,
        arena.staging,
        pool._flowkv_global_current[:rows],
        layout.capacity,
        arena.storage[layer],
        pool._flowkv_peer_offset(layer),
        pool._flowkv_storage_size,
        pool.flowkv_rank,
        blocks,
        pool._flowkv_namespaces > 1,
        group.tags,
        group.pins,
        group.cursors,
        group.totals,
        arena.step,
        arena.active,
        arena.current_keys,
        arena.current_values,
        *host_args,
    )


def pull_grid_blocks(pool, queries):
    # Four warps per CTA, one record per warp. Bound the whole batch's grid
    # by the device size; a large batch must not multiply empty pull CTAs.
    # Keep enough threads to resolve every remap column in the same kernel.
    topk = pool.flowkv_layout.topk
    remap_blocks = (topk + 127) // 128
    record_blocks = (topk + 3) // 4
    device_blocks = (pool._flowkv_pull_grid_budget + queries - 1) // queries
    blocks = min(record_blocks, max(remap_blocks, device_blocks))
    services = getattr(pool, "udcache_services", None)
    options = getattr(services, "config", None)
    cap = getattr(options, "low_query_pull_blocks", None)
    if cap is not None and queries <= 8:
        # Remap keeps one thread per TopK column. Opaque record copies use
        # the original grid-stride loop, including completely cold worksets.
        blocks = min(blocks, max(remap_blocks, cap))
    return blocks


def prefetch_decode(pool, layer, indices, current):
    # Native Index calls this hook; synchronous history pull runs at attention.
    return


def prepare_decode(pool, layer, indices, current):
    arena = pool.flowkv_arena
    rows = indices.shape[0]
    rows = plan_decode(pool, layer, indices, current)
    if rows:
        pull_records(
            pool, layer, rows, arena.kernel().pull, pull_grid_blocks(pool, rows)
        )
    pool._flowkv_mode, pool._flowkv_decode_layer = "decode", layer
    group = arena.groups[layer]
    return group.remap[:rows]
