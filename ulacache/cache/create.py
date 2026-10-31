"""Construct internal execution backends from explicit layout and services."""

from ulacache.bindings import (
    PositionShardBindings,
    RecordOwnerBindings,
    TensorParallelCacheBindings,
)


def validate_prefetch_configuration(cache_specification, cache_bindings):
    """Reject unsupported startup policies before constructing device resources."""
    options = cache_bindings.execution_options
    if not options.copy_follower_records_asynchronously:
        return
    if isinstance(cache_bindings, RecordOwnerBindings):
        raise NotImplementedError(
            "Distributed owner-paged asynchronous prefetch is not implemented"
        )
    sharing = cache_specification.layer_reuse
    if options.prefetch is not None and all(
        layer == owner for layer, owner in enumerate(sharing.sparse_selection_owners)
    ):
        raise ValueError(
            "Layer prefetch requires an explicitly shared TopK selection group"
        )
    if options.record_storage_backend == "native_paged":
        if options.follower_copy_mode != "layer":
            raise ValueError("Local native-paged prefetch requires layer completion")
        if len(cache_specification.storage_layout.record_planes) != 1:
            raise NotImplementedError(
                "Native multi-plane layer prefetch requires a separate gate"
            )
        if sharing.key_value_storage_owners != tuple(range(sharing.layer_count)):
            raise NotImplementedError(
                "Native layer prefetch requires independent KV writers"
            )
        follower_launch = options.follower_exchange_launch_dimensions
        if (
            follower_launch is not None
            and follower_launch != options.exchange_launch_dimensions
        ):
            raise NotImplementedError(
                "Native layer prefetch requires equal anchor/follower launch geometry"
            )


def create_execution_engine(cache_specification, cache_bindings):
    if isinstance(cache_bindings, TensorParallelCacheBindings):
        from ulacache.cache.tensor_parallel import NativeTensorParallelExecution

        return NativeTensorParallelExecution(cache_specification, cache_bindings)
    capacity = cache_specification.capacity_plan
    if capacity.maximum_prefill_union_record_count > capacity.device_record_capacity:
        raise ValueError(
            "This cache backend requires prefill union within device capacity"
        )
    from ulacache.cache.planning import create_rank_local_layer_groups
    from ulacache.cache.planning import (
        create_owner_paged_execution_plan,
        create_paged_cache_execution_plan,
    )

    validate_prefetch_configuration(cache_specification, cache_bindings)
    backend = cache_bindings.execution_options.record_storage_backend
    if isinstance(cache_bindings, PositionShardBindings):
        validate_position_configuration(cache_specification, cache_bindings)
    if isinstance(cache_bindings, RecordOwnerBindings):
        if backend != "native_paged":
            raise ValueError("Record-owner services require native-paged storage")
        plan = create_owner_paged_execution_plan(cache_specification)
        return create_record_owner_execution_engine(
            cache_specification, plan, cache_bindings
        )
    if backend == "contiguous_records":
        plan = create_rank_local_layer_groups(cache_specification)
    elif backend == "native_paged":
        plan = create_paged_cache_execution_plan(cache_specification)
    else:
        raise ValueError(f"Unknown record storage backend: {backend}")
    engine = create_rank_local_execution_engine(
        cache_specification, plan, cache_bindings
    )
    if isinstance(cache_bindings, PositionShardBindings):
        from .position_residency import PositionShardedResidency

        return PositionShardedResidency(engine, cache_bindings.position_layout)
    return engine


def validate_position_configuration(specification, bindings):
    """Reject unsupported owner/storage geometry before allocating CUDA memory."""
    capacity = specification.capacity_plan
    layout = bindings.position_layout
    if bindings.execution_options.record_storage_backend != "contiguous_records":
        raise NotImplementedError(
            "Position CP currently binds contiguous native records"
        )
    if (
        layout.device_records != capacity.device_record_capacity
        or layout.host_records != capacity.host_record_capacity
    ):
        raise ValueError(
            "Position ownership must cover the actual exclusive CPU/GPU tiers"
        )
    storage = specification.storage_layout
    if (
        storage.records_per_page != 64
        or len(storage.record_planes) != 1
        or storage.bytes_per_record != 656
    ):
        raise NotImplementedError(
            "Position CP currently requires native scaled656 FP8 page64"
        )
    if capacity.reserved_logical_record_count != 64:
        raise ValueError("Reserve the native zero page on every position owner")


def create_record_owner_execution_engine(
    cache_specification, execution_plan, cache_bindings
):
    from ulacache.cache.owner_cache import RecordOwnerPagedResidency

    return RecordOwnerPagedResidency(
        cache_specification, cache_bindings, execution_plan
    )


def create_rank_local_execution_engine(
    cache_specification, layer_groups, cache_bindings
):
    from ulacache.compiler import configure_compiler

    configure_compiler(cache_bindings.compile_device_module)
    if cache_bindings.execution_options.record_storage_backend == "native_paged":
        from ulacache.cache.local_paged_cache import NativePagedResidency

        return NativePagedResidency(cache_specification, cache_bindings, layer_groups)
    from ulacache.cache.local_cache import UnifiedLocalResidency

    capacity_plan = cache_specification.capacity_plan
    execution_options = cache_bindings.execution_options
    return UnifiedLocalResidency(
        groups=layer_groups,
        gpu_slots=capacity_plan.device_record_capacity,
        host_slots=capacity_plan.host_record_capacity,
        row_bytes=cache_specification.storage_layout.bytes_per_record,
        max_queries=cache_specification.sparse_access.maximum_padded_query_count,
        topk=cache_specification.sparse_access.maximum_selected_units,
        max_write_rows=capacity_plan.maximum_write_record_count,
        prefill_union_rows=capacity_plan.maximum_prefill_union_record_count,
        device=cache_bindings.device,
        host_pool_factory=cache_bindings.allocate_host_record_storage,
        parallel_expiry=execution_options.expire_residency_in_parallel,
        parallel_prepare=execution_options.prepare_residency_in_parallel,
        parallel_current=execution_options.prepare_current_records_in_parallel,
        parallel_victims=execution_options.select_victims_in_parallel,
        async_followers=execution_options.copy_follower_records_asynchronously,
        exchange_launch=execution_options.exchange_launch_dimensions,
        follower_exchange_launch=execution_options.follower_exchange_launch_dimensions,
        follower_copy_mode=execution_options.follower_copy_mode,
        prefill_exchange_launch=execution_options.prefill_exchange_launch_dimensions,
        prefill_union_mode=execution_options.prefill_union_method,
    )
