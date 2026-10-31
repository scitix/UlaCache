"""Logical capacity and physical GPU/CPU byte-budget planning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache

from ulacache.storage.layout import KeyValueStorageLayout
from ulacache.storage.workspace import needs_prefill_scratch


def request_history_capacity(max_requests, topk, multiplier, page_size=64):
    """Size historical residency from requests, independently of speculation.

    Query/current/scratch workspaces have separate bounds. Never multiply this
    capacity by a verify width or round a nonrepresentable requested ratio up.
    Execution backends validate their working-set requirements independently.
    """
    from fractions import Fraction

    if any(
        type(value) is not int or value < 1 for value in (max_requests, topk, page_size)
    ):
        raise ValueError("Request history geometry must contain positive integers")
    try:
        ratio = Fraction(str(multiplier))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Request history multiplier must be finite") from error
    slots = max_requests * topk * ratio
    if ratio < 1 or slots.denominator != 1:
        raise ValueError(
            "Requested history ratio must represent an integral capacity of at least one"
        )
    capacity = int(slots)
    if capacity % page_size or capacity >= 2**31:
        raise ValueError(
            "Requested history capacity must already be page aligned and fit int32"
        )
    return capacity


def shared_host_workspace_bytes(capacity, groups, gpu_slots):
    """Bytes for serialized transfer arrays and parallel group expiry scratch."""
    if min(capacity, groups, gpu_slots) < 1:
        raise ValueError("Shared host workspace dimensions must be positive")
    # Shared new/promote/candidates plus the five-word exchange plan.
    # Private expiry lists, eight int32 counters and three int64 prefixes.
    # Each ring counts its own descriptor, outside the workspace.
    return 32 * capacity + groups * (4 * min(capacity, gpu_slots) + 56)


def shared_host_workspace_fixed_bytes(capacity, groups):
    """Shared transfer arrays and private counters/prefixes, independent of GPU rows."""
    if min(capacity, groups) < 1:
        raise ValueError("Shared host workspace dimensions must be positive")
    return 32 * capacity + 56 * groups


def shared_host_workspace_expiry_bytes(capacity, groups, gpu_slots):
    """Private parallel expiry lists, each bounded by physical GPU residency."""
    if min(capacity, groups, gpu_slots) < 1:
        raise ValueError("Shared host workspace dimensions must be positive")
    return 4 * groups * min(capacity, gpu_slots)


def flowkv_cpu_pool_geometry(
    logical_slots_per_rank, *, cpu_slots, cpu_ratio, page_size=64
):
    """Split a logical upper bound into page-aligned GPU and CPU pools.

    Explicit CPU slots preserve the requested logical capacity. Otherwise,
    choose the largest GPU pool whose ``x + floor_page(ratio * x)`` fits the
    bound. A capacity produced here reconstructs exactly during allocation.
    """
    if cpu_slots:
        return logical_slots_per_rank - cpu_slots, cpu_slots

    from fractions import Fraction

    ratio = Fraction(str(cpu_ratio))
    pages = logical_slots_per_rank // page_size
    gpu_pages = ((pages + 1) * ratio.denominator - 1) // (
        ratio.numerator + ratio.denominator
    )
    cpu_pages = gpu_pages * ratio.numerator // ratio.denominator
    return gpu_pages * page_size, cpu_pages * page_size


@dataclass(frozen=True)
class CacheMemoryUsage:
    """Per-rank storage union, including native and temporary cache allocations."""

    device_bytes: int
    host_bytes: int


@dataclass(frozen=True)
class CacheMemoryBudget:
    device_bytes: int
    host_bytes: int

    def contains(self, usage: CacheMemoryUsage):
        return (
            0 <= usage.device_bytes <= self.device_bytes
            and 0 <= usage.host_bytes <= self.host_bytes
        )


@dataclass(frozen=True)
class CacheCapacityPlan:
    """Per-owner cold tokens with the complete per-rank allocation charge.

    Byte counts include every required bank and replicated native allocation.
    They are not multiplied by storage owners. Only disjoint history adds.
    """

    device_token_capacity_per_owner: int
    host_token_capacity_per_owner: int
    storage_owner_count: int
    memory_usage: CacheMemoryUsage
    memory_budget: CacheMemoryBudget

    def __post_init__(self):
        if (
            min(
                self.device_token_capacity_per_owner, self.host_token_capacity_per_owner
            )
            < 0
            or self.storage_owner_count < 1
        ):
            raise ValueError(
                "Cold capacities must be nonnegative with at least one storage owner"
            )

    def usable_history_token_capacity(self, native_page_size):
        total = self.storage_owner_count * (
            self.device_token_capacity_per_owner + self.host_token_capacity_per_owner
        )
        if total <= native_page_size or total % native_page_size:
            raise ValueError(
                "Combined history must cover whole native pages and their reserved page"
            )
        return total - native_page_size

    def validate_memory_budget(self, available_device_bytes=None):
        if not self.memory_budget.contains(self.memory_usage):
            raise ValueError(
                "Complete cache allocation exceeds its CPU or GPU byte limit"
            )
        if (
            available_device_bytes is not None
            and self.memory_usage.device_bytes > available_device_bytes
        ):
            raise ValueError(
                "Complete cache allocation exceeds native available GPU memory"
            )


def price_native_payload_replacement(
    *,
    native_device_bytes,
    native_payload_bytes,
    replacement_device_bytes,
    replacement_host_bytes,
    native_host_bytes,
    allocation_plan,
):
    """Retain every native non-payload charge under a complete external plan.

    The native total includes its constructor/runtime reservations. Only the
    explicitly replaced payload is subtracted; native Index, request tables,
    sliding state and allocation headroom remain in the returned charge.
    """
    values = (
        native_device_bytes,
        native_payload_bytes,
        replacement_device_bytes,
        replacement_host_bytes,
        native_host_bytes,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(
            "Native and replacement byte charges must be nonnegative integers"
        )
    if native_payload_bytes > native_device_bytes:
        raise ValueError(
            "The replaced payload must be included in the native device charge"
        )
    usage = CacheMemoryUsage(
        native_device_bytes - native_payload_bytes + replacement_device_bytes,
        native_host_bytes + replacement_host_bytes,
    )
    if (
        usage.device_bytes > allocation_plan.memory_usage.device_bytes
        or usage.host_bytes > allocation_plan.memory_usage.host_bytes
    ):
        raise ValueError(
            "Complete native and replacement charges exceed the declared allocation usage"
        )
    allocation_plan.validate_memory_budget()
    return usage


def plan_cache_capacity(
    *,
    memory_budget,
    measure_memory_usage,
    minimum_device_token_capacity,
    maximum_token_capacity_per_owner,
    page_size,
    storage_owner_count=1,
    minimum_host_token_capacity=0,
):
    """Maximize total cold history, then GPU residency at the chosen total.

    The callback receives per-owner GPU/CPU full-token counts and returns a
    complete per-rank ``CacheMemoryUsage``. Reuse native tensor constructors
    for Index, window/state, allocator and request tables; include UlaCache
    metadata, writer, attention, forward/Graph storage and runtime reserves.

    At fixed total history, moving a page from CPU to GPU must not decrease
    device usage or increase host usage. Increasing total history at the
    minimum host-fitting GPU capacity must not decrease device usage. Native
    paged layouts satisfy these monotonic conditions, including page padding.
    The upper bound must respect request addressing and identifier widths.
    Measurements may be CPU/meta predictions; the returned plan is not a
    substitute for actual startup inventories or runtime peak validation.
    """
    if (
        page_size < 1
        or storage_owner_count < 1
        or minimum_device_token_capacity < page_size
        or minimum_device_token_capacity % page_size
        or minimum_host_token_capacity < 0
        or minimum_host_token_capacity % page_size
        or maximum_token_capacity_per_owner
        < minimum_device_token_capacity + minimum_host_token_capacity
    ):
        raise ValueError(
            "Capacity bounds must include a page-aligned device working set"
        )
    minimum_device_pages = minimum_device_token_capacity // page_size
    minimum_host_pages = minimum_host_token_capacity // page_size
    minimum_total_pages = minimum_device_pages + minimum_host_pages
    maximum_total_pages = maximum_token_capacity_per_owner // page_size

    @lru_cache(maxsize=None)
    def usage(device_pages, host_pages):
        return measure_memory_usage(device_pages * page_size, host_pages * page_size)

    def minimum_device_pages_for_host_budget(total_pages):
        lower, upper = minimum_device_pages, total_pages - minimum_host_pages
        if usage(upper, minimum_host_pages).host_bytes > memory_budget.host_bytes:
            return None
        while lower < upper:
            middle = (lower + upper) // 2
            if (
                usage(middle, total_pages - middle).host_bytes
                <= memory_budget.host_bytes
            ):
                upper = middle
            else:
                lower = middle + 1
        return lower

    def fits(total_pages):
        device_pages = minimum_device_pages_for_host_budget(total_pages)
        return device_pages is not None and memory_budget.contains(
            usage(device_pages, total_pages - device_pages)
        )

    if not fits(minimum_total_pages):
        raise ValueError(
            "The minimum device working set exceeds the complete memory budget"
        )
    lower, upper = minimum_total_pages, maximum_total_pages
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if fits(middle):
            lower = middle
        else:
            upper = middle - 1
    total_pages = lower
    lower = minimum_device_pages_for_host_budget(total_pages)
    upper = total_pages - minimum_host_pages
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if memory_budget.contains(usage(middle, total_pages - middle)):
            lower = middle
        else:
            upper = middle - 1
    return CacheCapacityPlan(
        lower * page_size,
        (total_pages - lower) * page_size,
        storage_owner_count,
        usage(lower, total_pages - lower),
        memory_budget,
    )


@dataclass(frozen=True)
class RecordOwnerCapacity:
    """One disjoint cold-record set, possibly stored in several replicas."""

    owner_rank: int
    device_record_capacity: int
    host_record_capacity: int
    physical_replica_count: int = 1

    def __post_init__(self):
        if (
            min(self.device_record_capacity, self.host_record_capacity) < 0
            or self.physical_replica_count < 1
        ):
            raise ValueError(
                "Cold capacities must be nonnegative and each owner must have a physical replica"
            )


@dataclass(frozen=True)
class RequiredCachePartitionCapacity:
    """A required tier or native KV feature partition of the same history.

    Owners contain disjoint record IDs. Replicas contain identical IDs/bytes.
    Counts cover the complete set of payload banks in this partition, so model
    layers and storage aliases are not additional history-capacity multipliers.
    Bounded attention windows and recurrent state do not describe cold history.
    """

    request_namespace_id: int
    partition_name: str
    storage_layout: KeyValueStorageLayout
    record_owners: tuple[RecordOwnerCapacity, ...]
    reserved_logical_record_count: int


@dataclass(frozen=True)
class RequestNamespaceCapacityBounds:
    """One independent allocation/storage domain, bounded in history tokens.

    Routing ranks that share one allocator or cold pool are not independent
    namespaces. Represent that common domain once, or supply disjoint slices.
    """

    request_namespace_id: int
    native_allocator_usable_token_slots: int
    request_address_token_slots: int

    def __post_init__(self):
        if (
            min(
                self.native_allocator_usable_token_slots,
                self.request_address_token_slots,
            )
            < 0
        ):
            raise ValueError(
                "Native allocator and request-address limits must be nonnegative"
            )


def _describe_partition(partition: RequiredCachePartitionCapacity):
    owners = partition.record_owners
    if len({owner.owner_rank for owner in owners}) != len(owners):
        raise ValueError("A cold-record owner must appear once per required partition")
    device_records = sum(owner.device_record_capacity for owner in owners)
    host_records = sum(owner.host_record_capacity for owner in owners)
    if (
        not 0
        <= partition.reserved_logical_record_count
        <= device_records + host_records
    ):
        raise ValueError(
            "The allocation guard must fit inside the combined cold capacity"
        )
    tokens_per_record = partition.storage_layout.tokens_per_record
    return dict(
        request_namespace_id=partition.request_namespace_id,
        partition_name=partition.partition_name,
        storage_layout_name=partition.storage_layout.name,
        tokens_per_record=tokens_per_record,
        record_owner_count=len(owners),
        record_owners=[asdict(owner) for owner in owners],
        unique_device_cold_record_slots=device_records,
        unique_host_cold_record_slots=host_records,
        physical_device_cold_record_slots=sum(
            owner.device_record_capacity * owner.physical_replica_count
            for owner in owners
        ),
        physical_host_cold_record_slots=sum(
            owner.host_record_capacity * owner.physical_replica_count
            for owner in owners
        ),
        reserved_logical_record_count=partition.reserved_logical_record_count,
        usable_history_token_slots=(
            device_records + host_records - partition.reserved_logical_record_count
        )
        * tokens_per_record,
    )


def calculate_addressable_history_capacity(
    required_partitions: Iterable[RequiredCachePartitionCapacity],
    namespace_bounds: Iterable[RequestNamespaceCapacityBounds],
):
    """Sum disjoint owners, limit required partitions, then sum request spaces.

    Callers supply actual storage ownership and allocator inventories. Model TP
    does not establish record replication or ownership. A native allocation
    size argument is not a substitute for its actual usable allocator size.
    This is an allocation/accounting result; it does not validate the selected
    working set, maximum running concurrency or runtime memory peaks.
    """
    partitions = [_describe_partition(partition) for partition in required_partitions]
    bounds = tuple(namespace_bounds)
    namespace_ids = {bound.request_namespace_id for bound in bounds}
    if len(namespace_ids) != len(bounds):
        raise ValueError("Provide one native allocator bound per request namespace")
    if {partition["request_namespace_id"] for partition in partitions} != namespace_ids:
        raise ValueError(
            "Every request namespace needs its required partitions and address bounds"
        )
    namespaces = []
    for bound in bounds:
        required = [
            partition
            for partition in partitions
            if partition["request_namespace_id"] == bound.request_namespace_id
        ]
        cold_capacity = min(
            partition["usable_history_token_slots"] for partition in required
        )
        namespaces.append(
            dict(
                request_namespace_id=bound.request_namespace_id,
                required_partition_names=[
                    partition["partition_name"] for partition in required
                ],
                cold_storage_history_token_slots=cold_capacity,
                native_allocator_usable_token_slots=bound.native_allocator_usable_token_slots,
                request_address_token_slots=bound.request_address_token_slots,
                usable_history_token_slots=min(
                    cold_capacity,
                    bound.native_allocator_usable_token_slots,
                    bound.request_address_token_slots,
                ),
            )
        )
    return dict(
        required_partitions=partitions,
        request_namespaces=namespaces,
        total_usable_history_token_slots=sum(
            namespace["usable_history_token_slots"] for namespace in namespaces
        ),
    )


@dataclass(frozen=True)
class PlainPrefillBudget:
    logical_row_bytes: int
    fixed_bytes: int


def plain_prefill_pool_budget(layout):
    """Charge the actual window/staging once, instead of per logical row."""
    if layout.compression_ratio != 1:
        raise ValueError("Plain prefill budgeting requires uncompressed native records")
    if not needs_prefill_scratch(layout.disaggregation_mode):
        return PlainPrefillBudget(0, 0)
    if layout.prefill_window_rows:
        return PlainPrefillBudget(
            0,
            layout.prefill_buffer_bytes
            + layout.staging_buffer_bytes
            + layout.prefill_lookup_bytes,
        )
    return PlainPrefillBudget(layout.row_bytes, 0)


def position_residency_bytes(
    gpu_records,
    host_records,
    *,
    layers,
    groups,
    queries,
    topk,
    row_bytes,
    maximum_write_records,
    presence_bytes,
    prefill_union_method="sort",
):
    """Match the existing contiguous engine and its four CP mapping buffers.

    ``presence_bytes`` is the existing backend's descriptor/bitmap measurement;
    no serving framework or device dependency is imported by this module.
    """
    logical = gpu_records + host_records
    capacity = max(queries * (topk + 1), maximum_write_records, gpu_records + 1)
    state = groups * (16 * logical + 12 * gpu_records + 8 * host_records + 16)
    workspace = shared_host_workspace_bytes(capacity, groups, gpu_records)
    rings = groups * presence_bytes(gpu_records)
    # Native remap, three scalar counts, stats, thirty-one-word group frames and
    # the mapped host base pointer. Shared state/workspace views count once.
    native = 4 * groups * queries * topk + 12 + 24 * groups + 248 * groups + 8
    mapping = 4 * (
        groups * queries * topk
        + groups * queries
        + queries
        + max(queries, queries * topk)
    )
    extra = 0
    if prefill_union_method == "device":
        extra = 4 * (logical + min(gpu_records, capacity - 1) + 1)
    elif prefill_union_method != "sort":
        raise ValueError("Bind the native sort or device prefill union geometry")
    return dict(
        gpu_bytes=layers * gpu_records * row_bytes
        + state
        + workspace
        + rings
        + native
        + mapping
        + extra,
        cpu_bytes=layers * host_records * row_bytes,
        mapping_bytes=mapping,
    )


def plan_position_capacity(
    *,
    available_gpu_bytes,
    host_records,
    degree,
    layers,
    groups,
    queries,
    presence_bytes,
    draft_layers=1,
    row_bytes=656,
    topk=2048,
    maximum_write_records=8192,
    prefill_union_method="sort",
    index_communication_reserve_bytes=4 * 1024**3,
    native_auxiliary_reserve_bytes=1024**3,
):
    """Maximize per-owner GPU residency without changing CPU/MRQ/Graph bounds.

    Communication and native auxiliary reserves are estimates, not padding
    allocations. Startup inventories and runtime peaks must qualify the result.
    The unchanged official draft still owns the widened, replicated namespace.
    """
    values = (
        available_gpu_bytes,
        host_records,
        degree,
        layers,
        groups,
        queries,
        row_bytes,
        topk,
    )
    if (
        any(type(n) is not int or n < 1 for n in values)
        or degree < 2
        or degree & (degree - 1)
        or host_records % 64
        or groups > layers
        or type(draft_layers) is not int
        or draft_layers < 0
    ):
        raise ValueError("Bind a positive page64 position-owner geometry")
    cpu_bytes = host_records * layers * row_bytes
    minimum_gpu = ((queries * (topk + 1) + 1 + 63) // 64) * 64
    maximum_gpu = (
        min(
            available_gpu_bytes // (layers * row_bytes),
            (2**31 - 1) // degree - host_records,
        )
        // 64
        * 64
    )

    def measure(gpu, host):
        base = position_residency_bytes(
            gpu,
            host,
            layers=layers,
            groups=groups,
            queries=queries,
            topk=topk,
            row_bytes=row_bytes,
            maximum_write_records=maximum_write_records,
            presence_bytes=presence_bytes,
            prefill_union_method=prefill_union_method,
        )
        index = groups * 132 * (gpu + host)
        native_draft = draft_layers * degree * (gpu + host) * (row_bytes + 132)
        windows = 2 * 1048576 * row_bytes
        return CacheMemoryUsage(
            base["gpu_bytes"]
            + index
            + native_draft
            + windows
            + index_communication_reserve_bytes
            + native_auxiliary_reserve_bytes,
            base["cpu_bytes"],
        )

    result = plan_cache_capacity(
        memory_budget=CacheMemoryBudget(available_gpu_bytes, cpu_bytes),
        measure_memory_usage=measure,
        minimum_device_token_capacity=minimum_gpu,
        maximum_token_capacity_per_owner=maximum_gpu + host_records,
        minimum_host_token_capacity=host_records,
        page_size=64,
        storage_owner_count=degree,
    )
    return dict(
        gpu_records_per_owner=result.device_token_capacity_per_owner,
        host_records_per_owner=host_records,
        global_usable_records=degree
        * (result.device_token_capacity_per_owner + host_records - 64),
        predicted_gpu_bytes=result.memory_usage.device_bytes,
        cpu_KV_payload_bytes=cpu_bytes,
        available_gpu_bytes=available_gpu_bytes,
        index_communication_reserve_bytes=index_communication_reserve_bytes,
        native_auxiliary_reserve_bytes=native_auxiliary_reserve_bytes,
        draft_layout="native_replicated",
        prediction_only=True,
        actual_startup_inventory_required=True,
    )
