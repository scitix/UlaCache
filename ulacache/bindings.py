"""Allocator, compiler and communication callbacks supplied by the framework."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Protocol

from .cache_config import CacheExecutionOptions, CacheOptions
from ulacache.storage.layout import PositionShardLayout


class IPCManager(Protocol):
    def batch_get_handles(self, pointers: Sequence[int]) -> Iterable[Any]: ...
    def batch_open_handles(self, handles: Sequence[Any]) -> Iterable[int]: ...
    def destroy(self) -> None: ...


class IPCManagerFactory(Protocol):
    def __call__(self) -> IPCManager: ...


class HostRecordPool(Protocol):
    """Pinned uint8 records and device-visible pointers, closed by the owner."""

    storage: Any
    ptrs: Any
    nbytes: int

    def close(self) -> None: ...


class HostPoolFactory(Protocol):
    def __call__(
        self, *, layers: int, slots: int, row_bytes: int, device: Any
    ) -> HostRecordPool: ...


class LogicalSlotAllocator(Protocol):
    """One ID authority, bound at initialization; these calls do not register KV.

    ``alloc`` returns unique logical IDs as a device tensor, or None on capacity
    failure. ``free`` returns IDs only after cache readers and references retire.
    Native page alignment, admission and reference counting stay in this owner.
    """

    def alloc(self, n: int) -> Any: ...
    def free(self, ids: Any) -> None: ...


@dataclass(frozen=True)
class LocalCacheBindings:
    """Dependencies for an independent rank-local cache."""

    device: Any
    allocate_host_record_storage: HostPoolFactory
    compile_device_module: Callable
    execution_options: CacheExecutionOptions = field(
        default_factory=CacheExecutionOptions
    )
    logical_slot_allocator: LogicalSlotAllocator | None = None


@dataclass(frozen=True, kw_only=True)
class PositionShardBindings(LocalCacheBindings):
    """Per-owner tiers in one round-robin global allocator namespace.

    ``spec`` retains per-owner physical capacities. The allocator returns
    global IDs with owner=id%degree and owner-local ID=id//degree; its page
    spans degree native pages. Native attention merges partial O/LSE through
    the framework's existing DCP group. No peer KV cache is constructed.
    """

    position_layout: PositionShardLayout

    def __post_init__(self):
        if not isinstance(self.position_layout, PositionShardLayout):
            raise TypeError("Bind an explicit position-shard layout")


@dataclass(frozen=True)
class RecordOwnerBindings:
    """Dependencies for disjoint record owners in one logical namespace.

    The process group identifies storage owners, independently of model TP.
    Writers supply complete native record replicas within that owner group.
    Feature shards require the group replicating each complete native record;
    independent request producers cannot claim replicated writes from model TP.
    Replicated selections promise identical IDs, ordering, padded shapes and
    active counts, avoiding duplicate residency inputs without removing owners
    or reader publication. Capacities are per owner; allocator pages span the
    shared namespace. Calls use the caller's current CUDA stream.
    """

    device: Any
    owner_process_group: Any
    allocate_host_record_storage: HostPoolFactory
    compile_device_module: Callable
    ipc_manager_factory: IPCManagerFactory
    allocation_records_per_page: int
    maximum_prefill_record_count: int
    native_record_writes_replicated: bool
    execution_options: CacheExecutionOptions = field(
        default_factory=lambda: CacheExecutionOptions(
            record_storage_backend="native_paged", prefill_union_method="sort"
        )
    )
    native_selections_replicated: bool = False
    logical_slot_allocator: LogicalSlotAllocator | None = None


@dataclass(frozen=True)
class TensorParallelCacheBindings:
    """Adopt an initialized ordinary-TP pool and its existing shared lifecycle.

    This binding does not allocate another pool, directory or prefetch stream.
    Native allocation hooks may retain ID authority; in that case leave the
    allocator unbound and register/release through the public handle exactly
    once. A bound allocator must supply ID-only callbacks without cache hooks.
    Native current-write publication remains an explicit adapter boundary.
    """

    pool: Any
    logical_slot_allocator: LogicalSlotAllocator | None = None
    execution_options: CacheExecutionOptions = field(
        default_factory=lambda: CacheExecutionOptions(
            record_storage_backend="tensor_parallel"
        )
    )
    owns_storage: bool = False


class StorageGroup(Protocol):
    """Rank coordinates and the existing CPU process group for IPC setup."""

    rank_in_group: int
    cpu_group: Any
    device_group: Any


class MemoryRegionFactory(Protocol):
    def __call__(self) -> ContextManager[Any]: ...


class HostMemoryBudget(Protocol):
    def __call__(self, requested_bytes: int) -> int: ...


class WindowInitializer(Protocol):
    """Bind the native prefill representation after CUDA storage allocation."""

    def __call__(self, pool: Any) -> None: ...


@dataclass(frozen=True)
class BackendServices:
    degree: int
    group: StorageGroup
    namespaces: int
    config: CacheOptions
    memory_region: MemoryRegionFactory
    ipc_manager_factory: IPCManagerFactory
    host_pool_factory: HostPoolFactory
    host_memory_budget_bytes: HostMemoryBudget
    window_initializer: WindowInitializer | None = None

    def __post_init__(self):
        if self.degree < 1 or self.namespaces < 1:
            raise ValueError("Storage degree and namespace count must be positive")
        if not 0 <= self.group.rank_in_group < self.degree:
            raise ValueError("Storage rank lies outside the configured degree")
