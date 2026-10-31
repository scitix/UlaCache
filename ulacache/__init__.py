"""Unified distributed KV cache management.

The root namespace and optional feature namespaces have no device dependencies.
Core descriptors are resolved independently when requested.
"""

from importlib import import_module

__version__ = "0.1.0"
__all__ = [
    "CacheLayout",
    "ResidentQueryLayout",
    "InterleavedRecordLayout",
    "NativePageLayout",
    "LayerGroupPlan",
    "RequestPagePool",
    "RecordSpec",
    "history_capacity",
    "KeyValueRecordPlane",
    "KeyValueStorageLayout",
    "LayerReuseSpecification",
    "SparseAccessSpecification",
    "SparseAttentionAccessPlan",
    "ResidentAttentionInputs",
    "ResidentView",
    "MemoryTierCapacity",
    "UnifiedCacheSpecification",
    "UlaCache",
    "UnifiedMemoryCache",
    "LocalCacheBindings",
    "CacheExecutionOptions",
    "ForwardExecutionContext",
    "RecordOwnerBindings",
    "TensorParallelCacheBindings",
    "PositionShardBindings",
    "PositionShardLayout",
    "LogicalSlotAllocator",
    "FileRecordSource",
    "PrefetchPolicy",
    "init",
    "RecordOwnerCapacity",
    "RequiredCachePartitionCapacity",
    "RequestNamespaceCapacityBounds",
    "calculate_addressable_history_capacity",
    "CacheMemoryUsage",
    "CacheMemoryBudget",
    "CacheCapacityPlan",
    "plan_cache_capacity",
]

_EXPORTS = {
    "CacheLayout": (".storage.workspace", "FlowKVLayout"),
    "ResidentQueryLayout": (".storage.workspace", "ResidentQueryLayout"),
    "InterleavedRecordLayout": (".storage.layout", "InterleavedRecordLayout"),
    "NativePageLayout": (".storage.layout", "NativePageLayout"),
    "LayerGroupPlan": (".cache.layers", "LayerGroupPlan"),
    "RequestPagePool": (".cache.request_pages", "FlowKVRequestPagePool"),
    "RecordSpec": (".storage.layout", "FlowKVPackedMLASpec"),
    "history_capacity": (".storage.layout", "history_capacity"),
    "KeyValueRecordPlane": (".storage.layout", "KeyValueRecordPlane"),
    "KeyValueStorageLayout": (".storage.layout", "KeyValueStorageLayout"),
    "LayerReuseSpecification": (".cache.layers", "LayerReuseSpecification"),
    "SparseAccessSpecification": (".cache.access", "SparseAccessSpecification"),
    "SparseAttentionAccessPlan": (".cache.access", "SparseAttentionAccessPlan"),
    "ResidentAttentionInputs": (".cache.access", "ResidentAttentionInputs"),
    "ResidentView": (".cache.access", "ResidentView"),
    "MemoryTierCapacity": (".cache.specification", "MemoryTierCapacity"),
    "UnifiedCacheSpecification": (".cache.specification", "UnifiedCacheSpecification"),
    "UnifiedMemoryCache": (".client", "UnifiedMemoryCache"),
    "UlaCache": (".client", "UlaCache"),
    "init": (".client", "init"),
    "LocalCacheBindings": (".bindings", "LocalCacheBindings"),
    "PositionShardBindings": (".bindings", "PositionShardBindings"),
    "PositionShardLayout": (".storage.layout", "PositionShardLayout"),
    "CacheExecutionOptions": (".cache_config", "CacheExecutionOptions"),
    "PrefetchPolicy": (".cache_config", "PrefetchPolicy"),
    "RecordOwnerBindings": (".bindings", "RecordOwnerBindings"),
    "TensorParallelCacheBindings": (".bindings", "TensorParallelCacheBindings"),
    "LogicalSlotAllocator": (".bindings", "LogicalSlotAllocator"),
    "FileRecordSource": (".transfer.file_io", "FileRecordSource"),
    "ForwardExecutionContext": (".cache.access", "ForwardExecutionContext"),
    "RecordOwnerCapacity": (".storage.capacity", "RecordOwnerCapacity"),
    "RequiredCachePartitionCapacity": (
        ".storage.capacity",
        "RequiredCachePartitionCapacity",
    ),
    "RequestNamespaceCapacityBounds": (
        ".storage.capacity",
        "RequestNamespaceCapacityBounds",
    ),
    "calculate_addressable_history_capacity": (
        ".storage.capacity",
        "calculate_addressable_history_capacity",
    ),
    "CacheMemoryUsage": (".storage.capacity", "CacheMemoryUsage"),
    "CacheMemoryBudget": (".storage.capacity", "CacheMemoryBudget"),
    "CacheCapacityPlan": (".storage.capacity", "CacheCapacityPlan"),
    "plan_cache_capacity": (".storage.capacity", "plan_cache_capacity"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
