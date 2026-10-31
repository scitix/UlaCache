"""CUDA backend operations using an explicit, experimental pool binding ABI.

``bind_pool`` initializes framework-owned bindings; it does not construct a
framework-independent cache handle. Steady-state operations are direct aliases.
"""

from ulacache.cache.attention_buffers import FlowKVC4Arena as RecordArena
from ulacache.bindings import BackendServices, CacheOptions
from ulacache.cache.pool_lifecycle import close_pool
from ulacache.cache.tp_cache import (
    configure_pool as bind_pool,
    global_ids,
    get_key_buffer,
    begin_decode,
    plan_decode,
    prepare_decode,
    prepare_topk_lookup,
    pull_records,
    pull_grid_blocks,
)
from ulacache.cache.host import FlowKVHostTier as IndexResidency
from ulacache.cache.request_storage import FlowKVRequestTier as RequestResidency

__all__ = [
    "RecordArena",
    "BackendServices",
    "CacheOptions",
    "bind_pool",
    "close_pool",
    "global_ids",
    "get_key_buffer",
    "begin_decode",
    "plan_decode",
    "prepare_decode",
    "prepare_topk_lookup",
    "pull_records",
    "pull_grid_blocks",
    "IndexResidency",
    "RequestResidency",
]
