"""Collective shutdown of a bound CUDA record pool.

The caller must stop scheduling and retire every captured graph that refers to
the pool before closing it. This is a one-shot operation on every rank of the
same CPU process group, in the same order. It does not destroy caller-owned
process groups or free externally held tensor views.
"""

from __future__ import annotations

from typing import Any, Protocol

from ulacache.bindings import BackendServices


class Closeable(Protocol):
    def close(self) -> None: ...


class PoolResources(Protocol):
    """Resources installed by ``record_storage.configure_pool``.

    ``flowkv_step_publication`` exists only on pools bound with step publication;
    callers specify that binding explicitly. Native ``kv_buffer`` views and any
    adapter-owned window or model state remain the adapter's responsibility.
    """

    device: Any
    udcache_services: BackendServices
    flowkv_step_publication: Closeable
    flowkv_host: Closeable | None
    flowkv_request: Closeable | None
    flowkv_peers: Closeable
    flowkv_arena: Any
    flowkv_shards: Any
    flowkv_staging_positions: Any
    flowkv_staging_rows: Any
    flowkv_staging_target_rows: Any
    _flowkv_global_indices: Any
    _flowkv_global_current: Any
    _flowkv_lookup_descriptors: list[Any]


def close_pool(pool: PoolResources, *, step_publication: bool) -> None:
    """Drain CUDA, close transfer/IPC resources, then drop owned tensor refs.

    Shutdown order is local CUDA drain, optional step publication, host/device
    child or request exchange resources, payload IPC mappings, then a CPU-group
    barrier before releasing exported storage and arena references. Existing
    resource ``close`` methods handle their own streams and pinned memory.

    The explicit ``step_publication`` value must match the startup binding.
    This function neither detects active graphs nor makes concurrent scheduling
    safe; quiescence is a caller precondition. A failure stops shutdown before
    the final reference release and must be handled collectively by the caller.
    """
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize(pool.device)
    if step_publication:
        pool.flowkv_step_publication.close()
    if pool.flowkv_host is not None:
        pool.flowkv_host.close()
    if pool.flowkv_request is not None:
        pool.flowkv_request.close()
    pool.flowkv_peers.close()

    # Keep exported storage alive until all ranks have closed imported mappings.
    dist.barrier(group=pool.udcache_services.group.cpu_group)

    if step_publication:
        del pool.flowkv_step_publication
    pool.flowkv_host = None
    pool.flowkv_request = None
    del pool.flowkv_peers, pool.flowkv_arena, pool.flowkv_shards
    del pool.flowkv_staging_positions, pool.flowkv_staging_rows
    del pool.flowkv_staging_target_rows
    del pool._flowkv_global_indices, pool._flowkv_global_current
    del pool._flowkv_lookup_descriptors, pool.udcache_services
