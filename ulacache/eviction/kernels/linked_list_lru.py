# SPDX-License-Identifier: Apache-2.0
"""Plain single-GPU-thread doubly linked LRU for fixed shared-pool ablations.

GPU union/dedup is separate. Every group update launches exactly one thread;
resident hits move to MRU, then misses consume free slots before LRU victims.
Registered positive IDs and no outstanding leases are required.
"""

from typing import NamedTuple
from ulacache.compiler import cache_once, load_jit


@cache_once
def module():
    return load_jit(
        "linked_list_lru",
        "v1",
        cuda_files=["csrc/kvcacheio/linked_list_lru.cuh"],
        cuda_wrappers=[
            ("temporary_bytes", "sglang::linked_lru_temporary_bytes"),
            ("deduplicate", "sglang::linked_lru_deduplicate"),
            ("update", "sglang::linked_lru_update"),
        ],
    )


class LinkedWorkspace(NamedTuple):
    ids: object
    sorted_ids: object
    unique_ids: object
    unique_count: object
    previous: object
    following: object
    ends: object
    temporary: object


def initialize_order(state, scratch):
    """Untimed setup: oldest timestamp first, physical slot breaks initial ties."""
    import torch

    ages = torch.where(
        state.gpu_tags >= 0, state.access_time[state.gpu_tags.clamp_min(0).long()], -1
    )
    order = torch.argsort(ages, stable=True).int()
    scratch.previous[order.long()] = torch.cat([order.new_tensor([-1]), order[:-1]])
    scratch.following[order.long()] = torch.cat([order[1:], order.new_tensor([-1])])
    scratch.ends.copy_(torch.stack([order[0], order[-1]]))


def create_workspace(state, queries, topk):
    import torch

    if not 0 < queries <= 384 or topk < 1:
        raise ValueError("Expected 1..384 queries and positive TopK")
    if bool(state.leases.any()):
        raise ValueError("Plain linked-list adapter requires zero external leases")
    G = state.gpu_tags.numel()
    n = queries * (topk + 1)
    device = state.directory.device

    def alloc(count, dtype=torch.int32):
        return torch.empty(count, dtype=dtype, device=device)

    temp = module().temporary_bytes(n, state.directory.numel())
    result = LinkedWorkspace(
        alloc(n),
        alloc(n),
        alloc(n),
        alloc(1),
        alloc(G),
        alloc(G),
        alloc(2),
        alloc(temp, torch.uint8),
    )
    initialize_order(state, result)
    return result


def deduplicate(selected, current, active, state, scratch):
    import torch

    if selected.ndim != 2 or current.ndim != 1 or selected.shape[0] != current.numel():
        raise ValueError("Expected selected[query,topk] and current[query]")
    if selected.dtype not in (torch.int32, torch.int64) or current.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Expected int32/int64 positive registered IDs")
    module().deduplicate(selected, current, active, state.directory, *scratch)


def update(current, active, state, work, scratch):
    module().update(current, active, *state, *work, *scratch)


def prepare(selected, current, active, owner, state, work, scratch):
    if owner != 0:
        raise ValueError("Plain linked-list benchmark accepts owner-local IDs only")
    deduplicate(selected, current, active, state, scratch)
    update(current, active, state, work, scratch)
