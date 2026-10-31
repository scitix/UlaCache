# SPDX-License-Identifier: Apache-2.0
"""HiSparse single-request LRU over a GPU-deduplicated shared-pool union.

The large-pool adaptation retains the native single-CTA scan/compaction stages,
but widens slot indices and places temporary arrays in global memory. It is not
an unmodified native HiSparse implementation. Compiler services are injected.
"""

from typing import NamedTuple
from ulacache.compiler import cache_once, load_jit
from pathlib import Path


@cache_once
def module(gpu_slots, maximum_keys, queries):
    return load_jit(
        "hisparse_union",
        str(gpu_slots),
        str(maximum_keys),
        str(queries),
        cuda_files=[str(Path(__file__).parent / "kernels" / "hisparse_union.cuh")],
        cuda_wrappers=[
            ("temporary_bytes", "sglang::hisparse_union_temporary_bytes"),
            ("dedup", "sglang::hisparse_union_dedup"),
            (
                "update",
                f"sglang::hisparse_union_update<{gpu_slots},{maximum_keys},{queries}>",
            ),
            (
                "native_reference",
                f"sglang::hisparse_union_native_reference<{gpu_slots},{maximum_keys}>",
            ),
        ],
    )


class UnionWorkspace(NamedTuple):
    ids: object
    sorted_ids: object
    unique_ids: object
    unique_count: object
    lru_slots: object
    scratch: object
    locations: object
    temporary: object


def create_workspace(state, queries, topk):
    import torch

    if not 0 < queries <= 384 or topk < 1:
        raise ValueError("Expected1..384query rows and positive TopK")
    if bool(state.leases.any()):
        raise ValueError("The shared-pool adapter requires no external leases")
    G = state.gpu_tags.numel()
    count = queries * (topk + 1)
    maximum = min(count, G)
    chunks = (G + 31) // 32
    current = max(128, 1 << (2 * queries - 1).bit_length())
    storage = G + 5 * maximum + 2 * (chunks + 1) + 2 + current
    compiler = module(G, maximum, queries)
    temp = compiler.temporary_bytes(count, state.directory.numel())

    def allocate(n, dtype=torch.int32):
        return torch.empty(n, dtype=dtype, device=state.gpu_tags.device)

    live = state.gpu_tags >= 0
    ages = torch.where(live, state.access_time[state.gpu_tags.clamp_min(0).long()], -1)
    lru = torch.argsort(ages, stable=True).int()
    free = lru[~live[lru.long()]]
    if free.numel() != int(state.free_counts[0]):
        raise ValueError("GPU free count does not match empty tags")
    # The native stable partition consumes empty slots from the LRU front.
    # Match the common free stack to that order, outside measured updates.
    state.gpu_free[: free.numel()].copy_(free.flip(0))
    return UnionWorkspace(
        allocate(count),
        allocate(count),
        allocate(count),
        allocate(1),
        lru,
        allocate(storage),
        allocate(count),
        allocate(temp, torch.uint8),
    )


def _module(state, selected):
    q, k = selected.shape
    return module(state.gpu_tags.numel(), min(q * (k + 1), state.gpu_tags.numel()), q)


def deduplicate(selected, current, active, owner, state, scratch):
    if selected.ndim != 2 or current.ndim != 1 or selected.size(0) != current.numel():
        raise ValueError("Expected selected[query,topk] and current[query]")
    _module(state, selected).dedup(selected, current, active, owner, *state, *scratch)


def update(selected, current, active, owner, state, work, scratch):
    """Consume the already unique virtual request in one1024-thread CTA."""
    _module(state, selected).update(current, active, owner, *state, *work, *scratch)


def prepare(selected, current, active, owner, state, work, scratch):
    deduplicate(selected, current, active, owner, state, scratch)
    update(selected, current, active, owner, state, work, scratch)
