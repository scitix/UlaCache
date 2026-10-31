# SPDX-License-Identifier: Apache-2.0
"""Read-only pre-transaction GPU hit counters for shared-pool trace replay.

Metrics use positive registered owner-local IDs. Current-token admissions are
separate from historical TopK reads. This optional observer never changes the
cache state and must not be included in primary kernel latency measurements.
"""

from typing import NamedTuple
from ulacache.compiler import cache_once, load_jit

COLUMNS = (
    "raw_topk",
    "raw_topk_hot",
    "raw_current_occurrences",
    "raw_current_occurrences_hot",
    "unique_historical_topk",
    "unique_historical_topk_hot",
    "unique_current",
    "unique_current_hot",
    "unique_union",
    "unique_union_hot",
    "invalid_bindings",
    "unique_current_in_topk",
    "unique_current_in_topk_hot",
)


@cache_once
def module():
    return load_jit(
        "trace_hit_observer",
        "v1",
        cuda_files=["csrc/kvcacheio/trace_hits.cuh"],
        cuda_wrappers=[
            ("temporary_bytes", "sglang::trace_hits_temporary_bytes"),
            ("observe", "sglang::trace_hits_observe"),
        ],
    )


class HitWorkspace(NamedTuple):
    ids: object
    sorted_ids: object
    unique_ids: object
    unique_count: object
    current_hash: object
    current_selected: object
    stats: object
    temporary: object


def create_workspace(state, queries, topk, *, stats=None):
    import torch

    if not 0 < queries <= 384 or topk < 1:
        raise ValueError("Expected 1..384 query rows and positive TopK")
    n = queries * (topk + 1)
    capacity = max(128, 1 << (2 * queries - 1).bit_length())
    device = state.directory.device

    def alloc(count, dtype=torch.int32):
        return torch.empty(count, dtype=dtype, device=device)

    if stats is None:
        stats = alloc(len(COLUMNS), torch.int64)
    if (
        stats.shape != (len(COLUMNS),)
        or stats.dtype != torch.int64
        or not stats.is_contiguous()
        or stats.device != device
    ):
        raise ValueError("Expected contiguous int64[13] stats on the cache device")
    temporary = module().temporary_bytes(n, state.directory.numel())
    return HitWorkspace(
        alloc(n),
        alloc(n),
        alloc(n),
        alloc(1),
        alloc(capacity),
        alloc(capacity),
        stats,
        alloc(temporary, torch.uint8),
    )


def observe(selected, current, active, state, scratch):
    import torch

    if selected.ndim != 2 or current.ndim != 1 or selected.size(0) != current.numel():
        raise ValueError("Expected selected[query,topk] and current[query]")
    if selected.dtype not in (torch.int32, torch.int64) or current.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Expected int32/int64 positive registered IDs")
    module().observe(
        selected, current, active, state.directory, state.gpu_tags, *scratch
    )
