# SPDX-License-Identifier: Apache-2.0
"""Public HiSparse LRU body with fixed8192-slot per-request partitions.

Only MTP2 union construction and the common exclusive-cold I/O contract are
adapted. The existing int16/shared-memory HiSparse device body is unchanged.
"""

from typing import NamedTuple
from ulacache.compiler import cache_once, load_jit
from pathlib import Path


class RequestWorkspace(NamedTuple):
    unique_ids: object
    lru_slots: object
    locations: object
    output: object
    request_indices: object
    sequence_lengths: object
    active_requests: object
    miss_source: object
    miss_destination: object
    miss_count: object
    previous_tags: object
    ids: object
    sorted_ids: object
    offsets: object
    temporary: object
    use_cub: bool


@cache_once
def module():
    return load_jit(
        "hisparse_request",
        "native_v3_cub",
        cuda_files=[
            str(Path(__file__).parent / "kernels" / "hisparse_request_native.cuh")
        ],
        cuda_wrappers=[
            ("update", "sglang::hisparse_native_request"),
            ("temporary_bytes", "sglang::hisparse_request_temporary_bytes"),
            ("dedup", "sglang::hisparse_request_dedup"),
        ],
    )


@cache_once
def kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def begin(epoch, counters, active, requests, B: tl.constexpr):
        i = tl.arange(0, B)
        tl.store(counters + i, 0, i < 8)
        tl.store(epoch, tl.load(epoch) + 1)
        tl.store(requests, tl.load(active) // 3)

    @triton.jit
    def dedup(
        selected, current, unique, access_time, epoch, L: tl.constexpr, B: tl.constexpr
    ):
        request = tl.program_id(0)
        i = tl.arange(0, B)
        history = tl.load(selected + request * 6144 + i, i < 6144, other=L)
        writer = tl.load(
            current + request * 3 + i - 6144, (i >= 6144) & (i < 6147), other=L
        ).to(tl.int32)
        keys = tl.sort(tl.where(i < 6144, history, writer), descending=False)
        previous = tl.gather(keys, tl.maximum(i - 1, 0), axis=0)
        valid = (keys > 0) & (keys < L) & ((i == 0) | (keys != previous))
        position = tl.cumsum(valid.to(tl.int32), axis=0) - 1
        count = tl.sum(valid.to(tl.int32), axis=0)
        tl.store(unique + request * 6147 + position, keys, valid)
        tl.store(unique + request * 6147 + i, -1, (i >= count) & (i < 6147))
        tl.store(access_time + keys, tl.load(epoch), valid)

    @triton.jit
    def plan(
        tags,
        previous,
        directory,
        current,
        miss_destination,
        miss_count,
        promote_rows,
        plan_rows,
        counters,
        CAPACITY: tl.constexpr,
        B: tl.constexpr,
    ):
        request = tl.program_id(0)
        i = tl.arange(0, B)
        count = tl.load(miss_count + request)
        # The public native body already published its tag changes. Recover the
        # old tags for the common commit, preserving its inverse-map assertions.
        base = tl.atomic_add(counters + 2, count)
        tl.atomic_add(counters + 1, count)
        tl.atomic_add(counters + 3, count)
        tl.atomic_add(counters + 4, count)
        slot = tl.load(miss_destination + request * 6147 + i, i < count, other=0)
        uid = tl.load(tags + slot, i < count, other=0)
        old = tl.load(previous + slot, i < count, other=0)
        location = tl.load(directory + uid, i < count, other=-2)
        cpu = -location - 2
        a = tl.load(current + request * 3)
        b = tl.load(current + request * 3 + 1)
        c = tl.load(current + request * 3 + 2)
        writer = (uid == a) | (uid == b) | (uid == c)
        j = base + i
        tl.store(promote_rows + j, tl.where(writer, -uid - 1, uid), i < count)
        tl.store(plan_rows + j, uid, i < count)
        tl.store(plan_rows + CAPACITY + j, slot, i < count)
        tl.store(
            plan_rows + 2 * CAPACITY + j, tl.where(writer, -cpu - 2, cpu), i < count
        )
        tl.store(plan_rows + 3 * CAPACITY + j, cpu, i < count)
        tl.store(plan_rows + 4 * CAPACITY + j, old, i < count)
        tl.store(tags + slot, old, i < count)

    return begin, dedup, plan


def create_workspace(state, queries, topk, *, dedup_backend="cub"):
    import torch

    if queries % 3 or not 0 < queries <= 384 or topk != 2048:
        raise ValueError("Requires MTP2,1..128requests, TopK2048")
    if state.gpu_tags.numel() != 1048576 or bool(state.leases.any()):
        raise ValueError("Requires a1Mhot pool and no external leases")
    requests = queries // 3
    if bool((state.gpu_tags[: requests * 8192] < 0).any()):
        raise ValueError("Every active request partition must be full")
    if dedup_backend not in ("cub", "triton"):
        raise ValueError("Expected cub or triton dedup backend")
    device = state.gpu_tags.device

    def allocate(shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device=device)

    result = RequestWorkspace(
        allocate((requests, 6147)),
        torch.arange(8192, dtype=torch.int16, device=device)
        .expand(requests, -1)
        .contiguous(),
        torch.arange(requests * 8192, dtype=torch.int32, device=device).reshape(
            requests, 8192
        ),
        allocate((requests, 6147)),
        torch.arange(requests, dtype=torch.int32, device=device),
        torch.full(
            (requests,), state.directory.numel(), dtype=torch.int64, device=device
        ),
        allocate(1),
        allocate((requests, 6147), torch.int64),
        allocate((requests, 6147)),
        allocate(requests),
        torch.empty_like(state.gpu_tags),
        allocate(requests * 6147),
        allocate(requests * 6147),
        torch.arange(requests + 1, dtype=torch.int32, device=device) * 6147,
        allocate(
            module().temporary_bytes(requests, state.directory.numel()), torch.uint8
        ),
        dedup_backend == "cub",
    )
    module()
    kernels()
    return result


def prepare(
    selected, current, active, owner, state, work, scratch, *, profile_events=None
):
    if owner != 0 or tuple(selected.shape) != (current.numel(), 2048):
        raise ValueError(
            "Requires owner-local selected[3*requests,2048] and current[3*requests]"
        )
    start, dedup, plan = kernels()
    requests = current.numel() // 3
    if profile_events is not None:
        profile_events[0].record()
    start[(1,)](state.epoch, work.counters, active, scratch.active_requests, 32)
    if scratch.use_cub:
        module().dedup(
            selected,
            current,
            scratch.unique_ids,
            scratch.ids,
            scratch.sorted_ids,
            scratch.offsets,
            scratch.temporary,
            state.directory,
            state.access_time,
            state.epoch,
        )
    else:
        dedup[(requests,)](
            selected,
            current,
            scratch.unique_ids,
            state.access_time,
            state.epoch,
            state.directory.numel(),
            8192,
            num_warps=8,
        )
    scratch.previous_tags.copy_(state.gpu_tags)
    if profile_events is not None:
        profile_events[1].record()
    module().update(
        scratch.unique_ids,
        state.gpu_tags,
        state.directory,
        scratch.locations,
        scratch.output,
        scratch.request_indices,
        scratch.sequence_lengths,
        scratch.lru_slots,
        scratch.active_requests,
        scratch.miss_source,
        scratch.miss_destination,
        scratch.miss_count,
    )
    if profile_events is not None:
        profile_events[2].record()
    plan[(requests,)](
        state.gpu_tags,
        scratch.previous_tags,
        state.directory,
        current,
        scratch.miss_destination,
        scratch.miss_count,
        work.promote_rows,
        work.plan,
        work.counters,
        work.new_rows.numel(),
        8192,
        num_warps=8,
    )
    if profile_events is not None:
        profile_events[3].record()
