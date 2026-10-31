"""Explicit ragged FP32 input extension of the pinned H200 FlashAG protocol.

The existing CP page-table kernel produces aligned row prefixes and a device
payload count once per forward. Dense payloads retain the original linear
copy. No pack/unpack kernels, temporary score buffers or host length reads.
"""

import torch
from ulacache.compiler import cache_once, load_jit


@cache_once
def _module(world):
    if world != 8:
        raise ValueError("The H200 compact Index transport requires eight ranks")
    source = "csrc/flashag/h200/compact_adapter.cuh"
    return load_jit(
        "udcache_index_ragged_v1",
        str(world),
        cuda_files=[str(source)],
        cuda_wrappers=[("all_gather", f"direct_compact_all_gather<{world}>")],
    )


def compile_compact(world):
    _module(world)


def all_gather(owner, output, local, prefix, packs):
    if (
        not owner.compact_scores
        or local.ndim != 2
        or output.ndim != 2
        or local.dtype != torch.float32
        or output.dtype != torch.float32
        or local.device != owner.device
        or output.device != owner.device
        or not local.is_contiguous()
        or not output.is_contiguous()
        or output.shape != (owner.world_size * local.shape[0], local.shape[1])
        or output.numel() * 4 > owner.maximum_gather_bytes
        or output.data_ptr() != owner.gathered_workspace.data_ptr()
        or prefix.shape != (local.shape[0] + 1,)
        or prefix.dtype != torch.int32
        or packs.shape != (1,)
        or packs.dtype != torch.int64
        or prefix.device != owner.device
        or packs.device != owner.device
        or not prefix.is_contiguous()
        or not packs.is_contiguous()
        or local.data_ptr() % 16
        or output.data_ptr() % 16
    ):
        raise ValueError("Compact Index scores require stable aligned owner workspaces")
    band = next(b for b in owner._table if output.numel() * 4 <= b.max_bytes)
    if band.algo != 820:
        raise ValueError("Compact Index scores require the validated D31 band")
    family = owner._ops.family_of(band.algo, band.num_blocks, band.num_threads)
    region, stride, size = owner.layout.bases(family)
    _module(owner.world_size).all_gather(
        local,
        output,
        prefix,
        packs,
        owner.slab.data_ptr(),
        int(owner._handle.multicast_ptr),
        owner._peer_table.data_ptr(),
        owner.rank_in_group,
        region,
        stride,
        size,
        band.num_blocks,
        band.num_threads,
        owner._counters[family],
    )
