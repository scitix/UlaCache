"""Pinned H200 compact FlashAG host descriptors, independent of the framework.

Packet address arithmetic is the same D31/LL protocol as b300_layout. The
device reader remains the existing H200 compact kernel, without pack/unpack.
"""

from typing import NamedTuple

from .b300_layout import (
    NUM_SLOTS as NUM_SLOTS,
    Layout as Layout,
    family_of,
    make_layout as make_layout,
    wire_packs as wire_packs,
)

SUPPORTED_WORLD_SIZES = (8,)
compact_only = True


class Band(NamedTuple):
    max_bytes: int
    algo: int
    name: str
    num_blocks: int
    num_threads: int


POLICY_BY_WORLD = {
    8: (
        Band(4 * 1024, 800, "ag_ll", 16, 256),
        Band(32 * 1024, 820, "ag_d31_n", 8, 256),
        Band(64 * 1024, 820, "ag_d31_n2", 16, 256),
        Band(256 * 1024, 820, "ag_d31_m", 192, 256),
        Band(1024 * 1024, 820, "ag_d31_w", 256, 256),
        Band(16384 * 1024, 820, "ag_d31_x", 1024, 256),
    )
}


def target():
    import torch

    return "h200" if torch.cuda.get_device_capability() == (9, 0) else None


def counter_slots(world):
    result = {}
    for band in POLICY_BY_WORLD.get(world, ()):
        family = family_of(band.algo, band.num_blocks, band.num_threads)
        result[family] = max(result.get(family, 0), band.num_blocks * band.num_threads)
    return result
