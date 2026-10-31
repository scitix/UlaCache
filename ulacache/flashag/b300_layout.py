# SPDX-License-Identifier: Apache-2.0
"""Original B300 D31 slot layout for the library's packed Index transport.

Only host address arithmetic is provided here. Kernel bodies and launch
parameters remain in the existing packed transport module.
"""

from typing import Dict, NamedTuple, Tuple

NUM_SLOTS = 4
MODE_LL, MODE_LINE, MODE_D31 = 0, 1, 2
ALGO_MODE = {800: 0, 810: 1, 820: 2, 830: 2, 840: 0, 841: 0, 842: 1, 843: 2}
MODE_NAME = {0: "ll", 1: "ln", 2: "d31"}


def target():
    """Resolve only the measured B300 architecture, after CUDA device selection."""
    import torch

    return "b300" if torch.cuda.get_device_capability() == (10, 3) else None


def family_of(algo: int, num_blocks: int, num_threads: int) -> str:
    """A family is (wire layout, GRID) -- and the grid half is not cosmetic.

    ⚠ A FAMILY OWNS A SLAB REGION, A COUNTER REGION AND ONE FIXED GRID, and those
    are one decision. The device slot rotation counts one bump per WARP per call, so
    two bands sharing a counter at different grids leave the wider grid's extra
    warps holding a stale count: sender and receiver then derive different tags and
    the receiver spins forever. That is a HANG, not a slow row.

    ⚠ ``kSend`` IS ABSENT ON PURPOSE. The two send instructions write identical
    bytes, so they share a region and a grid; adding the send to this name would
    allocate a second region for the same wire layout. If a future send instruction
    ever changes the wire footprint, this signature and ``wire_packs`` both have to
    grow together.
    """
    return f"{MODE_NAME[algo_mode_of(algo)]}_{num_blocks}x{num_threads}"


def algo_mode_of(algo: int) -> int:
    """The WIRE LAYOUT an algo uses -- the slab's question, not the kernel's.

    For every id in 800-830 this returns what h200's ``algo_mode_of`` returned,
    because MODE_UNI is an alias for MODE_D31 here (see the MODE_UNI comment).
    """
    return ALGO_MODE[algo]


def wire_packs(mode: int, cm: int) -> int:
    """16 B wire units one source's ``cm``-pack shard occupies under ``mode``."""
    if mode == MODE_LL:
        return 2 * cm
    if mode == MODE_LINE:
        return -(-cm // 7) * 8
    w = 4 * cm  # D31 (and MODE_UNI, which is its alias) -- 31 data words per line
    return -(-w // 31) * 8


class Layout(NamedTuple):
    """Byte offsets of every region.

    Per-slot strides come from each mode's own band ceiling, never from the current
    message: a layout that moved with the message would let a stale slice from a
    bigger call land where a smaller one expects a different source's slot.
    """

    world: int
    slots: int
    total: int
    off: Dict[str, int]
    slice_packs: Dict[str, int]

    def bases(self, fam: str) -> Tuple[int, int, int]:
        """``(region, slot_stride, slice)``, all in 16 B pack units."""
        sl = self.slice_packs[fam]
        return self.off[fam] // 16, self.world * sl, sl


def make_layout(world: int, ceilings: Dict[str, int], slots: int = NUM_SLOTS) -> Layout:
    """``ceilings`` maps family -> the largest TOTAL payload that family serves."""
    off: Dict[str, int] = {}
    slice_packs: Dict[str, int] = {}
    cur = 0
    for fam in sorted(ceilings):
        cap = ceilings[fam]
        if cap <= 0:
            continue
        mode = (
            MODE_LL
            if fam.startswith("ll")
            else MODE_LINE
            if fam.startswith("ln")
            else MODE_D31
        )
        # The largest per-rank shard this family can be handed, in packs, then the
        # wire units that shard needs.
        cm = -(-(cap // 16) // world)
        sl = wire_packs(mode, cm)
        off[fam] = cur
        slice_packs[fam] = sl
        cur += slots * world * sl * 16
    return Layout(world=world, slots=slots, total=cur, off=off, slice_packs=slice_packs)
