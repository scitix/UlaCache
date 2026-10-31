"""Preserve native packed-region identity while moving opaque token records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackedRegionLayout:
    tokens_per_region: int = 8
    region_slots_per_query: int = 513
    logical_position_bits: int = 32

    @property
    def selected_token_width(self):
        return self.tokens_per_region * self.region_slots_per_query

    def __post_init__(self):
        if (
            self.tokens_per_region < 1
            or self.region_slots_per_query < 1
            or self.logical_position_bits != 32
        ):
            raise ValueError(
                "Declare positive region dimensions and the native 32/32 packed ABI"
            )


def expand_packed_regions_reference(
    packed, counts, *, layout=PackedRegionLayout(), active_rows=None
):
    """Expand complete physical regions; native attention retains causal masks.

    An invalid region expands to distinct negative tokens. The complete set of
    negative sentinels occupies one region in an eager sorted unique union,
    keeping every valid region aligned without changing the common planner.
    """
    if len(packed) != len(counts):
        raise ValueError("Packed rows and native counts must agree")
    active_rows = len(packed) if active_rows is None else active_rows
    if type(active_rows) is not int or not 0 <= active_rows <= len(packed):
        raise ValueError("Active rows must fit the padded selection rows")
    result = []
    for row, (regions, count) in enumerate(zip(packed, counts)):
        if len(regions) != layout.region_slots_per_query or not 0 <= count <= len(
            regions
        ):
            raise ValueError("Native region selection differs from its declared width")
        records = []
        for column, region in enumerate(regions):
            valid = row < active_rows and column < count and region >= 0
            start = (
                (region >> layout.logical_position_bits) * layout.tokens_per_region
                if valid
                else -layout.tokens_per_region
            )
            if valid and not 0 <= start <= 2**31 - layout.tokens_per_region:
                raise ValueError("Native physical token IDs exceed the int32 namespace")
            records.extend(start + offset for offset in range(layout.tokens_per_region))
        result.append(tuple(records))
    return tuple(result)


def remap_packed_regions_reference(
    packed, resident_records, *, layout=PackedRegionLayout()
):
    result = []
    mask = (1 << layout.logical_position_bits) - 1
    if len(packed) != len(resident_records):
        raise ValueError("Packed regions and resident token rows must agree")
    for regions, records in zip(packed, resident_records):
        if (
            len(regions) != layout.region_slots_per_query
            or len(records) != layout.selected_token_width
        ):
            raise ValueError("Resident token rows must include every complete region")
        output = []
        for column, region in enumerate(regions):
            first = records[column * layout.tokens_per_region]
            if region < 0 or first < 0:
                output.append(-1)
                continue
            group = records[
                column * layout.tokens_per_region : (column + 1)
                * layout.tokens_per_region
            ]
            if first % layout.tokens_per_region or tuple(group) != tuple(
                range(first, first + layout.tokens_per_region)
            ):
                raise ValueError(
                    "Resident native regions must remain aligned and contiguous"
                )
            output.append(
                ((first // layout.tokens_per_region) << layout.logical_position_bits)
                | (region & mask)
            )
        result.append(tuple(output))
    return tuple(result)
