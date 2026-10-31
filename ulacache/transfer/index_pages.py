"""Byte-exact native Index page plans for ordinary owner-striped TP.

Native page64 stores 64 FP8 keys followed by 64 FP32 scales. Selecting owners
must preserve this page layout; a page is not an array of 132-byte records.
The native allocator and transport own record allocation and write retirement.
This module imports no framework or device runtime.
"""

from dataclasses import dataclass


PAGE_RECORDS = 64
KEY_BYTES = 128
SCALE_BYTES = 4
PAGE_BYTES = PAGE_RECORDS * (KEY_BYTES + SCALE_BYTES)


@dataclass(frozen=True)
class IndexRecordCopy:
    degree: int
    rank: int
    logical_record: int
    received_record: int
    destination_record: int
    source_key_byte: int
    destination_key_byte: int
    source_scale_byte: int
    destination_scale_byte: int

    def __post_init__(self):
        if any(type(value) is not int for value in vars(self).values()):
            raise ValueError("Index record coordinates must be integers")
        if (
            self.degree < 1
            or self.degree & (self.degree - 1)
            or not 0 <= self.rank < self.degree
            or self.logical_record < self.degree * PAGE_RECORDS
            or self.logical_record % self.degree != self.rank
            or self.destination_record != self.logical_record // self.degree
            or self.received_record < 0
            or (self.source_key_byte, self.source_scale_byte)
            != _record_offsets(self.received_record)
            or (self.destination_key_byte, self.destination_scale_byte)
            != _record_offsets(self.destination_record)
        ):
            raise ValueError(
                "Index record plan disagrees with native owner/page coordinates"
            )


def _record_offsets(record):
    page, offset = divmod(record, PAGE_RECORDS)
    base = page * PAGE_BYTES
    return (
        base + offset * KEY_BYTES,
        base + PAGE_RECORDS * KEY_BYTES + offset * SCALE_BYTES,
    )


def plan_index_record_copies(logical_records, received_records, *, degree, rank):
    """Pair registered receive rows with canonical native Index destinations.

    Ordinary TP owns logical row ``r`` on rank ``r % degree`` at compact row
    ``r // degree``. Degree1 is the replicated native draft Index bank. The
    reserved first allocation superpage is never a real request destination.
    Request-policy physical page remapping must use its own explicit binding.
    """
    if (
        type(degree) is not int
        or degree < 1
        or degree & (degree - 1)
        or type(rank) is not int
        or not 0 <= rank < degree
    ):
        raise ValueError("Index ownership needs a power-of-two degree and valid rank")
    logical, received = tuple(logical_records), tuple(received_records)
    if len(logical) != len(received):
        raise ValueError("Logical and received Index record counts differ")
    if any(
        type(row) is not int or row < degree * PAGE_RECORDS for row in logical
    ) or any(type(row) is not int or row < 0 for row in received):
        raise ValueError(
            "Index records must exclude native padding and negative receive rows"
        )
    if len(set(logical)) != len(logical) or len(set(received)) != len(received):
        raise ValueError("Index transfer records must be unique")
    result = []
    for logical_record, received_record in zip(logical, received):
        if logical_record % degree != rank:
            continue
        destination_record = logical_record // degree
        source_key, source_scale = _record_offsets(received_record)
        destination_key, destination_scale = _record_offsets(destination_record)
        result.append(
            IndexRecordCopy(
                degree,
                rank,
                logical_record,
                received_record,
                destination_record,
                source_key,
                destination_key,
                source_scale,
                destination_scale,
            )
        )
    return tuple(result)


def publish_index_record_bytes(source, destination, plan, *, transport_quiesced):
    """Copy native CPU page bytes after the caller retires remote writes.

    This bounded CPU implementation also provides a byte oracle for a device
    binding. It neither registers addresses nor allocates/frees native rows.
    Validate the entire plan before changing any destination byte.
    """
    if transport_quiesced is not True:
        raise RuntimeError("Index publication requires retired remote writes")
    source_view, destination_view = memoryview(source), memoryview(destination)
    if not source_view.c_contiguous or not destination_view.c_contiguous:
        raise ValueError("Native Index page buffers must be contiguous")
    if source_view.obj is destination_view.obj:
        raise ValueError("Receive staging and canonical Index storage must not alias")
    source_view, destination_view = source_view.cast("B"), destination_view.cast("B")
    if (
        destination_view.readonly
        or source_view.nbytes % PAGE_BYTES
        or destination_view.nbytes % PAGE_BYTES
    ):
        raise ValueError(
            "Index storage must contain complete native pages and writable destinations"
        )
    plans = validate_index_record_plan(
        plan, source_view.nbytes, destination_view.nbytes
    )
    for item in plans:
        for source_start, destination_start, width in [
            (item.source_key_byte, item.destination_key_byte, KEY_BYTES),
            (item.source_scale_byte, item.destination_scale_byte, SCALE_BYTES),
        ]:
            destination_view[destination_start : destination_start + width] = (
                source_view[source_start : source_start + width]
            )
    return len(plans)


def validate_index_record_plan(plan, source_bytes, destination_bytes):
    """Validate a complete byte plan before CPU or device publication begins."""
    if (
        type(source_bytes) is not int
        or type(destination_bytes) is not int
        or source_bytes < 0
        or destination_bytes < 0
        or source_bytes % PAGE_BYTES
        or destination_bytes % PAGE_BYTES
    ):
        raise ValueError("Index storage must contain complete native pages")
    plans = tuple(plan)
    if any(type(item) is not IndexRecordCopy for item in plans):
        raise ValueError("Index publication requires validated native record plans")
    if len({(item.degree, item.rank) for item in plans}) > 1:
        raise ValueError("One Index bank must use one owner layout")
    if len({item.destination_record for item in plans}) != len(plans):
        raise ValueError("Index publication cannot write a destination twice")
    for item in plans:
        source_offsets = _record_offsets(item.received_record)
        destination_offsets = _record_offsets(item.destination_record)
        if (
            (item.source_key_byte, item.source_scale_byte) != source_offsets
            or (item.destination_key_byte, item.destination_scale_byte)
            != destination_offsets
            or item.received_record < 0
            or item.destination_record < PAGE_RECORDS
        ):
            raise ValueError("Index record plan disagrees with the native page ABI")
        for start, width, capacity in [
            (item.source_key_byte, KEY_BYTES, source_bytes),
            (item.source_scale_byte, SCALE_BYTES, source_bytes),
            (item.destination_key_byte, KEY_BYTES, destination_bytes),
            (item.destination_scale_byte, SCALE_BYTES, destination_bytes),
        ]:
            if start < 0 or start + width > capacity:
                raise ValueError("Index record copy exceeds native storage")
    return plans
