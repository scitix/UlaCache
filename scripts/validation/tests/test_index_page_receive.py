"""Native FP8 key and FP32 scale bytes must survive owner-page redistribution."""
from dataclasses import replace
import struct

import pytest

from ulacache.transfer.index_pages import (
    PAGE_BYTES, plan_index_record_copies, publish_index_record_bytes,
)


def put_native_record(buffer, row, key, scale):
    page, token = divmod(row, 64)
    start = page * 8448
    buffer[start + token * 128:start + (token + 1) * 128] = key
    buffer[start + 8192 + token * 4:start + 8192 + (token + 1) * 4] = scale


def test_all_eight_owners_preserve_21_native_banks_and_partial_page_sentinels():
    # Fragment both native logical pages and registered receive leases. The last
    # source page carries only61 real rows; its other bytes cannot be published.
    logical_pages = [8, 19, 11, 32, 9, 24, 18, 37]
    receive_pages = [3, 8, 0, 5, 7, 2, 9, 1]
    logical = [page * 64 + token for page in logical_pages for token in range(64)][:509]
    received = [page * 64 + token for page in receive_pages for token in range(64)][:509]
    owner_plans = [plan_index_record_copies(logical, received, degree=8, rank=rank)
                   for rank in range(8)]
    assert sorted(item.logical_record for plan in owner_plans for item in plan) == sorted(logical)
    for bank in range(21):
        source = bytearray([0xF7] * (10 * PAGE_BYTES))
        expected = [bytearray([0xDB] * (5 * PAGE_BYTES)) for _ in range(8)]
        for row, receive in zip(logical, received):
            key = bytes((row + bank * 17 + column) % 256 for column in range(128))
            # Preserve NaN scale payload bits without re-quantizing or
            # normalizing values. The draft case also covers zero/infinity.
            scale = struct.pack('<I', (0x7FC00000 + row + bank) & 0xFFFFFFFF)
            put_native_record(source, receive, key, scale)
            put_native_record(expected[row & 7], row >> 3, key, scale)
        for rank, plan in enumerate(owner_plans):
            destination = bytearray([0xDB] * (5 * PAGE_BYTES))
            assert publish_index_record_bytes(source, destination, plan,
                transport_quiesced=True) == len(plan)
            assert destination == expected[rank]


def test_draft_index_keeps_replicated_native_rows_and_exact_scale_bits():
    logical, received = [512, 575, 576], [0, 66, 129]
    source, destination = bytearray(3 * PAGE_BYTES), bytearray([0xA3] * (10 * PAGE_BYTES))
    expected = bytearray(destination)
    for index, (row, receive) in enumerate(zip(logical, received)):
        key, scale = bytes([index + 17] * 128), struct.pack('<I', [0, 0x80000000, 0x7F800000][index])
        put_native_record(source, receive, key, scale)
        put_native_record(expected, row, key, scale)
    plan = plan_index_record_copies(logical, received, degree=1, rank=0)
    publish_index_record_bytes(source, destination, plan, transport_quiesced=True)
    assert destination == expected


@pytest.mark.parametrize('logical,received,degree,rank', [
    ([512], [], 8, 0), ([512, 512], [0, 1], 8, 0),
    ([512, 520], [0, 0], 8, 0), ([511], [0], 8, 7),
    ([512], [-1], 8, 0), ([True], [0], 8, 0),
    ([512], [0], 3, 0), ([512], [0], 8, 8),
])
def test_invalid_ownership_and_record_plans_fail_before_publication(logical, received, degree, rank):
    with pytest.raises(ValueError):
        plan_index_record_copies(logical, received, degree=degree, rank=rank)


def test_publication_requires_quiescence_and_validates_all_records_before_writing():
    plan = plan_index_record_copies([512, 576], [0, 1], degree=8, rank=0)
    source, destination = bytearray(PAGE_BYTES), bytearray([0xBA] * (3 * PAGE_BYTES))
    before = bytes(destination)
    with pytest.raises(RuntimeError):
        publish_index_record_bytes(source, destination, plan, transport_quiesced=False)
    with pytest.raises(ValueError):
        publish_index_record_bytes(source, destination,
            [plan[0], replace(plan[1], destination_key_byte=999)], transport_quiesced=True)
    with pytest.raises(ValueError):
        publish_index_record_bytes(source, destination,
            [plan[0], replace(plan[1], received_record=640, source_key_byte=10*PAGE_BYTES,
                source_scale_byte=10*PAGE_BYTES+8192)], transport_quiesced=True)
    assert bytes(destination) == before


def test_plan_cannot_change_owner_or_publish_into_a_reserved_page():
    item = plan_index_record_copies([512], [0], degree=8, rank=0)[0]
    for changes in [dict(rank=1), dict(logical_record=513), dict(received_record=True),
                    dict(logical_record=504, destination_record=63,
                         destination_key_byte=63*128, destination_scale_byte=8192+63*4)]:
        with pytest.raises(ValueError):
            replace(item, **changes)


def test_receive_and_canonical_storage_cannot_alias_or_use_partial_pages():
    item = plan_index_record_copies([512], [0], degree=8, rank=0)[0]
    storage = bytearray(2 * PAGE_BYTES)
    with pytest.raises(ValueError):
        publish_index_record_bytes(storage, storage, [item], transport_quiesced=True)
    with pytest.raises(ValueError):
        publish_index_record_bytes(bytearray(PAGE_BYTES-1), storage, [item], transport_quiesced=True)


def test_owner_and_replicated_draft_layouts_cannot_share_a_publication_bank():
    owner = plan_index_record_copies([512], [0], degree=8, rank=0)
    draft = plan_index_record_copies([576], [1], degree=1, rank=0)
    destination = bytearray([0xBA] * (10 * PAGE_BYTES))
    before = bytes(destination)
    with pytest.raises(ValueError):
        publish_index_record_bytes(bytearray(PAGE_BYTES), destination,
            owner + draft, transport_quiesced=True)
    assert bytes(destination) == before
