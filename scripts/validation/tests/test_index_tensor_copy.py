"""Real CPU tensor scatter against the native Index byte oracle."""
import pytest

torch = pytest.importorskip('torch')

from ulacache.transfer.index_pages import (
    PAGE_BYTES, plan_index_record_copies, publish_index_record_bytes,
)
from ulacache.transfer.index_tensor_copy import publish_index_record_tensors


@pytest.mark.parametrize('rank', range(8))
def test_tensor_publication_matches_all_21_owner_banks_and_partial_tails(rank):
    logical = [page * 64 + offset for page in [8, 19, 11, 32, 9, 24, 18, 37]
               for offset in range(64)][:509]
    received = [page * 64 + offset for page in [3, 8, 0, 5, 7, 2, 9, 1]
                for offset in range(64)][:509]
    plan = plan_index_record_copies(logical, received, degree=8, rank=rank)
    pattern = torch.arange(10 * PAGE_BYTES, dtype=torch.int64)
    for bank in range(21):
        source = ((pattern + bank * 17) % 256).to(torch.uint8).reshape(10, PAGE_BYTES)
        source_before = source.clone()
        destination = torch.full((5, PAGE_BYTES), 0xDB, dtype=torch.uint8)
        expected = bytearray([0xDB] * destination.nbytes)
        publish_index_record_bytes(bytes(source.reshape(-1).tolist()), expected,
            plan, transport_quiesced=True)
        assert publish_index_record_tensors(source, destination, plan,
            transport_quiesced=True, copy_chunk_records=17) == len(plan)
        assert bytes(destination.reshape(-1).tolist()) == expected
        assert torch.equal(source, source_before)


def test_tensor_draft_is_replicated_and_preserves_raw_scale_payload_bits():
    source = torch.arange(3 * PAGE_BYTES, dtype=torch.int64).remainder(256).to(torch.uint8)
    # Signed zero, infinity and a NaN payload remain byte copies.
    source[8192:8196] = torch.tensor([0, 0, 0, 128], dtype=torch.uint8)
    source[PAGE_BYTES + 8192 + 8:PAGE_BYTES + 8192 + 12] = torch.tensor(
        [0, 0, 128, 127], dtype=torch.uint8)
    source[2 * PAGE_BYTES + 8192 + 4:2 * PAGE_BYTES + 8192 + 8] = torch.tensor(
        [37, 1, 192, 127], dtype=torch.uint8)
    destination = torch.full((10 * PAGE_BYTES,), 0xA3, dtype=torch.uint8)
    expected = bytearray([0xA3] * destination.nbytes)
    plan = plan_index_record_copies([512, 575, 576], [0, 66, 129], degree=1, rank=0)
    publish_index_record_bytes(bytes(source.tolist()), expected, plan, transport_quiesced=True)
    assert publish_index_record_tensors(source, destination, plan,
        transport_quiesced=True, copy_chunk_records=1) == 3
    assert bytes(destination.tolist()) == expected


def test_tensor_receive_rejects_late_out_of_bounds_and_unretired_writes_before_mutation():
    source = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
    destination = torch.full((3 * PAGE_BYTES,), 0xBA, dtype=torch.uint8)
    before = destination.clone()
    good = plan_index_record_copies([512], [0], degree=8, rank=0)
    invalid = plan_index_record_copies([576], [640], degree=8, rank=0)
    with pytest.raises(RuntimeError, match='retired'):
        publish_index_record_tensors(source, destination, good, transport_quiesced=False)
    with pytest.raises(ValueError, match='exceeds'):
        publish_index_record_tensors(source, destination, good + invalid, transport_quiesced=True)
    assert torch.equal(destination, before)


def test_tensor_receive_rejects_alias_layout_and_invalid_chunks():
    source = torch.zeros(3 * PAGE_BYTES, dtype=torch.uint8)
    destination = torch.zeros_like(source)
    plan = plan_index_record_copies([512], [0], degree=8, rank=0)
    for target in [source, source.view(3, PAGE_BYTES).transpose(0, 1),
                   destination.to(torch.int16), destination[:-1]]:
        with pytest.raises(ValueError):
            publish_index_record_tensors(source, target, plan, transport_quiesced=True)
    for chunk in [0, -1, True]:
        with pytest.raises(ValueError):
            publish_index_record_tensors(source, destination, plan,
                transport_quiesced=True, copy_chunk_records=chunk)
    assert publish_index_record_tensors(source, destination, (), transport_quiesced=True) == 0
