"""Full-record equivalence of bounded, quiescent request restore mappings."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from ulacache.tools.snapshot.restore import PreparedSequenceRestore, write_sequence_records


def make_cache(width, fragmented):
    layers, groups, slots, count = 78, 21, 192, 101
    directory = torch.full((groups, slots * 2), -1, dtype=torch.int64)
    rows = torch.arange(count, dtype=torch.int64) * 3 + 1
    generator = torch.Generator().manual_seed(176)
    for group in range(groups):
        # Each shared group intentionally has a different residency pattern.
        positions = (torch.arange(count) + group) % 3
        gpu = positions != 0
        for cpu, mask in [(False, gpu), (True, ~gpu)]:
            n = int(mask.sum())
            destinations = (torch.randperm(slots, generator=generator)[:n]
                            if fragmented else torch.arange(n) + 7)
            directory[group, rows[mask]] = -destinations - 2 if cpu else destinations
    return SimpleNamespace(
        groups=SimpleNamespace(group=lambda layer: layer % groups),
        directory=directory, row_bytes=width, gpu_slots=slots, host_slots=slots,
        gpu=torch.full((layers, slots, width), 173, dtype=torch.uint8),
        host=SimpleNamespace(storage=torch.full((layers, slots, width), 173, dtype=torch.uint8))), rows


@pytest.mark.parametrize("width", [576, 656])
@pytest.mark.parametrize("fragmented", [False, True])
def test_all_78_layers_exact_bytes_and_untouched_records(width, fragmented):
    candidate, rows = make_cache(width, fragmented)
    original, _ = make_cache(width, fragmented)
    quiesce = Mock()
    prepared = PreparedSequenceRestore(candidate, rows, quiesce=quiesce)
    quiesce.assert_called_once_with()
    for layer in range(78):
        payload = ((torch.arange(len(rows) * width, dtype=torch.int32).reshape(-1, width)
                    + layer * 37) % 251).to(torch.uint8)
        for start in range(0, len(rows), 17):
            chunk = payload[start:start + 17]
            prepared.write(layer, start, chunk, verify=True)
            write_sequence_records(original, layer, rows[start:start + len(chunk)], chunk,
                                   verify=True)
    assert torch.equal(candidate.gpu, original.gpu)
    assert torch.equal(candidate.host.storage, original.host.storage)
    assert torch.equal(candidate.directory, original.directory)


def test_reject_unallocated_and_invalid_destinations_before_writes():
    cache, rows = make_cache(576, False)
    cache.directory[0, rows[0]] = -1
    with pytest.raises(ValueError, match="allocated"):
        PreparedSequenceRestore(cache, rows, quiesce=lambda: None)
    cache.directory[0, rows[0]] = cache.gpu_slots
    prepared = PreparedSequenceRestore(cache, rows, quiesce=lambda: None)
    with pytest.raises(ValueError, match="outside"):
        prepared.write(0, 0, torch.zeros((1, 576), dtype=torch.uint8))
    assert torch.all(cache.gpu == 173)


def test_prepared_mapping_is_a_copy_and_checks_request_bounds():
    cache, rows = make_cache(576, False)
    prepared = PreparedSequenceRestore(cache, rows, quiesce=lambda: None)
    before = prepared.locations.clone()
    cache.directory.fill_(-1)
    assert torch.equal(prepared.locations, before)
    # This does not authorize writes after a cache mutation: the public contract
    # explicitly requires the allocator/residency namespace to stay quiescent.
    with pytest.raises(ValueError, match="outside"):
        prepared.write(0, len(rows), torch.zeros((1, 576), dtype=torch.uint8))
