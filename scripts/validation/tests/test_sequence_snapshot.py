"""Canonical order and planar Index ABI across a change in ownership topology."""
import pytest
import importlib.util

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                reason="Sequence byte oracles require optional Torch")

from ulacache.tools import SnapshotLayout, SnapshotReader, SnapshotRequest, SnapshotWriter, finalize_snapshot
from ulacache.tools.snapshot.restore import iter_sequence_plane, write_index_records


def snapshot(path):
    import torch
    layout = SnapshotLayout("bounded-fixture", 3, 2, 17, (0,), (7, 4), 8)
    request = SnapshotRequest("ragged", tuple(range(19)), (42,))
    canonical = {plane: torch.tensor([[(position * 29 + column * 7) % 256
        for column in range(width)] for position in range(19)], dtype=torch.uint8)
        for plane, width in layout.planes().items()}
    for owner in range(3):
        writer = SnapshotWriter(path, layout, [request], owner)
        for plane, value in canonical.items():
            raw = bytes(value[owner::3].reshape(-1).tolist())
            writer.write_plane(request.key, plane, [raw[i:i + 7] for i in range(0, len(raw), 7)])
        writer.finish()
    finalize_snapshot(path)
    return SnapshotReader(path), canonical


def test_reconstructs_ragged_owner_stripes_in_position_order(tmp_path):
    import torch
    reader, expected = snapshot(tmp_path)
    for plane, value in expected.items():
        for rows in (3, 6, 12, 24):
            chunks = list(iter_sequence_plane(reader, "ragged", plane, chunk_rows=rows))
            assert [start for start, _ in chunks] == list(range(0, 19, rows))
            assert torch.equal(torch.cat([data for _, data in chunks]), value)


def test_bridge_consumes_checksums_before_a_cohort_can_publish(tmp_path):
    reader, _ = snapshot(tmp_path)
    path = tmp_path / "owner_002/request_00000.mla_000.bin"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        list(iter_sequence_plane(reader, "ragged", "mla_000", chunk_rows=6))


def test_native_planar_index_writer_preserves_key_and_scale_bytes():
    import torch
    rows = torch.tensor([9, 22, 17, 31, 6], dtype=torch.int64)
    data = torch.arange(55, dtype=torch.uint8).reshape(5, 11)
    target = torch.full((5, 88), 255, dtype=torch.uint8)
    write_index_records(target, rows, data, page_rows=8, segment_bytes=(7, 4))
    for index, row in enumerate(rows.tolist()):
        page, offset = divmod(row, 8)
        assert torch.equal(target[page, offset * 7:(offset + 1) * 7], data[index, :7])
        assert torch.equal(target[page, 56 + offset * 4:56 + (offset + 1) * 4], data[index, 7:])
    assert torch.equal(target[4], torch.full((88,), 255, dtype=torch.uint8))
