"""CPU-only direct-read byte, integrity and caller-buffer lease checks."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ulacache.tools.snapshot.format import (
    SnapshotLayout, SnapshotReader, SnapshotRequest, SnapshotWriter, finalize_snapshot,
)

from ulacache.transfer import file_io as direct


def make_snapshot(path, *, owners=1, rows=23):
    layout = SnapshotLayout("actual-canonical-byte-fixture", owners, 2, 656, (0,))
    request = SnapshotRequest("saved-request", tuple(range(rows)), (17,))
    payload = {plane: bytes((i * 7 + tag) % 251 for i in range(rows * width))
               for tag, (plane, width) in enumerate(layout.planes().items())}
    for owner in range(owners):
        writer = SnapshotWriter(path, layout, [request], owner)
        for plane, width in layout.planes().items():
            data = payload[plane]
            striped = b"".join(data[i * width:(i + 1) * width]
                               for i in range(owner, rows, owners))
            writer.write_plane(request.key, plane, (striped,))
        writer.finish()
    finalize_snapshot(path)
    return SnapshotReader(path), request, payload


@pytest.fixture
def torch():
    value = pytest.importorskip("torch")
    assert not value.cuda.is_initialized()
    previous = value.get_num_threads()
    value.set_num_threads(1)
    yield value
    assert not value.cuda.is_initialized()
    value.set_num_threads(previous)


def test_optional_import_is_dependency_free():
    code = ("import sys; import ulacache.transfer.file_io; "
            "assert 'torch' not in sys.modules; "
            "assert not any(n=='sglang' or n.startswith('sglang.') for n in sys.modules)")
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


def test_ragged_chunks_equal_actual_old_iterator(tmp_path, torch):
    from ulacache.tools.snapshot.restore import iter_sequence_plane
    source, request, payload = make_snapshot(tmp_path / "snapshot")
    for plane, width in source.layout.planes().items():
        old = list(iter_sequence_plane(source, request.key, plane, chunk_rows=7))
        buffer = torch.empty((7, width), dtype=torch.uint8)
        with direct.CanonicalPlaneReader(source, request.key, plane) as reader:
            for (start, expected), count in zip(old, [7, 7, 7, 2]):
                chunk = reader.readinto(buffer)
                assert chunk.start == start and chunk.data.shape == (count, width)
                assert chunk.data.data_ptr() == buffer.data_ptr()
                assert torch.equal(chunk.data, expected)
                chunk.release()
            assert reader.readinto(buffer) is None
            assert reader.statistics()["payload_sha256_verified"]
        stats = reader.statistics()
        assert stats["bytes_read"] == len(payload[plane]) and stats["chunks_read"] == 4
        assert stats["borrowed_buffers"] == 0 and stats["closed"]
        assert stats["whole_call_wall_seconds"] >= 0


def test_two_caller_buffers_and_completion_boundary(tmp_path, torch):
    source, request, _ = make_snapshot(tmp_path / "snapshot")
    buffers = [torch.empty((7, 656), dtype=torch.uint8) for _ in range(2)]
    with direct.CanonicalPlaneReader(source, request.key, "mla_000") as reader:
        first = reader.readinto(buffers[0]); saved = first.data.clone()
        second = reader.readinto(buffers[1])
        assert torch.equal(first.data, saved)
        with pytest.raises(RuntimeError, match="still owns"):
            reader.readinto(buffers[0])
        # Aliases are checked by memory interval, not Python tensor identity.
        with pytest.raises(RuntimeError, match="still owns"):
            reader.readinto(buffers[1][1:])
        completed = []
        first.release(after=lambda: completed.append("consumer completed"))
        third = reader.readinto(buffers[0]); assert third.start == 14
        assert completed == ["consumer completed"]
        second.release(); third.release()
        last = reader.readinto(buffers[1]); assert last.start == 21
        last.release(); last.release()  # Release is idempotent.
        assert reader.statistics()["borrowed_buffers"] == 0


def test_wait_failure_keeps_lease_and_close_does_not_free_it(tmp_path, torch):
    source, request, _ = make_snapshot(tmp_path / "snapshot")
    reader = direct.CanonicalPlaneReader(source, request.key, "mla_000")
    buffer = torch.empty((7, 656), dtype=torch.uint8)
    chunk = reader.readinto(buffer)
    def failed_wait():
        raise RuntimeError("Consumer completion is unknown")
    with pytest.raises(RuntimeError, match="completion is unknown"):
        chunk.release(after=failed_wait)
    assert not chunk.released and reader.statistics()["borrowed_buffers"] == 1
    reader.close()
    assert reader.statistics()["borrowed_buffers"] == 1
    assert chunk.data.data_ptr() == buffer.data_ptr()
    with pytest.raises(RuntimeError, match="closed"):
        reader.readinto(buffer)
    chunk.release(); assert reader.statistics()["borrowed_buffers"] == 0


def test_payload_policy_and_hash_failure(tmp_path, torch):
    source, request, _ = make_snapshot(tmp_path / "snapshot")
    path = source.path / "owner_000/request_00000.mla_000.bin"
    raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
    buffer = torch.empty((31, 656), dtype=torch.uint8)
    with direct.CanonicalPlaneReader(source, request.key, "mla_000") as reader:
        with pytest.raises(ValueError, match="checksum mismatch"):
            reader.readinto(buffer)
        assert reader.statistics()["closed"] and not reader.statistics()["eof_verified"]
        with pytest.raises(RuntimeError, match="closed"):
            reader.readinto(buffer)
    source.verify_payload = False
    with direct.CanonicalPlaneReader(source, request.key, "mla_000") as reader:
        chunk = reader.readinto(buffer); chunk.release()
        assert reader.statistics()["eof_verified"]
        assert not reader.statistics()["payload_sha256_verified"]
    with direct.CanonicalPlaneReader(source, request.key, "mla_000", verify_payload=True) as reader:
        with pytest.raises(ValueError, match="checksum mismatch"):
            reader.readinto(buffer)


@pytest.mark.parametrize("difference", [-1, 1])
def test_changed_size_rejected_even_with_cached_owner(tmp_path, difference):
    source, request, _ = make_snapshot(tmp_path / "snapshot")
    source.validate_owner_manifest(0)
    path = source.path / "owner_000/request_00000.mla_000.bin"
    raw = path.read_bytes(); path.write_bytes(raw[:-1] if difference < 0 else raw + b"x")
    with pytest.raises(ValueError, match="truncated or extended"):
        direct.CanonicalPlaneReader(source, request.key, "mla_000")


@pytest.mark.parametrize("change", ["truncate", "extend"])
def test_eof_change_after_open_is_rejected(tmp_path, torch, change):
    source, request, _ = make_snapshot(tmp_path / "snapshot")
    buffer = torch.empty((7, 656), dtype=torch.uint8)
    with direct.CanonicalPlaneReader(source, request.key, "mla_000") as reader:
        first = reader.readinto(buffer); first.release()
        if change == "truncate":
            with reader.path.open("r+b") as file:
                file.truncate(21 * 656)
            error = "ended before"
        else:
            with reader.path.open("ab") as file:
                file.write(b"x")
            error = "excess bytes"
        with pytest.raises(ValueError, match=error):
            while chunk := reader.readinto(buffer):
                chunk.release()
        assert reader.statistics()["closed"] and not reader.statistics()["eof_verified"]


def test_partial_file_reads_are_filled_without_payload_reconstruction(tmp_path, torch):
    source, request, payload = make_snapshot(tmp_path / "snapshot")
    class ShortReads:
        def __init__(self, file): self.file, self.calls = file, 0
        def readinto(self, view):
            self.calls += 1
            return self.file.readinto(memoryview(view)[:17])
        def close(self): self.file.close()
    buffer = torch.empty((31, 656), dtype=torch.uint8)
    with direct.CanonicalPlaneReader(source, request.key, "mla_000") as reader:
        short = reader._handle = ShortReads(reader._handle)
        chunk = reader.readinto(buffer)
        assert bytes(chunk.data.flatten().tolist()) == payload["mla_000"]
        assert short.calls > 2 and reader.statistics()["payload_sha256_verified"]
        chunk.release()


def test_owner_request_plane_metadata_and_policy_validation(tmp_path):
    source, request, _ = make_snapshot(tmp_path / "single")
    for key, plane in [("missing", "mla_000"), (request.key, "mla_999")]:
        with pytest.raises(ValueError, match="Unknown"):
            direct.CanonicalPlaneReader(source, key, plane)
    with pytest.raises(ValueError, match="boolean"):
        direct.CanonicalPlaneReader(source, request.key, "mla_000", verify_payload=1)
    multi, request8, _ = make_snapshot(tmp_path / "striped", owners=8)
    with pytest.raises(ValueError, match="exactly one owner"):
        direct.CanonicalPlaneReader(multi, request8.key, "mla_000")
    shard = source.path / "owner_000/SHARD.json"
    value = json.loads(shard.read_text()); value["owner"] = 1
    shard.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="committed snapshot"):
        direct.CanonicalPlaneReader(SnapshotReader(source.path), request.key, "mla_000")


def test_buffer_shape_dtype_and_contiguity_validation(tmp_path, torch):
    source, request, _ = make_snapshot(tmp_path / "snapshot")
    with direct.CanonicalPlaneReader(source, request.key, "mla_000") as reader:
        for buffer in [torch.empty((7, 655), dtype=torch.uint8),
                       torch.empty((7, 656), dtype=torch.int32),
                       torch.empty((656, 7), dtype=torch.uint8).T,
                       torch.empty((0, 656), dtype=torch.uint8), bytearray(656)]:
            with pytest.raises(ValueError, match="contiguous CPU uint8"):
                reader.readinto(buffer)
        assert reader.statistics()["bytes_read"] == 0
