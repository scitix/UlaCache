"""Portable record integrity and decode-state semantics, without a device."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ulacache.tools.snapshot.format import (
    SnapshotLayout, SnapshotReader, SnapshotRequest, SnapshotWriter, finalize_snapshot,
)


def fixture(path):
    layout = SnapshotLayout("checkpoint-sha/native-v32", 3, 4, 17, (0, 2), (7, 4), 8)
    requests = [SnapshotRequest("one", tuple(range(19)), (77,)),
                SnapshotRequest("two", tuple(range(11)), (91, 92, 93))]
    expected = {}
    for owner in range(layout.owners):
        writer = SnapshotWriter(path, layout, requests, owner)
        for request in requests:
            for plane, width in layout.planes().items():
                rows = request.owned_rows(owner, layout.owners)
                salt = hashlib.sha256(f"{owner}:{request.key}:{plane}".encode()).digest()
                data = bytes(salt[index % len(salt)] ^ (index % 256) for index in range(rows * width))
                expected[owner, request.key, plane] = data
                writer.write_plane(request.key, plane, (data[i:i + 13] for i in range(0, len(data), 13)))
        writer.finish()
    finalize_snapshot(path)
    return layout, requests, expected


def test_preserves_every_owner_plane_and_next_decode_state(tmp_path):
    layout, requests, expected = fixture(tmp_path / "snapshot")
    reader = SnapshotReader(tmp_path / "snapshot")
    assert reader.layout == layout and reader.requests == tuple(requests)
    assert reader.requests[0].history_ids == tuple(range(19))
    assert reader.requests[0].next_token_id == 77 and reader.requests[0].next_position == 19
    assert reader.requests[1].history_ids == tuple(range(11)) + (91, 92)
    assert reader.requests[1].next_token_id == 93 and reader.requests[1].next_position == 13
    for (owner, key, plane), data in expected.items():
        assert b"".join(reader.iter_plane(owner, key, plane, chunk_bytes=29)) == data
    assert reader.verify()["verified_bytes"] == sum(map(len, expected.values()))


def test_pre_rename_snapshot_retains_metadata_hash_and_payload_checks(tmp_path, monkeypatch):
    from ulacache.tools.snapshot import format as snapshot_format

    with monkeypatch.context() as legacy:
        legacy.setattr(snapshot_format, "FORMAT", "ulacache-request-kv-v1")
        layout, requests, expected = fixture(tmp_path)
    reader = SnapshotReader(tmp_path)
    assert reader.layout == layout and reader.requests == tuple(requests)
    assert reader.metadata["format"] == "ulacache-request-kv-v1"
    assert reader.commit["format"] == reader.metadata["format"]
    assert reader.verify()["verified_bytes"] == sum(map(len, expected.values()))
    path = tmp_path / "owner_001/request_00000.mla_000.bin"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        reader.verify()


def test_missing_owner_cannot_publish_partial_snapshot(tmp_path):
    layout = SnapshotLayout("model", 2, 1, 5, (0,), (2, 1), 4)
    request = SnapshotRequest("a", (1, 2, 3), (4,))
    writer = SnapshotWriter(tmp_path, layout, [request], 0)
    for plane, width in layout.planes().items():
        writer.write_plane("a", plane, [bytes(2 * width)])
    writer.finish()
    with pytest.raises(FileNotFoundError):
        finalize_snapshot(tmp_path)
    assert not (tmp_path / "COMMITTED.json").exists()


def test_corrupted_payload_and_receipts_are_rejected(tmp_path):
    fixture(tmp_path)
    reader = SnapshotReader(tmp_path)
    path = tmp_path / "owner_001/request_00000.mla_000.bin"
    data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        list(reader.iter_plane(1, "one", "mla_000"))
    shard = tmp_path / "owner_002/SHARD.json"
    raw = json.loads(shard.read_text()); raw["owner"] = 1; shard.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="receipt"):
        reader.validate_owner_manifest(2)


def test_payload_optout_skips_hash_but_explicit_verify_still_checks(tmp_path, monkeypatch):
    fixture(tmp_path)
    path = tmp_path / "owner_001/request_00000.mla_000.bin"
    data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
    monkeypatch.setenv("UDCACHE_SNAPSHOT_VERIFY_PAYLOAD", "0")
    reader = SnapshotReader(tmp_path)
    assert not reader.verify_payload
    assert b"".join(reader.iter_plane(1, "one", "mla_000")) == data
    with pytest.raises(ValueError, match="checksum"):
        reader.verify()
    with pytest.raises(ValueError, match="checksum"):
        list(SnapshotReader(tmp_path, verify_payload=True).iter_plane(1, "one", "mla_000"))


def test_payload_optout_keeps_size_and_committed_manifest_checks(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setenv("UDCACHE_SNAPSHOT_VERIFY_PAYLOAD", "0")
    path = tmp_path / "owner_001/request_00000.mla_000.bin"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        list(SnapshotReader(tmp_path).iter_plane(1, "one", "mla_000"))


def test_truncation_and_incomplete_planes_are_rejected(tmp_path):
    layout = SnapshotLayout("model", 1, 1, 5, (0,), (2, 1), 4)
    writer = SnapshotWriter(tmp_path, layout, [SnapshotRequest("a", (1, 2), (3,))], 0)
    with pytest.raises(ValueError, match="truncated"):
        writer.write_plane("a", "mla_000", [b"short"])
    with pytest.raises(ValueError, match="Every request"):
        writer.finish()
    with pytest.raises(ValueError, match="next decode token"):
        SnapshotRequest("missing-seed", (1,), ())


def test_committed_snapshot_cannot_be_overwritten(tmp_path):
    layout, requests, _ = fixture(tmp_path)
    with pytest.raises(FileExistsError):
        SnapshotWriter(tmp_path, layout, requests, 0)
    with pytest.raises(FileExistsError):
        finalize_snapshot(tmp_path)


def test_optional_tools_import_and_cli_do_not_load_device_frameworks(tmp_path):
    fixture(tmp_path)
    code = "import sys; import ulacache.tools; assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    result = subprocess.run([sys.executable, "-m", "ulacache.tools.cli", "verify", str(tmp_path)],
                            check=True, text=True, capture_output=True, env=env)
    assert json.loads(result.stdout)["owners"] == 3
