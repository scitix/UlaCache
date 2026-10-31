"""Checksummed, owner-sharded request KV snapshots independent of serving code.

Only canonical MLA/Index bytes and decode state are persisted. Allocator IDs,
residency directories, CUDA pointers and captured graphs belong to the loader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import os
from typing import Iterable

FORMAT = "ulacache-request-kv-v1"
READ_FORMATS = frozenset((FORMAT, "ulacache-request-kv-v1"))


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _save(path, raw):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class SnapshotLayout:
    model_fingerprint: str
    owners: int
    layers: int
    mla_row_bytes: int
    index_anchors: tuple[int, ...]
    index_segment_bytes: tuple[int, ...] = (128, 4)
    index_page_rows: int = 64
    kv_format: str = "native-scaled-fp8-v32"

    def __post_init__(self):
        object.__setattr__(self, "index_anchors", tuple(self.index_anchors))
        object.__setattr__(self, "index_segment_bytes", tuple(self.index_segment_bytes))
        if not isinstance(self.model_fingerprint, str) or not self.model_fingerprint:
            raise ValueError("An exact model/weight/format fingerprint is required")
        for name in ("owners", "layers", "mla_row_bytes", "index_page_rows"):
            _positive(getattr(self, name), name)
        if (
            not self.index_anchors
            or len(set(self.index_anchors)) != len(self.index_anchors)
            or tuple(sorted(self.index_anchors)) != self.index_anchors
            or any(
                type(layer) is not int or not 0 <= layer < self.layers
                for layer in self.index_anchors
            )
        ):
            raise ValueError("Index anchor layers must be unique, ordered and in range")
        for width in self.index_segment_bytes:
            _positive(width, "Index segment width")
        if not self.index_segment_bytes or not self.kv_format:
            raise ValueError("Explicit Index geometry and KV format are required")

    @property
    def index_row_bytes(self):
        return sum(self.index_segment_bytes)

    def planes(self):
        return {
            **{f"mla_{layer:03d}": self.mla_row_bytes for layer in range(self.layers)},
            **{
                f"index_{layer:03d}": self.index_row_bytes
                for layer in self.index_anchors
            },
        }


@dataclass(frozen=True)
class SnapshotRequest:
    key: str
    input_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "input_ids", tuple(self.input_ids))
        object.__setattr__(self, "generated_ids", tuple(self.generated_ids))
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("A request key is required")
        if not self.input_ids or not self.generated_ids:
            raise ValueError("Input history and the next decode token must be captured")
        if any(
            type(token) is not int or token < 0
            for token in (*self.input_ids, *self.generated_ids)
        ):
            raise ValueError("Token IDs must be nonnegative integers")

    @property
    def history_ids(self):
        return self.input_ids + self.generated_ids[:-1]

    @property
    def next_token_id(self):
        return self.generated_ids[-1]

    @property
    def next_position(self):
        return len(self.input_ids) + len(self.generated_ids) - 1

    def owned_rows(self, owner, owners):
        return len(range(owner, self.next_position, owners))


def _metadata(layout, requests):
    requests = tuple(requests)
    if not requests or len({request.key for request in requests}) != len(requests):
        raise ValueError("Snapshot requests must be nonempty and unique")
    return {
        "format": FORMAT,
        "layout": asdict(layout),
        "requests": [asdict(request) for request in requests],
        "ownership": "token-position-modulo-owner-count",
        "state": "KV covers history_ids; next_token_id is not yet in KV",
    }


class SnapshotWriter:
    """One owner writes bounded byte chunks, then publishes its shard receipt.

    The supplied chunks must follow sequence position order for this owner.
    Framework/native storage is accessed only through the caller's byte reader.
    """

    def __init__(self, path, layout: SnapshotLayout, requests, owner: int):
        if type(owner) is not int or not 0 <= owner < layout.owners:
            raise ValueError("Owner outside snapshot topology")
        self.path, self.layout, self.owner = Path(path), layout, owner
        self.requests = tuple(requests)
        self.metadata = _metadata(layout, self.requests)
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / "COMMITTED.json").exists():
            raise FileExistsError("A committed snapshot cannot be overwritten")
        self.directory = self.path / f"owner_{owner:03d}"
        self.directory.mkdir(exist_ok=False)
        self.files = {}
        self.expected = {
            (index, plane)
            for index in range(len(self.requests))
            for plane in layout.planes()
        }
        _save(
            self.directory / "requests.json.gz",
            gzip.compress(_json(self.metadata), mtime=0),
        )

    def write_plane(self, request_key: str, plane: str, chunks: Iterable[bytes]):
        indices = {request.key: i for i, request in enumerate(self.requests)}
        if request_key not in indices or plane not in self.layout.planes():
            raise ValueError("Unknown snapshot request or byte plane")
        index = indices[request_key]
        name = f"request_{index:05d}.{plane}.bin"
        if name in self.files:
            raise ValueError("A byte plane cannot be written twice")
        expected = (
            self.requests[index].owned_rows(self.owner, self.layout.owners)
            * self.layout.planes()[plane]
        )
        size, digest = 0, hashlib.sha256()
        with (self.directory / name).open("xb") as handle:
            for chunk in chunks:
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TypeError("Snapshot chunks must contain raw bytes")
                size += len(chunk)
                if size > expected:
                    raise ValueError("Byte plane exceeds its canonical row count")
                handle.write(chunk)
                digest.update(chunk)
            if size != expected:
                raise ValueError("Byte plane is truncated")
            handle.flush()
            os.fsync(handle.fileno())
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        self.files[name] = {
            "bytes": size,
            "sha256": digest.hexdigest(),
            "request_index": index,
            "plane": plane,
        }

    def finish(self):
        actual = {
            (value["request_index"], value["plane"]) for value in self.files.values()
        }
        if actual != self.expected:
            raise ValueError("Every request MLA layer and Index anchor must be saved")
        receipt = {
            "owner": self.owner,
            "metadata_sha256": _sha(_json(self.metadata)),
            "files": self.files,
        }
        _save(self.directory / "SHARD.json", _json(receipt))
        return receipt


def finalize_snapshot(path):
    """Commit only when all owners have identical complete request metadata."""
    path = Path(path)
    if (path / "COMMITTED.json").exists():
        raise FileExistsError("Snapshot already committed")
    reader = SnapshotReader(path, require_committed=False)
    hashes = {}
    for owner in range(reader.layout.owners):
        reader.validate_owner_manifest(owner)
        hashes[str(owner)] = _sha((path / f"owner_{owner:03d}/SHARD.json").read_bytes())
    value = {
        "format": reader.metadata["format"],
        "metadata_sha256": _sha(_json(reader.metadata)),
        "owners": hashes,
        "request_count": len(reader.requests),
    }
    _save(path / "COMMITTED.json", _json(value))
    return value


class SnapshotReader:
    def __init__(self, path, *, require_committed=True, verify_payload=None):
        # Repeated benchmark restores may reuse a previously verified snapshot.
        # Explicit arguments override the opt-out; metadata and sizes still bind
        # every read to its committed layout. The default verifies all payloads.
        if verify_payload is None:
            setting = os.environ.get("UDCACHE_SNAPSHOT_VERIFY_PAYLOAD", "1")
            if setting not in ("0", "1"):
                raise ValueError("UDCACHE_SNAPSHOT_VERIFY_PAYLOAD must be 0 or 1")
            verify_payload = setting == "1"
        if type(verify_payload) is not bool:
            raise ValueError("verify_payload must be boolean")
        self.verify_payload = verify_payload
        self.path = Path(path)
        self.metadata = json.loads(
            gzip.decompress((self.path / "owner_000/requests.json.gz").read_bytes())
        )
        if (
            self.metadata.get("format") not in READ_FORMATS
            or self.metadata.get("ownership") != "token-position-modulo-owner-count"
        ):
            raise ValueError("Unsupported request KV snapshot format")
        self.layout = SnapshotLayout(**self.metadata["layout"])
        self.requests = tuple(
            SnapshotRequest(**row) for row in self.metadata["requests"]
        )
        expected = _metadata(self.layout, self.requests)
        expected["format"] = self.metadata["format"]
        if self.metadata != expected:
            # Normalize JSON tuple/list representations through serialization.
            if _json(self.metadata) != _json(expected):
                raise ValueError("Snapshot metadata contains unsupported state")
        self.commit = None
        self._owner_receipts = {}
        if require_committed:
            self.commit = json.loads((self.path / "COMMITTED.json").read_bytes())
            if (
                self.commit["format"] != self.metadata["format"]
                or self.commit["metadata_sha256"] != _sha(_json(self.metadata))
                or set(self.commit["owners"])
                != set(map(str, range(self.layout.owners)))
            ):
                raise ValueError("Snapshot commit does not match its metadata/topology")

    def validate_owner_manifest(self, owner):
        if type(owner) is not int or not 0 <= owner < self.layout.owners:
            raise ValueError("Owner outside snapshot topology")
        if owner in self._owner_receipts:
            return self._owner_receipts[owner]
        directory = self.path / f"owner_{owner:03d}"
        raw = (directory / "SHARD.json").read_bytes()
        if self.commit and _sha(raw) != self.commit["owners"][str(owner)]:
            raise ValueError("Owner receipt differs from the committed snapshot")
        receipt = json.loads(raw)
        metadata = json.loads(
            gzip.decompress((directory / "requests.json.gz").read_bytes())
        )
        if (
            receipt["owner"] != owner
            or _json(metadata) != _json(self.metadata)
            or receipt["metadata_sha256"] != _sha(_json(self.metadata))
        ):
            raise ValueError("Owners disagree on request/decode metadata")
        expected = {}
        for index, request in enumerate(self.requests):
            for plane, width in self.layout.planes().items():
                expected[f"request_{index:05d}.{plane}.bin"] = (
                    index,
                    plane,
                    request.owned_rows(owner, self.layout.owners) * width,
                )
        if set(receipt["files"]) != set(expected):
            raise ValueError("Owner byte-plane inventory is incomplete")
        for name, (index, plane, size) in expected.items():
            value = receipt["files"][name]
            if (value["request_index"], value["plane"], value["bytes"]) != (
                index,
                plane,
                size,
            ):
                raise ValueError("Owner byte-plane geometry differs")
            if (directory / name).stat().st_size != size:
                raise ValueError("Owner byte plane is truncated or extended")
        self._owner_receipts[owner] = receipt
        return receipt

    def iter_plane(
        self,
        owner,
        request_key,
        plane,
        *,
        chunk_bytes=4 * 1024 * 1024,
        verify_payload=None,
    ):
        _positive(chunk_bytes, "Chunk bytes")
        if verify_payload is None:
            verify_payload = self.verify_payload
        if type(verify_payload) is not bool:
            raise ValueError("verify_payload must be boolean")
        receipt = self.validate_owner_manifest(owner)
        index = next(
            (
                i
                for i, request in enumerate(self.requests)
                if request.key == request_key
            ),
            None,
        )
        if index is None or plane not in self.layout.planes():
            raise ValueError("Unknown snapshot request or byte plane")
        name = f"request_{index:05d}.{plane}.bin"
        digest, size = hashlib.sha256() if verify_payload else None, 0
        with (self.path / f"owner_{owner:03d}" / name).open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                if digest is not None:
                    digest.update(chunk)
                size += len(chunk)
                yield chunk
        if size != receipt["files"][name]["bytes"] or (
            digest is not None
            and digest.hexdigest() != receipt["files"][name]["sha256"]
        ):
            raise ValueError("Snapshot byte-plane checksum mismatch")

    def verify(self, *, chunk_bytes=4 * 1024 * 1024):
        total = 0
        for owner in range(self.layout.owners):
            for request in self.requests:
                for plane in self.layout.planes():
                    for chunk in self.iter_plane(
                        owner,
                        request.key,
                        plane,
                        chunk_bytes=chunk_bytes,
                        verify_payload=True,
                    ):
                        total += len(chunk)
        return {
            "format": FORMAT,
            "owners": self.layout.owners,
            "requests": len(self.requests),
            "verified_bytes": total,
        }
