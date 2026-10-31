"""Bounded file IO over caller-owned canonical record banks."""

from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ulacache.storage.layout import KeyValueRecordPlane, native_mla_encodings_match


@dataclass(frozen=True)
class FileRecordSource:
    """A range or position stripe of canonical record-major native bytes.

    ``sha256`` covers selected records in order, excluding headers and gaps.
    ``file_bytes`` binds the whole file size and defaults to the range end.
    Raw files without a trusted hash require explicit ``verify_payload=False``.
    Snapshot factories retain their existing verification policy and commits.
    """

    path: str | Path
    record_plane: KeyValueRecordPlane
    record_count: int
    offset_bytes: int = 0
    file_bytes: int | None = None
    sha256: str | None = None
    verify_payload: bool = True
    record_stride: int = 1

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.record_plane, KeyValueRecordPlane):
            raise TypeError("Supply an explicit native KeyValueRecordPlane")
        if (
            self.record_plane.page_memory_order != "record_major"
            or self.record_plane.page_alignment_bytes != 1
        ):
            raise NotImplementedError(
                "File sources require canonical record-major bytes without page padding"
            )
        if (
            type(self.record_count) is not int
            or self.record_count < 0
            or type(self.offset_bytes) is not int
            or self.offset_bytes < 0
        ):
            raise ValueError(
                "Native record count and file offset must be nonnegative integers"
            )
        if type(self.record_stride) is not int or self.record_stride < 1:
            raise ValueError("Native record stride must be a positive integer")
        end = self.offset_bytes + (
            0
            if not self.record_count
            else ((self.record_count - 1) * self.record_stride + 1) * self.row_bytes
        )
        if self.file_bytes is None:
            object.__setattr__(self, "file_bytes", end)
        elif type(self.file_bytes) is not int or self.file_bytes < end:
            raise ValueError(
                "The file size must cover the complete selected record range"
            )
        if type(self.verify_payload) is not bool:
            raise ValueError("verify_payload must be boolean")
        if self.sha256 is not None:
            if (
                not isinstance(self.sha256, str)
                or len(self.sha256) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in self.sha256)
            ):
                raise ValueError(
                    "Native record payload SHA256 must contain64 hex digits"
                )
            object.__setattr__(self, "sha256", self.sha256.lower())
        elif self.verify_payload:
            raise ValueError(
                "Payload verification requires a trusted native record SHA256"
            )

    @property
    def row_bytes(self):
        return self.record_plane.bytes_per_record

    @classmethod
    def from_snapshot(
        cls,
        reader,
        request_key,
        plane,
        *,
        owner=None,
        record_plane=None,
        verify_payload=None,
    ):
        """Bind one actual owner plane using SnapshotReader's commit validation.

        Owners=1 defaults to owner0. Multi-owner files require an explicit owner
        and expose only that owner's striped records, never a reconstructed
        global sequence. The caller supplies the matching owner-local IDs.
        An explicit layout must match the snapshot's native encoding, including
        only the aliases documented by native_mla_encodings_match. Equal widths
        alone do not prove encoding compatibility.
        """
        owners = reader.layout.owners
        if owner is None:
            if owners != 1:
                raise ValueError(
                    "Multi-owner snapshots require an explicit source owner"
                )
            owner = 0
        if type(owner) is not int or not 0 <= owner < owners:
            raise ValueError("Source owner is outside the snapshot topology")
        receipt = reader.validate_owner_manifest(owner)
        index = next(
            (i for i, row in enumerate(reader.requests) if row.key == request_key), None
        )
        planes = reader.layout.planes()
        if index is None or plane not in planes:
            raise ValueError("Unknown snapshot request or byte plane")
        width = planes[plane]
        if record_plane is None:
            record_plane = KeyValueRecordPlane(plane, reader.layout.kv_format, (width,))
        if record_plane.bytes_per_record != width or not native_mla_encodings_match(
            record_plane.native_encoding, reader.layout.kv_format
        ):
            raise ValueError(
                "Snapshot source encoding or complete record width differs"
            )
        rows = reader.requests[index].owned_rows(owner, owners)
        name = f"request_{index:05d}.{plane}.bin"
        expected = receipt["files"][name]
        if expected["bytes"] != rows * width:
            raise ValueError("Canonical plane differs from its request geometry")
        if verify_payload is None:
            verify_payload = reader.verify_payload
        return cls(
            Path(reader.path) / f"owner_{owner:03d}" / name,
            record_plane,
            rows,
            file_bytes=expected["bytes"],
            sha256=expected["sha256"],
            verify_payload=verify_payload,
        )

    def open(self):
        return FileRecordReader(self)

    def select_owner(self, owner, owners):
        """Select a canonical position stripe from an explicitly preverified file.

        The original whole-plane hash cannot authenticate a selected stripe.
        Keep hash-required sources on the original reader, or supply a committed
        owner-striped snapshot. Mapping is private and never writes the file.
        """
        from dataclasses import replace

        if (
            type(owners) is not int
            or owners < 1
            or type(owner) is not int
            or not 0 <= owner < owners
        ):
            raise ValueError("Select an explicit valid record owner")
        if self.verify_payload:
            raise ValueError("Owner views require an explicitly preverified source")
        count = len(range(owner, self.record_count, owners))
        offset = self.offset_bytes + owner * self.record_stride * self.row_bytes
        return replace(
            self,
            record_count=count,
            offset_bytes=min(offset, self.file_bytes),
            record_stride=self.record_stride * owners,
            sha256=None,
        )


class PlaneChunk:
    """A caller-buffer lease, released only after its consumer has finished.

    ``release(after=event.synchronize)`` waits before permitting CPU buffer
    reuse. A failed wait keeps the buffer leased. Closing its reader does not
    release buffers or prove completion of asynchronous GPU reads.
    """

    def __init__(self, reader, token, start, data):
        self.start, self.data = start, data
        self._reader, self._token = reader, token
        self.released = False

    def release(self, *, after=None):
        if not self.released:
            if after is not None:
                after()
            self._reader._release(self._token)
            self.released = True


class FileRecordReader:
    """The shared direct-read service used by canonical snapshot restoration.

    The caller owns a bounded contiguous CPU uint8 (rows, row_bytes) buffer,
    optionally pinned. No payload bytes/bytearray/Tensor reconstruction is
    performed. ``close`` closes the file only; borrowed views remain live.
    """

    def __init__(self, source):
        if not isinstance(source, FileRecordSource):
            raise TypeError("Supply a FileRecordSource")
        self.source = source
        self._started = time.perf_counter()
        self._closed = False
        self._handle = None
        self._mapping = None
        self._leases = {}
        self._next_token = 0
        self._rows_read = 0
        self._eof_verified = False
        self.bytes_read = self.chunks_read = 0
        self.whole_call_wall_seconds = None
        self.width, self.rows, self.path = (
            source.row_bytes,
            source.record_count,
            source.path,
        )
        self.verify_payload = source.verify_payload
        self._digest = hashlib.sha256() if self.verify_payload else None
        self._eof_probe = bytearray(1)
        self._handle = self.path.open("rb", buffering=0)
        self._fd = self._handle.fileno()
        try:
            if os.fstat(self._fd).st_size != source.file_bytes:
                raise ValueError("Owner byte plane is truncated or extended")
            self._handle.seek(source.offset_bytes)
            if source.record_stride > 1 and source.record_count:
                self._mapping = mmap.mmap(self._fd, 0, access=mmap.ACCESS_COPY)
        except BaseException:
            self.close()
            raise

    def readinto(self, buffer):
        """Return a leased complete-record chunk, or None after verified EOF.

        A live overlapping buffer cannot be reused. The final chunk is returned
        only after the complete selected range passes size/EOF/hash checks.
        """
        if self._closed:
            raise RuntimeError("Native file reader is closed")
        if self._eof_verified:
            return None
        if self._rows_read == self.rows:
            self._finish()
            return None
        import torch

        if (
            not isinstance(buffer, torch.Tensor)
            or buffer.device.type != "cpu"
            or buffer.dtype != torch.uint8
            or buffer.ndim != 2
            or buffer.shape[1] != self.width
            or buffer.shape[0] < 1
            or not buffer.is_contiguous()
        ):
            raise ValueError("Supply a contiguous CPU uint8 (rows, plane_width) buffer")
        pointer, end = buffer.data_ptr(), buffer.data_ptr() + buffer.numel()
        if any(
            pointer < right and left < end for left, right, _ in self._leases.values()
        ):
            raise RuntimeError("A consumer still owns an overlapping buffer")
        start = self._rows_read
        count = min(buffer.shape[0], self.rows - start)
        length = count * self.width
        # Preserve the existing canonical reader's direct view of caller storage.
        view = memoryview((ctypes.c_ubyte * length).from_address(pointer)).cast("B")
        try:
            if self._mapping is not None:
                if os.fstat(self._fd).st_size != self.source.file_bytes:
                    raise ValueError("Snapshot byte plane size changed during reading")
                base = torch.frombuffer(self._mapping, dtype=torch.uint8)
                selected = torch.as_strided(
                    base,
                    (count, self.width),
                    (self.source.record_stride * self.width, 1),
                    self.source.offset_bytes
                    + start * self.source.record_stride * self.width,
                )
                try:
                    buffer[:count].copy_(selected)
                finally:
                    del selected, base
            else:
                offset = 0
                while offset < length:
                    got = self._handle.readinto(view[offset:])
                    if not got:
                        raise ValueError(
                            "Snapshot byte plane ended before its canonical rows"
                        )
                    offset += got
            if self._digest is not None:
                self._digest.update(view)
        except BaseException:
            self.close()
            raise
        finally:
            view.release()
        self._rows_read += count
        self.bytes_read += length
        self.chunks_read += 1
        if self._rows_read == self.rows:
            self._finish()
        token = self._next_token
        self._next_token += 1
        self._leases[token] = (pointer, end, buffer)
        return PlaneChunk(self, token, start, buffer[:count])

    def _finish(self):
        try:
            # A bounded range can have a declared file trailer. Check that
            # complete size too; only the selected range contributes to SHA.
            if (
                self.source.record_stride == 1
                and self.source.offset_bytes + self.rows * self.width
                == self.source.file_bytes
            ):
                if self._handle.readinto(self._eof_probe):
                    raise ValueError("Snapshot byte plane contains excess bytes")
            if os.fstat(self._fd).st_size != self.source.file_bytes:
                raise ValueError("Snapshot byte plane size changed during reading")
            if (
                self._digest is not None
                and self._digest.hexdigest() != self.source.sha256
            ):
                raise ValueError("Snapshot byte-plane checksum mismatch")
        except BaseException:
            self.close()
            raise
        self._eof_verified = True

    def _release(self, token):
        del self._leases[token]

    def statistics(self):
        return dict(
            bytes_read=self.bytes_read,
            chunks_read=self.chunks_read,
            eof_verified=self._eof_verified,
            payload_sha256_verified=self.verify_payload and self._eof_verified,
            borrowed_buffers=len(self._leases),
            closed=self._closed,
            whole_call_wall_seconds=self.whole_call_wall_seconds,
            timing_scope="Entire reader call lifetime, including caller work/waits",
        )

    def close(self):
        if not self._closed:
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None
            if self._handle is not None:
                self._handle.close()
            self._closed = True
            self.whole_call_wall_seconds = time.perf_counter() - self._started

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()


class CanonicalPlaneReader(FileRecordReader):
    """Read a committed single-owner snapshot through the shared file reader."""

    def __init__(self, reader, request_key, plane, *, verify_payload=None):
        if reader.layout.owners != 1:
            raise ValueError("Direct canonical planes require exactly one owner")
        super().__init__(
            FileRecordSource.from_snapshot(
                reader, request_key, plane, verify_payload=verify_payload
            )
        )


class NativeRecordRouting:
    """The existing FastCudaRequestSnapshot route/copy core, without tools.

    Existing snapshot callers keep their own quiesce/invalidate scope. New
    callers should use QuiescentRecordIO to obtain the same blocking boundary.
    CPU-resident records are copied directly from CPU input to their CPU bank;
    only the GPU subset is transferred to the existing GPU bank.
    """

    def __init__(
        self,
        *,
        gpu_records,
        layer_groups,
        owner_rows,
        cpu_records=None,
        directories=None,
        request_page_map=None,
        page_rows=64,
    ):
        import torch

        self.torch = torch
        if (
            gpu_records.dtype != torch.uint8
            or gpu_records.ndim != 3
            or len(layer_groups) != gpu_records.shape[0]
        ):
            raise ValueError(
                "Native GPU banks must contain complete byte records for every layer"
            )
        if directories is not None and request_page_map is not None:
            raise ValueError("Bind exactly one native CPU placement policy")
        if (
            directories is not None or request_page_map is not None
        ) and cpu_records is None:
            raise ValueError(
                "Native CPU placement requires its existing CPU record bank"
            )
        if cpu_records is not None and (
            cpu_records.dtype != torch.uint8
            or cpu_records.ndim != 3
            or cpu_records.device.type != "cpu"
            or cpu_records.shape[0] != gpu_records.shape[0]
            or cpu_records.shape[2] != gpu_records.shape[2]
        ):
            raise ValueError(
                "Native CPU and GPU banks must have matching complete record widths"
            )
        rows = (
            owner_rows.detach().cpu().tolist()
            if hasattr(owner_rows, "detach")
            else list(owner_rows)
        )
        if len(set(rows)) != len(rows) or any(
            type(row) is not int or row < 0 for row in rows
        ):
            raise ValueError("Bind unique nonnegative owner-local record identities")
        if type(page_rows) is not int or page_rows < 1:
            raise ValueError("Native request pages need a positive record count")
        self.gpu_records, self.cpu_records = gpu_records, cpu_records
        self.layer_groups = tuple(layer_groups)
        self.directories, self.request_page_map = directories, request_page_map
        self.page_rows, self.owner_rows = page_rows, rows
        self.row_offsets = {row: index for index, row in enumerate(rows)}
        self.group_locations, self.chunk_routes = {}, {}

    def routes(self, layer, start, count):
        t = self.torch
        if (
            type(layer) is not int
            or not 0 <= layer < len(self.layer_groups)
            or type(start) is not int
            or type(count) is not int
            or start < 0
            or count < 1
            or start + count > len(self.owner_rows)
        ):
            raise ValueError(
                "Native IO chunk lies outside its owner-local layer or row range"
            )
        group = self.layer_groups[layer] if self.directories is not None else -1
        if group not in self.group_locations:
            all_rows = t.tensor(
                self.owner_rows, dtype=t.int64, device=self.gpu_records.device
            )
            if self.directories is not None:
                locations = self.directories[group].index_select(0, all_rows).cpu()
                if t.any(locations == -1):
                    raise ValueError("Snapshot refers to an unallocated MLA identity")
            elif self.request_page_map is not None:
                pages = all_rows // self.page_rows
                locations = self.request_page_map.index_select(0, pages).cpu()
                if t.any(locations == -1):
                    raise ValueError("Snapshot refers to an unallocated request page")
                offsets = all_rows.cpu() % self.page_rows
                locations = t.where(
                    locations >= 0,
                    locations * self.page_rows + offsets,
                    -((-locations - 2) * self.page_rows + offsets) - 2,
                )
            else:
                locations = all_rows.cpu()
            self.group_locations[group] = locations
        key = group, start, count
        if key not in self.chunk_routes:
            locations = self.group_locations[group][start : start + count]
            routes = []
            for cpu in (False, True):
                positions = t.where((locations < 0) if cpu else (locations >= 0))[0]
                indices = locations.index_select(0, positions)
                if cpu:
                    indices = -indices - 2
                else:
                    indices = indices.to(self.gpu_records.device)
                routes.append((cpu, positions, indices))
            self.chunk_routes[key] = routes
        return self.chunk_routes[key]

    def routes_for_rows(self, layer, rows):
        if not rows:
            raise ValueError("Native IO chunks must contain complete records")
        start = self.row_offsets[rows[0]]
        if list(rows) != self.owner_rows[start : start + len(rows)]:
            raise ValueError("Snapshot chunk is not in canonical owned-row order")
        return self.routes(layer, start, len(rows))

    def write(self, layer, start, data):
        t = self.torch
        if (
            data.device.type != "cpu"
            or data.dtype != t.uint8
            or data.ndim != 2
            or data.shape[1] != self.gpu_records.shape[2]
        ):
            raise ValueError("Native IO writes require complete CPU uint8 record rows")
        for cpu, positions, indices in self.routes(layer, start, len(data)):
            if not len(positions):
                continue
            storage = self.cpu_records[layer] if cpu else self.gpu_records[layer]
            storage.index_copy_(
                0, indices, data.index_select(0, positions).to(storage.device)
            )

    def read(self, layer, start, count):
        t = self.torch
        result = t.empty((count, self.gpu_records.shape[2]), dtype=t.uint8)
        for cpu, positions, indices in self.routes(layer, start, count):
            if not len(positions):
                continue
            storage = self.cpu_records[layer] if cpu else self.gpu_records[layer]
            result.index_copy_(0, positions, storage.index_select(0, indices).cpu())
        return result

    def close(self):
        self.group_locations.clear()
        self.chunk_routes.clear()


class QuiescentRecordIO(NativeRecordRouting):
    """Blocking file publication in the original quiesce/invalidate boundary.

    ``invalidate`` is zero-argument: the caller retains the original global IDs
    and invalidates their existing attention aliases. Even on a partial failure,
    retire queued writes before releasing the request-scoped routing state.
    """

    def __init__(self, *, quiesce, invalidate=None, **bindings):
        if not callable(quiesce) or (
            invalidate is not None and not callable(invalidate)
        ):
            raise ValueError(
                "Native IO requires an explicit quiesce and optional invalidation callback"
            )
        super().__init__(**bindings)
        self.quiesce, self.invalidate = quiesce, invalidate
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("Native IO already has an active quiescent scope")
        self.quiesce()
        self._active = True
        return self

    def write(self, layer, start, data):
        if not self._active:
            raise RuntimeError("Native IO writes require the quiescent scope")
        return super().write(layer, start, data)

    def __exit__(self, *_exception):
        try:
            if self.invalidate is not None:
                self.invalidate()
        finally:
            try:
                self.quiesce()
            finally:
                self._active = False
                self.close()
