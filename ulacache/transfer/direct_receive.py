"""Address plans for RDMA directly into allocated canonical KV records.

This module allocates metadata only. The allocator must hold the advertised
locations stable until native transport completion or a drained abort. A failed
batch submission does not establish that outstanding remote writes have ended.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import struct


@dataclass(frozen=True)
class RecordRegion:
    address: int
    size_bytes: int
    row_bytes: int

    def __post_init__(self):
        if (
            type(self.address) is not int
            or type(self.size_bytes) is not int
            or type(self.row_bytes) is not int
            or self.address <= 0
            or self.size_bytes <= 0
            or self.row_bytes <= 0
            or self.size_bytes % self.row_bytes
            or self.address + self.size_bytes > (1 << 63) - 1
        ):
            raise ValueError("Invalid canonical record region")

    @property
    def records(self):
        return self.size_bytes // self.row_bytes


@dataclass(frozen=True)
class RecordBank:
    bank_id: int
    group: int | None
    gpu: RecordRegion
    cpu: RecordRegion | None = None

    def __post_init__(self):
        if type(self.bank_id) is not int or self.bank_id < 0:
            raise ValueError("Bank IDs are nonnegative integers")
        if self.group is not None and (type(self.group) is not int or self.group < 0):
            raise ValueError("A target bank needs a nonnegative residency group")
        if self.cpu is not None and self.cpu.row_bytes != self.gpu.row_bytes:
            raise ValueError("CPU/GPU records in one bank must have equal widths")


def _array(value, *, dimensions):
    import numpy as np

    a = np.asarray(value)
    if a.ndim != dimensions:
        raise ValueError("Record indices must be integer arrays of the declared rank")
    if not a.size:
        return np.empty(a.shape, dtype=np.int64)
    if a.dtype.kind not in "iu":
        raise ValueError("Record indices must be integer arrays of the declared rank")
    if a.dtype.kind == "u" and a.size and a.max() > (1 << 63) - 1:
        raise ValueError("Record index exceeds signed address arithmetic")
    return np.ascontiguousarray(a, dtype=np.int64)


def coalesce_writes(source, destination, row_bytes):
    """Merge only runs contiguous at both ends; never reorder bytes."""
    import numpy as np

    source = _array(source, dimensions=1)
    destination = _array(destination, dimensions=1)
    if source.shape != destination.shape or type(row_bytes) is not int or row_bytes < 1:
        raise ValueError("Source/destination record counts and widths must match")
    if source.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    if (
        np.any(source <= 0)
        or np.any(destination <= 0)
        or np.any(source > (1 << 63) - 1 - row_bytes)
        or np.any(destination > (1 << 63) - 1 - row_bytes)
    ):
        raise ValueError("Transfer lies outside signed address arithmetic")
    breaks = (np.diff(source) != row_bytes) | (np.diff(destination) != row_bytes)
    starts = np.concatenate(([0], np.flatnonzero(breaks) + 1))
    ends = np.concatenate((starts[1:], [source.size]))
    return source[starts], destination[starts], (ends - starts) * row_bytes


def _write_geometry(rows, locations, width):
    """Coalesced offsets shared by banks with the same residency directory."""
    import numpy as np

    hot = locations >= 0
    slots = np.where(hot, locations, -locations - 2)
    breaks = (np.diff(rows) != 1) | (np.diff(slots) != 1) | (hot[1:] != hot[:-1])
    starts = np.concatenate(([0], np.flatnonzero(breaks) + 1))
    ends = np.concatenate((starts[1:], [rows.size]))
    mode = 0 if bool(hot.all()) else (1 if not bool(hot.any()) else 2)
    return (
        rows[starts] * width,
        slots[starts] * width,
        (ends - starts) * width,
        mode,
        hot[starts] if mode == 2 else None,
    )


def order_unpublished_cpu_locations(locations):
    """Order only a new receive's existing CPU slots; preserve GPU locations.

    This metadata operation is valid only before any remote writer or attention
    can use these newly allocated rows. It never selects a different slot set.
    The owning allocator must apply the resulting CPU bindings before publish.
    """
    import numpy as np

    if (
        not isinstance(locations, np.ndarray)
        or locations.ndim != 2
        or locations.dtype != np.dtype(np.int64)
        or not locations.flags.writeable
    ):
        raise ValueError(
            "Unpublished receive locations must be writable int64 metadata"
        )
    changed = False
    for group in locations:
        cold = group <= -2
        values = group[cold]
        if values.size > 1 and np.any(values[1:] > values[:-1]):
            group[cold] = np.sort(values)[::-1]
            changed = True
    return changed


class DirectReceivePlan:
    """An immutable snapshot of already allocated, leased destinations.

    Directory encoding is the native one: GPU slot >=0, CPU slot -slot-2.
    A bank without a residency group (e.g. a native draft bank) uses canonical
    logical row IDs directly. Existing prefixes are omitted by the caller.
    """

    MAGIC = b"ULAPD001"

    def __init__(self, logical_rows, locations, banks, *, page_size):
        import numpy as np
        from collections import Counter

        if type(page_size) is not int or page_size < 1:
            raise ValueError("Transfer page size must be positive")
        self.logical_rows = _array(logical_rows, dimensions=1).copy()
        self.locations = _array(locations, dimensions=2).copy()
        self.banks = tuple(banks)
        self.page_size = page_size
        if (
            self.locations.shape[1] != self.logical_rows.size
            or np.any(self.logical_rows <= 0)
            or np.any(np.diff(np.sort(self.logical_rows)) == 0)
            or np.any(self.locations == -1)
        ):
            raise ValueError("Receive rows must be unique, allocated and bound")
        if not self.banks or len({b.bank_id for b in self.banks}) != len(self.banks):
            raise ValueError("A direct plan needs distinct canonical banks")
        layouts = Counter((b.group, b.gpu.row_bytes) for b in self.banks)
        self._shared_transfer_layouts = frozenset(
            k for k, count in layouts.items() if count > 1
        )
        bounds = {}
        for b in self.banks:
            if b.group is not None and b.group >= self.locations.shape[0]:
                raise ValueError("Bank references an absent residency group")
            if b.group not in bounds:
                loc = self.logical_rows if b.group is None else self.locations[b.group]
                minimum = int(loc.min()) if loc.size else 0
                maximum = int(loc.max()) if loc.size else -1
                bounds[b.group] = (
                    maximum if maximum >= 0 else -1,
                    -minimum - 2 if minimum <= -2 else -1,
                )
            hot_max, cold_max = bounds[b.group]
            if hot_max >= b.gpu.records:
                raise ValueError("GPU record lies outside its registered region")
            if cold_max >= 0 and (b.cpu is None or cold_max >= b.cpu.records):
                raise ValueError("CPU record lies outside its registered region")
        self.logical_rows.flags.writeable = False
        self.locations.flags.writeable = False

    @property
    def record_count(self):
        return self.logical_rows.size

    def registration_regions(self):
        """Exact canonical allocations; this creates no staging allocation."""
        unique = {}
        for b in self.banks:
            for r in (b.gpu, b.cpu):
                if r is not None:
                    prior = unique.setdefault(r.address, r.size_bytes)
                    if prior != r.size_bytes:
                        raise ValueError(
                            "One base pointer has conflicting registration extents"
                        )
        return tuple(unique.items())

    def destination_addresses(self, bank_id, start, count):
        import numpy as np

        if (
            type(start) is not int
            or type(count) is not int
            or start < 0
            or count < 0
            or start + count > self.record_count
        ):
            raise ValueError("Chunk exceeds its leased record suffix")
        b = next((b for b in self.banks if b.bank_id == bank_id), None)
        if b is None:
            raise ValueError("Sender references an unadvertised bank")
        loc = (self.logical_rows if b.group is None else self.locations[b.group])[
            start : start + count
        ]
        hot = loc >= 0
        addresses = np.empty(count, dtype=np.int64)
        addresses[hot] = b.gpu.address + loc[hot] * b.gpu.row_bytes
        if np.any(~hot):
            addresses[~hot] = b.cpu.address + (-loc[~hot] - 2) * b.cpu.row_bytes
        return addresses

    def transfer_batch(self, sources, source_pages, *, destination_start, record_count):
        """Generate one coalesced scatter/gather batch for a native KV chunk.

        Sources map bank IDs to registered source regions. Partial final pages
        copy exactly record_count rows, preserving adjacent destination bytes.
        """
        import numpy as np

        pages = _array(source_pages, dimensions=1)
        if (
            type(record_count) is not int
            or record_count < 0
            or pages.size != (record_count + self.page_size - 1) // self.page_size
            or np.any(pages < 0)
        ):
            raise ValueError("Native source pages do not describe this record chunk")
        if set(sources) != {b.bank_id for b in self.banks}:
            raise ValueError("Source and receiver canonical bank sets differ")
        rows = (
            pages[:, None] * self.page_size + np.arange(self.page_size, dtype=np.int64)
        ).reshape(-1)[:record_count]
        parts = []
        for b in self.banks:
            src = sources[b.bank_id]
            if src.row_bytes != b.gpu.row_bytes or np.any(rows >= src.records):
                raise ValueError("Source record extent or width differs from receiver")
            source = src.address + rows * src.row_bytes
            destination = self.destination_addresses(
                b.bank_id, destination_start, record_count
            )
            parts.append(coalesce_writes(source, destination, src.row_bytes))
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))

    def iter_transfer_batches(
        self,
        sources,
        source_pages,
        *,
        destination_start,
        record_count,
        max_segments=65536,
    ):
        """Plan bounded bank slices without materializing a request-wide batch.

        A fragmented destination can require one descriptor per record per
        bank. Combine bounded slices across banks so small chunks still use
        one vector write, without concatenating the whole request. The caller
        retains the original canonical receive lease throughout.
        """
        import numpy as np
        from collections import OrderedDict

        if type(max_segments) is not int or max_segments < 1:
            raise ValueError("Direct descriptor batch size must be positive")
        pages = _array(source_pages, dimensions=1)
        if (
            type(record_count) is not int
            or record_count < 0
            or type(destination_start) is not int
            or destination_start < 0
            or destination_start + record_count > self.record_count
            or pages.size != (record_count + self.page_size - 1) // self.page_size
            or np.any(pages < 0)
        ):
            raise ValueError("Native source pages do not describe this record chunk")
        if set(sources) != {b.bank_id for b in self.banks}:
            raise ValueError("Source and receiver canonical bank sets differ")
        # Validate every bank before submitting the first write. Preserve the
        # all-or-nothing validation of the materialized planner.
        max_row = (
            int(pages.max()) * self.page_size + self.page_size - 1 if pages.size else -1
        )
        last_page_rows = record_count % self.page_size
        last_row = (
            int(pages[-1]) * self.page_size + last_page_rows - 1
            if last_page_rows
            else max_row
        )
        other_max = (
            int(pages[:-1].max()) * self.page_size + self.page_size - 1
            if pages.size > 1
            else -1
        )
        for b in self.banks:
            src = sources[b.bank_id]
            if (
                src.row_bytes != b.gpu.row_bytes
                or max(other_max, last_row) >= src.records
            ):
                raise ValueError("Source record extent or width differs from receiver")
        rows = (
            pages[:, None] * self.page_size + np.arange(self.page_size, dtype=np.int64)
        ).reshape(-1)[:record_count]
        workspace = None
        filled = 0
        # Cache offsets only for reused layouts, with a fixed entry bound. The
        # cache is local to this chunk, never retained across requests/chunks.
        # Visit all banks per record tile so long prefix chunks cannot evict
        # each shared layout before its next bank reuses it.
        geometry_cache = OrderedDict()
        cache_limit = len(self._shared_transfer_layouts)
        for begin in range(0, record_count, max_segments):
            end = min(begin + max_segments, record_count)
            for b in self.banks:
                src = sources[b.bank_id]
                key = (b.group, src.row_bytes, begin, end)
                geometry = geometry_cache.get(key)
                if geometry is None:
                    locations = (
                        self.logical_rows
                        if b.group is None
                        else self.locations[b.group]
                    )
                    geometry = _write_geometry(
                        rows[begin:end],
                        locations[destination_start + begin : destination_start + end],
                        src.row_bytes,
                    )
                    if (b.group, src.row_bytes) in self._shared_transfer_layouts:
                        if len(geometry_cache) == cache_limit:
                            geometry_cache.popitem(last=False)
                        geometry_cache[key] = geometry
                else:
                    geometry_cache.move_to_end(key)
                source_offsets, destination_offsets, lengths, mode, hot = geometry
                source = src.address + source_offsets
                if mode == 0:
                    destination = b.gpu.address + destination_offsets
                elif mode == 1:
                    destination = b.cpu.address + destination_offsets
                else:
                    destination = np.where(
                        hot,
                        b.gpu.address + destination_offsets,
                        b.cpu.address + destination_offsets,
                    )
                part = source, destination, lengths
                offset = 0
                while offset < len(part[0]):
                    if workspace is None:
                        workspace = np.empty((3, max_segments), dtype=np.int64)
                    count = min(max_segments - filled, len(part[0]) - offset)
                    for field in range(3):
                        workspace[field, filled : filled + count] = part[field][
                            offset : offset + count
                        ]
                    filled += count
                    offset += count
                    if filled == max_segments:
                        yield tuple(workspace[field] for field in range(3))
                        workspace = None
                        filled = 0
        if filled:
            yield tuple(workspace[field, :filled] for field in range(3))

    def to_bytes(self):
        """Bounded binary metadata, with contiguous directory runs compressed."""
        import numpy as np

        groups = [self.logical_rows, *self.locations]
        payloads, descriptors = [], []
        for values in groups:
            if values.size:
                delta = np.diff(values)
                cuts = np.abs(delta) != 1
                if delta.size > 1:
                    cuts[1:] |= delta[1:] != delta[:-1]
                run_count = 1 + int(np.count_nonzero(cuts))
            else:
                run_count = 0
            if run_count * 32 < values.nbytes:
                starts = np.concatenate(([0], np.flatnonzero(cuts) + 1))
                ends = np.concatenate((starts[1:], [values.size]))
                step = np.ones(starts.size, dtype=np.int64)
                multiple = ends - starts > 1
                step[multiple] = values[starts[multiple] + 1] - values[starts[multiple]]
                runs = np.column_stack(
                    (starts, ends - starts, values[starts], step)
                ).astype("<i8")
                descriptors.append(dict(encoding="runs", count=len(runs)))
                payloads.append(runs.tobytes())
            else:
                descriptors.append(dict(encoding="dense", count=values.size))
                payloads.append(values.astype("<i8", copy=False).tobytes())

        def region(r):
            return None if r is None else [r.address, r.size_bytes, r.row_bytes]

        header = json.dumps(
            dict(
                version=1,
                records=self.record_count,
                page_size=self.page_size,
                groups=descriptors,
                banks=[
                    dict(
                        bank=b.bank_id,
                        group=b.group,
                        gpu=region(b.gpu),
                        cpu=region(b.cpu),
                    )
                    for b in self.banks
                ],
            ),
            separators=(",", ":"),
        ).encode()
        return self.MAGIC + struct.pack("<I", len(header)) + header + b"".join(payloads)

    @classmethod
    def from_bytes(cls, payload, *, record_limit):
        import numpy as np

        if type(record_limit) is not int or record_limit < 0:
            raise ValueError("A native request record limit is required")
        if len(payload) < 12 or payload[:8] != cls.MAGIC:
            raise ValueError("Unknown direct receive protocol")
        size = struct.unpack_from("<I", payload, 8)[0]
        if size > 65536 or size > len(payload) - 12:
            raise ValueError("Invalid direct receive metadata header")
        h = json.loads(payload[12 : 12 + size])
        n = h["records"]
        if h["version"] != 1 or type(n) is not int or n < 0 or n > record_limit:
            raise ValueError("Direct metadata exceeds the native request suffix")
        offset = 12 + size
        groups = []
        for spec in h["groups"]:
            count = spec["count"]
            run = spec["encoding"] == "runs"
            if (
                type(count) is not int
                or count < 0
                or spec["encoding"] not in ["runs", "dense"]
            ):
                raise ValueError("Invalid direct directory encoding")
            length = count * (32 if run else 8)
            if length > len(payload) - offset:
                raise ValueError("Truncated direct directory")
            values = np.frombuffer(
                payload, offset=offset, count=count * (4 if run else 1), dtype="<i8"
            )
            offset += length
            if run:
                if not count:
                    if n:
                        raise ValueError("Incomplete direct directory")
                    rows = np.empty(0, dtype=np.int64)
                else:
                    starts, spans, first, steps = values.reshape(-1, 4).T
                    if (
                        n > (1 << 63) - 1
                        or np.any(starts < 0)
                        or np.any(starts >= n)
                        or np.any(spans <= 0)
                        or np.any(spans > n - starts)
                        or starts[0] != 0
                        or int(starts[-1]) + int(spans[-1]) != n
                        or np.any(np.diff(starts) != spans[:-1])
                        or np.any((steps != -1) & (steps != 1))
                    ):
                        raise ValueError("Direct directory runs overlap or have gaps")
                    distance = spans - 1
                    if np.any(
                        (steps == 1) & (first > (1 << 63) - 1 - distance)
                    ) or np.any((steps == -1) & (first < -(1 << 63) + distance)):
                        raise ValueError("Direct directory run arithmetic overflows")
                    owners = np.repeat(np.arange(count, dtype=np.int64), spans)
                    rows = first[owners] + steps[owners] * (
                        np.arange(n, dtype=np.int64) - starts[owners]
                    )
            else:
                if count != n:
                    raise ValueError("Direct directory count differs from request")
                rows = values
            groups.append(rows)
        if not groups or offset != len(payload):
            raise ValueError("Invalid direct metadata extent")
        banks = [
            RecordBank(
                b["bank"],
                b["group"],
                RecordRegion(*b["gpu"]),
                None if b["cpu"] is None else RecordRegion(*b["cpu"]),
            )
            for b in h["banks"]
        ]
        return cls(
            groups[0],
            np.asarray(groups[1:], dtype=np.int64).reshape(len(groups) - 1, n),
            banks,
            page_size=h["page_size"],
        )


def submit_direct_batch(engine, session_id, batch, *, max_segments=65536):
    """Submit vector writes through the normal Mooncake engine API.

    Return native status without releasing any destination lease. In particular,
    timeout/failure is not permission to reuse memory: the caller must perform
    native drained-abort handling before retiring advertised destinations.
    """
    source, destination, lengths = batch
    if (
        len(source) != len(destination)
        or len(source) != len(lengths)
        or type(max_segments) is not int
        or max_segments < 1
    ):
        raise ValueError("Invalid direct write batch")
    for start in range(0, len(source), max_segments):
        stop = start + max_segments
        status = engine.batch_transfer_sync(
            session_id,
            source[start:stop].tolist(),
            destination[start:stop].tolist(),
            lengths[start:stop].tolist(),
        )
        if status != 0:
            return status
    return 0
