"""Bounded diagnostic capture of pre-plan sparse addresses.

This module records addresses, not KV payloads or benchmark throughput. Fixed
GPU buffers participate in Graph capture; a serving adapter drains each chunk
before it is reused. Model timing and trace-replay timing remain separate.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SparseTraceGeometry:
    groups: int
    query_rows: int
    topk: int
    frames: int = 512
    chunk_frames: int = 64
    event_capacity: int = 262144

    def __post_init__(self):
        for name in (
            "groups",
            "query_rows",
            "topk",
            "frames",
            "chunk_frames",
            "event_capacity",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 2**31:
                raise ValueError(f"{name} must be a positive int32 count")
        if self.frames % self.chunk_frames:
            raise ValueError("The trace window must contain whole chunks")

    @property
    def payload_bytes(self):
        selected = self.chunk_frames * self.groups * self.query_rows * self.topk * 4
        current = self.chunk_frames * self.groups * self.query_rows * 4
        seen_and_epoch = self.chunk_frames * self.groups * (4 + 8)
        expiry = self.chunk_frames * self.groups * 8 * 8
        return (
            selected
            + current
            + seen_and_epoch
            + expiry
            + self.event_capacity * 3 * 8
            + 5 * 8
        )


def historical_lookup_reference(selected, query_requests, current):
    """CPU correctness oracle, never a measured scheduler implementation.

    Remove invalid/padding IDs and every current verification position, then
    union references across the valid queries of each request. Equal addresses
    in different requests retain distinct lookup identities.
    """
    if len(selected) != len(query_requests) or len(current) != len(query_requests):
        raise ValueError("Query metadata and sparse rows must agree")
    new_rows = {}
    for request, uid in zip(query_requests, current):
        if request >= 0 and uid >= 0:
            new_rows.setdefault(request, set()).add(uid)
    result = set()
    for request, indices in zip(query_requests, selected):
        if request < 0:
            continue
        for uid in indices:
            if uid > 0 and uid not in new_rows.get(request, ()):
                result.add((request, uid))
    return frozenset(result)


class GpuSparseTraceRecorder:
    """Passive Graph-compatible recorder with an explicit chunk-drain contract."""

    def __init__(self, geometry, *, device, max_gpu_bytes=3_000_000_000):
        import torch
        from ulacache.compiler import load_jit

        if not isinstance(geometry, SparseTraceGeometry):
            raise TypeError("Pass a validated SparseTraceGeometry")
        if type(max_gpu_bytes) is not int or geometry.payload_bytes > max_gpu_bytes:
            raise ValueError("Trace buffers exceed the diagnostic GPU allowance")
        self.geometry = geometry
        self.control = torch.zeros(5, dtype=torch.int64, device=device)
        self.selected = torch.empty(
            (
                geometry.chunk_frames,
                geometry.groups,
                geometry.query_rows,
                geometry.topk,
            ),
            dtype=torch.int32,
            device=device,
        )
        self.current = torch.empty(
            (geometry.chunk_frames, geometry.groups, geometry.query_rows),
            dtype=torch.int32,
            device=device,
        )
        self.seen = torch.zeros(
            (geometry.chunk_frames, geometry.groups), dtype=torch.int32, device=device
        )
        self.epochs = torch.zeros(
            (geometry.chunk_frames, geometry.groups), dtype=torch.int64, device=device
        )
        self.expiry = torch.full(
            (geometry.chunk_frames, geometry.groups, 8),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.events = torch.empty(
            (geometry.event_capacity, 3), dtype=torch.int64, device=device
        )
        self.module = load_jit(
            "udcache_sparse_trace_v1",
            cuda_files=["csrc/telemetry/sparse_trace.cuh"],
            cuda_wrappers=[
                ("record", "udcache::telemetry::record_sparse_trace"),
                ("event", "udcache::telemetry::record_sparse_trace_event"),
                ("expiry", "udcache::telemetry::record_sparse_trace_expiry"),
            ],
        )
        self._next_frame = 0
        self._drained_frames = 0
        self._valid_rows = []

    @property
    def gpu_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.control,
                self.selected,
                self.current,
                self.seen,
                self.epochs,
                self.expiry,
                self.events,
            )
        )

    def start(self):
        """Arm at a quiescent boundary after the initial state is saved."""
        self.control.zero_()
        self.control[0].fill_(1)
        self.seen.fill_(-1)
        self.expiry.fill_(-1)
        self._next_frame = self._drained_frames = 0
        self._valid_rows = []

    def begin_frame(self, valid_query_rows):
        """Call outside the model Graph; no host readback occurs here."""
        g = self.geometry
        if (
            type(valid_query_rows) is not int
            or not 0 <= valid_query_rows <= g.query_rows
        ):
            raise ValueError("Active query rows exceed the capture allocation")
        if self._next_frame >= g.frames:
            raise ValueError("The declared trace window is already full")
        if self._next_frame - self._drained_frames >= g.chunk_frames:
            raise RuntimeError("Drain the completed chunk before overwriting it")
        self.control[1].fill_(self._next_frame)
        self.control[2].fill_(valid_query_rows)
        self._valid_rows.append(valid_query_rows)
        self._next_frame += 1

    def observe(self, group, indices, current, active, epoch):
        """Bind before Graph capture, immediately before metadata promotion."""
        g = self.geometry
        if type(group) is not int or not 0 <= group < g.groups:
            raise ValueError("Invalid management group")
        self.module.record(
            indices,
            current,
            active,
            epoch,
            self.control,
            self.selected,
            self.current,
            self.seen,
            self.epochs,
            group,
            g.frames,
            g.chunk_frames,
            g.groups,
            g.query_rows,
            g.topk,
        )

    def observe_event(self, rows, *, release):
        """Record native allocation/free events; never infer per-token CPU LRU."""
        if type(release) is not bool:
            raise ValueError("Event kind must be explicit")
        self.module.event(
            rows, self.control, self.events, 2 if release else 1, self.geometry.frames
        )

    def observe_expiry(self, descriptors, *, after):
        """Read actual recent-bucket headers and counts around native expiry.

        Descriptors are retained int64[groups,5]: epoch pointer, bucket-epoch
        pointer, summary pointer, summary row stride, and recent bucket count.
        This observer never advances a clock or changes bucket membership.
        """
        if type(after) is not bool:
            raise ValueError("Expiry observation phase must be explicit")
        self.module.expiry(
            descriptors,
            self.control,
            self.expiry,
            int(after),
            self.geometry.frames,
            self.geometry.chunk_frames,
        )

    def drain_chunk(self):
        """Synchronizing diagnostic boundary, separate from production timing."""
        import torch

        count = self._next_frame - self._drained_frames
        if count != self.geometry.chunk_frames:
            raise RuntimeError("Only a complete chunk can be drained")
        torch.cuda.synchronize(self.control.device)
        status = self.control.cpu().tolist()
        if status[3]:
            raise RuntimeError(f"Sparse trace bounds error: {status[3]}")
        value = dict(
            first_frame=self._drained_frames,
            frame_count=count,
            selected=self.selected.cpu(),
            current=self.current.cpu(),
            seen=self.seen.cpu(),
            epochs=self.epochs.cpu(),
            expiry=self.expiry.cpu(),
        )
        expected = torch.tensor(
            self._valid_rows[self._drained_frames : self._next_frame], dtype=torch.int32
        )[:, None].expand_as(value["seen"])
        if not torch.equal(value["seen"], expected):
            raise RuntimeError(
                "Trace window has missing management groups or inconsistent query counts"
            )
        self._drained_frames = self._next_frame
        self.seen.fill_(-1)
        self.expiry.fill_(-1)
        return value

    def stop(self):
        """Finish after final acceptance/release events have been recorded."""
        self.control[0].zero_()

    def event_snapshot(self):
        import torch

        torch.cuda.synchronize(self.control.device)
        status = self.control.cpu().tolist()
        if status[3] or not 0 <= status[4] <= self.geometry.event_capacity:
            raise RuntimeError("Sparse trace event buffer overflow or invalid shape")
        return self.events[: status[4]].cpu()
