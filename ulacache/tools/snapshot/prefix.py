"""Canonical per-request prefix IO with explicit native byte-bank bindings.

An owners=1 snapshot describes sequence positions, not physical request slots.
The caller supplies native allocations and completion boundaries. Target and
draft model banks can be saved independently without stacking native planes.
"""

from __future__ import annotations
from pathlib import Path
from .format import (
    SnapshotLayout,
    SnapshotRequest as SnapshotRequest,
    SnapshotReader,
    SnapshotWriter,
    finalize_snapshot,
)


def stripe_prefix(source, destination, owners, *, max_plane_bytes=256 * 1024**2):
    """Partition checked canonical bytes once for direct owner-local reads.

    This offline conversion retains every real sequence row, including an
    uneven final stripe. One complete plane is materialized at a time; reject
    larger planes before creating output rather than exceeding the IO budget.
    It performs no model prefill or quantization.
    """
    from dataclasses import replace
    import numpy as np

    reader = SnapshotReader(source)
    if (
        reader.layout.owners != 1
        or len(reader.requests) != 1
        or type(owners) is not int
        or owners < 1
        or type(max_plane_bytes) is not int
        or max_plane_bytes < 1
    ):
        raise ValueError("Stripe one canonical prefix with explicit positive IO bounds")
    request = reader.requests[0]
    if any(
        request.next_position * width > max_plane_bytes
        for width in reader.layout.planes().values()
    ):
        raise ValueError("Canonical plane exceeds the explicit offline IO bound")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    layout = replace(reader.layout, owners=owners)
    writers = [
        SnapshotWriter(destination, layout, [request], owner) for owner in range(owners)
    ]
    for plane, width in layout.planes().items():
        raw = b"".join(reader.iter_plane(0, request.key, plane))
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(request.next_position, width)
        for owner, writer in enumerate(writers):
            writer.write_plane(request.key, plane, (rows[owner::owners].tobytes(),))
    for writer in writers:
        writer.finish()
    return finalize_snapshot(destination)


def write_prefix(path, layout, request, read_records, *, chunk_rows=4096):
    """Save one complete request via an explicit quiescent record reader."""
    if layout.owners != 1 or chunk_rows < 1:
        raise ValueError(
            "Portable prefixes require one sequence owner and positive chunks"
        )
    writer = SnapshotWriter(path, layout, [request], 0)
    for plane, width in layout.planes().items():

        def chunks():
            for start in range(0, request.next_position, chunk_rows):
                count = min(chunk_rows, request.next_position - start)
                raw = read_records(plane, start, count)
                if len(raw) != count * width:
                    raise ValueError(
                        "Native record reader did not return complete canonical rows"
                    )
                yield raw

        writer.write_plane(request.key, plane, chunks())
    writer.finish()
    return finalize_snapshot(path)


def iter_prefix_records(path, key, plane, *, chunk_rows=4096):
    """Yield canonical positions and checked bytes; no device or framework import."""
    reader = SnapshotReader(path)
    if reader.layout.owners != 1:
        raise ValueError("Use the striped reader for multi-owner snapshots")
    width = reader.layout.planes()[plane]
    position = 0
    for raw in reader.iter_plane(0, key, plane, chunk_bytes=chunk_rows * width):
        if len(raw) % width:
            raise ValueError("Canonical snapshot has an incomplete record")
        yield position, raw
        position += len(raw) // width


class NativeByteBanks:
    """Complete MLA records and planar Index pages, with lazy Torch loading."""

    def __init__(self, mla, index, row_bytes, *, index_segments=(128, 4), page_rows=64):
        import torch

        self.torch = torch
        self.mla = tuple(t.view(torch.uint8).reshape(-1, row_bytes) for t in mla)
        self.index = {layer: t.view(torch.uint8) for layer, t in index.items()}
        self.row_bytes, self.index_segments, self.page_rows = (
            row_bytes,
            index_segments,
            page_rows,
        )
        if row_bytes not in (576, 656) or not self.mla:
            raise ValueError("Bind explicit native raw576 or scaled656 record banks")

    def read(self, plane, rows):
        torch = self.torch
        layer = int(plane.rsplit("_", 1)[1])
        if plane.startswith("mla_"):
            bank = self.mla[layer]
            return (
                bank.index_select(0, rows.to(bank.device, dtype=torch.int64))
                .cpu()
                .contiguous()
                .numpy()
                .tobytes()
            )
        bank = self.index[layer]
        rows = rows.to(bank.device, dtype=torch.int64)
        pages, offsets = rows // self.page_rows, rows % self.page_rows
        segments, start = [], 0
        for width in self.index_segments:
            columns = (
                start
                + offsets[:, None] * width
                + torch.arange(width, device=bank.device)
            )
            segments.append(bank[pages[:, None], columns])
            start += self.page_rows * width
        return torch.cat(segments, dim=1).cpu().contiguous().numpy().tobytes()

    def write(self, plane, rows, raw, *, verify=False):
        torch = self.torch
        layer = int(plane.rsplit("_", 1)[1])
        width = self.row_bytes if plane.startswith("mla_") else sum(self.index_segments)
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, width)
        if len(rows) != len(data):
            raise ValueError("Restore allocation does not cover canonical records")
        if plane.startswith("mla_"):
            bank = self.mla[layer]
            bank.index_copy_(
                0, rows.to(bank.device, dtype=torch.int64), data.to(bank.device)
            )
        else:
            from .restore import write_index_records

            write_index_records(
                self.index[layer],
                rows,
                data,
                page_rows=self.page_rows,
                segment_bytes=self.index_segments,
            )
        if verify and self.read(plane, rows) != raw:
            raise RuntimeError(
                "Restored native prefix bytes differ from the canonical snapshot"
            )


def save_native_prefix(path, request, banks, rows, fingerprint, *, quiesce):
    """Save native byte banks only after the caller retires all producers."""
    quiesce()
    layout = SnapshotLayout(
        model_fingerprint=fingerprint,
        owners=1,
        layers=len(banks.mla),
        mla_row_bytes=banks.row_bytes,
        index_anchors=tuple(sorted(banks.index)),
        index_segment_bytes=banks.index_segments,
        index_page_rows=banks.page_rows,
        kv_format="native-raw-fp8-576"
        if banks.row_bytes == 576
        else "native-scaled-fp8-v32",
    )
    if len(rows) != request.next_position:
        raise ValueError("Native rows must cover the exact saved history")
    return write_prefix(
        path,
        layout,
        request,
        lambda plane, start, count: banks.read(plane, rows[start : start + count]),
    )


def restore_native_prefix(
    path, key, banks, rows, fingerprint, *, quiesce, verify=False
):
    """Restore exact records into caller-allocated, unpublished native IDs."""
    reader = SnapshotReader(path)
    layout = reader.layout
    if (
        layout.owners != 1
        or layout.model_fingerprint != fingerprint
        or layout.layers != len(banks.mla)
        or layout.mla_row_bytes != banks.row_bytes
        or layout.index_anchors != tuple(sorted(banks.index))
        or layout.index_segment_bytes != tuple(banks.index_segments)
        or layout.index_page_rows != banks.page_rows
    ):
        raise ValueError("Canonical prefix and bound native record geometry differ")
    request = next((r for r in reader.requests if r.key == key), None)
    if request is None or len(rows) != request.next_position:
        raise ValueError("Fresh native rows must cover the exact saved request history")
    quiesce()
    checked = 0
    for plane, width in layout.planes().items():
        start = 0
        for raw in reader.iter_plane(0, key, plane, chunk_bytes=4096 * width):
            count = len(raw) // width
            if len(raw) % width:
                raise ValueError("Canonical prefix has an incomplete native record")
            banks.write(plane, rows[start : start + count], raw, verify=verify)
            start += count
            checked += len(raw)
        if start != request.next_position:
            raise ValueError(
                "Canonical prefix does not cover the complete saved history"
            )
    quiesce()
    return dict(
        request_key=key,
        restored_bytes=checked,
        next_position=request.next_position,
        next_token_id=request.next_token_id,
    )
