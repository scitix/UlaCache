"""Bounded sequence reconstruction and quiescent native-record restoration."""

from __future__ import annotations

from ulacache.transfer.file_io import FileRecordSource


def iter_sequence_plane(reader, request_key, plane, *, chunk_rows=8192):
    """Yield (position, CPU byte tensor) without changing native record bytes.

    Every owner stream is consumed to EOF and its exact record count is checked.
    Payload checksums follow the reader's verification policy. Target allocators
    and residency policies belong to the caller.
    """
    import torch

    owners = reader.layout.owners
    if chunk_rows < owners or chunk_rows % owners:
        raise ValueError("Sequence chunks must contain a whole owner stripe")
    request = next(r for r in reader.requests if r.key == request_key)
    width = reader.layout.planes()[plane]
    streams = [
        iter(
            reader.iter_plane(
                owner, request_key, plane, chunk_bytes=chunk_rows // owners * width
            )
        )
        for owner in range(owners)
    ]
    for start in range(0, request.next_position, chunk_rows):
        count = min(chunk_rows, request.next_position - start)
        result = torch.empty((count, width), dtype=torch.uint8)
        for owner, stream in enumerate(streams):
            expected = len(range(owner, count, owners))
            if expected:
                raw = bytearray(next(stream))
                value = torch.frombuffer(raw, dtype=torch.uint8).reshape(-1, width)
                if len(value) != expected:
                    raise ValueError("Striped snapshot record count differs")
                result[owner::owners].copy_(value)
        yield start, result
    for stream in streams:
        try:
            next(stream)
        except StopIteration:
            continue
        raise ValueError("Striped snapshot contains excess records")


def write_index_records(target, rows, data, *, page_rows=64, segment_bytes=(128, 4)):
    """Write canonical token records into native planar Index pages."""
    import torch

    rows = rows.to(target.device, dtype=torch.int64)
    data = data.to(target.device)
    pages, offsets = rows // page_rows, rows % page_rows
    native_start = canonical_start = 0
    for width in segment_bytes:
        columns = (
            native_start
            + offsets[:, None] * width
            + torch.arange(width, device=target.device)
        )
        target[pages[:, None], columns] = data[
            :, canonical_start : canonical_start + width
        ]
        native_start += page_rows * width
        canonical_start += width


def write_sequence_records(cache, layer, logical_rows, data, *, verify=False):
    """Restore registered native logical IDs, without a new residency policy."""
    import torch

    group = cache.groups.group(layer)
    rows = logical_rows.to(cache.directory.device, dtype=torch.int64)
    locations = cache.directory[group].index_select(0, rows).cpu()
    if (locations == -1).any():
        raise ValueError("Snapshot restore requires allocated native identities")
    if data.dtype != torch.uint8 or data.shape != (rows.numel(), cache.row_bytes):
        raise ValueError("Snapshot bytes must contain complete native records")
    for cpu in (False, True):
        mask = locations < 0 if cpu else locations >= 0
        positions = mask.nonzero().flatten()
        if not positions.numel():
            continue
        source = data.index_select(0, positions)
        selected = locations.index_select(0, positions)
        storage = cache.host.storage[layer] if cpu else cache.gpu[layer]
        indices = (-selected - 2 if cpu else selected).to(storage.device)
        storage.index_copy_(0, indices, source.to(storage.device))
        if verify and not torch.equal(storage.index_select(0, indices).cpu(), source):
            raise RuntimeError("Restored unified KV differs from canonical bytes")


class PreparedSequenceRestore:
    """Request-scoped snapshot of a quiescent directory, never a decode policy.

    The caller must finish this request's writes before any allocation, release,
    or residency planning can change its directory entries. Discard the object
    after restore. No plan is attached to the cache or retained by decode.
    """

    def __init__(self, cache, logical_rows, *, quiesce):
        import torch

        if logical_rows.ndim != 1 or not logical_rows.numel():
            raise ValueError(
                "A prepared restore needs nonempty one-dimensional native rows"
            )
        quiesce()
        rows = logical_rows.to(cache.directory.device, dtype=torch.int64)
        # One device-to-host mapping copy for every group and the whole request.
        self.locations = cache.directory.index_select(1, rows).cpu()
        if (self.locations == -1).any():
            raise ValueError("Snapshot restore requires allocated native identities")
        self.cache, self.row_count = cache, rows.numel()
        self.placements = {}

    def _placements(self, group, start, count):
        import torch

        key = group, start, count
        if key in self.placements:
            return self.placements[key]
        locations = self.locations[group, start : start + count]
        result = []
        for cpu in (False, True):
            positions = (locations < 0 if cpu else locations >= 0).nonzero().flatten()
            if not positions.numel():
                continue
            selected = locations.index_select(0, positions)
            selected = -selected - 2 if cpu else selected
            bound = self.cache.host_slots if cpu else self.cache.gpu_slots
            if selected.min().item() < 0 or selected.max().item() >= bound:
                raise ValueError(
                    "Prepared restore points outside the bound record bank"
                )
            first = selected[0].item()
            contiguous = torch.equal(
                selected,
                torch.arange(
                    first,
                    first + len(selected),
                    device=selected.device,
                    dtype=selected.dtype,
                ),
            )
            device = self.cache.host.storage.device if cpu else self.cache.gpu.device
            indices = None if contiguous else selected.to(device)
            result.append(
                (
                    cpu,
                    None if len(positions) == count else positions,
                    first if contiguous else None,
                    indices,
                    len(selected),
                )
            )
        self.placements[key] = result
        return result

    def write(self, layer, start, data, *, verify=False):
        """Write the same full native bytes using stable request-local mappings."""
        import torch

        if data.dtype != torch.uint8 or data.ndim != 2 or data.device.type != "cpu":
            raise ValueError("Prepared restore expects a CPU matrix of native bytes")
        count = len(data)
        if start < 0 or count < 1 or start + count > self.row_count:
            raise ValueError("Prepared restore chunk lies outside its request")
        if data.shape[1] != self.cache.row_bytes:
            raise ValueError("Prepared restore needs complete native records")
        group = self.cache.groups.group(layer)
        for cpu, positions, first, indices, rows in self._placements(
            group, start, count
        ):
            source = data if positions is None else data.index_select(0, positions)
            storage = self.cache.host.storage[layer] if cpu else self.cache.gpu[layer]
            if first is not None:
                destination = storage[first : first + rows]
                destination.copy_(source)
                actual = destination if verify else None
            else:
                storage.index_copy_(0, indices, source.to(storage.device))
                actual = storage.index_select(0, indices) if verify else None
            if verify and not torch.equal(actual.cpu(), source):
                raise RuntimeError("Restored unified KV differs from canonical bytes")


def restore_public_request(snapshot, reader, request_key, native_rows, *, verify=False):
    """Keep native allocation, planar Index bytes and publication boundaries.

    Preverified canonical files expose only this owner's position stripe.
    Hash-required canonical files retain full payload verification, with bounded
    buffers. No physical filesystem bandwidth or model speedup is inferred.
    """
    t, b = snapshot.torch, snapshot.bindings
    source, target = reader.layout, snapshot.layout
    if (
        source.owners not in (1, target.owners)
        or source.model_fingerprint != target.model_fingerprint
        or source.layers != target.layers
        or source.mla_row_bytes != target.mla_row_bytes
        or source.index_anchors != target.index_anchors
        or source.index_segment_bytes != target.index_segment_bytes
        or source.index_page_rows != target.index_page_rows
    ):
        raise ValueError("Canonical source and native target geometry differ")
    request = next((r for r in reader.requests if r.key == request_key), None)
    if request is None:
        raise ValueError("Canonical request key is absent")
    owned = snapshot._owned(request, native_rows)
    replicated_index = snapshot.index_owners != target.owners
    if replicated_index and source.owners != 1:
        raise NotImplementedError(
            "Replicated Index restore requires a canonical source"
        )
    index_rows = (
        (
            native_rows.detach().cpu().tolist()
            if hasattr(native_rows, "detach")
            else list(native_rows)
        )
        if replicated_index
        else owned
    )
    plane = b.public_cache.cache_specification.storage_layout.record_planes[0]
    # Source descriptors are validated before the first write. Never publish a
    # request whose later owner, encoding or commit fails validation.
    sources = {}
    owner_view = source.owners == 1 and target.owners > 1 and not reader.verify_payload
    for name in source.planes():
        descriptor = FileRecordSource.from_snapshot(
            reader,
            request_key,
            name,
            owner=snapshot.owner if source.owners > 1 else 0,
            record_plane=plane if name.startswith("mla_") else None,
        )
        if owner_view and not (replicated_index and name.startswith("index_")):
            descriptor = descriptor.select_owner(snapshot.owner, target.owners)
        sources[name] = descriptor
    local_source = source.owners == target.owners or owner_view
    ids = native_rows[snapshot.owner :: target.owners] if local_source else native_rows
    ids = ids.contiguous()
    checked = io_bytes = 0
    try:
        with snapshot._boundary():
            snapshot._quiesce()
            try:
                for name, descriptor in sources.items():
                    layer = int(name.rsplit("_", 1)[1])
                    replica = replicated_index and name.startswith("index_")
                    plane_rows = index_rows if replica else owned
                    if name.startswith("mla_"):
                        b.public_cache.write(ids, layer, descriptor)
                        checked += len(owned) * descriptor.row_bytes
                        io_bytes += descriptor.record_count * descriptor.row_bytes
                        if not verify:
                            continue
                    buffer = t.empty(
                        (
                            max(
                                1,
                                min(
                                    descriptor.record_count,
                                    snapshot.chunk_bytes // descriptor.row_bytes,
                                ),
                            ),
                            descriptor.row_bytes,
                        ),
                        dtype=t.uint8,
                    )
                    with descriptor.open() as stream:
                        while (chunk := stream.readinto(buffer)) is not None:
                            try:
                                start = chunk.start
                                if local_source or replica:
                                    offset, data = start, chunk.data
                                else:
                                    first = (snapshot.owner - start) % target.owners
                                    offset = (start + first) // target.owners
                                    data = chunk.data[
                                        first :: target.owners
                                    ].contiguous()
                                rows = plane_rows[offset : offset + len(data)]
                                if not rows:
                                    continue
                                if name.startswith("index_"):
                                    snapshot._index_write(layer, rows, data)
                                    checked += data.numel()
                                if verify:
                                    read = (
                                        snapshot._mla_read
                                        if name.startswith("mla_")
                                        else snapshot._index_read
                                    )
                                    if not t.equal(read(layer, rows), data):
                                        raise RuntimeError(
                                            "Public native restore changed record bytes"
                                        )
                            finally:
                                chunk.release()
                        io_bytes += stream.bytes_read
                expected = sum(
                    (
                        len(index_rows)
                        if replicated_index and name.startswith("index_")
                        else len(owned)
                    )
                    * width
                    for name, width in source.planes().items()
                )
                if checked != expected:
                    raise ValueError(
                        "Public restore did not cover every complete owner record"
                    )
            finally:
                # Retire and invalidate even after partial writes; the native
                # request remains unpublished and its caller owns rollback.
                try:
                    if b.invalidate:
                        b.invalidate(native_rows)
                finally:
                    snapshot._quiesce()
        return dict(
            request_key=request_key,
            owner=snapshot.owner,
            restored_bytes=checked,
            next_position=request.next_position,
            next_token_id=request.next_token_id,
            restore_io=dict(
                backend="public-file-record-owner-view"
                if owner_view
                else "public-file-record-verified",
                logical_file_read_bytes=io_bytes,
                selected_payload_bytes=checked,
                physical_gpfs_read_bytes=None,
                payload_verification=reader.verify_payload,
                replicated_index=replicated_index,
            ),
        )
    finally:
        if getattr(snapshot, "_record_routing", None) is not None:
            snapshot._record_routing.close()
            snapshot._record_routing = None
        snapshot._route_rows = snapshot._route_offsets = None
        snapshot._group_locations = snapshot._chunk_routes = None
