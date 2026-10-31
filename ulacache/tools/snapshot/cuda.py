"""Canonical request records from UDCache GPU/CPU storage, with bounded IO."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

from .format import SnapshotLayout, SnapshotReader, SnapshotWriter
from ulacache.transfer.file_io import NativeRecordRouting


def release_unwritten_padding(pool, allocated_rows, history_length):
    """Leave ordinary-decode tail identities unregistered until native writes.

    Native page allocation owns the full physical page. The CPU/GPU directory
    binds only actual ordinary history records; alloc_decode registers each
    later token once. Speculative callers retain their whole-page reserve.
    """
    if not 0 < history_length <= len(allocated_rows):
        raise ValueError("The real history must fit the native page allocation")
    tail = allocated_rows[history_length:]
    tier = getattr(pool, "flowkv_host", None)
    if tier is None:
        tier = getattr(pool, "udcache_local", None)
    if tier is not None and len(tail):
        tier.release(tail)


@dataclass
class PoolSnapshotBindings:
    """Native tensors with independent MLA and Index ownership.

    Index owners default to the MLA topology; one Index owner means every TP
    rank restores a complete replica in its native logical-ID space.
    """

    layout: SnapshotLayout
    owner_rank: int
    gpu_mla: object
    index_gpu: dict
    layer_groups: tuple[int, ...]
    cpu_mla: object = None
    directories: object = None
    request_page_map: object = None
    index_cpu: dict = None
    quiesce: object = None
    invalidate: object = None
    boundary_scope: object = None
    public_cache: object = None
    index_owner_count: int | None = None


class CudaRequestSnapshot:
    def __init__(self, bindings: PoolSnapshotBindings, *, chunk_bytes=4 * 1024 * 1024):
        import torch

        self.torch, self.bindings, self.layout = torch, bindings, bindings.layout
        self.owner = bindings.owner_rank
        self.index_owners = (
            self.layout.owners
            if bindings.index_owner_count is None
            else bindings.index_owner_count
        )
        if type(self.index_owners) is not int or self.index_owners not in (
            1,
            self.layout.owners,
        ):
            raise ValueError("Index records must be replicated or share the MLA owners")
        if self.index_owners != self.layout.owners and (
            bindings.request_page_map is not None or bindings.index_cpu is not None
        ):
            raise ValueError("Replicated Index restore requires its direct GPU bank")
        self.chunk_bytes = chunk_bytes
        if type(chunk_bytes) is not int or chunk_bytes < max(
            self.layout.mla_row_bytes, self.layout.index_row_bytes
        ):
            raise ValueError("Snapshot chunk must hold at least one complete record")
        if not 0 <= self.owner < self.layout.owners:
            raise ValueError("Snapshot owner outside native topology")
        if (
            bindings.gpu_mla.dtype != torch.uint8
            or bindings.gpu_mla.ndim != 3
            or bindings.gpu_mla.shape[0] != self.layout.layers
            or bindings.gpu_mla.shape[2] != self.layout.mla_row_bytes
        ):
            raise ValueError(
                "MLA binding does not contain complete native byte records"
            )
        if len(bindings.layer_groups) != self.layout.layers:
            raise ValueError("Every MLA layer needs its residency group")
        if bindings.directories is not None and bindings.request_page_map is not None:
            raise ValueError("Bind exactly one CPU residency policy")
        if (
            bindings.directories is not None or bindings.request_page_map is not None
        ) and bindings.cpu_mla is None:
            raise ValueError("CPU residency requires the actual CPU MLA storage")
        if bindings.cpu_mla is not None and (
            bindings.cpu_mla.dtype != torch.uint8
            or bindings.cpu_mla.ndim != 3
            or bindings.cpu_mla.shape[0] != self.layout.layers
            or bindings.cpu_mla.shape[2] != self.layout.mla_row_bytes
            or bindings.cpu_mla.device.type != "cpu"
        ):
            raise ValueError(
                "CPU MLA binding must contain complete native byte records"
            )
        if set(bindings.index_gpu) != set(self.layout.index_anchors):
            raise ValueError("Index bindings must contain every native anchor")
        page_bytes = self.layout.index_page_rows * self.layout.index_row_bytes
        for tensor in bindings.index_gpu.values():
            if (
                tensor.dtype != torch.uint8
                or tensor.ndim != 2
                or tensor.shape[1] != page_bytes
            ):
                raise ValueError("Packed Index byte geometry differs")
        if bindings.request_page_map is not None and (
            bindings.index_cpu is None
            or set(bindings.index_cpu) != set(bindings.index_gpu)
        ):
            raise ValueError("Request policy requires both Index residency planes")
        for tensor in (bindings.index_cpu or {}).values():
            if (
                tensor.dtype != torch.uint8
                or tensor.ndim != 2
                or tensor.shape[1] != page_bytes
                or tensor.device.type != "cpu"
            ):
                raise ValueError("CPU Index byte geometry differs")

    def _boundary(self):
        return (
            self.bindings.boundary_scope()
            if self.bindings.boundary_scope
            else nullcontext()
        )

    def _quiesce(self):
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Snapshot IO must be outside CUDA Graph capture/replay")
        if self.bindings.quiesce is None:
            raise ValueError(
                "An explicit original-reader completion boundary is required"
            )
        self.bindings.quiesce()

    def _owned(self, request, native_rows):
        rows = (
            native_rows.detach().cpu().tolist()
            if hasattr(native_rows, "detach")
            else list(native_rows)
        )
        if len(rows) != request.next_position or len(set(rows)) != len(rows):
            raise ValueError(
                "Native history rows must uniquely cover the saved decode position"
            )
        if any(
            type(row) is not int
            or row <= 0
            or row % self.layout.owners != position % self.layout.owners
            for position, row in enumerate(rows)
        ):
            raise ValueError(
                "Native allocation must preserve sequence-position ownership"
            )
        return [
            row // self.layout.owners
            for position, row in enumerate(rows)
            if position % self.layout.owners == self.owner
        ]

    def _cuda_indices(self, values):
        return self.torch.tensor(
            values, dtype=self.torch.int64, device=self.bindings.gpu_mla.device
        )

    def _locations(self, rows, layer=None):
        b = self.bindings
        if b.directories is not None:
            locations = (
                b.directories[b.layer_groups[layer]]
                .index_select(0, self._cuda_indices(rows))
                .cpu()
                .tolist()
            )
            if any(location == -1 for location in locations):
                raise ValueError("Snapshot refers to an unallocated MLA identity")
            return [
                (False, location) if location >= 0 else (True, -location - 2)
                for location in locations
            ]
        if b.request_page_map is not None:
            pages = [row // self.layout.index_page_rows for row in rows]
            locations = (
                b.request_page_map.index_select(0, self._cuda_indices(pages))
                .cpu()
                .tolist()
            )
            if any(location == -1 for location in locations):
                raise ValueError("Snapshot refers to an unallocated request page")
            return [
                (
                    location < 0,
                    (location if location >= 0 else -location - 2)
                    * self.layout.index_page_rows
                    + row % self.layout.index_page_rows,
                )
                for row, location in zip(rows, locations)
            ]
        return [(False, row) for row in rows]

    def _mla_read(self, layer, rows):
        t, b = self.torch, self.bindings
        result = t.empty((len(rows), self.layout.mla_row_bytes), dtype=t.uint8)
        locations = self._locations(rows, layer)
        for cpu in (False, True):
            positions = [
                i for i, location in enumerate(locations) if location[0] == cpu
            ]
            if not positions:
                continue
            indices = [locations[i][1] for i in positions]
            storage = b.cpu_mla[layer] if cpu else b.gpu_mla[layer]
            index = t.tensor(indices, dtype=t.int64, device=storage.device)
            result[t.tensor(positions)] = storage.index_select(0, index).cpu()
        return result

    def _mla_write(self, layer, rows, data):
        t, b = self.torch, self.bindings
        locations = self._locations(rows, layer)
        for cpu in (False, True):
            positions = [
                i for i, location in enumerate(locations) if location[0] == cpu
            ]
            if not positions:
                continue
            storage = b.cpu_mla[layer] if cpu else b.gpu_mla[layer]
            index = t.tensor(
                [locations[i][1] for i in positions],
                dtype=t.int64,
                device=storage.device,
            )
            storage.index_copy_(0, index, data[t.tensor(positions)].to(storage.device))

    def _index_read(self, anchor, rows):
        t, b = self.torch, self.bindings
        # Index stays in logical GPU pages for active-history CPU offload.
        locations = (
            self._locations(rows)
            if b.request_page_map is not None
            else [(False, row) for row in rows]
        )
        result = t.empty((len(rows), self.layout.index_row_bytes), dtype=t.uint8)
        for cpu in (False, True):
            positions = [
                i for i, location in enumerate(locations) if location[0] == cpu
            ]
            if not positions:
                continue
            source = b.index_cpu[anchor] if cpu else b.index_gpu[anchor]
            values = [locations[i][1] for i in positions]
            pages = t.tensor(
                [row // self.layout.index_page_rows for row in values],
                dtype=t.int64,
                device=source.device,
            )
            offsets = t.tensor(
                [row % self.layout.index_page_rows for row in values],
                dtype=t.int64,
                device=source.device,
            )
            segments, start = [], 0
            for width in self.layout.index_segment_bytes:
                columns = (
                    start
                    + offsets[:, None] * width
                    + t.arange(width, device=source.device)
                )
                segments.append(source[pages[:, None], columns].cpu())
                start += self.layout.index_page_rows * width
            result[t.tensor(positions)] = t.cat(segments, dim=1)
        return result

    def _index_write(self, anchor, rows, data):
        t, b = self.torch, self.bindings
        locations = (
            self._locations(rows)
            if b.request_page_map is not None
            else [(False, row) for row in rows]
        )
        for cpu in (False, True):
            positions = [
                i for i, location in enumerate(locations) if location[0] == cpu
            ]
            if not positions:
                continue
            target = b.index_cpu[anchor] if cpu else b.index_gpu[anchor]
            values = [locations[i][1] for i in positions]
            pages = t.tensor(
                [row // self.layout.index_page_rows for row in values],
                dtype=t.int64,
                device=target.device,
            )
            offsets = t.tensor(
                [row % self.layout.index_page_rows for row in values],
                dtype=t.int64,
                device=target.device,
            )
            selected = data[t.tensor(positions)].to(target.device)
            native_start = canonical_start = 0
            for width in self.layout.index_segment_bytes:
                columns = (
                    native_start
                    + offsets[:, None] * width
                    + t.arange(width, device=target.device)
                )
                target[pages[:, None], columns] = selected[
                    :, canonical_start : canonical_start + width
                ]
                native_start += self.layout.index_page_rows * width
                canonical_start += width

    def save_owner(self, path, requests, native_rows_by_key):
        if self.index_owners != self.layout.owners:
            raise NotImplementedError(
                "Save mixed-ownership records as a canonical prefix"
            )
        with self._boundary():
            self._quiesce()
            writer = SnapshotWriter(path, self.layout, requests, self.owner)
            for request in requests:
                owned = self._owned(request, native_rows_by_key[request.key])
                for plane, width in self.layout.planes().items():
                    layer = int(plane.rsplit("_", 1)[1])
                    read = (
                        self._mla_read if plane.startswith("mla_") else self._index_read
                    )
                    chunk_rows = max(1, self.chunk_bytes // width)

                    def chunks():
                        for start in range(0, len(owned), chunk_rows):
                            tensor = read(layer, owned[start : start + chunk_rows])
                            yield tensor.contiguous().numpy().tobytes()

                    writer.write_plane(request.key, plane, chunks())
            return writer.finish()

    def restore_canonical_request(
        self, path, request_key, native_rows, *, verify_restored=False, chunk_rows=8192
    ):
        """Restore a topology-independent native sequence into this owner.

        Canonical sources contain complete sequence records. Owner selection
        follows sequence position, while allocator IDs remain caller supplied.
        The caller keeps the new request unpublished until all owners succeed.
        """
        from .restore import iter_sequence_plane

        reader = SnapshotReader(path)
        source, target = reader.layout, self.layout
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
        if source.owners == target.owners and self.index_owners == target.owners:
            self.restore_owner(
                path, {request_key: native_rows}, verify_restored=verify_restored
            )
            checked = request.owned_rows(self.owner, target.owners) * sum(
                source.planes().values()
            )
            return dict(
                request_key=request_key,
                owner=self.owner,
                restored_bytes=checked,
                next_position=request.next_position,
                next_token_id=request.next_token_id,
            )
        self._owned(request, native_rows)
        if chunk_rows % target.owners:
            raise ValueError("Canonical chunks must contain complete owner stripes")
        rows = (
            native_rows.detach().cpu().tolist()
            if hasattr(native_rows, "detach")
            else list(native_rows)
        )
        checked = 0
        with self._boundary():
            self._quiesce()
            for plane in source.planes():
                layer = int(plane.rsplit("_", 1)[1])
                degree = (
                    self.index_owners if plane.startswith("index_") else target.owners
                )
                owner = self.owner if degree > 1 else 0
                write = (
                    self._mla_write if plane.startswith("mla_") else self._index_write
                )
                read = self._mla_read if plane.startswith("mla_") else self._index_read
                for start, data in iter_sequence_plane(
                    reader, request_key, plane, chunk_rows=chunk_rows
                ):
                    positions = list(
                        range(
                            (owner - start) % degree,
                            len(data),
                            degree,
                        )
                    )
                    if not positions:
                        continue
                    local = [rows[start + p] // degree for p in positions]
                    selected = data[(owner - start) % degree :: degree].contiguous()
                    write(layer, local, selected)
                    if verify_restored and not self.torch.equal(
                        read(layer, local), selected
                    ):
                        raise RuntimeError(
                            "Canonical native restore changed record bytes"
                        )
                    checked += selected.numel()
            if self.bindings.invalidate:
                self.bindings.invalidate(native_rows)
            self._quiesce()
        return {
            "request_key": request_key,
            "owner": self.owner,
            "restored_bytes": checked,
            "next_position": request.next_position,
            "next_token_id": request.next_token_id,
        }

    def restore_owner(self, path, native_rows_by_key, *, verify_restored=False):
        if self.index_owners != self.layout.owners:
            raise NotImplementedError(
                "Restore mixed-ownership records through the canonical reader"
            )
        """Restore fresh, unpublished allocations; caller rolls back on error.

        The serving adapter must not publish request/prefix bindings until
        every owner's restore has succeeded and original writes have completed.
        """
        reader = SnapshotReader(path)
        if reader.layout != self.layout:
            raise ValueError("Snapshot model, format or owner topology differs")
        if not native_rows_by_key or not set(native_rows_by_key) <= {
            request.key for request in reader.requests
        }:
            raise ValueError(
                "Restore needs fresh native rows for selected saved requests"
            )
        selected = tuple(
            request for request in reader.requests if request.key in native_rows_by_key
        )
        reader.validate_owner_manifest(self.owner)
        with self._boundary():
            self._quiesce()
            for request in selected:
                owned = self._owned(request, native_rows_by_key[request.key])
                for plane, width in self.layout.planes().items():
                    layer = int(plane.rsplit("_", 1)[1])
                    write = (
                        self._mla_write
                        if plane.startswith("mla_")
                        else self._index_write
                    )
                    chunk_bytes = max(width, self.chunk_bytes // width * width)
                    start = 0
                    for raw in reader.iter_plane(
                        self.owner, request.key, plane, chunk_bytes=chunk_bytes
                    ):
                        data = self.torch.frombuffer(
                            bytearray(raw), dtype=self.torch.uint8
                        ).reshape(-1, width)
                        write(layer, owned[start : start + len(data)], data)
                        if verify_restored:
                            read = (
                                self._mla_read
                                if plane.startswith("mla_")
                                else self._index_read
                            )
                            if not self.torch.equal(
                                data, read(layer, owned[start : start + len(data)])
                            ):
                                raise ValueError(
                                    "Restored native bytes differ from snapshot"
                                )
                        start += len(data)
                    if start != len(owned):
                        raise ValueError("Restored row count differs")
            if self.bindings.invalidate:
                for rows in native_rows_by_key.values():
                    self.bindings.invalidate(rows)
            self._quiesce()
        return selected


def bind_udcache_pool(pool, model_fingerprint, *, quiesce, boundary_scope=None):
    """Bind public UDCache storage through the serving adapter's pool object."""
    if pool._flowkv_namespaces != 1:
        raise NotImplementedError(
            "Request snapshots currently require shared ordinary TP ownership"
        )
    native = pool.flowkv_layout
    if native.row_bytes not in (576, 656):
        raise NotImplementedError(
            "Pool snapshot adapter requires explicit raw576 or V32 656-byte records"
        )
    anchors = tuple(native.layer_groups.anchors)
    layout = SnapshotLayout(
        model_fingerprint=model_fingerprint,
        owners=pool._flowkv_storage_size,
        layers=native.layers,
        mla_row_bytes=native.row_bytes,
        index_anchors=anchors,
        kv_format="native-raw-fp8-576"
        if native.row_bytes == 576
        else "native-scaled-fp8-v32",
    )
    host, request = pool.flowkv_host, pool.flowkv_request
    index_gpu = {anchor: pool.index_key_cache.buffer[anchor] for anchor in anchors}
    bindings = PoolSnapshotBindings(
        layout=layout,
        owner_rank=pool.flowkv_rank,
        gpu_mla=pool.flowkv_shards,
        index_gpu=index_gpu,
        layer_groups=tuple(
            native.layer_groups.group(layer) for layer in range(native.layers)
        ),
        cpu_mla=host.host.storage
        if host is not None
        else request.host.storage
        if request is not None
        else None,
        directories=host.directory if host is not None else None,
        request_page_map=request.page_map if request is not None else None,
        index_cpu={
            anchor: request.index_host.storage[i] for i, anchor in enumerate(anchors)
        }
        if request is not None
        else None,
        quiesce=quiesce,
        boundary_scope=boundary_scope,
        public_cache=getattr(pool, "ulacache", None),
        index_owner_count=getattr(pool.index_key_cache, "degree", 1),
    )
    from ulacache.cache.tp_cache import global_ids

    bindings.invalidate = lambda rows: pool.flowkv_arena.invalidate(
        global_ids(pool, rows)
    )
    return CudaRequestSnapshot(bindings)


class FastCudaRequestSnapshot(CudaRequestSnapshot):
    """Reuse each group's physical routing across its MLA layers.

    Residency cannot change during restore. Canonical files, checksums,
    allocation, publication, and native record bytes are unchanged.
    """

    def _owned(self, request, native_rows):
        owned = super()._owned(request, native_rows)
        b = self.bindings
        self._record_routing = NativeRecordRouting(
            gpu_records=b.gpu_mla,
            cpu_records=b.cpu_mla,
            layer_groups=b.layer_groups,
            owner_rows=owned,
            directories=b.directories,
            request_page_map=b.request_page_map,
            page_rows=self.layout.index_page_rows,
        )
        self._route_rows = owned
        self._route_offsets = self._record_routing.row_offsets
        self._group_locations = self._record_routing.group_locations
        self._chunk_routes = self._record_routing.chunk_routes
        return owned

    def _routing(self, layer, rows):
        return self._record_routing.routes_for_rows(layer, rows)

    def _mla_read(self, layer, rows):
        if getattr(self, "_route_rows", None) is None:
            return super()._mla_read(layer, rows)
        self._routing(layer, rows)
        return self._record_routing.read(layer, self._route_offsets[rows[0]], len(rows))

    def _mla_write(self, layer, rows, data):
        if getattr(self, "_route_rows", None) is None:
            return super()._mla_write(layer, rows, data)
        self._routing(layer, rows)
        self._record_routing.write(layer, self._route_offsets[rows[0]], data)

    def restore_owner(self, *args, **kwargs):
        try:
            return super().restore_owner(*args, **kwargs)
        finally:
            if getattr(self, "_record_routing", None) is not None:
                self._record_routing.close()
                self._record_routing = None
            self._route_rows = self._route_offsets = None
            self._group_locations = self._chunk_routes = None

    def restore_canonical_request(
        self, path, request_key, native_rows, *, verify_restored=False, chunk_rows=8192
    ):
        try:
            if self.bindings.public_cache is not None:
                from .format import SnapshotReader
                from .restore import restore_public_request

                if (
                    type(chunk_rows) is not int
                    or chunk_rows < 1
                    or chunk_rows % self.layout.owners
                ):
                    raise ValueError(
                        "Canonical chunks must contain complete owner stripes"
                    )
                return restore_public_request(
                    self,
                    SnapshotReader(path),
                    request_key,
                    native_rows,
                    verify=verify_restored,
                )
            return super().restore_canonical_request(
                path,
                request_key,
                native_rows,
                verify_restored=verify_restored,
                chunk_rows=chunk_rows,
            )
        finally:
            if getattr(self, "_record_routing", None) is not None:
                self._record_routing.close()
                self._record_routing = None
            self._route_rows = self._route_offsets = None
            self._group_locations = self._chunk_routes = None
