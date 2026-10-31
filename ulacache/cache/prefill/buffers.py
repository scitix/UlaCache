"""Two bounded native KV windows for eager request-group prefill.

Records are copied from their current backing without residency promotion or
LRU mutation. Native TopK and causal masks remain unchanged. The two windows
overlap a ready request group's copies with another group's native attention.
Independent future layer selections must be computed before they can be read.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillQueryGroup:
    request: int
    start: int
    stop: int
    records: object
    indices: object


def request_query_ranges(query_counts, query_count):
    """Use native contiguous request ordering, including empty request rows."""
    if query_counts is None:
        raise ValueError("Prefill requires native per-request query counts")
    counts = tuple(query_counts)
    if (
        any(type(count) is not int or count < 0 for count in counts)
        or sum(counts) != query_count
    ):
        raise ValueError("Request query counts must partition the native query tensor")
    start = 0
    ranges = []
    for request, count in enumerate(counts):
        if count:
            ranges.append((request, start, start + count))
        start += count
    return tuple(ranges)


def prepare_query_groups(
    selected, query_counts, capacity, *, query_chunk=256, union=None
):
    """Keep complete TopK sets; split query groups when the union is too wide."""
    import torch

    if (
        type(capacity) is not int
        or capacity < 1
        or type(query_chunk) is not int
        or query_chunk < 1
    ):
        raise ValueError("Prefill record and query bounds must be positive integers")
    if selected.ndim != 2 or selected.dtype not in (torch.int32, torch.int64):
        raise ValueError("Prefill selections must be a two-dimensional integer tensor")

    def split(request, start, stop):
        ids = selected[start:stop]
        needed = torch.unique(ids[ids > 0].to(torch.int32), sorted=True)
        if union is not None:
            needed = union(needed)
        if needed.numel() > capacity:
            if stop - start == 1:
                raise ValueError(
                    "One complete query selection exceeds the prefill buffer"
                )
            middle = (start + stop) // 2
            yield from split(request, start, middle)
            yield from split(request, middle, stop)
            return
        # Compact row zero is valid. Native nonpositive logical IDs are padding.
        mapped = torch.searchsorted(
            needed, ids.contiguous().to(torch.int32), out_int32=True
        )
        mapped = torch.where(ids > 0, mapped, -1).contiguous()
        yield PrefillQueryGroup(request, start, stop, needed, mapped)

    result = []
    for request, first, last in request_query_ranges(query_counts, selected.shape[0]):
        for start in range(first, last, query_chunk):
            result.extend(split(request, start, min(last, start + query_chunk)))
    return tuple(result)


class NativePrefillRecordReader:
    """Copy owner-local CPU/GPU bytes, then gather published peer window rows."""

    def __init__(
        self,
        backend,
        specification,
        pages,
        *,
        device_group=None,
        ipc_manager_factory=None,
    ):
        import torch
        from ulacache.transfer.gpu_peer import NativePagedPeerRecordTransfer

        self.backend = backend
        self.specification = specification
        self.pages = pages
        self.local = getattr(backend, "owner_residency", backend)
        self.owner_count = getattr(backend, "owner_count", 1)
        self.owner_rank = getattr(backend, "owner_rank", 0)
        self.device_group = device_group
        self.plan = backend.execution_plan
        self.peer_pages = ()
        self.publication = None
        storage = self.local.record_storage
        self.storages = tuple(storage.plane_storage)
        self.capacity = (
            pages[0].shape[1] * specification.storage_layout.records_per_page
        )
        device = pages[0].device
        self.ordinals = torch.arange(self.capacity, dtype=torch.int32, device=device)
        self.locations = torch.empty(
            (2, 3, self.capacity), dtype=torch.int32, device=device
        )
        self.peer_transfers = tuple(
            NativePagedPeerRecordTransfer(
                plane, specification.storage_layout.records_per_page
            )
            for plane in specification.storage_layout.record_planes
        )
        if self.owner_count > 1:
            if device_group is None or ipc_manager_factory is None:
                raise ValueError(
                    "Owner prefill requires native device collectives and IPC services"
                )
            from ulacache.transfer.ipc import FlowKVPeerShards
            from ulacache.cache.reader_lifetime import DistributedResidencyPublication

            self.peer_pages = tuple(
                FlowKVPeerShards(
                    plane,
                    backend.owner_process_group,
                    ipc_manager_factory=ipc_manager_factory,
                )
                for plane in pages
            )
            self.publication = DistributedResidencyPublication(
                2,
                device,
                backend.owner_process_group,
                ipc_manager_factory=ipc_manager_factory,
            )

    def union(self, needed):
        """Every owner stages the same union, even when consumer TopK differs."""
        if self.owner_count == 1:
            return needed
        import torch
        import torch.distributed as dist

        largest = torch.tensor(
            [needed.numel()], dtype=torch.int64, device=needed.device
        )
        dist.all_reduce(largest, op=dist.ReduceOp.MAX, group=self.device_group)
        width = int(largest.item())
        if not width:
            return needed
        send = torch.full((width,), -1, dtype=torch.int32, device=needed.device)
        send[: needed.numel()].copy_(needed)
        received = torch.empty(
            self.owner_count * width, dtype=send.dtype, device=send.device
        )
        dist.all_gather_into_tensor(received, send, group=self.device_group)
        return torch.unique(received[received > 0], sorted=True)

    def copy(self, layer, records, slot):
        import torch
        from ulacache.cache.kernels.owner_lookup import map_owner_records
        from ulacache.storage.kernels.paged_write import map_native_write_locations

        count = records.numel()
        owned, gpu, host = self.locations[slot, :, :count]
        if self.publication is not None:
            self.publication.begin_selection(slot)
        domain = self.plan.residency_domain_indices_by_layer[layer]
        directory = self.local.residency_states[domain].directory
        map_owner_records(
            records,
            owned,
            owner_rank=self.owner_rank,
            owner_count=self.owner_count,
            owner_logical_capacity=directory.numel(),
        )
        map_native_write_locations(owned, directory, gpu, host)
        bank = self.plan.storage_bank_indices_by_layer[layer]
        destinations = self.ordinals[:count]
        for storage, output in zip(self.storages, self.pages):
            storage.record_transfer.copy(
                storage.device_record_pages[bank], output[slot], gpu, destinations
            )
            storage.record_transfer.copy(
                storage.host_record_pool.storage[bank], output[slot], host, destinations
            )
        if self.publication is not None:
            self.publication.publish_residency(slot)
            owners = (records % self.owner_count).to(torch.int32)
            owners = torch.where(owners != self.owner_rank, owners, -1).contiguous()
            for transfer, peers, output in zip(
                self.peer_transfers, self.peer_pages, self.pages
            ):
                transfer.gather(
                    peers.ptrs,
                    owners,
                    destinations,
                    output[slot],
                    destinations,
                    source_bank_byte_offset=slot * output.stride(0),
                )

    def finish(self, slot):
        if self.publication is not None:
            self.publication.finish_readers(slot)

    def inventory(self):
        tensors = [
            self.ordinals,
            self.locations,
            *(peers.ptrs for peers in self.peer_pages),
        ]
        result = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in tensors
        }
        if self.publication is not None:
            result.update(self.publication.get_device_storage_inventory())
        return result

    def close(self):
        if self.publication is not None:
            self.publication.close()
        for peers in self.peer_pages:
            peers.close()


class RequestPrefillBuffers:
    """Two reusable native windows shared by every layer and request group.

    The adapter serializes logical writes/releases with a model forward. Source
    backing stays fixed while these copies run. Read-only copies never promote
    records, choose victims, change directory entries or touch decode LRU state.
    """

    def __init__(
        self,
        backend,
        specification,
        record_capacity,
        *,
        device_group=None,
        ipc_manager_factory=None,
        query_chunk=256,
        prefetch=True,
    ):
        import torch

        if type(record_capacity) is not int or record_capacity < 1:
            raise ValueError("Prefill buffer record capacity must be positive")
        self.specification = specification
        self.record_capacity = record_capacity
        self.query_chunk = query_chunk
        self.prefetch = prefetch
        layout = specification.storage_layout
        page_count = (
            record_capacity + layout.records_per_page - 1
        ) // layout.records_per_page
        device = backend.active.device
        self.pages = tuple(
            torch.empty(
                (2, page_count, plane.bytes_per_page(layout.records_per_page)),
                dtype=torch.uint8,
                device=device,
            )
            for plane in layout.record_planes
        )
        self.reader = NativePrefillRecordReader(
            backend,
            specification,
            self.pages,
            device_group=device_group,
            ipc_manager_factory=ipc_manager_factory,
        )
        self.stream = torch.cuda.Stream(device=device)
        self.ready = tuple(torch.cuda.Event() for _ in range(2))
        self.retired = tuple(torch.cuda.Event() for _ in range(2))
        self.used = [False, False]
        self.pending_layer = None
        self.query_counts = None
        self.copy_count = 0
        self.group_count = 0
        self.maximum_union = 0

    def begin_forward(self, query_counts):
        if self.pending_layer is not None:
            raise RuntimeError(
                "Consume the pending prefill selection before starting another forward"
            )
        self.pending_layer = None
        self.query_counts = (
            None
            if query_counts is None
            else tuple(int(count) for count in query_counts)
        )

    def select(self, layer, ids):
        if self.pending_layer is not None:
            raise RuntimeError(
                "Consume the previous prefill selection before replacing it"
            )
        self.pending_layer = layer
        return ids

    def read_records(self, layer, records, read_native):
        """Reconstruct native MHA prefix rows through the same borrowed windows.

        A complete native prefix is one selection, with no query splitting or
        truncation. The native reader returns independent rows before the slot
        retires. Preserve this forward's real request partition for later sparse
        layers; the single row below describes a record read, not model queries.
        """
        if records.ndim != 1:
            raise ValueError("Native prefix reads require one-dimensional logical IDs")
        counts = self.query_counts
        selected = records.reshape(1, -1)
        self.select(layer, selected)
        self.query_counts = (1,)
        try:
            return self.run(
                selected,
                selected,
                lambda query, planes, indices: read_native(
                    planes, indices.reshape(-1)
                ).unsqueeze(0),
            )[0]
        finally:
            self.query_counts = counts

    def run(self, queries, selected, attend):
        """Native callback receives (query slice, native planes, compact IDs)."""
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Request prefill buffers require eager execution")
        if self.pending_layer is None:
            raise RuntimeError("Prepare native prefill selection before attention")
        layer = self.pending_layer
        groups = prepare_query_groups(
            selected,
            self.query_counts,
            self.record_capacity,
            query_chunk=self.query_chunk,
            union=self.reader.union,
        )
        compute = torch.cuda.current_stream(queries.device)
        outputs = []

        def prepare(index):
            slot, group = index % 2, groups[index]
            stream = self.stream if self.prefetch else compute
            if self.prefetch:
                stream.wait_stream(compute)
            with torch.cuda.stream(stream):
                group.records.record_stream(stream)
                if self.used[slot]:
                    self.retired[slot].wait(stream)
                self.reader.copy(layer, group.records, slot)
                self.ready[slot].record(stream)
            self.copy_count += group.records.numel()
            self.maximum_union = max(self.maximum_union, group.records.numel())

        if groups:
            prepare(0)
        for index, group in enumerate(groups):
            slot = index % 2
            self.ready[slot].wait(compute)
            if index + 1 < len(groups):
                prepare(index + 1)
            output = attend(
                queries[group.start : group.stop],
                tuple(plane[slot] for plane in self.pages),
                group.indices,
            )
            if output.shape[0] != group.stop - group.start:
                raise ValueError(
                    "Native prefill reader must return independent query-ordered output"
                )
            outputs.append(output)
            self.reader.finish(slot)
            self.retired[slot].record(compute)
            self.used[slot] = True
            self.group_count += 1
        self.pending_layer = None
        return (
            torch.cat(outputs, dim=0) if outputs else queries.new_empty(queries.shape)
        )

    def inventory(self):
        result = {
            plane.untyped_storage().data_ptr(): plane.untyped_storage().nbytes()
            for plane in self.pages
        }
        result.update(self.reader.inventory())
        return result

    def close(self):
        self.stream.synchronize()
        for used, retired in zip(self.used, self.retired):
            if used:
                retired.synchronize()
        self.reader.close()
