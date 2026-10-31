"""Bounded, registered CPU receive pages for an exclusive record cache.

The transport writes only this window. Publication runs on the owning thread
after transport completion, using the then-current GPU/CPU record directory.
Native allocation retirement supplies the abort/drain boundary.
No serving-framework or device dependency is imported at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReceivePageLease:
    key: object
    generation: int
    pages: tuple[int, ...]


class ReceivePageAllocator:
    def __init__(self, page_count):
        if type(page_count) is not int or page_count < 1:
            raise ValueError("Receive page capacity must be a positive integer")
        self._free = list(range(page_count - 1, -1, -1))
        self._leases = {}
        self._generation = 0

    @property
    def available_pages(self):
        return len(self._free)

    def reserve(self, key, page_count):
        if key in self._leases:
            raise ValueError("Receive key already owns a live lease")
        if type(page_count) is not int or page_count < 0:
            raise ValueError("Receive page count must be a nonnegative integer")
        if page_count > len(self._free):
            raise BufferError("Bounded receive window has insufficient pages")
        self._generation += 1
        lease = ReceivePageLease(
            key, self._generation, tuple(self._free.pop() for _ in range(page_count))
        )
        self._leases[key] = lease
        return lease

    def release(self, lease, *, transport_quiesced):
        if transport_quiesced is not True:
            raise RuntimeError("Receive pages cannot be reused before transport drains")
        if self._leases.get(lease.key) != lease:
            raise ValueError("Unknown or stale receive lease")
        del self._leases[lease.key]
        self._free.extend(reversed(lease.pages))


def incremental_state_indices(source, destination, prefix_tokens, page_size):
    """Retain exactly the native page suffix not promised by the receiver."""
    if type(prefix_tokens) is not int or prefix_tokens < 0:
        raise ValueError("Committed prefix must be a nonnegative token count")
    if type(page_size) is not int or page_size < 1 or prefix_tokens % page_size:
        raise ValueError("Committed transfer prefix must be page aligned")
    if len(source) != len(destination):
        raise ValueError("Source and destination state page counts differ")
    skip = prefix_tokens // page_size
    if skip > len(source):
        raise ValueError("Committed prefix exceeds the state payload")
    return source[skip:], destination[skip:]


def native_receive_record_count(page_count, page_size, record_count=None):
    """Exclude the unallocated tail of a native final transfer page."""
    if (
        type(page_count) is not int
        or page_count < 0
        or type(page_size) is not int
        or page_size < 1
    ):
        raise ValueError(
            "Native receive pages require nonnegative count and positive size"
        )
    capacity = page_count * page_size
    if record_count is None:
        return capacity
    if type(record_count) is not int or record_count < 0:
        raise ValueError("Valid native records require a nonnegative integer count")
    if record_count == capacity == 0:
        return 0
    if not max(0, capacity - page_size) < record_count <= capacity:
        raise ValueError(
            "Valid native records must cover all pages except the final tail"
        )
    return record_count


def incremental_receive_state_pages(
    canonical_pages, receive_pages, *, prefix_tokens, page_size
):
    """Keep the native full-state shape while staging its uncached suffix.

    Native senders remove the page-aligned committed prefix once. Preserve its
    placeholder entries and replace only remaining destinations with pages from
    the KV receive lease. Receive page zero is valid; native logical padding is
    not a canonical request page. This function does not retire remote writes.
    """
    canonical, received = tuple(canonical_pages), tuple(receive_pages)
    if (
        type(page_size) is not int
        or page_size < 1
        or type(prefix_tokens) is not int
        or prefix_tokens < 0
        or prefix_tokens % page_size
    ):
        raise ValueError("Native receive state prefix must be page aligned")
    if (
        any(type(page) is not int or page < 1 for page in canonical)
        or any(type(page) is not int or page < 0 for page in received)
        or len(set(canonical)) != len(canonical)
        or len(set(received)) != len(received)
    ):
        raise ValueError(
            "Native canonical and receive state pages must be unique and valid"
        )
    skip = prefix_tokens // page_size
    if skip > len(canonical) or len(received) != len(canonical) - skip:
        raise ValueError(
            "Native receive state suffix differs from the KV receive lease"
        )
    return canonical[:skip] + received


def transfer_pages_from_retired_allocation_pages(
    page_ids, *, allocation_page_size, transfer_page_size
):
    """Expand native allocation pages into their canonical transfer-page IDs.

    Ordinary TP reserves a page64 superpage per owner, while native transport
    and receive leases address page64 records. The caller must separately retire
    remote writes before applying these IDs to a receive window.
    """
    if (
        type(allocation_page_size) is not int
        or allocation_page_size < 1
        or type(transfer_page_size) is not int
        or transfer_page_size < 1
        or allocation_page_size % transfer_page_size
    ):
        raise ValueError("Allocation pages must contain complete native transfer pages")
    pages = tuple(page_ids)
    if any(type(page) is not int or page < 1 for page in pages) or len(
        set(pages)
    ) != len(pages):
        raise ValueError("Retired allocation pages must be unique and nonzero")
    ratio = allocation_page_size // transfer_page_size
    if any((page + 1) * ratio > 2**31 for page in pages):
        raise ValueError("Native transfer page IDs exceed their signed int32 metadata")
    return tuple(page * ratio + offset for page in pages for offset in range(ratio))


class PinnedRecordReceiveWindow:
    """Publish completed native bytes into preallocated CPU record locations.

    ``cache`` is a rank-local contiguous-record UnifiedMemoryCache. The caller
    excludes incoming records from model execution until publication returns.
    Their GPU/CPU locations may change while transfer is in flight. Native deferred
    release must retire all remote writes before calling logical_pages_freed.
    """

    def __init__(
        self,
        cache,
        *,
        record_capacity,
        allocate_host_record_storage,
        device,
        copy_chunk_records=4096,
        execution_engine=None,
    ):
        import torch

        engine = (
            cache._execution_engine if execution_engine is None else execution_engine
        )
        page_size = cache.cache_specification.storage_layout.records_per_page
        if record_capacity < 1 or record_capacity % page_size:
            raise ValueError("Receive capacity must contain complete native pages")
        if copy_chunk_records < 1:
            raise ValueError("Publication copy chunk must be positive")
        self.cache, self.engine = cache, engine
        self.page_size, self.record_capacity = page_size, record_capacity
        self.storage = allocate_host_record_storage(
            layers=engine.groups.layer_count,
            slots=record_capacity,
            row_bytes=engine.row_bytes,
            device=device,
        )
        if (
            self.storage.storage.device.type != "cpu"
            or not self.storage.storage.is_contiguous()
        ):
            raise ValueError(
                "Receive storage must be contiguous registered CPU records"
            )
        self.copy_scratch = torch.empty(
            (copy_chunk_records, engine.row_bytes),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self.device_copy_scratch = torch.empty_like(self.copy_scratch, device=device)
        self.pages = ReceivePageAllocator(record_capacity // page_size)
        self._pending = {}
        self._sidecars = {}
        self._allocate_host_record_storage = allocate_host_record_storage
        self._registration_frozen = False
        self._publication_layer_groups = None
        # Reuse the allocator's mapped host pointers. Providers without this
        # capability retain the portable complete-record copy implementation.
        if (
            engine.gpu.is_cuda
            and hasattr(self.storage, "ptrs")
            and hasattr(engine.host, "ptrs")
        ):
            for storage in (self.storage, engine.host):
                pointers = storage.ptrs
                if (
                    pointers.device != engine.gpu.device
                    or pointers.ndim != 1
                    or pointers.numel() != 1
                    or pointers.dtype not in (torch.int64, torch.uint64)
                ):
                    raise ValueError(
                        "Registered receive storage requires one device-accessible base pointer"
                    )
            self._publication_layer_groups = torch.tensor(
                [
                    engine.groups.group(layer)
                    for layer in range(engine.groups.layer_count)
                ],
                dtype=torch.int32,
                device=engine.gpu.device,
            )

    @property
    def available_records(self):
        return self.pages.available_pages * self.page_size

    def contiguous_buffer_infos(self):
        self._registration_frozen = True
        banks = self.storage.storage.unbind(0)
        return (
            [bank.data_ptr() for bank in banks],
            [bank.nbytes for bank in banks],
            [self.page_size * self.engine.row_bytes for _ in banks],
        )

    def add_sidecar_record_banks(self, name, destinations, *, page_size):
        """Stage additional canonical banks under the same transport lease.

        The caller supplies exclusively allocated logical destinations. Remote
        writes reach host staging only; publication updates those destinations
        after transport completion, before native execution can read them.
        Add every bank before exposing any registered buffer address.
        """
        import torch

        if self._registration_frozen or self._pending:
            raise RuntimeError(
                "Receive banks cannot change after registration or reservation"
            )
        if not isinstance(name, str) or not name or name in self._sidecars:
            raise ValueError("Receive sidecar requires a unique nonempty name")
        if page_size != self.page_size:
            raise ValueError("All receive banks must use the same native page size")
        banks = list(destinations)
        if not banks or any(
            bank.ndim < 2
            or not bank.is_contiguous()
            or bank.device != self.engine.gpu.device
            or bank.shape[0] < 1
            for bank in banks
        ):
            raise ValueError(
                "Sidecar destinations must be contiguous canonical record banks"
            )
        byte_banks = [
            bank.view(torch.uint8).reshape(bank.shape[0], -1) for bank in banks
        ]
        width = byte_banks[0].shape[1]
        if width < 1 or any(bank.shape[1] != width for bank in byte_banks):
            raise ValueError("Sidecar record widths must be positive and uniform")
        storage = self._allocate_host_record_storage(
            layers=len(banks),
            slots=self.record_capacity,
            row_bytes=width,
            device=self.engine.gpu.device,
        )
        if storage.storage.device.type != "cpu" or not storage.storage.is_contiguous():
            raise ValueError(
                "Sidecar receive storage must be contiguous registered CPU records"
            )
        scratch = torch.empty(
            (self.copy_scratch.shape[0], width),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        device_scratch = torch.empty_like(scratch, device=self.engine.gpu.device)
        self._sidecars[name] = dict(
            storage=storage,
            destinations=byte_banks,
            copy_scratch=scratch,
            device_copy_scratch=device_scratch,
            row_bytes=width,
        )

        def buffer_infos():
            self._registration_frozen = True
            receive_banks = storage.storage.unbind(0)
            return (
                [bank.data_ptr() for bank in receive_banks],
                [bank.nbytes for bank in receive_banks],
                [self.page_size * width for _ in receive_banks],
            )

        return buffer_infos

    def reserve(self, key, logical_page_ids, *, record_count=None):
        import torch

        if key in self._pending:
            raise ValueError("Receive room already has a pending publication")
        page_ids = logical_page_ids.to(device="cpu", dtype=torch.int64).reshape(-1)
        if page_ids.unique().numel() != page_ids.numel() or (page_ids <= 0).any():
            raise ValueError("Incoming native pages must be unique and nonzero")
        valid = native_receive_record_count(
            page_ids.numel(), self.page_size, record_count
        )
        rows = (
            page_ids[:, None] * self.page_size + torch.arange(self.page_size)
        ).reshape(-1)[:valid]
        if rows.numel() and any(
            rows.max().item() >= bank.shape[0]
            for group in self._sidecars.values()
            for bank in group["destinations"]
        ):
            raise ValueError(
                "Incoming sidecar records exceed canonical destination capacity"
            )
        self._validate_additional_destinations(rows)
        device_rows = rows.to(self.engine.directory.device)
        lease = self.pages.reserve(key, page_ids.numel())
        self._registration_frozen = True
        source_rows = (
            torch.tensor(lease.pages, dtype=torch.int64)[:, None] * self.page_size
            + torch.arange(self.page_size)
        ).reshape(-1)[:valid]
        self._pending[key] = dict(
            lease=lease,
            logical_pages=set(page_ids.tolist()),
            rows=device_rows,
            source_rows=source_rows,
        )
        return lease

    def transfer_pages(self, key):
        return self._pending[key]["lease"].pages

    def _target_record_indices(self, item):
        return item["rows"], item["source_rows"]

    def _validate_additional_destinations(self, rows):
        """Optional native state banks validate before acquiring receive pages."""

    def _publish_additional_banks(self, item):
        """Optional native state banks share the final stream/lease boundary."""

    def publish(self, key):
        import torch

        item = self._pending[key]
        self.engine._drain_transfers()
        count, chunk = item["source_rows"].numel(), self.copy_scratch.shape[0]
        target_rows, target_source_rows = self._target_record_indices(item)
        for start in range(0, target_source_rows.numel(), chunk):
            stop = min(start + chunk, target_source_rows.numel())
            device_locations = self.engine.directory.index_select(
                1, target_rows[start:stop]
            )
            locations = device_locations.cpu()
            if (
                (locations == -1).any()
                or (locations >= self.engine.gpu_slots).any()
                or (locations < -self.engine.host_slots - 1).any()
            ):
                raise RuntimeError(
                    "Incoming native records were released before publication"
                )
            temporary = self.copy_scratch[: stop - start]
            source_rows = target_source_rows[start:stop]
            if self._publication_layer_groups is not None:
                from ulacache.transfer.kernels.receive_publication import (
                    publish_registered_records,
                )

                publish_registered_records(
                    source_rows.to(self.engine.gpu.device),
                    device_locations,
                    self._publication_layer_groups,
                    self.storage,
                    self.engine.host,
                    self.engine.gpu,
                )
                continue
            for layer in range(self.engine.groups.layer_count):
                group = self.engine.groups.group(layer)
                torch.index_select(
                    self.storage.storage[layer], 0, source_rows, out=temporary
                )
                loc = locations[group]
                cold, hot = loc <= -2, loc >= 0
                if cold.any():
                    self.engine.host.storage[layer].index_copy_(
                        0, -loc[cold] - 2, temporary[cold]
                    )
                if hot.any():
                    rows = loc[hot].to(self.engine.gpu.device)
                    payload = self.device_copy_scratch[: rows.numel()]
                    payload.copy_(temporary[hot])
                    self.engine.gpu[layer].index_copy_(0, rows, payload)
        for group in self._sidecars.values():
            for start in range(0, count, chunk):
                stop = min(start + chunk, count)
                source_rows = item["source_rows"][start:stop]
                destination_rows = item["rows"][start:stop]
                temporary = group["copy_scratch"][: stop - start]
                payload = group["device_copy_scratch"][: stop - start]
                if self._publication_layer_groups is not None and hasattr(
                    group["storage"], "ptrs"
                ):
                    from ulacache.transfer.kernels.receive_publication import (
                        publish_registered_sidecar,
                    )

                    device_source_rows = source_rows.to(self.engine.gpu.device)
                    for layer, destination in enumerate(group["destinations"]):
                        publish_registered_sidecar(
                            device_source_rows,
                            destination_rows,
                            group["storage"],
                            layer,
                            destination,
                        )
                    continue
                for layer, destination in enumerate(group["destinations"]):
                    torch.index_select(
                        group["storage"].storage[layer], 0, source_rows, out=temporary
                    )
                    payload.copy_(temporary)
                    destination.index_copy_(0, destination_rows, payload)
        self._publish_additional_banks(item)
        torch.cuda.current_stream(self.engine.gpu.device).synchronize()
        self.pages.release(item["lease"], transport_quiesced=True)
        del self._pending[key]
        return count

    def logical_pages_freed(self, logical_pages):
        """Called only after the native allocator's remote-drain retirement."""
        released = set(logical_pages)
        for key, item in list(self._pending.items()):
            item["logical_pages"].difference_update(released)
            if not item["logical_pages"]:
                self.pages.release(item["lease"], transport_quiesced=True)
                del self._pending[key]

    def memory_inventory(self):
        metadata = sum(
            item[name].nbytes
            for item in self._pending.values()
            for name in ("source_rows",)
        )
        return dict(
            cpu_receive_bytes=self.storage.nbytes
            + sum(group["storage"].nbytes for group in self._sidecars.values()),
            cpu_copy_scratch_bytes=self.copy_scratch.nbytes
            + sum(group["copy_scratch"].nbytes for group in self._sidecars.values()),
            cpu_receive_metadata_bytes=metadata,
            gpu_receive_metadata_bytes=self.device_copy_scratch.nbytes
            + (
                self._publication_layer_groups.nbytes
                if self._publication_layer_groups is not None
                else 0
            )
            + sum(
                group["device_copy_scratch"].nbytes for group in self._sidecars.values()
            )
            + sum(item["rows"].nbytes for item in self._pending.values()),
            record_capacity=self.record_capacity,
            sidecar_bank_count=sum(
                len(group["destinations"]) for group in self._sidecars.values()
            ),
            pending_publications=len(self._pending),
        )
