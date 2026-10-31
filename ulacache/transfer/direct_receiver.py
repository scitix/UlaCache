"""Canonical rank-local receive ownership; no KV payload staging storage.

Calls run on the allocator/scheduler thread. Native logical-page ownership
protects unpublished CPU records; GPU destinations additionally leave the
eviction ring until native completion or drained-abort retirement.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from ulacache.transfer.direct_receive import (
    DirectReceivePlan,
    RecordBank,
    RecordRegion,
    order_unpublished_cpu_locations,
)


@dataclass
class _Pending:
    plan: DirectReceivePlan
    pages: set[int]
    timing: dict


class DirectRecordReceiver:
    def __init__(self, cache, *, audit=None):
        native = cache._execution_engine
        from ulacache.cache.local_cache import UnifiedLocalResidency

        if isinstance(native, UnifiedLocalResidency):
            engine = native
        else:
            from ulacache.transfer.native_paged_receive import native_record_receive_view

            engine = native_record_receive_view(cache)
            engine.frames = native.registration_frames
            engine.states = native.residency_states
            engine.rings = native.presence_rings
            engine.max_queries = native.maximum_query_count
            engine.topk = native.maximum_selection_width
            engine.logical_slots = (
                cache.cache_specification.capacity_plan.logical_record_capacity
            )
        self.native_engine = native
        self.cache, self.engine = cache, engine
        self.page_size = cache.cache_specification.storage_layout.records_per_page
        self._banks = [
            RecordBank(
                layer,
                engine.groups.group(layer),
                RecordRegion(
                    engine.gpu[layer].data_ptr(),
                    engine.gpu[layer].nbytes,
                    engine.row_bytes,
                ),
                RecordRegion(
                    engine.host.storage[layer].data_ptr(),
                    engine.host.storage[layer].nbytes,
                    engine.row_bytes,
                ),
            )
            for layer in range(engine.groups.layer_count)
        ]
        self._pending = {}
        self._waiting = {}
        self._audit = audit
        self._page_owner = {}
        storage = engine.gpu.untyped_storage()
        self._device_regions = {(storage.data_ptr(), storage.nbytes())}
        self._pinned_gpu = [0] * engine.groups.group_count
        # Required by the existing sparse planner, independent of the network
        # and its chunk sizes. This is recyclable GPU capacity, not a buffer.
        self._working_rows = engine.max_queries * (engine.topk + 1) + 1
        if self._working_rows >= engine.gpu_slots:
            raise ValueError("GPU records cannot cover the declared sparse working set")
        self._frozen = False

    def add_device_banks(self, tensors, *, page_size):
        if self._frozen or self._pending or page_size != self.page_size:
            raise ValueError("All canonical banks must be bound before registration")
        for tensor in tensors:
            if not tensor.is_cuda or not tensor.is_contiguous() or tensor.shape[0] < 1:
                raise ValueError(
                    "Draft destinations require canonical contiguous GPU records"
                )
            width = tensor.nbytes // tensor.shape[0]
            self._banks.append(
                RecordBank(
                    len(self._banks),
                    None,
                    RecordRegion(tensor.data_ptr(), tensor.nbytes, width),
                )
            )
            storage = tensor.untyped_storage()
            self._device_regions.add((storage.data_ptr(), storage.nbytes()))

    def device_registration_regions(self):
        """Exact underlying allocations, including canonical draft storage."""
        self._frozen = True
        return tuple(sorted(self._device_regions))

    def additional_registration_regions(self):
        """Register existing CPU allocations; native KVArgs registers GPU banks."""
        self._frozen = True
        # Host storage is one allocation containing all layer planes. Register
        # that allocation once, rather than separate adjacent plane slices.
        host = self.engine.host.storage
        return ((host.data_ptr(), host.nbytes),)

    @property
    def available_records(self):
        """Physical unbound destinations that leave room for active attention."""
        import torch

        free = (
            torch.stack([s.free_counts[:2] for s in self.engine.states]).cpu().tolist()
        )
        return min(
            int(cpu)
            + min(int(gpu), max(0, self.engine.gpu_slots - self._working_rows - pinned))
            for (gpu, cpu), pinned in zip(free, self._pinned_gpu)
        )

    def note_waiting(self, key):
        self._waiting.setdefault(key, time.monotonic())

    def can_accept(self, allocation_records):
        return allocation_records <= self.available_records

    def reserve(self, key, logical_rows, *, request_id=None, prefix_records=0):
        import numpy as np
        import torch
        from ulacache.cache.kernels.local_cache import (
            acquire_receive_locations,
            rebind_unpublished_cpu_receive,
        )

        if key in self._pending:
            raise ValueError("Native room already advertises a direct destination")
        started = time.monotonic()
        rows = logical_rows.to(
            device=self.engine.gpu.device, dtype=torch.int64
        ).contiguous()
        cpu_rows = rows.cpu().numpy()
        heads = cpu_rows[:: self.page_size]
        expected = (heads[:, None] + np.arange(self.page_size)).reshape(-1)[
            : cpu_rows.size
        ]
        if (
            np.any(heads % self.page_size)
            or not np.array_equal(cpu_rows, expected)
            or np.any(cpu_rows <= 0)
            or np.any(cpu_rows >= self.engine.logical_slots)
            or np.any(np.diff(np.sort(cpu_rows)) == 0)
        ):
            raise ValueError(
                "Native receive suffix must contain unique canonical pages"
            )
        for bank in self._banks:
            if bank.group is None and np.any(cpu_rows >= bank.gpu.records):
                raise ValueError("Native draft rows exceed canonical storage")
        pages = set((cpu_rows // self.page_size).tolist())
        if any(page in self._page_owner for page in pages):
            raise ValueError("Native receive pages already have remote writers")
        self.engine._drain_transfers()
        locations = acquire_receive_locations(rows, self.engine.frames)
        cpu_locations = locations.cpu().numpy()
        ordering_started = time.monotonic()
        reordered = order_unpublished_cpu_locations(cpu_locations)
        if reordered:
            locations.copy_(torch.from_numpy(cpu_locations))
            rebind_unpublished_cpu_receive(rows, self.engine.frames, locations)
        ordering_seconds = time.monotonic() - ordering_started
        # Freeze addresses on the owning stream before advertising to another
        # process. Retain CPU metadata only, not a GPU snapshot per live room.
        plan = DirectReceivePlan(
            cpu_rows, cpu_locations, self._banks, page_size=self.page_size
        )
        del locations
        for group in range(len(self._pinned_gpu)):
            self._pinned_gpu[group] += int((cpu_locations[group] >= 0).sum())
        timing = dict(
            room=key,
            request_id=request_id,
            prefix_records=prefix_records,
            received_records=plan.record_count,
            source="canonical_cpu_gpu_direct",
            admission_wait_seconds=started - self._waiting.pop(key, started),
            reserve_seconds=time.monotonic() - started,
            reserved_at=time.monotonic(),
            cpu_receive_slots_reordered=reordered,
            cpu_receive_ordering_seconds=ordering_seconds,
            gpu_records_per_group=[int((v >= 0).sum()) for v in plan.locations],
        )
        self._pending[key] = _Pending(plan, pages, timing)
        for page in pages:
            self._page_owner[page] = key
        self._frozen = True
        return plan

    def metadata_sent(self, key, *, encoding_seconds=None, metadata_bytes=None):
        timing = self._pending[key].timing
        timing["metadata_sent_at"] = time.monotonic()
        if encoding_seconds is not None:
            timing["metadata_encoding_seconds"] = encoding_seconds
        if metadata_bytes is not None:
            timing["metadata_bytes"] = metadata_bytes

    def _retire(self, key, *, transport_quiesced):
        import torch
        from ulacache.cache.kernels.local_cache import release_receive_locations

        if transport_quiesced is not True:
            raise RuntimeError("Remote writes must drain before destination retirement")
        pending = self._pending[key]
        plan = pending.plan
        device = self.engine.gpu.device
        rows = torch.tensor(plan.logical_rows, device=device)
        locations = torch.tensor(plan.locations, device=device)
        release_receive_locations(
            rows, self.engine.frames, locations, transport_quiesced=True
        )
        # Logical frees, subsequent allocations and Graph replay share this
        # stream. No whole-device synchronization or KV copying is required.
        for group in range(len(self._pinned_gpu)):
            self._pinned_gpu[group] -= int((plan.locations[group] >= 0).sum())
        for page in pending.pages:
            self._page_owner.pop(page)
        del self._pending[key]
        if self._audit is not None:
            now = time.monotonic()
            timing = pending.timing
            self._audit(
                dict(
                    timing,
                    retired_at=now,
                    destination_lease_seconds=now - timing["reserved_at"],
                    advertised_to_retired_seconds=now
                    - timing.get("metadata_sent_at", now),
                    transport_quiesced=True,
                    payload_staging_bytes=0,
                )
            )
        return plan.record_count

    def complete(self, key):
        """Called only after native metadata confirms all KV/state writes."""
        return self._retire(key, transport_quiesced=True)

    def logical_pages_freed(self, page_ids, *, transport_quiesced):
        retired = set(int(page) for page in page_ids)
        keys = {self._page_owner[page] for page in retired if page in self._page_owner}
        for key in keys:
            if not self._pending[key].pages.issubset(retired):
                raise RuntimeError(
                    "Native aborted receive did not retire all advertised pages"
                )
        for key in keys:
            self._retire(key, transport_quiesced=transport_quiesced)

    def metadata_inventory(self):
        return dict(
            payload_staging_bytes=0,
            pending_rooms=len(self._pending),
            cpu_directory_bytes=sum(
                p.plan.logical_rows.nbytes + p.plan.locations.nbytes
                for p in self._pending.values()
            ),
            pinned_gpu_rows=list(self._pinned_gpu),
            sparse_working_rows=self._working_rows,
        )


def send_record_chunk(
    engine,
    session_id,
    payload,
    sources,
    source_pages,
    *,
    page_offset,
    record_count,
    record_limit,
    audit=None,
    executors=(),
    max_inflight=16,
    max_segments=4096,
):
    """Write a native chunk through bounded canonical-address batches."""
    started = time.monotonic()
    plan = (
        payload
        if isinstance(payload, DirectReceivePlan)
        else DirectReceivePlan.from_bytes(payload, record_limit=record_limit)
    )
    if type(page_offset) is not int or page_offset < 0:
        raise ValueError("Native chunk offset must be a nonnegative page index")
    from ulacache.transfer.direct_receive import submit_direct_batch
    from concurrent.futures import FIRST_COMPLETED, wait

    executors = tuple(executors)
    if type(max_inflight) is not int or max_inflight < 1:
        raise ValueError("Direct write concurrency must be positive")
    if type(max_segments) is not int or not 1 <= max_segments <= 65536:
        raise ValueError("Direct descriptor batches must contain1..65536 segments")
    batches = iter(
        plan.iter_transfer_batches(
            sources,
            source_pages,
            destination_start=page_offset * plan.page_size,
            record_count=record_count,
            max_segments=max_segments,
        )
    )
    planning_seconds = write_seconds = 0.0
    segments = total_bytes = batches_submitted = peak_segments = peak_inflight = 0
    status = 0
    pending = set()
    exhausted = False
    first_error = None

    def write(batch):
        submitted = time.monotonic()
        result = submit_direct_batch(engine, session_id, batch)
        return result, time.monotonic() - submitted

    def collect(done):
        nonlocal status, write_seconds, first_error
        for future in done:
            try:
                result, seconds = future.result()
                write_seconds += seconds
                if result != 0 and status == 0:
                    status = result
            except BaseException as error:
                status = -1
                if first_error is None:
                    first_error = error

    while True:
        if pending:
            done = {f for f in pending if f.done()}
            collect(done)
            pending -= done
        if not exhausted and status == 0 and len(pending) < max_inflight:
            planned = time.monotonic()
            try:
                batch = next(batches)
            except StopIteration:
                exhausted = True
            except BaseException as error:
                status = -1
                first_error = error
                exhausted = True
            planning_seconds += time.monotonic() - planned
            if not exhausted:
                segments += len(batch[0])
                total_bytes += int(batch[2].sum())
                peak_segments = max(peak_segments, len(batch[0]))
                if executors and (len(batch[0]) == max_segments or pending):
                    executor = executors[batches_submitted % len(executors)]
                    try:
                        pending.add(executor.submit(write, batch))
                    except BaseException as error:
                        status = -1
                        first_error = error
                    peak_inflight = max(peak_inflight, len(pending))
                else:
                    try:
                        result, seconds = write(batch)
                    except BaseException as error:
                        status = -1
                        first_error = error
                    else:
                        write_seconds += seconds
                        if result != 0:
                            status = result
                batches_submitted += 1
                del batch
                continue
        if not pending:
            break
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        collect(done)
        pending -= done
        # Even after a failed submission, join every already submitted write.
        # A nonzero native status still requires the framework's poisoned ACK
        # handling: returning from a timed-out write does not prove DMA drained.
    if audit is not None:
        audit(
            dict(
                records=record_count,
                bytes=total_bytes,
                segments=segments,
                planning_seconds=planning_seconds,
                write_seconds=write_seconds,
                total_seconds=time.monotonic() - started,
                batches=batches_submitted,
                peak_descriptor_batch_segments=peak_segments,
                peak_inflight_batches=peak_inflight,
                descriptor_batch_limit=max_segments,
                max_inflight_limit=max_inflight,
                write_seconds_scope=(
                    "sum_of_possibly_overlapping_calls"
                    if peak_inflight > 1
                    else "serial_calls"
                ),
                status=status,
                page_offset=page_offset,
            )
        )
    if first_error is not None:
        raise first_error
    return status
