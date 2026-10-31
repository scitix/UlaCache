# SPDX-License-Identifier: Apache-2.0
"""Physical owner-page transactions for request-resident FlowKV.

This module does not allocate logical identities, protect requests, or choose
inactive prefixes. Native admission supplies those inputs under its own locks.
All MLA layers and packed Index anchors share one physical page location.
Native admission owns the logical IDs, locks and candidate order. GPU address
consumers are connected separately; no logical allocator is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Protocol, Sequence


FREE = -1


def cpu_location(page: int) -> int:
    return -page - 2


@dataclass(frozen=True)
class PageTransfer:
    kind: str  # onload, offload, swap
    logical_page: int
    gpu_page: int
    cpu_page: int
    victim_page: int | None = None


@dataclass(frozen=True)
class CapacityVerdict:
    fits: bool
    reason: str
    onload_pages: int
    future_gpu_pages: int
    inactive_gpu_pages_needed: int
    additional_cpu_pages_needed: int
    permanent_error: bool = False


@dataclass(frozen=True)
class BoundaryTransaction:
    transaction_id: int
    owner_rank: int
    waves: tuple[tuple[PageTransfer, ...], ...]
    map_updates: tuple[tuple[int, int], ...]
    future_gpu_pages: tuple[int, ...]
    reserved_free_gpu_pages: tuple[int, ...]
    reserved_free_cpu_pages: tuple[int, ...]

    def exchange_plan(self, *, plane: str, page_size: int = 64):
        """Original exchange plan rows, grouped by dependency wave.

        MLA expands each page to complete token records. Index copies one
        8448-byte packed page per anchor. Row0/4 are not consumed by exchange;
        directory publication belongs to the coherent page transaction.
        """
        if plane not in ("mla", "index"):
            raise ValueError("plane must be mla or index")
        records = page_size if plane == "mla" else 1
        plans = []
        for wave in self.waves:
            rows = [[], [], [], [], []]
            for transfer in wave:
                for offset in range(records):
                    gpu = transfer.gpu_page * records + offset
                    cpu = transfer.cpu_page * records + offset
                    rows[0].append(-1)
                    rows[1].append(gpu)
                    rows[2].append(cpu if transfer.kind != "offload" else -1)
                    rows[3].append(cpu if transfer.kind != "onload" else -1)
                    rows[4].append(-1)
            plans.append(tuple(tuple(row) for row in rows))
        return tuple(plans)


@dataclass(frozen=True)
class FuturePageReservation:
    reservation_id: int
    owner_rank: int
    gpu_pages: tuple[int, ...]


class BoundaryCompletion(Protocol):
    transaction_id: int

    def assert_complete(self) -> None:
        """Raise unless both planes and pre-existing readers have retired."""


class PageMapPublisher(Protocol):
    def publish(self, owner_rank: int, updates: tuple[tuple[int, int], ...]) -> None:
        """Publish the real fixed GPU page map after all-plane copy completion."""


class _Pages:
    def __init__(self, count: int, reserved: Sequence[int]):
        self.reserved = frozenset(reserved)
        if len(self.reserved) != len(reserved) or any(
            p < 0 or p >= count for p in reserved
        ):
            raise ValueError("invalid reserved physical pages")
        self.free = set(range(count)) - self.reserved
        self.heap = list(self.free)
        heapq.heapify(self.heap)

    def take(self, n: int) -> tuple[int, ...]:
        if n < 0 or n > len(self.free):
            raise ValueError("physical page capacity exceeded")
        result = tuple(heapq.heappop(self.heap) for _ in range(n))
        self.free.difference_update(result)
        return result

    def release(self, pages: Sequence[int]) -> None:
        if len(set(pages)) != len(pages) or any(
            p in self.free or p in self.reserved for p in pages
        ):
            raise ValueError("duplicate or reserved physical page release")
        for page in pages:
            self.free.add(page)
            heapq.heappush(self.heap, page)


class FlowKVRequestPagePool:
    """One owner-local physical map; existing native allocator owns logical IDs."""

    def __init__(
        self,
        *,
        owner_count: int,
        owner_rank: int,
        gpu_token_rows: int,
        cpu_token_rows: int,
        logical_owner_token_rows: int,
        layer_num: int,
        row_bytes: int,
        index_anchor_layers: Sequence[int],
        page_size: int = 64,
        reserved_gpu_pages: Sequence[int] = (),
        reserved_cpu_pages: Sequence[int] = (),
        sentinel_gpu_page: int | None = None,
    ):
        if owner_count < 1 or not 0 <= owner_rank < owner_count or page_size != 64:
            raise ValueError("ordinary owner-striped page64 geometry required")
        if any(
            n <= 0 or n % page_size
            for n in (gpu_token_rows, cpu_token_rows, logical_owner_token_rows)
        ):
            raise ValueError("capacities must be positive whole physical pages")
        if logical_owner_token_rows != gpu_token_rows + cpu_token_rows:
            raise ValueError(
                "native logical owner capacity must equal GPU+CPU capacity"
            )
        anchors = tuple(index_anchor_layers)
        if (
            layer_num < 1
            or row_bytes <= 0
            or row_bytes % 8
            or not anchors
            or len(set(anchors)) != len(anchors)
            or any(i < 0 or i >= layer_num for i in anchors)
        ):
            raise ValueError("invalid complete MLA/packed Index planes")
        if logical_owner_token_rows // page_size >= 2**31 - 1:
            raise ValueError("physical page map must fit signed int32")
        self.owner_count, self.owner_rank = owner_count, owner_rank
        self.page_size, self.allocation_page_size = page_size, page_size * owner_count
        self.layer_num, self.row_bytes, self.index_anchor_layers = (
            layer_num,
            row_bytes,
            anchors,
        )
        self.gpu_page_count, self.cpu_page_count = (
            gpu_token_rows // page_size,
            cpu_token_rows // page_size,
        )
        self.logical_owner_pages = logical_owner_token_rows // page_size
        self._gpu = _Pages(self.gpu_page_count, reserved_gpu_pages)
        self._cpu = _Pages(self.cpu_page_count, reserved_cpu_pages)
        self.gpu_usable_pages = len(self._gpu.free)
        self.cpu_usable_pages = len(self._cpu.free)
        self._map = [FREE] * self.logical_owner_pages
        self.sentinel_gpu_page = sentinel_gpu_page
        if sentinel_gpu_page is not None:
            if sentinel_gpu_page not in self._gpu.reserved:
                raise ValueError("sentinel must have a reserved physical GPU page")
            self._map[0] = sentinel_gpu_page
        self._consumed: dict[int, int] = {}
        self._pending: BoundaryTransaction | None = None
        self._submitted = False
        self._ticket = 0
        self._future: dict[int, set[int]] = {}

    @property
    def gpu_free_pages(self) -> int:
        return len(self._gpu.free)

    @property
    def cpu_free_pages(self) -> int:
        return len(self._cpu.free)

    @property
    def pending_native_binding_count(self) -> int:
        return len(self._consumed)

    def _ids(self, pages: Sequence[int]) -> tuple[int, ...]:
        result = tuple(dict.fromkeys(pages))
        if any(
            not isinstance(p, int)
            or p < 0
            or p >= self.logical_owner_pages
            or (p == 0 and self.sentinel_gpu_page is not None)
            for p in result
        ):
            raise ValueError("logical owner page outside native capacity")
        return result

    def _idle(self) -> None:
        if self._pending is not None:
            raise RuntimeError("a boundary transaction is still pending")

    def location(self, logical_page: int) -> int:
        return self._map[self._ids((logical_page,))[0]]

    def map_row_snapshot(self) -> tuple[int, ...]:
        """Int32 mirror row: GPU>=0, FREE=-1, CPU=-physical_page-2."""
        return tuple(self._map)

    def plane_shapes(self) -> dict:
        return {
            "mla_gpu": (self.layer_num, self.gpu_page_count * 64, self.row_bytes),
            "mla_cpu": (self.layer_num, self.cpu_page_count * 64, self.row_bytes),
            "index_gpu": (len(self.index_anchor_layers), self.gpu_page_count, 8448),
            "index_cpu": (len(self.index_anchor_layers), self.cpu_page_count, 8448),
        }

    def _inputs(
        self, logical_page_ids, future_gpu_pages, inactive_page_ids, protected_page_ids
    ):
        self._idle()
        if self._consumed:
            raise RuntimeError(
                "settle completed native GPU page bindings before admission"
            )
        required = self._ids(logical_page_ids)
        protected = set(self._ids(protected_page_ids)) | set(required)
        if not isinstance(future_gpu_pages, int) or future_gpu_pages < 0:
            raise ValueError(
                "future GPU reservation must be an exact nonnegative page count"
            )
        if any(self._map[p] == FREE for p in required):
            raise ValueError(
                "retained pages must already belong to the native allocator"
            )
        onload = tuple(p for p in required if self._map[p] < FREE)
        candidates = tuple(
            p
            for p in self._ids(inactive_page_ids)
            if p not in protected and self._map[p] >= 0
        )
        h = min(len(onload), self.gpu_free_pages)
        swap = len(onload) - h
        offload = max(0, future_gpu_pages - (self.gpu_free_pages - h))
        need = swap + offload
        reason = "ready"
        permanent_error = len(required) + future_gpu_pages > self.gpu_usable_pages
        if permanent_error:
            reason = "one request exceeds the complete GPU-resident page capacity"
        elif need > len(candidates):
            reason = "insufficient inactive GPU pages"
        elif offload > self.cpu_free_pages + h:
            reason = "insufficient CPU pages for inactive offload"
        return (
            required,
            onload,
            candidates,
            h,
            swap,
            offload,
            CapacityVerdict(
                reason == "ready",
                reason,
                len(onload),
                future_gpu_pages,
                need,
                offload,
                permanent_error,
            ),
        )

    def check_gpu_capacity(
        self,
        logical_page_ids,
        *,
        future_gpu_pages,
        inactive_page_ids=(),
        protected_page_ids=(),
    ) -> CapacityVerdict:
        return self._inputs(
            logical_page_ids, future_gpu_pages, inactive_page_ids, protected_page_ids
        )[-1]

    def reserve_request(
        self,
        logical_page_ids,
        *,
        future_gpu_pages,
        inactive_page_ids=(),
        protected_page_ids=(),
    ) -> BoundaryTransaction:
        _, onload, victims, h, swaps, offloads, verdict = self._inputs(
            logical_page_ids, future_gpu_pages, inactive_page_ids, protected_page_ids
        )
        if not verdict.fits:
            raise ValueError(verdict.reason)
        free_gpu = self._gpu.take(h + min(future_gpu_pages, self.gpu_free_pages - h))
        free_cpu = self._cpu.take(min(offloads, self.cpu_free_pages))
        updates, first, second = {}, [], []
        released_cpu = []
        for logical, gpu in zip(onload[:h], free_gpu[:h]):
            cpu = -self._map[logical] - 2
            first.append(PageTransfer("onload", logical, gpu, cpu))
            updates[logical] = gpu
            released_cpu.append(cpu)
        for logical, victim in zip(onload[h:], victims[:swaps]):
            cpu, gpu = -self._map[logical] - 2, self._map[victim]
            first.append(PageTransfer("swap", logical, gpu, cpu, victim))
            updates[logical], updates[victim] = gpu, cpu_location(cpu)
        destinations = free_cpu + tuple(released_cpu[: offloads - len(free_cpu)])
        future = list(free_gpu[h:])
        for victim, cpu in zip(victims[swaps : swaps + offloads], destinations):
            gpu = self._map[victim]
            second.append(PageTransfer("offload", victim, gpu, cpu))
            updates[victim] = cpu_location(cpu)
            future.append(gpu)
        assert len(future) == future_gpu_pages and len(second) == offloads
        self._ticket += 1
        tx = BoundaryTransaction(
            self._ticket,
            self.owner_rank,
            tuple(w for w in (tuple(first), tuple(second)) if w),
            tuple(sorted(updates.items())),
            tuple(future),
            free_gpu,
            free_cpu,
        )
        self._pending, self._submitted = tx, False
        return tx

    def ensure_resident(self, tx: BoundaryTransaction, executor) -> BoundaryCompletion:
        if self._pending is not tx or self._submitted:
            raise RuntimeError("transaction is not the reserved unsubmitted boundary")
        if executor is None:
            raise NotImplementedError(
                "request boundary requires a real MLA+Index transfer executor"
            )
        # A failed/partial submission must not be silently replayed: swaps would reverse.
        self._submitted = True
        return executor.submit(self, tx)

    def commit(
        self,
        tx: BoundaryTransaction,
        completion: BoundaryCompletion,
        publisher: PageMapPublisher,
    ) -> FuturePageReservation:
        if self._pending is not tx or not self._submitted:
            raise RuntimeError("transaction has not been submitted")
        if completion.transaction_id != tx.transaction_id:
            raise ValueError("completion belongs to another transaction")
        completion.assert_complete()
        if publisher is None:
            raise NotImplementedError(
                "real replicated page-map publication is required"
            )
        # If publication fails, retain the pending transaction; retry publication only.
        publisher.publish(self.owner_rank, tx.map_updates)
        old_cpu = {-self._map[p] - 2 for p, _ in tx.map_updates if self._map[p] < FREE}
        new_cpu = {-loc - 2 for _, loc in tx.map_updates if loc < FREE}
        self._cpu.release(tuple(old_cpu - new_cpu))
        for page, location in tx.map_updates:
            self._map[page] = location
        self._future[tx.transaction_id] = set(tx.future_gpu_pages)
        self._pending, self._submitted = None, False
        return FuturePageReservation(
            tx.transaction_id, self.owner_rank, tx.future_gpu_pages
        )

    def cancel(self, tx: BoundaryTransaction) -> None:
        if self._pending is not tx or self._submitted:
            raise RuntimeError("only an unsubmitted reservation can be cancelled")
        self._gpu.release(tx.reserved_free_gpu_pages)
        self._cpu.release(tx.reserved_free_cpu_pages)
        self._pending = None

    def attach_native_pages(
        self,
        logical_page_ids,
        *,
        publisher: PageMapPublisher,
        tier="GPU",
        reservation: FuturePageReservation | None = None,
    ):
        """Bind externally allocated fresh native pages before the original writer."""
        self._idle()
        if self._consumed:
            raise RuntimeError(
                "settle native GPU bindings before another map publication"
            )
        logical = self._ids(logical_page_ids)
        if any(self._map[p] != FREE for p in logical) or tier not in ("GPU", "CPU"):
            raise ValueError("fresh native pages must be unbound and select GPU or CPU")
        if publisher is None:
            raise NotImplementedError("page-map publication must precede native writes")
        pages_pool = self._gpu if tier == "GPU" else self._cpu
        if reservation is not None:
            available = self._future.get(reservation.reservation_id)
            if (
                tier != "GPU"
                or reservation.owner_rank != self.owner_rank
                or available is None
                or len(logical) > len(available)
            ):
                raise ValueError("invalid or exhausted native future-page reservation")
            physical = tuple(sorted(available)[: len(logical)])
        else:
            physical = pages_pool.take(len(logical))
        updates = tuple(
            (p, q if tier == "GPU" else cpu_location(q))
            for p, q in zip(logical, physical)
        )
        try:
            publisher.publish(self.owner_rank, updates)
        except Exception:
            if reservation is None:
                pages_pool.release(physical)
            raise
        if reservation is not None:
            available.difference_update(physical)
        for page, location in updates:
            self._map[page] = location
        return physical

    def consume_future_pages(
        self, reservation: FuturePageReservation, count: int
    ) -> tuple[int, ...]:
        """Lease targets for the native new-page GPU binding branch.

        The caller must complete the corresponding real GPU map binding before
        writes, and settle its logical IDs through record_consumed_pages at a
        boundary. No native GPU ID is read or synchronized by this method.
        """
        self._idle()
        if not isinstance(count, int) or count < 0:
            raise ValueError("new native page count must be nonnegative")
        available = self._future.get(reservation.reservation_id)
        if (
            reservation.owner_rank != self.owner_rank
            or available is None
            or count > len(available)
        ):
            raise ValueError("invalid or exhausted native future-page reservation")
        physical = tuple(sorted(available)[:count])
        available.difference_update(physical)
        self._consumed.update(dict.fromkeys(physical, reservation.reservation_id))
        return physical

    def return_consumed_pages(
        self, reservation: FuturePageReservation, physical_page_ids
    ) -> None:
        """Return targets only before the native GPU scatter is enqueued.

        The caller owns this submission boundary. A failed or partial CUDA
        submission must stop; it cannot roll back an already visible map.
        """
        self._idle()
        available = self._future.get(reservation.reservation_id)
        physical = tuple(physical_page_ids)
        if (
            reservation.owner_rank != self.owner_rank
            or available is None
            or len(set(physical)) != len(physical)
            or any(
                self._consumed.get(q) != reservation.reservation_id for q in physical
            )
        ):
            raise ValueError("targets do not belong to this pending native reservation")
        for q in physical:
            del self._consumed[q]
        available.update(physical)

    def record_consumed_pages(self, logical_page_ids, physical_page_ids) -> None:
        """Settle completed native GPU bindings at an explicit boundary.

        Inputs are CPU page metadata supplied by the native caller. This method
        does not inspect GPU tensors and must not run for a partial-page step.
        """
        self._idle()
        logical = self._ids(logical_page_ids)
        physical = tuple(physical_page_ids)
        if len(logical) != len(physical) or len(set(physical)) != len(physical):
            raise ValueError("completed page bindings must be unique and aligned")
        if any(self._map[p] != FREE for p in logical):
            raise ValueError("native logical page already bound")
        if any(
            not isinstance(q, int)
            or q < 0
            or q >= self.gpu_page_count
            or q not in self._consumed
            for q in physical
        ):
            raise ValueError("physical page is not a consumed unique GPU target")
        for p, q in zip(logical, physical):
            self._map[p] = q
        for q in physical:
            del self._consumed[q]

    def release_reservation(self, reservation: FuturePageReservation) -> None:
        self._idle()
        if (
            reservation.owner_rank != self.owner_rank
            or reservation.reservation_id not in self._future
        ):
            raise ValueError("unknown future-page reservation")
        self._gpu.release(tuple(self._future.pop(reservation.reservation_id)))

    def reservation_remaining_pages(self, reservation: FuturePageReservation) -> int:
        if (
            reservation.owner_rank != self.owner_rank
            or reservation.reservation_id not in self._future
        ):
            raise ValueError("unknown future-page reservation")
        return len(self._future[reservation.reservation_id])

    def retire_native_pages(
        self, logical_page_ids, *, readers_retired: bool, publisher: PageMapPublisher
    ) -> None:
        """Detach after original readDone; native caller permanently frees IDs."""
        self._idle()
        if self._consumed:
            raise RuntimeError(
                "settle native GPU bindings before permanent page release"
            )
        pages = self._ids(logical_page_ids)
        if not readers_retired:
            raise RuntimeError(
                "native readers must retire before physical page release"
            )
        if publisher is None:
            raise NotImplementedError("page-map retirement requires real publication")
        updates = tuple((p, FREE) for p in pages if self._map[p] != FREE)
        publisher.publish(self.owner_rank, updates)
        self._gpu.release(tuple(self._map[p] for p, _ in updates if self._map[p] >= 0))
        self._cpu.release(
            tuple(-self._map[p] - 2 for p, _ in updates if self._map[p] < FREE)
        )
        for page, _ in updates:
            self._map[page] = FREE

    def clear(self, *, readers_retired: bool) -> None:
        """Idle native-cache reset; never used by a decode transaction."""
        self._idle()
        if not readers_retired or self._consumed:
            raise RuntimeError(
                "clear requires retired readers and settled native bindings"
            )
        self._gpu = _Pages(self.gpu_page_count, tuple(self._gpu.reserved))
        self._cpu = _Pages(self.cpu_page_count, tuple(self._cpu.reserved))
        self._map[:] = [FREE] * self.logical_owner_pages
        if self.sentinel_gpu_page is not None:
            self._map[0] = self.sentinel_gpu_page
        self._future.clear()
