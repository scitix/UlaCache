"""Owner KV and native target/draft Index staging under one receive lease."""

from ulacache.transfer.index_pages import (
    PAGE_BYTES,
    PAGE_RECORDS,
    plan_index_record_copies,
    validate_index_record_plan,
)
from ulacache.transfer.owner_receive_window import OwnerStripedRecordReceiveWindow


class OwnerIndexRecordReceiveWindow(OwnerStripedRecordReceiveWindow):
    """Bind existing native Index banks without changing their ownership.

    The adapter supplies all canonical banks before registering any pointer.
    Native transport completion is the precondition for inherited ``publish``.
    Native remote-write drain is the precondition for logical-page retirement.
    Index keys retain their stable native logical addresses; only target KV
    uses the current CPU/GPU residency directory.
    """

    def __init__(self, cache, **kwargs):
        if cache.cache_specification.storage_layout.records_per_page != PAGE_RECORDS:
            raise ValueError("Native owner Index reception requires page64 records")
        super().__init__(cache, **kwargs)
        self._index_banks = {}

    def add_sidecar_record_banks(self, name, destinations, *, page_size):
        if name in self._index_banks:
            raise ValueError("Receive bank names must be unique across KV and Index")
        return super().add_sidecar_record_banks(name, destinations, page_size=page_size)

    def add_index_page_banks(self, name, destinations, *, replicated=False):
        """Register target-owner or replicated-draft native page64 byte banks."""
        import torch

        if self._registration_frozen or self._pending:
            raise RuntimeError(
                "Index receive banks cannot change after registration or reservation"
            )
        if (
            not isinstance(name, str)
            or not name
            or name in self._index_banks
            or name in self._sidecars
            or type(replicated) is not bool
        ):
            raise ValueError(
                "Index receive banks require a unique name and explicit ownership"
            )
        banks = list(destinations)
        if not banks or any(
            not isinstance(bank, torch.Tensor)
            or bank.dtype != torch.uint8
            or not bank.is_contiguous()
            or bank.device != self.engine.gpu.device
            or not bank.nbytes
            or bank.nbytes % PAGE_BYTES
            for bank in banks
        ):
            raise ValueError(
                "Index destinations must be complete contiguous native byte pages"
            )
        if len({bank.nbytes for bank in banks}) != 1:
            raise ValueError(
                "One native Index receive group must have uniform bank capacity"
            )
        intervals = sorted(
            (bank.data_ptr(), bank.data_ptr() + bank.nbytes)
            for bank in banks
            + [
                bank
                for group in self._index_banks.values()
                for bank in group["destinations"]
            ]
        )
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("Canonical native Index banks must not overlap")
        storage = self._allocate_host_record_storage(
            layers=len(banks),
            slots=self.record_capacity // PAGE_RECORDS,
            row_bytes=PAGE_BYTES,
            device=self.engine.gpu.device,
        )
        if (
            storage.storage.device.type != "cpu"
            or not storage.storage.is_contiguous()
            or storage.storage.dtype != torch.uint8
            or tuple(storage.storage.shape)
            != (len(banks), self.record_capacity // PAGE_RECORDS, PAGE_BYTES)
        ):
            raise ValueError(
                "Index receive storage must preserve registered native pages"
            )
        degree, rank = (1, 0) if replicated else (self.degree, self.rank)
        self._index_banks[name] = dict(
            storage=storage, destinations=banks, degree=degree, rank=rank
        )

        def buffer_infos():
            self._registration_frozen = True
            receive_banks = storage.storage.unbind(0)
            return (
                [bank.data_ptr() for bank in receive_banks],
                [bank.nbytes for bank in receive_banks],
                [PAGE_BYTES for _ in receive_banks],
            )

        return buffer_infos

    def _validate_additional_destinations(self, rows):
        logical = rows.tolist()
        for group in self._index_banks.values():
            plan = plan_index_record_copies(
                logical, range(len(logical)), degree=group["degree"], rank=group["rank"]
            )
            validate_index_record_plan(
                plan,
                group["storage"].storage[0].nbytes,
                group["destinations"][0].nbytes,
            )

    def _publish_additional_banks(self, item):
        from ulacache.transfer.index_tensor_copy import publish_index_record_tensors

        logical, received = item["rows"].cpu().tolist(), item["source_rows"].tolist()
        for group in self._index_banks.values():
            plan = plan_index_record_copies(
                logical, received, degree=group["degree"], rank=group["rank"]
            )
            for source, destination in zip(
                group["storage"].storage, group["destinations"]
            ):
                publish_index_record_tensors(
                    source,
                    destination,
                    plan,
                    transport_quiesced=True,
                    copy_chunk_records=self.copy_scratch.shape[0],
                )

    def memory_inventory(self):
        inventory = dict(super().memory_inventory())
        inventory["cpu_receive_bytes"] += sum(
            group["storage"].nbytes for group in self._index_banks.values()
        )
        inventory["index_receive_bank_count"] = sum(
            len(group["destinations"]) for group in self._index_banks.values()
        )
        inventory["index_publication_max_chunk_records"] = self.copy_scratch.shape[0]
        return inventory
