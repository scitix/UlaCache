"""Native receive leases with owner-local targets and replicated sidecars.

The destination view binds existing owner directory/storage tensors and the
native reader-drain hook. Allocation, eviction and transport remain native.
Device/RDMA integration requires separate native gates.
"""

from ulacache.transfer.receive_window import PinnedRecordReceiveWindow


class OwnerStripedRecordReceiveWindow(PinnedRecordReceiveWindow):
    def __init__(self, cache, *, degree, rank, namespaces=1, **kwargs):
        if (
            type(degree) is not int
            or degree < 2
            or degree & (degree - 1)
            or type(rank) is not int
            or not 0 <= rank < degree
            or type(namespaces) is not int
            or namespaces != 1
        ):
            raise ValueError(
                "Receive ownership requires one ordinary shared TP namespace"
            )
        engine = cache._execution_engine
        if (
            engine.directory.ndim != 2
            or engine.directory.shape[1] != engine.gpu_slots + engine.host_slots
        ):
            raise ValueError(
                "Owner directory must cover its GPU and CPU records exactly"
            )
        self.degree, self.rank = degree, rank
        super().__init__(cache, **kwargs)

    def reserve(self, key, logical_page_ids, *, record_count=None):
        import torch

        pages = logical_page_ids.to(device="cpu", dtype=torch.int64).reshape(-1)
        if pages.numel() and pages.min().item() < self.degree:
            raise ValueError(
                "Incoming native pages cannot address the reserved TP superpage"
            )
        if pages.numel() and (pages.max().item() + 1) * self.page_size > (
            self.degree * self.engine.directory.shape[1]
        ):
            raise ValueError(
                "Incoming native pages exceed owner-striped history capacity"
            )
        return super().reserve(key, pages, record_count=record_count)

    def _target_record_indices(self, item):
        rows = item["rows"]
        owned = (rows.remainder(self.degree) == self.rank).nonzero().flatten()
        target = rows.index_select(0, owned) // self.degree
        source = item["source_rows"].index_select(0, owned.to("cpu"))
        return target, source

    def memory_inventory(self):
        return dict(
            super().memory_inventory(),
            target_owner_degree=self.degree,
            target_owner_rank=self.rank,
            namespaces=1,
            target_owner_logical_capacity=self.engine.directory.shape[1],
        )
