"""Index communication over the pinned H200 and B300 FlashAG transports."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

__all__ = ["IndexTransport", "create_index_owner"]


class IndexTransport(ABC):
    """Shared lifecycle and dense/compact score interface for both devices."""

    device: Any
    gathered_workspace: Any
    world_size: int
    rank_in_group: int
    max_queries: int
    max_page_width: int
    compact_scores: bool
    unpacks_scores: bool

    def validate_bounds(self, max_queries, max_page_width):
        if self._closed or (max_queries, max_page_width) != (
            self.max_queries,
            self.max_page_width,
        ):
            raise ValueError("Shared Index transport requires identical Graph bounds")

    @contextmanager
    def graph_capture(self):
        if self._closed:
            raise RuntimeError("Index transport is closed")
        yield

    @abstractmethod
    def all_gather_compact(self, output, local, prefix, packs):
        """Write rank-major packed FP32 scores using GPU payload descriptors."""

    @abstractmethod
    def close(self):
        """Release the transport after its captured readers finish."""

    def _initialize_dense_gather(self):
        import torch

        self._dense_query_indices = torch.arange(
            self.max_queries + 1, dtype=torch.int32, device=self.device
        )
        self._dense_prefix = torch.empty_like(self._dense_query_indices)
        self._dense_packs = torch.empty(1, dtype=torch.int64, device=self.device)

    def _release_dense_gather(self):
        self._dense_query_indices = self._dense_prefix = self._dense_packs = None

    def all_gather_into_tensor(self, output, local):
        """Full rows are the compact format with uniformly full prefixes."""
        import torch

        if self._closed or local.ndim != 2:
            raise ValueError("Dense Index gather requires an open transport and matrix")
        queries, columns = local.shape
        if not 0 < queries <= self.max_queries or columns % 4:
            raise ValueError("Dense Index shape exceeds the aligned query bounds")
        prefix = self._dense_prefix[: queries + 1]
        torch.mul(self._dense_query_indices[: queries + 1], columns, out=prefix)
        self._dense_packs.fill_(queries * columns // 4)
        self.all_gather_compact(output, local, prefix, self._dense_packs)


def create_index_owner(
    serving_group, *, max_queries, max_page_width, device, ipc_manager_factory
) -> IndexTransport:
    """Construct one compact-score interface; hardware selection stays here."""
    import torch

    device = torch.device(device)
    capability = torch.cuda.get_device_capability(device)
    arguments = dict(
        max_queries=max_queries,
        max_page_width=max_page_width,
        device=device,
        compact_scores=True,
    )
    if capability == (9, 0):
        from . import h200_layout
        from .owner import FlashAGIndexOwner

        return FlashAGIndexOwner(serving_group, transport_ops=h200_layout, **arguments)
    if capability == (10, 3):
        from . import b300_layout
        from .packed import PackedFlashAGIndexOwner

        return PackedFlashAGIndexOwner(
            serving_group,
            transport_ops=b300_layout,
            ipc_manager_factory=ipc_manager_factory,
            **arguments,
        )
    raise NotImplementedError(
        f"Index communication is not validated for CUDA {capability}"
    )
