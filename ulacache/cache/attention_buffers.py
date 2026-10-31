"""Attention cache working buffers, grouped by shared Index ownership."""

from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from ulacache.storage.workspace import FlowKVLayout
from ulacache.cache.layers import LayerGroupPlan


@dataclass
class FlowKVC4Group:
    tags: torch.Tensor
    pins: torch.Tensor
    hints: torch.Tensor
    remap: torch.Tensor
    pull_rows: torch.Tensor
    pull_slots: torch.Tensor
    pull_counts: torch.Tensor
    histogram: torch.Tensor
    cursors: torch.Tensor
    totals: torch.Tensor
    errors: torch.Tensor


class FlowKVC4Arena:
    """FlowKV metadata plus native-paged local-KV and prefill buffers."""

    def __init__(
        self,
        layout: FlowKVLayout,
        *,
        device: torch.device,
        bytes_per_page: int,
        layer_groups: LayerGroupPlan,
    ) -> None:
        if bytes_per_page < layout.page_size * layout.row_bytes:
            raise ValueError("Invalid FlowKV record page byte geometry")
        self.layout = layout
        if (
            layer_groups.layer_count != layout.layers
            or layer_groups.group_count != layout.groups
        ):
            raise ValueError("Resident layers and planner groups disagree")
        self.layer_groups = layer_groups
        self.bytes_per_page = bytes_per_page
        self.local_kv_buffer_pages = layout.local_kv_buffer_pages
        self.pool_pages = layout.pool_rows // layout.page_size
        if (
            layout.rows_per_layer % layout.page_size
            or layout.pool_rows % layout.page_size
        ):
            raise ValueError("FlowKV record rows must be page aligned")

        self.storage = torch.zeros(
            (layout.layers, self.local_kv_buffer_pages, bytes_per_page),
            dtype=torch.uint8,
            device=device,
        )
        # A single native-paged buffer is reconstructed layer-by-layer for
        # prefill. Decode only touches the compact per-layer local KV buffer.
        self.prefill = torch.zeros(
            (layout.prefill_buffer_bytes // bytes_per_page, bytes_per_page),
            dtype=torch.uint8,
            device=device,
        )
        # Packing current rows and reconstructing history must use disjoint
        # storage: a physical destination may alias another staging source.
        self.staging = torch.zeros(
            (layout.staging_buffer_bytes // bytes_per_page, bytes_per_page),
            dtype=torch.uint8,
            device=device,
        )

        i32 = {"dtype": torch.int32, "device": device}
        self.hints = torch.full((layout.groups, layout.pool_rows), -1, **i32)
        self.tags = torch.full((layout.groups, layout.capacity), -1, **i32)
        self.pins = torch.zeros(
            (layout.groups, layout.capacity), dtype=torch.int64, device=device
        )
        self.counts = torch.zeros((layout.groups, layout.max_query_tokens), **i32)
        self.histograms = torch.zeros((layout.groups, 258), **i32)
        self.remaps = torch.empty(
            (layout.groups, layout.max_query_tokens, layout.topk), **i32
        )
        self.pull_rows = torch.empty_like(self.remaps)
        self.pull_slots = torch.empty_like(self.remaps)
        self.cursors = torch.zeros((layout.groups, 2), dtype=torch.int64, device=device)
        self.totals = torch.zeros((layout.groups, 2), **i32)
        self.errors = torch.zeros((layout.groups,), **i32)
        self.step = torch.zeros((1,), dtype=torch.int64, device=device)
        self.active = torch.zeros((1,), **i32)
        self.valid = torch.zeros((1,), **i32)
        self.current_keys = torch.full((layout.max_query_tokens * 4,), -1, **i32)
        self.current_values = torch.full_like(self.current_keys, -1)
        metadata_groups = tuple(
            FlowKVC4Group(
                tags=self.tags[group],
                pins=self.pins[group],
                hints=self.hints[group],
                remap=self.remaps[group],
                pull_rows=self.pull_rows[group],
                pull_slots=self.pull_slots[group],
                pull_counts=self.counts[group],
                histogram=self.histograms[group],
                cursors=self.cursors[group],
                totals=self.totals[group],
                errors=self.errors[group : group + 1],
            )
            for group in range(layout.groups)
        )
        self.groups = {
            layer: metadata_groups[group]
            for layer, group in enumerate(layer_groups.layer_to_group)
        }
        self._module = None

    @property
    def local_kv_buffer_bytes(self) -> int:
        return self.storage.nbytes

    @property
    def prefill_bytes(self) -> int:
        return self.prefill.nbytes

    @property
    def staging_bytes(self) -> int:
        return self.staging.nbytes

    @property
    def metadata_bytes(self) -> int:
        return sum(
            tensor.nbytes
            for tensor in (
                self.hints,
                self.tags,
                self.pins,
                self.counts,
                self.histograms,
                self.remaps,
                self.pull_rows,
                self.pull_slots,
                self.cursors,
                self.totals,
                self.errors,
                self.step,
                self.active,
                self.valid,
                self.current_keys,
                self.current_values,
            )
        )

    def kernel(self):
        if self._module is None:
            raise RuntimeError("Record transport must be compiled before capture")
        return self._module

    def begin(self, current: torch.Tensor, valid: torch.Tensor | None = None) -> None:
        if current.ndim != 1 or current.numel() > self.layout.max_query_tokens:
            raise ValueError(
                "FlowKV record decode batch exceeds the configured query bound"
            )
        if (
            current.dtype not in (torch.int32, torch.int64)
            or not current.is_contiguous()
        ):
            raise ValueError(
                "FlowKV record current rows must be contiguous device integers"
            )
        if valid is None:
            self.valid.fill_(current.numel())
            valid = self.valid
        self.kernel().begin(
            current,
            valid,
            self.layout.groups,
            self.layout.pool_rows,
            self.step,
            self.active,
            self.current_keys,
            self.current_values,
            self.cursors,
            self.totals,
            self.hints,
            self.counts,
            self.histograms,
        )

    def invalidate(self, rows: torch.Tensor, *, layer_id: int | None = None) -> None:
        if (
            rows.ndim != 1
            or rows.dtype not in (torch.int32, torch.int64)
            or rows.device != self.storage.device
            or not rows.is_contiguous()
        ):
            raise ValueError(
                "FlowKV record invalidation rows must be contiguous device integers"
            )
        group = None if layer_id is None else self.layer_groups.group(layer_id)
        hints = self.hints if group is None else self.hints[group : group + 1]
        self.kernel().invalidate(rows, hints.shape[0], self.layout.pool_rows, hints)

    def clear_local_residency(self, *, readers_retired: bool) -> None:
        """Invalidate drained resident copies without resetting published epochs.

        Cursors are local append/occupancy watermarks, not synchronization
        epochs. Keep every storage object and all payload bytes for captured
        graphs; the next begin overwrites its per-forward planning scratch.
        """
        if readers_retired is not True:
            raise RuntimeError(
                "Retire every resident reader before clearing local residency"
            )
        if (
            self.storage.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "Local residency cannot be cleared during CUDA Graph capture"
            )
        self.hints.fill_(-1)
        self.tags.fill_(-1)
        self.pins.zero_()
        self.cursors.zero_()

    def report(self) -> dict[str, int]:
        return {
            **self.layout.report(),
            "local_kv_buffer_bytes": self.local_kv_buffer_bytes,
            "prefill_buffer_bytes": self.prefill_bytes,
            "staging_buffer_bytes": self.staging_bytes,
            "prefill_scratch_extra_bytes": self.prefill_bytes,
            "shared_metadata_bytes": self.metadata_bytes,
        }
