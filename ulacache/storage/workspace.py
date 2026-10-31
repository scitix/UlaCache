"""Attention workspace geometry and lifetimes; no storage allocation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from functools import cached_property

from ulacache.cache.layers import LayerGroupPlan
from ulacache.storage.layout import NativePageLayout, history_capacity as history_capacity

DEFAULT_FLOWKV_LOGICAL_TOKENS_PER_KV_ROW = 1


def needs_prefill_scratch(disaggregation_mode: str) -> bool:
    if disaggregation_mode not in {"null", "prefill", "decode"}:
        raise ValueError(f"Unknown disaggregation mode: {disaggregation_mode}")
    return disaggregation_mode != "decode"


def round_up_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("Capacity must be positive")
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class FlowKVLayout:
    layers: int
    groups: int
    max_batch: int
    pool_rows: int
    row_bytes: int
    disaggregation_mode: str
    topk: int
    max_query_tokens: int
    history_slots: int
    page_size: int
    native_segment_bytes: tuple[int, ...]
    layer_anchors: tuple[int, ...]
    compression_ratio: int = DEFAULT_FLOWKV_LOGICAL_TOKENS_PER_KV_ROW
    page_storage_alignment: int = 1
    max_staging_tokens: int = 0
    prefill_window_rows: int = 0

    def __post_init__(self) -> None:
        self._validate_geometry(
            query_limit=512 if self.compression_ratio > 1 else 256,
            resident_queries=self.max_query_tokens,
        )

    def _validate_geometry(self, *, query_limit, resident_queries) -> None:
        if (
            min(
                self.layers,
                self.groups,
                self.max_batch,
                self.pool_rows,
                self.row_bytes,
                self.topk,
                self.max_query_tokens,
                self.history_slots,
                self.page_size,
                self.compression_ratio,
                self.page_storage_alignment,
            )
            <= 0
        ):
            raise ValueError("FlowKV layout dimensions must be positive")
        if self.groups > self.layers:
            raise ValueError("Invalid FlowKV group or local-KV-buffer capacity")
        if self.max_query_tokens < self.max_batch:
            raise ValueError("Query bound is smaller than the largest decode graph")
        if self.max_query_tokens > query_limit or self.topk % 256:
            raise ValueError(
                f"FlowKV kernel requires at most {query_limit} queries "
                "and TopK divisible by 256"
            )
        if self.max_query_tokens & (self.max_query_tokens - 1):
            raise ValueError("FlowKV query bound must be a power of two")
        if self.history_slots >= 2**31 or self.history_slots % self.page_size:
            raise ValueError(
                "FlowKV history capacity must be page aligned and fit int32"
            )
        if self.history_slots < resident_queries * self.topk:
            raise ValueError("History arena cannot hold the largest TopK batch")
        if self.row_bytes % 2:
            raise ValueError("FlowKV KV row must have an even byte width")
        if self.native_page.row_bytes != self.row_bytes:
            raise ValueError("Native segments must cover the complete KV row")
        if (
            self.layer_groups.layer_count != self.layers
            or self.layer_groups.group_count != self.groups
        ):
            raise ValueError("KV layer owners disagree with metadata group allocation")
        needs_prefill_scratch(self.disaggregation_mode)
        if self.prefill_window_rows < 0 or self.prefill_window_rows % self.page_size:
            raise ValueError("FlowKV prefill window must contain complete native pages")

    @cached_property
    def native_page(self) -> NativePageLayout:
        return NativePageLayout(
            self.page_size, self.native_segment_bytes, self.page_storage_alignment
        )

    @cached_property
    def layer_groups(self) -> LayerGroupPlan:
        return LayerGroupPlan.from_layer_anchors(self.layer_anchors)

    @property
    def capacity(self) -> int:
        return self.history_slots

    @property
    def rows_per_layer(self) -> int:
        # Reserve a current-token row for every request, then round to a page.
        rows = self.capacity + self.max_query_tokens
        return (rows + self.page_size - 1) // self.page_size * self.page_size

    @property
    def pool_pages(self) -> int:
        return self.pool_rows // self.page_size

    @property
    def local_kv_buffer_pages(self) -> int:
        return self.rows_per_layer // self.page_size

    @property
    def storage_bytes_per_page(self) -> int:
        """Physical bytes in one packed cache page.

        Plain MLA rows use an alignment of one, while a packed cache may
        require a format-defined padded page.  Keeping this in the common
        layout makes the budget use the same byte geometry as allocation.
        """
        page_bytes = self.page_size * self.row_bytes
        alignment = self.page_storage_alignment
        return (page_bytes + alignment - 1) // alignment * alignment

    @property
    def local_kv_buffer_bytes(self) -> int:
        return self.layers * self.local_kv_buffer_pages * self.storage_bytes_per_page

    @property
    def prefill_buffer_bytes(self) -> int:
        """Physical allocation for the native-paged prefill reconstruction view.

        Native reconstruction addresses all pages. The optional CP backend
        remaps complete query groups into a fixed window shared across layers.
        """
        return (
            (
                self.prefill_window_rows // self.page_size
                if self.prefill_window_rows
                else self.pool_pages
            )
            * self.storage_bytes_per_page
            if needs_prefill_scratch(self.disaggregation_mode)
            else 0
        )

    @property
    def prefill_lookup_bytes(self) -> int:
        if not self.prefill_window_rows or not needs_prefill_scratch(
            self.disaggregation_mode
        ):
            return 0
        slots = round_up_power_of_two(2 * self.prefill_window_rows)
        return (2 * slots + 1) * 4

    @property
    def staging_buffer_bytes(self) -> int:
        if self.compression_ratio == 1:
            return 0
        tokens = max(self.max_staging_tokens, self.max_query_tokens)
        return (
            (tokens + self.page_size - 1) // self.page_size
        ) * self.storage_bytes_per_page

    @property
    def selected_bytes(self) -> int:
        """Compatibility alias for the local KV buffer's physical allocation."""
        return self.local_kv_buffer_bytes

    @property
    def scratch_bytes(self) -> int:
        """Compatibility alias for :attr:`prefill_buffer_bytes`."""
        return self.prefill_buffer_bytes

    @property
    def hint_bytes(self) -> int:
        return self.groups * self.pool_rows * 4

    @property
    def fixed_metadata_bytes(self) -> int:
        # Per group: tags, pins, remap, pull lists, counters and histogram.
        # Global: step, active query count and current-token hash table.
        return (
            self.groups
            * (
                12 * self.capacity
                + 12 * self.max_query_tokens * self.topk
                + 4 * self.max_query_tokens
                + 28
                + 4 * 258
            )
            + 16
            + 8 * 4 * self.max_query_tokens
        )

    @property
    def metadata_bytes(self) -> int:
        return self.fixed_metadata_bytes + self.hint_bytes

    @property
    def staging_index_bytes(self) -> int:
        """Static C4 staging index tables owned by the packed MLA adapter.

        The packed C4 store path keeps one int64 source-position vector and
        two int32 row-id vectors so fused store and FlowKV transport can be
        captured without allocating or casting an index tensor.  They are not
        arena metadata, but they are persistent device allocations and must be
        included in the capacity budget.
        """
        if self.compression_ratio == DEFAULT_FLOWKV_LOGICAL_TOKENS_PER_KV_ROW:
            return self.max_staging_tokens * 4
        source_position_bytes = 8
        source_row_bytes = 4
        target_row_bytes = 4
        return self.pool_rows * (
            source_position_bytes + source_row_bytes + target_row_bytes
        )

    @property
    def extra_bytes(self) -> int:
        return (
            self.local_kv_buffer_bytes
            + self.prefill_buffer_bytes
            + self.prefill_lookup_bytes
            + self.metadata_bytes
        )

    def report(self) -> dict[str, int]:
        return {
            "rows_per_layer": self.rows_per_layer,
            "local_kv_buffer_bytes": self.local_kv_buffer_bytes,
            "prefill_buffer_bytes": self.prefill_buffer_bytes,
            "scratch_view_bytes": self.prefill_buffer_bytes,
            "prefill_scratch_extra_bytes": self.prefill_buffer_bytes,
            "prefill_lookup_bytes": self.prefill_lookup_bytes,
            "fixed_metadata_bytes": self.fixed_metadata_bytes,
            "hint_bytes": self.hint_bytes,
            "metadata_bytes": self.metadata_bytes,
            "staging_index_bytes": self.staging_index_bytes,
            "staging_buffer_bytes": self.staging_buffer_bytes,
            "global_history_slots": self.capacity,
            "max_query_tokens": self.max_query_tokens,
            "current_hash_entries": 4 * self.max_query_tokens,
        }


@dataclass(frozen=True)
class ResidentQueryLayout(FlowKVLayout):
    """Size large KV buffers from real queries, independent of hash padding.

    The existing native kernel ABI uses power-of-two query metadata. That
    bound does not determine the number of resident KV records. Zero retains
    the original sizing contract for callers without a narrower real bound.
    """

    max_resident_queries: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.max_query_tokens) is not int
            or not 0 < self.max_query_tokens <= 2048
            or self.max_query_tokens & (self.max_query_tokens - 1)
        ):
            raise ValueError(
                "Native resident query metadata requires a power-of-two bound at most2048"
            )
        if (
            type(self.max_resident_queries) is not int
            or self.max_resident_queries < 0
            or not self.max_batch
            <= self.resident_query_capacity
            <= self.max_query_tokens
        ):
            raise ValueError(
                "Resident query capacity must fit the native metadata bound"
            )
        self._validate_geometry(
            query_limit=2048, resident_queries=self.resident_query_capacity
        )

    @property
    def resident_query_capacity(self) -> int:
        return self.max_resident_queries or self.max_query_tokens

    @property
    def rows_per_layer(self) -> int:
        rows = self.capacity + self.resident_query_capacity
        return (rows + self.page_size - 1) // self.page_size * self.page_size


@dataclass(frozen=True)
class PlainPrefillLayout(ResidentQueryLayout):
    """Charge one fixed KV window, current staging, and bounded union workspace."""

    def __post_init__(self):
        super().__post_init__()
        if self.compression_ratio != 1:
            raise ValueError("Plain prefill requires uncompressed native records")
        if self.prefill_window_rows and (
            self.prefill_window_rows < self.topk or self.max_staging_tokens < 1
        ):
            raise ValueError(
                "Plain prefill needs a complete TopK window and a staging bound"
            )

    @property
    def prefill_lookup_bytes(self):
        if not self.prefill_window_rows or not needs_prefill_scratch(
            self.disaggregation_mode
        ):
            return 0
        queries = min(
            self.max_staging_tokens, 8 * self.prefill_window_rows // self.topk
        )
        return (
            96 * max(self.prefill_window_rows, queries * self.topk)
            + 16 * self.max_staging_tokens
        )

    @property
    def staging_buffer_bytes(self):
        if not self.prefill_window_rows or not needs_prefill_scratch(
            self.disaggregation_mode
        ):
            return 0
        tokens = max(self.max_staging_tokens, self.max_query_tokens)
        return (
            (tokens + self.page_size - 1) // self.page_size
        ) * self.storage_bytes_per_page

    @property
    def extra_bytes(self):
        return super().extra_bytes + self.staging_buffer_bytes


@dataclass(frozen=True)
class PipelineTPLayout(PlainPrefillLayout):
    """Keep actual verify rows and two independent encoded prefill windows."""

    prefill_layers: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.prefill_layers != 2:
            raise ValueError("Pipeline TP requires two independent prefill windows")

    @property
    def total_prefill_buffer_bytes(self):
        return self.prefill_layers * self.prefill_buffer_bytes

    @property
    def prefill_lookup_bytes(self):
        if not self.prefill_window_rows or not needs_prefill_scratch(
            self.disaggregation_mode
        ):
            return 0
        return 96 * self.prefill_window_rows + 16 * self.max_staging_tokens

    @classmethod
    def from_native(cls, layout):
        if isinstance(layout, cls):
            return layout
        return cls(
            **{
                field.name: getattr(layout, field.name, field.default)
                for field in fields(cls)
            }
        )


@dataclass(frozen=True)
class AttentionWorkspaceLifetime:
    storage_bank_index: int
    first_consumer_layer: int
    last_consumer_layer: int


@dataclass(frozen=True)
class AttentionWorkspacePlan:
    workspace_indices_by_storage_bank: tuple[int, ...]
    workspace_count: int


def create_attention_workspace_plan(
    storage_bank_count: int, lifetimes: Iterable[AttentionWorkspaceLifetime]
):
    """Assign static workspaces to banks with disjoint consumer intervals.

    A bank may have several successive selections. Its workspace stays fixed,
    but other banks can use it between those selections. Overlapping intervals
    require different workspaces. This assumes serialized readers and writers
    on the caller's current stream; cold payload banks retain their own storage.
    """
    lifetimes_by_bank = [[] for _ in range(storage_bank_count)]
    for lifetime in lifetimes:
        lifetimes_by_bank[lifetime.storage_bank_index].append(lifetime)

    def overlap(first_bank, second_bank):
        return any(
            first.first_consumer_layer <= second.last_consumer_layer
            and second.first_consumer_layer <= first.last_consumer_layer
            for first in lifetimes_by_bank[first_bank]
            for second in lifetimes_by_bank[second_bank]
        )

    workspace_banks = []
    workspace_indices = []
    for bank in range(storage_bank_count):
        for workspace, occupants in enumerate(workspace_banks):
            if all(not overlap(bank, occupant) for occupant in occupants):
                occupants.append(bank)
                workspace_indices.append(workspace)
                break
        else:
            workspace_indices.append(len(workspace_banks))
            workspace_banks.append([bank])
    return AttentionWorkspacePlan(tuple(workspace_indices), len(workspace_banks))
