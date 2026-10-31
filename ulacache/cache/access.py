"""Sparse access views and per-forward execution metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class SparseAccessSpecification:
    """Selection geometry before backend lowering.

    Record IDs name the stable logical storage domain, never GPU locations.
    Block selection and candidate-relative selection require explicit native
    composition into this domain before the initial backend can consume them.
    ``maximum_padded_query_count`` counts actual attention queries, including verify tokens.
    """

    maximum_padded_query_count: int
    maximum_selected_units: int
    selection_unit: Literal["record", "block"] = "record"
    records_per_selected_unit: int = 1
    selection_identifier_space: Literal["logical", "candidate"] = "logical"

    def __post_init__(self):
        if any(
            type(n) is not int or n < 1
            for n in (
                self.maximum_padded_query_count,
                self.maximum_selected_units,
                self.records_per_selected_unit,
            )
        ):
            raise ValueError("Sparse access bounds must be positive")
        if self.selection_unit not in ("record", "block"):
            raise ValueError("Unknown sparse selection unit")
        if self.selection_unit == "record" and self.records_per_selected_unit != 1:
            raise ValueError("A record selection names exactly one record")
        if self.selection_identifier_space not in ("logical", "candidate"):
            raise ValueError("Unknown sparse ID space")

    @property
    def maximum_selected_record_count(self) -> int:
        return self.maximum_selected_units * self.records_per_selected_unit


@dataclass(frozen=True)
class SparseAttentionAccessPlan:
    """Canonical padded int32[queries, width] IDs and current int32[queries].

    These are device tensors. IDs <= 0 are padding/sentinels; every positive
    ID must already be registered. Padded queries are masked by the active
    count supplied at begin_forward. The adapter retains request ordering,
    causal lengths and speculative acceptance; the cache does not infer them.
    Ragged native selections must be lowered without changing their meaning.
    """

    selected_logical_record_ids: Any
    current_logical_record_ids: Any


@dataclass(frozen=True)
class ResidentAttentionInputs:
    """Native byte planes and remapped GPU row IDs; no attention computation.

    Views are borrowed through the consumer's after_attention boundary. They
    must not be retained across subsequent plans, releases, clear or close.
    The reader's causal metadata remains a separate framework-owned input.
    ``record_layout`` describes the first plane for single-plane compatibility;
    readers of separate K/V planes use each plane's bound storage specification.
    """

    record_planes: tuple[Any, ...]
    device_record_indices: Any
    records_per_page: int
    record_layout: Literal["record_major", "segment_major"] = "record_major"

    @property
    def gpu_buffer(self):
        """Borrow the single native plane; separate K/V uses record_planes."""
        if len(self.record_planes) != 1:
            raise ValueError("Use record_planes for a view with separate native planes")
        return self.record_planes[0]

    @property
    def slot_ids(self):
        """Physical record indices in selection order, never stable logical IDs."""
        return self.device_record_indices


ResidentView = ResidentAttentionInputs


@dataclass(frozen=True)
class ForwardExecutionContext:
    """Graph padding and active queries are separate from request admission.

    ``active_query_count`` is a static integer or a persistent int32[1] CUDA tensor.
    Verification counts attention queries, including draft positions. The
    adapter supplies the original request ordering and causal reader metadata.
    """

    phase: Literal["prefill", "decode", "verify", "idle"]
    active_query_count: Any
    padded_query_count: int
