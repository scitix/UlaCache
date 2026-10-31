"""UlaCache tier adapter for the common two-window layer prefill pipeline.

No scheduling is implemented here. LayerPrefillPipeline owns streams/events;
this adapter supplies read-only history and the existing native writer bytes.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnifiedPrefillReadPlan:
    record_ids: Any
    fresh_positions: Any
    input_record_indices: Any
    current_rows: Any

    @property
    def record_count(self):
        return self.record_ids.numel()

    @property
    def keepalive_tensors(self):
        return (
            self.record_ids,
            self.fresh_positions,
            self.input_record_indices,
            self.current_rows,
        )

    def map_native_rows(self, rows):
        import torch

        mapped = torch.searchsorted(
            self.record_ids, rows.contiguous().to(torch.int32), out_int32=True
        )
        safe = mapped.clamp(max=max(0, self.record_count - 1))
        valid = (rows > 0) & (mapped < self.record_count)
        if self.record_count:
            valid &= self.record_ids[safe] == rows
        return torch.where(valid, mapped, -1).to(torch.int32).contiguous()


def prepare_read_plan(full_history_rows, current_rows, *, capacity_records=1048576):
    """Compact the complete native history, never a selected-TopK union."""
    import torch
    from ulacache.cache.prefill.window import compact_topk_rows

    if full_history_rows is None:
        full_history_rows = current_rows.new_empty((0,))
    if (
        full_history_rows.device != current_rows.device
        or current_rows.ndim != 1
        or type(capacity_records) is not int
        or capacity_records < 1
    ):
        raise ValueError(
            "The shared full-history plan requires a positive record bound and matching devices"
        )
    rows = full_history_rows.reshape(-1).to(torch.int32).contiguous()
    # Some native readers publish a prefix-only table. Always include the
    # current writer's extend rows; compaction removes overlap when present.
    complete = torch.cat((rows, current_rows.to(torch.int32)))
    lengths = torch.full((1,), complete.numel(), dtype=torch.int32, device=rows.device)
    needed, remapped, fresh = compact_topk_rows(
        complete.reshape(1, -1), lengths, current_rows
    )
    if needed.numel() > capacity_records:
        raise ValueError(
            "Full history exceeds prefill admission workspace bound; not a KV capacity endpoint"
        )
    return UnifiedPrefillReadPlan(
        needed,
        fresh,
        remapped.reshape(-1)[: rows.numel()].reshape(full_history_rows.shape),
        current_rows,
    )


class UnifiedPrefillMaterializer:
    """Callbacks supplied to LayerPrefillPipeline; canonical ownership is read-only."""

    def __init__(self, engine):
        self.engine = engine
        self._current = {}

    def begin(self):
        # Prior decode follower writes must finish before canonical readout.
        # This installs existing CUDA dependencies; it does not copy/promote KV.
        self.engine._drain_transfers()
        self._current.clear()

    def record_current(self, layer, rows, nope, rope):
        """Borrow tensors after the ordinary canonical native writer runs."""
        # A native first-chunk MHA layer may never read cached KV. Do not
        # retain one unused activation/staging payload for every target layer.
        self._current.clear()
        self._current[layer] = (rows, nope, rope)

    def gather_history(self, layer, plan, window):
        from ulacache.cache.kernels.read_only_prefill import gather_history

        group = self.engine.groups.group(layer)
        gather_history(
            plan.record_ids,
            plan.fresh_positions,
            self.engine.directory[group],
            self.engine.gpu[layer],
            self.engine.host.ptrs,
            window,
            layer=layer,
            host_slots=self.engine.host_slots,
            row_bytes=self.engine.row_bytes,
            launch=self.engine.prefill_exchange_launch,
        )

    def install_current(self, layer, plan, window):
        from ulacache.cache.kernels.read_only_prefill import install_current

        if layer not in self._current:
            raise RuntimeError(
                "Native current KV must be encoded/written before prefill read"
            )
        rows, nope, rope = self._current.pop(layer)
        install_current(
            plan.record_ids,
            plan.fresh_positions,
            rows,
            nope,
            rope,
            window,
            row_bytes=self.engine.row_bytes,
            launch=self.engine.prefill_exchange_launch,
        )

    def finish(self):
        """Call only after the shared pipeline has retired its outstanding copies."""
        self._current.clear()
