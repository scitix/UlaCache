"""Reuse bounded native prefill read plans across layers in one forward.

The public cache supplies complete encoded records. Native callbacks retain
dequantization and attention; plans own metadata, never layer payloads.
"""

from ulacache.cache.prefill.window import (
    CompactPrefillRecordReadPlan,
    CompactPrefillRecordWindow,
    compact_topk_rows,
)


class PrefillSequenceBudget:
    """Charge full reconstructed histories separately from new-token chunks."""

    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Prefill sequence capacity must be a positive integer")
        self.capacity, self.used = capacity, 0

    def can_reserve(self, prefix_len, extend_len):
        if prefix_len < 0 or extend_len < 0:
            raise ValueError("Prefill prefix and extend lengths must be nonnegative")
        if prefix_len + extend_len > self.capacity:
            raise ValueError("One prefill sequence exceeds the reconstruction window")
        return self.used + prefix_len + extend_len <= self.capacity

    def reserve(self, prefix_len, extend_len):
        if not self.can_reserve(prefix_len, extend_len):
            raise RuntimeError("Prefill sequence workspace reservation did not fit")
        self.used += prefix_len + extend_len


def prefill_request_groups(request_lengths, capacity):
    """Merge complete contiguous requests while preserving their native order."""
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Prefill window capacity must be a positive integer")
    offset = start = count = 0
    for length in request_lengths:
        if type(length) is not int or length < 0:
            raise ValueError("Prefill request lengths must be nonnegative CPU integers")
        if length > capacity:
            if count:
                yield start, count
            for at in range(offset, offset + length, capacity):
                yield at, min(capacity, offset + length - at)
            offset += length
            start, count = offset, 0
        else:
            if count + length > capacity:
                yield start, count
                start, count = offset, 0
            count += length
            offset += length
    if count:
        yield start, count


class NativePrefillReadPlans:
    """Keep one dense history plan and the current shared-Index group only.

    A forward reset retires all metadata. Source tensors remain referenced so
    Python cannot reuse their identities while a cached plan is live. Native
    writers preserve the current-row ordering across layers of that forward.
    """

    def __init__(self, cache, layer_groups, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Prefill window capacity must be a positive integer")
        self.cache, self.layer_groups, self.capacity = cache, layer_groups, capacity
        self.reset()

    def reset(self):
        self.dense = self.sparse = None

    def _validate(self, rows, request_lengths):
        if rows.numel() > self.capacity:
            raise ValueError("Prefill full histories exceed the fixed buffer capacity")
        lengths = (rows.numel(),) if request_lengths is None else tuple(request_lengths)
        if any(type(n) is not int or n < 0 for n in lengths):
            raise ValueError("Prefill request lengths must be nonnegative CPU integers")
        if sum(lengths) != rows.numel():
            raise ValueError(
                "Prefill request lengths disagree with native flattened rows"
            )
        return lengths

    def read(self, layer, rows, native_reader, *, request_lengths=None):
        """Gather full admitted histories and return the native result directly."""
        import torch

        lengths = self._validate(rows, request_lengths)
        key = (id(rows), lengths)
        if self.dense is None or self.dense[0] != key:
            native = rows.reshape(-1).to(torch.int32).contiguous()
            plans = []
            for offset, count in prefill_request_groups(lengths, self.capacity):
                part = native.narrow(0, offset, count)
                plans.append((part, self.cache.prepare_native_prefill_read_plan(part)))
            self.dense = (key, rows, plans)
        outputs = [
            self.cache.read_native_prefill_records(
                layer, part, native_reader, output=None, read_plan=plan
            )
            for part, plan in self.dense[2]
        ]
        if not outputs:
            raise ValueError("Native prefill reads require nonempty histories")
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    def read_selected(
        self, layer, rows, indices, current_rows, native_reader, *, request_lengths
    ):
        """Gather a small native TopK union once per shared selection group."""
        import torch

        self._validate(rows, request_lengths)
        if indices.numel() > rows.numel():
            return self.read(
                layer, rows, native_reader, request_lengths=request_lengths
            ), indices
        group = self.layer_groups.group(layer)
        key = (group, id(rows))
        if self.sparse is None or self.sparse[0] != key:
            native = torch.where(indices >= 0, rows[indices.clamp_min(0)], -1).int()
            lengths = torch.full(
                (indices.shape[0],),
                indices.shape[1],
                dtype=torch.int32,
                device=rows.device,
            )
            needed, remap, fresh = compact_topk_rows(native, lengths, current_rows)
            plan = CompactPrefillRecordReadPlan(
                (
                    CompactPrefillRecordWindow(
                        needed,
                        torch.arange(
                            needed.numel(), device=rows.device, dtype=torch.int32
                        ),
                        fresh,
                    ),
                )
            )
            self.sparse = (key, rows, needed, remap, plan)
        _, _, needed, remap, plan = self.sparse
        if not needed.numel():
            return self.read(
                layer, rows, native_reader, request_lengths=request_lengths
            ), indices
        output = self.cache.read_native_prefill_records(
            layer, needed, native_reader, output=None, read_plan=plan
        )
        return output, remap
