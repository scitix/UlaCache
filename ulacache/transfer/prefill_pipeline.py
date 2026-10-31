"""Two independent encoded-KV windows with the MR77 layer event ordering.

The backend owns record addressing, native Index/remaps, quantization and
canonical writes. This module only schedules its read-only history callback
and its current-row installation callback. It never allocates logical KV
slots, promotes records, changes a cache directory, or updates decode LRU.

MR77 uses two windows only on its P-only path. Using the same two-window
protocol on aggregated UlaCache and HiSparse is an explicit serving choice;
it must not enable MR77's P-only flag, which removes decode residency.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

DEFAULT_PREFILL_RECORDS = 1_048_576


class PrefillSequenceBudget:
    """Native batch admission counts complete prefix+extend histories."""

    def __init__(self, capacity: int = DEFAULT_PREFILL_RECORDS):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Prefill record capacity must be a positive integer")
        self.capacity = capacity
        self.used = 0

    def can_reserve(self, prefix_len: int, extend_len: int) -> bool:
        if any(type(n) is not int or n < 0 for n in (prefix_len, extend_len)):
            raise ValueError(
                "Native prefix and extend lengths must be nonnegative integers"
            )
        count = prefix_len + extend_len
        if count > self.capacity:
            raise ValueError("One full sequence exceeds the prefill record window")
        return self.used + count <= self.capacity

    def reserve(self, prefix_len: int, extend_len: int) -> None:
        if not self.can_reserve(prefix_len, extend_len):
            raise ValueError(
                "Full prefill histories exceed the remaining batch workspace"
            )
        self.used += prefix_len + extend_len


def allocate_native_prefill_windows(
    device, *, capacity_records=DEFAULT_PREFILL_RECORDS, row_bytes=656
):
    """Allocate exactly two encoded payloads; backend views allocate no storage.

    GLM's656-byte encoded record contains MLA values, scales and RoPE together.
    A backend may view the bytes as native pages without copying. Dequantized
    output, current-token staging and other scratch are separate allocations
    and must also appear in its normal startup/peak-memory inventory.
    """
    if (
        type(capacity_records) is not int
        or capacity_records < 1
        or type(row_bytes) is not int
        or row_bytes < 1
    ):
        raise ValueError("Prefill window dimensions must be positive integers")
    import torch

    return tuple(
        torch.empty((capacity_records, 1, row_bytes), dtype=torch.uint8, device=device)
        for _ in range(2)
    )


class CudaPrefillRuntime:
    """Ordinary CUDA stream/events; constructed during normal model startup."""

    def __init__(self, device):
        import torch

        self.torch = torch
        self.device = device

    def make_stream(self):
        return self.torch.cuda.Stream(device=self.device)

    def make_event(self):
        return self.torch.cuda.Event()

    def current_stream(self):
        return self.torch.cuda.current_stream(self.device)

    @contextmanager
    def stream_scope(self, stream):
        with self.torch.cuda.stream(stream):
            yield

    def wait_stream(self, destination, source):
        destination.wait_stream(source)

    def wait_event(self, stream, event):
        event.wait(stream)

    def record_event(self, event, stream):
        event.record(stream)

    def record_tensor(self, tensor, stream):
        tensor.record_stream(stream)

    def synchronize(self, stream):
        stream.synchronize()

    def is_capturing(self):
        return self.torch.cuda.is_current_stream_capturing()


class LayerPrefillPipeline:
    """Shared scheduling for UlaCache/HiSparse backend adapters.

    ``gather_history(layer, plan, window)`` runs on the prefetch stream. It
    reads stable encoded records into the separate window and skips every
    current/fresh row. It must not promote, evict, or mutate decode state.
    ``install_current(layer, plan, window)`` runs once per layer on the caller's
    stream, after history is ready and after that layer's native writer has
    produced fresh KV. Both callbacks own their native byte-layout/remap ABI.

    Caller order is begin -> prepare(layer) -> native writer -> read(layer)
    -> native attention -> required canonical-current writeback ->
    after_attention(layer). Normal finish joins prefetch before dropping the
    plan. Exceptional abort synchronizes its own stream before resetting.
    """

    def __init__(
        self,
        windows,
        *,
        gather_history: Callable[[int, Any, Any], None],
        install_current: Callable[[int, Any, Any], None],
        runtime=None,
        capacity_records=DEFAULT_PREFILL_RECORDS,
        row_bytes=656,
    ):
        windows = tuple(windows)
        if len(windows) != 2 or windows[0] is windows[1]:
            raise ValueError("Two independently owned prefill windows are required")
        if (
            type(capacity_records) is not int
            or capacity_records < 1
            or type(row_bytes) is not int
            or row_bytes < 1
        ):
            raise ValueError("Prefill window dimensions must be positive integers")
        expected = capacity_records * row_bytes
        for window in windows:
            if window.nbytes != expected:
                raise ValueError(
                    "Each encoded window must match its declared record budget"
                )
        # Different tensor views of the same allocation are not double buffers.
        storages = [window.untyped_storage() for window in windows]
        if storages[0].data_ptr() == storages[1].data_ptr():
            raise ValueError("Prefill windows must not alias one backing allocation")
        if not callable(gather_history) or not callable(install_current):
            raise TypeError("Both backend prefill callbacks are required")
        self.windows = windows
        self.capacity_records = capacity_records
        self.row_bytes = row_bytes
        self.gather_history = gather_history
        self.install_current = install_current
        self.runtime = runtime or CudaPrefillRuntime(windows[0].device)
        self.stream = self.runtime.make_stream()
        self.ready = tuple(self.runtime.make_event() for _ in windows)
        self.retired = tuple(self.runtime.make_event() for _ in windows)
        self._reset()

    def _reset(self):
        self.plan = None
        self.keepalive_tensors = ()
        self.layer_count = 0
        self.record_count = 0
        self.scheduled = set()
        self.installed = set()
        self.attended = set()
        self.slot_owner = [None, None]
        self.active_layer = None

    def begin(self, plan, *, layer_count: int, record_count: int, keepalive_tensors=()):
        if self.plan is not None or self.active_layer is not None:
            raise RuntimeError("Finish or abort the previous prefill forward first")
        if self.runtime.is_capturing():
            raise RuntimeError("Prefill reconstruction uses the native eager path")
        if plan is None or type(layer_count) is not int or layer_count < 1:
            raise ValueError("A backend plan and positive layer count are required")
        if (
            type(record_count) is not int
            or not 0 < record_count <= self.capacity_records
        ):
            raise ValueError(
                "The admitted full-history record count exceeds one window"
            )
        self.plan = plan
        self.layer_count = layer_count
        self.record_count = record_count
        self.keepalive_tensors = tuple(keepalive_tensors)

    def _layer(self, layer):
        if (
            self.plan is None
            or type(layer) is not int
            or not 0 <= layer < self.layer_count
        ):
            raise ValueError(
                "A valid local layer in an active prefill forward is required"
            )

    def _prefetch(self, layer):
        if layer in self.scheduled:
            return
        slot = layer % 2
        previous = self.slot_owner[slot]
        if previous is not None and previous not in self.attended:
            raise RuntimeError(
                "Prefill attempted to overwrite a layer before attention retirement"
            )
        main = self.runtime.current_stream()
        self.runtime.wait_stream(self.stream, main)
        with self.runtime.stream_scope(self.stream):
            if previous is not None:
                self.runtime.wait_event(self.stream, self.retired[slot])
            for tensor in self.keepalive_tensors:
                self.runtime.record_tensor(tensor, self.stream)
            self.gather_history(layer, self.plan, self.windows[slot])
            self.runtime.record_event(self.ready[slot], self.stream)
        self.scheduled.add(layer)
        self.slot_owner[slot] = layer

    def prepare(self, layer: int):
        self._layer(layer)
        if self.active_layer is not None and self.active_layer != layer:
            raise RuntimeError(
                "Retire the current attention before preparing another layer"
            )
        if layer != len(self.attended):
            raise RuntimeError("Native local prefill layers must be consumed in order")
        self.active_layer = layer
        self._prefetch(layer)
        if layer + 1 < self.layer_count:
            self._prefetch(layer + 1)
        return self.windows[layer % 2]

    def read(self, layer: int):
        self._layer(layer)
        if self.active_layer != layer or self.slot_owner[layer % 2] != layer:
            raise RuntimeError("Prepare the current layer before its native read")
        if layer not in self.installed:
            self.runtime.wait_event(
                self.runtime.current_stream(), self.ready[layer % 2]
            )
            self.install_current(layer, self.plan, self.windows[layer % 2])
            self.installed.add(layer)
        return self.windows[layer % 2]

    def after_attention(self, layer: int):
        self._layer(layer)
        if (
            self.active_layer != layer
            or layer not in self.installed
            or layer in self.attended
        ):
            raise RuntimeError(
                "Retire each successfully consumed prefill layer exactly once"
            )
        self.runtime.record_event(
            self.retired[layer % 2], self.runtime.current_stream()
        )
        self.attended.add(layer)
        self.active_layer = None

    def skip_unused(self, layer: int):
        """Retire a native layer which completed without reading this window.

        Call only after its real attention and canonical writes have completed
        on the caller stream. Native cold MHA may never need a prefix reader.
        A prefetched but unused window needs no main-stream wait: the prefetch
        stream's FIFO orders its old copy before any later overwrite, while
        the retired event preserves the native completion boundary.
        """
        self._layer(layer)
        if (
            layer != len(self.attended)
            or self.active_layer not in (None, layer)
            or layer in self.installed
        ):
            raise RuntimeError(
                "Only the next completed and unread native layer can be skipped"
            )
        if layer in self.scheduled:
            self.runtime.record_event(
                self.retired[layer % 2], self.runtime.current_stream()
            )
        self.attended.add(layer)
        self.active_layer = None

    def finish(self):
        if self.plan is None:
            return
        if self.active_layer is not None or len(self.attended) != self.layer_count:
            raise RuntimeError(
                "Unconsumed prefill layers require abort, not successful finish"
            )
        self.runtime.wait_stream(self.runtime.current_stream(), self.stream)
        self._reset()

    def abort(self):
        self.runtime.synchronize(self.stream)
        self._reset()

    def close(self):
        self.abort()

    def memory_inventory(self):
        """Encoded-window backing only; backend must add its other workspaces."""
        return dict(
            capacity_records_per_window=self.capacity_records,
            record_bytes=self.row_bytes,
            window_count=2,
            encoded_window_bytes=sum(
                w.untyped_storage().nbytes() for w in self.windows
            ),
            includes_MLA_scales_and_RoPE=True,
            excludes_native_dequant_output_and_backend_scratch=True,
        )
