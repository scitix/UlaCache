"""CPU-only stream/event model for the shared UlaCache/HiSparse pipeline."""
from contextlib import contextmanager
import unittest

from ulacache.transfer.prefill_pipeline import LayerPrefillPipeline, PrefillSequenceBudget


class Buffer:
    def __init__(self, pointer, size=64 * 656):
        self.pointer = pointer
        self.size = size
        self.nbytes = size

    def untyped_storage(self):
        return self

    def data_ptr(self):
        return self.pointer


class StorageBuffer(Buffer):
    def untyped_storage(self):
        outer = self

        class Storage:
            def data_ptr(self):
                return outer.pointer

            def nbytes(self):
                return outer.size

        return Storage()


class Event:
    def __init__(self, number):
        self.number = number
        self.generation = 0


class Runtime:
    def __init__(self):
        self.current = "main"
        self.events = []
        self.log = []
        self.capture = False

    def make_stream(self):
        return "prefetch"

    def make_event(self):
        event = Event(len(self.events))
        self.events.append(event)
        return event

    def current_stream(self):
        return self.current

    @contextmanager
    def stream_scope(self, stream):
        before = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = before

    def wait_stream(self, destination, source):
        self.log.append(("wait_stream", destination, source))

    def wait_event(self, stream, event):
        if not event.generation:
            raise AssertionError("Unrecorded CUDA event was waited")
        self.log.append(("wait_event", stream, event.number, event.generation))

    def record_event(self, event, stream):
        event.generation += 1
        self.log.append(("record_event", stream, event.number, event.generation))

    def record_tensor(self, tensor, stream):
        self.log.append(("record_tensor", tensor, stream))

    def synchronize(self, stream):
        self.log.append(("synchronize", stream))

    def is_capturing(self):
        return self.capture


class PipelineTest(unittest.TestCase):
    def pipeline(self):
        runtime = Runtime()
        windows = (StorageBuffer(100), StorageBuffer(200))
        # Distinct decode hot state is deliberately outside both callbacks.
        hot = {"directory": [10, -2, -3], "LRU": [6, 7], "pins": [0, 1]}
        def history(layer, plan, window):
            self.assertEqual(runtime.current, "prefetch")
            self.assertIs(plan, self.plan)
            runtime.log.append(("history", layer, window.pointer))
        def current(layer, plan, window):
            self.assertEqual(runtime.current, "main")
            runtime.log.append(("current", layer, window.pointer))
        pipeline = LayerPrefillPipeline(windows, gather_history=history,
            install_current=current, runtime=runtime, capacity_records=64)
        self.plan = object()
        return pipeline, runtime, hot

    def test_overlap_retirement_and_once_only_current_for_78_layers(self):
        pipeline, runtime, hot = self.pipeline()
        snapshot = repr(hot)
        pipeline.begin(self.plan, layer_count=78, record_count=64,
                       keepalive_tensors=("rows", "fresh"))
        pipeline.prepare(0)
        self.assertEqual([x for x in runtime.log if x[0] == "history"],
                         [("history", 0, 100), ("history", 1, 200)])
        # Next-layer history is scheduled before the current layer computes.
        for layer in range(78):
            if layer:
                pipeline.prepare(layer)
            window = pipeline.read(layer)
            self.assertIs(window, pipeline.read(layer))
            runtime.log.append(("native_attention_and_canonical_writeback", layer))
            pipeline.after_attention(layer)
        pipeline.finish()
        self.assertEqual(repr(hot), snapshot)
        self.assertEqual(len([x for x in runtime.log if x[0] == "history"]), 78)
        self.assertEqual(len([x for x in runtime.log if x[0] == "current"]), 78)
        self.assertFalse(any(x[0] == "synchronize" for x in runtime.log))
        self.assertEqual(runtime.log[-1], ("wait_stream", "main", "prefetch"))
        for layer in range(2, 78):
            at = runtime.log.index(("history", layer, 100 if layer % 2 == 0 else 200))
            retired = pipeline.retired[layer % 2]
            self.assertEqual(runtime.log[at - 3][0], "wait_event")
            wait = runtime.log[at - 3]
            self.assertEqual(wait[1:3], ("prefetch", retired.number))
            prior = runtime.log.index(("native_attention_and_canonical_writeback", layer - 2))
            self.assertLess(prior, at)
        self.assertIsNone(pipeline.plan)

    def test_reuse_before_retirement_and_partial_finish_are_rejected(self):
        pipeline, runtime, _ = self.pipeline()
        pipeline.begin(self.plan, layer_count=3, record_count=32)
        pipeline.prepare(0)
        with self.assertRaises(RuntimeError):
            pipeline.prepare(1)
        with self.assertRaises(RuntimeError):
            pipeline._prefetch(2)
        with self.assertRaises(RuntimeError):
            pipeline.finish()
        with self.assertRaises(RuntimeError):
            pipeline.after_attention(0)
        pipeline.abort()
        self.assertEqual(runtime.log[-1], ("synchronize", "prefetch"))

    def test_abort_then_new_forward_reuses_only_after_stream_join(self):
        pipeline, runtime, _ = self.pipeline()
        pipeline.begin(self.plan, layer_count=3, record_count=32)
        pipeline.prepare(0)
        pipeline.read(0)
        pipeline.abort()
        boundary = len(runtime.log)
        self.plan = object()
        pipeline.begin(self.plan, layer_count=1, record_count=16)
        pipeline.prepare(0)
        self.assertEqual(runtime.log[boundary], ("wait_stream", "prefetch", "main"))
        pipeline.read(0)
        pipeline.after_attention(0)
        pipeline.finish()

    def test_budget_uses_full_histories_not_only_fresh_chunk(self):
        budget = PrefillSequenceBudget(1_048_576)
        budget.reserve(512_000, 512)
        budget.reserve(512_000, 512)
        self.assertFalse(budget.can_reserve(64_000, 512))
        self.assertTrue(budget.can_reserve(0, 512))
        with self.assertRaises(ValueError):
            budget.can_reserve(1_048_576, 1)
        with self.assertRaises(ValueError):
            budget.reserve(-1, 1)

    def test_native_cold_MHA_unread_layers_need_no_forced_install(self):
        pipeline, runtime, _ = self.pipeline()
        pipeline.begin(self.plan, layer_count=5, record_count=32)
        pipeline.skip_unused(0)  # Native layer0 completed without a cache read.
        pipeline.prepare(1)
        pipeline.read(1)
        with self.assertRaises(RuntimeError):
            pipeline.skip_unused(1)
        pipeline.after_attention(1)
        # Layer2 was prefetched, but native attention did not consume it.
        before = len(runtime.log)
        pipeline.skip_unused(2)
        self.assertEqual(runtime.log[before][0], "record_event")
        pipeline.prepare(3)
        pipeline.read(3)
        pipeline.after_attention(3)
        pipeline.skip_unused(4)
        pipeline.finish()
        self.assertEqual([v[1] for v in runtime.log if v[0] == "current"], [1, 3])
        self.assertFalse(any(v[:2] == ("wait_event", "main") and v[2] == 0
                             for v in runtime.log))

    def test_capacity_capture_and_independent_backing(self):
        pipeline, runtime, _ = self.pipeline()
        with self.assertRaises(ValueError):
            pipeline.begin(self.plan, layer_count=78, record_count=65)
        runtime.capture = True
        with self.assertRaises(RuntimeError):
            pipeline.begin(self.plan, layer_count=78, record_count=64)
        with self.assertRaises(ValueError):
            LayerPrefillPipeline((StorageBuffer(1), StorageBuffer(1)),
                gather_history=lambda *a: None, install_current=lambda *a: None,
                runtime=runtime, capacity_records=64)
        inventory = pipeline.memory_inventory()
        self.assertEqual(inventory["encoded_window_bytes"], 2 * 64 * 656)
        self.assertTrue(inventory["excludes_native_dequant_output_and_backend_scratch"])


if __name__ == "__main__":
    unittest.main()
