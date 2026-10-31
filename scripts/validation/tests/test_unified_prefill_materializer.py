"""CPU metadata/callback contracts; these do not qualify CUDA payload kernels."""
import sys
import types
import unittest
from unittest import mock

import torch

from ulacache.cache.prefill.reader import (
    UnifiedPrefillMaterializer, prepare_read_plan,
)


class FullHistoryPlanTests(unittest.TestCase):
    def test_full_history_duplicate_padding_and_fresh_row_zero(self):
        full = torch.tensor([[7, 2, 7, 0, -1, 11], [2, 5, 11, 13, 13, 5]], dtype=torch.int64)
        current = torch.tensor([11, 5], dtype=torch.int64)
        p = prepare_read_plan(full, current)
        self.assertEqual(p.record_ids.tolist(), [2, 5, 7, 11, 13])
        self.assertEqual(p.fresh_positions.tolist(), [-1, 1, -1, 0, -1])
        self.assertEqual(p.input_record_indices.tolist(), [[2, 0, 2, -1, -1, 3], [0, 1, 3, 4, 4, 1]])
        self.assertEqual(p.map_native_rows(torch.tensor([[2, 0, -1, 13]])).tolist(), [[0, -1, -1, 4]])
        self.assertEqual(p.record_count, 5)
        self.assertIn(13, p.record_ids.tolist())  # Unselected history remains part of the plan.

    def test_prefix_only_table_includes_current_extend_rows(self):
        p = prepare_read_plan(torch.tensor([9, 3], dtype=torch.int32),
                              torch.tensor([7, 5], dtype=torch.int64))
        self.assertEqual(p.record_ids.tolist(), [3, 5, 7, 9])
        self.assertEqual(p.input_record_indices.tolist(), [3, 0])
        self.assertEqual(p.fresh_positions.tolist(), [-1, 1, 0, -1])

    def test_first_chunk_and_idle_do_not_invent_history(self):
        p = prepare_read_plan(None, torch.tensor([5, 2], dtype=torch.int64))
        self.assertEqual(p.record_ids.tolist(), [2, 5])
        self.assertEqual(p.fresh_positions.tolist(), [1, 0])
        idle = prepare_read_plan(None, torch.empty(0, dtype=torch.int64))
        self.assertEqual(idle.record_count, 0)

    def test_capacity_is_admission_bound_not_dynamic_pool_growth(self):
        for bound in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'positive record bound'):
                prepare_read_plan(torch.tensor([2]), torch.empty(0, dtype=torch.int32),
                                  capacity_records=bound)
        plan = prepare_read_plan(torch.arange(1, 65), torch.empty(0, dtype=torch.int32),
                                 capacity_records=64)
        self.assertEqual(plan.record_count, 64)
        with self.assertRaisesRegex(ValueError, 'admission workspace bound'):
            prepare_read_plan(torch.arange(1, 66), torch.empty(0, dtype=torch.int32),
                              capacity_records=64)
        with self.assertRaisesRegex(ValueError, 'admission workspace bound'):
            prepare_read_plan(torch.arange(1, 1048578, dtype=torch.int32), torch.empty(0, dtype=torch.int32))


class CallbackRoutingTests(unittest.TestCase):
    def test_history_uses_actual_group_and_layer_without_residency_planning(self):
        calls = []
        e = types.SimpleNamespace(
            groups=types.SimpleNamespace(group=lambda layer: layer // 2),
            directory=torch.zeros((2, 16), dtype=torch.int64),
            gpu=torch.zeros((4, 4, 656), dtype=torch.uint8),
            host=types.SimpleNamespace(ptrs=torch.tensor([1234], dtype=torch.int64)),
            host_slots=12, row_bytes=656, prefill_exchange_launch=(64, 256),
            _drain_transfers=lambda: calls.append('drain'),
        )
        directory_before = e.directory.clone()
        gpu_before = e.gpu.clone()
        p = prepare_read_plan(torch.tensor([2, 3]), torch.tensor([3]))
        materializer = UnifiedPrefillMaterializer(e)
        materializer.begin()
        fake = types.ModuleType('ulacache.cache.kernels.read_only_prefill')
        fake.gather_history = lambda *a, **kw: calls.append(('history', a, kw))
        fake.install_current = lambda *a, **kw: calls.append(('current', a, kw))
        window = torch.empty((8, 1, 656), dtype=torch.uint8)
        with mock.patch.dict(sys.modules, {'ulacache.cache.kernels.read_only_prefill': fake}):
            materializer.gather_history(2, p, window)
            with self.assertRaisesRegex(RuntimeError, 'encoded/written'):
                materializer.install_current(2, p, window)
            current = torch.tensor([3], dtype=torch.int64)
            n, r = torch.zeros((1, 528), dtype=torch.uint8), torch.zeros((1, 128), dtype=torch.uint8)
            materializer.record_current(1, current, n, r)
            materializer.record_current(2, current, n, r)
            self.assertEqual(set(materializer._current), {2})  # One borrowed layer, not78.
            materializer.install_current(2, p, window)
        history = next(v for v in calls if isinstance(v, tuple) and v[0] == 'history')
        self.assertEqual(history[1][2].data_ptr(), e.directory[1].data_ptr())
        self.assertEqual(history[1][3].data_ptr(), e.gpu[2].data_ptr())
        self.assertEqual(history[2]['launch'], (64, 256))
        self.assertEqual(history[2]['layer'], 2)
        self.assertTrue(torch.equal(e.directory, directory_before))
        self.assertTrue(torch.equal(e.gpu, gpu_before))
        self.assertFalse(materializer._current)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
