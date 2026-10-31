import sys
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from ulacache import (CacheLayout, InterleavedRecordLayout, NativePageLayout,
                     LayerGroupPlan, RecordSpec, history_capacity)
from ulacache.storage.capacity import flowkv_cpu_pool_geometry
from ulacache.storage.workspace import ResidentQueryLayout

class CoreContracts(unittest.TestCase):
    def test_import_does_not_load_framework_or_device_runtime(self):
        from test_module_boundaries import DEVICE_AND_FRAMEWORK, isolated

        process = isolated('''
from ulacache import (CacheLayout, InterleavedRecordLayout, NativePageLayout,
                     LayerGroupPlan, RecordSpec, history_capacity)
from ulacache.storage.capacity import flowkv_cpu_pool_geometry
from ulacache.storage.workspace import ResidentQueryLayout
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
''', DEVICE_AND_FRAMEWORK)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_position_interleave_preserves_rank_intervals(self):
        for peers in (1, 2, 4, 8):
            layout = InterleavedRecordLayout(peers, 1024, NativePageLayout(64, (576,), 1))
            for row in range(1, 256):
                gid = layout.native_to_global(row)
                self.assertEqual(layout.locate(gid), (row % peers, row // peers))
                self.assertEqual(layout.global_id(*layout.locate(gid)), gid)

    def test_cpu_ratio_reconstructs_page_aligned_capacity(self):
        for ratio in (0.125, 1, 2, 3.5):
            for logical_pages in range(1, 257):
                x, y = flowkv_cpu_pool_geometry(logical_pages * 64,
                                               cpu_slots=0, cpu_ratio=ratio)
                self.assertEqual(x % 64, 0)
                self.assertEqual(y % 64, 0)
                self.assertLessEqual(x + y, logical_pages * 64)
                self.assertEqual(y // 64, int((x // 64) * ratio))
                if x:
                    self.assertGreater(x + 64 + int((x // 64 + 1) * ratio) * 64,
                                       logical_pages * 64)

    def test_capacity_and_groups_do_not_depend_on_fixed_batch(self):
        for bs in (1, 2, 4, 8, 16, 32, 64, 128):
            groups = LayerGroupPlan.from_layer_anchors((0, 0, 2, 2))
            layout = CacheLayout(layers=4, groups=2, max_batch=bs, pool_rows=4096,
                row_bytes=584, disaggregation_mode="null", topk=512,
                max_query_tokens=bs, history_slots=history_capacity(bs, 512, 1.125, 64),
                page_size=64, native_segment_bytes=(576, 8), layer_anchors=(0, 0, 2, 2),
                compression_ratio=4, page_storage_alignment=576, prefill_window_rows=1024)
            self.assertEqual(layout.layer_groups, groups)
            self.assertEqual(layout.prefill_buffer_bytes, 16 * 37440)
            self.assertGreaterEqual(layout.history_slots, bs * 512)
            self.assertEqual(layout.native_page.row_bytes, 584)

    def test_resident_kv_capacity_does_not_include_native_metadata_padding(self):
        layout = ResidentQueryLayout(layers=78, groups=78, max_batch=1, pool_rows=4096,
            row_bytes=656, disaggregation_mode="decode", topk=2048,
            max_query_tokens=8, max_resident_queries=5,
            history_slots=history_capacity(5, 2048, 1.125, 64),
            page_size=64, native_segment_bytes=(512, 16, 128),
            layer_anchors=tuple(range(78)))
        self.assertEqual(layout.resident_query_capacity, 5)
        self.assertEqual(layout.history_slots, 11520)
        self.assertEqual(layout.rows_per_layer, 11584)
        with self.assertRaisesRegex(ValueError, "largest TopK"):
            replace(layout, history_slots=5 * 2048 - 64)
        with self.assertRaisesRegex(ValueError, "metadata bound"):
            replace(layout, max_resident_queries=9)
        with self.assertRaisesRegex(ValueError, "largest TopK"):
            replace(layout, max_resident_queries=0)

if __name__ == "__main__":
    unittest.main()
