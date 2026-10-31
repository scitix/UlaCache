"""Check real backing storage and parallel expiry isolation without CUDA."""
import unittest
import importlib.util

from ulacache.storage.capacity import shared_host_workspace_bytes

@unittest.skipIf(importlib.util.find_spec("torch") is None, "The optional CUDA backend requires Torch")
class SharedHostWorkspaceTests(unittest.TestCase):
    def test_serial_transfer_arrays_and_parallel_expiry_have_distinct_lifetimes(self):
        import torch
        from ulacache.cache.host import create_host_tier_workspaces

        work = create_host_tier_workspaces(97, 3, 64, "cpu")
        for name in ["new_rows", "promote_rows", "plan", "candidate_slots"]:
            self.assertEqual(len({getattr(w, name).data_ptr() for w in work}), 1)
        for name in ["counters", "selection_prefix", "leased_slots"]:
            self.assertEqual(len({getattr(w, name).data_ptr() for w in work}), 3)
        for group, w in enumerate(work):
            w.leased_slots.fill_(group + 1)
            w.counters.fill_(group + 11)
            self.assertEqual(w.leased_slots.numel(), 64)
        # All group expiry writes survive, while one completed transfer plan
        # can be replaced by the next group's plan without another allocation.
        work[0].plan.fill_(7)
        self.assertTrue(torch.all(work[1].plan == 7))
        work[1].plan.fill_(8)
        self.assertTrue(torch.all(work[2].plan == 8))
        for group, w in enumerate(work):
            self.assertTrue(torch.all(w.leased_slots == group + 1))
            self.assertTrue(torch.all(w.counters == group + 11))

    def test_allocation_sum_matches_capacity_accounting_below_and_above_gpu_slots(self):
        from ulacache.cache.host import create_host_tier_workspaces

        for capacity, groups, gpu_slots in [(31, 1, 64), (97, 3, 64), (1048704, 21, 262144)]:
            work = create_host_tier_workspaces(capacity, groups, gpu_slots, "cpu")
            allocations = {
                tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                for w in work for tensor in w if tensor.numel()
            }
            self.assertEqual(sum(allocations.values()), shared_host_workspace_bytes(capacity, groups, gpu_slots))
        self.assertEqual(shared_host_workspace_bytes(1048704, 21, 262144), 55579800)


if __name__ == "__main__":
    unittest.main()
