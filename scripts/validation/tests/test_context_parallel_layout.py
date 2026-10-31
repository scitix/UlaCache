"""Allocator conservation across ragged lengths, chunks and owner degrees."""

import unittest

from ulacache import PositionShardLayout


class PositionOwnershipTests(unittest.TestCase):
    def test_owners_partition_each_superpage_and_preserve_page64(self):
        for degree in (2, 4, 8):
            shards = [PositionShardLayout(rank, degree, 128, 256) for rank in range(degree)]
            first = shards[0].allocation_page_size
            seen = set()
            for slot in range(first, 3 * first):
                owners = [(rank, shard.local_slot(slot)) for rank, shard in enumerate(shards)
                          if shard.local_slot(slot) >= 0]
                self.assertEqual(owners, [(slot % degree, slot // degree)])
                self.assertNotIn(owners[0], seen)
                seen.add(owners[0])
            self.assertEqual(shards[0].usable_records, degree * (128 + 256 - 64))

    def test_ragged_chunks_and_verify_frontiers_conserve_all_tokens(self):
        for degree in (2, 4, 8):
            shards = [PositionShardLayout(rank, degree, 128, 256) for rank in range(degree)]
            for start in (0, 1, 63, 64, 65, 127, 257):
                for length in (0, 1, 2, 3, 7, 63, 64, 65, 129):
                    lengths = [shard.local_length(length, start=start) for shard in shards]
                    self.assertEqual(sum(lengths), length)
                    self.assertLessEqual(max(lengths) - min(lengths), 1)
                    for rank, count in enumerate(lengths):
                        self.assertEqual(count, sum(pos % degree == rank for pos in range(start, start + length)))

    def test_invalid_geometry_cannot_silently_change_ownership(self):
        for args in [(0, 3, 128, 256), (8, 8, 128, 256), (0, 8, 65, 256),
                     (0, 8, 128, 255), (0, True, 128, 256), (0, 8, 2**30, 0)]:
            with self.assertRaises(ValueError):
                PositionShardLayout(*args)
        shard = PositionShardLayout(0, 8, 128, 256)
        for slot in (-1, shard.global_records, True):
            with self.assertRaises(ValueError):
                shard.owner(slot)


if __name__ == '__main__':
    unittest.main()
