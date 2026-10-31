"""Stdlib protocol tests; no Torch import or device execution."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

import ulacache.cache.request_pages as m


class Publisher:
    def __init__(self):
        self.rows, self.fail = {}, False

    def publish(self, owner, updates):
        if self.fail:
            raise RuntimeError("controlled mirror publication failure")
        self.rows.setdefault(owner, {}).update(updates)


class Completion:
    def __init__(self, ticket, complete=True):
        self.transaction_id, self.complete = ticket, complete

    def assert_complete(self):
        if not self.complete:
            raise RuntimeError("both complete record planes are still pending")


class PayloadExecutor:
    """Explicit test executor models complete MLA tokens + packed Index pages."""
    def __init__(self, pool):
        self.pool = pool
        self.payload = {plane: {"GPU": {}, "CPU": {}} for plane in ("mla", "index")}
        self.calls = 0

    def seed(self, logical):
        location = self.pool.location(logical)
        tier, physical = ("GPU", location) if location >= 0 else ("CPU", -location - 2)
        for plane in self.payload:
            # These identities represent complete byte planes, not fabricated KV data.
            self.payload[plane][tier][physical] = (plane, logical, tuple(range(64)))

    def submit(self, pool, tx):
        self.calls += 1
        for wave in tx.waves:
            gpu_ids = [p.gpu_page for p in wave]
            cpu_ids = [p.cpu_page for p in wave]
            assert len(set(gpu_ids)) == len(gpu_ids)
            assert len(set(cpu_ids)) == len(cpu_ids)
            for transfer in wave:
                for plane in self.payload.values():
                    gpu, cpu = plane["GPU"], plane["CPU"]
                    g, c = transfer.gpu_page, transfer.cpu_page
                    if transfer.kind == "onload":
                        gpu[g] = cpu[c]
                    elif transfer.kind == "offload":
                        cpu[c] = gpu[g]
                    else:
                        gpu[g], cpu[c] = cpu[c], gpu[g]
        return Completion(tx.transaction_id)

    def validate(self):
        for logical, location in enumerate(self.pool.map_row_snapshot()):
            if location == m.FREE:
                continue
            tier, physical = ("GPU", location) if location >= 0 else ("CPU", -location - 2)
            for plane_name, plane in self.payload.items():
                assert plane[tier][physical] == (plane_name, logical, tuple(range(64)))


def setup(gpu=3, cpu=3, *, reserved_gpu=(), owner=1):
    pool = m.FlowKVRequestPagePool(owner_count=4, owner_rank=owner,
                                  gpu_token_rows=gpu * 64, cpu_token_rows=cpu * 64,
                                  logical_owner_token_rows=(gpu + cpu) * 64,
                                  layer_num=78, row_bytes=576,
                                  index_anchor_layers=tuple(range(0, 78, 4)),
                                  reserved_gpu_pages=reserved_gpu)
    publisher = Publisher()
    executor = PayloadExecutor(pool)
    return pool, publisher, executor


def attach(pool, publisher, executor, logical, tier="GPU", reservation=None):
    physical = pool.attach_native_pages(logical, publisher=publisher, tier=tier,
                                        reservation=reservation)
    for page in logical:
        executor.seed(page)
    return physical


def validate_allocator(test, pool):
    row = pool.map_row_snapshot()
    gpu = [p for p in row if p >= 0]
    cpu = [-p - 2 for p in row if p < m.FREE]
    test.assertEqual(len(gpu), len(set(gpu)))
    test.assertEqual(len(cpu), len(set(cpu)))
    held = set().union(*pool._future.values()) if pool._future else set()
    test.assertFalse(set(gpu) & held)
    test.assertEqual(set(gpu) | held | pool._gpu.free | pool._gpu.reserved,
                     set(range(pool.gpu_page_count)))
    test.assertEqual(set(cpu) | pool._cpu.free | pool._cpu.reserved,
                     set(range(pool.cpu_page_count)))
    test.assertEqual(len(gpu) + len(held) + len(pool._gpu.free) + len(pool._gpu.reserved), pool.gpu_page_count)
    test.assertEqual(len(cpu) + len(pool._cpu.free) + len(pool._cpu.reserved), pool.cpu_page_count)


class TestFlowKVRequestPagePool(unittest.TestCase):
    def test_shape_map_and_physical_capacity(self):
        pool, pub, ex = setup(reserved_gpu=(0,))
        self.assertEqual(pool.allocation_page_size, 256)
        self.assertEqual(pool.gpu_usable_pages, 2)
        attach(pool, pub, ex, [0, 1])
        self.assertEqual(pool.map_row_snapshot()[:2], (1, 2))
        self.assertEqual(pool.plane_shapes()["index_gpu"], (20, 3, 8448))
        self.assertEqual(pub.rows[1][0], 1)
        validate_allocator(self, pool)

    def test_full_CPU_swap_preserves_both_planes(self):
        pool, pub, ex = setup()
        attach(pool, pub, ex, [0, 1, 2])
        attach(pool, pub, ex, [3, 4, 5], "CPU")
        before = pool.map_row_snapshot()
        tx = pool.reserve_request([3], future_gpu_pages=0, inactive_page_ids=[0])
        self.assertEqual(tx.waves[0][0].kind, "swap")
        completion = pool.ensure_resident(tx, ex)
        self.assertEqual(pool.map_row_snapshot(), before)
        reservation = pool.commit(tx, completion, pub)
        self.assertEqual(pool.location(3), 0)
        self.assertEqual(pool.location(0), -2)
        ex.validate()
        pool.release_reservation(reservation)
        validate_allocator(self, pool)

    def test_CPU_source_reuse_requires_second_wave(self):
        pool, pub, ex = setup()
        attach(pool, pub, ex, [0, 1])
        attach(pool, pub, ex, [2, 3, 4], "CPU")
        tx = pool.reserve_request([2], future_gpu_pages=1, inactive_page_ids=[0])
        self.assertEqual(tuple(w[0].kind for w in tx.waves), ("onload", "offload"))
        self.assertEqual(tx.waves[0][0].cpu_page, tx.waves[1][0].cpu_page)
        plans = tx.exchange_plan(plane="mla")
        self.assertEqual(len(plans[0][0]), 64)
        self.assertEqual(plans[0][1], tuple(range(128, 192)))
        self.assertEqual(tx.exchange_plan(plane="index")[0][1], (2,))
        reservation = pool.commit(tx, pool.ensure_resident(tx, ex), pub)
        self.assertEqual(pool.reservation_remaining_pages(reservation), 1)
        attach(pool, pub, ex, [5], reservation=reservation)
        self.assertEqual(pool.reservation_remaining_pages(reservation), 0)
        ex.validate()
        pool.release_reservation(reservation)
        validate_allocator(self, pool)

    def test_capacity_protected_prefix_and_CPU_shortfall(self):
        pool, pub, ex = setup()
        attach(pool, pub, ex, [0, 1, 2])
        attach(pool, pub, ex, [3, 4, 5], "CPU")
        verdict = pool.check_gpu_capacity([3], future_gpu_pages=1, inactive_page_ids=[0, 1])
        self.assertFalse(verdict.fits)
        self.assertIn("CPU", verdict.reason)
        verdict = pool.check_gpu_capacity([3], future_gpu_pages=0, inactive_page_ids=[0], protected_page_ids=[0])
        self.assertFalse(verdict.fits)
        self.assertIn("inactive GPU", verdict.reason)
        with self.assertRaises(ValueError):
            pool.reserve_request([3], future_gpu_pages=0, inactive_page_ids=[3])
        self.assertEqual(pool.gpu_free_pages, 0)
        validate_allocator(self, pool)

    def test_incomplete_or_wrong_completion_does_not_publish(self):
        pool, pub, ex = setup()
        attach(pool, pub, ex, [0], "CPU")
        before = pool.map_row_snapshot()
        tx = pool.reserve_request([0], future_gpu_pages=0)
        done = pool.ensure_resident(tx, ex)
        done.complete = False
        with self.assertRaises(RuntimeError):
            pool.commit(tx, done, pub)
        self.assertEqual(pool.map_row_snapshot(), before)
        with self.assertRaises(ValueError):
            pool.commit(tx, Completion(tx.transaction_id + 1), pub)
        done.complete = True
        reservation = pool.commit(tx, done, pub)
        pool.release_reservation(reservation)
        ex.validate()

    def test_publication_retry_cannot_replay_swap(self):
        pool, pub, ex = setup(gpu=1, cpu=1)
        attach(pool, pub, ex, [0])
        attach(pool, pub, ex, [1], "CPU")
        tx = pool.reserve_request([1], future_gpu_pages=0, inactive_page_ids=[0])
        done = pool.ensure_resident(tx, ex)
        pub.fail = True
        with self.assertRaises(RuntimeError):
            pool.commit(tx, done, pub)
        with self.assertRaises(RuntimeError):
            pool.ensure_resident(tx, ex)
        with self.assertRaises(RuntimeError):
            pool.cancel(tx)
        pub.fail = False
        reservation = pool.commit(tx, done, pub)
        self.assertEqual(ex.calls, 1)
        ex.validate()
        pool.release_reservation(reservation)
        validate_allocator(self, pool)

    def test_cancel_returns_only_unsubmitted_reservations(self):
        pool, pub, ex = setup()
        attach(pool, pub, ex, [0], "CPU")
        before = (pool.gpu_free_pages, pool.cpu_free_pages)
        tx = pool.reserve_request([0], future_gpu_pages=1)
        with self.assertRaises(NotImplementedError):
            pool.ensure_resident(tx, None)
        pool.cancel(tx)
        self.assertEqual((pool.gpu_free_pages, pool.cpu_free_pages), before)
        validate_allocator(self, pool)

    def test_native_retire_rebind_and_headroom_release(self):
        pool, pub, ex = setup()
        tx = pool.reserve_request([], future_gpu_pages=2)
        reservation = pool.commit(tx, pool.ensure_resident(tx, ex), pub)
        attach(pool, pub, ex, [0], reservation=reservation)
        self.assertEqual(pool.gpu_free_pages, 1)
        with self.assertRaises(RuntimeError):
            pool.retire_native_pages([0], readers_retired=False, publisher=pub)
        pool.retire_native_pages([0], readers_retired=True, publisher=pub)
        self.assertEqual(pool.location(0), m.FREE)
        attach(pool, pub, ex, [0], "CPU")
        pool.release_reservation(reservation)
        self.assertEqual(pool.gpu_free_pages, 3)
        with self.assertRaises(ValueError):
            pool.reservation_remaining_pages(reservation)
        ex.validate()
        validate_allocator(self, pool)

    def test_no_mirror_no_fake_success(self):
        pool, pub, ex = setup()
        with self.assertRaises(NotImplementedError):
            pool.attach_native_pages([0], publisher=None)
        tx = pool.reserve_request([], future_gpu_pages=0)
        done = pool.ensure_resident(tx, ex)
        with self.assertRaises(NotImplementedError):
            pool.commit(tx, done, None)
        reservation = pool.commit(tx, done, pub)
        pool.release_reservation(reservation)

    def test_random_continuous_native_lifetimes(self):
        rng = random.Random(8161)
        pool, pub, ex = setup(gpu=7, cpu=9)
        for logical in range(12):
            attach(pool, pub, ex, [logical], "GPU" if logical < 6 else "CPU")
        transitions = 0
        for _ in range(200):
            resident = [p for p, loc in enumerate(pool.map_row_snapshot()) if loc != m.FREE]
            required = rng.sample(resident, min(rng.randrange(5), len(resident)))
            protected = rng.sample(resident, min(2, len(resident)))
            candidates = rng.sample(resident, len(resident))
            future = rng.randrange(3)
            verdict = pool.check_gpu_capacity(required, future_gpu_pages=future,
                                              inactive_page_ids=candidates,
                                              protected_page_ids=protected)
            if not verdict.fits:
                continue
            protected_locations = {p: pool.location(p) for p in protected if p not in required}
            tx = pool.reserve_request(required, future_gpu_pages=future,
                                      inactive_page_ids=candidates, protected_page_ids=protected)
            reservation = pool.commit(tx, pool.ensure_resident(tx, ex), pub)
            for p in required:
                self.assertGreaterEqual(pool.location(p), 0)
            for p, location in protected_locations.items():
                self.assertEqual(pool.location(p), location)
            unbound = [p for p, loc in enumerate(pool.map_row_snapshot()) if loc == m.FREE]
            if unbound and pool.reservation_remaining_pages(reservation):
                attach(pool, pub, ex, [unbound[0]], reservation=reservation)
            pool.release_reservation(reservation)
            if rng.randrange(3) == 0:
                victim = rng.choice([p for p, loc in enumerate(pool.map_row_snapshot()) if loc != m.FREE])
                pool.retire_native_pages([victim], readers_retired=True, publisher=pub)
            ex.validate()
            validate_allocator(self, pool)
            transitions += 1
        self.assertGreater(transitions, 100)


class TestNativeNewPageBindings(unittest.TestCase):
    def test_partial_page_consumption_is_a_pure_noop(self):
        pool, pub, ex = setup()
        tx = pool.reserve_request([], future_gpu_pages=2)
        reservation = pool.commit(tx, pool.ensure_resident(tx, ex), pub)
        before = (pool.map_row_snapshot(), set(pool._future[reservation.reservation_id]), set(pool._consumed))
        for _ in range(63):
            self.assertEqual(pool.consume_future_pages(reservation, 0), ())
        self.assertEqual(before, (pool.map_row_snapshot(), pool._future[reservation.reservation_id], set(pool._consumed)))
        physical = pool.consume_future_pages(reservation, 1)
        self.assertEqual(pool.reservation_remaining_pages(reservation), 1)
        with self.assertRaises(RuntimeError):
            pool.check_gpu_capacity([], future_gpu_pages=0)
        pool.record_consumed_pages([0], physical)
        self.assertEqual(pool.location(0), physical[0])
        ex.seed(0)
        ex.validate()
        pool.release_reservation(reservation)
        validate_allocator(self, pool)

    def test_binding_unique_targets_and_native_logical_reuse(self):
        pool, pub, ex = setup()
        tx = pool.reserve_request([], future_gpu_pages=2)
        reservation = pool.commit(tx, pool.ensure_resident(tx, ex), pub)
        physical = pool.consume_future_pages(reservation, 2)
        with self.assertRaises(ValueError):
            pool.record_consumed_pages([0, 1], [physical[0], physical[0]])
        with self.assertRaises(RuntimeError):
            pool.retire_native_pages([0], readers_retired=True, publisher=pub)
        pool.record_consumed_pages([0, 1], physical)
        with self.assertRaises(ValueError):
            pool.record_consumed_pages([2], physical[:1])
        pool.retire_native_pages([0], readers_retired=True, publisher=pub)
        self.assertEqual(pool.location(0), m.FREE)
        attach(pool, pub, ex, [0], "CPU")
        pool.release_reservation(reservation)
        validate_allocator(self, pool)

    def test_unsubmitted_consumption_rollback_checks_actual_handle_origin(self):
        pool, pub, ex = setup()
        old_tx = pool.reserve_request([], future_gpu_pages=1)
        old = pool.commit(old_tx, pool.ensure_resident(old_tx, ex), pub)
        targets = pool.consume_future_pages(old, 1)
        pool.return_consumed_pages(old, targets)
        self.assertEqual(pool.reservation_remaining_pages(old), 1)
        targets = pool.consume_future_pages(old, 1)
        pool.record_consumed_pages([0], targets)
        pool.retire_native_pages([0], readers_retired=True, publisher=pub)
        new_tx = pool.reserve_request([], future_gpu_pages=1)
        new = pool.commit(new_tx, pool.ensure_resident(new_tx, ex), pub)
        reused = pool.consume_future_pages(new, 1)
        self.assertEqual(reused, targets)
        with self.assertRaises(ValueError):
            pool.return_consumed_pages(old, reused)
        pool.return_consumed_pages(new, reused)
        self.assertEqual(pool.reservation_remaining_pages(new), 1)
        pool.release_reservation(new)
        pool.release_reservation(old)
        validate_allocator(self, pool)

    def test_oversize_is_permanent_but_active_pressure_is_waitable(self):
        pool, pub, ex = setup()
        v = pool.check_gpu_capacity([], future_gpu_pages=4)
        self.assertFalse(v.fits)
        self.assertTrue(v.permanent_error)
        attach(pool, pub, ex, [0, 1, 2])
        v = pool.check_gpu_capacity([], future_gpu_pages=1)
        self.assertFalse(v.fits)
        self.assertFalse(v.permanent_error)
        self.assertEqual(pool.gpu_free_pages, 0)

    def test_sentinel_and_clear_invalidate_old_handles(self):
        pool = m.FlowKVRequestPagePool(owner_count=4, owner_rank=0,
            gpu_token_rows=3*64, cpu_token_rows=3*64, logical_owner_token_rows=6*64,
            layer_num=78, row_bytes=576, index_anchor_layers=(0, 1),
            reserved_gpu_pages=(0,), sentinel_gpu_page=0)
        pub, ex = Publisher(), PayloadExecutor(pool)
        self.assertEqual(pool.map_row_snapshot()[0], 0)
        with self.assertRaises(ValueError):
            pool.attach_native_pages([0], publisher=pub)
        tx = pool.reserve_request([], future_gpu_pages=1)
        handle = pool.commit(tx, pool.ensure_resident(tx, ex), pub)
        with self.assertRaises(RuntimeError):
            pool.clear(readers_retired=False)
        pool.clear(readers_retired=True)
        self.assertEqual(pool.gpu_free_pages, 2)
        self.assertEqual(pool.map_row_snapshot(), (0, -1, -1, -1, -1, -1))
        with self.assertRaises(ValueError):
            pool.reservation_remaining_pages(handle)
        tx = pool.reserve_request([], future_gpu_pages=1)
        new = pool.commit(tx, pool.ensure_resident(tx, ex), pub)
        self.assertNotEqual(new.reservation_id, handle.reservation_id)


if __name__ == "__main__":
    unittest.main()
