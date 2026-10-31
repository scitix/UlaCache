"""CPU substitutes verify shutdown ordering, not CUDA quiescence."""

import gc
import sys
import threading
import types
import unittest
import weakref
from contextlib import contextmanager
from unittest.mock import patch

from ulacache.cache.pool_lifecycle import close_pool


OWNED_REFERENCES = (
    "flowkv_peers",
    "flowkv_arena",
    "flowkv_shards",
    "flowkv_staging_positions",
    "flowkv_staging_rows",
    "flowkv_staging_target_rows",
    "_flowkv_global_indices",
    "_flowkv_global_current",
    "_flowkv_lookup_descriptors",
    "udcache_services",
)


class Buffer:
    pass


class CPUGroup:
    def __init__(self, participants=1):
        self.participants = participants
        self.closed_payloads = set()
        self.lock = threading.Lock()
        self.rendezvous = threading.Barrier(participants)


class Pool(types.SimpleNamespace):
    def __delattr__(self, name):
        if name in OWNED_REFERENCES:
            # Exported storage must survive every rank's imported IPC close.
            if len(self.group.closed_payloads) != self.group.participants:
                raise AssertionError("owned references released before peer drain")
            self.events.append((self.rank, "release", name))
        super().__delattr__(name)


class Resource:
    def __init__(self, pool, name):
        self.pool = pool
        self.name = name
        self.closed = False

    def close(self):
        if self.closed:
            raise AssertionError("resource closed twice")
        self.pool.events.append((self.pool.rank, "close", self.name))
        self.closed = True


class PayloadPeers(Resource):
    def close(self):
        # IPC still holds exported storage during close, just as the backend.
        assert self.pool.flowkv_shards is self.pool.kv_buffer[0]
        assert self.pool.flowkv_arena is not None
        super().close()
        with self.pool.group.lock:
            self.pool.group.closed_payloads.add(self.pool.rank)


def make_pool(group, events, *, rank=0, step=True, tier=None):
    pool = Pool(device=f"cuda:{rank}", rank=rank, group=group, events=events)
    for name in OWNED_REFERENCES:
        setattr(pool, name, Buffer())
    pool._flowkv_lookup_descriptors = [Buffer()]
    pool.udcache_services = types.SimpleNamespace(
        group=types.SimpleNamespace(cpu_group=group)
    )
    pool.flowkv_peers = PayloadPeers(pool, "payload")
    pool.kv_buffer = [pool.flowkv_shards]
    pool.native_index = Buffer()
    if step:
        pool.flowkv_step_publication = Resource(pool, "publication")
    pool.flowkv_host = Resource(pool, "host-child-and-pin") if tier == "index" else None
    pool.flowkv_request = Resource(pool, "request-stream-and-pin") if tier == "request" else None
    return pool


@contextmanager
def fake_runtime(events):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        synchronize=lambda device: events.append((device, "cuda-drain"))
    )
    dist = types.ModuleType("torch.distributed")

    def barrier(*, group):
        events.append(("group", "barrier-enter"))
        group.rendezvous.wait(timeout=5)
        if len(group.closed_payloads) != group.participants:
            raise AssertionError("barrier completed before every IPC close")
        events.append(("group", "barrier-exit"))

    dist.barrier = barrier
    torch.distributed = dist
    with patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
        yield


class LifecycleContracts(unittest.TestCase):
    def test_index_tier_close_order_and_native_views(self):
        events = []
        pool = make_pool(CPUGroup(), events, tier="index")
        native_index = pool.native_index
        with fake_runtime(events):
            close_pool(pool, step_publication=True)
        self.assertEqual(
            events[:6],
            [
                ("cuda:0", "cuda-drain"),
                (0, "close", "publication"),
                (0, "close", "host-child-and-pin"),
                (0, "close", "payload"),
                ("group", "barrier-enter"),
                ("group", "barrier-exit"),
            ],
        )
        self.assertIs(pool.native_index, native_index)
        self.assertEqual(len(pool.kv_buffer), 1)
        self.assertIsNone(pool.flowkv_host)
        self.assertIsNone(pool.flowkv_request)
        self.assertFalse(hasattr(pool, "flowkv_step_publication"))
        self.assertTrue(all(not hasattr(pool, name) for name in OWNED_REFERENCES))

    def test_request_stream_and_pins_close_before_payload(self):
        events = []
        pool = make_pool(CPUGroup(), events, tier="request")
        with fake_runtime(events):
            close_pool(pool, step_publication=True)
        closes = [event[2] for event in events if event[1] == "close"]
        self.assertEqual(closes, ["publication", "request-stream-and-pin", "payload"])

    def test_packed_binding_has_no_step_publication(self):
        events = []
        pool = make_pool(CPUGroup(), events, step=False)
        with fake_runtime(events):
            close_pool(pool, step_publication=False)
        self.assertEqual([event[2] for event in events if event[1] == "close"], ["payload"])
        self.assertFalse(hasattr(pool, "flowkv_step_publication"))

    def test_arena_and_descriptor_references_are_released(self):
        events = []
        pool = make_pool(CPUGroup(), events)
        arena_ref = weakref.ref(pool.flowkv_arena)
        descriptor_ref = weakref.ref(pool._flowkv_lookup_descriptors[0])
        with fake_runtime(events):
            close_pool(pool, step_publication=True)
        gc.collect()
        self.assertIsNone(arena_ref())
        self.assertIsNone(descriptor_ref())

    def test_every_rank_closes_imports_before_storage_release(self):
        events = []
        group = CPUGroup(participants=2)
        pools = [make_pool(group, events, rank=rank, step=False) for rank in range(2)]
        errors = []

        def close_rank(pool):
            try:
                close_pool(pool, step_publication=False)
            except BaseException as error:
                errors.append(error)

        with fake_runtime(events):
            threads = [threading.Thread(target=close_rank, args=(pool,)) for pool in pools]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        last_close = max(i for i, event in enumerate(events) if event[1] == "close")
        first_release = min(i for i, event in enumerate(events) if event[1] == "release")
        self.assertGreater(first_release, last_close)

    def test_close_failure_keeps_owned_buffers_and_stops_sequence(self):
        events = []
        pool = make_pool(CPUGroup(), events, tier="index")
        with patch.object(pool.flowkv_host, "close", side_effect=RuntimeError("close failed")):
            with fake_runtime(events):
                with self.assertRaisesRegex(RuntimeError, "close failed"):
                    close_pool(pool, step_publication=True)
        self.assertTrue(all(hasattr(pool, name) for name in OWNED_REFERENCES))
        self.assertFalse(pool.flowkv_peers.closed)
        self.assertFalse(any(event[1] == "release" for event in events))

    def test_wrong_startup_binding_does_not_silently_skip_publication(self):
        events = []
        pool = make_pool(CPUGroup(), events, step=False)
        with fake_runtime(events):
            with self.assertRaises(AttributeError):
                close_pool(pool, step_publication=True)
        self.assertFalse(pool.flowkv_peers.closed)
        self.assertTrue(all(hasattr(pool, name) for name in OWNED_REFERENCES))


if __name__ == "__main__":
    unittest.main()
