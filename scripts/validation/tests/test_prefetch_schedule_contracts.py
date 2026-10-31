"""Exercise real scheduling methods with recorded streams, not CUDA transfers."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from ulacache.cache.layers import LayerGroupPlan
from ulacache.cache.local_paged_cache import NativePagedResidency
from ulacache.cache.prefetch import SharedIndexLifecycle
from ulacache.cache_config import PrefetchPolicy


def test_cuda_fixture_requires_an_explicit_cuda_device_without_initializing_it(monkeypatch):
    from cuda_prefetch_completion_fixture import run_prefetch_completion_fixture

    def forbidden(*args, **kwargs):
        raise AssertionError('CPU fixture validation must not initialize CUDA')

    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    with pytest.raises(ValueError, match='explicitly selected CUDA'):
        run_prefetch_completion_fixture('cpu')


@pytest.fixture
def streams(monkeypatch):
    events = []

    class Stream:
        def __init__(self, name): self.name = name
        def wait_stream(self, other): events.append((self.name, 'wait_stream', other.name))
        def wait_event(self, event): events.append((self.name, 'wait_event', event.name))

    class Event:
        count = 0
        def __init__(self):
            self.name = f'event{Event.count}'
            Event.count += 1
        def record(self, stream): events.append((stream.name, 'record', self.name))

    compute, transfer = Stream('compute'), Stream('transfer')
    current = [compute]

    @contextmanager
    def on_stream(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kwargs: transfer)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: current[0])
    monkeypatch.setattr(torch.cuda, 'stream', on_stream)
    return SimpleNamespace(events=events, current=current, compute=compute, transfer=transfer, Event=Event)


def tp_lifecycle(streams, *, prefetch=True, completion='auto'):
    events = streams.events
    pool = SimpleNamespace(device='cuda:0',
        flowkv_step_publication=SimpleNamespace(publish=lambda: events.append('publish')))
    lifecycle = SharedIndexLifecycle(LayerGroupPlan.from_layer_anchors((0, 0, 0)), pool,
        prefetch=PrefetchPolicy(enabled=prefetch, completion=completion))

    def plan(layer, ids):
        events.append((streams.current[0].name, 'plan', layer))
        lifecycle._planned.add(0)

    lifecycle._plan_index = plan
    lifecycle._pull_history = lambda layer: events.append((streams.current[0].name, 'pull', layer))
    return lifecycle


def test_tp_anchor_and_follower_start_points_and_group_wait(streams):
    lifecycle = tp_lifecycle(streams, completion='group')
    lifecycle.begin_forward()
    lifecycle.on_index_ready(0, object())
    assert streams.events == [
        ('transfer', 'wait_stream', 'compute'), ('transfer', 'plan', 0),
        ('transfer', 'pull', 0), ('transfer', 'record', lifecycle._anchor_ready[0].name)]
    lifecycle.before_history_consume(0)
    assert streams.events[-1] == ('compute', 'wait_event', lifecycle._anchor_ready[0].name)
    lifecycle.after_attention(0)
    assert streams.events[-4:] == [
        ('transfer', 'wait_stream', 'compute'), ('transfer', 'pull', 1),
        ('transfer', 'pull', 2), ('transfer', 'record', lifecycle._followers_ready[0].name)]
    lifecycle.before_layer_prepare(1)
    assert streams.events[-1] == ('compute', 'wait_event', lifecycle._followers_ready[0].name)
    lifecycle.before_history_consume(1)
    lifecycle.after_attention(1)
    lifecycle.before_history_consume(2)
    lifecycle.after_attention(2)
    lifecycle.finish_forward()
    assert streams.events[-1] == 'publish'
    assert not lifecycle.in_forward


def test_tp_default_waits_for_each_follower_before_its_own_prepare(streams):
    lifecycle = tp_lifecycle(streams)
    assert lifecycle.completion == 'layer' and lifecycle.prefetch is True
    assert lifecycle._followers_ready == (None,)
    lifecycle.begin_forward()
    lifecycle.before_layer_prepare(0)
    assert streams.events == []
    lifecycle.on_index_ready(0, object())
    lifecycle.before_history_consume(0)
    lifecycle.after_attention(0)
    first, second = lifecycle._layer_ready[1:]
    assert streams.events[-5:] == [
        ('transfer', 'wait_stream', 'compute'), ('transfer', 'pull', 1),
        ('transfer', 'record', first.name), ('transfer', 'pull', 2),
        ('transfer', 'record', second.name)]
    lifecycle.before_layer_prepare(1)
    assert streams.events[-1] == ('compute', 'wait_event', first.name)
    assert ('compute', 'wait_event', second.name) not in streams.events
    lifecycle.before_history_consume(1)
    lifecycle.after_attention(1)
    # Joining the first layer must not make its sibling appear ready.
    with pytest.raises(RuntimeError, match='join before Q'):
        lifecycle.before_history_consume(2)
    lifecycle.before_layer_prepare(2)
    assert streams.events[-1] == ('compute', 'wait_event', second.name)
    lifecycle.before_history_consume(2)
    lifecycle.after_attention(2)
    lifecycle.finish_forward()
    assert streams.events[-1] == 'publish'


@pytest.mark.parametrize('completion', ['group', 'layer'])
def test_tp_new_forward_cannot_reuse_old_follower_completion(streams, completion):
    lifecycle = tp_lifecycle(streams, completion=completion)
    for _ in range(2):
        lifecycle.begin_forward()
        lifecycle.on_index_ready(0, object())
        with pytest.raises(RuntimeError, match='scheduled'):
            lifecycle.before_layer_prepare(1)
        lifecycle.before_history_consume(0)
        lifecycle.after_attention(0)
        for layer in (1, 2):
            lifecycle.before_layer_prepare(layer)
            lifecycle.before_history_consume(layer)
            lifecycle.after_attention(layer)
        lifecycle.finish_forward()
    assert streams.events.count('publish') == 2


@pytest.mark.parametrize('completion', ['group', 'layer'])
def test_tp_completion_is_inherited_from_the_native_pool_policy(streams, completion):
    policy = PrefetchPolicy(enabled=True, completion=completion)
    groups = LayerGroupPlan.from_layer_anchors((0, 0, 2, 2))
    pool = SimpleNamespace(device='cuda:0', flowkv_prefetch_policy=policy)
    # This is the unchanged native factory signature: a bool plus its pool.
    lifecycle = SharedIndexLifecycle(groups, pool, prefetch=True)
    assert lifecycle.completion == completion
    assert lifecycle.prefetch is True
    assert lifecycle._stream is streams.transfer
    with pytest.raises(ValueError, match='match the policy'):
        SharedIndexLifecycle(groups, pool, prefetch=False)
    other = 'layer' if completion == 'group' else 'group'
    with pytest.raises(ValueError, match='completion differs'):
        SharedIndexLifecycle(groups, pool, prefetch=PrefetchPolicy(True, other))


def test_disabled_policy_does_not_construct_a_stream_or_events(streams):
    pool = SimpleNamespace(device='cuda:0')
    lifecycle = SharedIndexLifecycle(LayerGroupPlan.from_layer_anchors((0, 0)), pool,
        prefetch=PrefetchPolicy(enabled=False, completion='layer'))
    assert lifecycle.prefetch is False and not hasattr(lifecycle, '_stream')
    assert streams.Event.count == 0


@pytest.mark.parametrize('completion', ['group', 'layer'])
def test_tp_full_multigroup_lifecycle_keeps_anchor_and_follower_events_separate(streams, completion):
    groups = LayerGroupPlan.from_layer_anchors((0, 0, 0, 3, 3, 5))
    pool = SimpleNamespace(device='cuda:0', flowkv_step_publication=SimpleNamespace(
        publish=lambda: streams.events.append('publish')))
    lifecycle = SharedIndexLifecycle(groups, pool, prefetch=PrefetchPolicy(True, completion))
    lifecycle._plan_index = lambda layer, ids: lifecycle._planned.add(groups.group(layer))
    lifecycle._pull_history = lambda layer: streams.events.append(('transfer', 'pull', layer))
    for _ in range(2):
        lifecycle.begin_forward()
        for layer in range(groups.layer_count):
            before = len(streams.events)
            lifecycle.before_layer_prepare(layer)
            if layer in groups.anchors:
                assert len(streams.events) == before
                lifecycle.on_index_ready(layer, object())
            elif completion == 'layer':
                assert streams.events[-1] == ('compute', 'wait_event', lifecycle._layer_ready[layer].name)
            lifecycle.before_history_consume(layer)
            lifecycle.after_attention(layer)
        lifecycle.finish_forward()
    assert streams.events.count('publish') == 2
    for layer in (1, 2, 4):
        event = lifecycle._layer_ready[layer] if completion == 'layer' else lifecycle._followers_ready[groups.group(layer)]
        assert event is not None
    if completion == 'layer':
        assert len({id(lifecycle._layer_ready[layer]) for layer in (1, 2, 4)}) == 3


def test_tp_follower_cannot_consume_before_its_prepare_hook_joins(streams):
    lifecycle = tp_lifecycle(streams)
    lifecycle.begin_forward()
    lifecycle.on_index_ready(0, object())
    with pytest.raises(RuntimeError, match='scheduled'):
        lifecycle.before_layer_prepare(1)
    lifecycle.before_history_consume(0)
    lifecycle.after_attention(0)
    with pytest.raises(RuntimeError, match='join before Q'):
        lifecycle.before_history_consume(1)
    with pytest.raises(RuntimeError, match='unconsumed'):
        lifecycle.finish_forward()


def test_tp_without_prefetch_pulls_each_layer_on_the_compute_stream(streams):
    lifecycle = tp_lifecycle(streams, prefetch=False)
    lifecycle.begin_forward()
    lifecycle.on_index_ready(0, object())
    assert streams.events == [('compute', 'plan', 0)]
    for layer in range(3):
        lifecycle.before_history_consume(layer)
        lifecycle.after_attention(layer)
    lifecycle.finish_forward()
    assert streams.events == [('compute', 'plan', 0), ('compute', 'pull', 0),
        ('compute', 'pull', 1), ('compute', 'pull', 2), 'publish']


def test_native_paged_follower_waits_only_for_its_own_bank(streams, monkeypatch):
    import sys

    events = streams.events
    monkeypatch.setitem(sys.modules, 'ulacache.cache.kernels.cpu_residency',
        SimpleNamespace(commit_host_tier=lambda *args: events.append('commit')))
    runtime = NativePagedResidency.__new__(NativePagedResidency)
    runtime.device = 'cuda:0'
    runtime.execution_plan = SimpleNamespace(residency_domain_indices_by_layer=(0, 0, 0),
        storage_bank_indices_by_layer=(0, 1, 2))
    runtime.record_storage = SimpleNamespace(exchange_record_bank=lambda bank, work:
        events.append((streams.current[0].name, 'exchange', bank)))
    runtime.transfer_workspaces, runtime.residency_states = (object(),), (object(),)
    runtime._async_banks, runtime._anchor_banks = {0: (1, 2)}, {0: 0}
    runtime._transfer_start = {0: streams.Event()}
    runtime._bank_ready = {1: streams.Event(), 2: streams.Event()}
    runtime._transfer_done = {0: runtime._bank_ready[2]}
    runtime._transfer_stream = streams.transfer
    runtime._live_residency_domains, runtime._pending_transfers, runtime._joined_banks = set(), set(), set()
    runtime._exchange_and_commit(0, selection_layer=0)
    assert events[:2] == [('compute', 'exchange', 0), 'commit']
    assert ('transfer', 'exchange', 1) in events and ('transfer', 'exchange', 2) in events
    runtime.before_layer_prepare(1)
    assert events[-1] == ('compute', 'wait_event', runtime._bank_ready[1].name)
    assert ('compute', 'wait_event', runtime._bank_ready[2].name) not in events
    runtime.before_layer_prepare(2)
    assert events[-1] == ('compute', 'wait_event', runtime._bank_ready[2].name)
    runtime._drain_transfers()
    assert not runtime._pending_transfers
