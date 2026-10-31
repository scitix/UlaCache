"""Native reader epochs must retire before CPU receive publication can return."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
from ulacache.cache.host import FlowKVHostTier
from ulacache.cache.prefetch import SharedIndexLifecycle


def fixture(monkeypatch, *, prefetch=True, groups=21):
    events = []
    stream = SimpleNamespace(wait_stream=lambda other: events.append(('join', other)),
                             synchronize=lambda: events.append(('synchronize',)))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: stream)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    pool = SimpleNamespace(device='cuda:0')
    group_plan = SimpleNamespace(group_count=groups)
    lifecycle = object.__new__(SharedIndexLifecycle)
    lifecycle.pool, lifecycle.groups = pool, group_plan
    lifecycle.prefetch, lifecycle._planned = prefetch, None
    lifecycle._stream = 'native_history_stream'
    pool.flowkv_shared_index = lifecycle
    tier = object.__new__(FlowKVHostTier)
    tier.pool, tier.groups = pool, group_plan
    tier.states = tuple('group_' + str(g) for g in range(groups))
    tier.ready_peers = SimpleNamespace(ptrs='native_peer_ready_ptrs')
    import ulacache.cache.kernels.cpu_residency as kernel
    monkeypatch.setattr(kernel, 'drain_host_tier_readers',
        lambda state, ptrs, offset: events.append(('peer_wait', state, ptrs, offset)))
    return tier, lifecycle, events


@pytest.mark.parametrize('prefetch', [False, True])
def test_all21_native_reader_epochs_retire_before_host_publication(monkeypatch, prefetch):
    tier, lifecycle, events = fixture(monkeypatch, prefetch=prefetch)
    tier.drain_readers()
    expected = [('join', 'native_history_stream')] if prefetch else []
    expected += [('peer_wait', 'group_' + str(g), 'native_peer_ready_ptrs', 2*g+1)
                 for g in range(21)]
    expected += [('synchronize',)]
    assert events == expected
    assert not lifecycle.in_forward


def test_partial_forward_cannot_publish_or_enqueue_epoch_wait(monkeypatch):
    tier, lifecycle, events = fixture(monkeypatch)
    lifecycle._planned = {0}
    with pytest.raises(RuntimeError, match='during a shared Index forward'):
        tier.drain_readers()
    assert events == []


def test_capture_cannot_enter_host_publication(monkeypatch):
    tier, _, events = fixture(monkeypatch)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='CUDA Graph capture'):
        tier.drain_readers()
    assert events == []


@pytest.mark.parametrize('invalid', ['missing', 'other_pool', 'other_groups'])
def test_mismatched_native_lifecycle_rejected_before_mutation(monkeypatch, invalid):
    tier, lifecycle, events = fixture(monkeypatch)
    if invalid == 'missing':
        del tier.pool.flowkv_shared_index
    elif invalid == 'other_pool':
        lifecycle.pool = SimpleNamespace()
    else:
        lifecycle.groups = SimpleNamespace()
    with pytest.raises(ValueError, match='actual native shared Index lifecycle'):
        tier.drain_readers()
    assert events == []
