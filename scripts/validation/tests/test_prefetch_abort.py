"""Partial-forward cleanup retires prefetch readers before dropping metadata."""

from types import SimpleNamespace

import pytest

from ulacache.cache.prefetch import SharedIndexLifecycle


@pytest.mark.parametrize("drain_fails", [False, True])
def test_abort_discards_partial_state_and_keeps_publication_owned_by_the_adapter(drain_fails):
    pool = SimpleNamespace(flowkv_step_publication=SimpleNamespace(
        publish=lambda: pytest.fail("The adapter must decide whether an epoch began")))
    lifecycle = SharedIndexLifecycle(SimpleNamespace(layer_count=2), pool, prefetch=False)
    lifecycle.begin_forward()
    lifecycle._planned.add(0)
    lifecycle._consumed.add(0)
    lifecycle._deferred_pending = True
    lifecycle._keepalive[0] = object()
    lifecycle._followers_scheduled.add(1)
    lifecycle._followers_joined.add(1)
    drained = []

    def drain():
        assert lifecycle._keepalive
        drained.append(True)
        if drain_fails:
            raise RuntimeError("failed device drain")

    lifecycle.prefetch = True
    lifecycle._stream = SimpleNamespace(synchronize=drain)
    if drain_fails:
        with pytest.raises(RuntimeError, match="failed device drain"):
            lifecycle.abort_forward()
    else:
        lifecycle.abort_forward()
    assert drained == [True]
    assert not lifecycle.in_forward and not lifecycle._deferred_pending
    assert not lifecycle._keepalive and not lifecycle._consumed
    assert not lifecycle._followers_scheduled and not lifecycle._followers_joined
    lifecycle.begin_forward()
