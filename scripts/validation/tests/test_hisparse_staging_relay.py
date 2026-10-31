"""A staged MTP proposal is materialized once, including repeated startup binds."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def relay(monkeypatch):
    root = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if not root:
        pytest.skip("Set HISPARSE_NATIVE_SOURCE_ROOT to the patched official checkout")
    path = Path(root) / "python/sglang/srt/ulacache/hisparse_prefill_relay_adapter.py"
    spec = importlib.util.spec_from_file_location("_hisparse_staging_relay", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_materialized_draft_preserves_proposal_on_repeated_resolution(relay):
    proposal, selected = object(), object()
    draft = SimpleNamespace(future_indices=selected, topk_p=None)
    batch = SimpleNamespace(
        enable_overlap=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        spec_info=draft,
    )
    resolved = []

    def resolve(value):
        assert value.spec_info.future_indices is selected
        value.spec_info.topk_p = proposal
        resolved.append(value)

    scheduler = SimpleNamespace(future_map=SimpleNamespace(_resolve_spec_extras=resolve))
    assert relay.resolve_hisparse_staged_input(scheduler, batch) is batch
    assert relay.resolve_hisparse_staged_input(scheduler, batch) is batch
    assert resolved == [batch]
    assert draft.future_indices is None and draft.topk_p is proposal


def test_repeated_startup_binding_keeps_one_staging_transition(relay, monkeypatch):
    class Scheduler:
        def run_batch(self, batch):
            return batch

        def _build_hisparse_decode_batch(self, requests):
            return requests

    native = ModuleType("sglang.srt.managers.scheduler")
    native.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, native.__name__, native)
    relay.install()
    first = Scheduler.run_batch, Scheduler._build_hisparse_decode_batch
    relay.install()
    assert (Scheduler.run_batch, Scheduler._build_hisparse_decode_batch) == first
