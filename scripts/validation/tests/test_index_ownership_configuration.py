"""Index placement can change independently of TP KV placement and CPU capacity."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def owner_configuration(monkeypatch):
    root = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if not root:
        pytest.skip("Set HISPARSE_NATIVE_SOURCE_ROOT to the patched official checkout")
    path = Path(root) / "python/sglang/srt/ulacache/v0522_owner_tp.py"
    name = "_test_owner_index_configuration"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("sharded", [False, True])
def test_score_mode_switch_preserves_native_context_and_gpu_only_capacity(
    owner_configuration, monkeypatch, sharded
):
    owner = owner_configuration
    native = SimpleNamespace(kernel=SimpleNamespace(dsa_decode_score_mode="unchanged"))
    context = ModuleType("sglang.srt.runtime_context")
    context.get_exec = lambda: native
    monkeypatch.setitem(sys.modules, context.__name__, context)
    settings = owner.OwnerTPConfiguration(enabled=True, index_sharding=sharded)
    owner.configure(settings)
    selected = owner.get_exec()
    assert selected.kernel.dsa_decode_score_mode == ("cp" if sharded else "replicated")
    assert native.kernel.dsa_decode_score_mode == "unchanged"
    assert settings.cpu_offload is False and settings.cpu_slots == 0


def test_sharded_draft_requires_sharded_target(owner_configuration):
    with pytest.raises(ValueError, match="sharded target"):
        owner_configuration.OwnerTPConfiguration(
            enabled=True, index_sharding=False, owner_index_draft=True
        )


def test_index_choice_rejects_non_boolean_value(owner_configuration):
    with pytest.raises(ValueError, match="index_sharding must be boolean"):
        owner_configuration.OwnerTPConfiguration(enabled=True, index_sharding="false")
