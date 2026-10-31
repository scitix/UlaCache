"""The common Index transport starts with or without experiment receipts."""

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def adapter(monkeypatch):
    root = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if not root:
        pytest.skip("Set HISPARSE_NATIVE_SOURCE_ROOT to the patched official checkout")
    source = Path(root) / "python/sglang/srt/distributed/device_communicators/flash_ag_index.py"
    calls = []

    class Owner:
        def __init__(self, group, **kwargs):
            calls.append((group, kwargs))
            self.rank_in_group = 0
            self.world_size = 8
            self.compact_scores = True
            self.unpacks_scores = False
            self.max_queries = kwargs["max_queries"]
            self.max_page_width = kwargs["max_page_width"]
            self.maximum_grid = 512
            self.resident_grid_limit = 512

    modules = {
        "sglang.kernels.jit.utils": {"load_jit": object()},
        "sglang.kernels.ops.communication.all_reduce": {"IPCManager": object()},
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    import ulacache.compiler
    import ulacache.flashag

    monkeypatch.setattr(ulacache.compiler, "configure_compiler", lambda compiler: None)
    monkeypatch.setattr(ulacache.flashag, "create_index_owner", Owner)
    spec = importlib.util.spec_from_file_location("_packed_adapter_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, calls=calls)


def test_packed_transport_without_experiment_directory(adapter, monkeypatch):
    monkeypatch.delenv("UDCACHE_MATRIX_AUDIT_DIR", raising=False)
    group = object()
    owner = adapter.module.create_index_owner(
        group, max_queries=192, max_page_width=4585, device="cuda:0"
    )
    assert owner.world_size == 8
    assert len(adapter.calls) == 1
    assert adapter.calls[0][0] is group
    assert adapter.calls[0][1]["max_queries"] == 192


def test_requested_transport_receipt_keeps_actual_geometry(adapter, monkeypatch, tmp_path):
    monkeypatch.setenv("UDCACHE_MATRIX_AUDIT_DIR", str(tmp_path))
    adapter.module.create_index_owner(
        object(), max_queries=192, max_page_width=4585, device="cuda:0"
    )
    receipt = json.loads((tmp_path / "index_transport.rank0.json").read_text())
    assert receipt["complete"]
    assert receipt["world"] == 8
    assert receipt["max_queries"] == 192
    assert receipt["max_page_width"] == 4585


@pytest.mark.parametrize("capability", [(9, 0), (10, 3)])
def test_hardware_selection_preserves_the_same_creation_arguments(monkeypatch, capability):
    import ulacache.flashag

    calls = []
    device = object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        device=lambda value: device,
        cuda=SimpleNamespace(get_device_capability=lambda value: capability)))

    def construct(group, **kwargs):
        calls.append((group, kwargs))
        return "transport"

    for name, symbol in (("owner", "FlashAGIndexOwner"), ("packed", "PackedFlashAGIndexOwner")):
        module = ModuleType("ulacache.flashag." + name)
        setattr(module, symbol, construct)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    group, ipc = object(), object()
    assert ulacache.flashag.create_index_owner(group, max_queries=192,
        max_page_width=4585, device="cuda:0", ipc_manager_factory=ipc) == "transport"
    assert len(calls) == 1
    actual_group, options = calls[0]
    assert actual_group is group
    assert options["max_queries"] == 192
    assert options["max_page_width"] == 4585
    assert options["device"] is device
    assert options["compact_scores"] is True


def test_dense_api_uses_the_same_compact_descriptor_path():
    import torch
    from ulacache.flashag import IndexTransport

    calls = []

    class Transport(IndexTransport):
        def all_gather_compact(self, output, local, prefix, packs):
            calls.append((output, local, prefix.clone(), packs.clone()))

        def close(self):
            self._closed = True

    owner = Transport()
    owner.device, owner.max_queries, owner._closed = torch.device("cpu"), 8, False
    owner._initialize_dense_gather()
    local, output = torch.empty(3, 16), object()
    owner.all_gather_into_tensor(output, local)
    actual_output, actual_local, prefix, packs = calls.pop()
    assert actual_output is output and actual_local is local
    assert prefix.tolist() == [0, 16, 32, 48]
    assert packs.tolist() == [12]
    owner._closed = True
    with pytest.raises(ValueError, match="open transport"):
        owner.all_gather_into_tensor(output, local)
    assert not calls
