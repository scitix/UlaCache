"""Startup inventory must preserve PortArgs for native cohort admission."""
import ast
from functools import wraps
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]


def adapter_source(name):
    relative = "python/sglang/srt/ulacache/" + name + ".py"
    patch = (ROOT / "patches/sglang.patch").read_text()
    section = patch.split("diff --git a/" + relative + " b/" + relative + "\n", 1)[1]
    section = section.split("\ndiff --git ", 1)[0]
    return "\n".join(line[1:] for line in section.splitlines()
                     if line.startswith("+") and not line.startswith("+++")) + "\n"


@pytest.mark.parametrize("keyword", [False, True])
def test_inventory_and_admission_share_the_actual_constructor_signature(tmp_path, monkeypatch, keyword):
    calls = []

    class Scheduler:
        def __init__(self, server_args, port_args):
            calls.append((server_args, port_args))
            self.tp_worker = SimpleNamespace(model_runner="actual-runner")
            self.tree_cache = "native-prefix-cache"

    original = Scheduler.__init__
    install = next(node for node in ast.parse(adapter_source("matrix_startup_audit")).body
                   if isinstance(node, ast.FunctionDef) and node.name == "install")
    wrapper = next(node for node in install.body
                   if isinstance(node, ast.FunctionDef) and node.name == "scheduler_init")
    namespace = dict(original_scheduler=original, wraps=wraps,
                     audit=lambda runner, coordinator=None, cache_tree=None:
                         calls.append((runner, coordinator, cache_tree)))
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), "matrix_startup_audit", "exec"), namespace)
    Scheduler.__init__ = namespace["scheduler_init"]
    assert inspect.signature(Scheduler.__init__) == inspect.signature(original)

    admission = ModuleType("scheduler_receipt_under_test")
    exec(compile(adapter_source("native_scheduler_admission"), "native_scheduler_admission", "exec"),
         admission.__dict__)
    admission.process_identity = lambda pid: dict(pid=pid, start_ticks="123")
    io = ModuleType("sglang.srt.managers.io_struct")
    io.__file__ = __file__
    managers = ModuleType("sglang.srt.managers")
    managers.io_struct = io
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = lambda: SimpleNamespace(tp_rank=0)
    monkeypatch.setitem(sys.modules, managers.__name__, managers)
    monkeypatch.setitem(sys.modules, io.__name__, io)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    admission.install_scheduler_receipts(Scheduler, tmp_path)
    ports = SimpleNamespace(rpc_ipc_name="ipc:///tmp/ulacache-scheduler-test")
    if keyword:
        Scheduler(server_args="settings", port_args=ports)
    else:
        Scheduler("settings", ports)
    receipt = json.loads((tmp_path / "scheduler_control.rank0.json").read_text())
    assert receipt["endpoint"] == ports.rpc_ipc_name
    assert calls == [("settings", ports), ("actual-runner", None, "native-prefix-cache")]


@pytest.mark.parametrize("endpoint", ["tcp://0.0.0.0:39001", "tcp://127.0.0.1:0",
    "tcp://127.0.0.1:39001/path", "tcp://127.0.0.1:39001?query=1",
    "tcp://user@127.0.0.1:39001", "ipc://relative", "ipc:///" + "a" * 108])
def test_invalid_endpoint_is_rejected_before_model_initialization(tmp_path, endpoint):
    class Scheduler:
        def __init__(self, server_args, port_args):
            raise AssertionError("Invalid benchmark control endpoint reached model initialization")

    admission = ModuleType("scheduler_receipt_validation_under_test")
    exec(compile(adapter_source("native_scheduler_admission"), "native_scheduler_admission", "exec"),
         admission.__dict__)
    admission.install_scheduler_receipts(Scheduler, tmp_path)
    with pytest.raises(ValueError, match="absolute IPC"):
        Scheduler("settings", SimpleNamespace(rpc_ipc_name=endpoint))
