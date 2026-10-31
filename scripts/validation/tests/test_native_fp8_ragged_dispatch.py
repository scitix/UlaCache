"""Reconstructed prefill keys retain the exact native callback ABI."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ulacache.indexer.native_fp8 import NativeFP8IndexRuntime


def runtime(monkeypatch, *, capturing=False):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        is_current_stream_capturing=lambda: capturing)))
    value=NativeFP8IndexRuntime.__new__(NativeFP8IndexRuntime)
    value.ragged_score_mode='native'
    value.ownership=SimpleNamespace(logical_record_capacity=4096)
    value.score_group=Mock()
    return value


@pytest.mark.parametrize('clean_logits', [False, True])
def test_reconstructed_keys_use_one_unchanged_native_call(monkeypatch, clean_logits):
    value=runtime(monkeypatch)
    q=SimpleNamespace(shape=(2048,32,128))
    keys=SimpleNamespace(shape=(4096,128))
    scales,weights,starts,ends=(object() for _ in range(4))
    output=object()
    native=Mock(return_value=output)
    actual=value.score_ragged(q,keys,scales,weights,starts,ends,
        native_score=native,clean_logits=clean_logits)
    assert actual is output
    native.assert_called_once_with(q,keys,scales,weights,starts,ends,
        clean_logits=clean_logits)
    value.score_group.all_gather_into_tensor.assert_not_called()


def test_native_ragged_mode_keeps_namespace_and_capture_checks(monkeypatch):
    value=runtime(monkeypatch)
    native=Mock()
    with pytest.raises(ValueError,match='admitted logical history'):
        value.score_ragged(SimpleNamespace(shape=(1,)),SimpleNamespace(shape=(4097,)),
            None,None,None,None,native_score=native,clean_logits=True)
    value=runtime(monkeypatch,capturing=True)
    with pytest.raises(NotImplementedError,match='eager-only'):
        value.score_ragged(None,None,None,None,None,None,native_score=native,clean_logits=True)
    native.assert_not_called()


def test_invalid_ragged_policy_is_rejected_before_device_construction():
    with pytest.raises(ValueError,match='native or distributed'):
        NativeFP8IndexRuntime(None,None,maximum_queries=1,maximum_columns=1,
            score_group=None,cpu_group=None,ipc_manager_factory=None,
            schedule_builder=None,ragged_score_mode='unknown')


def test_native_dpa_startup_keeps_the_shared_fp8_loader(monkeypatch):
    import ast
    import os
    from pathlib import Path

    root = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if not root:
        pytest.skip("Set HISPARSE_NATIVE_SOURCE_ROOT to the patched official checkout")
    path = Path(root) / "python/sglang/srt/ulacache/model_integration.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "install_model_integration")
    events = []
    monkeypatch.setitem(sys.modules, "sglang.srt.ulacache.v0522_fp8_loading",
                        SimpleNamespace(install=lambda: events.append("fp8_loader")))
    monkeypatch.setitem(sys.modules, "sglang.srt.ulacache.v0522_dpa",
                        SimpleNamespace(install=lambda pool: events.append(("dpa", pool))))
    monkeypatch.delenv("UDCACHE_MATRIX_AUDIT_DIR", raising=False)
    namespace = dict(_installed=None, LEGACY_CONFIGURATION_VARIABLES=(),
                     POOL_BINDINGS={}, os=os, sys=sys,
                     configure_launch=lambda *args: None)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    pool = object()
    configuration = SimpleNamespace(framework_release="v0.5.22", adapter="native_dpa", pool=pool)
    namespace[function.name](configuration)
    namespace[function.name](configuration)
    assert events == ["fp8_loader", ("dpa", pool)]


@pytest.mark.parametrize("budget", [None, 192_000_000_000])
def test_dpa_pool_construction_uses_one_explicit_host_budget(monkeypatch, budget):
    import ast
    from contextlib import contextmanager
    from functools import wraps
    import os
    from pathlib import Path

    root = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if not root:
        pytest.skip("Set HISPARSE_NATIVE_SOURCE_ROOT to the patched official checkout")
    path = Path(root) / "python/sglang/srt/ulacache/v0522_dpa.py"
    install = next(node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == "install")
    build = next(node for node in install.body
                 if isinstance(node, ast.FunctionDef) and node.name == "build")
    events = []

    @contextmanager
    def scope(value):
        events.append(("enter", value))
        try:
            yield
        finally:
            events.append(("exit", value))

    def original(owner, **kwargs):
        events.append(("allocate", kwargs))
        return "native-pool"

    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.pool_host.base",
                        SimpleNamespace(host_memory_budget_scope=scope))
    namespace = dict(wraps=wraps, original=original,
                     configuration=SimpleNamespace(host_cache_budget_bytes=budget),
                     dsa_layer_skips_topk=lambda config, layer: False,
                     get_dsa_index_topk=lambda config: 2048,
                     create_pool_class=lambda *args: "DPA-pool")
    exec(compile(ast.Module(body=[build], type_ignores=[]), str(path), "exec"), namespace)
    owner = SimpleNamespace(is_draft_worker=False,
                            layer_info=SimpleNamespace(start_layer=0, end_layer=2),
                            model_config=SimpleNamespace(hf_config=object()))
    assert namespace[build.name](owner, max_total_num_tokens=128) == "native-pool"
    allocation = ("allocate", dict(max_total_num_tokens=128, dsa_pool_class="DPA-pool"))
    expected = [allocation] if budget is None else [("enter", budget), allocation, ("exit", budget)]
    assert events == expected
