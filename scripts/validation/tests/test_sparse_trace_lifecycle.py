"""Trace hooks must preserve the native allocation API, including CPU placement."""

from pathlib import Path
from unittest.mock import Mock
import inspect

import pytest


def adapter():
    patch = (Path(__file__).resolve().parents[3] / "patches/sglang.patch").read_text()
    name = "python/sglang/srt/ulacache/sparse_trace_adapter.py"
    section = patch.split(f"diff --git a/{name} b/{name}\n", 1)[1].split(
        "\ndiff --git ", 1
    )[0]
    source = "\n".join(
        line[1:]
        for line in section.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    namespace = {}
    exec(compile(source, name, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("placement", [None, True, False])
def test_allocation_preserves_default_or_explicit_native_placement(placement):
    module = adapter()
    recorder = Mock()

    class Cache:
        pass

    cache, rows = Cache(), object()
    module["_observers"][cache] = {"recorder": recorder}
    calls = []
    result = object()

    def allocate_unbound(cache, rows, *, prefer_gpu=True):
        calls.append((cache, rows, prefer_gpu))
        return result

    wrapped = module["trace_lifecycle_call"](allocate_unbound, release=False)
    kwargs = {} if placement is None else {"prefer_gpu": placement}
    assert wrapped(cache, rows, **kwargs) is result
    assert calls == [(cache, rows, True if placement is None else placement)]
    assert inspect.signature(wrapped) == inspect.signature(allocate_unbound)
    recorder.observe_event.assert_called_once_with(rows, release=False)


def test_release_and_nonrecording_calls_keep_native_results_and_errors():
    module = adapter()

    class Cache:
        pass

    cache, rows, recorder = Cache(), object(), Mock()

    def release(cache, rows):
        return rows

    wrapped = module["trace_lifecycle_call"](release, release=True)
    assert wrapped(cache, rows) is rows
    module["_observers"][cache] = {"recorder": recorder}
    assert wrapped(cache, rows) is rows
    recorder.observe_event.assert_called_once_with(rows, release=True)
    with pytest.raises(TypeError, match="unexpected keyword"):
        wrapped(cache, rows, unsupported=True)
