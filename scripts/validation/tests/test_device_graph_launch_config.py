"""Startup geometry variants must not alias compiled device graphs."""
import importlib

import pytest


def test_device_graph_variants_keep_distinct_bindings(monkeypatch):
    import ulacache.transfer.kernels.device_graph as source

    module = importlib.reload(source)
    calls = []

    def compiler(*args, **kwargs):
        result = object()
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(module, "load_jit", compiler)
    baseline = module.device_graph_module()
    candidate = module.device_graph_module(exchange_blocks=64, exchange_threads=256)
    assert module.device_graph_module() is baseline
    assert module.device_graph_module(exchange_blocks=64, exchange_threads=256) is candidate
    assert baseline is not candidate
    assert [args[0] for args, _, _ in calls] == [
        "flowkv_host_device_graph", "flowkv_host_device_graph_b64_t256"]
    assert calls[0][1]["extra_cuda_cflags"] == [
        "-DUDCACHE_HOST_EXCHANGE_BLOCKS=4", "-DUDCACHE_HOST_EXCHANGE_THREADS=1024"]
    assert calls[1][1]["extra_cuda_cflags"] == [
        "-DUDCACHE_HOST_EXCHANGE_BLOCKS=64", "-DUDCACHE_HOST_EXCHANGE_THREADS=256"]


@pytest.mark.parametrize("blocks,threads", [
    (0, 256), (-1, 256), (True, 256), (1.5, 256), (2147483648, 256),
    (4, 0), (4, 33), (4, 1056), (4, True), (4, 256.0),
])
def test_invalid_launch_is_rejected_before_compilation(monkeypatch, blocks, threads):
    import ulacache.transfer.kernels.device_graph as source

    module = importlib.reload(source)

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid geometry reached the compiler")

    monkeypatch.setattr(module, "load_jit", unexpected)
    with pytest.raises(ValueError):
        module.device_graph_module(exchange_blocks=blocks, exchange_threads=threads)


def test_parallel_selector_does_not_reuse_a_serial_compiler_binding(monkeypatch):
    import ulacache.transfer.kernels.device_graph as source

    module = importlib.reload(source)
    calls = []

    def compiler(*args, **kwargs):
        value = object()
        calls.append((args, kwargs))
        return value

    monkeypatch.setattr(module, "load_jit", compiler)
    serial = module.device_graph_module()
    parallel = module.device_graph_module(parallel_victims=True)
    assert serial is not parallel
    assert module.device_graph_module(parallel_victims=True) is parallel
    assert calls[0][0][0] != calls[1][0][0]
    assert "-DUDCACHE_PARALLEL_VICTIMS=1" in calls[1][1]["extra_cuda_cflags"]
    assert ("finish_parallel", "sglang::flowkv_device_graph_finish_parallel") in calls[1][1]["cuda_wrappers"]
