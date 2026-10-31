"""Device-launched cold residency graphs using the common record kernels."""

from ulacache.compiler import cache_once, load_jit


@cache_once
def device_graph_module(
    *, exchange_blocks=4, exchange_threads=1024, parallel_victims=False
):
    """Bind cold-copy geometry before capture; retain the original protocol.

    Defaults remain 4x1024. Explicit dimensions support isolated operator
    comparisons in one process. The compiler and module cache distinguish
    geometry variants; captured graphs retain their original launch shape.
    """
    blocks, threads = exchange_blocks, exchange_threads
    if type(blocks) is not int or not 1 <= blocks <= 2147483647:
        raise ValueError("exchange_blocks must be a positive CUDA grid dimension")
    if type(threads) is not int or not 1 <= threads <= 1024 or threads % 32:
        raise ValueError("exchange_threads must be a warp multiple at most 1024")
    if type(parallel_victims) is not bool:
        raise ValueError("parallel_victims must be boolean")
    name = "flowkv_host_device_graph"
    if (blocks, threads) != (4, 1024):
        name += f"_b{blocks}_t{threads}"
    flags = [
        f"-DUDCACHE_HOST_EXCHANGE_BLOCKS={blocks}",
        f"-DUDCACHE_HOST_EXCHANGE_THREADS={threads}",
    ]
    if parallel_victims:
        name += "_parallel_victims"
        flags.append("-DUDCACHE_PARALLEL_VICTIMS=1")
    return load_jit(
        name,
        cuda_files=["csrc/transfer/device_graph.cuh"],
        extra_cuda_cflags=flags,
        cuda_wrappers=[
            ("create", "sglang::flowkv_device_graph_create"),
            ("finish", "sglang::flowkv_device_graph_finish"),
            ("finish_prepared", "sglang::flowkv_device_graph_finish_prepared"),
            ("finish_parallel", "sglang::flowkv_device_graph_finish_parallel"),
            ("destroy", "sglang::flowkv_device_graph_destroy"),
        ],
    )
