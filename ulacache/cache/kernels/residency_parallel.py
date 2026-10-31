"""Explicit experimental binding; the original residency planner remains default."""

from ulacache.compiler import cache_once, load_jit


@cache_once
def prepare_function(parallel_victims=False):
    module = load_jit(
        "flowkv_host_parallel_prepare",
        "parent_words_v2",
        str(int(parallel_victims)),
        cuda_files=["csrc/cache/residency_parallel.cuh"],
        cuda_wrappers=[("prepare", "sglang::flowkv_host_prepare_parallel")],
        extra_cuda_cflags=[f"-DUDCACHE_PARALLEL_VICTIMS={int(parallel_victims)}"],
    )
    return module.prepare
