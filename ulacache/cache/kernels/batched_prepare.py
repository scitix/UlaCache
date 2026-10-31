"""Existing owner-local transactions over explicitly independent domains."""

from ulacache.compiler import cache_once, load_jit


@cache_once
def module(parallel_victims=False):
    return load_jit(
        "native_batched_current_v1",
        str(int(parallel_victims)),
        cuda_files=["csrc/cache/batched_prepare.cuh"],
        cuda_wrappers=[
            ("begin", "sglang::flowkv_begin_native_domains"),
            ("prepare", "sglang::flowkv_host_prepare_after_begin"),
        ],
        extra_cuda_cflags=[f"-DUDCACHE_PARALLEL_VICTIMS={int(parallel_victims)}"],
    )


def begin_native_domains(current, active, frames, *, parallel_victims=False):
    module(parallel_victims).begin(current, active, frames)
