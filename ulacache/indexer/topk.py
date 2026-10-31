# SPDX-License-Identifier: Apache-2.0
"""Compile the validated compact input reader with the native TopK algorithm.

The integration supplies rendered native template arguments and occupancy
flags. Selection kernels and their supporting headers are packaged here.
"""

from ulacache.compiler import cache_once, load_jit


@cache_once
def rank_major_topk_module(template_args: tuple[str, ...], cuda_flags: tuple[str, ...]):
    """Expose the existing native reader for explicit dense or compact input.

    The six native template parameters and the vendored selection implementation
    are unchanged. A dense-unpacking packed transport uses the false input
    specialization and supplies two absent compact descriptors at the call.
    """
    if len(template_args) != 6 or template_args[-1] not in ("true", "false"):
        raise ValueError(
            "Rank-major TopK requires an explicit native input specialization"
        )
    kernel = "TopKRankMajorKernel<" + ", ".join(template_args) + ">"
    return load_jit(
        "ulacache_index_rank_major_topk",
        *template_args,
        extra_cuda_cflags=list(cuda_flags),
        cuda_files=["csrc/indexer/topk_v2.cuh"],
        cuda_wrappers=[
            ("topk_transform_paged_rank_major", f"{kernel}::transform_paged")
        ],
    )


@cache_once
def compact_topk_module(template_args: tuple[str, ...], cuda_flags: tuple[str, ...]):
    if len(template_args) != 6 or template_args[-1] != "true":
        raise ValueError(
            "Compact TopK requires its explicit native input specialization"
        )
    kernel = "TopKRankMajorKernel<" + ", ".join(template_args) + ">"
    return load_jit(
        "udcache_index_compact_topk",
        *template_args,
        extra_cuda_cflags=list(cuda_flags),
        cuda_files=["csrc/indexer/topk_v2.cuh"],
        cuda_wrappers=[
            ("topk_transform_paged_rank_major", f"{kernel}::transform_paged")
        ],
    )
