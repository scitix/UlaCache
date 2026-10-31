# SPDX-License-Identifier: Apache-2.0
"""Experimental launch-only tuning of the unchanged PB transaction kernels."""

from typing import NamedTuple

from ulacache.compiler import cache_once, load_jit, recency_policy


class PBLaunch(NamedTuple):
    pin_blocks: int = 128
    pin_threads: int = 256
    victim_blocks: int = 128
    assignment_blocks: int = 128
    assignment_threads: int = 256
    small_query_parallel: bool = False


def validate_config(config):
    if any(
        type(x) is not int or x < 1 or x > 1024
        for x in (config.pin_blocks, config.victim_blocks, config.assignment_blocks)
    ):
        raise ValueError("PB launch grids must be between 1 and 1024 blocks")
    if config.pin_threads not in (128, 256, 512) or config.assignment_threads not in (
        128,
        256,
        512,
    ):
        raise ValueError("PB pin and assignment blocks require 128, 256 or 512 threads")
    if type(config.small_query_parallel) is not bool:
        raise TypeError("small_query_parallel must be a bool")
    return config


@cache_once
def module():
    if recency_policy() != "pb":
        raise ValueError("Launch tuning requires the PB recency policy")
    return load_jit(
        "pb_launch",
        "v1",
        cuda_files=["csrc/eviction/launch_config.cuh"],
        cuda_wrappers=[("prepare", "sglang::pb_launch_prepare")],
    )


def prepare(
    selected,
    current,
    active,
    owner,
    state,
    workspace,
    *,
    selected_counts=None,
    config=PBLaunch(),
):
    """Change launch geometry only; expiry, commit and lease release are external."""
    validate_config(config)
    if selected.ndim != 2 or current.ndim != 1 or selected.size(0) != current.numel():
        raise ValueError("Expected selected[query,topk] and current[query]")
    counts = active if selected_counts is None else selected_counts
    module().prepare(
        selected,
        current,
        active,
        counts,
        selected_counts is not None,
        owner,
        *state,
        *workspace,
        *config,
    )
