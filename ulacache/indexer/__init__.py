# SPDX-License-Identifier: Apache-2.0
"""Distributed Index scoring and transport with explicit framework services.

Importing this namespace does not import Torch, Triton, or a serving framework.
"""

from importlib import import_module

__all__ = [
    "CPIndexScorer",
    "CPScoreBucket",
    "CPScorePlan",
    "FlashAGIndexOwner",
    "RankMajorScoreLayout",
    "NativeIndexOwnership",
    "NativeIndexWriteMapping",
    "NativeIndexQueryLayout",
    "NativeOwnerIndexScorer",
    "NativeIndexStagingWriter",
    "NativeFP8IndexRuntime",
]

_EXPORTS = {
    "CPIndexScorer": (".scorer", "CPIndexScorer"),
    "CPScoreBucket": (".scorer", "CPScoreBucket"),
    "CPScorePlan": (".scorer", "CPScorePlan"),
    "FlashAGIndexOwner": (".flash_ag", "FlashAGIndexOwner"),
    "RankMajorScoreLayout": (".score_layout", "RankMajorScoreLayout"),
    "NativeIndexOwnership": (".ownership", "NativeIndexOwnership"),
    "NativeIndexWriteMapping": (".ownership", "NativeIndexWriteMapping"),
    "NativeIndexQueryLayout": (".ownership", "NativeIndexQueryLayout"),
    "NativeOwnerIndexScorer": (".native_owner", "NativeOwnerIndexScorer"),
    "NativeIndexStagingWriter": (".native_writer", "NativeIndexStagingWriter"),
    "NativeFP8IndexRuntime": (".native_fp8", "NativeFP8IndexRuntime"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
