# SPDX-License-Identifier: Apache-2.0
"""Explicit startup binding for the original header-only JIT entry points.

The compiler accepts the existing ``load_jit`` keyword arguments, including
absolute source paths, include directories, flags, and typed FFI exports. It
must compile CUDA C++20 with CUDA/CCCL, TVM-FFI, and DLPack available. UDCache
does not discover or import a framework compiler and does not provide a
fallback. SGLang's integration supplies its existing compiler at startup.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)
_compiler: Callable | None = None
_recency_policy: str | None = None
_package = Path(__file__).resolve().parent
_native = _package / "csrc"


def recency_policy() -> str:
    """Freeze one explicit process-wide policy before descriptors or JIT bind.

    The ablation changes metadata maintenance and victim selection only. A
    policy cannot be switched underneath already captured modules or views.
    """
    global _recency_policy
    requested = os.environ.get("UDCACHE_RECENCY_POLICY", "pb")
    if requested not in ("pb", "naive"):
        raise ValueError("UDCACHE_RECENCY_POLICY must be pb or naive")
    if _recency_policy is None:
        _recency_policy = requested
    elif requested != _recency_policy:
        raise RuntimeError("UDCache recency policy is frozen for this process")
    return _recency_policy


def configure_compiler(compiler: Callable) -> None:
    """Bind one compiler before module creation or Graph capture.

    Repeating the same binding is harmless. Replacing it would split the
    cached module/FFI identity, so a different binding is rejected.
    """
    global _compiler
    recency_policy()
    if not callable(compiler):
        raise TypeError("The kernel compiler must be callable")
    if _compiler is not None and _compiler is not compiler:
        raise RuntimeError("The UDCache kernel compiler is already configured")
    _compiler = compiler


def cache_once(fn: F) -> F:
    """Preserve the original cache key and Torch compile compatible wrapper."""
    result_map = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper


def load_jit(*args: str, **kwargs):
    """Forward an unchanged kernel request using the packaged native sources."""
    if _compiler is None:
        raise RuntimeError(
            "Call ulacache.compiler.configure_compiler during startup "
            "before loading CUDA kernels"
        )
    request = dict(kwargs)
    if recency_policy() == "naive":
        # Every writer, pin, native allocation/release and finish translation
        # unit must agree. Keep the original PB module names and flags intact.
        if not args:
            raise ValueError("Naive recency requires an explicit JIT module name")
        args = (args[0] + "_naive_lru", *args[1:])
        for key in ("extra_cflags", "extra_cuda_cflags"):
            request[key] = [*(request.get(key) or []), "-DUDCACHE_NAIVE_LRU=1"]
    for name in ("cpp_files", "cuda_files"):
        if request.get(name) is not None:
            request[name] = [
                str(
                    path.resolve()
                    if path.is_absolute()
                    else (_package / path).resolve()
                )
                for path in map(Path, request[name])
            ]
    include = str(_native / "include")
    # The existing compiler appends its own include directories after caller
    # flags. Put the vendored support headers first without changing any macro
    # definition, launch setting, or kernel body.
    for name in ("extra_cflags", "extra_cuda_cflags"):
        request[name] = [f"-I{include}", f"-I{_package}", *(request.get(name) or [])]
    request["extra_include_paths"] = [
        include,
        str(_package),
        *(request.get("extra_include_paths") or []),
    ]
    return _compiler(*args, **request)
