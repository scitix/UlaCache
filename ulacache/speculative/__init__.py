# SPDX-License-Identifier: Apache-2.0
"""Speculative query bindings that reuse the ordinary UDCache lifecycle.

This namespace does not import device or serving-framework dependencies.
"""

from .linear import (
    LinearVerifyConfig,
    is_cache_step,
    index_query_metadata,
    draft_kv_cell_bytes,
)

__all__ = [
    "LinearVerifyConfig",
    "is_cache_step",
    "index_query_metadata",
    "draft_kv_cell_bytes",
]
