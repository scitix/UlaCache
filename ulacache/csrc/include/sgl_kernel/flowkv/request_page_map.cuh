// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <assert.h>
#include <stdint.h>

namespace sglang::flowkv {

#if defined(__CUDACC__)
#define FLOW_REQUEST_INLINE __device__ __forceinline__
#else
#define FLOW_REQUEST_INLINE inline
#endif

// The mirrored map resolves an owner-local logical page. Native token IDs
// and owners stay unchanged; only the payload address becomes physical.
FLOW_REQUEST_INLINE int64_t request_physical_row(
    int64_t logical_row, const int32_t* page_map, int64_t logical_rows,
    int64_t gpu_rows) {
  assert(logical_row >= 0 && logical_row < logical_rows);
  const int32_t page = page_map[logical_row / 64];
  // Admission must complete every active-page onload before native readers.
  // A CPU/free encoding here is an execution error, never a polling fallback.
  assert(page >= 0 && int64_t(page) < gpu_rows / 64);
  return int64_t(page) * 64 + logical_row % 64;
}

#undef FLOW_REQUEST_INLINE
}  // namespace sglang::flowkv
