// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define FLOWKV_PAGE_HD __host__ __device__
#else
#define FLOWKV_PAGE_HD
#endif

namespace sglang::flowkv {

// Native pages may store one contiguous row or page-separated row segments.
// Cold-store records are always contiguous. Callers pass a valid row/byte;
// transport words are bounded by the immutable row geometry.
template<int PageSize, int PageBytes, int... Segments>
struct NativePageLayout {
  static constexpr int kSegments = sizeof...(Segments);
  static constexpr int kRowBytes = (Segments + ...);
  static_assert(PageSize > 0 && kSegments > 0);
  static_assert(((Segments > 0 && Segments % 8 == 0) && ...));
  static_assert(PageBytes >= PageSize * kRowBytes && PageBytes % 8 == 0);

  FLOWKV_PAGE_HD static constexpr int64_t offset(int64_t row, int byte) {
    constexpr int widths[] = {Segments...};
    const int64_t page = row / PageSize * PageBytes;
    const int slot = row % PageSize;
    int row_prefix = 0, page_prefix = 0;
#ifdef __CUDACC__
#pragma unroll
#endif
    for (int segment = 0; segment < kSegments - 1; ++segment) {
      const int width = widths[segment];
      if (byte < row_prefix + width)
        return page + page_prefix + slot * width + byte - row_prefix;
      row_prefix += width;
      page_prefix += PageSize * width;
    }
    return page + page_prefix + slot * widths[kSegments - 1] + byte - row_prefix;
  }
};

}  // namespace sglang::flowkv

#undef FLOWKV_PAGE_HD
