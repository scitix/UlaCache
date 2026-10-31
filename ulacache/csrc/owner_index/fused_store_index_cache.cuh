#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>
#include <sgl_kernel/flowkv/request_page_map.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

struct FusedStoreCacheParam {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
  uint32_t owner_rank;
  const int32_t* owner_page_map;
  int64_t logical_rows;
  int64_t gpu_rows;
};

[[maybe_unused]]
SGL_DEVICE float fp8_e4m3_clip(float val) {
  namespace math = device::math;
  return math::max(math::min(val, kFP8E4M3Max), -kFP8E4M3Max);
}

[[maybe_unused]]
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};
}

template <typename KeyT, typename IndicesT, uint32_t kPageBits, bool kUsePDL, uint32_t kOwnerDegree = 1, bool kRequestMap = false>
__global__ void fused_store_indexer_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  /// NOTE: 132 = 128 + 4
  constexpr int64_t kPageBytes = 132 << kPageBits;

  // each warp handles 128 elements, each block handles multiple rows
  const auto& [input, cache, indices, num_tokens, owner_rank, owner_page_map, logical_rows, gpu_rows] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;

  if (global_wid >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  // prefetch the index
  auto index = static_cast<const IndicesT*>(indices)[global_wid];
  if constexpr (kOwnerDegree > 1) {
    if (index <= 0 || index % kOwnerDegree != owner_rank) {
      PDLTriggerSecondary<kUsePDL>();
      return;
    }
    index /= kOwnerDegree;
  }
  if constexpr (kRequestMap)
    index = sglang::flowkv::request_physical_row(index, owner_page_map, logical_rows, gpu_rows);
  // always load the value from input (don't store if invalid)
  using KeyT2 = packed_t<KeyT>;
  using InStorage = AlignedVector<KeyT2, 2>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
  const auto elems = static_cast<const InStorage*>(input)[global_tid];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabs(x0), fabs(x1)), fmaxf(fabs(y0), fabs(y1)));
  const auto abs_max = warp::reduce_max(local_max);
  // use normal fp32 scale
  const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
  const auto inv_scale = 1.0f / scale;
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(cache, page * kPageBytes);
  const auto value_ptr = pointer::offset(page_ptr, offset * 128);
  const auto scale_ptr = pointer::offset(page_ptr, 128 << kPageBits, offset * 4);
  OutStorage result;
  result[0] = pack_fp8(x0 * inv_scale, x1 * inv_scale);
  result[1] = pack_fp8(y0 * inv_scale, y1 * inv_scale);
  static_cast<OutStorage*>(value_ptr)[lane_id] = result;
  static_cast<float*>(scale_ptr)[0] = scale;

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

template <typename KeyT, typename IndicesT, uint32_t kPageSize, bool kUsePDL, uint32_t kOwnerDegree = 1, bool kRequestMap = false>
struct FusedStoreCacheIndexerKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  /// NOTE: 132 = 128 + 4 (128 represent K and 4 represent scale)
  static constexpr int64_t kPageBytes = 132 * kPageSize;
  static constexpr auto kernel = fused_store_indexer_cache<KeyT, IndicesT, kLogSize, kUsePDL, kOwnerDegree, kRequestMap>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);
  static_assert(std::has_single_bit(kOwnerDegree), "owner degree must be a power of 2");
  static_assert(!kRequestMap || (kPageSize == 64 && kOwnerDegree > 1), "request map requires owner page64");

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    run_owner(input, cache, indices, 0);
  }

  static void run_owner(
      tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices, int64_t owner_rank) {
    run_owner_impl(input, cache, indices, owner_rank, nullptr, 0);
  }

  static void run_owner_mapped(
      tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices,
      int64_t owner_rank, tvm::ffi::TensorView page_map) {
    static_assert(kRequestMap, "mapped writer requires its separate JIT specialization");
    using namespace host;
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(cache.device()).verify(page_map);
    RuntimeCheck(page_map.stride(0) == 1, "Request map must be contiguous");
    run_owner_impl(input, cache, indices, owner_rank,
        static_cast<const int32_t*>(page_map.data_ptr()), page_map.shape()[0] * 64);
  }

  static void run_owner_impl(
      tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices,
      int64_t owner_rank, const int32_t* page_map, int64_t logical_rows) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128})  // input
        .with_dtype<KeyT>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);
    RuntimeCheck(owner_rank >= 0 && owner_rank < kOwnerDegree, "Invalid Index owner rank");
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedStoreCacheParam{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
        .owner_rank = static_cast<uint32_t>(owner_rank),
        .owner_page_map = page_map, .logical_rows = logical_rows,
        .gpu_rows = cache.shape()[0] * kPageSize,
    };
    const auto kBlockSize = 128;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
