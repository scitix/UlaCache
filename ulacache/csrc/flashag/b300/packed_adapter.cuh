#pragma once
#include "csrc/flashag/b300/ag_kernels_dynamic.cuh"
#include "csrc/indexer/ipc_compact.cuh"

namespace udcache::index {

template<int World>
int64_t packed_resident_blocks() {
  namespace ka = sglang::flash_ag_kernels_b300;
  namespace kc = ulacache_flashag::flash_ag_common_b300;
  const auto kernel = ka::k_ag_mcst<kc::PkF32, World, ka::kModeD31, ka::kSendUni, 1,
                                   ka::kPollDense, false, false, false, true>;
  int device = 0, per_sm = 0;
  cudaDeviceProp properties{};
  using namespace sglang::host;
  RuntimeCheck(cudaGetDevice(&device) == cudaSuccess &&
               cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
               cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 256, 0) == cudaSuccess,
               "Cannot establish the FlashAG spinning-grid residency bound");
  return int64_t(per_sm) * properties.multiProcessorCount;
}

__global__ void pack_scores(const uint4* input, uint4* staging,
    const int32_t* prefix, const int64_t* packs, int queries, int columns) {
  sglang::device::PDLWaitPrimary<true>();
  const int64_t count = *packs;
  assert(count >= 0 && count <= int64_t(queries) * columns / 4);
  if (count == 0 && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    staging[0] = make_uint4(0, 0, 0, 0);
  const int query = blockIdx.y;
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int begin = prefix[query] / 4;
  const int length = (prefix[query + 1] - prefix[query]) / 4;
  if (offset < length) {
    staging[int64_t(begin) + offset] = input[int64_t(query) * (columns / 4) + offset];
  }
  sglang::device::PDLTriggerSecondary<true>();
}

template<int World>
__global__ void unpack_scores(const uint4* packed, uint4* dense,
    const int32_t* prefix, const int64_t* packs, int queries, int columns) {
  sglang::device::PDLWaitPrimary<true>();
  const int64_t count = *packs;
  const int peer = blockIdx.z;
  const int query = blockIdx.y;
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int begin = prefix[query] / 4;
  const int length = (prefix[query + 1] - prefix[query]) / 4;
  if (offset < length) {
    const int64_t source = int64_t(peer) * count + begin + offset;
    const int64_t target = (int64_t(peer) * queries + query) * (columns / 4) + offset;
    dense[target] = packed[source];
  }
  // Padding remains unspecified, as with native clean_logits=False scoring.
  // Native TopK validity continues to use the original logical lengths.
  sglang::device::PDLTriggerSecondary<true>();
}

template<int World>
void packed_all_gather(tvm::ffi::TensorView input, tvm::ffi::TensorView output,
    tvm::ffi::TensorView prefix, tvm::ffi::TensorView packs,
    tvm::ffi::TensorView staging, tvm::ffi::TensorView packed,
    tvm::ffi::TensorView slab_peers, tvm::ffi::TensorView slab,
    tvm::ffi::TensorView counters, int64_t rank, int64_t region,
    int64_t stride, int64_t slice, int64_t blocks, bool unpack) {
  using namespace sglang::host;
  namespace ka = sglang::flash_ag_kernels_b300;
  namespace kc = ulacache_flashag::flash_ag_common_b300;
  RuntimeCheck(input.ndim() == 2 && output.numel() == input.numel() * World &&
               input.size(0) > 0 && input.size(1) > 0 &&
               input.dtype().bits == 32 && output.dtype() == input.dtype(), "Invalid FP32 score bounds");
  RuntimeCheck(staging.numel() >= input.numel() && packed.numel() >= output.numel(), "Packed workspace too small");
  RuntimeCheck(input.size(1) % 4 == 0 && prefix.numel() == input.size(0) + 1 &&
               prefix.dtype().bits == 32 && packs.numel() == 1 && packs.dtype().bits == 64,
               "Invalid aligned GPU descriptors");
  RuntimeCheck(slab_peers.numel() == World &&
               rank >= 0 && rank < World, "Invalid shared collective state");
  RuntimeCheck(blocks > 0 && blocks * 256 <= kc::kMaxSpinThreads && counters.numel() >= blocks * 8,
               "Invalid co-resident FlashAG grid/counters");
  RuntimeCheck(region >= 0 && stride >= World * slice && slice > 0 &&
               (region + kc::kNumSlots * stride) * 16 <= slab.numel() * 4,
               "FlashAG slot region exceeds exported slab");
  auto device = LaunchKernel::resolve_device(input.device());
  // Each CTA copies one contiguous query tile. GPU prefixes determine live
  // bounds; no per-vector binary search or dynamic 64-bit division is needed.
  const int copy_threads = 256;
  const int copy_tiles = (input.size(1) / 4 + copy_threads - 1) / copy_threads;
  LaunchKernel(dim3(copy_tiles, input.size(0)), dim3(copy_threads), device).enable_pdl(true)(pack_scores,
      reinterpret_cast<const uint4*>(input.data_ptr()), reinterpret_cast<uint4*>(staging.data_ptr()),
      static_cast<const int32_t*>(prefix.data_ptr()), static_cast<const int64_t*>(packs.data_ptr()),
      int(input.size(0)), int(input.size(1)));
  LaunchKernel(dim3(blocks), dim3(256), device).enable_pdl(true)(
      ka::k_ag_mcst<kc::PkF32, World, ka::kModeD31, ka::kSendUni, 1,
                   ka::kPollDense, false, false, false, true>,
      reinterpret_cast<const uint4*>(staging.data_ptr()), reinterpret_cast<uint4*>(packed.data_ptr()),
      reinterpret_cast<uint4*>(slab.data_ptr()), static_cast<uint4*>(nullptr),
      reinterpret_cast<void* const*>(slab_peers.data_ptr()), int(rank), size_t(output.numel() / 4),
      static_cast<uint32_t*>(counters.data_ptr()), size_t(region), size_t(stride), size_t(slice),
      static_cast<const int64_t*>(packs.data_ptr()));
  if (unpack) {
    LaunchKernel(dim3(copy_tiles, input.size(0), World), dim3(copy_threads), device).enable_pdl(true)(unpack_scores<World>,
        reinterpret_cast<const uint4*>(packed.data_ptr()), reinterpret_cast<uint4*>(output.data_ptr()),
        static_cast<const int32_t*>(prefix.data_ptr()), static_cast<const int64_t*>(packs.data_ptr()),
        int(input.size(0)), int(input.size(1)));
  }
}
}  // namespace udcache::index
