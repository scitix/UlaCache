#pragma once
#include <cuda/atomic>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

namespace udcache::index {

using Flag = unsigned long long;
using SystemFlag = cuda::atomic_ref<Flag, cuda::thread_scope_system>;
constexpr int kCompactBlocks = 8;
constexpr int kCompactThreads = 256;

__device__ inline void wait_epoch(Flag* pointer, Flag epoch) {
  SystemFlag flag(*pointer);
  while (flag.load(cuda::memory_order_acquire) < epoch) __nanosleep(64);
}

template<int World>
__global__ void compact_ipc_pack(const uint4* input, uint4* staging,
    const int32_t* prefix, const int64_t* packs, Flag* flags,
    const uint64_t* flag_peers, int queries, int columns) {
  sglang::device::PDLWaitPrimary<true>();
  const Flag epoch = flags[0] + 1;
  if (threadIdx.x == 0) {
    // Every previous reader retires before any CTA overwrites exported scores.
    for (int peer = 0; peer < World; ++peer)
      wait_epoch(reinterpret_cast<Flag*>(flag_peers[peer]) + 2, epoch - 1);
  }
  __syncthreads();
  const int64_t count = *packs;
  assert(count >= 0 && count <= int64_t(queries) * columns / 4);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    int low = 0, high = queries;
    while (low + 1 < high) {
      const int middle = (low + high) / 2;
      if (prefix[middle] / 4 <= i) low = middle; else high = middle;
    }
    const int64_t source = int64_t(low) * (columns / 4) + i - prefix[low] / 4;
    assert(prefix[low] % 4 == 0 && source < int64_t(queries) * columns / 4);
    staging[i] = input[source];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    cuda::atomic_ref<Flag, cuda::thread_scope_device> finished(flags[3]);
    if (finished.fetch_add(1, cuda::memory_order_acq_rel) == gridDim.x - 1) {
      flags[3] = 0;
      flags[0] = epoch;
      flags[5] = count;
      SystemFlag(flags[1]).store(epoch, cuda::memory_order_release);
    }
  }
  sglang::device::PDLTriggerSecondary<true>();
}

template<int World>
__global__ void compact_ipc_pull(uint4* output, const int64_t* packs,
    Flag* flags, const uint64_t* score_peers, const uint64_t* flag_peers) {
  sglang::device::PDLWaitPrimary<true>();
  const Flag epoch = flags[0];
  const int64_t count = *packs;
  if (threadIdx.x == 0) {
    for (int peer = 0; peer < World; ++peer) {
      auto* remote = reinterpret_cast<Flag*>(flag_peers[peer]);
      wait_epoch(remote + 1, epoch);
      assert(remote[5] == count);
    }
  }
  sglang::device::PDLTriggerSecondary<true>();
  __syncthreads();
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count * World;
       i += gridDim.x * blockDim.x) {
    const int peer = i / count;
    const int64_t offset = i % count;
    output[i] = reinterpret_cast<const uint4*>(score_peers[peer])[offset];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    cuda::atomic_ref<Flag, cuda::thread_scope_device> finished(flags[4]);
    if (finished.fetch_add(1, cuda::memory_order_acq_rel) == gridDim.x - 1) {
      flags[4] = 0;
      SystemFlag(flags[2]).store(epoch, cuda::memory_order_release);
    }
  }
}

template<int World>
void compact_ipc_gather(tvm::ffi::TensorView input, tvm::ffi::TensorView output,
    tvm::ffi::TensorView prefix, tvm::ffi::TensorView packs,
    tvm::ffi::TensorView staging, tvm::ffi::TensorView flags,
    tvm::ffi::TensorView score_peers, tvm::ffi::TensorView flag_peers, int64_t rank) {
  using namespace sglang::host;
  RuntimeCheck(input.ndim() == 2 && output.numel() == input.numel() * World,
               "Invalid compact IPC score bounds");
  RuntimeCheck(staging.numel() >= input.numel() && flags.numel() == 8,
               "Invalid compact IPC staging/flags");
  RuntimeCheck(prefix.numel() == input.size(0) + 1 && packs.numel() == 1,
               "Invalid compact IPC device descriptors");
  RuntimeCheck(score_peers.numel() == World && flag_peers.numel() == World &&
               rank >= 0 && rank < World, "Invalid compact IPC peer table");
  auto device = LaunchKernel::resolve_device(input.device());
  auto* state = reinterpret_cast<Flag*>(flags.data_ptr());
  LaunchKernel(dim3(kCompactBlocks), dim3(kCompactThreads), device).enable_pdl(true)(compact_ipc_pack<World>,
      reinterpret_cast<const uint4*>(input.data_ptr()), reinterpret_cast<uint4*>(staging.data_ptr()),
      static_cast<const int32_t*>(prefix.data_ptr()), static_cast<const int64_t*>(packs.data_ptr()),
      state, static_cast<const uint64_t*>(flag_peers.data_ptr()), int(input.size(0)), int(input.size(1)));
  LaunchKernel(dim3(kCompactBlocks), dim3(kCompactThreads), device).enable_pdl(true)(compact_ipc_pull<World>,
      reinterpret_cast<uint4*>(output.data_ptr()), static_cast<const int64_t*>(packs.data_ptr()),
      state, static_cast<const uint64_t*>(score_peers.data_ptr()),
      static_cast<const uint64_t*>(flag_peers.data_ptr()));
}

}  // namespace udcache::index
