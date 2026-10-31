// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "csrc/cache/residency_fused.cuh"
#include "csrc/cache/residency_parallel.cuh"

namespace sglang::flowkv_host {
__global__ void begin_native_domains(const void* current, int bytes, int queries,
    const int* active, const int64_t* frames) {
  __shared__ recent::Storage<256> storage;
  const Frame frame{frames + int64_t(blockIdx.x) * kFrameWords};
  assert(frame.d[FPeers] == 1 && frame.d[FRank] == 0);
  // Exactly the original transaction, including CPU-current promotion and
  // fresh-current copy suppression. Only independent domains run together.
  begin_current_body(current, bytes, queries, active, 0,
      frame.state(), frame.work(), storage);
}
}
namespace sglang {
void flowkv_begin_native_domains(TensorView current, TensorView active, TensorView frames) {
  TVM_FFI_ICHECK(current.ndim() == 1 && (current.dtype().bits == 32 || current.dtype().bits == 64));
  TVM_FFI_ICHECK(active.numel() == 1 && active.dtype().bits == 32);
  TVM_FFI_ICHECK(frames.ndim() == 2 && frames.dtype().bits == 64 &&
      frames.size(1) == flowkv_host::kFrameWords);
  const auto stream = host::LaunchKernel::resolve_device(frames.device());
  flowkv_host::begin_native_domains<<<int(frames.size(0)), 256, 0, stream>>>(
      current.data_ptr(), current.dtype().bits / 8, int(current.numel()),
      static_cast<const int*>(active.data_ptr()), static_cast<const int64_t*>(frames.data_ptr()));
  host::RuntimeDeviceCheck(cudaGetLastError());
}
void flowkv_host_prepare_after_begin(TensorView selected, TensorView current,
    TensorView real_queries, TensorView selected_counts, bool use_counts, int64_t owner,
    TensorView directory, TensorView gpu_tags, TensorView host_tags,
    TensorView access_time, TensorView leases, TensorView epoch,
    TensorView gpu_free, TensorView host_free, TensorView free_counts,
    TensorView new_rows, TensorView promote_rows, TensorView plan, TensorView counters,
    TensorView histogram, TensorView prefix, TensorView candidates, TensorView leased_slots) {
  const auto stream = host::LaunchKernel::resolve_device(directory.device());
  const flowkv_host::State s{
    static_cast<int64_t*>(directory.data_ptr()), static_cast<int*>(gpu_tags.data_ptr()),
    static_cast<int*>(host_tags.data_ptr()), static_cast<int64_t*>(access_time.data_ptr()),
    static_cast<int*>(leases.data_ptr()), static_cast<int64_t*>(epoch.data_ptr()),
    static_cast<int*>(gpu_free.data_ptr()), static_cast<int*>(host_free.data_ptr()),
    static_cast<int*>(free_counts.data_ptr()), int(directory.shape()[0]),
    int(gpu_tags.shape()[0]), int(host_tags.shape()[0])};
  const flowkv_host::Workspace w{
    static_cast<int*>(new_rows.data_ptr()), static_cast<int*>(promote_rows.data_ptr()),
    static_cast<int*>(plan.data_ptr()), static_cast<int*>(counters.data_ptr()),
    static_cast<int*>(histogram.data_ptr()), static_cast<int64_t*>(prefix.data_ptr()),
    static_cast<int*>(candidates.data_ptr()), static_cast<int*>(leased_slots.data_ptr()),
    int(new_rows.shape()[0])};
  const int queries = int(selected.shape()[0]), width = int(selected.shape()[1]);
  const auto* active = static_cast<const int*>(real_queries.data_ptr());
  if (queries && width)
    flowkv_host_parallel::pin_rows<<<std::min(128, (queries * width + 255) / 256), 256, 0, stream>>>(
        selected.data_ptr(), selected.dtype().bits / 8, queries, width, active,
        static_cast<const int*>(selected_counts.data_ptr()), use_counts, owner, s, w);
#if UDCACHE_PARALLEL_VICTIMS
  constexpr bool parallel_victims = true;
#else
  constexpr bool parallel_victims = false;
#endif
  flowkv_host_parallel::finish_metadata(stream, s, w, queries, parallel_victims);
}
} // namespace sglang
