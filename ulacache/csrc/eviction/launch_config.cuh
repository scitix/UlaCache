// SPDX-License-Identifier: Apache-2.0
#pragma once
// Launch geometry is the only experimental variable. All device bodies below
// are the existing PB implementation, including its original unordered ties.
#include "csrc/cache/residency_parallel.cuh"

namespace sglang {
void pb_launch_prepare(TensorView selected, TensorView current,
    TensorView real_queries, TensorView selected_counts, bool use_counts, int64_t owner,
    TensorView directory, TensorView gpu_tags, TensorView host_tags,
    TensorView access_time, TensorView leases, TensorView epoch,
    TensorView gpu_free, TensorView host_free, TensorView free_counts,
    TensorView new_rows, TensorView promote_rows, TensorView plan, TensorView counters,
    TensorView histogram, TensorView prefix, TensorView candidates, TensorView leased_slots,
    int64_t pin_blocks, int64_t pin_threads, int64_t victim_blocks,
    int64_t assignment_blocks, int64_t assignment_threads, bool small_query_parallel) {
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
  flowkv_host::begin_current<<<1, 256, 0, stream>>>(
      current.data_ptr(), current.dtype().bits / 8, queries, active, owner, s, w);
  if (queries && width) {
    const int blocks = std::min(int(pin_blocks),
        (queries * width + int(pin_threads) - 1) / int(pin_threads));
    flowkv_host_parallel::pin_rows<<<blocks, int(pin_threads), 0, stream>>>(
        selected.data_ptr(), selected.dtype().bits / 8, queries, width, active,
        static_cast<const int*>(selected_counts.data_ptr()), use_counts, owner, s, w);
  }
  // The bucket selector uses a 256-thread BlockScan and eight warps per CTA.
  // Its thread count and bucket partitioning remain fixed.
  if (queries >= 8 || small_query_parallel) {
    flowkv_host_parallel::plan_victim_buckets<<<1, 256, 0, stream>>>(s, w);
    flowkv_host_parallel::select_victim_buckets<<<int(victim_blocks), 256, 0, stream>>>(s, w);
  } else {
    flowkv_host_parallel::select_victims<<<1, 256, 0, stream>>>(s, w);
  }
  const int blocks = std::min(int(assignment_blocks),
      (w.capacity + int(assignment_threads) - 1) / int(assignment_threads));
  flowkv_host_parallel::assign_parallel<<<blocks, int(assignment_threads), 0, stream>>>(s, w);
  host::RuntimeDeviceCheck(cudaGetLastError());
}
}  // namespace sglang
