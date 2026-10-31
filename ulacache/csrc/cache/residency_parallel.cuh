// SPDX-License-Identifier: Apache-2.0
#pragma once
// Experimental owner-local preparation. Keep the published serial transaction
// unchanged; reuse its victim selector, counters, record plan and commit ABI.
#include "csrc/cache/residency_finish.cuh"

namespace sglang::flowkv_host_parallel {
namespace base = flowkv_host;
namespace recent = flowkv_ring_presence_recent;
namespace ring = flowkv_bitmap_ring;
namespace warp = flowkv_presence_warp;

__global__ void pin_rows(const void* rows, int row_bytes, int queries, int width,
    const int* real_queries, const int* selected_counts, bool use_counts,
    int owner, base::State s, base::Workspace w) {
  const auto context = recent::producer_context(
      *reinterpret_cast<const ring::Group*>(w.histogram), *s.epoch);
  const int count = queries * width;
  for (int first = blockIdx.x * blockDim.x; first < count;
       first += gridDim.x * blockDim.x) {
    const int i = first + threadIdx.x;
    const int query = i / width;
    const bool active = i < count && query < *real_queries &&
        (!use_counts || i % width < selected_counts[query]);
    const int row = active ? base::local_row(
        base::read_row(rows, i, row_bytes), owner, s.logical_slots) : -1;
    const int64_t location = row >= 0 ? s.directory[row] : -1;
    if (row >= 0) assert(location != -1);
    const int slot = row >= 0 && location >= 0 ? int(location) : -1;
    if (slot >= 0) assert(slot < s.gpu_slots && s.gpu_tags[slot] == row);
    // Every lane participates, including partial tails, padding and CPU rows.
    // The original inverse-CAS claim still deduplicates repeated query IDs.
    recent::mark_warp(context, slot, ring::Operation::Access);
    if (row < 0) continue;
    auto time = cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.access_time[row]);
    if (location >= 0) {
      if (time.load(cuda::memory_order_relaxed) != *s.epoch)
        time.store(*s.epoch, cuda::memory_order_relaxed);
    } else if (time.load(cuda::memory_order_relaxed) != *s.epoch &&
               time.exchange(*s.epoch, cuda::memory_order_relaxed) != *s.epoch) {
      const int out = atomicAdd(w.counters + base::PromoteCount, 1);
      assert(out < w.capacity);
      w.promote_rows[out] = row;
    }
  }
}

} // namespace sglang::flowkv_host_parallel

namespace sglang {
void flowkv_host_prepare_parallel(TensorView selected, TensorView current,
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
  flowkv_host::begin_current<<<1, 256, 0, stream>>>(
      current.data_ptr(), current.dtype().bits / 8, queries, active, owner, s, w);
  if (queries && width)
    flowkv_host_parallel::pin_rows<<<std::min(128, (queries * width + 255) / 256), 256, 0, stream>>>(
        selected.data_ptr(), selected.dtype().bits / 8, queries, width, active,
        static_cast<const int*>(selected_counts.data_ptr()), use_counts, owner, s, w);
#if UDCACHE_PARALLEL_VICTIMS
  constexpr bool parallel_victims = true;
#else
  constexpr bool parallel_victims = false;
#endif
  flowkv_host_parallel::finish_metadata(stream, s, w, queries, parallel_victims,
      flowkv_host_parallel::PlanDiagnostic{nullptr, nullptr, 0, 0, int(owner), -1});
}
} // namespace sglang
