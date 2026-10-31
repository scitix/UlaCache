// SPDX-License-Identifier: Apache-2.0
#pragma once
// Shared owner-local finish after every residency producer has retired.
// TP retains its original owner/IPC and child-graph publication boundaries;
// DPA supplies its existing native begin/pin producers. No epoch is advanced
// here, and no incoming slot aliases plan[0]. Selection retires before the
// unique parallel assignments update ring membership and write the same plan.
#include "csrc/cache/cpu_residency.cuh"
#include "csrc/eviction/victims.cuh"

namespace sglang::flowkv_host_parallel {
namespace base = flowkv_host;
namespace recent = flowkv_ring_presence_recent;
namespace ring = flowkv_bitmap_ring;
namespace warp = flowkv_presence_warp;

__global__ void assign_parallel(base::State s, base::Workspace w) {
  const int count = w.counters[base::PlanCount];
  const int free_used = w.counters[base::FreeUsed];
#if !UDCACHE_NAIVE_LRU
  const auto& group = *reinterpret_cast<const ring::Group*>(w.histogram);
  const int current = int(*s.epoch % group.recent_count);
#endif
  for (int first = blockIdx.x * blockDim.x; first < count;
       first += gridDim.x * blockDim.x) {
    const int i = first + threadIdx.x;
    const bool active = i < count;
    int gpu = -1, source = -1;
    if (active) {
      gpu = i < free_used ? s.gpu_free[s.free_counts[0] + i]
                         : w.candidates[i - free_used];
#if !UDCACHE_NAIVE_LRU
      const int expected = i < free_used ? ring::kFree : group.gpu_bucket[gpu];
      assert(i < free_used || expected == ring::kCold ||
          (expected >= 0 && expected < group.recent_count && expected != current));
      source = expected == ring::kCold ? group.recent_count : expected;
      assert(atomicCAS(group.gpu_bucket + gpu, expected, current) == expected);
#endif
    }
    // Source views are delete-only; the current view is insert-only. The
    // existing unique-bit hierarchy supports independent producer CTAs.
#if !UDCACHE_NAIVE_LRU
    warp::transfer_warp_unique(group.buckets, group.recent_count + 1,
                              source, current, gpu);
#endif
    if (!active) continue;
    base::assign_plan_item(s, w, i, gpu);
  }
}
inline void finish_metadata(cudaStream_t stream, base::State s,
    base::Workspace w, int queries, bool parallel_victims,
    PlanDiagnostic diagnostic = {}) {
#if UDCACHE_NAIVE_LRU
  base::naive_finish_select(stream, s, w);
#else
  // Native DPA begin/current and selected-row producers have already retired
  // on this stream. For small static shapes, reuse their serial reference
  // finish: its empty-plan path retains reservation/lease counters without
  // launching a capacity-sized parallel assignment wave. Nonempty plans keep
  // the same reference victim/ring/record transaction. No CPU read of the
  // dynamic plan, additional epoch, payload copy or readiness boundary exists.
  if (queries > 0 && queries < 8) {
    base::metadata_finish<<<1, 256, 0, stream>>>(s, w);
    host::RuntimeDeviceCheck(cudaGetLastError());
    return;
  }
  // Preserve the existing launch policy for the remaining static Graph shapes.
  if (parallel_victims && queries >= 8) {
    plan_victim_buckets<<<1, 256, 0, stream>>>(s, w, diagnostic);
    select_victim_buckets<<<128, 256, 0, stream>>>(s, w);
  } else {
    select_victims<<<1, 256, 0, stream>>>(s, w, diagnostic);
  }
#endif
  assign_parallel<<<std::min(128, (w.capacity + 255) / 256), 256, 0, stream>>>(s, w);
  host::RuntimeDeviceCheck(cudaGetLastError());
}
} // namespace sglang::flowkv_host_parallel
