// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "csrc/cache/cpu_residency.cuh"

namespace sglang::flowkv_host_parallel {
namespace base = flowkv_host;
namespace recent = flowkv_ring_presence_recent;
namespace ring = flowkv_bitmap_ring;
namespace warp = flowkv_presence_warp;
}

namespace sglang::flowkv_host_parallel {
// Reuse existing ready-table identities only on the failure path. No scratch
// allocation or device-to-host synchronization is needed for diagnostics.
struct PlanDiagnostic {
  const uint64_t* ready_peers = nullptr;
  const int64_t* local_ready = nullptr;
  int peers = 0;
  int ready_offset = 0;
  int owner = -1;
  int group = -1;
};

__device__ __forceinline__ void check_plan_capacity(
    base::State s, base::Workspace w, const ring::Group& group,
    PlanDiagnostic diagnostic) {
  const int current = int(*s.epoch % group.recent_count);
  const uint32_t pinned = flowkv_presence_bitmap::count_completed(group.buckets[current]);
  const int planned = w.counters[base::PlanCount];
  const bool fits = uint64_t(pinned) + uint64_t(planned) <= uint32_t(s.gpu_slots);
  if (!fits) {
    if (threadIdx.x == 0) {
      int owner = diagnostic.owner;
      if (diagnostic.ready_peers && diagnostic.local_ready) {
        const auto address = reinterpret_cast<uint64_t>(diagnostic.local_ready);
        for (int peer = 0; peer < diagnostic.peers; ++peer)
          if (diagnostic.ready_peers[peer] + int64_t(diagnostic.ready_offset) * sizeof(int64_t) == address)
            owner = peer;
      }
      printf("FlowKV owner working-set overflow: owner=%d group=%d epoch=%lld "
             "pinned=%u plan=%d gpu_slots=%d new=%d promote=%d free_used=%d victims=%d\n",
             owner, diagnostic.group, static_cast<long long>(*s.epoch), pinned,
             planned, s.gpu_slots, w.counters[base::NewCount],
             w.counters[base::PromoteCount], w.counters[base::FreeUsed],
             w.counters[base::VictimCount]);
    }
    // Let the diagnostic producer finish before any lane triggers the assert.
    __syncthreads();
  }
  assert(fits);
}

__global__ void select_victims(base::State s, base::Workspace w, PlanDiagnostic diagnostic = {}) {
  __shared__ recent::Storage<256> storage;
  if (threadIdx.x == 0) {
    base::reserve_counts(s, w);
    w.counters[base::PendingReserve] = 0;
    w.counters[base::LeaseCount] = 0;
    w.prefix[0] = 0;
  }
  __syncthreads();
  const auto& group = *reinterpret_cast<const ring::Group*>(w.histogram);
  const int current = int(*s.epoch % group.recent_count);
  assert(group.bucket_epochs[current] == *s.epoch);
  check_plan_capacity(s, w, group, diagnostic);
  recent::enumerate_victims<256>(group, *s.epoch,
      w.counters[base::VictimCount], storage);
  if (threadIdx.x == 0)
    w.counters[base::Candidates] = w.counters[base::VictimCount];
}

// Prefix only bucket totals; never scan the resident slots or change ages.
// The original expiry scratch is dead until the next begin/release boundary.
// Small workspaces and non-default large ring geometries keep the CTA selector.
__device__ bool bucket_parallel_supported(const ring::Group& group,
                                          const base::Workspace& w) {
  return group.recent_count <= 255 &&
      min(w.capacity, recent::capacity(group)) >= 4 * group.recent_count;
}

__global__ void plan_victim_buckets(base::State s, base::Workspace w, PlanDiagnostic diagnostic = {}) {
  __shared__ recent::Storage<256> storage;
  const auto& group = *reinterpret_cast<const ring::Group*>(w.histogram);
  if (threadIdx.x == 0) {
    base::reserve_counts(s, w);
    w.counters[base::PendingReserve] = 0;
    w.counters[base::LeaseCount] = 0;
    w.prefix[0] = 0;
  }
  __syncthreads();
  const int current = int(*s.epoch % group.recent_count);
  assert(group.bucket_epochs[current] == *s.epoch);
  check_plan_capacity(s, w, group, diagnostic);
  // Reservation has published free-slot usage and reset bucket descriptors.
  // A free-only or empty plan has no old buckets to enumerate.
  if (w.counters[base::VictimCount] == 0) {
    if (threadIdx.x == 0) w.counters[base::Candidates] = 0;
    return;
  }
  if (!bucket_parallel_supported(group, w)) {
    recent::enumerate_victims<256>(group, *s.epoch,
        w.counters[base::VictimCount], storage);
  } else {
    const int order = threadIdx.x, R = group.recent_count;
    const int bid = order == 0 ? R : (current + order) % R;
    const int count = order < R
        ? int(flowkv_presence_bitmap::count_completed(group.buckets[bid])) : 0;
    int prefix, total;
    cub::BlockScan<int32_t, 256>(storage.scan).ExclusiveSum(count, prefix, total);
    assert(total >= w.counters[base::VictimCount]);
    const int remaining = w.counters[base::VictimCount] - prefix;
    const int quota = order < R && remaining > 0
        ? (count < remaining ? count : remaining) : 0;
    // Compact active buckets before assigning several CTAs to each bucket.
    // prefix[0:2] and leased[] are retired expiry scratch until lease/release.
    __syncthreads();
    int descriptor, descriptors;
    cub::BlockScan<int32_t, 256>(storage.scan).ExclusiveSum(
        int(quota != 0), descriptor, descriptors);
    if (quota) {
      if (quota && bid < R)
        assert(group.bucket_epochs[bid] >= 0 && group.bucket_epochs[bid] < *s.epoch);
      w.leased[4 * descriptor] = bid;
      w.leased[4 * descriptor + 1] = quota;
      w.leased[4 * descriptor + 2] = prefix;
      w.leased[4 * descriptor + 3] = 0;
    }
    if (threadIdx.x == 0) {
      w.prefix[0] = descriptors;
      w.prefix[1] = descriptors ? min(32, max(1, 128 / descriptors)) : 1;
    }
  }
  if (threadIdx.x == 0)
    w.counters[base::Candidates] = w.counters[base::VictimCount];
}

// One level-0 summary word covers 1024 resident slots. Partition those words
// directly: do not make every warp walk the same upper tree just to discard
// other warps' branches. Membership is stable throughout this consumer wave.
__device__ __forceinline__ void select_parent_words(
    const flowkv_presence_bitmap::View& view, uint32_t take, int* output,
    uint32_t* reserved, int part, int parts) {
  namespace pb = flowkv_presence_bitmap;
  constexpr uint32_t all = UINT32_MAX;
  const int lane = int(threadIdx.x) & 31;
  for (int parent = part; parent < view.lengths[0]; parent += parts) {
    // Different CTAs share this ticket counter; block-scope loads are invalid.
    uint32_t done = lane == 0
        ? cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*reserved)
              .load(cuda::memory_order_relaxed) : 0;
    if (__shfl_sync(all, done, 0) >= take) break;
    const uint32_t presence = view.summary[view.offsets[0] + parent];
    if (!presence) continue;
    const int leaf = parent * 32 + lane;
    const uint32_t bits = presence & (uint32_t(1) << lane) ? view.bits[leaf] : 0;
    assert(!(presence & (uint32_t(1) << lane)) ||
        (leaf < pb::leaf_words(view) && bits && !(bits & ~pb::valid_mask(view, leaf))));
    const uint32_t count = pb::pop(bits);
    uint32_t prefix = count;
    for (int delta = 1; delta < 32; delta <<= 1) {
      const uint32_t value = __shfl_up_sync(all, prefix, delta);
      if (lane >= delta) prefix += value;
    }
    const uint32_t total = __shfl_sync(all, prefix, 31);
    uint32_t base = lane == 0 ? atomicAdd(reserved, total) : 0;
    base = __shfl_sync(all, base, 0);
    if (base >= take) break;
    uint32_t mask = bits, dst = base + prefix - count;
    while (mask && dst < take) {
      const int bit = pb::first(mask);
      mask &= mask - 1;
      output[dst++] = leaf * 32 + bit;
    }
  }
}

__global__ void select_victim_buckets(base::State s, base::Workspace w) {
  const auto& group = *reinterpret_cast<const ring::Group*>(w.histogram);
  const int descriptors = int(w.prefix[0]);
  if (!descriptors) return; // Includes the unsupported-geometry CTA fallback.
  const int parts = int(w.prefix[1]);
  for (int task = blockIdx.x; task < descriptors * parts; task += gridDim.x) {
    const int descriptor = task / parts, part = task % parts;
    const int bid = w.leased[4 * descriptor];
    const int quota = w.leased[4 * descriptor + 1];
    const int base = w.leased[4 * descriptor + 2];
    auto* reserved = reinterpret_cast<uint32_t*>(w.leased + 4 * descriptor + 3);
    select_parent_words(group.buckets[bid], quota, w.candidates + base, reserved,
                        part * 8 + int(threadIdx.x) / 32, parts * 8);
  }
}


}  // namespace sglang::flowkv_host_parallel
