#pragma once
// Same independent recent-ring + unified-cold policy. Presence words replace
// population trees; no pager/position/CSR/whole-residency scan is introduced.
#include "csrc/eviction/presence_types.cuh"
#include "csrc/eviction/presence_warp.cuh"
#include <cub/block/block_scan.cuh>
#ifndef FLOWKV_CPU_EXECUTION
#include <cuda/atomic>
#endif
namespace flowkv_ring_presence_recent {
namespace rt = flowkv_bitmap_ring;
namespace pb = flowkv_presence_bitmap;
namespace pw = flowkv_presence_warp;
struct Descriptor { int32_t bucket, quota, base; uint32_t reserved; };
template<int NT = 256> struct Storage {
  pb::Storage<NT> take;
  typename cub::BlockScan<int32_t, NT>::TempStorage scan;
  Descriptor descriptors[NT];
  int32_t descriptor_count;
};
__device__ inline int capacity(const rt::Group& g) { return g.buckets[g.recent_count].capacity; }
__device__ inline void store_time(int64_t* p, int64_t E) {
#ifdef FLOWKV_CPU_EXECUTION
  __atomic_store_n(p, E, __ATOMIC_SEQ_CST);
#else
  cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*p).store(E, cuda::memory_order_relaxed);
#endif
}
// Read immutable producer geometry/epoch once, outside each slot claim. The
// runtime can construct this after original begin and pass it to scalar pins.
struct ProducerContext {
  rt::BucketView* views;
  int32_t* inverse;
  const int32_t* tags;
  int64_t* time;
  int32_t x, R, current, logical_capacity;
  int64_t E;
};
__device__ inline ProducerContext producer_context(const rt::Group& g, int64_t E) {
#if UDCACHE_NAIVE_LRU
  assert(E >= 0 && E == *g.epoch);
  return {nullptr, nullptr, g.gpu_tags, g.access_time, capacity(g), 0, 0,
      g.logical_capacity, E};
#else
  assert(g.recent_count > 0 && E >= 0 && E == *g.epoch);
  const int current = int(E % g.recent_count);
  assert(g.bucket_epochs[current] == E);
  return {g.buckets, g.gpu_bucket, g.gpu_tags, g.access_time, capacity(g),
      g.recent_count, current, g.logical_capacity, E};
#endif
}
__device__ inline bool claim(const ProducerContext& c, int slot, rt::Operation operation,
    int& source, bool stamp_time = false) {
  assert(slot >= 0 && slot < c.x);
#if UDCACHE_NAIVE_LRU
  if (stamp_time && operation == rt::Operation::Access) {
    const int logical = c.tags[slot];
    assert(logical >= 0 && logical < c.logical_capacity);
    store_time(c.time + logical, c.E);
  }
  return false; // No bucket membership exists under timestamp recency.
#else
#ifdef FLOWKV_CPU_EXECUTION
  const int old = __atomic_load_n(c.inverse + slot, __ATOMIC_RELAXED);
#else
  const int old = cuda::atomic_ref<int32_t, cuda::thread_scope_device>(c.inverse[slot]).load(cuda::memory_order_relaxed);
#endif
  if (old == rt::kFree) { assert(operation == rt::Operation::Detach); return false; }
  assert(old == rt::kCold || (old >= 0 && old < c.R));
  if (stamp_time && operation == rt::Operation::Access) {
    const int logical = c.tags[slot]; assert(logical >= 0 && logical < c.logical_capacity);
    store_time(c.time + logical, c.E);
  }
  if (operation == rt::Operation::Access && old == c.current) return false;
  const int desired = operation == rt::Operation::Access ? c.current : rt::kFree;
  if (atomicCAS(c.inverse + slot, old, desired) != old) return false;
  source = old == rt::kCold ? c.R : old; return true;
#endif
}
__device__ inline bool mark_slot(const ProducerContext& c, int slot, rt::Operation operation,
    bool stamp_time = false) {
  int source = -1; if (!claim(c, slot, operation, source, stamp_time)) return false;
  pb::erase_mask_unique(c.views[source], slot >> 5, uint32_t(1) << (slot & 31));
  if (operation == rt::Operation::Access)
    pb::insert_mask_unique(c.views[c.current], slot >> 5, uint32_t(1) << (slot & 31));
  return true;
}
__device__ inline bool mark_slot(const rt::Group& g, int slot, rt::Operation operation,
    int64_t E, bool stamp_time = false) { return mark_slot(producer_context(g, E), slot, operation, stamp_time); }
__device__ inline void mark_warp(const ProducerContext& c, int slot, rt::Operation operation,
    bool stamp_time = false) {
#if UDCACHE_NAIVE_LRU
  int ignored;
  if (slot >= 0) claim(c, slot, operation, ignored, stamp_time);
#else
  int source = -1; const bool winner = slot >= 0 && claim(c, slot, operation, source, stamp_time);
  pw::change_warp_unique(c.views, c.R + 1, source, winner ? slot : -1, false);
  if (operation == rt::Operation::Access)
    pw::change_warp_unique(c.views[c.current], winner ? slot : -1, true);
#endif
}
__device__ inline void mark_warp(const rt::Group& g, int slot, rt::Operation operation,
    int64_t E, bool stamp_time = false) { mark_warp(producer_context(g, E), slot, operation, stamp_time); }
__device__ inline void mark(const rt::Group& g, const rt::Transaction& tx, bool stamp_time = false) {
  assert(tx.slot_count >= 0);
  const int64_t first = int64_t(blockIdx.x) * blockDim.x;
  // Empty CTAs touch neither Group.epoch nor its dependent native epoch value.
  if (first >= tx.slot_count) return;
  const auto c = producer_context(g, tx.epoch);
  for (int64_t base = first; base < tx.slot_count; base += int64_t(blockDim.x) * gridDim.x) {
    const int64_t item = base + threadIdx.x;
    mark_warp(c, item < tx.slot_count ? tx.slots[item] : -1, tx.operation, stamp_time);
  }
}

template<int NT> __device__ uint32_t take_single(const rt::BucketView& view,
    uint32_t requested, int32_t* out, uint32_t out_capacity, Storage<NT>& s) {
  const uint32_t take = pb::select_completed_cta<NT>(view, requested, out, out_capacity, s.take);
  for (uint32_t base = 0; base < take; base += NT) {
    const uint32_t item = base + threadIdx.x;
    pw::change_warp_unique(view, item < take ? out[item] : -1, false);
  }
  __syncthreads(); return take;
}
template<int NT = 256> __device__ void advance(const rt::Group& g, int64_t E, Storage<NT>& s) {
#if UDCACHE_NAIVE_LRU
  // Canonical epoch advancement remains in the original begin producer.
  return;
#else
  assert(E >= 0 && E >= *g.epoch && g.recent_count > 0);
  assert(g.work.capacity > 0 && g.work.capacity <= capacity(g));
  const int64_t gap = E - *g.epoch;
  const int span = gap < g.recent_count ? int(gap) : g.recent_count;
  const int64_t first = span ? E - span + 1 : E;
  for (int i = 0; i < span; ++i) {
    const int bid = int((first + i) % g.recent_count);
    const int64_t age = g.bucket_epochs[bid];
    assert(age == -1 || (age >= 0 && age <= E));
    __syncthreads(); // all lanes capture age before thread 0 retags the header
    if (age < 0 || E - age < g.recent_count) continue;
    const auto& old = g.buckets[bid];
    for (;;) {
      const uint32_t n = pb::count_completed(old); if (!n) break;
      const uint32_t chunk = n < uint32_t(g.work.capacity) ? n : uint32_t(g.work.capacity);
      const uint32_t taken = take_single<NT>(old, chunk, g.work.expire_slots, uint32_t(g.work.capacity), s);
      assert(taken == chunk);
      for (uint32_t base = 0; base < taken; base += NT) {
        const uint32_t item = base + threadIdx.x;
        const int slot = item < taken ? g.work.expire_slots[item] : -1;
        if (slot >= 0) assert(g.gpu_bucket[slot] == bid);
        pw::change_warp_unique(g.buckets[g.recent_count], slot, true);
        if (slot >= 0) g.gpu_bucket[slot] = rt::kCold;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) g.bucket_epochs[bid] = -1;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int current = int(E % g.recent_count);
    assert(g.bucket_epochs[current] == E || pb::count_completed(g.buckets[current]) == 0);
    g.bucket_epochs[current] = E;
  }
  __syncthreads();
#endif
}
template<int NT = 256> __device__ void begin(const rt::Group& g, const rt::Transaction& tx, Storage<NT>& s) {
  advance<NT>(g, tx.epoch, s);
  if (threadIdx.x == 0) *g.epoch = tx.epoch;
  __syncthreads();
}

template<int NT> __device__ void enumerate_descriptors(const rt::Group& g, Storage<NT>& s) {
  const int m = s.descriptor_count;
#ifdef FLOWKV_CPU_EXECUTION
  if (threadIdx.x == 0) for (int d = 0; d < m; ++d) {
    auto& desc = s.descriptors[d];
    desc.reserved = pb::select_host_completed(g.buckets[desc.bucket], uint32_t(desc.quota),
        g.work.victims + desc.base, uint32_t(g.work.capacity - desc.base), NT / 32);
  }
#else
  constexpr int W = NT / 32; const int warp = int(threadIdx.x) / 32;
  if (m <= W) {
    if (m) {
      const int d = warp % m, part = warp / m, parts = (W - 1 - d) / m + 1;
      auto& desc = s.descriptors[d];
      pb::select_warp_completed(g.buckets[desc.bucket], uint32_t(desc.quota), g.work.victims + desc.base,
          &desc.reserved, s.take.paths[warp], part, parts);
    }
  } else for (int d = warp; d < m; d += W) {
    auto& desc = s.descriptors[d];
    pb::select_warp_completed(g.buckets[desc.bucket], uint32_t(desc.quota), g.work.victims + desc.base,
        &desc.reserved, s.take.paths[warp], 0, 1);
  }
#endif
  __syncthreads();
  if (threadIdx.x < m) assert(s.descriptors[threadIdx.x].reserved >= uint32_t(s.descriptors[threadIdx.x].quota));
}
template<int NT = 256> __device__ void enumerate_victims(const rt::Group& g, int64_t E, int K, Storage<NT>& s) {
  assert(E == *g.epoch && K >= 0 && K <= g.work.capacity);
  if (!K) return;
  const int R = g.recent_count, current = int(E % R);
  const uint32_t cold = pb::count_completed(g.buckets[R]);
  int emitted = 0;
  if (cold >= uint32_t(K)) {
    if (threadIdx.x == 0) { s.descriptors[0] = {R, K, 0, 0}; s.descriptor_count = 1; }
    __syncthreads(); enumerate_descriptors<NT>(g, s); emitted = K;
  } else for (int64_t base = 0; base < R && emitted < K; base += NT) {
    const int64_t order = base + threadIdx.x;
    const int bid = order == 0 ? R : int((int64_t(current) + order) % R);
    const int count = order < R ? int(pb::count_completed(g.buckets[bid])) : 0;
    int prefix, total;
    cub::BlockScan<int32_t, NT>(s.scan).ExclusiveSum(count, prefix, total);
    const int need = K - emitted;
    const int quota = prefix < need ? (count < need - prefix ? count : need - prefix) : 0;
    const int out_base = emitted + prefix;
    if (quota && bid < R) assert(g.bucket_epochs[bid] >= 0 && g.bucket_epochs[bid] < E);
    __syncthreads(); // retire the first CUB scan before flag compaction
    int dest, descriptors;
    cub::BlockScan<int32_t, NT>(s.scan).ExclusiveSum(int(quota != 0), dest, descriptors);
    if (quota) s.descriptors[dest] = {bid, quota, out_base, 0};
    if (threadIdx.x == 0) s.descriptor_count = descriptors;
    __syncthreads();
    enumerate_descriptors<NT>(g, s);
    emitted += total < need ? total : need;
    __syncthreads(); // descriptor/prefix storage is retired before next chunk
  }
  assert(emitted == K);
}
template<int NT = 256> __device__ void take_victims(const rt::Group& g, int64_t E, int K, Storage<NT>& s) {
  // Preserve the public destructive take contract; finish alone transfers.
  enumerate_victims<NT>(g, E, K, s);
  if (!K) return;
  const int R = g.recent_count, current = int(E % R);
  // Every output is complete before any source View or inverse is modified.
  for (int64_t base = 0; base < K; base += NT) {
    const int64_t item = base + threadIdx.x;
    const int slot = item < K ? g.work.victims[item] : -1;
    int source = -1;
    if (slot >= 0) {
      assert(slot < capacity(g)); const int old = g.gpu_bucket[slot];
      assert(old == rt::kCold || (old >= 0 && old < R && old != current));
      source = old == rt::kCold ? R : old;
    }
    pw::change_warp_unique(g.buckets, R + 1, source, slot, false);
    if (slot >= 0) g.gpu_bucket[slot] = rt::kFree;
  }
  __syncthreads();
}
template<int NT = 256> __device__ void finish(const rt::Group& g, const rt::Transaction& tx, Storage<NT>& s) {
  if (tx.operation == rt::Operation::Detach) { assert(tx.free_count == 0 && tx.victim_count == 0); return; }
  assert(tx.free_count >= 0 && tx.victim_count >= 0);
  const int64_t P = int64_t(tx.free_count) + tx.victim_count; assert(P <= g.work.capacity);
  const int current = int(tx.epoch % g.recent_count);
  assert(g.bucket_epochs[current] == tx.epoch && tx.epoch == *g.epoch);
  assert(uint64_t(pb::count_completed(g.buckets[current])) + uint64_t(P) <= uint32_t(capacity(g)));
  // K=0 skips the selector completion barrier: retire all prologue reads
  // before any incoming warp can modify the current bucket total.
  if (tx.victim_count == 0 && P != 0) __syncthreads();
  // All quotas and victim outputs retire before either bitmap can change.
  enumerate_victims<NT>(g, tx.epoch, tx.victim_count, s);
  for (int64_t base = 0; base < P; base += NT) {
    const int64_t item = base + threadIdx.x; int slot = -1, source = -1;
    if (item < P) {
      slot = item < tx.free_count ? tx.free_targets[item] : g.work.victims[item - tx.free_count];
      assert(slot >= 0 && slot < capacity(g));
      int expected = rt::kFree;
      if (item >= tx.free_count) {
        expected = g.gpu_bucket[slot];
        assert(expected == rt::kCold || (expected >= 0 && expected < g.recent_count && expected != current));
        source = expected == rt::kCold ? g.recent_count : expected;
      }
      // The uniquely selected victim has no externally observed FREE phase.
      assert(atomicCAS(g.gpu_bucket + slot, expected, current) == expected);
      g.work.incoming[item] = slot;
    }
    pw::transfer_warp_unique(g.buckets, g.recent_count + 1, source, current, slot);
  }
  __syncthreads();
}
template<int NT = 256> __device__ void attach_native(const rt::Group& g, int64_t age,
    const int32_t* slots, int count, bool stamp_time = false) {
  assert(age >= 0 && age <= *g.epoch && count >= 0 && count <= capacity(g));
  if (!count) return;
  const bool cold = *g.epoch - age >= g.recent_count;
  const int bid = cold ? g.recent_count : int(age % g.recent_count);
  if (!cold && threadIdx.x == 0) {
    assert(g.bucket_epochs[bid] == age || pb::count_completed(g.buckets[bid]) == 0);
    g.bucket_epochs[bid] = age;
  }
  __syncthreads();
  for (int64_t base = 0; base < count; base += NT) {
    const int64_t item = base + threadIdx.x; int slot = -1;
    if (item < count) {
      slot = slots[item]; assert(slot >= 0 && slot < capacity(g));
      assert(atomicCAS(g.gpu_bucket + slot, rt::kFree, cold ? rt::kCold : bid) == rt::kFree);
      if (stamp_time) {
        const int logical = g.gpu_tags[slot]; assert(logical >= 0 && logical < g.logical_capacity);
        store_time(g.access_time + logical, age);
      }
    }
    pw::change_warp_unique(g.buckets[bid], slot, true);
  }
  __syncthreads();
}
} // namespace flowkv_ring_presence_recent
