#pragma once

// Private bounded recent/cold presence ABI. Native logical time, tags, and
// group epoch retain the original State9 pointers. No pager or count tree.
#include "csrc/eviction/presence_bitmap.cuh"
#include <stddef.h>
#include <stdint.h>

#ifndef UDCACHE_NAIVE_LRU
#define UDCACHE_NAIVE_LRU 0
#endif

namespace flowkv_bitmap_ring {
using BucketView = flowkv_presence_bitmap::View;
constexpr int32_t kFree = -1;
constexpr int32_t kCold = -2;
// Compatibility encodings only: this route generates no pending inverse IDs.
__host__ __device__ inline int32_t pending_recent(int32_t bucket) { return -3 - bucket; }
__host__ __device__ inline int32_t pending_cold(int32_t recent_count) { return -3 - recent_count; }

enum class Operation : int32_t { Access = 0, Detach = 1 };

struct Work {
  int32_t* expire_slots;  // Native leased_slots[:U]; expiry batches retire before take.
  int32_t* victims;       // Native candidates[:U]; retained through original assign.
  int32_t* incoming;      // Native plan.flat[:U]; retired before assign overwrites it.
  int32_t capacity;       // U>0, U<=x; native F+K<=U.
};

struct Group {
  BucketView* buckets;     // [R+1], cold at R; every View has the same capacity x.
  int64_t* bucket_epochs;  // [R], recent ID=age modulo R.
  int64_t* epoch;          // Existing monotonic group E, independent of arena.step.
  int32_t* gpu_bucket;     // [x], completed state FREE/COLD/recent only.
  const int32_t* gpu_tags; // Existing physical GPU slot -> logical identity.
  int64_t* access_time;    // Existing sole logical time; membership precedes dedup.
  int32_t recent_count;   // Default total256 gives recent255+cold1.
  int32_t logical_capacity;
  Work work;
#if UDCACHE_NAIVE_LRU
  int32_t* naive_histogram; // 256 GPU radix bins; no presence buckets maintained.
#endif
};

struct Transaction {
  Operation operation;
  int64_t epoch;
  const int32_t* slots;  // GPU touches/detaches; duplicate and -1 entries permitted.
  int32_t slot_count;
  const int32_t* free_targets;
  int32_t free_count;
  int32_t victim_count;  // Native CPU promotion P=F+K, no separate CPU policy.
};

// Presence consumers run only after all producer kernels have completed.
// summary[sum(lengths)] is the sole exact total; earlier words are nonempty bits.
static_assert(sizeof(BucketView) == 88);
static_assert(sizeof(Work) == 32);
static_assert(sizeof(Group) == (UDCACHE_NAIVE_LRU ? 96 : 88) && offsetof(Group, work) == 56);
static_assert(sizeof(Transaction) == 48 && offsetof(Transaction, epoch) == 8);
} // namespace flowkv_bitmap_ring
