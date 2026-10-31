#pragma once
// Private presence summary. One mutation wave has one sign per View; consumers
// require the complete producer kernel. Counts are per bucket, never per node.
#include <cassert>
#include <climits>
#include <cstdint>
#include <type_traits>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda/atomic>
#define FPB_HD __host__ __device__ __forceinline__
#define FPB_D __device__ __forceinline__
#else
#include <atomic>
#define FPB_HD inline
#define FPB_D inline
#endif
namespace flowkv_presence_bitmap {
constexpr int kMaxLevels = 8;
struct View {
  uint32_t* bits;
  uint32_t* summary;
  int32_t offsets[kMaxLevels], lengths[kMaxLevels], levels, capacity;
};
static_assert(sizeof(View) == 88 && std::is_trivially_copyable_v<View>);
FPB_HD int32_t leaf_words(const View& v) { return int32_t((int64_t(v.capacity) + 31) >> 5); }
FPB_HD View make_view(uint32_t* bits, uint32_t* summary, int32_t capacity) {
  assert(capacity > 0 && capacity <= INT_MAX - 2);
  View v{}; v.bits = bits; v.summary = summary; v.capacity = capacity;
  int32_t n = int32_t((int64_t(leaf_words(v)) + 31) >> 5), total = 0;
  for (;;) {
    assert(v.levels < kMaxLevels);
    v.offsets[v.levels] = total; v.lengths[v.levels++] = n; total += n;
    if (n == 1) break;
    n = int32_t((int64_t(n) + 31) >> 5);
  }
  return v;
}
FPB_HD int32_t total_offset(const View& v) { return v.offsets[v.levels - 1] + 1; }
FPB_HD int32_t summary_words(const View& v) { return total_offset(v) + 1; }
// Factory compatibility: this includes the appended single total, not counts.
FPB_HD int32_t population_words(const View& v) { return summary_words(v); }
FPB_HD uint64_t array_bytes(const View& v) { return uint64_t(leaf_words(v) + summary_words(v)) * 4; }
FPB_HD uint32_t pop(uint32_t x) {
#if defined(__CUDA_ARCH__)
  return __popc(x);
#else
  return uint32_t(__builtin_popcount(x));
#endif
}
FPB_HD int first(uint32_t x) { assert(x);
#if defined(__CUDA_ARCH__)
  return __ffs(x) - 1;
#else
  return __builtin_ctz(x);
#endif
}
FPB_HD uint32_t after(int lane) { return lane == 31 ? 0 : UINT32_MAX << (lane + 1); }
FPB_HD uint32_t lower(int lane) { return lane == 0 ? 0 : (uint32_t(1) << lane) - 1; }
FPB_HD uint32_t valid_mask(const View& v, int32_t word) {
  assert(word >= 0 && word < leaf_words(v));
  const int tail = v.capacity & 31;
  return word == leaf_words(v) - 1 && tail ? (uint32_t(1) << tail) - 1 : UINT32_MAX;
}
FPB_D uint32_t bit_or(uint32_t* p, uint32_t mask) {
#if defined(__CUDACC__)
  return atomicOr(p, mask);
#else
  return std::atomic_ref<uint32_t>(*p).fetch_or(mask, std::memory_order_relaxed);
#endif
}
FPB_D uint32_t bit_and(uint32_t* p, uint32_t mask) {
#if defined(__CUDACC__)
  return atomicAnd(p, mask);
#else
  return std::atomic_ref<uint32_t>(*p).fetch_and(mask, std::memory_order_relaxed);
#endif
}
// Shared finish/scalar primitive. Both pointers must be valid aligned global
// addresses, including inactive predicates. Inactive operations do not access
// memory. The returned old values are exposed only after both instructions.
struct BitPair { uint32_t old_value, new_value; };
FPB_D BitPair paired_bit_and_or(uint32_t* old_ptr, uint32_t* new_ptr,
    uint32_t and_mask, uint32_t or_mask, bool old_active, bool new_active) {
  BitPair result{0, 0};
#if defined(__CUDACC__)
  const unsigned long long old_address = __cvta_generic_to_global(old_ptr);
  const unsigned long long new_address = __cvta_generic_to_global(new_ptr);
  asm volatile(
      "{\n\t"
      ".reg .pred p_old, p_new;\n\t"
      ".reg .b32 returned_old, returned_new;\n\t"
      "mov.b32 returned_old, 0;\n\t"
      "mov.b32 returned_new, 0;\n\t"
      "setp.ne.u32 p_old, %6, 0;\n\t"
      "setp.ne.u32 p_new, %7, 0;\n\t"
      "@p_old atom.relaxed.gpu.global.and.b32 returned_old, [%2], %4;\n\t"
      "@p_new atom.relaxed.gpu.global.or.b32 returned_new, [%3], %5;\n\t"
      "mov.b32 %0, returned_old;\n\t"
      "mov.b32 %1, returned_new;\n\t"
      "}\n"
      : "=r"(result.old_value), "=r"(result.new_value)
      : "l"(old_address), "l"(new_address), "r"(and_mask), "r"(or_mask),
        "r"(int(old_active)), "r"(int(new_active))
      : "memory");
#else
  if (old_active) result.old_value = bit_and(old_ptr, and_mask);
  if (new_active) result.new_value = bit_or(new_ptr, or_mask);
#endif
  return result;
}
FPB_D void add_total(const View& v, int32_t delta) {
  // Unique member mutations and the single-sign View producer contract bound
  // every partial count. Consumers validate totals after producer completion.
  // Discarding this result permits a reduction instruction instead of waiting
  // for an atomic return used only by redundant per-update bounds assertions.
#if defined(__CUDACC__)
  atomicAdd(v.summary + total_offset(v), uint32_t(delta));
#else
  std::atomic_ref<uint32_t>(v.summary[total_offset(v)]).fetch_add(uint32_t(delta), std::memory_order_relaxed);
#endif
}
FPB_HD uint32_t count_completed(const View& v) { return v.summary[total_offset(v)]; }
FPB_D void change_mask_unique(const View& v, int32_t word, uint32_t mask, bool present) {
  assert(mask && (mask & ~valid_mask(v, word)) == 0);
  const uint32_t previous = present ? bit_or(v.bits + word, mask) : bit_and(v.bits + word, ~mask);
  assert(present ? (previous & mask) == 0 : (previous & mask) == mask);
  bool transition = present ? previous == 0 : (previous & ~mask) == 0;
  for (int level = 0; transition && level < v.levels; ++level) {
    const uint32_t bit = uint32_t(1) << (word & 31); word >>= 5;
    const uint32_t before = present ? bit_or(v.summary + v.offsets[level] + word, bit)
                                    : bit_and(v.summary + v.offsets[level] + word, ~bit);
    assert(present ? (before & bit) == 0 : (before & bit) == bit);
    transition = present ? before == 0 : (before & ~bit) == 0;
  }
  add_total(v, present ? int32_t(pop(mask)) : -int32_t(pop(mask)));
}
FPB_D void insert_mask_unique(const View& v, int32_t word, uint32_t mask) { change_mask_unique(v, word, mask, true); }
FPB_D void erase_mask_unique(const View& v, int32_t word, uint32_t mask) { change_mask_unique(v, word, mask, false); }

// A cursor visits only nonempty summary branches. Several warps may traverse
// the small upper tree, but level-0 words are partitioned by word % stride.
struct Cursor { int32_t nodes[kMaxLevels]; uint32_t pending[kMaxLevels]; };
FPB_HD int32_t first_parent_completed(const View& v, Cursor& p) {
  int32_t node = 0;
  for (int level = v.levels - 1; level > 0; --level) {
    const uint32_t mask = v.summary[v.offsets[level] + node]; assert(mask);
    const int digit = first(mask); p.nodes[level] = node; p.pending[level] = mask & after(digit);
    node = int32_t(int64_t(node) * 32 + digit);
    assert(node < v.lengths[level - 1]);
  }
  return v.summary[v.offsets[0] + node] ? node : -1;
}
FPB_HD int32_t next_parent_completed(const View& v, Cursor& p) {
  int level = 1;
  while (level < v.levels && p.pending[level] == 0) ++level;
  if (level == v.levels) return -1;
  int digit = first(p.pending[level]); p.pending[level] &= ~(uint32_t(1) << digit);
  int32_t node = int32_t(int64_t(p.nodes[level]) * 32 + digit);
  for (--level; level > 0; --level) {
    assert(node < v.lengths[level]);
    const uint32_t mask = v.summary[v.offsets[level] + node]; assert(mask);
    digit = first(mask); p.nodes[level] = node; p.pending[level] = mask & after(digit);
    node = int32_t(int64_t(node) * 32 + digit);
  }
  assert(node < v.lengths[0]); return node;
}
// CPU reference executes the same sparse cursor and partitioned leaf batches.
// It is not an oracle for CUDA scheduling; any same-bucket tie is legal.
inline uint32_t select_host_completed(const View& v, uint32_t requested,
    int32_t* out, uint32_t out_capacity, uint32_t warps = 8) {
  assert(warps && requested <= out_capacity);
  const uint32_t total = count_completed(v), take = requested < total ? requested : total;
  uint32_t reserved = 0;
  for (uint32_t warp = 0; warp < warps && reserved < take; ++warp) {
    Cursor p{}; int32_t word = take ? first_parent_completed(v, p) : -1;
    while (word >= 0 && reserved < take) {
      if (uint32_t(word) % warps == warp) {
        const uint32_t presence = v.summary[v.offsets[0] + word]; assert(presence);
        for (int child = 0; child < 32 && reserved < take; ++child) if (presence & (uint32_t(1) << child)) {
          const int32_t leaf = int32_t(int64_t(word) * 32 + child); assert(leaf < leaf_words(v));
          uint32_t bits = v.bits[leaf]; assert(bits && !(bits & ~valid_mask(v, leaf)));
          while (bits && reserved < take) { const int bit = first(bits); bits &= bits - 1; out[reserved++] = leaf * 32 + bit; }
        }
      }
      word = next_parent_completed(v, p);
    }
  }
  assert(reserved == take); return take;
}
#if defined(__CUDACC__)
FPB_D void reset(const View& v) {
  for (int64_t i = threadIdx.x; i < leaf_words(v); i += blockDim.x) v.bits[i] = 0;
  for (int64_t i = threadIdx.x; i < summary_words(v); i += blockDim.x) v.summary[i] = 0;
  __syncthreads();
}
// Startup/fixture only. Never a decode activation or runtime residency rebuild.
FPB_D void initialize_counts(const View& v) {
  for (int64_t i = threadIdx.x; i < summary_words(v); i += blockDim.x) v.summary[i] = 0;
  __syncthreads();
  uint32_t local = 0;
  for (int64_t i = threadIdx.x; i < leaf_words(v); i += blockDim.x) {
    const uint32_t bits = v.bits[i]; assert(!(bits & ~valid_mask(v, int(i))));
    local += pop(bits);
    if (bits) bit_or(v.summary + v.offsets[0] + (i >> 5), uint32_t(1) << (i & 31));
  }
  if (local) add_total(v, int32_t(local));
  __syncthreads();
  for (int level = 1; level < v.levels; ++level) {
    for (int64_t i = threadIdx.x; i < v.lengths[level - 1]; i += blockDim.x)
      if (v.summary[v.offsets[level - 1] + i]) bit_or(v.summary + v.offsets[level] + (i >> 5), uint32_t(1) << (i & 31));
    __syncthreads();
  }
}
template<int NT> struct Storage {
  static_assert(NT > 0 && NT <= 1024 && NT % 32 == 0);
  Cursor paths[NT / 32]; uint32_t reserved, take;
};
// No CTA synchronization: warp groups concurrently enumerate stable Views.
// Caller assigns disjoint level-0 parent words and a shared quota counter.
FPB_D void select_warp_completed(const View& v, uint32_t take, int32_t* out,
    uint32_t* reserved, Cursor& path, int part, int parts) {
  constexpr uint32_t all = UINT32_MAX; const int lane = int(threadIdx.x) & 31;
  assert(__activemask() == all && part >= 0 && part < parts);
  int word = lane == 0 ? first_parent_completed(v, path) : -1;
  word = __shfl_sync(all, word, 0);
  while (word >= 0) {
    uint32_t done = lane == 0 ? cuda::atomic_ref<uint32_t, cuda::thread_scope_block>(*reserved).load(cuda::memory_order_relaxed) : 0;
    done = __shfl_sync(all, done, 0); if (done >= take) break;
    if (word % parts == part) {
      const uint32_t presence = v.summary[v.offsets[0] + word]; assert(presence);
      const int leaf = int(int64_t(word) * 32 + lane);
      const uint32_t bits = (presence & (uint32_t(1) << lane)) ? v.bits[leaf] : 0;
      assert(!(presence & (uint32_t(1) << lane)) || (leaf < leaf_words(v) && bits && !(bits & ~valid_mask(v, leaf))));
      const uint32_t count = pop(bits); uint32_t prefix = count;
      for (int delta = 1; delta < 32; delta <<= 1) { const uint32_t x = __shfl_up_sync(all, prefix, delta); if (lane >= delta) prefix += x; }
      const uint32_t total = __shfl_sync(all, prefix, 31);
      uint32_t base = lane == 0 ? atomicAdd(reserved, total) : 0; base = __shfl_sync(all, base, 0);
      // base is warp-uniform; this ticket has no output within the quota.
      if (base >= take) break;
      uint32_t mask = bits, dst = base + prefix - count;
      while (mask && dst < take) { const int bit = first(mask); mask &= mask - 1;
        out[dst++] = leaf * 32 + bit;
      }
    }
    word = lane == 0 ? next_parent_completed(v, path) : -1;
    word = __shfl_sync(all, word, 0);
  }
}
template<int NT> FPB_D uint32_t select_completed_cta(const View& v, uint32_t requested,
    int32_t* out, uint32_t out_capacity, Storage<NT>& s) {
  assert(blockDim.x == NT && requested <= out_capacity);
  if (threadIdx.x == 0) { const uint32_t n = count_completed(v); s.take = requested < n ? requested : n; s.reserved = 0; }
  __syncthreads();
  const uint32_t take = s.take;
  if (take) select_warp_completed(v, take, out, &s.reserved, s.paths[threadIdx.x / 32], int(threadIdx.x) / 32, NT / 32);
  __syncthreads(); assert(s.reserved >= take); return take;
}
#endif
} // namespace flowkv_presence_bitmap
#undef FPB_HD
#undef FPB_D
