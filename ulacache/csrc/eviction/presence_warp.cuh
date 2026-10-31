#pragma once
#include "csrc/eviction/presence_bitmap.cuh"
namespace flowkv_presence_warp {
namespace pb = flowkv_presence_bitmap;
#if defined(__CUDACC__)
__device__ __forceinline__ uint32_t mask_or(uint32_t group, uint32_t bit) {
  return (group & (group - 1)) ? __reduce_or_sync(group, bit) : bit;
}
// Full-warp heterogeneous View update. The total needs only a winner ballot
// and matchAny(bucket), never a subgroup reduce-add at each hierarchy level.
__device__ __forceinline__ void change_warp_unique(const pb::View* views, int view_count,
    int bucket, int slot, bool present) {
  constexpr uint32_t all = UINT32_MAX;
  // All lanes participate, including inactive slots after divergent arrival.
  assert(__ballot_sync(all, true) == all);
  const int lane = int(threadIdx.x) & 31; const bool valid = slot >= 0;
  assert(!valid || (bucket >= 0 && bucket < view_count && slot < views[bucket].capacity));
  const uint32_t winners = __ballot_sync(all, valid); if (!winners) return;
  const uint64_t key = valid ? (uint64_t(uint32_t(bucket)) << 32) | (uint32_t(slot) >> 5) : UINT64_MAX;
  const uint32_t same = __match_any_sync(all, key);
  const uint32_t own = valid ? uint32_t(1) << (slot & 31) : 0;
  const uint32_t mask = valid ? mask_or(same, own) : 0;
  assert(!valid || pb::pop(mask) == pb::pop(same));
  uint32_t node = UINT32_MAX, bit = 0;
  if (valid && lane == pb::first(same)) {
    const auto& v = views[bucket]; const uint32_t word = uint32_t(slot) >> 5;
    assert(!(mask & ~pb::valid_mask(v, int(word))));
    const uint32_t old = present ? pb::bit_or(v.bits + word, mask) : pb::bit_and(v.bits + word, ~mask);
    assert(present ? !(old & mask) : (old & mask) == mask);
    if (present ? old == 0 : (old & ~mask) == 0) { node = word >> 5; bit = uint32_t(1) << (word & 31); }
  }
  for (int level = 0; level < views[0].levels; ++level) {
    const bool changing = node != UINT32_MAX;
    if (__ballot_sync(all, changing) == 0) break;
    const uint64_t k = changing ? (uint64_t(uint32_t(bucket)) << 32) | node : UINT64_MAX;
    const uint32_t group = __match_any_sync(all, k);
    const uint32_t update = changing ? mask_or(group, bit) : 0;
    uint32_t next = UINT32_MAX, next_bit = 0;
    if (changing && lane == pb::first(group)) {
      const auto& v = views[bucket];
      const uint32_t old = present ? pb::bit_or(v.summary + v.offsets[level] + node, update)
                                   : pb::bit_and(v.summary + v.offsets[level] + node, ~update);
      assert(present ? !(old & update) : (old & update) == update);
      if (present ? old == 0 : (old & ~update) == 0) { next = node >> 5; next_bit = uint32_t(1) << (node & 31); }
    }
    node = next; bit = next_bit;
  }
  const uint32_t bucket_group = __match_any_sync(all, valid ? bucket : -1);
  if (valid && lane == pb::first(bucket_group)) {
    const int delta = int(pb::pop(bucket_group & winners));
    pb::add_total(views[bucket], present ? delta : -delta);
  }
}
__device__ __forceinline__ void change_warp_unique(const pb::View& view, int slot, bool present) {
  change_warp_unique(&view, 1, 0, slot, present);
}
// Finish-only transfer: sources are delete-only, the uniform destination is
// insert-only. All output IDs are complete before this producer starts.
// source==-1 admits an already-free target; slot==-1 is an inactive lane.
__device__ __forceinline__ void transfer_warp_unique(const pb::View* views, int view_count,
    int source, int destination, int slot) {
  constexpr uint32_t all = UINT32_MAX;
  // All lanes participate, including inactive slots after divergent arrival.
  assert(__ballot_sync(all, true) == all);
  const int lane = int(threadIdx.x) & 31;
  const bool valid = slot >= 0, removing = valid && source >= 0;
  assert(destination >= 0 && destination < view_count);
  assert(!valid || slot < views[destination].capacity);
  assert(!removing || (source < view_count && source != destination));
  const uint32_t incoming = __ballot_sync(all, valid); if (!incoming) return;
  const uint32_t outgoing = __ballot_sync(all, removing);
  const uint32_t word = valid ? uint32_t(slot) >> 5 : UINT32_MAX;
  const uint64_t old_key = removing ? (uint64_t(uint32_t(source)) << 32) | word : UINT64_MAX;
  const uint32_t old_same = __match_any_sync(all, old_key);
  const uint32_t new_same = __match_any_sync(all, word);
  const uint32_t own = valid ? uint32_t(1) << (slot & 31) : 0;
  const uint32_t old_mask = removing ? mask_or(old_same, own) : 0;
  const uint32_t new_mask = valid ? mask_or(new_same, own) : 0;
  assert(!removing || pb::pop(old_mask) == pb::pop(old_same));
  assert(!valid || pb::pop(new_mask) == pb::pop(new_same));
  const bool old_leader = removing && lane == pb::first(old_same);
  const bool new_leader = valid && lane == pb::first(new_same);
  // Prepare legal dummy bases before the paired predicated instruction block.
  // No sentinel word/node participates in inactive pointer arithmetic.
  uint32_t* old_pointer = views[destination].bits;
  uint32_t* new_pointer = views[destination].bits;
  if (old_leader) old_pointer = views[source].bits + word;
  if (new_leader) new_pointer = views[destination].bits + word;
  const auto leaf_pair = pb::paired_bit_and_or(old_pointer, new_pointer,
      ~old_mask, new_mask, old_leader, new_leader);
  const uint32_t old_before = leaf_pair.old_value, new_before = leaf_pair.new_value;
  uint32_t old_node = UINT32_MAX, new_node = UINT32_MAX, old_bit = 0, new_bit = 0;
  if (old_leader) {
    assert((old_before & old_mask) == old_mask);
    if ((old_before & ~old_mask) == 0) { old_node = word >> 5; old_bit = uint32_t(1) << (word & 31); }
  }
  if (new_leader) {
    assert(!(new_before & new_mask));
    if (new_before == 0) { new_node = word >> 5; new_bit = uint32_t(1) << (word & 31); }
  }
  for (int level = 0; level < views[0].levels; ++level) {
    const bool old_changing = old_node != UINT32_MAX, new_changing = new_node != UINT32_MAX;
    if (__ballot_sync(all, old_changing || new_changing) == 0) break;
    const uint64_t old_k = old_changing ? (uint64_t(uint32_t(source)) << 32) | old_node : UINT64_MAX;
    const uint32_t old_group = __match_any_sync(all, old_k);
    const uint32_t new_group = __match_any_sync(all, new_node);
    const uint32_t old_update = old_changing ? mask_or(old_group, old_bit) : 0;
    const uint32_t new_update = new_changing ? mask_or(new_group, new_bit) : 0;
    const bool old_writer = old_changing && lane == pb::first(old_group);
    const bool new_writer = new_changing && lane == pb::first(new_group);
    uint32_t* old_parent = views[destination].summary;
    uint32_t* new_parent = views[destination].summary;
    if (old_writer) old_parent = views[source].summary + views[source].offsets[level] + old_node;
    if (new_writer) new_parent = views[destination].summary + views[destination].offsets[level] + new_node;
    const auto parent_pair = pb::paired_bit_and_or(old_parent, new_parent,
        ~old_update, new_update, old_writer, new_writer);
    const uint32_t before_old = parent_pair.old_value, before_new = parent_pair.new_value;
    uint32_t next_old = UINT32_MAX, next_new = UINT32_MAX, bit_old = 0, bit_new = 0;
    if (old_writer) {
      assert((before_old & old_update) == old_update);
      if ((before_old & ~old_update) == 0) { next_old = old_node >> 5; bit_old = uint32_t(1) << (old_node & 31); }
    }
    if (new_writer) {
      assert(!(before_new & new_update));
      if (before_new == 0) { next_new = new_node >> 5; bit_new = uint32_t(1) << (new_node & 31); }
    }
    old_node = next_old; old_bit = bit_old; new_node = next_new; new_bit = bit_new;
  }
  const uint32_t old_bucket_group = __match_any_sync(all, removing ? source : -1);
  if (removing && lane == pb::first(old_bucket_group))
    pb::add_total(views[source], -int(pb::pop(old_bucket_group & outgoing)));
  if (lane == pb::first(incoming)) pb::add_total(views[destination], int(pb::pop(incoming)));
}
#else
inline void change_warp_unique(const pb::View* views, int view_count, int bucket, int slot, bool present) {
  if (slot < 0) return;
  assert(bucket >= 0 && bucket < view_count && slot < views[bucket].capacity);
  pb::change_mask_unique(views[bucket], slot >> 5, uint32_t(1) << (slot & 31), present);
}
inline void change_warp_unique(const pb::View& view, int slot, bool present) {
  if (slot >= 0) pb::change_mask_unique(view, slot >> 5, uint32_t(1) << (slot & 31), present);
}
// CPU actual-body contracts execute scalar atomics; a separate exact-body warp
// shim verifies the CUDA branch's independent grouping/transition propagation.
inline void transfer_warp_unique(const pb::View* views, int view_count,
    int source, int destination, int slot) {
  if (slot < 0) return;
  assert(destination >= 0 && destination < view_count && slot < views[destination].capacity);
  const uint32_t mask = uint32_t(1) << (slot & 31);
  if (source >= 0) {
    assert(source < view_count && source != destination);
    pb::erase_mask_unique(views[source], slot >> 5, mask);
  }
  pb::insert_mask_unique(views[destination], slot >> 5, mask);
}
#endif
} // namespace flowkv_presence_warp
