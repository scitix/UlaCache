// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda/atomic>

#include <cstdint>

namespace sglang::device::flowkv {

// One lookup epoch owns at most active * topk history misses. If that entire
// union fits after head, allocation cannot overwrite a resident hit.
__device__ __forceinline__ bool append_requires_pins(int64_t head, int active, int topk, int capacity) {
  return head + int64_t(active) * topk > capacity;
}

// The same immutable lookup view is used by the standalone planner and an
// indexer's output epilogue. Allocation starts only after every lookup retires.
struct LookupState {
  bool pin_hits;
  const int64_t* step;
  const int* active;
  const int* keys;
  const int* values;
  const int* tags;
  int64_t* pins;
  int* hints;
  int* remap;
  int* sources;
  int* counts;
  int* totals;
  int capacity;
  int hash_size;
  int topk;
  int64_t rows;

  template <bool Direct, bool RelaxedPin = true>
  __device__ __forceinline__ int lookup(int row, int& owner) const {
    if (row < 0 || row >= rows) return -1;
    const int64_t now = *step;
    const auto pin = [&](int entry) {
      if (!pin_hits) return;
      if constexpr (RelaxedPin)
        cuda::atomic_ref<int64_t, cuda::thread_scope_device>(pins[entry]).store(now, cuda::memory_order_relaxed);
      else
        atomicExch(reinterpret_cast<unsigned long long*>(pins + entry), static_cast<unsigned long long>(now));
    };
    auto hint = cuda::atomic_ref<int, cuda::thread_scope_device>(hints[row]);
    if constexpr (Direct) {
      // begin invalidates every current row before the indexer runs.
      const int entry = hint.load(cuda::memory_order_relaxed);
      if (entry >= 0 && entry < capacity && tags[entry] == row) {
        pin(entry);
        return entry;
      }
    }
    unsigned h = (uint32_t(row) * 2654435761u) >> (32 - (__ffs(hash_size) - 1));
    for (int probe = 0; probe < hash_size; ++probe) {
      const int key = Direct ? __ldg(keys + h) : keys[h];
      if (key == row) return capacity + (Direct ? __ldg(values + h) : values[h]);
      if (key < 0) break;
      h = (h + 1) & (hash_size - 1);
    }
    int entry = hint.load(cuda::memory_order_relaxed);
    for (;;) {
      if (entry >= 0 && entry < capacity && tags[entry] == row) {
        pin(entry);
        return entry;
      }
      if (entry == -2) return -2;
      int expected = entry;
      if (hint.compare_exchange_strong(expected, -2, cuda::memory_order_relaxed)) {
        owner = 1;
        return -2;
      }
      entry = expected;
    }
  }

  __device__ __forceinline__ void emit(int query, int column, int row) const {
    int owner = 0;
    const int mapped = query < *active ? lookup<true>(row, owner) : -1;
    remap[int64_t(query) * topk + column] = mapped;
    if (owner) {
      const int offset = atomicAdd(counts + query, 1);
      sources[int64_t(query) * topk + offset] = row;
      atomicAdd(totals, 1);
    }
  }
};

// Device-resident descriptor, built once per cache group before Graph capture.
// Storage geometry is runtime data; the indexer has no model or rank constants.
struct NativeLookup {
  const int64_t* descriptor;

  // Called by every lane of a full warp at the native page-transform output.
  // `valid` masks tail columns; lookup/CAS still elects one owner globally when
  // different queries or warps select the same row. The allocator runs only
  // after this kernel retires, so source ordering inside a query is immaterial.
  __device__ __forceinline__ void emit(int query, int column, int row, bool valid) const {
    const auto* d = descriptor;
    const LookupState state{
        append_requires_pins(*reinterpret_cast<const int64_t*>(d[19]),
            *reinterpret_cast<const int*>(d[1]), int(d[14]), int(d[12])),
        reinterpret_cast<const int64_t*>(d[0]), reinterpret_cast<const int*>(d[1]),
        reinterpret_cast<const int*>(d[2]), reinterpret_cast<const int*>(d[3]),
        reinterpret_cast<const int*>(d[4]), reinterpret_cast<int64_t*>(d[5]),
        reinterpret_cast<int*>(d[6]), reinterpret_cast<int*>(d[7]),
        reinterpret_cast<int*>(d[8]), reinterpret_cast<int*>(d[9]),
        reinterpret_cast<int*>(d[10]), int(d[12]), int(d[13]), int(d[14]), d[15]};
    auto* global = reinterpret_cast<int*>(d[11]);
    const int64_t start = int64_t(query) * state.topk;
    int owner = 0;
    if (valid) {
      const int logical = row + int(d[16]);
      row = row > 0 ? (logical % int(d[18])) * int(d[17]) + logical / int(d[18]) : row;
      global[start + column] = row;
      state.remap[start + column] = query < *state.active ? state.lookup<true>(row, owner) : -1;
    }
    constexpr unsigned FullWarp = 0xffffffffu;
    const unsigned owners = __ballot_sync(FullWarp, owner != 0);
    if (owners == 0) return;
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(owners) - 1;
    int base = 0;
    if (lane == leader) {
      const int count = __popc(owners);
      base = atomicAdd(state.counts + query, count);
      atomicAdd(state.totals, count);
    }
    base = __shfl_sync(FullWarp, base, leader);
    if (owner) {
      const int prefix = __popc(owners & ((1u << lane) - 1u));
      state.sources[start + base + prefix] = row;
    }
  }
};

}  // namespace sglang::device::flowkv
