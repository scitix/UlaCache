// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifndef GLOBAL_SMALL_HIST_BLOCKS
#define GLOBAL_SMALL_HIST_BLOCKS 512
#endif
#ifndef GLOBAL_SINGLE_QUERY_ALLOC
#define GLOBAL_SINGLE_QUERY_ALLOC 1
#endif
#ifndef GLOBAL_HIST_BLOCKS
#define GLOBAL_HIST_BLOCKS 512
#endif
#ifndef GLOBAL_ALLOC_ITEMS
#define GLOBAL_ALLOC_ITEMS 1
#endif
#ifndef GLOBAL_STRIPED
#define GLOBAL_STRIPED 1
#endif
#ifndef GLOBAL_RELAXED_PIN_STORE
#define GLOBAL_RELAXED_PIN_STORE 1
#endif
#ifndef FLOW_ZERO_SENTINEL
#define FLOW_ZERO_SENTINEL 0
#endif
#ifndef FLOW_CIRCULAR_MAX_QUERIES
#define FLOW_CIRCULAR_MAX_QUERIES 0
#endif
#ifndef FLOW_WARP_CIRCULAR
#define FLOW_WARP_CIRCULAR 0
#endif
#ifndef FLOW_PARALLEL_CIRCULAR
#define FLOW_PARALLEL_CIRCULAR 0
#endif
#ifndef FLOW_PDL
#define FLOW_PDL 1
#endif
#ifndef FLOW_DIRECT_CURRENT_LOOKUP
#define FLOW_DIRECT_CURRENT_LOOKUP 0
#endif
// Global compact KV cache. Persistent slots have no request/batch ownership.
// Kernel boundaries separate pin/claim, LRU histogram, and allocation.
#include <sgl_kernel/flowkv/lookup.cuh>

#include <cub/block/block_scan.cuh>
#include <cuda/atomic>

#include <algorithm>
#include <cuda_runtime.h>
#include <stdint.h>
#if FLOW_RECORD_HOST_TIER
#include "csrc/cache/residency_fused.cuh"
#endif
#ifndef FLOW_TOPK
#error "FlowKV requires a compile-time TopK"
#endif
#ifndef FLOW_MAX_QUERIES
#error "FlowKV requires a compile-time query bound"
#endif
#ifndef FLOW_ROW_BYTES
#error "FlowKV requires a compile-time KV row size"
#endif
constexpr int K = FLOW_TOPK, NT = 256, MAX_QUERIES = FLOW_MAX_QUERIES;
// Preserve existing hash geometry through1024; cap2048 shared keys/values at32KiB.
constexpr int HASH = (MAX_QUERIES > 1024 ? 2 : 4) * MAX_QUERIES;
static_assert(K >= NT && K % NT == 0);
static_assert(MAX_QUERIES >= 1 && MAX_QUERIES <= 2048 && (MAX_QUERIES & (MAX_QUERIES - 1)) == 0);
static_assert(FLOW_ROW_BYTES > 0 && FLOW_ROW_BYTES % 2 == 0);
constexpr unsigned FULL = 0xffffffffu;

template <typename Kernel, typename... Args>
cudaError_t cache_launch(dim3 grid, dim3 block, cudaStream_t stream, Kernel kernel, Args... args) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = FLOW_PDL;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = stream;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, args...);
}

__device__ __forceinline__ int read_int(const void* p, int i, int bytes) {
  return bytes == 8 ? int(((const int64_t*)p)[i]) : ((const int*)p)[i];
}
__device__ __forceinline__ unsigned hash_row(int x) {
  return (uint32_t(x) * 2654435761u) >> (32 - __builtin_ctz(HASH));
}

template <bool FromNative = false, bool StepPublication = false>
__global__ void begin_global_step(
    const void* current,
    int cur_bytes,
    const void* valid,
    int valid_bytes,
    int queries,
    int max_queries,
    int groups,
    long nrows,
    int64_t* step,
    int* active,
    int* current_keys,
    int* current_values,
    int64_t* cursors,
    int* totals,
    int* hints,
    int* counts,
    int* histogram,
    int* global_current = nullptr, int native_base = 0, int native_slots = 0, int native_peers = 1,
    const uint64_t* ready_peers = nullptr, int ready_count = 0
#if FLOW_RECORD_HOST_TIER
    , const int64_t* host_groups = nullptr
#endif
    ) {
  const int t = threadIdx.x;
  const int n = valid ? max(0, min(queries, read_int(valid, 0, valid_bytes))) : queries;
  if (t == 0) {
    if constexpr (StepPublication) {
      // TP peers publish once after their preceding decode's writer tasks.
      const int64_t prior = *step;
      for (int peer = 0; peer < ready_count; ++peer) {
        auto* ready = reinterpret_cast<int64_t*>(ready_peers[peer]);
        while (cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*ready)
                   .load(cuda::memory_order_acquire) < prior) {}
      }
    }
    ++*step;
    *active = n;
  }
#if FLOW_RECORD_HOST_TIER
  // Every lane must acquire prior peer retirement before clearing counts
  // that another owner's deferred PB producers can still be reading.
  if constexpr (StepPublication) __syncthreads();
#endif
  for (int j = t; j < HASH; j += blockDim.x) {
    current_keys[j] = -1;
    current_values[j] = -1;
  }
  for (int g = t; g < groups; g += blockDim.x) {
    cursors[g * 2] = cursors[g * 2 + 1];
    totals[g * 2] = 0;
    totals[g * 2 + 1] = 0;
  }
  for (int i = t; i < groups * max_queries; i += blockDim.x)
    counts[i] = 0;
  for (int i = t; i < groups * 258; i += blockDim.x)
    histogram[i] = 0;
  if constexpr (FromNative) {
    for (int q = t; q < queries; q += blockDim.x) {
      const int row = read_int(current, q, cur_bytes);
      const int logical = row + native_base;
      global_current[q] = row > 0 ? (logical % native_peers) * native_slots + logical / native_peers : row;
    }
  }
  __syncthreads();
  for (int q = t; q < n; q += blockDim.x) {
    const int row = FromNative ? global_current[q] : read_int(current, q, cur_bytes);
    if (row < 0 || (FLOW_ZERO_SENTINEL && row == 0) || row >= nrows) continue;
    unsigned h = hash_row(row);
    for (;;) {
      const int old = atomicCAS(current_keys + h, -1, row);
      if (old == -1 || old == row) {
        atomicCAS(current_values + h, -1, q);
        break;
      }
      h = (h + 1) & (HASH - 1);
    }
  }
  // Parallelize across groups as well as query tokens.
  for (int i = t; i < groups * n; i += blockDim.x) {
    const int row = FromNative ? global_current[i % n] : read_int(current, i % n, cur_bytes);
    if (row >= 0 && row < nrows)
      cuda::atomic_ref<int, cuda::thread_scope_device>(hints[long(i / n) * nrows + row])
          .store(-1, cuda::memory_order_relaxed);
  }
#if FLOW_RECORD_HOST_TIER
  if constexpr (StepPublication)
    if (host_groups)
      sglang::flowkv_host::begin_groups(host_groups, groups, current, cur_bytes, n);
#endif
}
extern "C" int flowkv_global_begin(
    const void* current,
    int cur_bytes,
    const void* valid,
    int valid_bytes,
    int queries,
    int max_queries,
    int groups,
    long nrows,
    int64_t* step,
    int* active,
    int* keys,
    int* values,
    int64_t* cursors,
    int* totals,
    int* hints,
    int* counts,
    int* histogram,
    cudaStream_t stream) {
  if (queries < 0 || queries > max_queries || max_queries != MAX_QUERIES || groups < 1 ||
      (cur_bytes != 4 && cur_bytes != 8))
    return int(cudaErrorInvalidValue);
  begin_global_step<false><<<1, 128, 0, stream>>>(
      current,
      cur_bytes,
      valid,
      valid_bytes,
      queries,
      max_queries,
      groups,
      nrows,
      step,
      active,
      keys,
      values,
      cursors,
      totals,
      hints,
      counts,
      histogram);
  return int(cudaGetLastError());
}
__global__ void invalidate_global_rows(const void* rows, int bytes, int count, int groups, long nrows, int* hints) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count * groups) {
    const int row = read_int(rows, i % count, bytes);
    if (row >= 0 && row < nrows)
      cuda::atomic_ref<int, cuda::thread_scope_device>(hints[long(i / count) * nrows + row])
          .store(-1, cuda::memory_order_relaxed);
  }
}
extern "C" int flowkv_global_invalidate(
    const void* rows, int bytes, int count, int groups, long nrows, int* hints, cudaStream_t stream) {
  if (count < 0 || (bytes != 4 && bytes != 8)) return int(cudaErrorInvalidValue);
  if (count)
    invalidate_global_rows<<<(count * groups + 255) / 256, 256, 0, stream>>>(rows, bytes, count, groups, nrows, hints);
  return int(cudaGetLastError());
}

template<bool FromNative = false>
__global__ void global_lookup_pin(
    const int* idx,
    const int64_t* step,
    const int* active,
    const int* current_keys,
    const int* current_values,
    const int* tags,
    int64_t* pins,
    int* hint,
    int* remap,
    int* sources,
    int* counts,
    int* bases,
    int* totals,
    const int64_t* cursors,
    int capacity,
    long nrows,
    int* global_idx = nullptr, int native_base = 0, int native_slots = 0,
    int native_peers = 1) {
  // Permit the dependent grid to launch early, but require the preceding
  // grid to finish before reading any of its metadata or KV outputs.
  if constexpr (FLOW_PDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if constexpr (FLOW_PDL) asm volatile("griddepcontrol.wait;" ::: "memory");

  constexpr int Threads = NT;
  const int b = blockIdx.x / (K / Threads), t = threadIdx.x, j = (blockIdx.x % (K / Threads)) * Threads + t;
  const long io = long(b) * K;
  int converted_row = 0;
  if constexpr (FromNative) {
    const int row = idx[io + j];
    const int logical = row + native_base;
    converted_row = row > 0 ? (logical % native_peers) * native_slots + logical / native_peers : row;
#if FLOW_RECORD_HOST_TIER
    // Native slot zero is the graph padding sentinel, with no KV residency.
    if (converted_row == 0) converted_row = -1;
#endif
    global_idx[io + j] = converted_row;
  }
  if (b >= *active) {
    remap[io + j] = -1;
    return;
  }
  __shared__ int keys[FLOW_DIRECT_CURRENT_LOOKUP ? 1 : HASH], values[FLOW_DIRECT_CURRENT_LOOKUP ? 1 : HASH], query_base;
  using Scan = cub::BlockScan<int, Threads>;
  __shared__ typename Scan::TempStorage scan;
  if constexpr (!FLOW_DIRECT_CURRENT_LOOKUP) {
    for (int h = t; h < HASH; h += Threads) {
      keys[h] = current_keys[h];
      values[h] = current_values[h];
    }
    __syncthreads();
  }
  const sglang::device::flowkv::LookupState state{
      !FLOW_PARALLEL_CIRCULAR || sglang::device::flowkv::append_requires_pins(
          cursors[0], *active, K, capacity),
      step,
      active,
      FLOW_DIRECT_CURRENT_LOOKUP ? current_keys : keys,
      FLOW_DIRECT_CURRENT_LOOKUP ? current_values : values,
      tags,
      pins,
      hint,
      remap,
      sources,
      counts,
      totals,
      capacity,
      HASH,
      K,
      nrows};
  int own = 0;
  int row;
  if constexpr (FromNative) row = converted_row;
  else row = idx[io + j];
  const int mapped = state.lookup<FLOW_DIRECT_CURRENT_LOOKUP, GLOBAL_RELAXED_PIN_STORE>(row, own);
  remap[io + j] = mapped;
#if FLOW_C4_SKIP_EMPTY_SCAN
  // Counts are cleared by begin. A hit-only CTA has no miss list to compact.
  if (!__syncthreads_or(own)) return;
#endif
  int prefix, total;
  Scan(scan).ExclusiveSum(own, prefix, total);
  if (t == 0) {
    query_base = total ? atomicAdd(counts + b, total) : 0;
    if (total) atomicAdd(totals, total);
  }
  __syncthreads();
  if (own) sources[io + query_base + prefix] = row;
}

// Ages 1..254 are exact steps, 255 groups older rows, 256 means empty.
// Current TopK hits already have age zero and are excluded from eviction.
__device__ __forceinline__ int lru_bucket(int tag, int64_t pin, int64_t now) {
  return tag < 0 ? 256 : int(min(int64_t(255), now - pin));
}
__global__ void global_lru_histogram(
    const int* tags,
    const int64_t* pins,
    const int64_t* step,
    const int* totals,
    int* histogram,
    const int64_t* cursors,
    int capacity) {
  if constexpr (FLOW_PDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if constexpr (FLOW_PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (totals[0] == 0 || cursors[0] + totals[0] <= capacity) return;
  const int t = threadIdx.x, lane = t & 31;
  __shared__ int hist[257];
  for (int i = t; i < 257; i += NT)
    hist[i] = 0;
  __syncthreads();
  const int64_t now = *step;
  for (int slot = blockIdx.x * NT + t; slot < capacity; slot += gridDim.x * NT) {
    const int bucket = lru_bucket(tags[slot], pins[slot], now);
    const unsigned live = __ballot_sync(FULL, bucket > 0);
    if (bucket > 0) {
      const unsigned peers = __match_any_sync(live, bucket);
      if (lane == __ffs(peers) - 1) atomicAdd(hist + bucket, __popc(peers));
    }
  }
  __syncthreads();
  for (int i = t; i < 257; i += NT)
    if (hist[i]) atomicAdd(histogram + i, hist[i]);
}
// A verify batch can exceed the 256 threads in an allocation CTA. Scan query
// counts in blocked thread order so every query contributes, including the
// second half of a BS64 x DSPARK6 graph's 512-query capacity.
__device__ __forceinline__ void query_count_ends(const int* counts, int n, int* ends) {
  constexpr int ITEMS = (MAX_QUERIES + NT - 1) / NT;
  using Scan = cub::BlockScan<int, NT>;
  __shared__ typename Scan::TempStorage temp;
  int values[ITEMS];
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int q = threadIdx.x * ITEMS + i;
    values[i] = q < n ? counts[q] : 0;
  }
  Scan(temp).InclusiveSum(values, values);
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int q = threadIdx.x * ITEMS + i;
    if (q < MAX_QUERIES) ends[q] = values[i];
  }
  __syncthreads();
}

// Raw cursors advance across epochs and may exceed one revolution. Keep the
// mask path for power-of-two capacities; arbitrary capacities require modulo.
__device__ __forceinline__ int history_cursor_slot(unsigned long long cursor, int capacity) {
  if ((capacity & (capacity - 1)) == 0) return int(cursor & (capacity - 1));
  return int(cursor % static_cast<unsigned long long>(capacity));
}

// A normalized start and a valid bounded-scan offset are both below capacity.
// Their sum is below 2*capacity, so one subtraction replaces per-slot division.
__device__ __forceinline__ int history_window_slot(int start, int offset, int capacity) {
  const unsigned slot = unsigned(start) + unsigned(offset);
  return int(slot >= unsigned(capacity) ? slot - unsigned(capacity) : slot);
}

// 0: LRU, 1: per-miss circular probes, 2: parallel circular scan.
template <int POLICY = 0>
__global__ void global_allocate(
    const int* sources,
    int* targets,
    const int* counts,
    const int* active,
    int* tags,
    int* hint,
    int64_t* pins,
    const int64_t* step,
    int64_t* cursors,
    int* totals,
    int* histogram,
    int capacity,
    int* errors
#if FLOW_RECORD_HOST_TIER
    , int64_t* selected_ready = nullptr, const int64_t* host_group = nullptr
#endif
    ) {
#if FLOW_RECORD_HOST_TIER
  const bool cold_owner = sglang::flowkv_host::has_cpu_rows(host_group);
  if constexpr (FLOW_PDL)
    if (!cold_owner) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#else
  if constexpr (FLOW_PDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  if constexpr (FLOW_PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
#if FLOW_RECORD_HOST_TIER
  if (selected_ready && blockIdx.x == 0 && threadIdx.x == 0)
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*selected_ready)
        .store(*step, cuda::memory_order_release);
#endif
  const int t = threadIdx.x, lane = t & 31, total = totals[0], n = *active;
  if (total == 0) return;
  const int64_t now = *step, head = cursors[0];
  // Unused capacity needs no eviction selection or full-pool age scan.
  if (head + total <= capacity) {
    const int q = blockIdx.x;
    if (q >= n) return;
    __shared__ int append_base;
    // Decode often has only a few live queries even when the graph/arena
    // bound is 64 or 512. Use the existing warp scan for that active prefix;
    // the padded graph width need not force a full-CTA scan and barrier.
    if (MAX_QUERIES <= 32 || n <= 32) {
      if (t < 32) {
        const int count = t < n ? counts[t] : 0;
        int sum = count;
        for (int d = 1; d < 32; d *= 2) {
          const int v = __shfl_up_sync(FULL, sum, d);
          if (t >= d) sum += v;
        }
        if (t == q) append_base = sum - count;
      }
    } else if constexpr (MAX_QUERIES > NT) {
      __shared__ int ends[MAX_QUERIES];
      query_count_ends(counts, n, ends);
      if (t == 0) append_base = ends[q] - counts[q];
    } else {
      using QueryScan = cub::BlockScan<int, NT>;
      __shared__ typename QueryScan::TempStorage query_scan;
      int prefix;
      QueryScan(query_scan).ExclusiveSum(t < n ? counts[t] : 0, prefix);
      if (t == q) append_base = prefix;
    }
    __syncthreads();
    for (int m = t; m < counts[q]; m += NT) {
      const int slot = int(head) + append_base + m, row = sources[long(q) * K + m];
      tags[slot] = row;
      pins[slot] = now;
      hint[row] = slot;
      targets[long(q) * K + m] = slot;
    }
    if (q == 0 && t == 0) cursors[1] = head + total;
    return;
  }
  if constexpr (POLICY == 1) {
    // With at least twice the maximum current TopK union in the arena, a
    // circular hand can find unpinned victims without a full-pool LRU scan.
    // Lookup has pinned every hit before this kernel starts. Claiming a pin
    // atomically also protects slots assigned to concurrent miss owners.
    // Do not protect the previous step: a preceding larger batch may have
    // pinned the entire arena, even when this step has only one request.
    const int q = blockIdx.x;
    if (q >= n) return;
    const bool power_of_two = (capacity & (capacity - 1)) == 0;
    int wrapped_head = 0;
    if (!power_of_two) {
      // Reduce the raw epoch cursor once per warp, without a CTA barrier.
      if (lane == 0) wrapped_head = history_cursor_slot(head, capacity);
      wrapped_head = __shfl_sync(FULL, wrapped_head, 0);
    }
    // This policy requires capacity >= 2*batch*K. Before completing M misses,
    // rejected H hit slots, successful claims and outstanding reservations
    // total at most H+M <= n*K. No reservation can reach a second revolution.
    if constexpr (FLOW_WARP_CIRCULAR) {
      // Reserve adjacent candidate slots once per participating warp, rather
      // than once per miss. Hit pins and successful CAS claims stay protected
      // for the current step. Every lane participates in each ballot/shuffle.
      const int rounds = (counts[q] + NT - 1) / NT;
      for (int round = 0; round < rounds; ++round) {
        const int m = round * NT + t;
        bool pending = m < counts[q];
        unsigned mask = __ballot_sync(FULL, pending);
        while (mask) {
          const int leader = __ffs(mask) - 1;
          unsigned long long base = 0;
          if (lane == leader)
            base = atomicAdd(reinterpret_cast<unsigned long long*>(cursors + 1),
                             static_cast<unsigned long long>(__popc(mask)));
          base = __shfl_sync(FULL, base, leader);
          if (pending) {
            const unsigned preceding = mask & unsigned((1ULL << lane) - 1);
            const auto cursor = base + __popc(preceding);
            const int slot = power_of_two
                ? int(cursor & (capacity - 1))
                : history_window_slot(wrapped_head, int(cursor - head), capacity);
            const auto previous = cuda::atomic_ref<int64_t, cuda::thread_scope_device>(
                pins[slot]).load(cuda::memory_order_relaxed);
            if (previous != now && atomicCAS(
                reinterpret_cast<unsigned long long*>(pins + slot),
                static_cast<unsigned long long>(previous),
                static_cast<unsigned long long>(now)) ==
                static_cast<unsigned long long>(previous)) {
              const int row = sources[long(q) * K + m];
              tags[slot] = row;
              hint[row] = slot;
              targets[long(q) * K + m] = slot;
              pending = false;
            }
          }
          mask = __ballot_sync(FULL, pending);
        }
      }
      return;
    }
    for (int m = t; m < counts[q]; m += NT) {
      int slot;
      for (;;) {
        const auto cursor = atomicAdd(reinterpret_cast<unsigned long long*>(cursors + 1), 1ULL);
        slot = power_of_two
            ? int(cursor & (capacity - 1))
            : history_window_slot(wrapped_head, int(cursor - head), capacity);
        const auto previous =
            cuda::atomic_ref<int64_t, cuda::thread_scope_device>(pins[slot]).load(cuda::memory_order_relaxed);
        if (previous == now) continue;
        if (atomicCAS(
                reinterpret_cast<unsigned long long*>(pins + slot),
                static_cast<unsigned long long>(previous),
                static_cast<unsigned long long>(now)) == static_cast<unsigned long long>(previous))
          break;
      }
      const int row = sources[long(q) * K + m];
      tags[slot] = row;
      hint[row] = slot;
      targets[long(q) * K + m] = slot;
    }
    return;
  }
  if constexpr (POLICY == 2) {
    // Each CTA visits a disjoint range of the circular arena. Unlike serial
    // probing, this also bounds the cost when nearly every slot is pinned.
    // Kernel boundaries make hit pins immutable here; each candidate slot has
    // exactly one writer, so allocation needs no per-slot CAS or global barrier.
    // One TopK union per group and step pins H hits and needs M miss slots,
    // with H + M <= n*K. Any n*K-slot circular window has enough victims.
    const int limit = min(capacity, n * K);
    constexpr int ITEMS = GLOBAL_ALLOC_ITEMS;
    const int first = blockIdx.x * NT * ITEMS;
    if (first >= limit) return;
    using Scan = cub::BlockScan<int, NT>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int ends[MAX_QUERIES], base, wrapped_head;
    const bool power_of_two = (capacity & (capacity - 1)) == 0;
    // Reuse the existing query-scan barrier to publish one reduction per CTA.
    if (t == 0 && !power_of_two) wrapped_head = history_cursor_slot(head, capacity);
    if constexpr (MAX_QUERIES > NT) {
      query_count_ends(counts, n, ends);
    } else {
      int end;
      Scan(scan).InclusiveSum(t < n ? counts[t] : 0, end);
      if (t < n) ends[t] = end;
      __syncthreads();
    }
    int flags[ITEMS], prefix[ITEMS], count;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      const int offset = first + (GLOBAL_STRIPED ? i * NT + t : t * ITEMS + i);
      flags[i] = offset < limit && pins[power_of_two
          ? int((head + offset) & (capacity - 1))
          : history_window_slot(wrapped_head, offset, capacity)] != now;
    }
    Scan(scan).ExclusiveSum(flags, prefix, count);
    if (t == 0) base = count ? atomicAdd(totals + 1, count) : 0;
    __syncthreads();
    int last = 0;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      const int ordinal = base + prefix[i];
      if (flags[i] && ordinal < total) {
        int lo = 0, hi = n;
        while (lo < hi) {
          const int mid = (lo + hi) / 2;
          if (ordinal < ends[mid])
            hi = mid;
          else
            lo = mid + 1;
        }
        const int q = lo, m = ordinal - (q ? ends[q - 1] : 0);
        const int offset = first + (GLOBAL_STRIPED ? i * NT + t : t * ITEMS + i);
        const int slot = power_of_two
            ? int((head + offset) & (capacity - 1))
            : history_window_slot(wrapped_head, offset, capacity);
        const int row = sources[long(q) * K + m];
        tags[slot] = row;
        pins[slot] = now;
        hint[row] = slot;
        targets[long(q) * K + m] = slot;
        last = max(last, offset + 1);
      }
    }
    last = __reduce_max_sync(FULL, last);
    if (lane == 0 && last)
      atomicMax(reinterpret_cast<unsigned long long*>(cursors + 1), static_cast<unsigned long long>(head + last));
    return;
  }
  constexpr int ITEMS = GLOBAL_ALLOC_ITEMS;
  __shared__ int cutoff, strict_total, ends[MAX_QUERIES], strict_base, tie_base;
  using Scan = cub::BlockScan<uint64_t, NT>;
  __shared__ typename Scan::TempStorage scan;
  if (t == 0) {
    cutoff = -1;
    strict_total = 0;
  }
  __syncthreads();
  if (t < 32) {
    int buckets[9], sum = 0;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const int h = 256 - lane * 9 - i;
      buckets[i] = h > 0 ? histogram[h] : 0;
      sum += buckets[i];
    }
    int inclusive = sum;
    for (int d = 1; d < 32; d *= 2) {
      const int v = __shfl_up_sync(FULL, inclusive, d);
      if (lane >= d) inclusive += v;
    }
    int before = inclusive - sum;
    if (before < total && inclusive >= total) {
#pragma unroll
      for (int i = 0; i < 9; ++i) {
        if (before < total && before + buckets[i] >= total) {
          cutoff = 256 - lane * 9 - i;
          strict_total = before;
        }
        before += buckets[i];
      }
    }
    if constexpr (MAX_QUERIES == 32) {
      int end = t < n ? counts[t] : 0;
      for (int d = 1; d < 32; d *= 2) {
        const int v = __shfl_up_sync(FULL, end, d);
        if (t >= d) end += v;
      }
      ends[t] = end;
    }
  }
  if constexpr (MAX_QUERIES > NT) {
    query_count_ends(counts, n, ends);
  } else if constexpr (MAX_QUERIES != 32) {
    using QueryScan = cub::BlockScan<int, NT>;
    __shared__ typename QueryScan::TempStorage query_scan;
    int end;
    QueryScan(query_scan).InclusiveSum(t < n ? counts[t] : 0, end);
    if (t < MAX_QUERIES) ends[t] = end;
  }
  __syncthreads();
  if (cutoff < 1) {
    if (t == 0) atomicAdd(errors, 1);
    asm volatile("trap;");
  }
  uint64_t flags[ITEMS], prefix[ITEMS], count;
  int buckets[ITEMS];
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int slot = blockIdx.x * NT * ITEMS + (GLOBAL_STRIPED ? i * NT + t : t * ITEMS + i);
    buckets[i] = slot < capacity ? lru_bucket(tags[slot], pins[slot], now) : 0;
    flags[i] = uint64_t(buckets[i] > cutoff) + (uint64_t(buckets[i] == cutoff) << 32);
  }
  Scan(scan).ExclusiveSum(flags, prefix, count);
  if (t == 0) {
    strict_base = uint32_t(count) ? atomicAdd(totals + 1, uint32_t(count)) : 0;
    tie_base = uint32_t(count >> 32) ? atomicAdd(histogram + 257, uint32_t(count >> 32)) : 0;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int slot = blockIdx.x * NT * ITEMS + (GLOBAL_STRIPED ? i * NT + t : t * ITEMS + i);
    const bool strict = buckets[i] > cutoff, tie = buckets[i] == cutoff;
    const int ordinal =
        strict ? strict_base + int(uint32_t(prefix[i])) : strict_total + tie_base + int(uint32_t(prefix[i] >> 32));
    if ((strict || tie) && ordinal < total) {
      int q = 0, m = ordinal;
      // The overwhelmingly common low-QPS case has one logical query.
      // Its global miss ordinal already IS its local miss index.
      if (!GLOBAL_SINGLE_QUERY_ALLOC || n != 1) {
        int lo = 0, hi = n;
        while (lo < hi) {
          const int mid = (lo + hi) / 2;
          if (ordinal < ends[mid])
            hi = mid;
          else
            lo = mid + 1;
        }
        q = lo;
        m = ordinal - (q ? ends[q - 1] : 0);
      }
      if (q >= n || m < 0 || m >= counts[q] || pins[slot] == now) {
        atomicAdd(errors, 1);
        asm volatile("trap;");
      }
      const int row = sources[long(q) * K + m];
      tags[slot] = row;
      pins[slot] = now;
      hint[row] = slot;
      targets[long(q) * K + m] = slot;
    }
  }
  if (blockIdx.x == 0 && t == 0) cursors[1] = capacity;
}
extern "C" int flowkv_global_plan(
    const int* idx,
    const int64_t* step,
    const int* active,
    const int* keys,
    const int* values,
    int* tags,
    int64_t* pins,
    int* hint,
    int* remap,
    int* sources,
    int* targets,
    int* counts,
    int* bases,
    int* free_offsets,
    int64_t* cursors,
    int* totals,
    int* errors,
    int* free_counts,
    int batch,
    int capacity,
    long nrows,
    cudaStream_t stream,
    bool lookup_done
#if FLOW_RECORD_HOST_TIER
    , int64_t* selected_ready = nullptr, const int64_t* host_group = nullptr
#endif
    ) {
  // The history union is at most batch*K rows. Current candidate rows use
  // dedicated slots outside this arena. Hits are pinned before any eviction.
  if (batch < 1 || batch > MAX_QUERIES || capacity < batch * K)
    return int(cudaErrorInvalidValue);
  if (!lookup_done) {
    const auto rc = cache_launch(
        dim3(batch * (K / NT)),
        dim3(NT),
        stream,
        global_lookup_pin<false>,
        idx,
        step,
        active,
        keys,
        values,
        tags,
        pins,
        hint,
        remap,
        sources,
        counts,
        bases,
        totals,
        cursors,
        capacity,
        nrows,
        static_cast<int*>(nullptr), 0, 0, 1);
    if (rc != cudaSuccess) return int(rc);
  }
  const bool circular = batch <= FLOW_CIRCULAR_MAX_QUERIES && capacity >= 2 * batch * K;
  int* histogram = free_offsets;
  if (!circular && !FLOW_PARALLEL_CIRCULAR) {
    const auto rc = cache_launch(
        dim3(batch < 16 ? GLOBAL_SMALL_HIST_BLOCKS : GLOBAL_HIST_BLOCKS),
        dim3(NT),
        stream,
        global_lru_histogram,
        tags,
        pins,
        step,
        totals,
        histogram,
        cursors,
        capacity);
    if (rc != cudaSuccess) return int(rc);
  }
  {
    const int scan_slots = FLOW_PARALLEL_CIRCULAR ? min(capacity, batch * K) : capacity;
    const auto rc = cache_launch(
        // The append path assigns one CTA per query; the eviction path scans
        // ITEMS slots per thread. Cover both, including sub-tile capacities.
        dim3(
          circular ? batch : max(batch, (scan_slots + NT * GLOBAL_ALLOC_ITEMS - 1) / (NT * GLOBAL_ALLOC_ITEMS))),
        dim3(NT),
        stream,
        circular ? global_allocate<1> : (FLOW_PARALLEL_CIRCULAR ? global_allocate<2> : global_allocate<0>),
        sources,
        targets,
        counts,
        active,
        tags,
        hint,
        pins,
        step,
        cursors,
        totals,
        histogram,
        capacity,
        errors
#if FLOW_RECORD_HOST_TIER
        , selected_ready, host_group
#endif
        );
    if (rc != cudaSuccess) return int(rc);
  }
  #if FLOW_RECORD_HOST_TIER
  // The existing lookup/list publication stays inside the native allocator.
  // Complete that grid before the owner consumes any peer miss list; captured
  // batch sets the generic query grid, independent of the allocation Qmax.
  if (host_group) {
#if !UDCACHE_NAIVE_LRU
    if (batch < 8)
      sglang::flowkv_host::prepare_actual_misses_and_finish<<<FLOW_RECORD_PEERS * batch * 4, 256, 0, stream>>>(
          host_group, step, active, batch, 4);
    else
#endif
    sglang::flowkv_host::prepare_actual_misses<<<FLOW_RECORD_PEERS * batch * 4, 256, 0, stream>>>(
        host_group, step, active, batch, 4);
  }
  #endif
  return int(cudaGetLastError());
}

// Native packed-page geometry shared with the record transport.
#ifndef FLOW_C4_PAGE_SIZE
#define FLOW_C4_PAGE_SIZE 1
#endif
#ifndef FLOW_C4_BYTES_PER_PAGE
#define FLOW_C4_BYTES_PER_PAGE FLOW_ROW_BYTES
#endif

__device__ __forceinline__ long c4_byte_offset(long row, int byte) {
  // FlashMLA stores all 576-byte value records first, followed by the
  // page's 8-byte scale records. A logical 584-byte row is not contiguous.
  constexpr int value_bytes = FLOW_ROW_BYTES - 8;
  const long page = (row / FLOW_C4_PAGE_SIZE) * long(FLOW_C4_BYTES_PER_PAGE);
  const int offset = row % FLOW_C4_PAGE_SIZE;
  return page + (byte < value_bytes ? offset * value_bytes + byte
                                    : FLOW_C4_PAGE_SIZE * value_bytes + offset * 8 + byte - value_bytes);
}

struct C4PullCacheState {
  int* tags;
  int64_t* pins;
  int64_t* cursors;
  const int* totals;
  const int64_t* step;
  const int* active;
  const int* current_keys;
  const int* current_values;
};

#include <sgl_kernel/utils.cuh>

namespace sglang {
using tvm::ffi::TensorView;

void flowkv_begin(
    TensorView current,
    TensorView valid,
    int64_t groups,
    int64_t pool_rows,
    TensorView step,
    TensorView active,
    TensorView keys,
    TensorView values,
    TensorView cursors,
    TensorView totals,
    TensorView hints,
    TensorView counts,
    TensorView histogram) {
  auto stream = host::LaunchKernel::resolve_device(current.device());
  host::RuntimeDeviceCheck(
      static_cast<cudaError_t>(flowkv_global_begin(
          current.data_ptr(),
          current.dtype().bits / 8,
          valid.data_ptr(),
          valid.dtype().bits / 8,
          current.shape()[0],
          MAX_QUERIES,
          groups,
          pool_rows,
          static_cast<int64_t*>(step.data_ptr()),
          static_cast<int*>(active.data_ptr()),
          static_cast<int*>(keys.data_ptr()),
          static_cast<int*>(values.data_ptr()),
          static_cast<int64_t*>(cursors.data_ptr()),
          static_cast<int*>(totals.data_ptr()),
          static_cast<int*>(hints.data_ptr()),
          static_cast<int*>(counts.data_ptr()),
          static_cast<int*>(histogram.data_ptr()),
          stream)));
}

void flowkv_invalidate(TensorView rows, int64_t groups, int64_t pool_rows, TensorView hints) {
  auto stream = host::LaunchKernel::resolve_device(rows.device());
  host::RuntimeDeviceCheck(
      static_cast<cudaError_t>(flowkv_global_invalidate(
          rows.data_ptr(),
          rows.dtype().bits / 8,
          rows.shape()[0],
          groups,
          pool_rows,
          static_cast<int*>(hints.data_ptr()),
          stream)));
}

template <bool LookupDone>
void flowkv_plan(
    TensorView idx,
    TensorView step,
    TensorView active,
    TensorView keys,
    TensorView values,
    TensorView tags,
    TensorView pins,
    TensorView hint,
    TensorView remap,
    TensorView sources,
    TensorView targets,
    TensorView counts,
    TensorView histogram,
    TensorView cursors,
    TensorView totals,
    TensorView errors,
    int64_t capacity,
    int64_t pool_rows) {
  auto stream = host::LaunchKernel::resolve_device(idx.device());
  host::RuntimeDeviceCheck(
      static_cast<cudaError_t>(flowkv_global_plan(
          static_cast<const int*>(idx.data_ptr()),
          static_cast<const int64_t*>(step.data_ptr()),
          static_cast<const int*>(active.data_ptr()),
          static_cast<const int*>(keys.data_ptr()),
          static_cast<const int*>(values.data_ptr()),
          static_cast<int*>(tags.data_ptr()),
          static_cast<int64_t*>(pins.data_ptr()),
          static_cast<int*>(hint.data_ptr()),
          static_cast<int*>(remap.data_ptr()),
          static_cast<int*>(sources.data_ptr()),
          static_cast<int*>(targets.data_ptr()),
          static_cast<int*>(counts.data_ptr()),
          nullptr,
          static_cast<int*>(histogram.data_ptr()),
          static_cast<int64_t*>(cursors.data_ptr()),
          static_cast<int*>(totals.data_ptr()),
          static_cast<int*>(errors.data_ptr()),
          nullptr,
          idx.shape()[0],
          capacity,
          pool_rows,
          stream,
          LookupDone)));
}

}  // namespace sglang

// Complete-record format adapters reuse this same planner and transport.
#if defined(FLOW_RECORD_PEERS) && defined(FLOW_RECORD_SLOTS)
// Physical IDs have contiguous rank intervals. Sequence ownership comes from
// native page offsets, never from physical gid % N. M and N are compiled from
// the immutable allocation so non-power-of-two M uses a constant fast divide.
__device__ __forceinline__ int record_global_slot(int logical) {
  return (logical % FLOW_RECORD_PEERS) * FLOW_RECORD_SLOTS + logical / FLOW_RECORD_PEERS;
}
__device__ __forceinline__ int record_owner(int gid) { return gid / FLOW_RECORD_SLOTS; }
__device__ __forceinline__ int record_local(int gid) { return gid % FLOW_RECORD_SLOTS; }

#ifndef FLOW_RECORD_REQUEST_TIER
#define FLOW_RECORD_REQUEST_TIER 0
#endif
#if FLOW_RECORD_REQUEST_TIER
#include <sgl_kernel/flowkv/request_page_map.cuh>
__device__ __forceinline__ int64_t record_request_local(int gid, const int32_t* page_map) {
  return sglang::flowkv::request_physical_row(record_local(gid), page_map,
      FLOW_RECORD_SLOTS, FLOW_RECORD_GPU_SLOTS);
}
#endif

#ifndef FLOW_RECORD_HOST_TIER
#define FLOW_RECORD_HOST_TIER 0
#endif
#if FLOW_RECORD_HOST_TIER
#include "csrc/cache/owner_directory.cuh"
#define FLOW_HOST_DEVICE_PARAM , const int64_t* host
#define FLOW_HOST_DEVICE_ARG , host
#define FLOW_HOST_TENSOR_PARAM , TensorView host_descriptor
#define FLOW_HOST_TENSOR_ARG , host_descriptor
#define FLOW_HOST_LAUNCH_ARG , static_cast<const int64_t*>(host_descriptor.data_ptr())
#elif FLOW_RECORD_REQUEST_TIER
#define FLOW_HOST_DEVICE_PARAM , const int32_t* request_page_map
#define FLOW_HOST_DEVICE_ARG , request_page_map
#define FLOW_HOST_TENSOR_PARAM , TensorView request_page_map
#define FLOW_HOST_TENSOR_ARG , request_page_map
#define FLOW_HOST_LAUNCH_ARG , static_cast<const int32_t*>(request_page_map.data_ptr())
#else
#define FLOW_HOST_DEVICE_PARAM
#define FLOW_HOST_DEVICE_ARG
#define FLOW_HOST_TENSOR_PARAM
#define FLOW_HOST_TENSOR_ARG
#define FLOW_HOST_LAUNCH_ARG
#endif



template<bool History, bool Current, bool UpdateMetadata, class Page>
__device__ __forceinline__ void record_pull_cached_impl(
    const uint64_t* peers, const int* sources, const int* targets,
    const int* counts, const int* idx, const int* hints, int* remap,
    const uint8_t* staging, const int* current, int capacity,
    uint8_t* resident, long layer_offset, int npeers, int rank,
    bool sole_producer, C4PullCacheState cache FLOW_HOST_DEVICE_PARAM) {
  if constexpr (FLOW_PDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if constexpr (FLOW_PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  const int b = blockIdx.y, lane = threadIdx.x & 31;
  const int warp = blockIdx.x * 4 + (threadIdx.x >> 5);
  const int nwarps = gridDim.x * 4, count = History ? counts[b] : 0;
#if FLOW_RECORD_HOST_TIER
  // Residency publication is carried by each source directory entry.
#endif
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (!UpdateMetadata && !Current) {
    // A follower has no remap/current work. Its empty miss warps retire
    // without keeping the anchor's TopK-sized metadata grid alive.
    if (warp >= count) return;
  } else {
    if (linear >= K && linear * 8 >= FLOW_ROW_BYTES && warp >= count) return;
  }

  const int live = *cache.active;
  const int64_t append_base = cache.cursors[0] + cache.totals[0];
  const bool retain = append_base + live <= capacity;
  if (History && UpdateMetadata && retain && b == 0 && blockIdx.x == 0 && threadIdx.x == 0)
    cache.cursors[1] = append_base + live;
  if (History && UpdateMetadata && retain && b < live && current[b] > 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    cache.tags[append_base + b] = current[b];
    cache.pins[append_base + b] = *cache.step;
    unsigned h = hash_row(current[b]);
    for (int probe = 0; probe < HASH; ++probe) {
      const int key = __ldg(cache.current_keys + h);
      if (key == current[b]) {
        if (__ldg(cache.current_values + h) == b)
          cuda::atomic_ref<int, cuda::thread_scope_device>(const_cast<int*>(hints)[current[b]])
              .store(int(append_base) + b, cuda::memory_order_relaxed);
        break;
      }
      if (key < 0) break;
      h = (h + 1) & (HASH - 1);
    }
  }
  if (History && UpdateMetadata && linear < K && remap[long(b) * K + linear] == -2)
    remap[long(b) * K + linear] = hints[idx[long(b) * K + linear]];

  const int gid = current[b];
  if (Current && gid > 0 && b < live) {
    const int owner = record_owner(gid);
    const bool publish = sole_producer || owner == rank;
    #if FLOW_RECORD_HOST_TIER
    uint8_t* cold = publish ? record_host_address(peers, gid, layer_offset, host, rank, true) : nullptr;
#elif FLOW_RECORD_REQUEST_TIER
    uint8_t* cold = reinterpret_cast<uint8_t*>(peers[owner] + layer_offset)
                    + record_request_local(gid, request_page_map) * FLOW_ROW_BYTES;
#else
    uint8_t* cold = reinterpret_cast<uint8_t*>(peers[owner] + layer_offset)
                    + long(record_local(gid)) * FLOW_ROW_BYTES;
#endif
    bool wrote = false;
    for (int word = linear; word < FLOW_ROW_BYTES / 8; word += gridDim.x * blockDim.x) {
      const uint64_t value = *reinterpret_cast<const uint64_t*>(staging + Page::offset(b, word * 8));
      *reinterpret_cast<uint64_t*>(resident + Page::offset(capacity + b, word * 8)) = value;
      if (retain)
        *reinterpret_cast<uint64_t*>(resident + Page::offset(append_base + b, word * 8)) = value;
      if (publish) { reinterpret_cast<uint64_t*>(cold)[word] = value; wrote = true; }
    }
    if (wrote) __threadfence_system();
  }

  // Issue independent peer loads before storing their results, as in the
  // validated shared-Index pull. Eight-byte words cover the entire packed
  // row without reading beyond its value and scale regions.
  constexpr int kPipe = 4;
  constexpr int kWords = FLOW_ROW_BYTES / sizeof(uint64_t);
  constexpr int kWordsPerLane = (kWords + 31) / 32;
  for (int base = warp; History && base < count; base += nwarps * kPipe) {
    uint64_t values[kPipe][kWordsPerLane];
    int slots[kPipe];
    #pragma unroll
    for (int j = 0; j < kPipe; ++j) {
      const int miss = base + j * nwarps;
      slots[j] = -1;
      if (miss < count) {
        const int source = sources[long(b) * K + miss];
        slots[j] = targets[long(b) * K + miss];
        #if FLOW_RECORD_HOST_TIER
        auto* physical_rows = reinterpret_cast<int*>(host[HostMissPhysical]);
        const uint64_t* src;
        if constexpr (UpdateMetadata) {
          src = reinterpret_cast<const uint64_t*>(
              record_host_address(peers, source, layer_offset, host, rank, false));
          if (lane == 0) {
            const int physical = (reinterpret_cast<uint64_t>(src)
                - peers[record_owner(source)] - layer_offset) / FLOW_ROW_BYTES;
            physical_rows[long(b) * K + miss] = physical;
            record_host_touch(source, host, physical);
          }
        } else {
          // The anchor's completed pull precedes every follower in this
          // shared-Index group. Its pinned source cannot move this epoch.
          const int physical = __ldg(physical_rows + long(b) * K + miss);
          assert(physical >= 0 && physical < host[HostGpuRows]);
          src = reinterpret_cast<const uint64_t*>(
              peers[record_owner(source)] + layer_offset) + long(physical) * kWords;
        }
#elif FLOW_RECORD_REQUEST_TIER
        const uint64_t* src = reinterpret_cast<const uint64_t*>(
            peers[record_owner(source)] + layer_offset)
            + record_request_local(source, request_page_map) * kWords;
#else
        const uint64_t* src = reinterpret_cast<const uint64_t*>(
            peers[record_owner(source)] + layer_offset)
            + long(record_local(source)) * kWords;
#endif
        #pragma unroll
        for (int word = 0; word < kWordsPerLane; ++word)
          if (lane + word * 32 < kWords)
            values[j][word] = src[lane + word * 32];
      }
    }
    #pragma unroll
    for (int j = 0; j < kPipe; ++j) {
      if (slots[j] < 0) continue;
      #pragma unroll
      for (int word = 0; word < kWordsPerLane; ++word)
        if (lane + word * 32 < kWords)
          *reinterpret_cast<uint64_t*>(resident + Page::offset(
              slots[j], (lane + word * 32) * sizeof(uint64_t))) = values[j][word];
    }
  }
}

#endif
