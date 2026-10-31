// SPDX-License-Identifier: Apache-2.0
#pragma once

// Keep the complete-record transport identical to HiSparse's pinned-host IO.
// Swap is the one extra case: hold a record chunk in registers before replacing
// its CPU slot with the victim.  No permanent host/GPU duplicate is introduced.
#include "csrc/cache/directory.cuh"
#include "csrc/transfer/record_exchange.cuh"
#include "csrc/eviction/presence_recent.cuh"

#include <assert.h>
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>
#include <limits.h>

#ifndef UDCACHE_HOST_EXCHANGE_BLOCKS
#define UDCACHE_HOST_EXCHANGE_BLOCKS 4
#endif
#ifndef UDCACHE_HOST_EXCHANGE_THREADS
#define UDCACHE_HOST_EXCHANGE_THREADS 1024
#endif
static_assert(UDCACHE_HOST_EXCHANGE_BLOCKS > 0);
static_assert(UDCACHE_HOST_EXCHANGE_THREADS > 0 && UDCACHE_HOST_EXCHANGE_THREADS <= 1024 &&
              UDCACHE_HOST_EXCHANGE_THREADS % 32 == 0);

#if CUDART_VERSION < 12040
#error "FlowKV CPU residency requires CUDA 12.4 or newer for conditional graphs"
#endif

namespace sglang::flowkv_host {

constexpr int kThreads = 256;
__device__ __forceinline__ int64_t read_row(const void* rows, int i, int bytes) {
  return bytes == 8 ? static_cast<const int64_t*>(rows)[i] : static_cast<const int*>(rows)[i];
}

__device__ __forceinline__ int local_row(int64_t global, int owner, int slots) {
  const int64_t first = int64_t(owner) * slots;
  return global > 0 && global >= first && global < first + slots ? int(global - first) : -1;
}

__device__ __forceinline__ int pop_free(int* stack, int* count) {
  int observed = atomicAdd(count, 0);
  while (observed > 0) {
    const int prior = atomicCAS(count, observed, observed - 1);
    if (prior == observed) return stack[observed - 1];
    observed = prior;
  }
  return -1;
}

__device__ __forceinline__ void push_free(int* stack, int* count, int capacity, int slot) {
  const int offset = atomicAdd(count, 1);
  assert(offset < capacity);
  stack[offset] = slot;
}

namespace ring = flowkv_bitmap_ring;
namespace recent = flowkv_ring_presence_recent;

// Native registration attaches at the actual current group epoch. It is a
// separate insert-only producer, completed before begin/prepare consumes it.
__device__ __forceinline__ void attach_registered(ring::Group const& g, int slot, int64_t age) {
  assert(slot >= 0 && slot < recent::capacity(g));
  assert(age >= 0 && age == *g.epoch);
#if !UDCACHE_NAIVE_LRU
  const int bid = int(age % g.recent_count);
  auto* label = reinterpret_cast<unsigned long long*>(g.bucket_epochs + bid);
  const auto prior = atomicCAS(label, ULLONG_MAX, static_cast<unsigned long long>(age));
  assert(prior == ULLONG_MAX || prior == static_cast<unsigned long long>(age));
  assert(atomicCAS(g.gpu_bucket + slot, ring::kFree, bid) == ring::kFree);
  flowkv_presence_bitmap::insert_mask_unique(g.buckets[bid], slot >> 5, uint32_t(1) << (slot & 31));
#endif
}

__device__ __forceinline__ void register_row(int64_t global, int owner, bool prefer_gpu, State s, const ring::Group* group) {
  const int row = local_row(global, owner, s.logical_slots);
  if (row < 0) return;
  // Register inputs are newly allocated unique rows.  A repeated row is an
  // allocator error, rather than a reason to silently reuse stale records.
  assert(s.directory[row] == -1);
  cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.access_time[row])
      .store(*s.epoch, cuda::memory_order_relaxed);
  int gpu = prefer_gpu ? pop_free(s.gpu_free, s.free_counts) : -1;
  if (gpu >= 0) {
    s.gpu_tags[gpu] = row;
    attach_registered(*group, gpu, *s.epoch);
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.directory[row])
        .store(gpu, cuda::memory_order_release);
  } else {
    int host = pop_free(s.host_free, s.free_counts + 1);
    if (host < 0 && !prefer_gpu) gpu = pop_free(s.gpu_free, s.free_counts);
    assert(host >= 0 || gpu >= 0);
    if (host >= 0) {
      s.host_tags[host] = row;
      cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.directory[row])
          .store(-int64_t(host) - 2, cuda::memory_order_release);
    } else {
      s.gpu_tags[gpu] = row;
      attach_registered(*group, gpu, *s.epoch);
      cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.directory[row])
          .store(gpu, cuda::memory_order_release);
    }
  }
}

__global__ void register_rows(const void* rows, int row_bytes, int count, int owner, bool prefer_gpu, State s, const ring::Group* group) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
    register_row(read_row(rows, i, row_bytes), owner, prefer_gpu, s, group);
}

// Keep the scalar 3837 GPU/CPU operation order. Ring membership changes
// before canonical-time dedup, including a legal stale-time native attach.
__device__ __forceinline__ void pin_row(int row, bool allow_new, State s, Workspace w) {
  if (row < 0) return;
  const int64_t epoch = *s.epoch;
  const int64_t location = s.directory[row];
  auto time = cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.access_time[row]);
  if (location >= 0) {
    assert(location < s.gpu_slots && s.gpu_tags[location] == row);
    recent::mark_slot(*reinterpret_cast<const ring::Group*>(w.histogram), int(location), ring::Operation::Access, epoch);
    if (time.load(cuda::memory_order_relaxed) != epoch) time.store(epoch, cuda::memory_order_relaxed);
    return;
  }
  if (time.load(cuda::memory_order_relaxed) == epoch) return;
  if (time.exchange(epoch, cuda::memory_order_relaxed) == epoch) return;
  const bool is_new = location == -1;
  assert(!is_new || allow_new);
  const int out = atomicAdd(w.counters + (is_new ? NewCount : PromoteCount), 1);
  assert(out < w.capacity);
  (is_new ? w.new_rows : w.promote_rows)[out] = !is_new && allow_new ? -row - 1 : row;
}

__device__ __forceinline__ void begin_current_body(
    const void* rows, int row_bytes, int queries, const int* real_queries,
    int owner, State s, Workspace w, recent::Storage<256>& storage) {
  assert(*s.epoch < LLONG_MAX);
  recent::advance<256>(*reinterpret_cast<const ring::Group*>(w.histogram), *s.epoch + 1, storage);
  if (threadIdx.x == 0) ++*s.epoch;
  if (threadIdx.x < 8) w.counters[threadIdx.x] = 0;
  if (threadIdx.x < 3) w.prefix[threadIdx.x] = 0;
  __syncthreads();
  for (int query = threadIdx.x; query < queries && query < *real_queries; query += blockDim.x)
    pin_row(local_row(read_row(rows, query, row_bytes), owner, s.logical_slots), true, s, w);
}

__global__ void begin_current(const void* rows, int row_bytes, int queries,
    const int* real_queries, int owner, State s, Workspace w) {
  __shared__ recent::Storage<256> storage;
  begin_current_body(rows, row_bytes, queries, real_queries, owner, s, w, storage);
}

__global__ void pin_rows(
    const void* rows, int row_bytes, int queries, int width, const int* real_queries,
    const int* selected_counts, bool use_counts, int owner, State s, Workspace w) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < queries * width; i += gridDim.x * blockDim.x) {
    const int query = i / width;
    if (query >= *real_queries || (use_counts && i % width >= selected_counts[query])) continue;
    pin_row(local_row(read_row(rows, i, row_bytes), owner, s.logical_slots), false, s, w);
  }
}

__device__ __forceinline__ bool reserve_counts(State s, Workspace w) {
  const int count = w.counters[NewCount] + w.counters[PromoteCount];
  assert(count <= w.capacity);
  const int free_used = min(count, s.free_counts[0]);
  w.counters[PlanCount] = count;
  w.counters[FreeUsed] = free_used;
  w.counters[VictimCount] = count - free_used;
  w.prefix[2] = count - free_used;
  s.free_counts[0] -= free_used;
  return count > free_used;
}

// Decode preparation is consumed only after its whole kernel completes.
__device__ __forceinline__ void finalize_pending_reserve(State s, Workspace w) {
  if (w.counters[PendingReserve] == 0) return;
  reserve_counts(s, w);
  w.counters[PendingReserve] = 0;
}

__global__ void reserve(State s, Workspace w, cudaGraphConditionalHandle evict) {
  if (threadIdx.x) return;
  const bool needs_eviction = reserve_counts(s, w);
  if (evict) {
    cudaGraphSetConditional(evict, needs_eviction);
  }
}

__global__ void publish_group(const int64_t* epoch, int64_t* ready) {
  cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*ready)
      .store(*epoch, cuda::memory_order_release);
}

// A zero upper bound makes this leading byte zero for every eligible key.
// Both radix kernels must skip together, leaving the prefix and rank unchanged.
__device__ __forceinline__ bool skip_epoch_byte(const int64_t* epoch, bool physical, int shift) {
  return !physical && (uint64_t(*epoch) >> shift) == 0;
}

// Physical keys are bounded for the lifetime of the pool, unlike epoch keys.
inline int first_physical_shift(int gpu_slots) {
  uint32_t highest_slot = uint32_t(gpu_slots - 1);
  int shift = 0;
  while ((highest_slot >>= 8) != 0) shift += 8;
  return shift;
}

__device__ __forceinline__ void assign_plan_item(State s, Workspace w, int i, int gpu) {
  const bool is_new = i < w.counters[NewCount];
  const int encoded = is_new ? w.new_rows[i] : w.promote_rows[i - w.counters[NewCount]];
  const bool current_cpu = !is_new && encoded < 0;
  const int row = current_cpu ? -encoded - 1 : encoded;
  const int old_row = s.gpu_tags[gpu];
  const int old_cpu = is_new ? -1 : int(-s.directory[row] - 2);
  const int cpu_src = is_new ? -1 : (current_cpu ? -old_cpu - 2 : old_cpu);
  const int cpu_dst = old_row < 0 ? -1 : (is_new ? pop_free(s.host_free, s.free_counts + 1) : old_cpu);
  assert(old_row < 0 || cpu_dst >= 0);
  w.plan[i] = row;
  w.plan[w.capacity + i] = gpu;
  w.plan[2 * w.capacity + i] = cpu_src;
  w.plan[3 * w.capacity + i] = cpu_dst;
  w.plan[4 * w.capacity + i] = old_row;
}

__device__ __forceinline__ void assign_plan_cta(State s, Workspace w) {
  const int count = w.counters[PlanCount];
  const int free_used = w.counters[FreeUsed];
  assert(w.counters[Candidates] == w.counters[VictimCount]);
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    // Allocate free GPU slots to new rows first.  Hence new GPU victims always
    // have a free CPU destination, without reusing another promotion's source.
    const int gpu = i < free_used ? s.gpu_free[s.free_counts[0] + i] : w.candidates[i - free_used];
    assign_plan_item(s, w, i, gpu);
  }
}

// Whole prepare completion precedes this owner-local CTA. Incoming aliases
// plan[0] only until finish's final barrier, before native assign overwrites it.
__device__ __forceinline__ void metadata_finish_body(
    State s, Workspace w, recent::Storage<256>& storage) {
#if UDCACHE_NAIVE_LRU
  assert(false && "Timestamp LRU requires its GPU full-pool selector");
#else
  if (threadIdx.x == 0) { reserve_counts(s, w); w.counters[PendingReserve] = 0; }
  __syncthreads();
  // No incoming rows means no ring transfer or victim selection. Keep the
  // reservation outputs and scratch counters exact for the child selector.
  if (w.counters[PlanCount] == 0) {
    if (threadIdx.x == 0) { w.counters[Candidates] = 0; w.counters[LeaseCount] = 0; }
    return;
  }
  const auto& g = *reinterpret_cast<const ring::Group*>(w.histogram);
  ring::Transaction tx{ring::Operation::Access, *s.epoch, nullptr, 0,
      s.gpu_free + s.free_counts[0], w.counters[FreeUsed], w.counters[VictimCount]};
  recent::finish<256>(g, tx, storage);
  if (threadIdx.x == 0) { w.counters[Candidates] = tx.victim_count; w.counters[LeaseCount] = 0; }
  __syncthreads();
  assign_plan_cta(s, w);
#endif
}

__global__ void metadata_finish(State s, Workspace w) {
  __shared__ recent::Storage<256> storage;
  metadata_finish_body(s, w, storage);
}

__global__ void commit(State s, Workspace w) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < w.counters[PlanCount]; i += gridDim.x * blockDim.x) {
    const int row = w.plan[i], gpu = w.plan[w.capacity + i];
    const int src = w.plan[2 * w.capacity + i], dst = w.plan[3 * w.capacity + i];
    const int old_row = w.plan[4 * w.capacity + i];
    if (old_row >= 0) {
      assert(dst >= 0 && s.gpu_tags[gpu] == old_row && s.directory[old_row] == gpu);
      cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.directory[old_row])
          .store(-int64_t(dst) - 2, cuda::memory_order_relaxed);
      s.host_tags[dst] = old_row;
    } else if (src != -1) {
      const int released_cpu = src >= 0 ? src : -src - 2;
      s.host_tags[released_cpu] = -1;
      push_free(s.host_free, s.free_counts + 1, s.host_slots, released_cpu);
    }
    s.gpu_tags[gpu] = row;
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.directory[row])
        .store(gpu, cuda::memory_order_release);
  }
}

__global__ void release_leases(int* leases, const int* leased, int* counters, int capacity) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < counters[LeaseCount]; i += blockDim.x * gridDim.x) {
    assert(i < capacity);
    const int before = atomicSub(leases + leased[i], 1);
    assert(before > 0);
  }
}

__device__ __forceinline__ void wait_readers(State s, const uint64_t* ready_peers,
    int peers, int offset) {
  if (threadIdx.x == 0) {
    const uint64_t started = clock64();
    for (int peer = 0; peer < peers; ++peer) {
      auto* ready = reinterpret_cast<int64_t*>(ready_peers[peer]) + offset;
      while (cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*ready)
                 .load(cuda::memory_order_acquire) < *s.epoch) {
        assert(clock64() - started < 100000000000ULL);
        __nanosleep(64);
      }
    }
  }
  __syncthreads();
}

__global__ void drain_readers(State s, const uint64_t* ready_peers, int peers, int offset) {
  wait_readers(s, ready_peers, peers, offset);
}

__global__ void release_rows(const void* rows, int bytes, int count, int owner, State s, const ring::Group* group,
    const uint64_t* ready_peers = nullptr, int peers = 0, int ready_offset = 0) {
  if (ready_peers) wait_readers(s, ready_peers, peers, ready_offset);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    const int row = local_row(read_row(rows, i, bytes), owner, s.logical_slots);
    if (row < 0) continue;
    const int64_t location = s.directory[row];
    // Native allocators release whole pages; unwritten tail padding never
    // acquired a physical GPU/CPU binding and has nothing to release.
    if (location == -1) continue;
    if (location >= 0) {
      assert(s.gpu_tags[location] == row);
      recent::mark_slot(*group, int(location), ring::Operation::Detach, *s.epoch);
      s.gpu_tags[location] = -1;
      push_free(s.gpu_free, s.free_counts, s.gpu_slots, int(location));
    } else {
      const int host = int(-location - 2);
      assert(s.host_tags[host] == row);
      s.host_tags[host] = -1;
      push_free(s.host_free, s.free_counts + 1, s.host_slots, host);
    }
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.directory[row])
        .store(-1, cuda::memory_order_release);
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.access_time[row])
        .store(-1, cuda::memory_order_relaxed);
  }
}

struct Capture {
  cudaGraph_t graph{};
  const cudaGraphNode_t* dependencies{};
  size_t count{};
};

inline Capture capture_info(cudaStream_t stream) {
  Capture capture;
  cudaStreamCaptureStatus status{};
#if CUDART_VERSION >= 13000
  host::RuntimeDeviceCheck(cudaStreamGetCaptureInfo(
      stream, &status, nullptr, &capture.graph, &capture.dependencies, nullptr, &capture.count));
#else
  host::RuntimeDeviceCheck(cudaStreamGetCaptureInfo_v2(
      stream, &status, nullptr, &capture.graph, &capture.dependencies, &capture.count));
#endif
  if (status == cudaStreamCaptureStatusInvalidated)
    host::RuntimeDeviceCheck(cudaErrorStreamCaptureInvalidated);
  return capture;
}

struct IfBody {
  cudaGraphNode_t node;
  cudaGraph_t body;
};

inline IfBody add_if(cudaGraph_t graph, const cudaGraphNode_t* dependencies,
                     size_t count, cudaGraphConditionalHandle handle) {
  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
  params.conditional.size = 1;
  cudaGraphNode_t node;
#if CUDART_VERSION >= 13000
  host::RuntimeDeviceCheck(cudaGraphAddNode(&node, graph, dependencies, nullptr, count, &params));
#else
  host::RuntimeDeviceCheck(cudaGraphAddNode(&node, graph, dependencies, count, &params));
#endif
  return {node, params.conditional.phGraph_out[0]};
}

inline void set_frontier(cudaStream_t stream, cudaGraphNode_t node) {
#if CUDART_VERSION >= 13000
  host::RuntimeDeviceCheck(cudaStreamUpdateCaptureDependencies(
      stream, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
#else
  host::RuntimeDeviceCheck(cudaStreamUpdateCaptureDependencies(
      stream, &node, 1, cudaStreamSetCaptureDependencies));
#endif
}

template <typename Kernel, typename... Args>
inline cudaGraphNode_t add_kernel(cudaGraph_t graph, cudaGraphNode_t previous, dim3 grid,
                                  dim3 block, Kernel kernel, Args... arguments) {
  void* pointers[]{static_cast<void*>(&arguments)...};
  cudaKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(kernel);
  params.gridDim = grid;
  params.blockDim = block;
  params.kernelParams = pointers;
  cudaGraphNode_t node;
  host::RuntimeDeviceCheck(cudaGraphAddKernelNode(
      &node, graph, previous ? &previous : nullptr, previous ? 1 : 0, &params));
  return node;
}

#if UDCACHE_NAIVE_LRU
#include "csrc/eviction/naive_lru.cuh"
#endif

inline void finish_prepare(cudaStream_t stream, State s, Workspace w) {
#if UDCACHE_NAIVE_LRU
  naive_finish_select(stream, s, w);
  naive_assign_cta<<<1, 256, 0, stream>>>(s, w);
#else
  metadata_finish<<<1, 256, 0, stream>>>(s, w);
#endif
  host::RuntimeDeviceCheck(cudaGetLastError());
}

__global__ void clear_ring(const ring::Group* group) {
  const auto& g = *group;
#if UDCACHE_NAIVE_LRU
  g.naive_histogram[threadIdx.x] = 0;
#else
  const int x = recent::capacity(g);
  for (int bucket = 0; bucket <= g.recent_count; ++bucket) {
    const auto& v = g.buckets[bucket];
    for (int64_t i = threadIdx.x; i < flowkv_presence_bitmap::leaf_words(v); i += blockDim.x) v.bits[i] = 0;
    for (int64_t i = threadIdx.x; i < flowkv_presence_bitmap::summary_words(v); i += blockDim.x) v.summary[i] = 0;
  }
  for (int64_t i = threadIdx.x; i < x; i += blockDim.x) g.gpu_bucket[i] = ring::kFree;
  for (int i = threadIdx.x; i < g.recent_count; i += blockDim.x) g.bucket_epochs[i] = -1;
#endif
}

}  // namespace sglang::flowkv_host

namespace sglang {
using tvm::ffi::TensorView;

#define FLOW_HOST_STATE_ARGS \
  TensorView directory, TensorView gpu_tags, TensorView host_tags, TensorView access_time, \
  TensorView leases, TensorView epoch, TensorView gpu_free, \
  TensorView host_free, TensorView free_counts
#define FLOW_HOST_WORK_ARGS \
  TensorView new_rows, TensorView promote_rows, TensorView plan, TensorView counters, \
  TensorView histogram, TensorView selection_prefix, TensorView candidate_slots, TensorView leased_slots
#define FLOW_HOST_STATE \
  flowkv_host::State{static_cast<int64_t*>(directory.data_ptr()), static_cast<int*>(gpu_tags.data_ptr()), \
    static_cast<int*>(host_tags.data_ptr()), static_cast<int64_t*>(access_time.data_ptr()), \
    static_cast<int*>(leases.data_ptr()), \
    static_cast<int64_t*>(epoch.data_ptr()), static_cast<int*>(gpu_free.data_ptr()), \
    static_cast<int*>(host_free.data_ptr()), static_cast<int*>(free_counts.data_ptr()), \
    int(directory.shape()[0]), int(gpu_tags.shape()[0]), int(host_tags.shape()[0])}
#define FLOW_HOST_WORK \
  flowkv_host::Workspace{static_cast<int*>(new_rows.data_ptr()), static_cast<int*>(promote_rows.data_ptr()), \
    static_cast<int*>(plan.data_ptr()), static_cast<int*>(counters.data_ptr()), \
    static_cast<int*>(histogram.data_ptr()), static_cast<int64_t*>(selection_prefix.data_ptr()), \
    static_cast<int*>(candidate_slots.data_ptr()), static_cast<int*>(leased_slots.data_ptr()), int(new_rows.shape()[0])}

void flowkv_host_register(TensorView rows, int64_t owner, bool prefer_gpu, TensorView manager, FLOW_HOST_STATE_ARGS) {
  const auto stream = host::LaunchKernel::resolve_device(directory.device());
  const int count = int(rows.shape()[0]);
  if (count) flowkv_host::register_rows<<<std::min(128, (count + 255) / 256), 256, 0, stream>>>(
      rows.data_ptr(), rows.dtype().bits / 8, count, owner, prefer_gpu, FLOW_HOST_STATE, static_cast<const flowkv_host::ring::Group*>(manager.data_ptr()));
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_host_prepare(TensorView selected, TensorView current, TensorView num_real_queries,
                         TensorView selected_counts, bool use_counts, int64_t owner,
                         FLOW_HOST_STATE_ARGS, FLOW_HOST_WORK_ARGS) {
  const auto stream = host::LaunchKernel::resolve_device(directory.device());
  const auto s = FLOW_HOST_STATE;
  const auto w = FLOW_HOST_WORK;
  const int queries = int(selected.shape()[0]), width = int(selected.shape()[1]);
  const auto* real = static_cast<const int*>(num_real_queries.data_ptr());
  const auto* lengths = static_cast<const int*>(selected_counts.data_ptr());
  flowkv_host::begin_current<<<1, 256, 0, stream>>>(
      current.data_ptr(), current.dtype().bits / 8, queries, real, owner, s, w);
  if (queries) {
    if (width) flowkv_host::pin_rows<<<std::min(128, (queries * width + 255) / 256), 256, 0, stream>>>(
        selected.data_ptr(), selected.dtype().bits / 8, queries, width, real, lengths, use_counts, owner, s, w);
  }
  flowkv_host::finish_prepare(stream, s, w);
}

void flowkv_host_exchange(TensorView host_cache, TensorView gpu_cache, TensorView plan, TensorView counters,
                          int64_t item_size_bytes, int64_t host_stride_bytes, int64_t gpu_stride_bytes,
                          int64_t layers, int64_t host_layer_stride, int64_t gpu_layer_stride) {
  const auto stream = host::LaunchKernel::resolve_device(gpu_cache.device());
  // The default preserves HiSparse's side-stream grid. Explicit startup
  // variants tune memory-level parallelism without changing the transfer plan.
  flowkv_host::exchange<<<UDCACHE_HOST_EXCHANGE_BLOCKS, UDCACHE_HOST_EXCHANGE_THREADS, 0, stream>>>(
      static_cast<char*>(host_cache.data_ptr()), static_cast<char*>(gpu_cache.data_ptr()),
      static_cast<const int*>(plan.data_ptr()), static_cast<const int*>(counters.data_ptr()),
      int(plan.shape()[1]), item_size_bytes, host_stride_bytes, gpu_stride_bytes,
      int(layers), host_layer_stride, gpu_layer_stride);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_host_commit(FLOW_HOST_STATE_ARGS, FLOW_HOST_WORK_ARGS) {
  const auto stream = host::LaunchKernel::resolve_device(directory.device());
  const auto w = FLOW_HOST_WORK;
  flowkv_host::commit<<<std::min(128, (w.capacity + 255) / 256), 256, 0, stream>>>(FLOW_HOST_STATE, w);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_host_release_leases(TensorView leases, TensorView leased_slots, TensorView counters) {
  const auto stream = host::LaunchKernel::resolve_device(leases.device());
  const int capacity = int(leased_slots.shape()[0]);
  flowkv_host::release_leases<<<std::min(128, (capacity + 255) / 256), 256, 0, stream>>>(
      static_cast<int*>(leases.data_ptr()), static_cast<const int*>(leased_slots.data_ptr()),
      static_cast<int*>(counters.data_ptr()), capacity);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_host_release_rows(TensorView rows, int64_t owner, TensorView manager, FLOW_HOST_STATE_ARGS) {
  const auto stream = host::LaunchKernel::resolve_device(directory.device());
  const int count = int(rows.shape()[0]);
  if (count) flowkv_host::release_rows<<<std::min(128, (count + 255) / 256), 256, 0, stream>>>(
      rows.data_ptr(), rows.dtype().bits / 8, count, owner, FLOW_HOST_STATE, static_cast<const flowkv_host::ring::Group*>(manager.data_ptr()));
  host::RuntimeDeviceCheck(cudaGetLastError());
}


void flowkv_host_release_rows_retired(TensorView rows, int64_t owner,
    TensorView ready_peers, int64_t ready_offset, TensorView manager, FLOW_HOST_STATE_ARGS) {
  const auto stream = host::LaunchKernel::resolve_device(directory.device());
  const int count = int(rows.shape()[0]);
  if (count) flowkv_host::release_rows<<<std::min(128, (count + 255) / 256), 256, 0, stream>>>(
      rows.data_ptr(), rows.dtype().bits / 8, count, owner, FLOW_HOST_STATE, static_cast<const flowkv_host::ring::Group*>(manager.data_ptr()),
      static_cast<const uint64_t*>(ready_peers.data_ptr()), int(ready_peers.shape()[0]), int(ready_offset));
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_host_drain_readers(TensorView ready_peers, int64_t ready_offset, FLOW_HOST_STATE_ARGS) {
  const auto stream = host::LaunchKernel::resolve_device(directory.device());
  flowkv_host::drain_readers<<<1, 256, 0, stream>>>(FLOW_HOST_STATE,
      static_cast<const uint64_t*>(ready_peers.data_ptr()), int(ready_peers.shape()[0]), int(ready_offset));
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_host_clear_ring(TensorView manager) {
  TVM_FFI_ICHECK(manager.dtype().bits == 64 && manager.ndim() == 2 && manager.shape()[0] == 1 &&
                manager.shape()[1] == sizeof(flowkv_host::ring::Group) / sizeof(int64_t));
  const auto stream = host::LaunchKernel::resolve_device(manager.device());
  flowkv_host::clear_ring<<<1, 256, 0, stream>>>(static_cast<const flowkv_host::ring::Group*>(manager.data_ptr()));
  host::RuntimeDeviceCheck(cudaGetLastError());
}

#undef FLOW_HOST_STATE_ARGS
#undef FLOW_HOST_WORK_ARGS
#undef FLOW_HOST_STATE
#undef FLOW_HOST_WORK
}  // namespace sglang
