// CPU residency consumes the existing actual miss lists.
#pragma once
#include "csrc/cache/cpu_residency.cuh"

namespace sglang::flowkv_host {
enum FrameField {
  FDirectory, FGpuTags, FHostTags, FAccessTime, FLeases, FEpoch, FGpuFree, FHostFree, FFreeCounts, FNewRows, FPromoteRows, FPlan, FCounters, FHistogram, FPrefix, FCandidates, FLeasedSlots, FMissRowsPeers, FMissCountsPeers, FMissReadyPeers, FGroup, FReady, FGpuSlots, FHostSlots, FCapacity, FRank, FPeers, FMaxQueries, FTopK, FColdGraph, FDeferredEpoch, FFrameWords
};
constexpr int kFrameWords = FFrameWords;

struct Frame {
  const int64_t* d;
  __device__ __forceinline__ State state() const {
    return {reinterpret_cast<int64_t*>(d[FDirectory]), reinterpret_cast<int*>(d[FGpuTags]),
      reinterpret_cast<int*>(d[FHostTags]), reinterpret_cast<int64_t*>(d[FAccessTime]),
      reinterpret_cast<int*>(d[FLeases]),
      reinterpret_cast<int64_t*>(d[FEpoch]), reinterpret_cast<int*>(d[FGpuFree]),
      reinterpret_cast<int*>(d[FHostFree]), reinterpret_cast<int*>(d[FFreeCounts]),
      int(d[FGpuSlots] + d[FHostSlots]), int(d[FGpuSlots]), int(d[FHostSlots])};
  }
  __device__ __forceinline__ const ring::Group& group() const {
    return *reinterpret_cast<const ring::Group*>(d[FHistogram]);
  }
  __device__ __forceinline__ Workspace work() const {
    return {reinterpret_cast<int*>(d[FNewRows]), reinterpret_cast<int*>(d[FPromoteRows]),
      reinterpret_cast<int*>(d[FPlan]), reinterpret_cast<int*>(d[FCounters]),
      reinterpret_cast<int*>(d[FHistogram]), reinterpret_cast<int64_t*>(d[FPrefix]),
      reinterpret_cast<int*>(d[FCandidates]), reinterpret_cast<int*>(d[FLeasedSlots]), int(d[FCapacity])};
  }
};

// Match record_globalize_current's int32 slot ABI for either input dtype.
__device__ __forceinline__ int native_global_row(int row, int base, int peers, int slots) {
  if (row <= 0) return row;
  const int logical = row + base;
  return (logical % peers) * slots + logical / peers;
}

template <bool NativeRows>
__global__ void register_groups(const void* rows, int bytes, int count,
    const int64_t* frames, int base, bool prefer_gpu) {
  const Frame frame{frames + int64_t(blockIdx.y) * kFrameWords};
  const State state = frame.state();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    int64_t row = read_row(rows, i, bytes);
    if constexpr (NativeRows)
      row = native_global_row(int(row), base, int(frame.d[FPeers]), state.logical_slots);
    register_row(row, int(frame.d[FRank]), prefer_gpu, state, &frame.group());
  }
}

__device__ __forceinline__ bool has_cpu_rows(const int64_t* descriptor) {
  if (!descriptor) return false;
  const State s = Frame{descriptor}.state();
  return s.free_counts[1] != s.host_slots;
}

__device__ __forceinline__ void pin_gpu_or_cpu(
    int row, State s, Workspace w, const recent::ProducerContext& context) {
  if (row < 0) return;
  const int64_t epoch = *s.epoch;
  const int64_t location = s.directory[row];
  auto time = cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.access_time[row]);
  if (location >= 0) {
    assert(location < s.gpu_slots && s.gpu_tags[location] == row);
    // Imported readers may already have stamped E: structural ring membership
    // must be updated before the original timestamp dedup can return.
    recent::mark_slot(context, int(location), ring::Operation::Access, false);
    if (time.load(cuda::memory_order_relaxed) != epoch)
      time.store(epoch, cuda::memory_order_relaxed);
    return;
  }
  // Preserve native CPU promotion dedup and the complete-producer boundary.
  if (time.load(cuda::memory_order_relaxed) == epoch) return;
  if (time.exchange(epoch, cuda::memory_order_relaxed) == epoch) return;
  assert(location <= -2); // Decode misses refer to allocated logical rows.
  const int out = atomicAdd(w.counters + PromoteCount, 1);
  assert(out < w.capacity);
  w.promote_rows[out] = row;
}

// The original per-group transaction. Its State, Workspace and ring belong
// only to this shared-Index group; callers retire prior readers first.
__device__ __forceinline__ void begin_group(const Frame& f,
    const void* current, int bytes, int queries, recent::Storage<256>& storage) {
  assert(blockDim.x == 256);
    const State s = f.state();
    assert(s.epoch == f.group().epoch && *s.epoch < LLONG_MAX);
    recent::advance<256>(f.group(), *s.epoch + 1, storage);
    if (threadIdx.x == 0) ++*s.epoch;
    if (threadIdx.x < 8) f.work().counters[threadIdx.x] = 0;
    __syncthreads();
    // Canonical current time also suppresses a fresh CPU current promotion.
    // Registered GPU currents move structurally before the timestamp store.
    for (int q = threadIdx.x; q < queries; q += blockDim.x) {
      const int row = local_row(read_row(current, q, bytes), int(f.d[FRank]), s.logical_slots);
      if (row < 0) continue;
      const int64_t location = s.directory[row];
      assert(location != -1);
      if (location >= 0) {
        assert(location < s.gpu_slots && s.gpu_tags[location] == row);
        recent::mark_slot(f.group(), int(location), ring::Operation::Access, *s.epoch);
      }
      cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.access_time[row])
          .store(*s.epoch, cuda::memory_order_relaxed);
    }
    __syncthreads(); // retire each group's producers and shared storage
}

// Preserve the existing inline contract for other callers.
__device__ __forceinline__ void begin_groups(const int64_t* frames, int groups,
    const void* current, int bytes, int queries) {
  __shared__ recent::Storage<256> storage;
  for (int g = 0; g < groups; ++g) {
    const Frame f{frames + int64_t(g) * kFrameWords};
    begin_group(f, current, bytes, queries, storage);
  }
}

// Ordinary same-stream predecessor: original published begin has completed
// its one StepPublication retirement and written active before this launch.
// Each CTA uses only its existing group state/scratch and local shared storage.
__global__ void begin_groups_parallel(const int64_t* frames, int groups,
    const void* current, int bytes, int queries, const int* active) {
  __shared__ recent::Storage<256> storage;
  const int g = int(blockIdx.x);
  assert(g < groups);
  const int n = *active;
  assert(n >= 0 && n <= queries);
  const Frame f{frames + int64_t(g) * kFrameWords};
  begin_group(f, current, bytes, n, storage);
}

template <bool SetPendingReserve>
__device__ __forceinline__ void prepare_actual_misses_body(const int64_t* descriptor,
    const int64_t* step, const int* active,
    int queries, int tiles) {
  if (!descriptor) return;
  assert(queries > 0 && tiles > 0);
  const Frame f{descriptor};
  const auto s = f.state();
  const auto w = f.work();
  const int peer = blockIdx.x / (queries * tiles);
  const int query = (blockIdx.x / tiles) % queries;
  const int tile = blockIdx.x % tiles;
  assert(peer < f.d[FPeers]);
  if constexpr (SetPendingReserve)
    if (threadIdx.x == 0 && blockIdx.x == 0) w.counters[PendingReserve] = 1;
  if (query >= *active) return;
  const auto* ready_peers = reinterpret_cast<const uint64_t*>(f.d[FMissReadyPeers]);
  if (threadIdx.x == 0) {
    auto* ready = reinterpret_cast<int64_t*>(ready_peers[peer]) + f.d[FGroup];
    const uint64_t started = clock64();
    while (cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*ready)
        .load(cuda::memory_order_acquire) < *step) {
      assert(clock64() - started < 100000000000ULL);
      __nanosleep(64);
    }
  }
  __syncthreads();
  const auto* counts_peers = reinterpret_cast<const uint64_t*>(f.d[FMissCountsPeers]);
  const auto* rows_peers = reinterpret_cast<const uint64_t*>(f.d[FMissRowsPeers]);
  const auto* counts = reinterpret_cast<const int*>(counts_peers[peer])
      + f.d[FGroup] * f.d[FMaxQueries];
  const auto* rows = reinterpret_cast<const int*>(rows_peers[peer])
      + f.d[FGroup] * f.d[FMaxQueries] * f.d[FTopK];
  const int count = counts[query];
  assert(count >= 0 && count <= f.d[FTopK]);
  // Cache immutable producer geometry only for CTAs that own actual rows.
  if (int64_t(tile) * blockDim.x >= count) return;
  const auto context = recent::producer_context(f.group(), *s.epoch);
  for (int miss = tile * blockDim.x + threadIdx.x; miss < count; miss += tiles * blockDim.x) {
    const int row = local_row(rows[int64_t(query) * f.d[FTopK] + miss],
        int(f.d[FRank]), s.logical_slots);
    pin_gpu_or_cpu(row, s, w, context);
  }
}

__global__ void prepare_actual_misses(const int64_t* descriptor,
    const int64_t* step, const int* active, int queries, int tiles) {
  prepare_actual_misses_body<true>(descriptor, step, active, queries, tiles);
}

__device__ __forceinline__ bool arrive_actual_miss_cta(Workspace w, int ctas) {
  const int prior = cuda::atomic_ref<int, cuda::thread_scope_device>(w.counters[PendingReserve])
      .fetch_add(1, cuda::memory_order_acq_rel);
  assert(prior >= 0 && prior < ctas);
  return prior == ctas - 1;
}

// Low-query TP retains the complete parallel producer grid. The last CTA
// finishes its owner-local plan after every producer has retired, without a
// spinning grid barrier or another metadata kernel launch. begin_group zeroes
// this existing scratch word; metadata_finish_body resets it after the vote.
__device__ __forceinline__ void finish_actual_miss_producers(const int64_t* descriptor,
    const int64_t* step, const int* active, int queries, int tiles) {
  assert(descriptor && queries > 0 && queries < 8);
  __shared__ recent::Storage<256> storage;
  __shared__ bool last;
  prepare_actual_misses_body<false>(descriptor, step, active, queries, tiles);
  // Every lane publishes its own directory/ring/time writes before its CTA
  // votes. The acquire/release RMW chain publishes all producers to the last
  // CTA; its barrier makes those results visible to all finish lanes.
  __threadfence();
  __syncthreads();
  const auto w = Frame{descriptor}.work();
  if (threadIdx.x == 0) last = arrive_actual_miss_cta(w, int(gridDim.x));
  __syncthreads();
  if (last) metadata_finish_body(Frame{descriptor}.state(), w, storage);
}
// Internal TP scheduling choice, frozen before any child promotion can change
// CPU free counts. The epoch tag is private to this group and never inferred
// again after exchange/commit. A null flag retains synchronous preparation.
__device__ __forceinline__ bool defer_gpu_bookkeeping(const Frame& f) {
  const auto s = f.state();
  return f.d[FDeferredEpoch] && s.free_counts[1] == s.host_slots &&
      f.work().counters[NewCount] == 0;
}

__global__ void prepare_actual_misses_and_finish(const int64_t* descriptor,
    const int64_t* step, const int* active, int queries, int tiles) {
  const Frame f{descriptor};
  const bool deferred = defer_gpu_bookkeeping(f);
  if (threadIdx.x == 0 && blockIdx.x == 0 && f.d[FDeferredEpoch])
    *reinterpret_cast<int64_t*>(f.d[FDeferredEpoch]) = deferred ? *step : 0;
  if (!deferred) finish_actual_miss_producers(descriptor, step, active, queries, tiles);
}

// Enqueued after the anchor payload event on the existing prefetch stream.
// Only the early route tag authorizes this phase; a formerly cold owner whose
// last CPU row was promoted must not run its producers/reservation twice.
__global__ void finish_deferred_gpu_bookkeeping(const int64_t* descriptor,
    const int64_t* step, const int* active, int queries, int tiles) {
  const Frame f{descriptor};
  if (!f.d[FDeferredEpoch]) return;
  const int64_t route_step = *reinterpret_cast<const int64_t*>(f.d[FDeferredEpoch]);
  if (route_step <= 0 || route_step != *step) return;
  assert(defer_gpu_bookkeeping(f));
  assert(f.work().counters[PromoteCount] == 0 && f.work().counters[PlanCount] == 0);
  finish_actual_miss_producers(descriptor, step, active, queries, tiles);
}

} // namespace sglang::flowkv_host

namespace sglang {
template <bool NativeRows>
void flowkv_host_register_groups_impl(TensorView rows, TensorView frames, int64_t base, bool prefer_gpu) {
  TVM_FFI_ICHECK(rows.ndim() == 1 && (rows.dtype().bits == 32 || rows.dtype().bits == 64));
  TVM_FFI_ICHECK(frames.ndim() == 2 && frames.dtype().bits == 64 &&
                frames.shape()[1] == flowkv_host::kFrameWords);
  const auto stream = host::LaunchKernel::resolve_device(frames.device());
  const int count = int(rows.shape()[0]), groups = int(frames.shape()[0]);
  if (count && groups)
    flowkv_host::register_groups<NativeRows><<<dim3(std::min(128, (count + 255) / 256), groups), 256, 0, stream>>>(
        rows.data_ptr(), rows.dtype().bits / 8, count, static_cast<const int64_t*>(frames.data_ptr()), int(base), prefer_gpu);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_host_register_groups(TensorView rows, TensorView frames, bool prefer_gpu) {
  flowkv_host_register_groups_impl<false>(rows, frames, 0, prefer_gpu);
}

void flowkv_host_register_native_groups(TensorView rows, TensorView frames, int64_t base, bool prefer_gpu) {
  flowkv_host_register_groups_impl<true>(rows, frames, base, prefer_gpu);
}
} // namespace sglang

#ifdef FLOW_RECORD_PEERS
namespace sglang {
void flowkv_finish_deferred_gpu_bookkeeping(TensorView descriptor,
    TensorView step, TensorView active, int64_t queries) {
  TVM_FFI_ICHECK(descriptor.ndim() == 2 && descriptor.shape()[0] == 1 &&
      descriptor.shape()[1] == flowkv_host::kFrameWords && descriptor.dtype().bits == 64 &&
      step.numel() == 1 && step.dtype().bits == 64 && active.numel() == 1 &&
      active.dtype().bits == 32 && queries > 0 && queries < 8);
  const auto stream = host::LaunchKernel::resolve_device(descriptor.device());
  flowkv_host::finish_deferred_gpu_bookkeeping<<<FLOW_RECORD_PEERS * int(queries) * 4, 256, 0, stream>>>(
      static_cast<const int64_t*>(descriptor.data_ptr()),
      static_cast<const int64_t*>(step.data_ptr()), static_cast<const int*>(active.data_ptr()), int(queries), 4);
  host::RuntimeDeviceCheck(cudaGetLastError());
}
} // namespace sglang

#endif // FLOW_RECORD_PEERS
