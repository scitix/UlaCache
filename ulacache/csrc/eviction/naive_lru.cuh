// SPDX-License-Identifier: Apache-2.0
#pragma once
// Included inside sglang::flowkv_host after State, Workspace and graph helpers.
// Exact timestamp LRU: scan resident GPU slots, select the GPU smallest-k
// lexicographic (last-access epoch, physical slot), then reuse the native plan.
// Recovered selector provenance: frozen flowkv_host.cuh SHA256 6604a30ba4df09d77e2dcd09272f0ae475150f0000a227ef784b623de4e22cc4.
// Only the selector is reused; current allocator, leases, mapping, publication,
// record IO, and speculative native release remain the production path.

__device__ __forceinline__ int* naive_bins(Workspace w) {
  return reinterpret_cast<const ring::Group*>(w.histogram)->naive_histogram;
}

__device__ __forceinline__ bool naive_reserve_counts(State s, Workspace w) {
  const bool needed = reserve_counts(s, w);
  w.prefix[0] = 0;
  w.prefix[1] = 0;
  w.counters[Candidates] = 0;
  w.counters[PendingReserve] = 0;
  w.counters[LeaseCount] = 0;
  return needed;
}

__global__ void naive_reserve(State s, Workspace w, cudaGraphConditionalHandle evict) {
  if (threadIdx.x) return;
  const bool needed = naive_reserve_counts(s, w);
  if (evict) cudaGraphSetConditional(evict, needed);
}

// TP already launches one cold payload child only when PlanCount is nonzero.
// Keep its parent free of conditional graph nodes, and put selection inside
// that same child. Expiry scratch is dead until the next begin/release.
__global__ void naive_reserve_child(State s, Workspace w, int parallel) {
  if (threadIdx.x) return;
  naive_reserve_counts(s, w);
  w.leased[0] = parallel;
}

__global__ void naive_assign_child(State s, Workspace w) {
  if (!w.leased[0]) {
    if (blockIdx.x == 0) assign_plan_cta(s, w);
    return;
  }
  assert(w.counters[Candidates] == w.counters[VictimCount]);
  const int free_used = w.counters[FreeUsed];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < w.counters[PlanCount]; i += gridDim.x * blockDim.x) {
    const int gpu = i < free_used ? s.gpu_free[s.free_counts[0] + i]
                                : w.candidates[i - free_used];
    assign_plan_item(s, w, i, gpu);
  }
}

__global__ void naive_assign_cta(State s, Workspace w) {
  assign_plan_cta(s, w);
}

// Exact lexicographic selection of (last_access_epoch, physical_slot), using
// one pool scan per byte.  GPU-free allocation bypasses every scan.  Each
// histogram/select pair is a fixed launch, so the plan is graph-compatible.
__global__ void naive_histogram(State s, Workspace w, bool physical, int shift) {
  if (w.counters[VictimCount] == 0 || skip_epoch_byte(s.epoch, physical, shift)) return;
  __shared__ int bins[256];
  bins[threadIdx.x] = 0;
  __syncthreads();
  const uint64_t prefix = uint64_t(w.prefix[physical ? 1 : 0]);
  for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < s.gpu_slots; slot += gridDim.x * blockDim.x) {
    const int row = s.gpu_tags[slot];
    if (row < 0 || s.leases[slot] != 0) continue;
    const int64_t time = cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.access_time[row])
        .load(cuda::memory_order_relaxed);
    if (time >= *s.epoch) continue;
    const uint64_t epoch = uint64_t(time);
    if (physical && epoch != uint64_t(w.prefix[0])) continue;
    const uint64_t key = physical ? uint64_t(slot) : epoch;
    const bool prefix_match = !physical && shift == 56 ? true : (key >> (shift + 8)) == (prefix >> (shift + 8));
    if (prefix_match) atomicAdd(bins + ((key >> shift) & 255), 1);
  }
  __syncthreads();
  if (bins[threadIdx.x]) atomicAdd(naive_bins(w) + threadIdx.x, bins[threadIdx.x]);
}

__global__ void naive_select_prefix(Workspace w, bool physical, int shift, const int64_t* epoch) {
  if (w.counters[VictimCount] == 0 || skip_epoch_byte(epoch, physical, shift)) return;
  using Scan = cub::BlockScan<int64_t, kThreads>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ int64_t rank;
  if (threadIdx.x == 0) rank = w.prefix[2];
  const int64_t bin = naive_bins(w)[threadIdx.x];
  int64_t inclusive;
  // BlockScan's CTA synchronization publishes the immutable rank snapshot.
  Scan(scan).InclusiveSum(bin, inclusive);
  if (threadIdx.x == kThreads - 1) assert(rank > 0 && rank <= inclusive);
  const int64_t preceding = inclusive - bin;
  if (bin > 0 && preceding < rank && rank <= inclusive) {
    w.prefix[physical ? 1 : 0] |= int64_t(uint64_t(threadIdx.x) << shift);
    w.prefix[2] = rank - preceding;
  }
  __syncthreads();
  naive_bins(w)[threadIdx.x] = 0;
}

__global__ void naive_collect_victims(State s, Workspace w) {
  const int need = w.counters[VictimCount];
  if (need == 0) return;
  const int64_t threshold = w.prefix[0];
  const int tie_slot = int(w.prefix[1]);
  for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < s.gpu_slots; slot += gridDim.x * blockDim.x) {
    const int row = s.gpu_tags[slot];
    if (row < 0 || s.leases[slot] != 0) continue;
    const int64_t timestamp = cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.access_time[row])
        .load(cuda::memory_order_relaxed);
    if (timestamp >= *s.epoch) continue;
    if (timestamp < threshold || (timestamp == threshold && slot <= tie_slot)) {
      const int out = atomicAdd(w.counters + Candidates, 1);
      assert(out < need);
      w.candidates[out] = slot;
    }
  }
}


inline void naive_conditional_eviction(cudaStream_t stream, cudaGraphConditionalHandle evict, State s, Workspace w, int blocks) {
  const auto capture = capture_info(stream);
  const auto eviction = add_if(capture.graph, capture.dependencies, capture.count, evict);
  cudaGraphNode_t previous{};
  for (int shift = 56; shift >= 0; shift -= 8) {
    previous = add_kernel(eviction.body, previous, dim3(blocks), dim3(256), naive_histogram, s, w, false, shift);
    previous = add_kernel(eviction.body, previous, dim3(1), dim3(256), naive_select_prefix, w, false, shift, s.epoch);
  }
  for (int shift = first_physical_shift(s.gpu_slots); shift >= 0; shift -= 8) {
    previous = add_kernel(eviction.body, previous, dim3(blocks), dim3(256), naive_histogram, s, w, true, shift);
    previous = add_kernel(eviction.body, previous, dim3(1), dim3(256), naive_select_prefix, w, true, shift, s.epoch);
  }
  add_kernel(eviction.body, previous, dim3(blocks), dim3(256), naive_collect_victims, s, w);
  // Continue the caller's stream capture after the conditional plan.  The
  // graph owns both bodies and their handles; no CPU-side replay state exists.
  set_frontier(stream, eviction.node);
}

inline void naive_finish_select(cudaStream_t stream, State s, Workspace w) {
  const auto capture = capture_info(stream);
  cudaGraphConditionalHandle evict{};
  if (capture.graph)
    host::RuntimeDeviceCheck(cudaGraphConditionalHandleCreate(&evict, capture.graph, 0, cudaGraphCondAssignDefault));
  naive_reserve<<<1, 1, 0, stream>>>(s, w, evict);
  const int blocks = std::min(128, (s.gpu_slots + 255) / 256);
  if (capture.graph) {
    naive_conditional_eviction(stream, evict, s, w, blocks);
  } else {
    for (int shift = 56; shift >= 0; shift -= 8) {
      naive_histogram<<<blocks, 256, 0, stream>>>(s, w, false, shift);
      naive_select_prefix<<<1, 256, 0, stream>>>(w, false, shift, s.epoch);
    }
    for (int shift = first_physical_shift(s.gpu_slots); shift >= 0; shift -= 8) {
      naive_histogram<<<blocks, 256, 0, stream>>>(s, w, true, shift);
      naive_select_prefix<<<1, 256, 0, stream>>>(w, true, shift, s.epoch);
    }
    naive_collect_victims<<<blocks, 256, 0, stream>>>(s, w);
  }
  host::RuntimeDeviceCheck(cudaGetLastError());
}
