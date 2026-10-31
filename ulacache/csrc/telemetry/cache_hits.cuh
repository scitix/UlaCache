// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <cuda/atomic>

namespace udcache::telemetry {
using tvm::ffi::TensorView;

__global__ void cache_hit_samples(const void* indices, int index_bytes,
    int queries, int topk, int64_t query_stride, const int64_t* step,
    const int* active, const int* current_keys, int hash_size,
    const int* hints, const int* tags, const uint64_t* directories,
    const int* enabled, unsigned long long* counters, int capacity,
    int slots, int peers, int group, int rank) {
  if (!*enabled || *step % UDCACHE_HIT_STEP_STRIDE || blockIdx.x >= *active) return;
  const int query = blockIdx.x, lane = threadIdx.x;
  if (lane == 0) atomicAdd(counters + 8, 1ULL);
  for (int sample = lane; sample < UDCACHE_HIT_SAMPLES; sample += 32) {
    const int column = int((uint64_t(*step)*173 + uint64_t(query)*19
        + uint64_t(group)*7 + uint64_t(rank)*11 + uint64_t(sample)*67) % topk);
    const int64_t offset = int64_t(query)*query_stride + column;
    const int native = index_bytes == 4 ? static_cast<const int*>(indices)[offset]
        : int(static_cast<const int64_t*>(indices)[offset]);
    int category = 7;
    if (native > 0 && native / peers < slots) {
      const int owner = native % peers, local = native / peers;
      const int row = owner*slots + local;
      bool current = false;
      unsigned hash = (uint32_t(row)*2654435761u) >> (32-(__ffs(hash_size)-1));
      for (int probe = 0; probe < hash_size; ++probe) {
        const int key = current_keys[hash];
        if (key == row) { current = true; break; }
        if (key < 0) break;
        hash = (hash+1) & (hash_size-1);
      }
      if (current) category = 6;
      else {
        const int hint = hints[row];
        if (hint >= 0 && hint < capacity && tags[hint] == row) category = 1;
        else {
          const auto* directory = reinterpret_cast<const int64_t*>(directories[owner])
              + int64_t(group)*slots;
          const int64_t location = cuda::atomic_ref<int64_t, cuda::thread_scope_system>(const_cast<int64_t&>(directory[local]))
              .load(cuda::memory_order_acquire);
          category = location >= 0 ? 3 : (location <= -2 ? 4 : 5);
        }
      }
    }
    // All lanes iterate equally except the last partial sample warp. Mask its
    // inactive lanes explicitly so arbitrary positive sample counts are legal.
    const unsigned participating = __activemask();
    for (int counter = 0; counter < 8; ++counter) {
      const bool yes = counter == 0 ? category >= 1 && category <= 5
          : counter == 2 ? category >= 3 && category <= 5 : category == counter;
      const unsigned votes = __ballot_sync(participating, yes);
      if (lane == __ffs(participating)-1 && votes) atomicAdd(counters+counter, uint64_t(__popc(votes)));
    }
  }
}

void sample_cache_hits(TensorView indices, TensorView step, TensorView active,
    TensorView current_keys, TensorView hints, TensorView tags, int64_t directories,
    TensorView enabled, TensorView counters, int64_t capacity, int64_t slots,
    int64_t peers, int64_t group, int64_t rank) {
  TVM_FFI_ICHECK(indices.ndim() == 2 && indices.stride(1) == 1 &&
      (indices.dtype().bits == 32 || indices.dtype().bits == 64));
  TVM_FFI_ICHECK(counters.numel() == 9 && counters.dtype().bits == 64);
  TVM_FFI_ICHECK(step.numel() == 1 && step.dtype().bits == 64 && active.numel() == 1);
  TVM_FFI_ICHECK(enabled.numel() == 1 && enabled.dtype().bits == 32 && directories);
  TVM_FFI_ICHECK(capacity > 0 && slots > 0 && peers > 0 &&
      hints.numel() == peers*slots && tags.numel() == capacity);
  const int queries = int(indices.size(0));
  if (!queries) return;
  const auto stream = sglang::host::LaunchKernel::resolve_device(indices.device());
  cache_hit_samples<<<queries, 32, 0, stream>>>(indices.data_ptr(), indices.dtype().bits/8,
      queries, int(indices.size(1)), indices.stride(0), static_cast<const int64_t*>(step.data_ptr()),
      static_cast<const int*>(active.data_ptr()), static_cast<const int*>(current_keys.data_ptr()),
      int(current_keys.numel()), static_cast<const int*>(hints.data_ptr()),
      static_cast<const int*>(tags.data_ptr()), reinterpret_cast<const uint64_t*>(directories),
      static_cast<const int*>(enabled.data_ptr()), static_cast<unsigned long long*>(counters.data_ptr()),
      int(capacity), int(slots), int(peers), int(group), int(rank));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}
} // namespace udcache::telemetry
