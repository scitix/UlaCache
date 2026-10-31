// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <climits>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>

namespace udcache::telemetry {
using tvm::ffi::TensorView;

// Passive counters are diagnostic work, excluded from native cache timings.
__global__ void sparse_replay_unique_hits(const int* selected, const int* current,
    const int* requests, const int64_t* directory, unsigned long long* keys,
    unsigned long long* counts, int queries, int width, int topk, int slots, int table_size) {
  const int64_t item = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (item >= int64_t(queries)*topk) return;
  const int query = int(item / topk), request = query / width;
  const int uid = selected[item];
  if (uid <= 0) return;
  if (uid >= slots || requests[request] < 0) {
    atomicAdd(counts + 4, 1ULL); return;
  }
  for (int lane = 0; lane < width; ++lane)
    if (current[request*width + lane] == uid) return;
  const uint64_t key = (uint64_t(uint32_t(requests[request]) + 1) << 32) | uint32_t(uid);
  uint64_t mixed = key;
  mixed ^= mixed >> 30; mixed *= 0xbf58476d1ce4e5b9ULL;
  mixed ^= mixed >> 27; mixed *= 0x94d049bb133111ebULL; mixed ^= mixed >> 31;
  unsigned at = unsigned(mixed) & unsigned(table_size - 1);
  for (int probe = 0; probe < table_size; ++probe) {
    const uint64_t previous = atomicCAS(keys + at, ULLONG_MAX, key);
    if (previous == key) return;
    if (previous == ULLONG_MAX) {
      const int64_t location = directory[uid];
      atomicAdd(counts, 1ULL);
      atomicAdd(counts + (location >= 0 ? 1 : location <= -2 ? 2 : 3), 1ULL);
      return;
    }
    at = (at + 1) & unsigned(table_size - 1);
  }
  atomicAdd(counts + 4, 1ULL);
}

inline void sparse_replay_count_hits(TensorView selected, TensorView current,
    TensorView requests, TensorView directory, TensorView keys, TensorView counts,
    int64_t width) {
  TVM_FFI_ICHECK(selected.ndim() == 2 && selected.dtype().bits == 32 &&
      selected.stride(1) == 1 && selected.stride(0) == selected.size(1));
  TVM_FFI_ICHECK(width > 0 && selected.size(0) % width == 0);
  TVM_FFI_ICHECK(current.numel() == selected.size(0) && current.dtype().bits == 32);
  TVM_FFI_ICHECK(requests.numel() * width == selected.size(0) && requests.dtype().bits == 32);
  TVM_FFI_ICHECK(directory.ndim() == 1 && directory.dtype().bits == 64);
  TVM_FFI_ICHECK(keys.dtype().bits == 64 && keys.numel() >= 2*selected.numel() &&
      (keys.numel() & (keys.numel() - 1)) == 0);
  TVM_FFI_ICHECK(counts.numel() == 8 && counts.dtype().bits == 64);
  if (!selected.numel()) return;
  const auto stream = sglang::host::LaunchKernel::resolve_device(selected.device());
  sparse_replay_unique_hits<<<(selected.numel() + 255)/256, 256, 0, stream>>>(
      static_cast<const int*>(selected.data_ptr()), static_cast<const int*>(current.data_ptr()),
      static_cast<const int*>(requests.data_ptr()), static_cast<const int64_t*>(directory.data_ptr()),
      static_cast<unsigned long long*>(keys.data_ptr()), static_cast<unsigned long long*>(counts.data_ptr()),
      int(selected.size(0)), int(width), int(selected.size(1)), int(directory.numel()), int(keys.numel()));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

__global__ void sparse_replay_plan_bytes(const int* plan, const int* counters,
    unsigned long long* counts, int capacity, int layers, int row_bytes, int bound) {
  const int count = counters[2]; // Original PlanCount ABI.
  if (count > bound) {
    if (!blockIdx.x && !threadIdx.x) atomicAdd(counts + 4, 1ULL);
    return;
  }
  const int item = int(blockIdx.x)*blockDim.x + threadIdx.x;
  if (item >= count) return;
  if (plan[2*capacity + item] >= 0) atomicAdd(counts + 5, uint64_t(layers)*row_bytes);
  if (plan[3*capacity + item] >= 0) atomicAdd(counts + 6, uint64_t(layers)*row_bytes);
}

inline void sparse_replay_count_plan(TensorView plan, TensorView counters,
    TensorView counts, int64_t layers, int64_t row_bytes, int64_t bound) {
  TVM_FFI_ICHECK(plan.ndim() == 2 && plan.size(0) == 5 && plan.dtype().bits == 32);
  TVM_FFI_ICHECK(counters.numel() == 8 && counters.dtype().bits == 32);
  TVM_FFI_ICHECK(counts.numel() == 8 && counts.dtype().bits == 64);
  TVM_FFI_ICHECK(layers > 0 && row_bytes > 0 && bound > 0 && bound <= plan.size(1));
  const auto stream = sglang::host::LaunchKernel::resolve_device(plan.device());
  sparse_replay_plan_bytes<<<(bound + 255)/256, 256, 0, stream>>>(
      static_cast<const int*>(plan.data_ptr()), static_cast<const int*>(counters.data_ptr()),
      static_cast<unsigned long long*>(counts.data_ptr()), int(plan.size(1)), int(layers), int(row_bytes), int(bound));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

__global__ void sparse_replay_current_bytes(const int* current, const int64_t* directory,
    unsigned long long* counts, int queries, int slots, int layers, int row_bytes) {
  const int query = int(blockIdx.x)*blockDim.x + threadIdx.x;
  if (query >= queries) return;
  const int uid = current[query];
  if (uid > 0 && uid < slots && directory[uid] <= -2)
    atomicAdd(counts + 7, uint64_t(layers)*row_bytes);
}

inline void sparse_replay_count_current(TensorView current, TensorView directory,
    TensorView counts, int64_t layers, int64_t row_bytes) {
  TVM_FFI_ICHECK(current.ndim() == 1 && current.dtype().bits == 32);
  TVM_FFI_ICHECK(directory.ndim() == 1 && directory.dtype().bits == 64);
  TVM_FFI_ICHECK(counts.numel() == 8 && counts.dtype().bits == 64);
  if (!current.numel()) return;
  const auto stream = sglang::host::LaunchKernel::resolve_device(current.device());
  sparse_replay_current_bytes<<<(current.numel() + 255)/256, 256, 0, stream>>>(
      static_cast<const int*>(current.data_ptr()), static_cast<const int64_t*>(directory.data_ptr()),
      static_cast<unsigned long long*>(counts.data_ptr()), int(current.numel()), int(directory.numel()), int(layers), int(row_bytes));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}
}  // namespace udcache::telemetry
