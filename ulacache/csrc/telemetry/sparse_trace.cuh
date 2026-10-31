// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <climits>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>

namespace udcache::telemetry {
using tvm::ffi::TensorView;

__device__ inline int64_t trace_read(const void* data, int bytes, int64_t offset) {
  return bytes == 4 ? static_cast<const int*>(data)[offset]
                    : static_cast<const int64_t*>(data)[offset];
}

__global__ void sparse_trace_kernel(const void* ids, int id_bytes,
    int source_rows, int64_t source_stride, const void* current, int current_bytes,
    int current_rows, const int* active, const int64_t* epoch, int64_t* control,
    int* output, int* output_current, int* seen, int64_t* epochs,
    int group, int total_frames, int chunks, int groups, int rows, int topk) {
  if (!control[0]) return;
  const int64_t frame = control[1];
  if (frame < 0 || frame >= total_frames) return;
  const int valid = int(control[2]);
  if (valid < 0 || valid > *active || valid > rows || valid > source_rows || valid > current_rows) {
    if (!blockIdx.x && !threadIdx.x) atomicOr(reinterpret_cast<unsigned long long*>(control + 3), 1ULL);
    return;
  }
  const int chunk = int(frame % chunks);
  const int64_t base = (int64_t(chunk)*groups + group)*rows;
  const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i < int64_t(rows)*topk) {
    const int row = int(i / topk), column = int(i % topk);
    const int64_t uid = row < valid ? trace_read(ids, id_bytes, int64_t(row)*source_stride + column) : -1;
    if (uid < -1 || uid >= INT_MAX) atomicOr(reinterpret_cast<unsigned long long*>(control + 3), 2ULL);
    output[base*topk + i] = int(uid);
    if (!column) {
      const int64_t value = row < valid ? trace_read(current, current_bytes, row) : -1;
      if (value < -1 || value >= INT_MAX) atomicOr(reinterpret_cast<unsigned long long*>(control + 3), 2ULL);
      output_current[base + row] = int(value);
    }
  }
  if (!blockIdx.x && !threadIdx.x) {
    seen[int64_t(chunk)*groups + group] = valid;
    epochs[int64_t(chunk)*groups + group] = *epoch;
  }
}

inline void record_sparse_trace(TensorView ids, TensorView current,
    TensorView active, TensorView epoch, TensorView control,
    TensorView output, TensorView output_current, TensorView seen, TensorView epochs,
    int64_t group, int64_t frames, int64_t chunks, int64_t groups,
    int64_t rows, int64_t topk) {
  TVM_FFI_ICHECK(ids.ndim() == 2 && ids.stride(1) == 1 && ids.size(1) == topk);
  TVM_FFI_ICHECK(ids.dtype().bits == 32 || ids.dtype().bits == 64);
  TVM_FFI_ICHECK(current.ndim() == 1 && current.stride(0) == 1);
  TVM_FFI_ICHECK(current.dtype().bits == 32 || current.dtype().bits == 64);
  TVM_FFI_ICHECK(active.numel() == 1 && active.dtype().bits == 32);
  TVM_FFI_ICHECK(epoch.numel() == 1 && epoch.dtype().bits == 64);
  TVM_FFI_ICHECK(control.numel() == 5 && control.dtype().bits == 64);
  TVM_FFI_ICHECK(group >= 0 && group < groups && chunks > 0 && rows > 0 && topk > 0);
  TVM_FFI_ICHECK(output.numel() == chunks*groups*rows*topk && output.dtype().bits == 32);
  TVM_FFI_ICHECK(output_current.numel() == chunks*groups*rows && output_current.dtype().bits == 32);
  TVM_FFI_ICHECK(seen.numel() == chunks*groups && seen.dtype().bits == 32);
  TVM_FFI_ICHECK(epochs.numel() == chunks*groups && epochs.dtype().bits == 64);
  auto stream = sglang::host::LaunchKernel::resolve_device(ids.device());
  sparse_trace_kernel<<<(rows*topk + 255)/256, 256, 0, stream>>>(
      ids.data_ptr(), ids.dtype().bits/8, int(ids.size(0)), ids.stride(0),
      current.data_ptr(), current.dtype().bits/8, int(current.numel()),
      static_cast<const int*>(active.data_ptr()), static_cast<const int64_t*>(epoch.data_ptr()),
      static_cast<int64_t*>(control.data_ptr()), static_cast<int*>(output.data_ptr()),
      static_cast<int*>(output_current.data_ptr()), static_cast<int*>(seen.data_ptr()),
      static_cast<int64_t*>(epochs.data_ptr()), int(group), int(frames), int(chunks),
      int(groups), int(rows), int(topk));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

__global__ void sparse_trace_events_kernel(const void* rows, int bytes,
    int count, int64_t* control, int64_t* events, int capacity, int kind, int frames) {
  if (!control[0] || control[1] < 0 || control[1] >= frames) return;
  __shared__ unsigned long long base;
  if (!threadIdx.x) base = atomicAdd(reinterpret_cast<unsigned long long*>(control + 4), static_cast<unsigned long long>(count));
  __syncthreads();
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const int64_t at = int64_t(base) + i;
    const int64_t uid = trace_read(rows, bytes, i);
    if (at >= capacity || uid < 0 || uid >= INT_MAX) {
      atomicOr(reinterpret_cast<unsigned long long*>(control + 3), 4ULL);
    } else {
      events[at*3] = control[1];
      events[at*3 + 1] = kind;
      events[at*3 + 2] = uid;
    }
  }
}

inline void record_sparse_trace_event(TensorView rows, TensorView control,
    TensorView events, int64_t kind, int64_t frames) {
  TVM_FFI_ICHECK(rows.ndim() == 1 && rows.stride(0) == 1);
  TVM_FFI_ICHECK(rows.dtype().bits == 32 || rows.dtype().bits == 64);
  TVM_FFI_ICHECK(control.numel() == 5 && control.dtype().bits == 64);
  TVM_FFI_ICHECK(events.ndim() == 2 && events.size(1) == 3 && events.dtype().bits == 64);
  TVM_FFI_ICHECK(kind == 1 || kind == 2);
  if (!rows.numel()) return;
  auto stream = sglang::host::LaunchKernel::resolve_device(rows.device());
  sparse_trace_events_kernel<<<1, 256, 0, stream>>>(rows.data_ptr(), rows.dtype().bits/8,
      int(rows.numel()), static_cast<int64_t*>(control.data_ptr()),
      static_cast<int64_t*>(events.data_ptr()), int(events.size(0)), int(kind), int(frames));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

__global__ void sparse_trace_expiry_kernel(const int64_t* descriptors,
    const int64_t* control, int64_t* output, int groups, int after, int frames, int chunks) {
  if (!control[0] || control[1] < 0 || control[1] >= frames) return;
  const int group = int(blockIdx.x)*blockDim.x + threadIdx.x;
  if (group >= groups) return;
  const auto* d = descriptors + int64_t(group)*5;
  const auto* epoch = reinterpret_cast<const int64_t*>(d[0]);
  const auto* ages = reinterpret_cast<const int64_t*>(d[1]);
  const auto* summary = reinterpret_cast<const uint32_t*>(d[2]);
  const int64_t stride = d[3], recent = d[4];
  auto* row = output + (int64_t(control[1] % chunks)*groups + group)*8;
  const int64_t next_epoch = *epoch + 1;
  const int64_t bid = after ? row[1] : next_epoch % recent;
  if (!after) {
    row[0] = next_epoch;
    row[1] = bid;
    row[2] = ages[bid];
    row[3] = summary[bid*stride + stride - 1];
    row[6] = summary[recent*stride + stride - 1];
  } else {
    row[4] = ages[bid];
    row[5] = summary[bid*stride + stride - 1];
    row[7] = summary[recent*stride + stride - 1];
  }
}

inline void record_sparse_trace_expiry(TensorView descriptors, TensorView control,
    TensorView output, int64_t after, int64_t frames, int64_t chunks) {
  TVM_FFI_ICHECK(descriptors.ndim() == 2 && descriptors.size(1) == 5 &&
      descriptors.stride(1) == 1 && descriptors.stride(0) == 5 && descriptors.dtype().bits == 64);
  TVM_FFI_ICHECK(control.numel() == 5 && control.dtype().bits == 64);
  TVM_FFI_ICHECK(output.numel() == chunks*descriptors.size(0)*8 && output.dtype().bits == 64);
  TVM_FFI_ICHECK((after == 0 || after == 1) && chunks > 0);
  const int groups = int(descriptors.size(0));
  const auto stream = sglang::host::LaunchKernel::resolve_device(descriptors.device());
  sparse_trace_expiry_kernel<<<(groups + 127)/128, 128, 0, stream>>>(
      static_cast<const int64_t*>(descriptors.data_ptr()), static_cast<const int64_t*>(control.data_ptr()),
      static_cast<int64_t*>(output.data_ptr()), groups, int(after), int(frames), int(chunks));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}
}  // namespace udcache::telemetry
