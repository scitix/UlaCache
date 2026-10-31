// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <climits>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>

namespace udcache::telemetry {
using tvm::ffi::TensorView;

__global__ void tp_miss_trace_kernel(const int* rows, int64_t stride,
    const int* counts, int source_rows, const int* active, int64_t* control,
    int* output, int* output_counts, int* bucket_rows,
    int group, int frames, int chunks, int groups, int bound, int topk) {
  if (!control[0] || control[1] < 0 || control[1] >= frames) return;
  const int valid = int(control[2]);
  if (valid < 0 || valid > *active || valid > source_rows || valid > bound) {
    if (!blockIdx.x && !threadIdx.x)
      atomicOr(reinterpret_cast<unsigned long long*>(control + 3), 8ULL);
    return;
  }
  const int64_t frame_group = (control[1] % chunks) * groups + group;
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < int64_t(bound) * topk) {
    const int query = int(i / topk), column = int(i % topk);
    const int count = query < valid ? counts[query] : -1;
    if (query < valid && (count < 0 || count > topk))
      atomicOr(reinterpret_cast<unsigned long long*>(control + 3), 16ULL);
    const int uid = query < valid && column < count
        ? rows[int64_t(query) * stride + column] : -1;
    if (uid < -1 || uid >= INT_MAX)
      atomicOr(reinterpret_cast<unsigned long long*>(control + 3), 32ULL);
    output[frame_group * bound * topk + i] = uid;
    if (!column) output_counts[frame_group * bound + query] = count;
  }
  if (!blockIdx.x && !threadIdx.x) bucket_rows[frame_group] = source_rows;
}

inline void record_tp_miss_trace(TensorView rows, TensorView counts,
    TensorView active, TensorView control, TensorView output,
    TensorView output_counts, TensorView bucket_rows,
    int64_t group, int64_t frames, int64_t chunks, int64_t groups,
    int64_t bound, int64_t topk) {
  TVM_FFI_ICHECK(rows.ndim() == 2 && rows.dtype().bits == 32 &&
      rows.stride(1) == 1 && rows.size(1) == topk);
  TVM_FFI_ICHECK(counts.ndim() == 1 && counts.stride(0) == 1 &&
      counts.dtype().bits == 32 && counts.numel() >= rows.size(0));
  TVM_FFI_ICHECK(active.numel() == 1 && active.dtype().bits == 32);
  TVM_FFI_ICHECK(control.numel() == 5 && control.dtype().bits == 64);
  TVM_FFI_ICHECK(group >= 0 && group < groups && frames > 0 && chunks > 0 &&
      bound > 0 && topk > 0);
  TVM_FFI_ICHECK(output.numel() == chunks * groups * bound * topk && output.dtype().bits == 32);
  TVM_FFI_ICHECK(output_counts.numel() == chunks * groups * bound && output_counts.dtype().bits == 32);
  TVM_FFI_ICHECK(bucket_rows.numel() == chunks * groups && bucket_rows.dtype().bits == 32);
  auto stream = sglang::host::LaunchKernel::resolve_device(rows.device());
  tp_miss_trace_kernel<<<(bound * topk + 255) / 256, 256, 0, stream>>>(
      static_cast<const int*>(rows.data_ptr()), rows.stride(0),
      static_cast<const int*>(counts.data_ptr()), int(rows.size(0)),
      static_cast<const int*>(active.data_ptr()), static_cast<int64_t*>(control.data_ptr()),
      static_cast<int*>(output.data_ptr()), static_cast<int*>(output_counts.data_ptr()),
      static_cast<int*>(bucket_rows.data_ptr()), int(group), int(frames), int(chunks),
      int(groups), int(bound), int(topk));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}
}  // namespace udcache::telemetry
