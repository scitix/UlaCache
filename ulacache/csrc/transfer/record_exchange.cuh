// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "csrc/cache/directory.cuh"
#include "csrc/cache/hisparse.cuh"

namespace sglang::flowkv_host {
__device__ __forceinline__ void swap_record(int lane, char* host, char* gpu, int64_t bytes) {
  // HiSparse's transfer_item_warp load/store shape, with both originals loaded
  // before either destination is overwritten.  Only two uint64 registers per
  // endpoint are needed regardless of record width.
  for (int64_t pair = lane; pair < bytes / 16; pair += 32) {
    auto* h = reinterpret_cast<uint64_t*>(host) + 2 * pair;
    auto* g = reinterpret_cast<uint64_t*>(gpu) + 2 * pair;
    uint64_t h0, h1, g0, g1;
    if (((reinterpret_cast<uintptr_t>(h) | reinterpret_cast<uintptr_t>(g)) & 15) == 0) {
      asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(h0), "=l"(h1) : "l"(h) : "memory");
      asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(g0), "=l"(g1) : "l"(g) : "memory");
      asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" :: "l"(h), "l"(g0), "l"(g1) : "memory");
      asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" :: "l"(g), "l"(h0), "l"(h1) : "memory");
    } else {
      // A legal eight-byte record stride can alternate between 0/8 modulo16.
      // PTX vector loads require sixteen-byte alignment; scalar words do not.
      asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(h0) : "l"(h) : "memory");
      asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(h1) : "l"(h + 1) : "memory");
      asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(g0) : "l"(g) : "memory");
      asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(g1) : "l"(g + 1) : "memory");
      asm volatile("st.global.cg.b64 [%0],%1;" :: "l"(h), "l"(g0) : "memory");
      asm volatile("st.global.cg.b64 [%0],%1;" :: "l"(h + 1), "l"(g1) : "memory");
      asm volatile("st.global.cg.b64 [%0],%1;" :: "l"(g), "l"(h0) : "memory");
      asm volatile("st.global.cg.b64 [%0],%1;" :: "l"(g + 1), "l"(h1) : "memory");
    }
  }
  if (bytes % 16 && lane == 0) {
    auto* h = reinterpret_cast<uint64_t*>(host + bytes - 8);
    auto* g = reinterpret_cast<uint64_t*>(gpu + bytes - 8);
    const uint64_t h0 = *h, g0 = *g;
    *h = g0;
    *g = h0;
  }
}

__device__ __forceinline__ void copy_record(int lane, const char* src, char* dst, int64_t bytes) {
  if (((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0) {
    transfer_item_warp(lane, src, dst, bytes);
  } else {
    const auto* source = reinterpret_cast<const uint64_t*>(src);
    auto* target = reinterpret_cast<uint64_t*>(dst);
    for (int64_t word = lane; word < bytes / 8; word += 32) {
      uint64_t value;
      asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(value) : "l"(source + word) : "memory");
      asm volatile("st.global.cg.b64 [%0],%1;" :: "l"(target + word), "l"(value) : "memory");
    }
  }
}

__global__ void exchange(char* host, char* gpu, const int* plan, const int* counters, int capacity,
                         int64_t bytes, int64_t host_stride, int64_t gpu_stride,
                         int layers, int64_t host_layer_stride, int64_t gpu_layer_stride) {
  const int count = counters[PlanCount];
  if (count == 0) return;
  const int lane = threadIdx.x & 31;
  for (int64_t operation = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
       operation < int64_t(count) * layers; operation += gridDim.x * (blockDim.x / 32)) {
    const int i = operation % count, layer = operation / count;
    const int g = plan[capacity + i], src = plan[2 * capacity + i], dst = plan[3 * capacity + i];
    auto* gpu_record = gpu + int64_t(layer) * gpu_layer_stride + int64_t(g) * gpu_stride;
    auto* host_layer = host + int64_t(layer) * host_layer_stride;
    if (src >= 0 && dst >= 0) {
      assert(src == dst);
      swap_record(lane, host_layer + int64_t(src) * host_stride, gpu_record, bytes);
    } else if (src >= 0) {
      copy_record(lane, host_layer + int64_t(src) * host_stride, gpu_record, bytes);
    } else if (dst >= 0) {
      copy_record(lane, gpu_record, host_layer + int64_t(dst) * host_stride, bytes);
    }
  }
  __threadfence_system();
}


}  // namespace sglang::flowkv_host
