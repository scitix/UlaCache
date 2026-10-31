#pragma once
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include "csrc/flashag/h200/ag_kernels_compact.cuh"

template<int World>
void compact_all_gather(tvm::ffi::TensorView input, tvm::ffi::TensorView output,
    tvm::ffi::TensorView input_packs, int64_t uc, int64_t mc, int64_t peer_table,
    int64_t rank, int64_t region, int64_t slot_stride, int64_t slice,
    int64_t blocks, int64_t threads, tvm::ffi::TensorView counters) {
  using namespace sglang::host;
  namespace ka = sglang::flash_ag_kernels_h200;
  namespace kc = ulacache_flashag::flash_ag_common_h200;
  RuntimeCheck(input.dtype().bits == 32 && output.dtype() == input.dtype(), "FP32 scores required");
  RuntimeCheck(output.numel() == input.numel() * World, "Declared upper bounds differ");
  RuntimeCheck(input_packs.numel() == 1 && input_packs.dtype().bits == 64, "Input pack count must be one int64");
  RuntimeCheck(blocks > 0 && threads > 0 && threads % 32 == 0 &&
      blocks * threads <= kc::kMaxSpinThreads, "Invalid co-resident launch domain");
  RuntimeCheck(counters.numel() >= blocks * threads / 32, "Counter buffer too small");
  RuntimeCheck(rank >= 0 && rank < World && uc && mc && peer_table, "Missing collective mapping");
  RuntimeCheck(uintptr_t(input.data_ptr()) % 16 == 0 && uintptr_t(output.data_ptr()) % 16 == 0,
      "Score buffers must be 16-byte aligned");
  LaunchKernel(dim3(blocks), dim3(threads), LaunchKernel::resolve_device(input.device()))
      .enable_pdl(true)(ka::k_ag_mcst<kc::PkF32, World, ka::kModeD31, 1,
          ka::kPollDense, false, false, false, true>,
          reinterpret_cast<const uint4*>(input.data_ptr()), reinterpret_cast<uint4*>(output.data_ptr()),
          reinterpret_cast<uint4*>(uc), reinterpret_cast<uint4*>(mc),
          reinterpret_cast<void* const*>(peer_table), int(rank), size_t(output.numel() / 4),
          reinterpret_cast<uint32_t*>(counters.data_ptr()), size_t(region), size_t(slot_stride),
          size_t(slice), static_cast<const int64_t*>(input_packs.data_ptr()), nullptr, size_t(0), int(0));
}


template<int World>
void direct_compact_all_gather(tvm::ffi::TensorView input, tvm::ffi::TensorView output,
    tvm::ffi::TensorView query_offsets, tvm::ffi::TensorView input_packs,
    int64_t uc, int64_t mc, int64_t peer_table,
    int64_t rank, int64_t region, int64_t slot_stride, int64_t slice,
    int64_t blocks, int64_t threads, tvm::ffi::TensorView counters) {
  using namespace sglang::host;
  namespace ka = sglang::flash_ag_kernels_h200;
  namespace kc = ulacache_flashag::flash_ag_common_h200;
  RuntimeCheck(input.ndim() == 2 && input.dtype().bits == 32 && output.dtype() == input.dtype(), "FP32 score matrix required");
  RuntimeCheck(input.stride(1) == 1 && input.stride(0) == input.size(1), "Contiguous local scores required");
  RuntimeCheck(output.numel() == input.numel() * World, "Declared upper bounds differ");
  RuntimeCheck(query_offsets.numel() == input.size(0) + 1 && query_offsets.dtype().bits == 32, "Invalid row prefix");
  RuntimeCheck(input_packs.numel() == 1 && input_packs.dtype().bits == 64, "Input pack count must be one int64");
  RuntimeCheck(blocks > 0 && threads > 0 && threads % 32 == 0 &&
      blocks * threads <= kc::kMaxSpinThreads, "Invalid co-resident launch domain");
  RuntimeCheck(counters.numel() >= blocks * threads / 32, "Counter buffer too small");
  RuntimeCheck(rank >= 0 && rank < World && uc && mc && peer_table, "Missing collective mapping");
  RuntimeCheck(uintptr_t(input.data_ptr()) % 16 == 0 && uintptr_t(output.data_ptr()) % 16 == 0,
      "Score buffers must be 16-byte aligned");
  LaunchKernel(dim3(blocks), dim3(threads), LaunchKernel::resolve_device(input.device()))
      .enable_pdl(true)(ka::k_ag_mcst<kc::PkF32, World, ka::kModeD31, 1,
          ka::kPollDense, false, false, false, true, true>,
          reinterpret_cast<const uint4*>(input.data_ptr()), reinterpret_cast<uint4*>(output.data_ptr()),
          reinterpret_cast<uint4*>(uc), reinterpret_cast<uint4*>(mc),
          reinterpret_cast<void* const*>(peer_table), int(rank), size_t(output.numel() / 4),
          reinterpret_cast<uint32_t*>(counters.data_ptr()), size_t(region), size_t(slot_stride),
          size_t(slice), static_cast<const int64_t*>(input_packs.data_ptr()),
          static_cast<const int32_t*>(query_offsets.data_ptr()), size_t(input.size(1)), int(input.size(0)));
}
