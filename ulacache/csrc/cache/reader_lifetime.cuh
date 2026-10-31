// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <cuda/atomic>
#include <tvm/ffi/container/tensor.h>

namespace ulacache {
namespace residency_publication {

enum Phase { SelectionsReady, ResidencyReady, ReadersRetired, PhaseCount };

__device__ void wait_for_phase(const uint64_t* peers, int64_t owner_count,
    int64_t domain, Phase phase, int64_t epoch) {
  for (int64_t owner = threadIdx.x; owner < owner_count; owner += blockDim.x) {
    auto* flag = reinterpret_cast<int64_t*>(peers[owner]) + domain * PhaseCount + phase;
    while (cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*flag)
               .load(cuda::memory_order_acquire) < epoch)
      __nanosleep(64);
  }
  __syncthreads();
}

template <Phase PublishedPhase>
__global__ void publish_kernel(int64_t* epochs, int64_t* flags, const uint64_t* peers,
    int64_t owner_count, int64_t domain) {
  int64_t epoch = epochs[domain];
  if constexpr (PublishedPhase == SelectionsReady) {
    wait_for_phase(peers, owner_count, domain, ReadersRetired, epoch);
    ++epoch;
    if (threadIdx.x == 0) epochs[domain] = epoch;
  }
  // Ordinary stream/Graph task completion orders all preceding native work.
  // System release/acquire follows the existing TP step-publication protocol.
  if (threadIdx.x == 0)
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(flags[domain * PhaseCount + PublishedPhase])
        .store(epoch, cuda::memory_order_release);
  __syncthreads();
  if constexpr (PublishedPhase != ReadersRetired)
    wait_for_phase(peers, owner_count, domain, PublishedPhase, epoch);
}

__global__ void wait_readers_kernel(const int64_t* epochs, const uint64_t* peers,
    int64_t owner_count, int64_t first_domain, int64_t domain_count) {
  for (int64_t domain = first_domain; domain < first_domain + domain_count; ++domain)
    wait_for_phase(peers, owner_count, domain, ReadersRetired, epochs[domain]);
}

__global__ void finish_all_readers_kernel(const int64_t* epochs, int64_t* flags,
    int64_t domain_count) {
  // Completion of preceding stream/Graph tasks orders every local reader.
  // Each owner publishes its own epochs; this operation does not wait for peers.
  for (int64_t domain = threadIdx.x; domain < domain_count; domain += blockDim.x)
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(
        flags[domain * PhaseCount + ReadersRetired])
        .store(epochs[domain], cuda::memory_order_release);
}

inline void check_domain(tvm::ffi::TensorView epochs, int64_t domain) {
  TVM_FFI_ICHECK(epochs.ndim() == 1 && epochs.dtype().bits == 64 && epochs.stride(0) == 1);
  TVM_FFI_ICHECK(domain >= 0 && domain < epochs.numel());
}

template <Phase PublishedPhase>
void publish(tvm::ffi::TensorView epochs, tvm::ffi::TensorView flags,
    tvm::ffi::TensorView peers, int64_t domain) {
  check_domain(epochs, domain);
  TVM_FFI_ICHECK(flags.ndim() == 2 && flags.size(0) == epochs.numel() &&
                flags.size(1) == PhaseCount && flags.dtype().bits == 64 &&
                flags.stride(0) == PhaseCount && flags.stride(1) == 1);
  if constexpr (PublishedPhase != ReadersRetired)
    TVM_FFI_ICHECK(peers.ndim() == 1 && peers.dtype().bits == 64 && peers.stride(0) == 1);
  const auto stream = sglang::host::LaunchKernel::resolve_device(epochs.device());
  publish_kernel<PublishedPhase><<<1, 32, 0, stream>>>(static_cast<int64_t*>(epochs.data_ptr()),
      static_cast<int64_t*>(flags.data_ptr()),
      PublishedPhase == ReadersRetired ? nullptr : static_cast<const uint64_t*>(peers.data_ptr()),
      PublishedPhase == ReadersRetired ? 0 : peers.numel(), domain);
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

}  // namespace residency_publication

void residency_publication_begin_selection(tvm::ffi::TensorView epochs, tvm::ffi::TensorView flags,
    tvm::ffi::TensorView peers, int64_t domain) {
  residency_publication::publish<residency_publication::SelectionsReady>(epochs, flags, peers, domain);
}

void residency_publication_publish_residency(tvm::ffi::TensorView epochs, tvm::ffi::TensorView flags,
    tvm::ffi::TensorView peers, int64_t domain) {
  residency_publication::publish<residency_publication::ResidencyReady>(epochs, flags, peers, domain);
}

void residency_publication_finish_readers(tvm::ffi::TensorView epochs, tvm::ffi::TensorView flags,
    int64_t domain) {
  residency_publication::publish<residency_publication::ReadersRetired>(epochs, flags, flags, domain);
}

void residency_publication_finish_all_readers(tvm::ffi::TensorView epochs,
    tvm::ffi::TensorView flags) {
  residency_publication::check_domain(epochs, 0);
  TVM_FFI_ICHECK(flags.ndim() == 2 && flags.size(0) == epochs.numel() &&
      flags.size(1) == residency_publication::PhaseCount && flags.dtype().bits == 64 &&
      flags.stride(0) == residency_publication::PhaseCount && flags.stride(1) == 1);
  const auto stream = sglang::host::LaunchKernel::resolve_device(epochs.device());
  residency_publication::finish_all_readers_kernel<<<1, 32, 0, stream>>>(
      static_cast<const int64_t*>(epochs.data_ptr()), static_cast<int64_t*>(flags.data_ptr()),
      epochs.numel());
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

void residency_publication_wait_for_readers(tvm::ffi::TensorView epochs, tvm::ffi::TensorView peers,
    int64_t domain) {
  residency_publication::check_domain(epochs, domain);
  TVM_FFI_ICHECK(peers.ndim() == 1 && peers.dtype().bits == 64 && peers.stride(0) == 1);
  const auto stream = sglang::host::LaunchKernel::resolve_device(epochs.device());
  residency_publication::wait_readers_kernel<<<1, 32, 0, stream>>>(
      static_cast<const int64_t*>(epochs.data_ptr()), static_cast<const uint64_t*>(peers.data_ptr()),
      peers.numel(), domain, 1);
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

void residency_publication_wait_for_all_readers(tvm::ffi::TensorView epochs, tvm::ffi::TensorView peers) {
  TVM_FFI_ICHECK(epochs.ndim() == 1 && epochs.dtype().bits == 64 && epochs.stride(0) == 1);
  TVM_FFI_ICHECK(peers.ndim() == 1 && peers.dtype().bits == 64 && peers.stride(0) == 1);
  const auto stream = sglang::host::LaunchKernel::resolve_device(epochs.device());
  residency_publication::wait_readers_kernel<<<1, 32, 0, stream>>>(
      static_cast<const int64_t*>(epochs.data_ptr()), static_cast<const uint64_t*>(peers.data_ptr()),
      peers.numel(), 0, epochs.numel());
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

// Reuse the existing native TP startup check before any captured publication.
void residency_publication_validate_peer(tvm::ffi::TensorView local, int64_t pointer) {
  sglang::host::LaunchKernel::resolve_device(local.device());
  TVM_FFI_ICHECK(pointer != 0 && (uint64_t(pointer) & 7) == 0);
  cudaPointerAttributes attributes{};
  sglang::host::RuntimeDeviceCheck(cudaPointerGetAttributes(&attributes,
      reinterpret_cast<const void*>(uintptr_t(pointer))));
  TVM_FFI_ICHECK(attributes.type == cudaMemoryTypeDevice && attributes.device >= 0);
  int device = -1, supported = 1;
  sglang::host::RuntimeDeviceCheck(cudaGetDevice(&device));
  if (device != attributes.device)
    sglang::host::RuntimeDeviceCheck(cudaDeviceGetP2PAttribute(&supported,
        cudaDevP2PAttrNativeAtomicSupported, device, attributes.device));
  TVM_FFI_ICHECK(supported == 1) << "Residency peers require system-scope GPU atomics";
}

}  // namespace ulacache
