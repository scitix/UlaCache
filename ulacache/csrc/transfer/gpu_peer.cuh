// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "csrc/transfer/cpu_gpu.cuh"

namespace ulacache::paged_records {

template <typename Page, size_t Segment, int VectorBytes>
__device__ void copy_published_peer_segment(int lane, const char* source, char* destination,
    int64_t source_record, int64_t destination_record) {
  constexpr int width = Page::template segment_width<Segment>();
  source += Page::template offset<Segment>(source_record);
  destination += Page::template offset<Segment>(destination_record);
  if constexpr (VectorBytes == 16) {
    if (((reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination)) & 15) == 0) {
      for (int word = lane; word < width / 16; word += 32) {
        uint64_t low, high;
        asm volatile("ld.global.cg.v2.u64 {%0, %1}, [%2];"
            : "=l"(low), "=l"(high) : "l"(source + word * 16) : "memory");
        *reinterpret_cast<ulonglong2*>(destination + word * 16) = make_ulonglong2(low, high);
      }
      source += width / 16 * 16;
      destination += width / 16 * 16;
      if constexpr (width % 16 >= 8) {
        if (lane == 0) {
          uint64_t value;
          asm volatile("ld.global.cg.b64 %0,[%1];" : "=l"(value) : "l"(source) : "memory");
          *reinterpret_cast<uint64_t*>(destination) = value;
        }
        source += 8;
        destination += 8;
      }
      for (int byte = lane; byte < width % 8; byte += 32) {
        uint32_t value;
        asm volatile("ld.global.cg.u8 %0,[%1];" : "=r"(value) : "l"(source + byte) : "memory");
        destination[byte] = static_cast<char>(value);
      }
      return;
    }
  }
  if (((reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination)) & 7) == 0) {
    for (int word = lane; word < width / 8; word += 32) {
      uint64_t value;
      // Owner publications order the writes. Bypass the read-only cache for
      // records whose physical pages can change between residency selections.
      asm volatile("ld.global.cg.b64 %0,[%1];" : "=l"(value) : "l"(source + word * 8) : "memory");
      *reinterpret_cast<uint64_t*>(destination + word * 8) = value;
    }
    source += width / 8 * 8;
    destination += width / 8 * 8;
    for (int byte = lane; byte < width % 8; byte += 32) {
      uint32_t value;
      asm volatile("ld.global.cg.u8 %0,[%1];" : "=r"(value) : "l"(source + byte) : "memory");
      destination[byte] = static_cast<char>(value);
    }
  } else {
    for (int byte = lane; byte < width; byte += 32) {
      uint32_t value;
      asm volatile("ld.global.cg.u8 %0,[%1];" : "=r"(value) : "l"(source + byte) : "memory");
      destination[byte] = static_cast<char>(value);
    }
  }
}

template <typename Page, int VectorBytes, size_t... Segments>
__device__ void copy_published_peer_record(int lane, const char* source, char* destination,
    int64_t source_record, int64_t destination_record, std::index_sequence<Segments...>) {
  (copy_published_peer_segment<Page, Segments, VectorBytes>(lane, source, destination,
      source_record, destination_record), ...);
}

template <typename Page, size_t Segments, int VectorBytes>
__global__ void gather_peer_kernel(const uint64_t* peers, const void* source_owners,
    const void* source_records, char* destination, const void* destination_records,
    int owner_index_bytes, int source_index_bytes, int destination_index_bytes,
    int64_t owner_count, int64_t record_count, int64_t source_bank_byte_offset) {
  const int lane = threadIdx.x & 31;
  for (int64_t index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
       index < record_count; index += gridDim.x * (blockDim.x / 32)) {
    const int64_t owner = sglang::flowkv_host::read_row(source_owners, index, owner_index_bytes);
    const int64_t source_record = sglang::flowkv_host::read_row(source_records, index, source_index_bytes);
    const int64_t destination_record = sglang::flowkv_host::read_row(
        destination_records, index, destination_index_bytes);
    if (owner < 0 || source_record < 0 || destination_record < 0) continue;
    assert(owner < owner_count);
    const auto* source = reinterpret_cast<const char*>(peers[owner]) + source_bank_byte_offset;
    copy_published_peer_record<Page, VectorBytes>(lane, source, destination, source_record, destination_record,
        std::make_index_sequence<Segments>{});
  }
}

}  // namespace ulacache::paged_records

namespace ulacache {

template <int RecordsPerPage, int PageBytes, bool SegmentMajor, int VectorBytes, int... SegmentBytes>
void gather_peer_paged_records(tvm::ffi::TensorView peers, tvm::ffi::TensorView source_owners,
    tvm::ffi::TensorView source_records, tvm::ffi::TensorView destination,
    tvm::ffi::TensorView destination_records, int64_t source_bank_byte_offset,
    int64_t blocks, int64_t threads) {
  TVM_FFI_ICHECK(peers.ndim() == 1 && peers.dtype().bits == 64 && peers.stride(0) == 1);
  TVM_FFI_ICHECK(destination.ndim() == 2 && destination.dtype().bits == 8 &&
                destination.size(1) == PageBytes && destination.stride(0) == PageBytes &&
                destination.stride(1) == 1);
  for (const auto& indices : {source_owners, source_records, destination_records}) {
    TVM_FFI_ICHECK(indices.ndim() == 1 && indices.numel() == source_records.numel() &&
                  indices.stride(0) == 1 &&
                  (indices.dtype().bits == 32 || indices.dtype().bits == 64));
  }
  TVM_FFI_ICHECK(source_bank_byte_offset >= 0);
  static_assert(VectorBytes == 8 || VectorBytes == 16);
  using Page = paged_records::NativePage<RecordsPerPage, PageBytes, SegmentMajor, SegmentBytes...>;
  const auto stream = sglang::host::LaunchKernel::resolve_device(destination.device());
  if (source_records.numel())
    paged_records::gather_peer_kernel<Page, sizeof...(SegmentBytes), VectorBytes><<<blocks, threads, 0, stream>>>(
        static_cast<const uint64_t*>(peers.data_ptr()), source_owners.data_ptr(), source_records.data_ptr(),
        static_cast<char*>(destination.data_ptr()), destination_records.data_ptr(),
        source_owners.dtype().bits / 8, source_records.dtype().bits / 8,
        destination_records.dtype().bits / 8, peers.numel(), source_records.numel(), source_bank_byte_offset);
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

}  // namespace ulacache
