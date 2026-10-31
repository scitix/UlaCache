// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "csrc/cache/cpu_residency.cuh"

#include <utility>

namespace ulacache::paged_records {

template <int RecordsPerPage, int PageBytes, bool SegmentMajor, int... SegmentBytes>
struct NativePage {
  static constexpr int record_bytes = (SegmentBytes + ...);

  template <size_t Segment>
  static constexpr int segment_width() {
    constexpr int widths[]{SegmentBytes...};
    return widths[Segment];
  }

  template <size_t Segment>
  static constexpr int prefix_bytes() {
    constexpr int widths[]{SegmentBytes...};
    int prefix = 0;
    for (int index = 0; index < int(Segment); ++index) prefix += widths[index];
    return prefix;
  }

  template <size_t Segment>
  __device__ static int64_t offset(int64_t record) {
    constexpr int prefix = prefix_bytes<Segment>();
    constexpr int width = segment_width<Segment>();
    const int64_t page = record / RecordsPerPage;
    const int64_t slot = record % RecordsPerPage;
    if constexpr (SegmentMajor)
      return page * PageBytes + int64_t(prefix) * RecordsPerPage + slot * width;
    return page * PageBytes + slot * record_bytes + prefix;
  }

  template <int Bytes>
  __device__ static void copy_segment(int lane, const char* source, char* destination) {
    if constexpr (PageBytes % 8 == 0 && ((SegmentBytes % 8 == 0) && ...)) {
      sglang::flowkv_host::copy_record(lane, source, destination, Bytes);
    } else {
      for (int byte = lane; byte < Bytes; byte += 32) destination[byte] = source[byte];
    }
  }

  __device__ static void swap_eight_byte_word(char* host, char* device) {
    uint64_t host_value, device_value;
    asm volatile(
        "ld.global.cg.b64 %0,[%2];\n"
        "ld.global.cg.b64 %1,[%3];\n"
        "st.global.cg.b64 [%2],%1;\n"
        "st.global.cg.b64 [%3],%0;"
        : "=&l"(host_value), "=&l"(device_value)
        : "l"(host), "l"(device) : "memory");
  }

  template <int Bytes>
  __device__ static void swap_segment(int lane, char* host, char* device) {
    if constexpr (PageBytes % 8 == 0 && ((SegmentBytes % 8 == 0) && ...)) {
      // Both endpoints change in this kernel. Read-only .nc loads can move
      // past their endpoint's store during assembly and destroy the original.
      if (((reinterpret_cast<uintptr_t>(host) | reinterpret_cast<uintptr_t>(device)) & 15) == 0) {
        for (int pair = lane; pair < Bytes / 16; pair += 32) {
          uint64_t host_low, host_high, device_low, device_high;
          const int byte_offset = pair * 16;
          asm volatile(
              "ld.global.cg.v2.b64 {%0,%1},[%4];\n"
              "ld.global.cg.v2.b64 {%2,%3},[%5];\n"
              "st.global.cg.v2.b64 [%4],{%2,%3};\n"
              "st.global.cg.v2.b64 [%5],{%0,%1};"
              : "=&l"(host_low), "=&l"(host_high), "=&l"(device_low), "=&l"(device_high)
              : "l"(host + byte_offset), "l"(device + byte_offset) : "memory");
        }
        if constexpr (Bytes % 16 != 0) {
          if (lane == 0) swap_eight_byte_word(host + Bytes - 8, device + Bytes - 8);
        }
      } else {
        for (int word = lane; word < Bytes / 8; word += 32)
          swap_eight_byte_word(host + word * 8, device + word * 8);
      }
    } else {
      for (int byte = lane; byte < Bytes; byte += 32) {
        const char host_value = host[byte];
        const char device_value = device[byte];
        host[byte] = device_value;
        device[byte] = host_value;
      }
    }
  }

  template <size_t... Segments>
  __device__ static void copy(int lane, const char* source, char* destination,
                             int64_t source_record, int64_t destination_record,
                             std::index_sequence<Segments...>) {
    (copy_segment<segment_width<Segments>()>(lane, source + offset<Segments>(source_record),
                                  destination + offset<Segments>(destination_record)), ...);
  }

  template <size_t... Segments>
  __device__ static void exchange_segment(int lane, char* host, char* device,
      int device_record, int host_source, int host_destination,
      std::index_sequence<Segments...>) {
    if (host_source >= 0 && host_destination >= 0) {
      assert(host_source == host_destination);
      (swap_segment<segment_width<Segments>()>(lane, host + offset<Segments>(host_source),
                                     device + offset<Segments>(device_record)), ...);
    } else if (host_source >= 0) {
      copy(lane, host, device, host_source, device_record, std::index_sequence<Segments...>{});
    } else if (host_destination >= 0) {
      copy(lane, device, host, device_record, host_destination, std::index_sequence<Segments...>{});
    }
  }
};

template <typename Page, size_t Segments>
__global__ void exchange_kernel(char* host, char* device, const int* plan,
    const int* counters, int capacity, int banks, int64_t host_bank_stride,
    int64_t device_bank_stride) {
  const int count = counters[sglang::flowkv_host::PlanCount];
  if (count == 0) return;
  const int lane = threadIdx.x & 31;
  for (int64_t operation = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
       operation < int64_t(count) * banks;
       operation += gridDim.x * (blockDim.x / 32)) {
    const int index = operation % count;
    const int bank = operation / count;
    Page::exchange_segment(lane, host + int64_t(bank) * host_bank_stride,
        device + int64_t(bank) * device_bank_stride, plan[capacity + index],
        plan[2 * capacity + index], plan[3 * capacity + index],
        std::make_index_sequence<Segments>{});
  }
  // Publish host victims before the existing residency commit changes owners.
  __threadfence_system();
}

template <typename Page, size_t Segments>
__global__ void copy_kernel(const char* source, char* destination,
    const void* source_records, const void* destination_records,
    int source_index_bytes, int destination_index_bytes, int64_t count) {
  const int lane = threadIdx.x & 31;
  for (int64_t index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
       index < count; index += gridDim.x * (blockDim.x / 32)) {
    const int64_t source_record = sglang::flowkv_host::read_row(source_records, index, source_index_bytes);
    const int64_t destination_record = sglang::flowkv_host::read_row(destination_records, index, destination_index_bytes);
    if (source_record >= 0 && destination_record >= 0)
      Page::copy(lane, source, destination, source_record, destination_record,
                 std::make_index_sequence<Segments>{});
  }
  __threadfence_system();
}

}  // namespace ulacache::paged_records

namespace ulacache {

using tvm::ffi::TensorView;

template <int RecordsPerPage, int PageBytes, bool SegmentMajor, int... SegmentBytes>
void exchange_paged_records(TensorView host_storage, TensorView device_storage, TensorView plan,
                            TensorView counters, int64_t blocks, int64_t threads) {
  TVM_FFI_ICHECK(host_storage.ndim() == 3 && device_storage.ndim() == 3 &&
                host_storage.size(0) == device_storage.size(0));
  TVM_FFI_ICHECK(host_storage.dtype().bits == 8 && device_storage.dtype().bits == 8);
  TVM_FFI_ICHECK(host_storage.size(2) == PageBytes && device_storage.size(2) == PageBytes);
  TVM_FFI_ICHECK(host_storage.stride(1) == PageBytes && device_storage.stride(1) == PageBytes &&
                host_storage.stride(2) == 1 && device_storage.stride(2) == 1);
  TVM_FFI_ICHECK(plan.ndim() == 2 && plan.size(0) == 5 && plan.dtype().bits == 32 &&
                plan.stride(0) == plan.size(1) && plan.stride(1) == 1);
  TVM_FFI_ICHECK(counters.dtype().bits == 32 && counters.numel() >= 8);
  using Page = paged_records::NativePage<RecordsPerPage, PageBytes, SegmentMajor, SegmentBytes...>;
  const auto stream = sglang::host::LaunchKernel::resolve_device(device_storage.device());
  paged_records::exchange_kernel<Page, sizeof...(SegmentBytes)><<<blocks, threads, 0, stream>>>(
      static_cast<char*>(host_storage.data_ptr()), static_cast<char*>(device_storage.data_ptr()),
      static_cast<const int*>(plan.data_ptr()), static_cast<const int*>(counters.data_ptr()),
      int(plan.size(1)), int(host_storage.size(0)), host_storage.stride(0), device_storage.stride(0));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

template <int RecordsPerPage, int PageBytes, bool SegmentMajor, int... SegmentBytes>
void copy_paged_records(TensorView source, TensorView destination,
    TensorView source_records, TensorView destination_records,
    int64_t blocks, int64_t threads) {
  TVM_FFI_ICHECK(source.ndim() == 2 && destination.ndim() == 2 &&
                source.size(1) == PageBytes && destination.size(1) == PageBytes);
  TVM_FFI_ICHECK(source.dtype().bits == 8 && destination.dtype().bits == 8 &&
                source.stride(0) == PageBytes && destination.stride(0) == PageBytes &&
                source.stride(1) == 1 && destination.stride(1) == 1);
  TVM_FFI_ICHECK(source_records.ndim() == 1 && destination_records.ndim() == 1 &&
                source_records.numel() == destination_records.numel() &&
                source_records.stride(0) == 1 && destination_records.stride(0) == 1 &&
                (source_records.dtype().bits == 32 || source_records.dtype().bits == 64) &&
                (destination_records.dtype().bits == 32 || destination_records.dtype().bits == 64));
  using Page = paged_records::NativePage<RecordsPerPage, PageBytes, SegmentMajor, SegmentBytes...>;
  const auto stream = sglang::host::LaunchKernel::resolve_device(source_records.device());
  if (source_records.numel())
    paged_records::copy_kernel<Page, sizeof...(SegmentBytes)><<<blocks, threads, 0, stream>>>(
        static_cast<const char*>(source.data_ptr()), static_cast<char*>(destination.data_ptr()),
        source_records.data_ptr(), destination_records.data_ptr(), source_records.dtype().bits / 8,
        destination_records.dtype().bits / 8, source_records.numel());
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

}  // namespace ulacache
