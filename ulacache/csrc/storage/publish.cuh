// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "csrc/transfer/cpu_gpu.cuh"

namespace ulacache::paged_records {

// Publish an existing native writer result to its exclusive final tier. This
// combines the old location-map and two disjoint copies. NativePage::copy is
// the unchanged byte/segment implementation; no quantizer or KV math is here.
template <typename Page, size_t Segments, bool LocalDeviceReader = false>
__global__ void publish_kernel(const char* source, char* device, char* host,
    const void* logical_records, const void* source_records,
    const int64_t* directory, int logical_index_bytes, int source_index_bytes,
    int64_t count, int64_t logical_capacity, int64_t source_capacity,
    int64_t device_capacity, int64_t host_capacity) {
  const int lane = threadIdx.x & 31;
  for (int64_t index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
       index < count; index += gridDim.x * (blockDim.x / 32)) {
    const int64_t logical = sglang::flowkv_host::read_row(logical_records, index, logical_index_bytes);
    const int64_t source_record = sglang::flowkv_host::read_row(source_records, index, source_index_bytes);
    if (logical <= 0 || source_record < 0) continue;
    assert(logical < logical_capacity && source_record < source_capacity);
    const int64_t location = directory[logical];
    assert(location != -1);
    const bool on_device = location >= 0;
    if constexpr (LocalDeviceReader) {
      assert(on_device && "Committed native currents must have local GPU destinations");
    }
    const int64_t destination_record = on_device ? location : -location - 2;
    assert(destination_record >= 0 && destination_record <
        (on_device ? device_capacity : host_capacity));
    Page::copy(lane, source, on_device ? device : host, source_record,
        destination_record, std::make_index_sequence<Segments>{});
  }
  // Retain the old copy's publication fence, including pinned-host writes.
  if constexpr (!LocalDeviceReader) __threadfence_system();
}

}  // namespace ulacache::paged_records

namespace ulacache {

template <bool LocalDeviceReader, int RecordsPerPage, int PageBytes, bool SegmentMajor, int... SegmentBytes>
void publish_record_pages_impl(TensorView source, TensorView device, TensorView host,
    TensorView logical_records, TensorView source_records, TensorView directory) {
  TVM_FFI_ICHECK(source.ndim() == 2 && device.ndim() == 2 && host.ndim() == 2);
  for (auto storage : {source, device, host}) {
    TVM_FFI_ICHECK(storage.dtype().bits == 8 && storage.size(1) == PageBytes &&
        storage.stride(0) == PageBytes && storage.stride(1) == 1);
  }
  TVM_FFI_ICHECK(logical_records.ndim() == 1 && source_records.ndim() == 1 &&
      logical_records.numel() == source_records.numel() &&
      logical_records.stride(0) == 1 && source_records.stride(0) == 1 &&
      (logical_records.dtype().bits == 32 || logical_records.dtype().bits == 64) &&
      (source_records.dtype().bits == 32 || source_records.dtype().bits == 64));
  TVM_FFI_ICHECK(directory.ndim() == 1 && directory.dtype().bits == 64 && directory.stride(0) == 1);
  if constexpr (LocalDeviceReader) {
    const auto device_id = device.device().device_id;
    for (auto tensor : {source, device, logical_records, source_records, directory}) {
      TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA && tensor.device().device_id == device_id);
    }
  }
  const int64_t count = logical_records.numel();
  if (count) {
    using Page = paged_records::NativePage<RecordsPerPage, PageBytes, SegmentMajor, SegmentBytes...>;
    const auto stream = sglang::host::LaunchKernel::resolve_device(logical_records.device());
    constexpr int threads = 256;
    const int blocks = int(std::min<int64_t>(128, (count + threads / 32 - 1) / (threads / 32)));
    paged_records::publish_kernel<Page, sizeof...(SegmentBytes), LocalDeviceReader><<<blocks, threads, 0, stream>>>(
        static_cast<const char*>(source.data_ptr()), static_cast<char*>(device.data_ptr()),
        static_cast<char*>(host.data_ptr()), logical_records.data_ptr(), source_records.data_ptr(),
        static_cast<const int64_t*>(directory.data_ptr()), logical_records.dtype().bits / 8,
        source_records.dtype().bits / 8, count, directory.numel(),
        source.size(0) * RecordsPerPage, device.size(0) * RecordsPerPage,
        host.size(0) * RecordsPerPage);
    sglang::host::RuntimeDeviceCheck(cudaGetLastError());
  }
}

template <int RecordsPerPage, int PageBytes, bool SegmentMajor, int... SegmentBytes>
void publish_paged_records(TensorView source, TensorView device, TensorView host,
    TensorView logical_records, TensorView source_records, TensorView directory) {
  publish_record_pages_impl<false, RecordsPerPage, PageBytes, SegmentMajor, SegmentBytes...>(
      source, device, host, logical_records, source_records, directory);
}

// Private native decode entry: the transaction has committed these exact
// current IDs, and the only following reader is on this same GPU/stream.
// Ordinary public/prefill/host publication retains the system fence above.
template <int RecordsPerPage, int PageBytes, bool SegmentMajor, int... SegmentBytes>
void publish_local_current_records(TensorView source, TensorView device, TensorView host,
    TensorView logical_records, TensorView source_records, TensorView directory) {
  publish_record_pages_impl<true, RecordsPerPage, PageBytes, SegmentMajor, SegmentBytes...>(
      source, device, host, logical_records, source_records, directory);
}

}  // namespace ulacache
