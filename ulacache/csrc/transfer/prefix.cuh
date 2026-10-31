// SPDX-License-Identifier: Apache-2.0
// UDCache-derived bounded Plain MLA prefill. Included after the audited
// flowkv_record.cuh. Current records use query addressing; history is gathered
// to a compact page view without changing canonical native slot identities.
#ifdef FLOW_RECORD_NATIVE_LAYOUT
namespace sglang {
using tvm::ffi::TensorView;
template<typename Element>
__global__ void plain_prefill_write_staged_kernel(
    const Element* nope, const Element* rope, const void* loc, int loc_bytes,
    int nope_dim, long nope_row_stride, long nope_col_stride,
    long rope_row_stride, long rope_col_stride, const uint64_t* peers,
    uint8_t* resident, int* native_rows, int base, long layer_offset,
    int rank, bool sole_producer, int capacity, const int64_t* cursors,
    const int* totals, const int* active FLOW_HOST_DEVICE_PARAM) {
  if constexpr (FLOW_PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  // Match the native writer: permit the dependent kernel's independent
  // prologue to overlap these stores. Its dependency wait still requires
  // this grid to complete and flush before it consumes the written KV.
  if constexpr (FLOW_PDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int query = blockIdx.x, row = read_int(loc, query, loc_bytes);
  if (threadIdx.x == 0) native_rows[query] = row;
  const int live = gridDim.x;
  if (row > 0 && query < live) {
    const int gid = record_global_slot(row + base), owner = record_owner(gid);
    const bool publish = sole_producer || owner == rank;
    #if FLOW_RECORD_HOST_TIER
    uint8_t* cold = publish ? record_host_address(peers, gid, layer_offset, host, rank, true) : nullptr;
#elif FLOW_RECORD_REQUEST_TIER
    uint8_t* cold = reinterpret_cast<uint8_t*>(peers[owner] + layer_offset)
                    + record_request_local(gid, request_page_map) * FLOW_ROW_BYTES;
#else
    uint8_t* cold = reinterpret_cast<uint8_t*>(peers[owner] + layer_offset)
                    + long(record_local(gid)) * FLOW_ROW_BYTES;
#endif

    for (int word = threadIdx.x; word < FLOW_ROW_BYTES / 8; word += blockDim.x) {
      const int column = word * (8 / sizeof(Element));
      const uint64_t value = column < nope_dim
          ? record_plain_word(nope, query, column, nope_row_stride, nope_col_stride)
          : record_plain_word(rope, query, column - nope_dim, rope_row_stride, rope_col_stride);
      const int output_row = query;
      *reinterpret_cast<uint64_t*>(resident + RecordNativePage::offset(output_row, word * 8)) = value;
      if (publish) reinterpret_cast<uint64_t*>(cold)[word] = value;
    }
#if FLOW_RECORD_HOST_TIER
    if (publish && threadIdx.x == 0) record_host_writer_touch(gid, host, rank);
#endif
    // Prefill retains owner publication independently of decode step state.
    if (publish) __threadfence_system();
  }
}

template<typename Element>
void plain_prefill_write_staged_typed(
    TensorView nope, TensorView rope, TensorView loc, int64_t peer_ptrs,
    TensorView resident, TensorView native_rows, int64_t base, int64_t layer_offset,
    int64_t rank, bool sole, int64_t capacity, TensorView cursors,
    TensorView totals, TensorView active FLOW_HOST_TENSOR_PARAM) {
  TVM_FFI_ICHECK((nope.ndim() == 2 || (nope.ndim() == 3 && nope.shape()[1] == 1)) &&
      (rope.ndim() == 2 || (rope.ndim() == 3 && rope.shape()[1] == 1)));
  TVM_FFI_ICHECK(loc.ndim() == 1 && loc.strides()[0] == 1);
  TVM_FFI_ICHECK(nope.shape()[0] == loc.shape()[0] && rope.shape()[0] == loc.shape()[0]);
  const int nope_dim = nope.shape()[nope.ndim() - 1];
  const int rope_dim = rope.shape()[rope.ndim() - 1];
  TVM_FFI_ICHECK(nope.dtype().bits == 8 * sizeof(Element) &&
      (rope_dim == 0 || rope.dtype().bits == 8 * sizeof(Element)));
  TVM_FFI_ICHECK((nope_dim + rope_dim) * sizeof(Element) == FLOW_ROW_BYTES &&
      nope_dim % (8 / sizeof(Element)) == 0 && rope_dim % (8 / sizeof(Element)) == 0);
  TVM_FFI_ICHECK((loc.dtype().bits == 32 || loc.dtype().bits == 64) && native_rows.dtype().bits == 32 && native_rows.shape()[0] == loc.shape()[0]);
  const int count = loc.shape()[0];
  TVM_FFI_ICHECK(resident.ndim() == 2 && resident.dtype().bits == 8);
  TVM_FFI_ICHECK(resident.shape()[0] * FLOW_C4_PAGE_SIZE >= count);
  if (!count) return;
  auto stream = host::LaunchKernel::resolve_device(resident.device());
  host::RuntimeDeviceCheck(cache_launch(dim3(count), dim3(128), stream,
      plain_prefill_write_staged_kernel<Element>, static_cast<const Element*>(nope.data_ptr()),
      static_cast<const Element*>(rope.data_ptr()), loc.data_ptr(), loc.dtype().bits / 8,
      nope_dim, long(nope.strides()[0]), long(nope.strides()[nope.ndim() - 1]),
      long(rope.strides()[0]), long(rope.strides()[rope.ndim() - 1]), reinterpret_cast<const uint64_t*>(peer_ptrs),
      static_cast<uint8_t*>(resident.data_ptr()), static_cast<int*>(native_rows.data_ptr()),
      int(base), long(layer_offset), int(rank), sole, int(capacity),
      static_cast<const int64_t*>(cursors.data_ptr()), static_cast<const int*>(totals.data_ptr()),
      static_cast<const int*>(active.data_ptr()) FLOW_HOST_LAUNCH_ARG));
}

void plain_prefill_write_staged(
    TensorView nope, TensorView rope, TensorView loc, int64_t peer_ptrs,
    TensorView resident, TensorView native_rows, int64_t base, int64_t layer_offset,
    int64_t rank, bool sole, int64_t capacity, TensorView cursors,
    TensorView totals, TensorView active FLOW_HOST_TENSOR_PARAM) {
  TVM_FFI_ICHECK(nope.dtype().bits == 8 || nope.dtype().bits == 16);
  if (nope.dtype().bits == 8)
    plain_prefill_write_staged_typed<uint8_t>(nope, rope, loc, peer_ptrs,
        resident, native_rows, base, layer_offset, rank, sole, capacity, cursors, totals, active FLOW_HOST_TENSOR_ARG);
  else
    plain_prefill_write_staged_typed<uint16_t>(nope, rope, loc, peer_ptrs,
        resident, native_rows, base, layer_offset, rank, sole, capacity, cursors, totals, active FLOW_HOST_TENSOR_ARG);
}

#if FLOW_RECORD_HOST_TIER
__global__ void plain_prefill_acquire_kernel(const int64_t* host) {
  record_host_acquire_all(host);
}
#endif

__global__ void plain_prefill_fill_compact_kernel(const uint64_t* peers,
    const int* rows, const int* fresh_positions, int count,
    const uint8_t* staging, uint8_t* window, long offset,
    int rank, int base FLOW_HOST_DEVICE_PARAM) {
  const int lane = threadIdx.x & 31;
  for (int item = blockIdx.x * 4 + (threadIdx.x >> 5); item < count;
       item += gridDim.x * 4) {
    const int row = rows[item], fresh = fresh_positions[item];
    assert(row > 0);
    const int gid = record_global_slot(base + row);
    const uint8_t* source = nullptr;
    if (fresh < 0) {
#if FLOW_RECORD_HOST_TIER
      source = record_host_address(peers, gid, offset, host, rank, false);
#elif FLOW_RECORD_REQUEST_TIER
      source = reinterpret_cast<const uint8_t*>(peers[record_owner(gid)] + offset)
               + record_request_local(gid, request_page_map) * FLOW_ROW_BYTES;
#else
      source = reinterpret_cast<const uint8_t*>(peers[record_owner(gid)] + offset)
               + long(record_local(gid)) * FLOW_ROW_BYTES;
#endif
    }
    for (int word = lane; word < FLOW_ROW_BYTES / 8; word += 32) {
      const uint64_t value = fresh >= 0
          ? *reinterpret_cast<const uint64_t*>(staging + RecordNativePage::offset(fresh, word * 8))
          : reinterpret_cast<const uint64_t*>(source)[word];
      *reinterpret_cast<uint64_t*>(window + RecordNativePage::offset(item, word * 8)) = value;
    }
#if FLOW_RECORD_HOST_TIER
    if (fresh < 0 && lane == 0) record_host_touch(gid, host);
#endif
  }
}

void plain_prefill_fill_compact(int64_t ptrs, TensorView rows, TensorView staging,
    TensorView fresh_positions, TensorView window, int64_t offset, int64_t npeers,
    int64_t rank, int64_t base, bool sole FLOW_HOST_TENSOR_PARAM) {
  TVM_FFI_ICHECK(rows.ndim() == 1 && rows.dtype().bits == 32 && rows.strides()[0] == 1);
  TVM_FFI_ICHECK(fresh_positions.ndim() == 1 && fresh_positions.dtype().bits == 32 &&
      fresh_positions.strides()[0] == 1 && fresh_positions.shape()[0] == rows.shape()[0]);
  TVM_FFI_ICHECK(window.ndim() == 2 && window.dtype().bits == 8 &&
      window.shape()[0] * FLOW_C4_PAGE_SIZE >= rows.shape()[0]);
  const int count = rows.shape()[0];
  if (!count) return;
  auto stream = host::LaunchKernel::resolve_device(window.device());
#if FLOW_RECORD_HOST_TIER
  // One waiting CTA precedes the copy grid: do not occupy every SM with
  // peer waits while a producer still needs to finish its transfers.
  plain_prefill_acquire_kernel<<<1, 32, 0, stream>>>(
      static_cast<const int64_t*>(host_descriptor.data_ptr()));
  host::RuntimeDeviceCheck(cudaGetLastError());
#endif
  plain_prefill_fill_compact_kernel<<<std::min((count + 3) / 4, 65536), 128, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(ptrs), static_cast<const int*>(rows.data_ptr()),
      static_cast<const int*>(fresh_positions.data_ptr()), count,
      static_cast<const uint8_t*>(staging.data_ptr()), static_cast<uint8_t*>(window.data_ptr()),
      long(offset), int(rank), int(base) FLOW_HOST_LAUNCH_ARG);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

}  // namespace sglang
#endif
