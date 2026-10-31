// SPDX-License-Identifier: Apache-2.0
// Included after flowkv.cuh by load_jit. Reuse its proven planner and cache
// ABI; these entry points replace the feature-sharded transport completely.
// Every physical cold-store row is one aligned, contiguous complete record.
// Packed records keep the original C4 address calculation. Other native
// geometries supply only a compile-time page layout to the same transport.
#ifndef FLOW_RECORD_STEP_PUBLICATION
#define FLOW_RECORD_STEP_PUBLICATION 0
#endif

struct RecordPackedPage {
  __device__ __forceinline__ static long offset(long row, int byte) {
    return c4_byte_offset(row, byte);
  }
};
#ifdef FLOW_RECORD_NATIVE_LAYOUT
#include <sgl_kernel/flowkv/native_page_layout.cuh>
using RecordNativePage = sglang::flowkv::NativePageLayout<
    FLOW_C4_PAGE_SIZE, FLOW_C4_BYTES_PER_PAGE, FLOW_RECORD_PAGE_SEGMENT_BYTES>;
#else
using RecordNativePage = RecordPackedPage;
#endif

#ifdef FLOW_RECORD_NATIVE_LAYOUT
// The ordinary MLA writer reads projection views directly. One CTA emits a
// complete native row to its owner, the attention view, and (when the
// planner retained it) the resident history slot. No temporary concat/cast.
template<typename Element>
__device__ __forceinline__ uint64_t record_plain_word(
    const Element* input, int query, int column, long row_stride, long col_stride) {
  const Element* src = input + long(query) * row_stride + long(column) * col_stride;
  if (col_stride == 1 && (reinterpret_cast<uintptr_t>(src) & 7) == 0)
    return *reinterpret_cast<const uint64_t*>(src);
  uint64_t word = 0;
  #pragma unroll
  for (int i = 0; i < 8 / sizeof(Element); ++i)
    word |= uint64_t(src[long(i) * col_stride]) << (8 * sizeof(Element) * i);
  return word;
}

template<bool Decode, typename Element>
__global__ void record_write_plain_kernel(
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
  const int live = Decode ? *active : gridDim.x;
  if (row > 0 && query < live) {
    const int gid = record_global_slot(row + base), owner = record_owner(gid);
    const bool publish = sole_producer || owner == rank;
    #if FLOW_RECORD_HOST_TIER
    uint8_t* cold = publish ? record_host_writer_address_warp(peers, gid, layer_offset, host, rank) : nullptr;
#elif FLOW_RECORD_REQUEST_TIER
    uint8_t* cold = reinterpret_cast<uint8_t*>(peers[owner] + layer_offset)
                    + record_request_local(gid, request_page_map) * FLOW_ROW_BYTES;
#else
    uint8_t* cold = reinterpret_cast<uint8_t*>(peers[owner] + layer_offset)
                    + long(record_local(gid)) * FLOW_ROW_BYTES;
#endif
    const int64_t append = Decode ? cursors[0] + totals[0] : 0;
    for (int word = threadIdx.x; word < FLOW_ROW_BYTES / 8; word += blockDim.x) {
      const int column = word * (8 / sizeof(Element));
      const uint64_t value = column < nope_dim
          ? record_plain_word(nope, query, column, nope_row_stride, nope_col_stride)
          : record_plain_word(rope, query, column - nope_dim, rope_row_stride, rope_col_stride);
      const int output_row = Decode ? capacity + query : row;
      *reinterpret_cast<uint64_t*>(resident + RecordNativePage::offset(output_row, word * 8)) = value;
      if constexpr (Decode)
        if (append + live <= capacity)
          *reinterpret_cast<uint64_t*>(resident + RecordNativePage::offset(append + query, word * 8)) = value;
      if (publish) reinterpret_cast<uint64_t*>(cold)[word] = value;
    }
#if FLOW_RECORD_HOST_TIER
    // Published begin already pins each current GPU row in this epoch's
    // protected bucket and stamps GPU/CPU currents once per Index group.
    // Victim selection excludes that bucket; CPU currents keep their native
    // location. Prefill and nonpublished writers still own their first touch.
    if constexpr (!Decode || !FLOW_RECORD_STEP_PUBLICATION)
      if (publish && threadIdx.x == 0) record_host_writer_touch(gid, host, rank);
#endif
    // Only this ordinary MLA decode specialization uses step publication.
    // Prefill keeps its existing per-writer publication contract.
    if constexpr (!Decode || !FLOW_RECORD_STEP_PUBLICATION)
      if (publish) __threadfence_system();
  }
}

#endif

__global__ void record_globalize_kernel(const int* native_idx, const void* native_current,
    int current_bytes, int* global_idx, int* global_current, int queries,
    int base, bool update_current, bool already_global) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < queries * K) {
    const int row = native_idx[i];
    global_idx[i] = row > 0 && !already_global ? record_global_slot(row + base) : row;
  }
  if (update_current && i < queries) {
    const int row = read_int(native_current, i, current_bytes);
    global_current[i] = row > 0 ? record_global_slot(row + base) : row;
  }
}

__global__ void record_fill_native_kernel(const uint64_t* peers,
    const int* history, int nhistory, const uint8_t* staging,
    const int* current, int ncurrent, uint8_t* resident, long layer_offset,
    int npeers, int rank, int base, bool sole_producer FLOW_HOST_DEVICE_PARAM) {
  const int lane = threadIdx.x & 31;
#if FLOW_RECORD_HOST_TIER
  if (nhistory) record_host_acquire_all(host);
#endif
  const int total = nhistory + ncurrent;
  for (int item = blockIdx.x * 4 + (threadIdx.x >> 5); item < total;
       item += gridDim.x * 4) {
    const bool fresh = item >= nhistory;
    const int query = item - nhistory;
    const int row = fresh ? current[query] : history[item];
    if (row <= 0) continue;
    const int gid = record_global_slot(base + row), owner = record_owner(gid);
    const bool publish = fresh && (sole_producer || owner == rank);
#if FLOW_RECORD_HOST_TIER
    uint8_t* cold = (!fresh || publish)
        ? record_host_address(peers, gid, layer_offset, host, rank, fresh) : nullptr;
#elif FLOW_RECORD_REQUEST_TIER
    uint8_t* cold = reinterpret_cast<uint8_t*>(peers[owner] + layer_offset)
                    + record_request_local(gid, request_page_map) * FLOW_ROW_BYTES;
#else
    uint8_t* cold = reinterpret_cast<uint8_t*>(peers[owner] + layer_offset)
                    + long(record_local(gid)) * FLOW_ROW_BYTES;
#endif
    for (int word = lane; word < FLOW_ROW_BYTES / 8; word += 32) {
      const uint64_t value = fresh
          ? *reinterpret_cast<const uint64_t*>(staging + RecordNativePage::offset(query, word * 8))
          : reinterpret_cast<const uint64_t*>(cold)[word];
      *reinterpret_cast<uint64_t*>(resident + RecordNativePage::offset(row, word * 8)) = value;
      if (publish) reinterpret_cast<uint64_t*>(cold)[word] = value;
    }
#if FLOW_RECORD_HOST_TIER
    if (lane == 0 && (!fresh || publish)) {
      if (fresh) record_host_writer_touch(gid, host, rank);
      else record_host_touch(gid, host);
    }
#endif
    if (publish) __threadfence_system();
  }
}

__global__ void record_seed_native_kernel(const uint8_t* packed,
    const int* history, int nhistory, const int* current, int ncurrent,
    uint8_t* resident, int* tags, int64_t* pins, int* hints,
    int64_t* cursors, int base) {
  const int lane = threadIdx.x & 31, count = nhistory + ncurrent;
  if (blockIdx.x == 0 && threadIdx.x == 0) cursors[0] = cursors[1] = count;
  for (int slot = blockIdx.x * 4 + (threadIdx.x >> 5); slot < count;
       slot += gridDim.x * 4) {
    const int row = slot < nhistory ? history[slot] : current[slot - nhistory];
    if (row <= 0) continue;
    for (int word = lane; word < FLOW_ROW_BYTES / 8; word += 32)
      *reinterpret_cast<uint64_t*>(resident + RecordNativePage::offset(slot, word * 8)) =
          *reinterpret_cast<const uint64_t*>(packed + RecordNativePage::offset(row, word * 8));
    if (lane == 0) {
      const int gid = record_global_slot(row + base);
      tags[slot] = gid; pins[slot] = 0; hints[gid] = slot;
    }
  }
}

__global__ void record_publish_kernel(
    const uint64_t* peers, const uint8_t* staging, const int* source,
    const int* gids, int count, long layer_offset, int npeers,
    int rank, bool sole_producer
#if FLOW_RECORD_REQUEST_TIER
    , const int32_t* request_page_map
#endif
    ) {
  const int item = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (item >= count) return;
  const int gid = gids[item], row = source[item];
  if (gid <= 0 || row < 0) return;
  const int owner = record_owner(gid);
  // Replicated TP/CP producers elect the storage owner. A partitioned DPA/CP
  // producer is unique and may publish to a remote storage owner directly.
  if (!sole_producer && owner != rank) return;
  uint8_t* dst = reinterpret_cast<uint8_t*>(peers[owner] + layer_offset)
#if FLOW_RECORD_REQUEST_TIER
                 + record_request_local(gid, request_page_map) * FLOW_ROW_BYTES;
#else
                 + long(record_local(gid)) * FLOW_ROW_BYTES;
#endif
  const int lane = threadIdx.x & 31;
  for (int word = lane; word < FLOW_ROW_BYTES / 8; word += 32) {
    const uint64_t value = *reinterpret_cast<const uint64_t*>(staging + RecordNativePage::offset(row, word * 8));
    reinterpret_cast<uint64_t*>(dst)[word] = value;
  }
  __threadfence_system();
}

__global__ void record_gather_kernel(
    const uint64_t* peers, const int* gids, int count, uint8_t* resident,
    long layer_offset, int npeers
#if FLOW_RECORD_REQUEST_TIER
    , const int32_t* request_page_map
#endif
    ) {
  const int warp = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  for (int item = warp; item < count; item += gridDim.x * 4) {
    const int gid = gids[item];
    const uint8_t* src = gid >= 0
        ? reinterpret_cast<const uint8_t*>(peers[record_owner(gid)] + layer_offset)
#if FLOW_RECORD_REQUEST_TIER
            + record_request_local(gid, request_page_map) * FLOW_ROW_BYTES : nullptr;
#else
            + long(record_local(gid)) * FLOW_ROW_BYTES : nullptr;
#endif
    for (int word = lane; word < FLOW_ROW_BYTES / 8; word += 32) {
      const uint64_t value = src ? reinterpret_cast<const uint64_t*>(src)[word] : 0;
      *reinterpret_cast<uint64_t*>(resident + RecordNativePage::offset(item, word * 8)) = value;
    }
  }
}

template<bool History, bool Current>
__global__ void record_pull_cached_kernel(
    const uint64_t* peers, const int* sources, const int* targets,
    const int* counts, const int* idx, const int* hints, int* remap,
    const uint8_t* staging, const int* current, int capacity,
    uint8_t* resident, long layer_offset, int npeers, int rank,
    bool sole_producer, C4PullCacheState cache FLOW_HOST_DEVICE_PARAM) {
  record_pull_cached_impl<History, Current, true, RecordPackedPage>(
      peers, sources, targets, counts, idx, hints, remap,
      staging, current, capacity, resident, layer_offset, npeers, rank,
      sole_producer, cache FLOW_HOST_DEVICE_ARG);
}

#ifdef FLOW_RECORD_NATIVE_LAYOUT
template<bool History, bool Current, bool UpdateMetadata>
__global__ void record_pull_native_cached_kernel(
    const uint64_t* peers, const int* sources, const int* targets,
    const int* counts, const int* idx, const int* hints, int* remap,
    const uint8_t* staging, const int* current, int capacity,
    uint8_t* resident, long layer_offset, int npeers, int rank,
    bool sole_producer, C4PullCacheState cache FLOW_HOST_DEVICE_PARAM) {
  record_pull_cached_impl<History, Current, UpdateMetadata, RecordNativePage>(
      peers, sources, targets, counts, idx, hints, remap,
      staging, current, capacity, resident, layer_offset, npeers, rank,
      sole_producer, cache FLOW_HOST_DEVICE_ARG);
}
#endif

namespace sglang {

#if FLOW_RECORD_HOST_TIER
void flowkv_record_host_publish(TensorView descriptor, TensorView ready, int64_t rank) {
  TVM_FFI_ICHECK(descriptor.shape()[0] == HostRecordFields && descriptor.dtype().bits == 64 && ready.dtype().bits == 64);
  auto stream = host::LaunchKernel::resolve_device(ready.device());
  record_host_publish_kernel<<<1, 1, 0, stream>>>(
      static_cast<const int64_t*>(descriptor.data_ptr()), static_cast<int64_t*>(ready.data_ptr()));
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_record_host_prefill_read_done(TensorView descriptor, TensorView ready, int64_t rank) {
  auto stream = host::LaunchKernel::resolve_device(ready.device());
  record_host_read_done_kernel<<<1, 1, 0, stream>>>(
      static_cast<const int64_t*>(descriptor.data_ptr()), static_cast<int64_t*>(ready.data_ptr()));
  host::RuntimeDeviceCheck(cudaGetLastError());
}
#endif
#ifdef FLOW_RECORD_NATIVE_LAYOUT
template<bool Decode, typename Element>
void flowkv_record_write_plain_typed(
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
  if (!count) return;
  auto stream = host::LaunchKernel::resolve_device(resident.device());
  host::RuntimeDeviceCheck(cache_launch(dim3(count), dim3(Decode ? 32 : 128), stream,
      record_write_plain_kernel<Decode, Element>, static_cast<const Element*>(nope.data_ptr()),
      static_cast<const Element*>(rope.data_ptr()), loc.data_ptr(), loc.dtype().bits / 8,
      nope_dim, long(nope.strides()[0]), long(nope.strides()[nope.ndim() - 1]),
      long(rope.strides()[0]), long(rope.strides()[rope.ndim() - 1]), reinterpret_cast<const uint64_t*>(peer_ptrs),
      static_cast<uint8_t*>(resident.data_ptr()), static_cast<int*>(native_rows.data_ptr()),
      int(base), long(layer_offset), int(rank), sole, int(capacity),
      static_cast<const int64_t*>(cursors.data_ptr()), static_cast<const int*>(totals.data_ptr()),
      static_cast<const int*>(active.data_ptr()) FLOW_HOST_LAUNCH_ARG));
}

template<bool Decode>
void flowkv_record_write_plain(
    TensorView nope, TensorView rope, TensorView loc, int64_t peer_ptrs,
    TensorView resident, TensorView native_rows, int64_t base, int64_t layer_offset,
    int64_t rank, bool sole, int64_t capacity, TensorView cursors,
    TensorView totals, TensorView active FLOW_HOST_TENSOR_PARAM) {
  TVM_FFI_ICHECK(nope.dtype().bits == 8 || nope.dtype().bits == 16);
  if (nope.dtype().bits == 8)
    flowkv_record_write_plain_typed<Decode, uint8_t>(nope, rope, loc, peer_ptrs,
        resident, native_rows, base, layer_offset, rank, sole, capacity, cursors, totals, active FLOW_HOST_TENSOR_ARG);
  else
    flowkv_record_write_plain_typed<Decode, uint16_t>(nope, rope, loc, peer_ptrs,
        resident, native_rows, base, layer_offset, rank, sole, capacity, cursors, totals, active FLOW_HOST_TENSOR_ARG);
}
#endif

// Normal, non-PDL task: stream/Graph completion dependencies acquire all
// earlier writer grids before this thread cumulatively publishes their data.
__global__ void record_publish_step_kernel(const int64_t* step, int64_t* ready
#if FLOW_RECORD_HOST_TIER
    , const int64_t* host_groups, int groups
#endif
    ) {
#if FLOW_RECORD_HOST_TIER
  for (int group = 0; group < groups; ++group) {
    const auto* frame = host_groups + int64_t(group) * sglang::flowkv_host::kFrameWords;
    auto* read_done = reinterpret_cast<int64_t*>(frame[sglang::flowkv_host::FReady]) + 1;
    const int64_t epoch = *reinterpret_cast<const int64_t*>(frame[sglang::flowkv_host::FEpoch]);
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*read_done)
        .store(epoch, cuda::memory_order_release);
  }
#endif
  cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*ready)
      .store(*step, cuda::memory_order_release);
}

void flowkv_record_publish_step(TensorView step, TensorView ready
#if FLOW_RECORD_HOST_TIER
    , TensorView host_groups
#endif
    ) {
  TVM_FFI_ICHECK(step.shape()[0] == 1 && ready.shape()[0] == 1 &&
      step.dtype().bits == 64 && ready.dtype().bits == 64);
  auto stream = host::LaunchKernel::resolve_device(step.device());
  record_publish_step_kernel<<<1, 1, 0, stream>>>(
      static_cast<const int64_t*>(step.data_ptr()), static_cast<int64_t*>(ready.data_ptr())
#if FLOW_RECORD_HOST_TIER
      , static_cast<const int64_t*>(host_groups.data_ptr()), int(host_groups.shape()[0])
#endif
      );
  host::RuntimeDeviceCheck(cudaGetLastError());
}

// Startup-only. System atomics on GPU memory require native atomic support
// from each accessing GPU to the GPU owning the peer flag.
void flowkv_record_validate_step_peer(TensorView local, int64_t pointer) {
  host::LaunchKernel::resolve_device(local.device());
  TVM_FFI_ICHECK(pointer != 0 && (uint64_t(pointer) & 7) == 0);
  cudaPointerAttributes attributes{};
  host::RuntimeDeviceCheck(cudaPointerGetAttributes(&attributes,
      reinterpret_cast<const void*>(uintptr_t(pointer))));
  TVM_FFI_ICHECK(attributes.type == cudaMemoryTypeDevice && attributes.device >= 0);
  int device = -1, supported = 1;
  host::RuntimeDeviceCheck(cudaGetDevice(&device));
  if (device != attributes.device)
    host::RuntimeDeviceCheck(cudaDeviceGetP2PAttribute(&supported,
        cudaDevP2PAttrNativeAtomicSupported, device, attributes.device));
  TVM_FFI_ICHECK(supported == 1) << "TP peer does not support system-scope GPU atomics";
}

void flowkv_record_begin_published(
    TensorView current, TensorView valid, int64_t groups, int64_t pool_rows,
    TensorView step, TensorView active, TensorView keys, TensorView values,
    TensorView cursors, TensorView totals, TensorView hints, TensorView counts,
    TensorView histogram, TensorView ready_peers
#if FLOW_RECORD_HOST_TIER
    , TensorView host_groups
#endif
    ) {
  TVM_FFI_ICHECK(current.shape()[0] <= MAX_QUERIES &&
      (current.dtype().bits == 32 || current.dtype().bits == 64) &&
      ready_peers.dtype().bits == 64 && ready_peers.shape()[0] == FLOW_RECORD_PEERS);
  auto stream = host::LaunchKernel::resolve_device(current.device());
  // The mature ring's owner-local expiry uses the NT256 CTA collective.
  // GPU-only step publication retains its existing NT128 launch.
  begin_global_step<false, true><<<1, FLOW_RECORD_HOST_TIER ? 256 : 128, 0, stream>>>(
      current.data_ptr(), current.dtype().bits / 8, valid.data_ptr(), valid.dtype().bits / 8,
      int(current.shape()[0]), MAX_QUERIES, int(groups), long(pool_rows),
      static_cast<int64_t*>(step.data_ptr()), static_cast<int*>(active.data_ptr()),
      static_cast<int*>(keys.data_ptr()), static_cast<int*>(values.data_ptr()),
      static_cast<int64_t*>(cursors.data_ptr()), static_cast<int*>(totals.data_ptr()),
      static_cast<int*>(hints.data_ptr()), static_cast<int*>(counts.data_ptr()),
      static_cast<int*>(histogram.data_ptr()), nullptr, 0, 0, 1,
      static_cast<const uint64_t*>(ready_peers.data_ptr()), FLOW_RECORD_PEERS
#if FLOW_RECORD_HOST_TIER
      , nullptr // group transaction follows this kernel's complete retirement
#endif
      );
  host::RuntimeDeviceCheck(cudaGetLastError());
#if FLOW_RECORD_HOST_TIER
  if (host_groups.data_ptr()) {
    // No new peer barrier or device child: ordinary launch completion orders
    // all group epochs/current pins before the original lookup/allocator.
    sglang::flowkv_host::begin_groups_parallel<<<int(groups), 256, 0, stream>>>(
        static_cast<const int64_t*>(host_groups.data_ptr()), int(groups),
        current.data_ptr(), current.dtype().bits / 8, int(current.shape()[0]),
        static_cast<const int*>(active.data_ptr()));
    host::RuntimeDeviceCheck(cudaGetLastError());
  }
#endif
}

void flowkv_record_begin(
    TensorView current, TensorView global, int64_t base, int64_t groups, int64_t pool_rows,
    TensorView step, TensorView active, TensorView keys, TensorView values,
    TensorView cursors, TensorView totals, TensorView hints, TensorView counts, TensorView histogram) {
  TVM_FFI_ICHECK(current.shape()[0] == global.shape()[0] && current.shape()[0] <= MAX_QUERIES &&
      (current.dtype().bits == 32 || current.dtype().bits == 64) && global.dtype().bits == 32);
  const auto stream = host::LaunchKernel::resolve_device(current.device());
  begin_global_step<true><<<1, 128, 0, stream>>>(
      current.data_ptr(), current.dtype().bits / 8, nullptr, 0, int(current.shape()[0]),
      MAX_QUERIES, int(groups), long(pool_rows), static_cast<int64_t*>(step.data_ptr()),
      static_cast<int*>(active.data_ptr()), static_cast<int*>(keys.data_ptr()),
      static_cast<int*>(values.data_ptr()), static_cast<int64_t*>(cursors.data_ptr()),
      static_cast<int*>(totals.data_ptr()), static_cast<int*>(hints.data_ptr()),
      static_cast<int*>(counts.data_ptr()), static_cast<int*>(histogram.data_ptr()),
      static_cast<int*>(global.data_ptr()), int(base), FLOW_RECORD_SLOTS, FLOW_RECORD_PEERS);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

// Convert native rows inside the existing lookup, as in the verified GLM
// prefetch path. V4 native page allocation gives a linear owner namespace.
void flowkv_record_plan_native(
    TensorView native, TensorView idx, int64_t base,
    TensorView step, TensorView active, TensorView keys, TensorView values,
    TensorView tags, TensorView pins, TensorView hint, TensorView remap,
    TensorView sources, TensorView targets, TensorView counts, TensorView histogram,
    TensorView cursors, TensorView totals, TensorView errors,
    int64_t capacity, int64_t pool_rows
#if FLOW_RECORD_HOST_TIER
    , TensorView selected_ready, TensorView host_group
#endif
    ) {
  TVM_FFI_ICHECK(native.shape()[0] == idx.shape()[0] &&
      native.shape()[1] == K && idx.shape()[1] == K &&
      native.dtype().bits == 32 && pool_rows == FLOW_RECORD_PEERS * FLOW_RECORD_SLOTS);
  auto stream = host::LaunchKernel::resolve_device(native.device());
  host::RuntimeDeviceCheck(cache_launch(
      dim3(native.shape()[0] * (K / NT)), dim3(NT), stream,
      global_lookup_pin<true>, static_cast<const int*>(native.data_ptr()),
      static_cast<const int64_t*>(step.data_ptr()), static_cast<const int*>(active.data_ptr()),
      static_cast<const int*>(keys.data_ptr()), static_cast<const int*>(values.data_ptr()),
      static_cast<const int*>(tags.data_ptr()), static_cast<int64_t*>(pins.data_ptr()),
      static_cast<int*>(hint.data_ptr()), static_cast<int*>(remap.data_ptr()),
      static_cast<int*>(sources.data_ptr()), static_cast<int*>(counts.data_ptr()),
      static_cast<int*>(nullptr), static_cast<int*>(totals.data_ptr()),
      static_cast<const int64_t*>(cursors.data_ptr()), int(capacity), long(pool_rows),
      static_cast<int*>(idx.data_ptr()), int(base),
      int(FLOW_RECORD_SLOTS), int(FLOW_RECORD_PEERS)));
#if FLOW_RECORD_HOST_TIER
  TVM_FFI_ICHECK(selected_ready.numel() == 1 && selected_ready.dtype().bits == 64);
  host::RuntimeDeviceCheck(static_cast<cudaError_t>(flowkv_global_plan(
      static_cast<const int*>(idx.data_ptr()),
      static_cast<const int64_t*>(step.data_ptr()),
      static_cast<const int*>(active.data_ptr()),
      static_cast<const int*>(keys.data_ptr()),
      static_cast<const int*>(values.data_ptr()),
      static_cast<int*>(tags.data_ptr()), static_cast<int64_t*>(pins.data_ptr()),
      static_cast<int*>(hint.data_ptr()), static_cast<int*>(remap.data_ptr()),
      static_cast<int*>(sources.data_ptr()), static_cast<int*>(targets.data_ptr()),
      static_cast<int*>(counts.data_ptr()), nullptr,
      static_cast<int*>(histogram.data_ptr()), static_cast<int64_t*>(cursors.data_ptr()),
      static_cast<int*>(totals.data_ptr()), static_cast<int*>(errors.data_ptr()), nullptr,
      int(idx.shape()[0]), int(capacity), long(pool_rows), stream, true,
      static_cast<int64_t*>(selected_ready.data_ptr()),
      static_cast<const int64_t*>(host_group.data_ptr()))));
#else
  flowkv_plan<true>(idx, step, active, keys, values, tags, pins, hint, remap,
      sources, targets, counts, histogram, cursors, totals, errors, capacity, pool_rows);
#endif
}

__global__ void record_globalize_current_kernel(const void* native, int bytes, int* global,
    int queries, int base) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < queries) {
    const int row = read_int(native, i, bytes);
    global[i] = row > 0 ? record_global_slot(row + base) : row;
  }
}

void flowkv_record_globalize_current(TensorView native, TensorView global, int64_t base) {
  auto stream = host::LaunchKernel::resolve_device(native.device());
  if (native.shape()[0]) record_globalize_current_kernel<<<(native.shape()[0] + 127) / 128, 128, 0, stream>>>(
      native.data_ptr(), native.dtype().bits / 8, static_cast<int*>(global.data_ptr()), native.shape()[0], base);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_record_globalize(TensorView native_idx, TensorView native_current,
    TensorView global_idx, TensorView global_current, int64_t base, bool update_current) {
  const int n = native_idx.shape()[0];
  auto stream = host::LaunchKernel::resolve_device(native_idx.device());
  if (n) record_globalize_kernel<<<(n * K + 255) / 256, 256, 0, stream>>>(
      static_cast<const int*>(native_idx.data_ptr()), native_current.data_ptr(),
      native_current.dtype().bits / 8, static_cast<int*>(global_idx.data_ptr()),
      static_cast<int*>(global_current.data_ptr()), n, base, update_current, false);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_record_globalize_bound(TensorView native_idx, TensorView native_current,
    TensorView global_idx, TensorView global_current, int64_t base,
    bool update_current, bool already_global) {
  const int n = native_idx.shape()[0];
  auto stream = host::LaunchKernel::resolve_device(native_idx.device());
  if (n) record_globalize_kernel<<<(n * K + 255) / 256, 256, 0, stream>>>(
      static_cast<const int*>(native_idx.data_ptr()), native_current.data_ptr(),
      native_current.dtype().bits / 8, static_cast<int*>(global_idx.data_ptr()),
      static_cast<int*>(global_current.data_ptr()), n, base, update_current, already_global);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_record_fill_native(int64_t ptrs, TensorView history, TensorView staging,
    TensorView current, TensorView resident, int64_t offset, int64_t npeers,
    int64_t rank, int64_t base, bool sole FLOW_HOST_TENSOR_PARAM) {
  const int nh = history.shape()[0], nc = current.shape()[0], count = nh + nc;
  auto stream = host::LaunchKernel::resolve_device(resident.device());
  if (count) record_fill_native_kernel<<<std::min((count + 3) / 4, 65536), 128, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(ptrs), static_cast<const int*>(history.data_ptr()), nh,
      static_cast<const uint8_t*>(staging.data_ptr()), static_cast<const int*>(current.data_ptr()), nc,
      static_cast<uint8_t*>(resident.data_ptr()), offset, npeers, rank, base, sole FLOW_HOST_LAUNCH_ARG);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_record_seed_native(TensorView packed, TensorView history,
    TensorView current, TensorView resident, TensorView tags, TensorView pins,
    TensorView hints, TensorView cursors, int64_t base) {
  const int nh = history.shape()[0], nc = current.shape()[0], count = nh + nc;
  auto stream = host::LaunchKernel::resolve_device(resident.device());
  TVM_FFI_ICHECK(count <= tags.shape()[0]);
  host::RuntimeDeviceCheck(cudaMemsetAsync(tags.data_ptr(), 0xff, tags.shape()[0] * sizeof(int), stream));
  record_seed_native_kernel<<<std::max(1, std::min((count + 3) / 4, 65536)), 128, 0, stream>>>(
      static_cast<const uint8_t*>(packed.data_ptr()), static_cast<const int*>(history.data_ptr()), nh,
      static_cast<const int*>(current.data_ptr()), nc, static_cast<uint8_t*>(resident.data_ptr()),
      static_cast<int*>(tags.data_ptr()), static_cast<int64_t*>(pins.data_ptr()),
      static_cast<int*>(hints.data_ptr()), static_cast<int64_t*>(cursors.data_ptr()), base);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_record_publish(int64_t ptrs, TensorView staging, TensorView source,
    TensorView gids, int64_t offset, int64_t npeers, int64_t rank, bool sole
#if FLOW_RECORD_REQUEST_TIER
    , TensorView request_page_map
#endif
    ) {
  const int n = gids.shape()[0];
  auto stream = host::LaunchKernel::resolve_device(staging.device());
  if (n) record_publish_kernel<<<(n + 3) / 4, 128, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(ptrs), static_cast<const uint8_t*>(staging.data_ptr()),
      static_cast<const int*>(source.data_ptr()), static_cast<const int*>(gids.data_ptr()),
      n, offset, npeers, rank, sole
#if FLOW_RECORD_REQUEST_TIER
      , static_cast<const int32_t*>(request_page_map.data_ptr())
#endif
      );
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void flowkv_record_gather(int64_t ptrs, TensorView gids, TensorView resident,
    int64_t offset, int64_t npeers
#if FLOW_RECORD_REQUEST_TIER
    , TensorView request_page_map
#endif
    ) {
  const int n = gids.shape()[0];
  auto stream = host::LaunchKernel::resolve_device(resident.device());
  if (n) record_gather_kernel<<<std::min((n + 3) / 4, 4096), 128, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(ptrs), static_cast<const int*>(gids.data_ptr()),
      n, static_cast<uint8_t*>(resident.data_ptr()), offset, npeers
#if FLOW_RECORD_REQUEST_TIER
      , static_cast<const int32_t*>(request_page_map.data_ptr())
#endif
      );
  host::RuntimeDeviceCheck(cudaGetLastError());
}

template<bool History = true, bool Current = true>
void flowkv_record_pull_cached(int64_t peer_ptrs, TensorView sources,
    TensorView targets, TensorView counts, TensorView idx, TensorView hints,
    TensorView remap, TensorView staging, TensorView current, int64_t capacity,
    TensorView resident, int64_t layer_offset, int64_t npeers, int64_t rank,
    int64_t blocks, bool sole_producer, TensorView tags, TensorView pins,
    TensorView cursors, TensorView totals, TensorView step, TensorView active,
    TensorView current_keys, TensorView current_values FLOW_HOST_TENSOR_PARAM) {
  const int batch = idx.shape()[0];
  if (batch < 1 || batch > MAX_QUERIES || blocks < 1 || capacity < 1 ||
      npeers < 1 || rank < 0 || rank >= npeers) {
    host::RuntimeDeviceCheck(cudaErrorInvalidValue); return;
  }
  const C4PullCacheState cache{static_cast<int*>(tags.data_ptr()),
      static_cast<int64_t*>(pins.data_ptr()), static_cast<int64_t*>(cursors.data_ptr()),
      static_cast<const int*>(totals.data_ptr()), static_cast<const int64_t*>(step.data_ptr()),
      static_cast<const int*>(active.data_ptr()), static_cast<const int*>(current_keys.data_ptr()),
      static_cast<const int*>(current_values.data_ptr())};
  auto stream = host::LaunchKernel::resolve_device(resident.device());
  host::RuntimeDeviceCheck(cache_launch(dim3(blocks, batch), dim3(128), stream,
      record_pull_cached_kernel<History, Current>, reinterpret_cast<const uint64_t*>(peer_ptrs),
      static_cast<const int*>(sources.data_ptr()), static_cast<const int*>(targets.data_ptr()),
      static_cast<const int*>(counts.data_ptr()), static_cast<const int*>(idx.data_ptr()),
      static_cast<const int*>(hints.data_ptr()), static_cast<int*>(remap.data_ptr()),
      static_cast<const uint8_t*>(staging.data_ptr()), static_cast<const int*>(current.data_ptr()),
      int(capacity), static_cast<uint8_t*>(resident.data_ptr()), long(layer_offset),
      int(npeers), int(rank), sole_producer, cache FLOW_HOST_LAUNCH_ARG));
}

#ifdef FLOW_RECORD_NATIVE_LAYOUT
template<bool History = true, bool Current = true, bool UpdateMetadata = true>
void flowkv_record_pull_native_cached(int64_t peer_ptrs, TensorView sources,
    TensorView targets, TensorView counts, TensorView idx, TensorView hints,
    TensorView remap, TensorView staging, TensorView current, int64_t capacity,
    TensorView resident, int64_t layer_offset, int64_t npeers, int64_t rank,
    int64_t blocks, bool sole_producer, TensorView tags, TensorView pins,
    TensorView cursors, TensorView totals, TensorView step, TensorView active,
    TensorView current_keys, TensorView current_values FLOW_HOST_TENSOR_PARAM) {
  const int batch = idx.shape()[0];
  if (batch < 1 || batch > MAX_QUERIES || blocks < 1 || capacity < 1 ||
      npeers < 1 || rank < 0 || rank >= npeers) {
    host::RuntimeDeviceCheck(cudaErrorInvalidValue); return;
  }
  const C4PullCacheState cache{static_cast<int*>(tags.data_ptr()),
      static_cast<int64_t*>(pins.data_ptr()), static_cast<int64_t*>(cursors.data_ptr()),
      static_cast<const int*>(totals.data_ptr()), static_cast<const int64_t*>(step.data_ptr()),
      static_cast<const int*>(active.data_ptr()), static_cast<const int*>(current_keys.data_ptr()),
      static_cast<const int*>(current_values.data_ptr())};
  auto stream = host::LaunchKernel::resolve_device(resident.device());
  host::RuntimeDeviceCheck(cache_launch(dim3(blocks, batch), dim3(128), stream,
      record_pull_native_cached_kernel<History, Current, UpdateMetadata>, reinterpret_cast<const uint64_t*>(peer_ptrs),
      static_cast<const int*>(sources.data_ptr()), static_cast<const int*>(targets.data_ptr()),
      static_cast<const int*>(counts.data_ptr()), static_cast<const int*>(idx.data_ptr()),
      static_cast<const int*>(hints.data_ptr()), static_cast<int*>(remap.data_ptr()),
      static_cast<const uint8_t*>(staging.data_ptr()), static_cast<const int*>(current.data_ptr()),
      int(capacity), static_cast<uint8_t*>(resident.data_ptr()), long(layer_offset),
      int(npeers), int(rank), sole_producer, cache FLOW_HOST_LAUNCH_ARG));
}
#endif
}  // namespace sglang

#if FLOW_RECORD_REQUEST_TIER
// Admission reuses the original prior-step retirement, without advancing
// its epoch or mutating any native decode state.
__global__ void record_retire_request_pages_kernel(
    const int64_t* step, const uint64_t* ready_peers) {
  if (threadIdx.x == 0) {
    const int64_t prior = *step;
    for (int peer = 0; peer < FLOW_RECORD_PEERS; ++peer) {
      auto* ready = reinterpret_cast<int64_t*>(ready_peers[peer]);
      while (cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*ready)
                 .load(cuda::memory_order_acquire) < prior) {}
    }
  }
}

namespace sglang {
void flowkv_record_retire_request_pages(TensorView step, TensorView ready_peers) {
  TVM_FFI_ICHECK(step.shape()[0] == 1 && step.dtype().bits == 64 &&
      ready_peers.shape()[0] == FLOW_RECORD_PEERS && ready_peers.dtype().bits == 64);
  auto stream = host::LaunchKernel::resolve_device(step.device());
  record_retire_request_pages_kernel<<<1, 1, 0, stream>>>(
      static_cast<const int64_t*>(step.data_ptr()),
      static_cast<const uint64_t*>(ready_peers.data_ptr()));
  host::RuntimeDeviceCheck(cudaGetLastError());
}
}  // namespace sglang
#endif
