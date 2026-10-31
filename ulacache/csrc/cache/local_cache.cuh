// Rank-local native allocation bindings; no IPC namespace or peer publication.
#pragma once
#include "csrc/cache/residency_fused.cuh"

namespace sglang::flowkv_host {
__global__ void register_unbound_local(const void* rows, int bytes, int count,
                                     const int64_t* frames, bool prefer_gpu) {
  const Frame frame{frames + int64_t(blockIdx.y) * kFrameWords};
  const State state = frame.state();
  assert(frame.d[FPeers] == 1 && frame.d[FRank] == 0);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    const auto row = read_row(rows, i, bytes);
    if (row > 0 && row < state.logical_slots && state.directory[row] == -1)
      register_row(row, 0, prefer_gpu, state, &frame.group());
  }
}

// Reserve an ordered tile, rather than one stack pop per CUDA lane. Canonical
// receive destinations then retain source-record contiguity whenever the free
// stack contains an extent. Fragmented stacks remain valid scatter targets.
__device__ __forceinline__ int reserve_free_tile(int* counts, int wanted, int* take) {
  int available = atomicAdd(counts, 0);
  while (true) {
    if (available < 0) { available = atomicAdd(counts, 0); continue; }
    const int n = min(wanted, available);
    const int observed = atomicCAS(counts, available, available - n);
    if (observed == available) { *take = n; return available - n; }
    available = observed;
  }
}

__global__ void register_receive_ordered(const void* rows, int bytes, int count,
                                        const int64_t* frames, bool prefer_gpu) {
  __shared__ int warp_base[8], gpu_base, host_base, gpu_count, host_count;
  const Frame frame{frames + int64_t(blockIdx.y) * kFrameWords};
  const State s = frame.state();
  assert(frame.d[FPeers] == 1 && frame.d[FRank] == 0);
  const int i = blockIdx.x * 256 + threadIdx.x;
  const int64_t row = i < count ? read_row(rows, i, bytes) : -1;
  const bool valid = row > 0 && row < s.logical_slots && s.directory[row] == -1;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint32_t bits = __ballot_sync(0xffffffffu, valid);
  const int prefix = __popc(bits & ((uint32_t(1) << lane) - 1));
  if (lane == 0) warp_base[warp] = __popc(bits);
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int w = 0; w < 8; ++w) { const int n = warp_base[w]; warp_base[w] = total; total += n; }
    if (prefer_gpu) {
      gpu_base = reserve_free_tile(s.free_counts, total, &gpu_count);
      host_base = reserve_free_tile(s.free_counts + 1, total - gpu_count, &host_count);
    } else {
      host_base = reserve_free_tile(s.free_counts + 1, total, &host_count);
      gpu_base = reserve_free_tile(s.free_counts, total - host_count, &gpu_count);
    }
    assert(gpu_count + host_count == total);
  }
  __syncthreads();
  if (!valid) return;
  const int ordinal = warp_base[warp] + prefix;
  const bool gpu = prefer_gpu ? ordinal < gpu_count : ordinal >= host_count;
  cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.access_time[row])
      .store(*s.epoch, cuda::memory_order_relaxed);
  if (gpu) {
    const int slot = s.gpu_free[gpu_base + ordinal - (prefer_gpu ? 0 : host_count)];
    s.gpu_tags[slot] = row;
    attach_registered(frame.group(), slot, *s.epoch);
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.directory[row])
        .store(slot, cuda::memory_order_release);
  } else {
    const int slot = s.host_free[host_base + ordinal - (prefer_gpu ? gpu_count : 0)];
    s.host_tags[slot] = row;
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.directory[row])
        .store(-int64_t(slot) - 2, cuda::memory_order_release);
  }
}

__global__ void acquire_receive_locations(const void* rows, int bytes, int count,
                                          const int64_t* frames, int64_t* locations) {
  const Frame frame{frames + int64_t(blockIdx.y) * kFrameWords};
  const State s = frame.state();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    const int64_t row = read_row(rows, i, bytes);
    assert(row > 0 && row < s.logical_slots);
    const int64_t location = s.directory[row];
    assert(location != -1);
    if (location >= 0) {
      assert(location < s.gpu_slots && s.gpu_tags[location] == row);
      const int before = atomicAdd(s.leases + location, 1);
      if (before == 0) {
#if !UDCACHE_NAIVE_LRU
        // The optimized selectors enumerate presence buckets, rather than
        // scanning lease counters. Exclude this destination from those buckets
        // until remote writers retire; free-stack ownership remains unchanged.
        recent::mark_slot(frame.group(), int(location), ring::Operation::Detach, *s.epoch);
#endif
      }
    } else {
      const int64_t host = -location - 2;
      assert(host >= 0 && host < s.host_slots && s.host_tags[host] == row);
    }
    locations[int64_t(blockIdx.y) * count + i] = location;
  }
}

__global__ void release_receive_locations(const void* rows, int bytes, int count,
                                          const int64_t* frames, const int64_t* locations) {
  const Frame frame{frames + int64_t(blockIdx.y) * kFrameWords};
  const State s = frame.state();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    const int64_t row = read_row(rows, i, bytes);
    const int64_t location = locations[int64_t(blockIdx.y) * count + i];
    assert(row > 0 && row < s.logical_slots && s.directory[row] == location);
    if (location >= 0) {
      assert(location < s.gpu_slots && s.gpu_tags[location] == row);
      const int before = atomicSub(s.leases + location, 1);
      assert(before > 0);
      if (before == 1) {
        cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.access_time[row])
            .store(*s.epoch, cuda::memory_order_relaxed);
        attach_registered(frame.group(), int(location), *s.epoch);
      }
    } else {
      const int64_t host = -location - 2;
      assert(host >= 0 && host < s.host_slots && s.host_tags[host] == row);
    }
  }
}

// The caller permutes only CPU slots already owned by this unpublished receive.
// Every selected host slot has exactly one new owner; existing KV bytes stay put.
__global__ void rebind_cpu_receive(const void* rows, int bytes, int count,
                                 const int64_t* frames, const int64_t* locations) {
  const Frame frame{frames + int64_t(blockIdx.y) * kFrameWords};
  const State s = frame.state();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    const int64_t row = read_row(rows, i, bytes);
    assert(row > 0 && row < s.logical_slots);
    const int64_t before = s.directory[row];
    const int64_t after = locations[int64_t(blockIdx.y) * count + i];
    if (before >= 0) { assert(after == before); continue; }
    assert(before <= -2 && after <= -2);
    const int64_t slot = -after - 2;
    assert(slot >= 0 && slot < s.host_slots);
    s.host_tags[slot] = row;
    cuda::atomic_ref<int64_t, cuda::thread_scope_system>(s.directory[row])
        .store(after, cuda::memory_order_release);
  }
}

// Each group owns its expiry list and recent/cold metadata. Transfer plans
// remain shared and serialized. Advance only the bucket headers here; the
// original per-group begin still advances canonical time and pins currents.
__global__ void expire_groups_parallel(const int64_t* frames) {
  __shared__ recent::Storage<256> storage;
  const Frame frame{frames + int64_t(blockIdx.x) * kFrameWords};
  const State state = frame.state();
  assert(frame.d[FPeers] == 1 && frame.d[FRank] == 0);
  assert(state.epoch == frame.group().epoch && *state.epoch < LLONG_MAX);
  recent::advance<256>(frame.group(), *state.epoch + 1, storage);
}
} // namespace sglang::flowkv_host

namespace sglang {
void unified_local_register_unbound(TensorView rows, TensorView frames, bool prefer_gpu) {
  TVM_FFI_ICHECK(rows.ndim() == 1 && (rows.dtype().bits == 32 || rows.dtype().bits == 64));
  TVM_FFI_ICHECK(frames.ndim() == 2 && frames.dtype().bits == 64 &&
                frames.shape()[1] == flowkv_host::kFrameWords);
  const auto stream = host::LaunchKernel::resolve_device(frames.device());
  const int count = int(rows.shape()[0]), groups = int(frames.shape()[0]);
  if (count && groups)
    flowkv_host::register_unbound_local<<<dim3(std::min(128, (count + 255) / 256), groups), 256, 0, stream>>>(
      rows.data_ptr(), rows.dtype().bits / 8, count,
      static_cast<const int64_t*>(frames.data_ptr()), prefer_gpu);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void unified_local_expire_groups(TensorView frames) {
  TVM_FFI_ICHECK(frames.ndim() == 2 && frames.dtype().bits == 64 &&
                frames.shape()[1] == flowkv_host::kFrameWords);
  const auto stream = host::LaunchKernel::resolve_device(frames.device());
  const int groups = int(frames.shape()[0]);
  if (groups)
    flowkv_host::expire_groups_parallel<<<groups, 256, 0, stream>>>(
        static_cast<const int64_t*>(frames.data_ptr()));
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void unified_local_register_receive(TensorView rows, TensorView frames, bool prefer_gpu) {
  TVM_FFI_ICHECK(rows.ndim() == 1 && (rows.dtype().bits == 32 || rows.dtype().bits == 64));
  TVM_FFI_ICHECK(frames.ndim() == 2 && frames.dtype().bits == 64 && frames.shape()[1] == flowkv_host::kFrameWords);
  const auto stream = host::LaunchKernel::resolve_device(frames.device());
  const int count = int(rows.shape()[0]), groups = int(frames.shape()[0]);
  if (count && groups)
    flowkv_host::register_receive_ordered<<<dim3((count + 255) / 256, groups), 256, 0, stream>>>(
        rows.data_ptr(), rows.dtype().bits / 8, count, static_cast<const int64_t*>(frames.data_ptr()), prefer_gpu);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void unified_local_receive_locations(TensorView rows, TensorView frames, TensorView locations, bool acquire) {
  TVM_FFI_ICHECK(rows.ndim() == 1 && (rows.dtype().bits == 32 || rows.dtype().bits == 64));
  TVM_FFI_ICHECK(frames.ndim() == 2 && frames.dtype().bits == 64 && frames.shape()[1] == flowkv_host::kFrameWords);
  TVM_FFI_ICHECK(locations.ndim() == 2 && locations.dtype().bits == 64 && locations.shape()[0] == frames.shape()[0] && locations.shape()[1] == rows.shape()[0]);
  const auto stream = host::LaunchKernel::resolve_device(frames.device());
  const int count = int(rows.shape()[0]), groups = int(frames.shape()[0]);
  if (count && groups) {
    const dim3 grid(std::min(128, (count + 255) / 256), groups);
    if (acquire)
      flowkv_host::acquire_receive_locations<<<grid, 256, 0, stream>>>(rows.data_ptr(), rows.dtype().bits / 8, count,
          static_cast<const int64_t*>(frames.data_ptr()), static_cast<int64_t*>(locations.data_ptr()));
    else
      flowkv_host::release_receive_locations<<<grid, 256, 0, stream>>>(rows.data_ptr(), rows.dtype().bits / 8, count,
          static_cast<const int64_t*>(frames.data_ptr()), static_cast<const int64_t*>(locations.data_ptr()));
  }
  host::RuntimeDeviceCheck(cudaGetLastError());
}

void unified_local_rebind_cpu_receive(TensorView rows, TensorView frames, TensorView locations) {
  TVM_FFI_ICHECK(rows.ndim() == 1 && (rows.dtype().bits == 32 || rows.dtype().bits == 64));
  TVM_FFI_ICHECK(frames.ndim() == 2 && frames.dtype().bits == 64 &&
                frames.shape()[1] == flowkv_host::kFrameWords);
  TVM_FFI_ICHECK(locations.ndim() == 2 && locations.dtype().bits == 64 &&
                locations.shape()[0] == frames.shape()[0] && locations.shape()[1] == rows.shape()[0]);
  const auto stream = host::LaunchKernel::resolve_device(frames.device());
  const int count = int(rows.shape()[0]), groups = int(frames.shape()[0]);
  if (count && groups)
    flowkv_host::rebind_cpu_receive<<<dim3(std::min(128, (count + 255) / 256), groups), 256, 0, stream>>>(
        rows.data_ptr(), rows.dtype().bits / 8, count,
        static_cast<const int64_t*>(frames.data_ptr()), static_cast<const int64_t*>(locations.data_ptr()));
  host::RuntimeDeviceCheck(cudaGetLastError());
}
} // namespace sglang
