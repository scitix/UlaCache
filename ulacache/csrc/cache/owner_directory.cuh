// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <assert.h>

// descriptor: directory IPC table, local mapped host base, readiness IPC
// table, logical/GPU/host rows per owner, group, local group epoch pointer
// and the anchor's local physical-row cache for this group's actual misses.
// Only an owner writes its registered CPU memory. Peer readers use GPU VA.
enum HostRecordField {
  HostDirectoryPeers, HostLocalCpu, HostReadyPeers, HostLogicalRows, HostGpuRows, HostCpuRows, HostGroup, HostEpoch, HostAccessTimePeers, HostMissPhysical, HostRingGroup, HostRecordFields
};
__device__ __forceinline__ int64_t record_host_location(int gid, const int64_t* host) {
  const auto* directories = reinterpret_cast<const uint64_t*>(host[HostDirectoryPeers]);
  const auto* directory = reinterpret_cast<const int64_t*>(directories[record_owner(gid)]);
  return cuda::atomic_ref<int64_t, cuda::thread_scope_system>(
      const_cast<int64_t*>(directory)[host[HostGroup] * host[HostLogicalRows] + record_local(gid)])
      .load(cuda::memory_order_acquire);
}

__device__ __forceinline__ uint8_t* record_host_address(
    const uint64_t* peers, int gid, long layer_offset,
    const int64_t* host, int rank, bool writer) {
  const int owner = record_owner(gid);
  int64_t location = record_host_location(gid, host);
  if (!writer && location <= -2) {
    const uint64_t started = clock64();
    do {
      assert(clock64() - started < 100000000000ULL);
      __nanosleep(64);
      location = record_host_location(gid, host);
    } while (location <= -2);
  }
  if (location >= 0) {
    assert(location < host[HostGpuRows]);
    return reinterpret_cast<uint8_t*>(peers[owner] + layer_offset)
           + location * FLOW_ROW_BYTES;
  }
  assert(writer && owner == rank && location <= -2);
  const int64_t cpu_slot = -location - 2;
  assert(cpu_slot < host[HostCpuRows]);
  const int64_t layer = layer_offset / (host[HostGpuRows] * FLOW_ROW_BYTES);
  return reinterpret_cast<uint8_t*>(host[HostLocalCpu])
         + (layer * host[HostCpuRows] + cpu_slot) * FLOW_ROW_BYTES;
}

// A native writer CTA owns one query, so every participating lane has the
// same gid and publication predicate. Keep the original writer address path
// (including mapped CPU storage), but acquire its directory once per warp.
__device__ __forceinline__ uint8_t* record_host_writer_address_warp(
    const uint64_t* peers, int gid, long layer_offset,
    const int64_t* host, int rank) {
  uint64_t address = 0;
  if ((threadIdx.x & 31) == 0)
    address = reinterpret_cast<uint64_t>(
        record_host_address(peers, gid, layer_offset, host, rank, true));
  const uint32_t low = __shfl_sync(0xffffffffu, uint32_t(address), 0);
  const uint32_t high = __shfl_sync(0xffffffffu, uint32_t(address >> 32), 0);
  // A register shuffle alone is not an acquire/release memory barrier.
  __syncwarp(0xffffffffu);
  return reinterpret_cast<uint8_t*>((uint64_t(high) << 32) | low);
}

__device__ __forceinline__ void record_host_touch(int gid, const int64_t* host, int64_t physical = -1) {
  const auto* peers = reinterpret_cast<const uint64_t*>(host[HostAccessTimePeers]);
  auto* time = reinterpret_cast<int64_t*>(peers[record_owner(gid)])
      + host[HostGroup] * host[HostLogicalRows] + record_local(gid);
  cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*time)
      .store(*reinterpret_cast<const int64_t*>(host[HostEpoch]), cuda::memory_order_relaxed);
}

// Only the local owner writer mutates its ring. Peer readers retain the
// original SYS canonical-time touch and never access owner-local bitmaps.
__device__ __forceinline__ void record_host_writer_touch(int gid, const int64_t* host, int rank) {
  assert(record_owner(gid) == rank);
  const int64_t location = record_host_location(gid, host);
  if (location >= 0) {
    const auto& g = *reinterpret_cast<const flowkv_bitmap_ring::Group*>(host[HostRingGroup]);
    flowkv_ring_presence_recent::mark_slot(g, int(location), flowkv_bitmap_ring::Operation::Access,
        *reinterpret_cast<const int64_t*>(host[HostEpoch]));
  }
  record_host_touch(gid, host);
}

__device__ __forceinline__ void record_host_wait(
    const int64_t* host, int owner, bool readers) {
  const auto* peers = reinterpret_cast<const uint64_t*>(host[HostReadyPeers]);
  auto* ready = reinterpret_cast<int64_t*>(peers[owner]) + 2 * host[HostGroup] + int(readers);
  const int64_t epoch = *reinterpret_cast<const int64_t*>(host[HostEpoch]);
  const uint64_t started = clock64();
  while (cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*ready)
             .load(cuda::memory_order_acquire) < epoch) {
    // A missing producer is a real execution error, never a silent fallback.
    assert(clock64() - started < 100000000000ULL);
    __nanosleep(64);
  }
}

__device__ __forceinline__ void record_host_acquire_all(const int64_t* host) {
  if (threadIdx.x == 0)
    for (int owner = 0; owner < FLOW_RECORD_PEERS; ++owner)
      record_host_wait(host, owner, false);
  __syncthreads();
}

__global__ void record_host_publish_kernel(const int64_t* host, int64_t* ready) {
  cuda::atomic_ref<int64_t, cuda::thread_scope_system>(ready[2 * host[HostGroup]])
      .store(*reinterpret_cast<const int64_t*>(host[HostEpoch]), cuda::memory_order_release);
}

__global__ void record_host_read_done_kernel(const int64_t* host, int64_t* ready) {
  cuda::atomic_ref<int64_t, cuda::thread_scope_system>(ready[2 * host[HostGroup] + 1])
      .store(*reinterpret_cast<const int64_t*>(host[HostEpoch]), cuda::memory_order_release);
  for (int owner = 0; owner < FLOW_RECORD_PEERS; ++owner)
    record_host_wait(host, owner, true);
}
