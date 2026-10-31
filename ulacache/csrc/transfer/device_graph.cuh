// Device graph launch is confined to this module.
#pragma once
#include "csrc/cache/residency_finish.cuh"

namespace sglang::flowkv_host {
// Only captured execution uses this ordinary (non-PDL) kernel, matching the
// already-validated device-graph launch mode. Eager plans use host launch.
__global__ void launch_planned_group(State s, Workspace w, cudaGraphExec_t cold, int64_t* ready) {
  if (w.counters[PlanCount] == 0) return;
  assert(cudaGraphLaunch(cold, cudaStreamGraphFireAndForget) == cudaSuccess);
  const uint64_t started = clock64();
  while (cuda::atomic_ref<int64_t, cuda::thread_scope_system>(*ready)
             .load(cuda::memory_order_acquire) < *s.epoch) {
    assert(clock64() - started < 100000000000ULL);
    __nanosleep(64);
  }
}
} // namespace sglang::flowkv_host

namespace sglang {
using tvm::ffi::TensorView;
#define HOST_STATE_ARGS \
 TensorView directory, TensorView gpu_tags, TensorView host_tags, TensorView access_time, \
 TensorView leases, TensorView epoch, TensorView gpu_free, \
 TensorView host_free, TensorView free_counts
#define HOST_WORK_ARGS \
 TensorView new_rows, TensorView promote_rows, TensorView plan, TensorView counters, \
 TensorView histogram, TensorView selection_prefix, TensorView candidate_slots, TensorView leased_slots
#define HOST_STATE \
 flowkv_host::State{static_cast<int64_t*>(directory.data_ptr()), static_cast<int*>(gpu_tags.data_ptr()), \
 static_cast<int*>(host_tags.data_ptr()), static_cast<int64_t*>(access_time.data_ptr()), \
 static_cast<int*>(leases.data_ptr()), \
 static_cast<int64_t*>(epoch.data_ptr()), static_cast<int*>(gpu_free.data_ptr()), \
 static_cast<int*>(host_free.data_ptr()), static_cast<int*>(free_counts.data_ptr()), \
 int(directory.shape()[0]), int(gpu_tags.shape()[0]), int(host_tags.shape()[0])}
#define HOST_WORK \
 flowkv_host::Workspace{static_cast<int*>(new_rows.data_ptr()), static_cast<int*>(promote_rows.data_ptr()), \
 static_cast<int*>(plan.data_ptr()), static_cast<int*>(counters.data_ptr()), \
 static_cast<int*>(histogram.data_ptr()), static_cast<int64_t*>(selection_prefix.data_ptr()), \
 static_cast<int*>(candidate_slots.data_ptr()), static_cast<int*>(leased_slots.data_ptr()), int(new_rows.shape()[0])}

int64_t flowkv_device_graph_create(TensorView host_cache, TensorView gpu_cache, TensorView ready,
    HOST_STATE_ARGS, HOST_WORK_ARGS) {
  const auto stream = host::LaunchKernel::resolve_device(gpu_cache.device());
  const auto s = HOST_STATE;
  const auto w = HOST_WORK;
  const int work_blocks = std::min(128, (w.capacity + 255) / 256);
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  host::RuntimeDeviceCheck(cudaGraphCreate(&graph, 0));
  cudaGraphNode_t previous{};
#if UDCACHE_NAIVE_LRU
  const int select_blocks = std::min(128, (s.gpu_slots + 255) / 256);
  for (int shift = 56; shift >= 0; shift -= 8) {
    previous = flowkv_host::add_kernel(graph, previous, dim3(select_blocks), dim3(256),
        flowkv_host::naive_histogram, s, w, false, shift);
    previous = flowkv_host::add_kernel(graph, previous, dim3(1), dim3(256),
        flowkv_host::naive_select_prefix, w, false, shift, s.epoch);
  }
  for (int shift = flowkv_host::first_physical_shift(s.gpu_slots); shift >= 0; shift -= 8) {
    previous = flowkv_host::add_kernel(graph, previous, dim3(select_blocks), dim3(256),
        flowkv_host::naive_histogram, s, w, true, shift);
    previous = flowkv_host::add_kernel(graph, previous, dim3(1), dim3(256),
        flowkv_host::naive_select_prefix, w, true, shift, s.epoch);
  }
  previous = flowkv_host::add_kernel(graph, previous, dim3(select_blocks), dim3(256),
      flowkv_host::naive_collect_victims, s, w);
  previous = flowkv_host::add_kernel(graph, previous, dim3(work_blocks), dim3(256),
      flowkv_host::naive_assign_child, s, w);
#endif
  previous = flowkv_host::add_kernel(graph, previous, dim3(UDCACHE_HOST_EXCHANGE_BLOCKS),
      dim3(UDCACHE_HOST_EXCHANGE_THREADS), flowkv_host::exchange,
      static_cast<char*>(host_cache.data_ptr()), static_cast<char*>(gpu_cache.data_ptr()),
      w.plan, w.counters, w.capacity, int64_t(gpu_cache.shape()[2]),
      int64_t(host_cache.strides()[1]), int64_t(gpu_cache.strides()[1]), int(gpu_cache.shape()[0]),
      int64_t(host_cache.strides()[0]), int64_t(gpu_cache.strides()[0]));
  previous = flowkv_host::add_kernel(graph, previous, dim3(work_blocks), dim3(256), flowkv_host::commit, s, w);
  flowkv_host::add_kernel(graph, previous, dim3(1), dim3(1), flowkv_host::publish_group,
      s.epoch, static_cast<int64_t*>(ready.data_ptr()));
  host::RuntimeDeviceCheck(cudaGraphInstantiateWithFlags(&executable, graph, cudaGraphInstantiateFlagDeviceLaunch));
  host::RuntimeDeviceCheck(cudaGraphUpload(executable, stream));
  host::RuntimeDeviceCheck(cudaStreamSynchronize(stream)); // Startup only.
  host::RuntimeDeviceCheck(cudaGraphDestroy(graph));
  return reinterpret_cast<int64_t>(executable);
}

inline void flowkv_device_graph_launch(cudaStream_t stream, flowkv_host::State s,
    flowkv_host::Workspace w, int64_t executable, TensorView ready) {
  if (flowkv_host::capture_info(stream).graph) {
    flowkv_host::launch_planned_group<<<1, 1, 0, stream>>>(s, w,
        reinterpret_cast<cudaGraphExec_t>(executable), static_cast<int64_t*>(ready.data_ptr()));
    host::RuntimeDeviceCheck(cudaGetLastError());
  } else {
    host::RuntimeDeviceCheck(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(executable), stream));
  }
}

void flowkv_device_graph_finish(TensorView gpu_cache, int64_t executable, TensorView ready,
    TensorView ready_peers, int64_t ready_offset, HOST_STATE_ARGS, HOST_WORK_ARGS) {
  const auto stream = host::LaunchKernel::resolve_device(gpu_cache.device());
  // Complete metadata before the original payload/commit/publication child.
#if UDCACHE_NAIVE_LRU
  flowkv_host::naive_reserve_child<<<1, 1, 0, stream>>>(HOST_STATE, HOST_WORK, 0);
#else
  flowkv_host::metadata_finish<<<1, 256, 0, stream>>>(HOST_STATE, HOST_WORK);
#endif
  host::RuntimeDeviceCheck(cudaGetLastError());
  flowkv_device_graph_launch(stream, HOST_STATE, HOST_WORK, executable, ready);
}

void flowkv_device_graph_finish_prepared(TensorView gpu_cache, int64_t executable,
    TensorView ready, TensorView ready_peers, int64_t ready_offset,
    HOST_STATE_ARGS, HOST_WORK_ARGS) {
  const auto stream = host::LaunchKernel::resolve_device(gpu_cache.device());
  // Low-query TP's last producer CTA already executed metadata_finish_body.
  // Retain the same payload/commit/publication child and capture semantics.
  flowkv_device_graph_launch(stream, HOST_STATE, HOST_WORK, executable, ready);
}

void flowkv_device_graph_finish_parallel(TensorView gpu_cache, int64_t executable,
    TensorView ready, TensorView ready_peers, int64_t ready_offset, int64_t queries,
    HOST_STATE_ARGS, HOST_WORK_ARGS) {
  const auto stream = host::LaunchKernel::resolve_device(gpu_cache.device());
#if UDCACHE_PARALLEL_VICTIMS
  constexpr bool parallel_victims = true;
#else
  constexpr bool parallel_victims = false;
#endif
  // The TP miss grid has already completed all owners' pin producers. Reuse
  // DPA's finish without another begin, epoch advance, or plan/incoming alias.
#if UDCACHE_NAIVE_LRU
  flowkv_host::naive_reserve_child<<<1, 1, 0, stream>>>(HOST_STATE, HOST_WORK, 1);
  host::RuntimeDeviceCheck(cudaGetLastError());
#else
  flowkv_host_parallel::finish_metadata(stream, HOST_STATE, HOST_WORK,
      int(queries), parallel_victims,
      flowkv_host_parallel::PlanDiagnostic{
          static_cast<const uint64_t*>(ready_peers.data_ptr()),
          static_cast<const int64_t*>(ready.data_ptr()), int(ready_peers.shape()[0]),
          int(ready_offset), -1, int(ready_offset / 2)});
#endif
  flowkv_device_graph_launch(stream, HOST_STATE, HOST_WORK, executable, ready);
}

void flowkv_device_graph_destroy(TensorView gpu_cache, int64_t executable) {
  const auto stream = host::LaunchKernel::resolve_device(gpu_cache.device());
  host::RuntimeDeviceCheck(cudaStreamSynchronize(stream));
  host::RuntimeDeviceCheck(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(executable)));
}
#undef HOST_STATE_ARGS
#undef HOST_WORK_ARGS
#undef HOST_STATE
#undef HOST_WORK
} // namespace sglang
