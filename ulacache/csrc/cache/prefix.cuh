// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "csrc/cache/residency_fused.cuh"

namespace sglang::flowkv_host {
// Native eager prefill supplies unique current identities. A CTA batches a
// small set of rows: one reservation per free list and one system fence per
// copying thread, rather than contending on both counters for every row.
template<int RowBytes,int RowsPerBlock=8>
__global__ void offload_prefill_rows(const void* rows,int bytes,int count,
    const int64_t* frames,unsigned char* gpu,const uint64_t* host_ptrs,
    int first_layer,int layers,int64_t layer_stride) {
  const Frame frame{frames};const State state=frame.state();
  __shared__ int ids[RowsPerBlock],gs[RowsPerBlock],hs[RowsPerBlock];
  __shared__ int host_top,gpu_base,live_count;
  constexpr int words=RowBytes/8;
  for(int base=blockIdx.x*RowsPerBlock;base<count;base+=gridDim.x*RowsPerBlock) {
    if(threadIdx.x<32) {
      const int lane=threadIdx.x;int row=-1,slot=-1;
      if(lane<RowsPerBlock && base+lane<count) {
        row=int(read_row(rows,base+lane,bytes));
        if(row>0) {
          assert(row<state.logical_slots && state.directory[row]!=-1);
          const auto at=state.directory[row];
          if(at>=0) {
            assert(at<state.gpu_slots && state.gpu_tags[at]==row && state.leases[at]==0);
            slot=int(at);
          }
        }
      }
      const unsigned mask=__ballot_sync(0xffffffffu,slot>=0);
      const int n=__popc(mask);
      if(lane==0) {
        live_count=n;
        if(n) {
          host_top=atomicSub(state.free_counts+1,n);
          assert(host_top>=n);
        }
      }
      __syncwarp();
      if(lane<RowsPerBlock) {
        ids[lane]=row;gs[lane]=slot;hs[lane]=-1;
        if(slot>=0) {
          const int rank=__popc(mask & ((1u<<lane)-1));
          hs[lane]=state.host_free[host_top-1-rank];
        }
      }
    }
    __syncthreads();
    if(live_count) {
      auto* host=reinterpret_cast<unsigned char*>(host_ptrs[0]);
      for(int layer=first_layer;layer<first_layer+layers;++layer) {
        for(int at=threadIdx.x;at<RowsPerBlock*words;at+=blockDim.x) {
          const int row=at/words,word=at%words;
          if(gs[row]>=0) {
            auto* src=reinterpret_cast<const uint64_t*>(gpu+int64_t(layer)*layer_stride+int64_t(gs[row])*RowBytes);
            auto* dst=reinterpret_cast<uint64_t*>(host+(int64_t(layer)*state.host_slots+hs[row])*RowBytes);
            dst[word]=src[word];
          }
        }
      }
      // Every writer fences its own stores before publishing CPU ownership.
      __threadfence_system();
      __syncthreads();
      if(threadIdx.x==0) {
        gpu_base=atomicAdd(state.free_counts,live_count);
        assert(gpu_base+live_count<=state.gpu_slots);
      }
      __syncthreads();
      if(threadIdx.x<32) {
        const int lane=threadIdx.x;
        const bool live=lane<RowsPerBlock && gs[lane]>=0;
        const unsigned mask=__ballot_sync(0xffffffffu,live);
        if(live) {
          const int row=ids[lane],slot=gs[lane],host_slot=hs[lane];
          state.host_tags[host_slot]=row;
          recent::mark_slot(frame.group(),slot,ring::Operation::Detach,*state.epoch);
          state.gpu_tags[slot]=-1;
          cuda::atomic_ref<int64_t,cuda::thread_scope_system>(state.directory[row])
              .store(-int64_t(host_slot)-2,cuda::memory_order_release);
          const int rank=__popc(mask & ((1u<<lane)-1));
          state.gpu_free[gpu_base+rank]=slot;
        }
      }
    }
    // This kernel never consumes GPU free entries; subsequent kernels on the
    // same stream see the fully written batch, after all producers retire.
    __syncthreads();
  }
}
} // namespace sglang::flowkv_host

namespace sglang {
void unified_local_offload_prefill(TensorView rows,TensorView frames,TensorView gpu,
    TensorView host_ptrs,int64_t first_layer,int64_t layers) {
  TVM_FFI_ICHECK(rows.ndim()==1 && (rows.dtype().bits==32 || rows.dtype().bits==64));
  TVM_FFI_ICHECK(frames.ndim()==2 && frames.size(0)==1 && frames.size(1)==flowkv_host::kFrameWords);
  TVM_FFI_ICHECK(gpu.ndim()==3 && gpu.dtype().bits==8 && gpu.stride(2)==1 && gpu.stride(1)==gpu.size(2));
  TVM_FFI_ICHECK((gpu.size(2)==576 || gpu.size(2)==656) && first_layer>=0 && layers>0 && first_layer+layers<=gpu.size(0));
  TVM_FFI_ICHECK(host_ptrs.dtype().bits==64 && host_ptrs.numel()>=1);
  const auto stream=host::LaunchKernel::resolve_device(gpu.device());
  const int blocks=int(std::min<int64_t>(1024,(rows.numel()+7)/8));
  if(rows.numel()) {
    if(gpu.size(2)==656)flowkv_host::offload_prefill_rows<656><<<blocks,128,0,stream>>>(
        rows.data_ptr(),rows.dtype().bits/8,int(rows.numel()),static_cast<const int64_t*>(frames.data_ptr()),
        static_cast<unsigned char*>(gpu.data_ptr()),static_cast<const uint64_t*>(host_ptrs.data_ptr()),
        int(first_layer),int(layers),gpu.stride(0));
    else flowkv_host::offload_prefill_rows<576><<<blocks,128,0,stream>>>(
        rows.data_ptr(),rows.dtype().bits/8,int(rows.numel()),static_cast<const int64_t*>(frames.data_ptr()),
        static_cast<unsigned char*>(gpu.data_ptr()),static_cast<const uint64_t*>(host_ptrs.data_ptr()),
        int(first_layer),int(layers),gpu.stride(0));
  }
  host::RuntimeDeviceCheck(cudaGetLastError());
}
} // namespace sglang
