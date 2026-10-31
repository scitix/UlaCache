// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <climits>
#include <sgl_kernel/runtime.cuh>
// Supplied explicitly by the baseline adapter, with its source SHA recorded.
#include <kvcacheio/hisparse_spec.cuh>

namespace udcache::telemetry {
using tvm::ffi::TensorView;

template<int Hot>
__global__ void hisparse_probe_observe(const int* selected, const int64_t* requests,
    const int* seq_lens, const int64_t* positions, const int* buffer_tokens,
    const int* buffer_locs, const int64_t* cache_index, const int* scratch,
    const int64_t* host_locs, const int* real_requests, const int* enabled,
    unsigned long long* keys, unsigned* flags, unsigned long long* counts,
    int batches, int width, int topk, int token_stride, int hash_size,
    int request_capacity, int host_stride, int page_size, int table_size) {
  if (!*enabled) return;
  const int64_t item=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (item>=int64_t(batches)*width*topk) return;
  const int query=int(item/topk), batch=query/width;
  if (batch>=*real_requests) return;
  const int token=selected[item];
  if (token<0 || token>=seq_lens[query]) return;
  for (int j=0;j<width;++j)
    if (int64_t(token)==positions[batch*width+j]) return;
  const int64_t rid=requests[batch];
  if (rid<0 || rid>=request_capacity || token>=host_stride) {
    atomicAdd(counts+4,1ULL); return;
  }
  const int* tokens=buffer_tokens+rid*token_stride;
  const int* locs=buffer_locs+rid*token_stride;
  int extra=-1;
  bool hit=sglang::try_get_extra_page_device_loc<Hot>(
      token,seq_lens[query],tokens,locs,page_size,&extra);
  if (!hit) {
    const int64_t* primary=cache_index+rid*2*hash_size;
    const int64_t* secondary=primary+hash_size;
    const bool degraded=uint32_t(scratch[3*request_capacity+rid]) & sglang::HASH_DEGRADED_FLAG;
    hit=sglang::hot_cache_lookup<Hot>(primary,secondary,hash_size,token,tokens,degraded)>=0;
  }
  const bool mapped=host_locs[rid*host_stride+token]>=0;
  const uint64_t key=(uint64_t(rid+1)<<32)|uint32_t(token);
  uint64_t mixed=key;
  mixed^=mixed>>30; mixed*=0xbf58476d1ce4e5b9ULL;
  mixed^=mixed>>27; mixed*=0x94d049bb133111ebULL; mixed^=mixed>>31;
  unsigned at=unsigned(mixed)&unsigned(table_size-1);
  for(int probe=0;probe<table_size;++probe) {
    const uint64_t previous=atomicCAS(keys+at,ULLONG_MAX,key);
    if(previous==ULLONG_MAX || previous==key) {
      // A repeated address can use an extra-page hit in only some queries.
      // Reduce after all occurrences, rather than letting the first lane win.
      atomicOr(flags+at,1u|(hit?2u:0u)|(mapped?4u:0u));
      return;
    }
    at=(at+1)&unsigned(table_size-1);
  }
  atomicAdd(counts+4,1ULL);
}

__global__ void hisparse_probe_reduce(const unsigned long long* keys,
    const unsigned* flags,unsigned long long* counts,const int* enabled,int size) {
  if(!*enabled) return;
  const int at=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(at>=size || keys[at]==ULLONG_MAX) return;
  const unsigned f=flags[at];
  atomicAdd(counts,1ULL);
  atomicAdd(counts+((f&2u)?1:(f&4u)?2:3),1ULL);
}

inline void hisparse_metrics_observe(TensorView selected,TensorView requests,
    TensorView seq_lens,TensorView positions,TensorView buffer_tokens,
    TensorView buffer_locs,TensorView cache_index,TensorView scratch,
    TensorView host_locs,TensorView real_requests,TensorView enabled,
    TensorView keys,TensorView flags,TensorView counts,int64_t hot,int64_t page_size) {
  TVM_FFI_ICHECK(selected.ndim()==3 && selected.dtype().bits==32 &&
      selected.stride(2)==1 && selected.stride(1)==selected.size(2) &&
      selected.stride(0)==selected.size(1)*selected.size(2));
  TVM_FFI_ICHECK(requests.dtype().bits==64 && requests.numel()==selected.size(0));
  TVM_FFI_ICHECK(seq_lens.dtype().bits==32 && seq_lens.numel()==selected.size(0)*selected.size(1));
  TVM_FFI_ICHECK(positions.dtype().bits==64 && positions.numel()>=seq_lens.numel());
  TVM_FFI_ICHECK(buffer_tokens.ndim()==2 && buffer_tokens.dtype().bits==32 &&
      buffer_tokens.stride(1)==1 && buffer_tokens.stride(0)==buffer_tokens.size(1));
  TVM_FFI_ICHECK(buffer_locs.ndim()==2 && buffer_locs.size(0)==buffer_tokens.size(0) &&
      buffer_locs.size(1)==buffer_tokens.size(1) && buffer_locs.dtype().bits==32 &&
      buffer_locs.stride(1)==1 && buffer_locs.stride(0)==buffer_locs.size(1));
  TVM_FFI_ICHECK(cache_index.ndim()==3 && cache_index.size(0)==buffer_tokens.size(0) &&
      cache_index.size(1)==2 && cache_index.dtype().bits==64);
  TVM_FFI_ICHECK(scratch.dtype().bits==32 && scratch.numel()>=4*buffer_tokens.size(0));
  TVM_FFI_ICHECK(host_locs.ndim()==2 && host_locs.dtype().bits==64 &&
      host_locs.size(0)==buffer_tokens.size(0));
  TVM_FFI_ICHECK(keys.dtype().bits==64 && flags.dtype().bits==32 &&
      keys.numel()==flags.numel() && keys.numel()>=2*selected.numel() &&
      (keys.numel()&(keys.numel()-1))==0);
  TVM_FFI_ICHECK(counts.dtype().bits==64 && counts.numel()==8);
  TVM_FFI_ICHECK(real_requests.dtype().bits==32 && real_requests.numel()==1 &&
      enabled.dtype().bits==32 && enabled.numel()==1);
  TVM_FFI_ICHECK(page_size>0 && buffer_tokens.size(1)>=hot+page_size);
  const auto stream=sglang::host::LaunchKernel::resolve_device(selected.device());
#define HS_PROBE_CASE(H) case H: hisparse_probe_observe<H><<<(selected.numel()+255)/256,256,0,stream>>>( \
    static_cast<const int*>(selected.data_ptr()),static_cast<const int64_t*>(requests.data_ptr()), \
    static_cast<const int*>(seq_lens.data_ptr()),static_cast<const int64_t*>(positions.data_ptr()), \
    static_cast<const int*>(buffer_tokens.data_ptr()),static_cast<const int*>(buffer_locs.data_ptr()), \
    static_cast<const int64_t*>(cache_index.data_ptr()),static_cast<const int*>(scratch.data_ptr()), \
    static_cast<const int64_t*>(host_locs.data_ptr()),static_cast<const int*>(real_requests.data_ptr()), \
    static_cast<const int*>(enabled.data_ptr()),static_cast<unsigned long long*>(keys.data_ptr()), \
    static_cast<unsigned*>(flags.data_ptr()),static_cast<unsigned long long*>(counts.data_ptr()), \
    int(selected.size(0)),int(selected.size(1)),int(selected.size(2)),int(buffer_tokens.size(1)), \
    int(cache_index.size(2)),int(buffer_tokens.size(0)),int(host_locs.size(1)),int(page_size),int(keys.numel())); break
  switch(hot) {
    HS_PROBE_CASE(4096); HS_PROBE_CASE(6912); HS_PROBE_CASE(8192);
    HS_PROBE_CASE(15552); HS_PROBE_CASE(16384); HS_PROBE_CASE(32768);
    default: TVM_FFI_ICHECK(false)<<"Unsupported native HiSparse hot buffer size";
  }
#undef HS_PROBE_CASE
  hisparse_probe_reduce<<<(keys.numel()+255)/256,256,0,stream>>>(
      static_cast<const unsigned long long*>(keys.data_ptr()),static_cast<const unsigned*>(flags.data_ptr()),
      static_cast<unsigned long long*>(counts.data_ptr()),static_cast<const int*>(enabled.data_ptr()),int(keys.numel()));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

__global__ void hisparse_probe_miss_bytes(const int* misses,const int* real,
    const int* enabled,unsigned long long* counts,int batches,int layers,int bytes) {
  if(!*enabled) return;
  const int bid=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(bid>=batches || bid>=*real) return;
  if(misses[bid]<0) { atomicAdd(counts+4,1ULL); return; }
  atomicAdd(counts+5,uint64_t(misses[bid])*layers*bytes);
}

inline void hisparse_metrics_miss_bytes(TensorView misses,TensorView real,
    TensorView enabled,TensorView counts,int64_t layers,int64_t bytes) {
  TVM_FFI_ICHECK(misses.dtype().bits==32 && misses.ndim()==1 && layers>0 && bytes>0);
  const auto stream=sglang::host::LaunchKernel::resolve_device(misses.device());
  hisparse_probe_miss_bytes<<<(misses.numel()+255)/256,256,0,stream>>>(
      static_cast<const int*>(misses.data_ptr()),static_cast<const int*>(real.data_ptr()),
      static_cast<const int*>(enabled.data_ptr()),static_cast<unsigned long long*>(counts.data_ptr()),
      int(misses.numel()),int(layers),int(bytes));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}

__global__ void hisparse_probe_backup(const int* enabled,unsigned long long* counts,uint64_t bytes) {
  if(*enabled && !threadIdx.x) { atomicAdd(counts+6,bytes); atomicAdd(counts+7,bytes); }
}

inline void hisparse_metrics_backup(TensorView enabled,TensorView counts,int64_t bytes) {
  TVM_FFI_ICHECK(bytes>=0);
  const auto stream=sglang::host::LaunchKernel::resolve_device(counts.device());
  hisparse_probe_backup<<<1,32,0,stream>>>(static_cast<const int*>(enabled.data_ptr()),
      static_cast<unsigned long long*>(counts.data_ptr()),uint64_t(bytes));
  sglang::host::RuntimeDeviceCheck(cudaGetLastError());
}
} // namespace udcache::telemetry
