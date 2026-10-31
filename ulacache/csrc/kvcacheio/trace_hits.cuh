// SPDX-License-Identifier: Apache-2.0
#pragma once
// Read-only observer. Cache state is never written by these kernels.
#include "csrc/cache/cpu_residency.cuh"
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

namespace sglang::trace_hits {
constexpr int Columns=13,Threads=256;
enum Column {Raw=0,RawHot,RawCurrent,RawCurrentHot,Historical,HistoricalHot,
             Current,CurrentHot,Union,UnionHot,Invalid,CurrentSelected,CurrentSelectedHot};
struct Counts {int v[Columns];};
struct Sum {__device__ Counts operator()(const Counts& a,const Counts& b)const{
 Counts r{};
 #pragma unroll
 for(int i=0;i<Columns;++i)r.v[i]=a.v[i]+b.v[i];
 return r;
}};
__device__ void publish(Counts x,int64_t* stats){
 using Reduce=cub::BlockReduce<Counts,Threads>;
 __shared__ typename Reduce::TempStorage temporary;
 Counts total=Reduce(temporary).Reduce(x,Sum{});
 if(!threadIdx.x)for(int j=0;j<Columns;++j)
  atomicAdd(reinterpret_cast<unsigned long long*>(stats+j),static_cast<unsigned long long>(total.v[j]));
}
__device__ int writer_slot(int row,const int* hash,int capacity){
 unsigned h=(unsigned(row)*2654435761u)&(capacity-1);
 for(int j=0;j<capacity;++j){int v=hash[h];if(v==row)return int(h);if(v==-1)return -1;h=(h+1)&(capacity-1);}
 return -1;
}
__device__ bool hot(int row,const int64_t* directory,const int* tags,int G,Counts& counts){
 int64_t loc=directory[row];
 if(loc==-1){++counts.v[Invalid];return false;}
 if(loc<0)return false;
 if(loc>=G||tags[loc]!=row){++counts.v[Invalid];return false;}
 return true;
}
__global__ void begin(int* hash,int* selected,int capacity,int64_t* stats){
 for(int i=threadIdx.x;i<capacity;i+=blockDim.x){hash[i]=-1;selected[i]=0;}
 if(threadIdx.x<Columns)stats[threadIdx.x]=0;
}
__global__ void writers(const void* current,int bytes,const int* active,int Q,int L,int* hash,int capacity){
 for(int i=threadIdx.x;i<Q&&i<*active;i+=blockDim.x){
  int64_t row=flowkv_host::read_row(current,i,bytes);assert(row>0&&row<L);
  unsigned h=(unsigned(row)*2654435761u)&(capacity-1);
  while(true){int old=atomicCAS(hash+h,-1,int(row));if(old==-1||old==row)break;h=(h+1)&(capacity-1);}
 }
}
__global__ void merge(const void* selected,int sb,const void* current,int cb,const int* active,
 int Q,int K,const int64_t* directory,const int* tags,int L,int G,int* ids,
 const int* hash,int* flags,int capacity,int64_t* stats){
 Counts counts{};
 for(int i=blockIdx.x*Threads+threadIdx.x;i<Q*(K+1);i+=gridDim.x*Threads){
  bool raw=i<Q*K;int query=raw?i/K:i-Q*K;int uid=L;
  if(query<*active){
   int64_t row=flowkv_host::read_row(raw?selected:current,raw?i:query,raw?sb:cb);
   if(row>0&&row<L){uid=int(row);if(raw){
    ++counts.v[Raw];bool found=hot(uid,directory,tags,G,counts);counts.v[RawHot]+=int(found);
    int writer=writer_slot(uid,hash,capacity);
    if(writer>=0){++counts.v[RawCurrent];counts.v[RawCurrentHot]+=int(found);atomicExch(flags+writer,1);}
   }}else ++counts.v[Invalid];
  }
  ids[i]=uid;
 }
 publish(counts,stats);
}
__global__ void unique_stats(const int* unique,const int* count,int n,const int64_t* directory,
 const int* tags,int L,int G,const int* hash,const int* flags,int capacity,int64_t* stats){
 Counts counts{};
 for(int i=blockIdx.x*Threads+threadIdx.x;i<n&&i<*count;i+=gridDim.x*Threads){
  int uid=unique[i];if(uid==L)continue;assert(uid>0&&uid<L);
  bool found=hot(uid,directory,tags,G,counts);++counts.v[Union];counts.v[UnionHot]+=int(found);
  int writer=writer_slot(uid,hash,capacity);
  if(writer>=0){++counts.v[Current];counts.v[CurrentHot]+=int(found);
   if(flags[writer]){++counts.v[CurrentSelected];counts.v[CurrentSelectedHot]+=int(found);}
  }else{++counts.v[Historical];counts.v[HistoricalHot]+=int(found);}
 }
 publish(counts,stats);
}
inline int bits(int L){int n=0;do{++n;L>>=1;}while(L);return n;}
}
namespace sglang {
int64_t trace_hits_temporary_bytes(int64_t n,int64_t logical){
 size_t sort=0,unique=0;
 host::RuntimeDeviceCheck(cub::DeviceRadixSort::SortKeys(nullptr,sort,(int*)nullptr,(int*)nullptr,int(n),0,trace_hits::bits(int(logical))));
 host::RuntimeDeviceCheck(cub::DeviceSelect::Unique(nullptr,unique,(int*)nullptr,(int*)nullptr,(int*)nullptr,int(n)));
 return std::max(sort,unique);
}
void trace_hits_observe(TensorView selected,TensorView current,TensorView active,TensorView directory,TensorView gpu_tags,
 TensorView ids,TensorView sorted_ids,TensorView unique_ids,TensorView unique_count,TensorView current_hash,
 TensorView current_selected,TensorView stats,TensorView temporary){
 using namespace trace_hits;
 auto stream=host::LaunchKernel::resolve_device(directory.device());int Q=selected.shape()[0],K=selected.shape()[1],n=Q*(K+1);
 int L=directory.shape()[0],G=gpu_tags.shape()[0],capacity=current_hash.shape()[0];
 TVM_FFI_ICHECK(ids.shape()[0]==n&&stats.shape()[0]==Columns&&Q==current.shape()[0]);
 auto hash=static_cast<int*>(current_hash.data_ptr());auto flags=static_cast<int*>(current_selected.data_ptr());auto output=static_cast<int64_t*>(stats.data_ptr());
 begin<<<1,Threads,0,stream>>>(hash,flags,capacity,output);
 writers<<<1,Threads,0,stream>>>(current.data_ptr(),current.dtype().bits/8,static_cast<int*>(active.data_ptr()),Q,L,hash,capacity);
 int blocks=std::min(256,(n+Threads-1)/Threads);
 merge<<<blocks,Threads,0,stream>>>(selected.data_ptr(),selected.dtype().bits/8,current.data_ptr(),current.dtype().bits/8,static_cast<int*>(active.data_ptr()),Q,K,static_cast<int64_t*>(directory.data_ptr()),static_cast<int*>(gpu_tags.data_ptr()),L,G,static_cast<int*>(ids.data_ptr()),hash,flags,capacity,output);
 size_t bytes=temporary.shape()[0];host::RuntimeDeviceCheck(cub::DeviceRadixSort::SortKeys(temporary.data_ptr(),bytes,static_cast<int*>(ids.data_ptr()),static_cast<int*>(sorted_ids.data_ptr()),n,0,bits(L),stream));
 bytes=temporary.shape()[0];host::RuntimeDeviceCheck(cub::DeviceSelect::Unique(temporary.data_ptr(),bytes,static_cast<int*>(sorted_ids.data_ptr()),static_cast<int*>(unique_ids.data_ptr()),static_cast<int*>(unique_count.data_ptr()),n,stream));
 unique_stats<<<blocks,Threads,0,stream>>>(static_cast<int*>(unique_ids.data_ptr()),static_cast<int*>(unique_count.data_ptr()),n,static_cast<int64_t*>(directory.data_ptr()),static_cast<int*>(gpu_tags.data_ptr()),L,G,hash,flags,capacity,output);
 host::RuntimeDeviceCheck(cudaGetLastError());
}
}
