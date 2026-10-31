// SPDX-License-Identifier: Apache-2.0
#pragma once
// New plain baseline: sorted GPU union first; exactly one GPU thread owns each
// doubly linked hot-slot list. No PB, radix victim selector or parallel LRU work.
#include "csrc/cache/cpu_residency.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
namespace sglang::linked_list_lru {
using State=flowkv_host::State;using Work=flowkv_host::Workspace;
struct Scratch {int *ids,*sorted_ids,*unique_ids,*unique_count,*previous,*following,*ends;char* temporary;size_t temporary_bytes;};
inline int bits(int logical){int n=0;do{++n;logical>>=1;}while(logical);return n;}
__global__ void merge(const void* selected,int sb,const void* current,int cb,const int* active,int Q,int K,int L,int* ids){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<Q*(K+1);i+=gridDim.x*blockDim.x){
  const bool history=i<Q*K;const int q=history?i/K:i-Q*K;
  int64_t row=q<*active?flowkv_host::read_row(history?selected:current,history?i:q,history?sb:cb):L;
  assert(row>0&&row<=L);ids[i]=int(row);
 }
}
__device__ void move_to_mru(int slot,Scratch x){
 if(x.ends[1]==slot)return;
 const int left=x.previous[slot],right=x.following[slot];
 if(left>=0)x.following[left]=right;else x.ends[0]=right;
 if(right>=0)x.previous[right]=left;
 const int tail=x.ends[1];assert(tail>=0&&tail!=slot);
 x.following[tail]=slot;x.previous[slot]=tail;x.following[slot]=-1;x.ends[1]=slot;
}
__global__ void update(const void* current,int cb,const int* active,int Q,State s,Work w,Scratch x){
 assert(gridDim.x==1&&blockDim.x==1&&blockIdx.x==0&&threadIdx.x==0);
 assert(s.free_counts[0]>=0&&s.free_counts[0]<=s.gpu_slots&&*s.epoch<LLONG_MAX);
 const int64_t epoch=++*s.epoch;
 for(int j=0;j<8;++j)w.counters[j]=0;
 // Current writer stamps distinguish cold writes from historical CPU reads.
 for(int q=0;q<Q&&q<*active;++q){const int row=int(flowkv_host::read_row(current,q,cb));assert(row>0&&row<s.logical_slots&&s.directory[row]!=-1);s.access_time[row]=epoch;}
 int unique=*x.unique_count;if(unique&&x.unique_ids[unique-1]==s.logical_slots)--unique;
 assert(unique<=s.gpu_slots&&unique<=w.capacity);
 int misses=0;
 // Hits first: protect every required resident before selecting any victims.
 // Sorted UID order is the explicit within-transaction access/tie order.
 for(int j=0;j<unique;++j){
  const int row=x.unique_ids[j];const int64_t location=s.directory[row];assert(row>0&&row<s.logical_slots&&location!=-1);
  if(location>=0){assert(location<s.gpu_slots&&s.gpu_tags[location]==row&&s.leases[location]==0);move_to_mru(int(location),x);}
  else{assert(location<=-2);const bool writer=s.access_time[row]==epoch;w.promote_rows[misses++]=writer?-row-1:row;}
  s.access_time[row]=epoch;
 }
 const int free_used=min(misses,s.free_counts[0]);
 s.free_counts[0]-=free_used;
 w.counters[flowkv_host::PromoteCount]=misses;w.counters[flowkv_host::PlanCount]=misses;
 w.counters[flowkv_host::FreeUsed]=free_used;
 w.counters[flowkv_host::VictimCount]=misses-free_used;w.counters[flowkv_host::Candidates]=misses-free_used;
 for(int j=0;j<misses;++j){
  const int slot=j<free_used?s.gpu_free[s.free_counts[0]+j]:x.ends[0];
  assert(slot>=0&&s.leases[slot]==0);
  const int victim=s.gpu_tags[slot];
  if(j<free_used){assert(victim<0);}
  else{assert(victim>0&&s.access_time[victim]<epoch);w.candidates[j-free_used]=slot;}
  flowkv_host::assign_plan_item(s,w,j,slot);move_to_mru(slot,x);
 }
}
}
namespace sglang {
#define LL_STATE_ARGS TensorView directory,TensorView gpu_tags,TensorView host_tags,TensorView access_time,TensorView leases,TensorView epoch,TensorView gpu_free,TensorView host_free,TensorView free_counts
#define LL_WORK_ARGS TensorView new_rows,TensorView promote_rows,TensorView plan,TensorView counters,TensorView histogram,TensorView selection_prefix,TensorView candidate_slots,TensorView leased_slots
#define LL_SCRATCH_ARGS TensorView ids,TensorView sorted_ids,TensorView unique_ids,TensorView unique_count,TensorView previous,TensorView following,TensorView ends,TensorView temporary
#define LL_STATE flowkv_host::State{static_cast<int64_t*>(directory.data_ptr()),static_cast<int*>(gpu_tags.data_ptr()),static_cast<int*>(host_tags.data_ptr()),static_cast<int64_t*>(access_time.data_ptr()),static_cast<int*>(leases.data_ptr()),static_cast<int64_t*>(epoch.data_ptr()),static_cast<int*>(gpu_free.data_ptr()),static_cast<int*>(host_free.data_ptr()),static_cast<int*>(free_counts.data_ptr()),int(directory.shape()[0]),int(gpu_tags.shape()[0]),int(host_tags.shape()[0])}
#define LL_WORK flowkv_host::Workspace{static_cast<int*>(new_rows.data_ptr()),static_cast<int*>(promote_rows.data_ptr()),static_cast<int*>(plan.data_ptr()),static_cast<int*>(counters.data_ptr()),static_cast<int*>(histogram.data_ptr()),static_cast<int64_t*>(selection_prefix.data_ptr()),static_cast<int*>(candidate_slots.data_ptr()),static_cast<int*>(leased_slots.data_ptr()),int(new_rows.shape()[0])}
#define LL_SCRATCH linked_list_lru::Scratch{static_cast<int*>(ids.data_ptr()),static_cast<int*>(sorted_ids.data_ptr()),static_cast<int*>(unique_ids.data_ptr()),static_cast<int*>(unique_count.data_ptr()),static_cast<int*>(previous.data_ptr()),static_cast<int*>(following.data_ptr()),static_cast<int*>(ends.data_ptr()),static_cast<char*>(temporary.data_ptr()),size_t(temporary.shape()[0])}
int64_t linked_lru_temporary_bytes(int64_t n,int64_t L){
 size_t sort=0,unique=0;host::RuntimeDeviceCheck(cub::DeviceRadixSort::SortKeys(nullptr,sort,(int*)nullptr,(int*)nullptr,int(n),0,linked_list_lru::bits(int(L))));host::RuntimeDeviceCheck(cub::DeviceSelect::Unique(nullptr,unique,(int*)nullptr,(int*)nullptr,(int*)nullptr,int(n)));return std::max(sort,unique);
}
void linked_lru_deduplicate(TensorView selected,TensorView current,TensorView active,TensorView directory,LL_SCRATCH_ARGS){
 const auto stream=host::LaunchKernel::resolve_device(directory.device());auto x=LL_SCRATCH;const int Q=selected.shape()[0],K=selected.shape()[1],L=directory.shape()[0],n=Q*(K+1);
 linked_list_lru::merge<<<std::min(256,(n+255)/256),256,0,stream>>>(selected.data_ptr(),selected.dtype().bits/8,current.data_ptr(),current.dtype().bits/8,static_cast<int*>(active.data_ptr()),Q,K,L,x.ids);
 size_t bytes=x.temporary_bytes;host::RuntimeDeviceCheck(cub::DeviceRadixSort::SortKeys(x.temporary,bytes,x.ids,x.sorted_ids,n,0,linked_list_lru::bits(L),stream));bytes=x.temporary_bytes;host::RuntimeDeviceCheck(cub::DeviceSelect::Unique(x.temporary,bytes,x.sorted_ids,x.unique_ids,x.unique_count,n,stream));
}
void linked_lru_update(TensorView current,TensorView active,LL_STATE_ARGS,LL_WORK_ARGS,LL_SCRATCH_ARGS){
 const auto stream=host::LaunchKernel::resolve_device(directory.device());
 linked_list_lru::update<<<1,1,0,stream>>>(current.data_ptr(),current.dtype().bits/8,static_cast<int*>(active.data_ptr()),int(current.shape()[0]),LL_STATE,LL_WORK,LL_SCRATCH);host::RuntimeDeviceCheck(cudaGetLastError());
}
#undef LL_STATE_ARGS
#undef LL_WORK_ARGS
#undef LL_SCRATCH_ARGS
#undef LL_STATE
#undef LL_WORK
#undef LL_SCRATCH
}
