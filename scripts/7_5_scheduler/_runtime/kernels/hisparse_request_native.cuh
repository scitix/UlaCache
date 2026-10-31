// SPDX-License-Identifier: Apache-2.0
#pragma once
// Invoke the existing public HiSparse device body without modifying it.
// Outer adapters provide per-request MTP2 unions and exclusive-cold metadata.
#include "csrc/cache/cpu_residency.cuh"
#include <cub/device/device_segmented_radix_sort.cuh>
namespace sglang {
namespace hisparse_request_detail {
inline int bits(int value) {int n=0;do{++n;value>>=1;}while(value);return n;}
__global__ void merge(const int32_t* selected,const void* current,int bytes,int* ids) {
 const int request=blockIdx.x;
 for(int i=threadIdx.x;i<6147;i+=blockDim.x)
  ids[request*6147+i]=i<6144?selected[request*6144+i]:int(flowkv_host::read_row(current,request*3+i-6144,bytes));
}
__global__ void compact(const int* sorted,int* unique,int64_t* times,const int64_t* epoch) {
 const int request=blockIdx.x,tid=threadIdx.x;
 using Scan=cub::BlockScan<int,256>;
 __shared__ typename Scan::TempStorage scratch;
 __shared__ int base;
 if(!tid)base=0;
 __syncthreads();
 for(int start=0;start<6147;start+=256) {
  const int i=start+tid,uid=i<6147?sorted[request*6147+i]:-1;
  const int keep=i<6147&&(i==0||uid!=sorted[request*6147+i-1]);
  int offset,total;Scan(scratch).ExclusiveSum(keep,offset,total);
  if(keep){unique[request*6147+base+offset]=uid;times[uid]=*epoch;}
  __syncthreads();
  if(!tid)base+=total;
  __syncthreads();
 }
 for(int i=base+tid;i<6147;i+=256)unique[request*6147+i]=-1;
}
}
int64_t hisparse_request_temporary_bytes(int64_t requests,int64_t logical) {
 size_t bytes=0;
 host::RuntimeDeviceCheck(cub::DeviceSegmentedRadixSort::SortKeys(nullptr,bytes,
  (int*)nullptr,(int*)nullptr,int(requests*6147),int(requests),(int*)nullptr,(int*)nullptr,
  0,hisparse_request_detail::bits(int(logical))));
 return bytes;
}
void hisparse_request_dedup(TensorView selected,TensorView current,TensorView unique,
 TensorView ids,TensorView sorted,TensorView offsets,TensorView temporary,
 TensorView directory,TensorView times,TensorView epoch) {
 const auto stream=host::LaunchKernel::resolve_device(selected.device());
 const int requests=current.shape()[0]/3;
 auto input=static_cast<int*>(ids.data_ptr());auto output=static_cast<int*>(sorted.data_ptr());
 auto segments=static_cast<int*>(offsets.data_ptr());
 hisparse_request_detail::merge<<<requests,256,0,stream>>>(
  static_cast<int32_t*>(selected.data_ptr()),current.data_ptr(),current.dtype().bits/8,input);
 size_t bytes=temporary.shape()[0];
 host::RuntimeDeviceCheck(cub::DeviceSegmentedRadixSort::SortKeys(temporary.data_ptr(),bytes,
  input,output,requests*6147,requests,segments,segments+1,0,
  hisparse_request_detail::bits(int(directory.shape()[0])),stream));
 hisparse_request_detail::compact<<<requests,256,0,stream>>>(output,
  static_cast<int*>(unique.data_ptr()),static_cast<int64_t*>(times.data_ptr()),static_cast<int64_t*>(epoch.data_ptr()));
 host::RuntimeDeviceCheck(cudaGetLastError());
}
void hisparse_native_request(TensorView unique,TensorView tags,TensorView directory,
 TensorView device_locations,TensorView output,TensorView request_indices,
 TensorView sequence_lengths,TensorView lru,TensorView active_requests,
 TensorView miss_source,TensorView miss_destination,TensorView miss_count) {
 constexpr int N=1024,M=6147,G=8192;
 constexpr size_t bytes=SmemLayout<M,G>::BYTES;
 auto kernel=load_cache_to_device_buffer_kernel<N,M,G,true,false,1,true,true,true,int64_t,int32_t>;
 static const bool configured=[] {
   host::RuntimeDeviceCheck(cudaFuncSetAttribute(
    load_cache_to_device_buffer_kernel<N,M,G,true,false,1,true,true,true,int64_t,int32_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
   return true;
 }();
 (void)configured;
 auto stream=host::LaunchKernel::resolve_device(tags.device());
 kernel<<<unique.shape()[0],N,bytes,stream>>>(
   static_cast<int32_t*>(unique.data_ptr()),static_cast<int32_t*>(tags.data_ptr()),
   static_cast<int64_t*>(directory.data_ptr()),static_cast<int32_t*>(device_locations.data_ptr()),
   nullptr,nullptr,nullptr,nullptr,static_cast<int32_t*>(output.data_ptr()),
   static_cast<int32_t*>(request_indices.data_ptr()),static_cast<int64_t*>(sequence_lengths.data_ptr()),
   static_cast<int16_t*>(lru.data_ptr()),static_cast<int32_t*>(active_requests.data_ptr()),
   G,0,G,M,M,1,0,static_cast<int64_t*>(miss_source.data_ptr()),
   static_cast<int32_t*>(miss_destination.data_ptr()),static_cast<int32_t*>(miss_count.data_ptr()),M);
 host::RuntimeDeviceCheck(cudaGetLastError());
}
}
