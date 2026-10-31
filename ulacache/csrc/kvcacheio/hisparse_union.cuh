// SPDX-License-Identifier: Apache-2.0
#pragma once
// Single virtual request per Index group. Derived native stages are recorded
// in hisparse_union_provenance.json; this global-scratch version is an adaptation.
#include "csrc/cache/cpu_residency.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
namespace sglang::hisparse_union {
using State=flowkv_host::State;
using Work=flowkv_host::Workspace;
struct Scratch {int *ids,*sorted,*unique,*count,*lru,*buffer,*locations; char* temporary; size_t temporary_bytes;};
__global__ void merge(const void* selected,int sb,const void* current,int cb,int q,int k,const int* active,int owner,State s,int* ids) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<q*(k+1);i+=gridDim.x*blockDim.x) {
    int query=i<q*k?i/k:i-q*k;
    int row=query<*active?flowkv_host::local_row(i<q*k?flowkv_host::read_row(selected,i,sb):flowkv_host::read_row(current,query,cb),owner,s.logical_slots):-1;
    ids[i]=row<0?s.logical_slots:row;
  }
}
template<int GPU_SLOTS,int MAX_KEYS,int QUERIES>
__global__ void update(const void* current,int current_bytes,const int* active,int owner,State s,Work w,Scratch x) {
  constexpr int BLOCK_SIZE=1024,NUM_WARPS=32,HOT_BUFFER_SIZE=GPU_SLOTS;
  constexpr int NUM_BUFFER_CHUNKS=(HOT_BUFFER_SIZE+31)/32,HASH_SIZE=2*MAX_KEYS;
  constexpr bool TopKIsBlocks=false; constexpr int SPARSE_BLOCK_SIZE=1;
  const int tid=threadIdx.x,warp_id=tid/32,lane_id=tid%32;
  const BallotMask lanes_before=(BallotMask(1)<<lane_id)-1;
  int U=*x.count;if(U && x.unique[U-1]==s.logical_slots)--U;
  assert(U<=GPU_SLOTS && U<=MAX_KEYS);
  const int NUM_TOP_K_TOKENS=U,NUM_TOKEN_CHUNKS=(U+31)/32;
  int* s_top_k_tokens=x.buffer;
  int* s_chunk_offset=s_top_k_tokens+MAX_KEYS;
  int* s_evict_chunk_offset=s_chunk_offset+NUM_BUFFER_CHUNKS+1;
  int* s_hash_keys=s_evict_chunk_offset+NUM_BUFFER_CHUNKS+1;
  int& s_total_hits=s_hash_keys[HASH_SIZE];
  int& s_total_misses=s_hash_keys[HASH_SIZE+1];
  int* s_lru_slots_out=s_hash_keys+HASH_SIZE+2;
  int* s_hash_vals=s_lru_slots_out+GPU_SLOTS;
  int* current_hash=s_hash_vals+HASH_SIZE;
  constexpr int current_capacity=QUERIES<=64?128:(QUERIES<=128?256:(QUERIES<=256?512:1024));
  const int32_t* req_top_k=x.unique;int* req_top_k_device_locs=x.locations;
  const int* req_device_buffer_tokens=s.gpu_tags;int* req_lru_slots=x.lru;
  const int newest_slot=GPU_SLOTS,newest_token=-1;const int64_t seq_len=s.logical_slots;
  if(!tid) {assert(*s.epoch<LLONG_MAX);++*s.epoch;s_total_hits=0;s_total_misses=0;}
  if(tid<8)w.counters[tid]=0;
  for(int i=tid;i<HASH_SIZE;i+=BLOCK_SIZE)s_hash_keys[i]=HASH_EMPTY;
  for(int i=tid;i<NUM_BUFFER_CHUNKS+1;i+=BLOCK_SIZE){s_chunk_offset[i]=0;s_evict_chunk_offset[i]=0;}
  for(int i=tid;i<current_capacity;i+=BLOCK_SIZE)current_hash[i]=-1;
  __syncthreads();
  for(int i=tid;i<QUERIES && i<*active;i+=BLOCK_SIZE) {
    int row=flowkv_host::local_row(flowkv_host::read_row(current,i,current_bytes),owner,s.logical_slots);
    if(row<0)continue;int h=(uint32_t(row)*2654435761u)&(current_capacity-1);
    while(true){int old=atomicCAS(current_hash+h,-1,row);if(old==-1||old==row)break;h=(h+1)&(current_capacity-1);}
  }
  __syncthreads();
  // Insert top-k tokens into shared-memory hash table.
  for (int i = tid; i < NUM_TOP_K_TOKENS; i += BLOCK_SIZE) {
    const int32_t token_idx = resolve_selected_token<SPARSE_BLOCK_SIZE, TopKIsBlocks>(req_top_k, i);
    if constexpr (TopKIsBlocks) {
      if (token_idx < 0 || token_idx >= seq_len) {
        s_top_k_tokens[i] = TOKEN_HIT;
        req_top_k_device_locs[i] = -1;
        continue;
      }
    }
    if (token_idx == newest_token) {
      // If topk includes the latest token, bind its canonical occurrence to newest_slot (at HOT_BUFFER_SIZE) and mark
      // it as a hit. newest_slot is at the first position of the extra page, excluded from LRU tracking.
      s_top_k_tokens[i] = TOKEN_HIT;
      req_top_k_device_locs[i] = newest_slot;
    } else {
      int slot = hash_slot(token_idx, HASH_SIZE);
      while (true) {
        int32_t old = atomicCAS(&s_hash_keys[slot], HASH_EMPTY, token_idx);
        if (old == HASH_EMPTY || old == token_idx) {
          s_hash_vals[slot] = i;
          break;
        }
        slot = (slot + 1) % HASH_SIZE;
      }
      s_top_k_tokens[i] = token_idx;
      s.access_time[token_idx] = *s.epoch;
    }
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_BUFFER = (NUM_BUFFER_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  int total_hit_count = 0;
  int total_evict_count = 0;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;

    const int slot_idx = chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_slot = has_valid_chunk && (slot_idx < HOT_BUFFER_SIZE);
    const int32_t buf_slot = has_valid_slot ? req_lru_slots[slot_idx] : -1;
    int32_t my_buffer_token = (buf_slot >= 0) ? req_device_buffer_tokens[buf_slot] : -1;
    int my_found_top_k_idx = -1;
    if (my_buffer_token >= 0) {
      int h = hash_slot(my_buffer_token, HASH_SIZE);
      while (true) {
        int32_t k = s_hash_keys[h];
        if (k == my_buffer_token) {
          my_found_top_k_idx = static_cast<int32_t>(s_hash_vals[h]);
          break;
        }
        if (k == HASH_EMPTY) break;
        h = (h + 1) % HASH_SIZE;
      }
    }
    bool is_hit = my_found_top_k_idx >= 0;
    bool is_evictable = has_valid_slot && !is_hit;

    // Record hits
    if (is_hit) {
      s_top_k_tokens[my_found_top_k_idx] = TOKEN_HIT;
      req_top_k_device_locs[my_found_top_k_idx] = buf_slot;
    }

    int local_hit_offset = 0;
    int local_evict_offset = 0;
    if (has_valid_chunk) {
      const BallotMask hit_mask = __ballot_sync(FULL_WARP_MASK, is_hit);
      const BallotMask evict_mask = __ballot_sync(FULL_WARP_MASK, is_evictable);
      local_hit_offset = popc_mask(hit_mask & lanes_before);
      local_evict_offset = popc_mask(evict_mask & lanes_before);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = popc_mask(hit_mask);
        s_evict_chunk_offset[chunk_idx + 1] = popc_mask(evict_mask);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
#ifdef USE_ROCM
      // ROCm wavefront64: WARP_SIZE (64) > NUM_WARPS (16 at block_size=1024),
      // so the wide-count form below would let lanes beyond this iteration's
      // NUM_WARPS-wide window write the accumulator into s_chunk_offset
      // positions belonging to future iterations, corrupting their reads.
      // Bound the scan window to NUM_WARPS lanes.
      const int scan_offset = iter * NUM_WARPS + 1;
      const int scan_count = min(scan_offset + NUM_WARPS, NUM_BUFFER_CHUNKS + 1);
      total_hit_count = warp_inclusive_scan(s_chunk_offset, lane_id, scan_offset, scan_count, total_hit_count);
      total_evict_count =
          warp_inclusive_scan(s_evict_chunk_offset, lane_id, scan_offset, scan_count, total_evict_count);
#else
      total_hit_count =
          warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_hit_count);
      total_evict_count =
          warp_inclusive_scan(s_evict_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_evict_count);
#endif
      if (tid == 0) {
        s_total_hits = total_hit_count;
      }
    }
    __syncthreads();

    // Hits grow forward from index 0
    if (is_hit) {
      int hit_offset = s_chunk_offset[chunk_idx] + local_hit_offset;
      s_lru_slots_out[hit_offset] = buf_slot;
    }
    // Evictables grow backward from HOT_BUFFER_SIZE - 1
    if (is_evictable) {
      int evict_offset = s_evict_chunk_offset[chunk_idx] + local_evict_offset;
      s_lru_slots_out[HOT_BUFFER_SIZE - 1 - evict_offset] = buf_slot;
    }
  }
  __syncthreads();

  // Reset offsets for the miss counting phase (only NUM_TOKEN_CHUNKS + 1 entries needed).
  for (int i = tid; i < NUM_TOKEN_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  // Third pass to identify misses and their evictable slots
  int total_misses = 0;
  const int ITERATIONS_PER_WARP_TOKEN = (NUM_TOKEN_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_TOKEN; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_TOKEN_CHUNKS;

    const int chunk_token_start = chunk_idx * WARP_SIZE;
    const int my_token_idx = chunk_token_start + lane_id;
    const bool has_valid_token = has_valid_chunk && (my_token_idx < NUM_TOP_K_TOKENS);

    int32_t my_token = 0;
    bool is_miss = false;
    int local_miss_offset = 0;

    if (has_valid_token) {
      is_miss = s_top_k_tokens[my_token_idx] != TOKEN_HIT;
      if (is_miss) {
        my_token = s_top_k_tokens[my_token_idx];
      }
    }

    if (has_valid_chunk) {
      const BallotMask miss_mask = __ballot_sync(FULL_WARP_MASK, is_miss);
      local_miss_offset = popc_mask(miss_mask & lanes_before);
      const int warp_miss_count = popc_mask(miss_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_miss_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
#ifdef USE_ROCM
      const int scan_offset = iter * NUM_WARPS + 1;
      const int scan_count = min(scan_offset + NUM_WARPS, NUM_TOKEN_CHUNKS + 1);
      total_misses = warp_inclusive_scan(s_chunk_offset, lane_id, scan_offset, scan_count, total_misses);
#else
      total_misses = warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_TOKEN_CHUNKS + 1, total_misses);
#endif
      if (tid == 0) {
        s_total_misses = total_misses;
      }
    }
    __syncthreads();

    if (is_miss) {
      int miss_offset = s_chunk_offset[chunk_idx] + local_miss_offset;
      int32_t evict_slot = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - miss_offset];
      // Reuse s_top_k_tokens as miss scratch: miss_offset < my_token_idx always
      // holds (hits are skipped), so compacted writes never overrun pending reads.
      s_top_k_tokens[miss_offset] = my_token;
      req_top_k_device_locs[my_token_idx] = evict_slot;
      int h = (uint32_t(my_token) * 2654435761u) & (current_capacity - 1);
      while (current_hash[h] != -1 && current_hash[h] != my_token) h = (h + 1) & (current_capacity - 1);
      w.promote_rows[miss_offset] = current_hash[h] == my_token ? -my_token - 1 : my_token;
      flowkv_host::assign_plan_item(s, w, miss_offset, evict_slot);

    }
  }
  __syncthreads();

  total_misses = s_total_misses;
  if (tid == 0) {
    w.counters[flowkv_host::PromoteCount] = total_misses;
    w.counters[flowkv_host::PlanCount] = total_misses;
    w.counters[flowkv_host::VictimCount] = total_misses;
    w.counters[flowkv_host::Candidates] = total_misses;
  }
  // Write back LRU order: evictables at front (LRU), hits at back (MRU).
  {
    const int total_evictable = HOT_BUFFER_SIZE - s_total_hits;
#ifdef USE_ROCM
    // ROCm: cap writeback threads at 512 for large kernels.
    constexpr int LRU_WRITEBACK_THREADS = (BLOCK_SIZE > 512) ? 512 : BLOCK_SIZE;
    if (tid < LRU_WRITEBACK_THREADS) {
      for (int i = tid; i < HOT_BUFFER_SIZE; i += LRU_WRITEBACK_THREADS) {
        if (i < total_misses) {
          // Misses: just loaded from host, place right before hits
          req_lru_slots[total_evictable - total_misses + i] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
        } else if (i < total_evictable) {
          // Remaining evictables: truly stale, dest at LRU front
          req_lru_slots[i - total_misses] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
        } else {
          // Hits: source at forward end, dest at MRU back
          req_lru_slots[i] = s_lru_slots_out[i - total_evictable];
        }
      }
    }
#else
    for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
      if (i < total_misses) {
        // Misses: just loaded from host, place right before hits
        req_lru_slots[total_evictable - total_misses + i] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
      } else if (i < total_evictable) {
        // Remaining evictables: truly stale, dest at LRU front
        req_lru_slots[i - total_misses] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
      } else {
        // Hits: source at forward end, dest at MRU back
        req_lru_slots[i] = s_lru_slots_out[i - total_evictable];
      }
    }
#endif
  }


}
__global__ void pack_native(const int* unique,const int* count,int* selected,int n,int logical) {
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)selected[i]=i<*count && unique[i]<logical?unique[i]:-1;
}
}
namespace sglang {
#define HU_STATE_ARGS TensorView directory,TensorView gpu_tags,TensorView host_tags,TensorView access_time,TensorView leases,TensorView epoch,TensorView gpu_free,TensorView host_free,TensorView free_counts
#define HU_WORK_ARGS TensorView new_rows,TensorView promote_rows,TensorView plan,TensorView counters,TensorView histogram,TensorView selection_prefix,TensorView candidate_slots,TensorView leased_slots
#define HU_SCRATCH_ARGS TensorView ids,TensorView sorted_ids,TensorView unique_ids,TensorView unique_count,TensorView lru_slots,TensorView scratch,TensorView locations,TensorView temporary
#define HU_STATE flowkv_host::State{static_cast<int64_t*>(directory.data_ptr()),static_cast<int*>(gpu_tags.data_ptr()),static_cast<int*>(host_tags.data_ptr()),static_cast<int64_t*>(access_time.data_ptr()),static_cast<int*>(leases.data_ptr()),static_cast<int64_t*>(epoch.data_ptr()),static_cast<int*>(gpu_free.data_ptr()),static_cast<int*>(host_free.data_ptr()),static_cast<int*>(free_counts.data_ptr()),int(directory.shape()[0]),int(gpu_tags.shape()[0]),int(host_tags.shape()[0])}
#define HU_WORK flowkv_host::Workspace{static_cast<int*>(new_rows.data_ptr()),static_cast<int*>(promote_rows.data_ptr()),static_cast<int*>(plan.data_ptr()),static_cast<int*>(counters.data_ptr()),static_cast<int*>(histogram.data_ptr()),static_cast<int64_t*>(selection_prefix.data_ptr()),static_cast<int*>(candidate_slots.data_ptr()),static_cast<int*>(leased_slots.data_ptr()),int(new_rows.shape()[0])}
#define HU_SCRATCH hisparse_union::Scratch{static_cast<int*>(ids.data_ptr()),static_cast<int*>(sorted_ids.data_ptr()),static_cast<int*>(unique_ids.data_ptr()),static_cast<int*>(unique_count.data_ptr()),static_cast<int*>(lru_slots.data_ptr()),static_cast<int*>(scratch.data_ptr()),static_cast<int*>(locations.data_ptr()),static_cast<char*>(temporary.data_ptr()),size_t(temporary.shape()[0])}
inline int hu_bits(int x){int n=0;do{++n;x>>=1;}while(x);return n;}
int64_t hisparse_union_temporary_bytes(int64_t count,int64_t logical) {
 size_t sort=0,unique=0;
 host::RuntimeDeviceCheck(cub::DeviceRadixSort::SortKeys(nullptr,sort,(int*)nullptr,(int*)nullptr,int(count),0,hu_bits(logical)));
 host::RuntimeDeviceCheck(cub::DeviceSelect::Unique(nullptr,unique,(int*)nullptr,(int*)nullptr,(int*)nullptr,int(count)));
 return std::max(sort,unique);
}
void hisparse_union_dedup(TensorView selected,TensorView current,TensorView active,int64_t owner,HU_STATE_ARGS,HU_SCRATCH_ARGS) {
 auto s=HU_STATE;auto x=HU_SCRATCH;auto stream=host::LaunchKernel::resolve_device(gpu_tags.device());
 int q=selected.shape()[0],k=selected.shape()[1],n=q*(k+1);
 hisparse_union::merge<<<std::min(256,(n+255)/256),256,0,stream>>>(selected.data_ptr(),selected.dtype().bits/8,current.data_ptr(),current.dtype().bits/8,q,k,static_cast<int*>(active.data_ptr()),owner,s,x.ids);
 size_t bytes=x.temporary_bytes;
 host::RuntimeDeviceCheck(cub::DeviceRadixSort::SortKeys(x.temporary,bytes,x.ids,x.sorted,n,0,hu_bits(s.logical_slots),stream));
 bytes=x.temporary_bytes;
 host::RuntimeDeviceCheck(cub::DeviceSelect::Unique(x.temporary,bytes,x.sorted,x.unique,x.count,n,stream));
}
template<int G,int M,int Q>
void hisparse_union_update(TensorView current,TensorView active,int64_t owner,HU_STATE_ARGS,HU_WORK_ARGS,HU_SCRATCH_ARGS) {
 auto stream=host::LaunchKernel::resolve_device(gpu_tags.device());
 hisparse_union::update<G,M,Q><<<1,1024,0,stream>>>(current.data_ptr(),current.dtype().bits/8,static_cast<int*>(active.data_ptr()),int(owner),HU_STATE,HU_WORK,HU_SCRATCH);
 host::RuntimeDeviceCheck(cudaGetLastError());
}
template<int G,int M>
void hisparse_union_native_reference(TensorView unique_ids,TensorView unique_count,TensorView selected,TensorView tokens,TensorView host_locs,TensorView device_locs,TensorView output,TensorView lru,TensorView request,TensorView sequence,TensorView active,TensorView miss_src,TensorView miss_dst,TensorView miss_count) {
 if constexpr(G<=32768 && M<=32767) {
   constexpr int N=1024;constexpr size_t bytes=SmemLayout<M,G>::BYTES;
   auto stream=host::LaunchKernel::resolve_device(tokens.device());
   hisparse_union::pack_native<<<(M+255)/256,256,0,stream>>>(static_cast<int*>(unique_ids.data_ptr()),static_cast<int*>(unique_count.data_ptr()),static_cast<int*>(selected.data_ptr()),M,int(host_locs.shape()[0]));
   auto kernel=load_cache_to_device_buffer_kernel<N,M,G,true,false,1,true,true,true,int32_t,int32_t>;
   host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
   kernel<<<1,N,bytes,stream>>>(static_cast<int*>(selected.data_ptr()),static_cast<int*>(tokens.data_ptr()),static_cast<int64_t*>(host_locs.data_ptr()),static_cast<int*>(device_locs.data_ptr()),nullptr,nullptr,nullptr,nullptr,static_cast<int*>(output.data_ptr()),static_cast<int*>(request.data_ptr()),static_cast<int*>(sequence.data_ptr()),static_cast<int16_t*>(lru.data_ptr()),static_cast<int*>(active.data_ptr()),G+1,host_locs.shape()[0],G,M,M,1,0,static_cast<int64_t*>(miss_src.data_ptr()),static_cast<int*>(miss_dst.data_ptr()),static_cast<int*>(miss_count.data_ptr()),M);
   host::RuntimeDeviceCheck(cudaGetLastError());
 } else { throw std::runtime_error("Native HiSparse reference slot layout is unsupported"); }
}
#undef HU_STATE_ARGS
#undef HU_WORK_ARGS
#undef HU_SCRATCH_ARGS
#undef HU_STATE
#undef HU_WORK
#undef HU_SCRATCH
}
