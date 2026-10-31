"""Execute the actual native writer body and address helpers as scalar C++.

CUDA scheduling/memory ordering still requires the separately prepared GPU
fixture. This check covers encoded bytes, pointer reconstruction and policy
branches; it does not claim CUDA correctness or performance.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


def definition(text, marker):
    start = text.index(marker)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def test_actual_writer_bytes_publication_and_uniform_address(tmp_path):
    compiler = shutil.which('c++') or shutil.which('clang++')
    if compiler is None:
        pytest.skip('A C++ compiler is required for native writer semantics')
    records = (ROOT / 'ulacache/csrc/storage/records.cuh').read_text()
    owners = (ROOT / 'ulacache/csrc/cache/owner_directory.cuh').read_text()
    enum = owners[owners.index('enum HostRecordField'):owners.index('};') + 2]
    functions = '\n'.join(definition(owners, '__device__ __forceinline__ ' + marker)
        for marker in ['int64_t record_host_location(', 'uint8_t* record_host_address(',
                       'uint8_t* record_host_writer_address_warp(',
                       'void record_host_touch(', 'void record_host_writer_touch('])
    body = '\n'.join(definition(records, marker) for marker in [
        'template<typename Element>\n__device__ __forceinline__ uint64_t record_plain_word(',
        'template<bool Decode, typename Element>\n__global__ void record_write_plain_kernel('])
    code = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define __device__
#define __global__
#define __forceinline__ inline
#define FLOW_PDL 0
#define FLOW_RECORD_HOST_TIER 1
#define FLOW_RECORD_STEP_PUBLICATION PUBLISHED
#define FLOW_ROW_BYTES 656
#define FLOW_HOST_DEVICE_PARAM , const int64_t* host
struct Dim { int x; } threadIdx, blockIdx, blockDim, gridDim;
int loads=0, stores=0, marks=0, fences=0, syncs=0, shuffles=0;
uint32_t broadcast[2];
uint32_t __shfl_sync(unsigned mask, uint32_t value, int source) {
  assert(mask == 0xffffffffu && source == 0);
  const int half=shuffles++ % 2;
  if ((threadIdx.x & 31) == 0) broadcast[half]=value;
  return broadcast[half];
}
void __syncwarp(unsigned mask) { assert(mask == 0xffffffffu); ++syncs; }
void __threadfence_system() { ++fences; }
void __nanosleep(int) { assert(false); }
uint64_t clock64() { return 0; }
namespace cuda {
enum {thread_scope_system, memory_order_acquire, memory_order_relaxed};
template<typename T, int Scope> struct atomic_ref {
  T& value;
  atomic_ref(T& v):value(v) {}
  T load(int) { ++loads; return value; }
  void store(T next,int) { ++stores; value=next; }
};
}
namespace flowkv_bitmap_ring {
struct Group { int64_t epoch; };
enum class Operation { Access };
}
namespace flowkv_ring_presence_recent {
void mark_slot(const flowkv_bitmap_ring::Group& g,int,flowkv_bitmap_ring::Operation,int64_t epoch) {
  assert(g.epoch == epoch); ++marks;
}
}
constexpr int PEERS=2, SLOTS=8, GPU=4, CPU=4, LAYERS=2, CAP=8;
int read_int(const void* p,int i,int bytes) { return bytes==8 ? int(((int64_t*)p)[i]) : ((int*)p)[i]; }
int record_global_slot(int logical) { return (logical%PEERS)*SLOTS+logical/PEERS; }
int record_owner(int gid) { return gid/SLOTS; }
int record_local(int gid) { return gid%SLOTS; }
struct RecordNativePage { static long offset(long row,int byte) { return row*FLOW_ROW_BYTES+byte; } };
'''
    tail = r'''
template<bool Decode> void run(bool cpu_row, int rank, bool sole) {
  constexpr int Q=4;
  std::vector<uint8_t> gpu0(LAYERS*GPU*FLOW_ROW_BYTES,0xa1), gpu1(gpu0.size(),0xa1);
  std::vector<uint8_t> cpu(LAYERS*CPU*FLOW_ROW_BYTES,0xb2), resident((CAP+Q)*FLOW_ROW_BYTES,0xc3);
  int64_t dirs[PEERS][SLOTS], times[PEERS][SLOTS];
  for(int o=0;o<PEERS;++o) for(int i=0;i<SLOTS;++i) { dirs[o][i]=cpu_row ? -i%CPU-2 : i%GPU; times[o][i]=97; }
  uint64_t peers[PEERS]={uint64_t(gpu0.data()),uint64_t(gpu1.data())};
  uint64_t directory_peers[PEERS]={uint64_t(dirs[0]),uint64_t(dirs[1])};
  uint64_t time_peers[PEERS]={uint64_t(times[0]),uint64_t(times[1])};
  int64_t epoch=97;
  flowkv_bitmap_ring::Group ring{97};
  int64_t host[HostRecordFields]{};
  host[HostDirectoryPeers]=int64_t(directory_peers); host[HostLocalCpu]=int64_t(cpu.data());
  host[HostLogicalRows]=SLOTS; host[HostGpuRows]=GPU; host[HostCpuRows]=CPU;
  host[HostEpoch]=int64_t(&epoch); host[HostAccessTimePeers]=int64_t(time_peers); host[HostRingGroup]=int64_t(&ring);
  std::vector<uint8_t> nope(Q*528),rope(Q*128);
  for(int i=0;i<int(nope.size());++i) nope[i]=(i*7+13)%251;
  for(int i=0;i<int(rope.size());++i) rope[i]=(i*11+29)%251;
  int64_t loc[Q]={1,2,3,4}; int rows[Q]={-1,-1,-1,-1};
  int active=3,totals[2]={0,0}; int64_t cursors[2]={0,0};
  if(sole) {loc[0]=rank ? 1:2;loc[1]=rank ? 3:4;loc[2]=rank ? 5:6;loc[3]=0;}
  loads=stores=marks=fences=syncs=shuffles=0;
  blockDim.x=Decode ? 32:128; gridDim.x=Q;
  for(int q=0;q<Q;++q) {
    blockIdx.x=q;
    for(int t=0;t<blockDim.x;++t) {
      threadIdx.x=t;
      record_write_plain_kernel<Decode,uint8_t>(nope.data(),rope.data(),loc,8,528,528,1,128,1,
          peers,resident.data(),rows,0,GPU*FLOW_ROW_BYTES,rank,sole,CAP,cursors,totals,&active,host);
    }
  }
  int published=0,live=Decode ? active:Q;
  for(int q=0;q<Q;++q) {
    assert(rows[q]==loc[q]);
    if(loc[q]<=0 || q>=live) continue;
    const int gid=record_global_slot(loc[q]), owner=record_owner(gid), local=record_local(gid);
    const int output=Decode ? CAP+q : int(loc[q]);
    std::vector<uint8_t> expected;
    expected.insert(expected.end(),nope.begin()+q*528,nope.begin()+(q+1)*528);
    expected.insert(expected.end(),rope.begin()+q*128,rope.begin()+(q+1)*128);
    assert(std::memcmp(resident.data()+output*FLOW_ROW_BYTES,expected.data(),FLOW_ROW_BYTES)==0);
    if(Decode) assert(std::memcmp(resident.data()+q*FLOW_ROW_BYTES,expected.data(),FLOW_ROW_BYTES)==0);
    if(sole || owner==rank) {
      ++published;
      uint8_t* destination=cpu_row ? cpu.data()+(CPU+local%CPU)*FLOW_ROW_BYTES
          : (owner ? gpu1.data():gpu0.data())+(GPU+local%GPU)*FLOW_ROW_BYTES;
      assert(std::memcmp(destination,expected.data(),FLOW_ROW_BYTES)==0);
    }
  }
  const bool touch=!Decode || !PUBLISHED;
  assert(loads == published*(blockDim.x/32 + int(touch)));
  assert(stores == published*int(touch));
  assert(marks == published*int(touch)*int(!cpu_row));
  assert(fences == published*blockDim.x*int(touch));
  assert(syncs == published*blockDim.x && shuffles == 2*syncs);
  assert(epoch==97);
  for(auto& owner:times) for(auto time:owner) assert(time==97);
}
int main() {
  for(bool cpu:{false,true}) for(int rank:{0,1}) for(bool sole:{false,true}) {
    run<true>(cpu,rank,sole);run<false>(cpu,rank,sole);
  }
  puts("actual writer byte/pointer/publication branches passed");
}
'''
    for published in (0, 1):
        source = tmp_path / f'writer_{published}.cpp'
        executable = tmp_path / f'writer_{published}'
        source.write_text(f'#define PUBLISHED {published}\n' + code + enum + functions + body + tail)
        result = subprocess.run([compiler, '-std=c++17', '-O1', str(source), '-o', str(executable)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        result = subprocess.run([str(executable)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert 'actual writer byte/pointer/publication branches passed' in result.stdout
