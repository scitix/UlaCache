"""Exercise actual metadata bodies and low-Q dispatch without a CUDA context."""
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / 'ulacache/csrc/cache'


def cpp_function(source, name):
    match = re.search(r'\b' + name + r'\s*\(', source)
    assert match, name
    start = source.rfind('\n', 0, match.start()) + 1
    opened = source.index('{', match.end())
    depth = 1
    end = opened + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def test_actual_finish_math_and_last_cta_release_chain(tmp_path):
    """CPU atomics model completion ordering; CUDA barriers need the GPU gate."""
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('A C++ compiler is required for the actual-body CPU check')
    residency = (KERNELS / 'cpu_residency.cuh').read_text()
    fused = (KERNELS / 'residency_fused.cuh').read_text()
    structs = (KERNELS / 'directory.cuh').read_text().split('namespace sglang::flowkv_host {', 1)[1].rsplit('}', 1)[0]
    functions = '\n'.join(cpp_function(residency, name) for name in (
        'pop_free', 'reserve_counts', 'assign_plan_item', 'assign_plan_cta', 'metadata_finish_body'))
    functions += '\n' + cpp_function(fused, 'arrive_actual_miss_cta')
    # The real CUDA atomic API is replaced only by the same CPU memory-order
    # primitive; the tested reservation/plan/completion bodies are not rewritten.
    source = r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>
using std::min;
#define __device__
#define __forceinline__ inline
#define __syncthreads() ((void)0)
#define UDCACHE_NAIVE_LRU 0
struct Dim { int x; } threadIdx{0}, blockDim{256};
inline int atomicAdd(int* p, int n) { return __atomic_fetch_add(p, n, __ATOMIC_SEQ_CST); }
inline int atomicCAS(int* p, int old, int desired) {
  __atomic_compare_exchange_n(p, &old, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); return old;
}
namespace cuda {
enum Scope { thread_scope_device }; enum Order { memory_order_acq_rel };
template<class T, Scope> struct atomic_ref {
  T& value; explicit atomic_ref(T& x): value(x) {}
  T fetch_add(T n, Order) { return __atomic_fetch_add(&value,n,__ATOMIC_ACQ_REL); }
};
}
int ring_finishes=0;
namespace ring {
enum class Operation { Access };
struct Group {};
struct Transaction {
  Operation op; int64_t epoch; const int* slots; int count;
  const int* free_targets; int free_count; int victim_count;
};
}
namespace recent {
template<int> struct Storage {};
template<int N> void finish(const ring::Group&, const ring::Transaction&, Storage<N>&) {
  if (threadIdx.x==0) ++ring_finishes;
}
}
'''
    source += structs + functions + r'''
int main() {
  for (int fresh=0; fresh<=2; ++fresh) for (int promote=0; promote<=2; ++promote)
    for (int free=0; free<=4; ++free) for (bool cpu_current:{false,true}) {
      int directory[32],tags[8],host_tags[8]{},leases[8]{},gpu_free[8],host_free[8];
      int64_t times[32]{},epoch=42; int free_counts[2]{free,8};
      for(int i=0;i<32;++i) directory[i]=-1;
      for(int i=0;i<8;++i) { tags[i]=i<free?-1:16+i; gpu_free[i]=i; host_free[i]=i; }
      for(int i=0;i<promote;++i) directory[8+i]=-2-i;
      int new_rows[8]{2,3},promote_rows[8],plan[40],counters[8]{fresh,promote,91,92,93,94,95,96};
      int64_t prefix[3]{31,32,33};int candidates[8],leased[8]{};
      for(int i=0;i<8;++i) { promote_rows[i]=cpu_current?-9-i:8+i; candidates[i]=free+i; }
      std::fill(plan,plan+40,-777);ring::Group group;
      State s{reinterpret_cast<int64_t*>(nullptr),tags,host_tags,times,leases,&epoch,
        gpu_free,host_free,free_counts,32,8,8};
      int64_t wide_directory[32];for(int i=0;i<32;++i)wide_directory[i]=directory[i];
      s.directory=wide_directory;
      Workspace w{new_rows,promote_rows,plan,counters,reinterpret_cast<int*>(&group),prefix,candidates,leased,8};
      ring_finishes=0;
      // Lane 0 executes the original reservation before CPU sequential lanes;
      // GPU execution uses the unchanged CTA barrier for the same dependency.
      for(int lane=0;lane<256;++lane) { threadIdx.x=lane;recent::Storage<256> scratch;metadata_finish_body(s,w,scratch); }
      int count=fresh+promote, used=std::min(count,free), victims=count-used;
      assert(counters[PlanCount]==count && counters[FreeUsed]==used && counters[VictimCount]==victims);
      assert(counters[Candidates]==victims && counters[LeaseCount]==0 && counters[PendingReserve]==0);
      assert(free_counts[0]==free-used && prefix[0]==31 && prefix[1]==32 && prefix[2]==victims);
      assert(epoch==42 && ring_finishes==(count!=0));
      if(!count) for(int value:plan)assert(value==-777);
      for(int i=0;i<count;++i) {
        int row=i<fresh?new_rows[i]:8+i-fresh;
        int gpu=i<used?free-used+i:candidates[i-used];
        assert(plan[i]==row && plan[8+i]==gpu);
        if(i<fresh)assert(plan[16+i]==-1);
        else { int cpu=i-fresh;assert(plan[16+i]==(cpu_current?-cpu-2:cpu)); }
      }
    }
  // Execute the actual arrive function under many concurrent CTA completion
  // orders, including grids whose producer bodies have no active rows.
  for(int q=1;q<8;++q)for(int repetition=0;repetition<12;++repetition) {
    int ctas=8*q*4,counters[8]{};std::vector<int> writes(ctas,0);
    Workspace w{};w.counters=counters;std::atomic<int> completed{0};
    std::vector<std::thread> workers;
    for(int worker=0;worker<8;++worker)workers.emplace_back([&,worker] {
      for(int item=worker;item<ctas;item+=8) {
        int cta=repetition%2?ctas-1-item:item;
        writes[cta]=cta+1;
        if(arrive_actual_miss_cta(w,ctas)) {
          for(int i=0;i<ctas;++i)assert(writes[i]==i+1);
          ++completed;
        }
      }
    });
    for(auto& worker:workers)worker.join();
    assert(completed==1 && counters[PendingReserve]==ctas);
    // The finish body subsequently retires/reuses the word for the next group step.
    counters[PendingReserve]=0;
  }
}
'''
    # nullptr casts differ among host compilers; State receives its real array below.
    source = source.replace('reinterpret_cast<int64_t*>(nullptr)', 'nullptr')
    path = tmp_path / 'actual_metadata.cpp'
    path.write_text(source)
    binary = tmp_path / 'actual_metadata'
    subprocess.run([compiler, '-std=c++17', '-pthread', str(path), '-o', str(binary)],
                   check=True, capture_output=True, text=True, timeout=30)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=20)


def test_inactive_and_empty_tiles_cannot_bypass_the_completion_vote():
    source = (KERNELS / 'residency_fused.cuh').read_text()
    body = cpp_function(source, 'prepare_actual_misses_body')
    fused = cpp_function(source, 'finish_actual_miss_producers')
    assert 'if (query >= *active) return;' in body
    assert 'if (int64_t(tile) * blockDim.x >= count) return;' in body
    # Early returns are in the called producer body, never in its fused caller.
    assert re.search(r'\breturn\b', fused) is None
    operations = ['prepare_actual_misses_body<false>', '__threadfence();', '__syncthreads();',
                  'arrive_actual_miss_cta', '__syncthreads();', 'metadata_finish_body']
    rest=fused
    for op in operations:
        assert op in rest
        rest=rest.split(op,1)[1]
    assert 'while' not in fused
    begin=cpp_function(source,'begin_group')
    assert 'if (threadIdx.x < 8) f.work().counters[threadIdx.x] = 0;' in begin
    # Exhaustive grids retain exactly one vote per CTA, regardless of the
    # producer's query-active and tile-empty branches (the grid is unchanged).
    for q in range(1,8):
        for active in range(q+1):
            counts=[0,1,255,256,257,1024,2048]
            for count in counts:
                visited=[(peer,query,tile,query<active and tile*256<count)
                         for peer in range(8) for query in range(q) for tile in range(4)]
                assert len(visited)==8*q*4


@pytest.mark.parametrize('policy,q,parallel,expected',[
    ('pb',0,True,'finish'),('pb',1,False,'finish_prepared'),
    ('pb',3,True,'finish_prepared'),('pb',7,True,'finish_prepared'),
    ('pb',8,True,'finish_parallel'),('pb',24,True,'finish_parallel'),
    ('pb',8,False,'finish'),('naive',3,True,'finish'),
    ('naive',8,True,'finish_parallel'),
])
def test_actual_prepare_group_dispatch_agrees_with_compiled_query_boundary(monkeypatch,policy,q,parallel,expected):
    import ulacache.compiler as compiler
    from ulacache.cache.host import FlowKVHostTier
    monkeypatch.setattr(compiler,'_recency_policy',None)
    monkeypatch.setenv('UDCACHE_RECENCY_POLICY',policy)
    called=[]
    tier=object.__new__(FlowKVHostTier)
    tier.groups=NS(group=lambda layer:0,anchors=(0,),followers=((1,2),))
    tier.parallel_finish=parallel
    tier._cold_module=NS(**{name:(lambda *args,name=name:called.append((name,args)))
                          for name in ('finish','finish_prepared','finish_parallel')})
    tier.pool=NS(flowkv_shards=[0,1,2]);tier.cold_execs=[11]
    class Ready:
        def __getitem__(self,key):return ('ready',key)
    tier.ready=Ready();tier.ready_peers=NS(ptrs=17)
    tier.states=[('state',)];tier.workspaces=[('work',)]
    tier.prepare_group(0,queries=q)
    assert len(called)==1 and called[0][0]==expected
    args=called[0][1]
    assert args[-2:]==('state','work')
    assert args[5:-2]==((q,) if expected=='finish_parallel' else ())
    # This is the actual C++ launch predicate, not a copied CPU implementation.
    common=(KERNELS/'common.cuh').read_text()
    position=common.index('prepare_actual_misses_and_finish<<<')
    assert '#if !UDCACHE_NAIVE_LRU\n    if (batch < 8)' in common[position-100:position]
    prepared=(policy!='naive' and 0<q<8)
    assert (expected=='finish_prepared')==prepared
