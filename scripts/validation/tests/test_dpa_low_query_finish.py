"""Actual native launch bodies and serial reference math without CUDA.

Host tests do not validate device barriers, PB bitmap kernels or Graph replay;
those remain part of the root-owned component campaign.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / 'ulacache/csrc/cache'


def cpp_function(source, name):
    match = re.search(r'\b'+name+r'\s*\(',source)
    assert match, name
    start = source.rfind('\n',0,match.start())+1
    parameters, cursor = 1, match.end()
    while parameters:
        parameters += (source[cursor] == '(') - (source[cursor] == ')')
        cursor += 1
    opened = source.index('{', cursor)
    depth, end = 1, opened+1
    while depth:
        depth += (source[end]=='{')-(source[end]=='}')
        end += 1
    return source[start:end]


def run_cpp(tmp_path, source, *definitions):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('A C++ compiler is required for actual-body CPU checks')
    path = tmp_path/'actual_dpa_finish.cpp'
    path.write_text(source)
    output = tmp_path/'actual_dpa_finish'
    subprocess.run([compiler,'-std=c++17',*definitions,str(path),'-o',str(output)],
        check=True,capture_output=True,text=True,timeout=30)
    subprocess.run([str(output)],check=True,capture_output=True,text=True,timeout=20)


def cpu_launches(source):
    # Strip only CUDA host launch syntax; execute the original host decisions
    # and arguments with kernel-entry recorders. No dispatch is reimplemented.
    return re.sub(r'<<<.*?>>>','',source,flags=re.DOTALL)


@pytest.mark.parametrize('naive',[0,1])
@pytest.mark.parametrize('parallel_victims',[0,1])
def test_actual_native_prepare_and_finish_launches(tmp_path,naive,parallel_victims):
    directory = (KERNELS/'directory.cuh').read_text().split(
        'namespace sglang::flowkv_host {',1)[1].rsplit('}',1)[0]
    finish = cpp_function((KERNELS/'residency_finish.cuh').read_text(),'finish_metadata')
    parallel = cpp_function((KERNELS/'residency_parallel.cuh').read_text(),
        'flowkv_host_prepare_parallel')
    after_begin = cpp_function((KERNELS/'batched_prepare.cuh').read_text(),
        'flowkv_host_prepare_after_begin')
    declarations = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
using cudaStream_t=int;
using std::min;
std::vector<std::string> calls;
struct Type { int bits=32; };
struct TensorView {
  int dimensions[2]; void* data_ptr(){return nullptr;}
  Type dtype(){return {};}
  const int* shape(){return dimensions;}
  int device(){return 0;}
};
int cudaGetLastError(){return 0;}
namespace host {
void RuntimeDeviceCheck(int){}
struct LaunchKernel { static int resolve_device(int){return 0;} };
}
namespace flowkv_host {
'''+directory+r'''
template<class... A> void begin_current(A...){calls.push_back("begin_current");}
void metadata_finish(State,Workspace){calls.push_back("serial_reference_finish");}
void naive_finish_select(int,State,Workspace){calls.push_back("naive_finish");}
}
namespace flowkv_host_parallel {
namespace base=flowkv_host;
struct PlanDiagnostic {
  const uint64_t* ready_peers=nullptr;
  const int64_t* local_ready=nullptr;
  int peers=0, ready_offset=0, owner=-1, group=-1;
};
template<class... A> void pin_rows(A...){calls.push_back("pin_rows");}
void select_victims(base::State,base::Workspace,PlanDiagnostic){calls.push_back("select_victims");}
void plan_victim_buckets(base::State,base::Workspace,PlanDiagnostic){calls.push_back("plan_victim_buckets");}
void select_victim_buckets(base::State,base::Workspace){calls.push_back("select_victim_buckets");}
void assign_parallel(base::State,base::Workspace){calls.push_back("assign_parallel");}
'''+cpu_launches(finish)+'\n}\n'+cpu_launches(parallel)+'\n'+cpu_launches(after_begin)
    signature = parallel[:parallel.index('{')]
    parameters = re.findall(r'(TensorView|int64_t|bool)\s+(\w+)',signature)
    arguments = ','.join({'owner':'0','use_counts':'false'}.get(name,
        'selected' if name=='selected' else 'value') for _,name in parameters)
    source = declarations+r'''
int main(){
  TensorView value{{1,1}};
  for(int queries: {0,1,2,3,6,7,8,12,24})for(bool batched: {false,true}) {
    TensorView selected{{queries,2048}};
    calls.clear();
    if(batched)flowkv_host_prepare_after_begin('''+arguments+r''');
    else flowkv_host_prepare_parallel('''+arguments+r''');
    std::vector<std::string> expected;
    if(!batched)expected.push_back("begin_current");
    if(queries)expected.push_back("pin_rows");
#if !UDCACHE_NAIVE_LRU
    if(queries>0 && queries<8)expected.push_back("serial_reference_finish");
    else {
      if(UDCACHE_PARALLEL_VICTIMS && queries>=8){
        expected.push_back("plan_victim_buckets");expected.push_back("select_victim_buckets");
      }else expected.push_back("select_victims");
      expected.push_back("assign_parallel");
    }
#else
    expected.push_back("naive_finish");expected.push_back("assign_parallel");
#endif
    assert(calls==expected);
  }
}
'''
    run_cpp(tmp_path,source,f'-DUDCACHE_NAIVE_LRU={naive}',
        f'-DUDCACHE_PARALLEL_VICTIMS={parallel_victims}')


def test_actual_current_selected_recency_and_serial_finish_counters(tmp_path):
    """Execute actual pin/reserve/assignment math with a recording ring service."""
    source = (KERNELS/'cpu_residency.cuh').read_text()
    structs = (KERNELS/'directory.cuh').read_text().split(
        'namespace sglang::flowkv_host {',1)[1].rsplit('}',1)[0]
    functions = '\n'.join(cpp_function(source,name) for name in
        ('pop_free','pin_row','reserve_counts','assign_plan_item',
         'assign_plan_cta','metadata_finish_body'))
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
using std::min;
#define __device__
#define __forceinline__ inline
#define __syncthreads() ((void)0)
#define UDCACHE_NAIVE_LRU 0
struct Dim{int x;} threadIdx{0},blockDim{256};
int atomicAdd(int* p,int n){int old=*p;*p+=n;return old;}
int atomicCAS(int* p,int old,int value){int prior=*p;if(prior==old)*p=value;return prior;}
namespace cuda {
enum Scope{thread_scope_system};enum Order{memory_order_relaxed};
template<class T,Scope>struct atomic_ref{
  T& value;explicit atomic_ref(T& v):value(v){}
  T load(Order){return value;}void store(T v,Order){value=v;}
  T exchange(T v,Order){T old=value;value=v;return old;}
};}
namespace ring {
enum class Operation{Access};
struct Group{std::vector<int> touched;int finish_calls=0;};
struct Transaction{Operation operation;int64_t epoch;const int* incoming;
  int incoming_count;const int* free_targets;int free_count;int victim_count;};
}
namespace recent {
template<int>struct Storage{};
void mark_slot(const ring::Group& g,int slot,ring::Operation,int64_t){
  const_cast<ring::Group&>(g).touched.push_back(slot);
}
template<int N>void finish(const ring::Group& g,const ring::Transaction&,Storage<N>&){
  if(threadIdx.x==0)++const_cast<ring::Group&>(g).finish_calls;
}
}
'''+structs+functions+r'''
int main(){
  for(bool fresh:{false,true})for(bool cpu_current:{false,true})
    for(bool cpu_selected:{false,true})for(int available:{0,1,4}) {
      int64_t directory[32],times[32]{},epoch=42;
      int tags[8],host_tags[8]{},leases[8],gpu_free[8],host_free[8];
      for(int i=0;i<32;++i)directory[i]=-1;
      for(int i=0;i<8;++i){tags[i]=i<available?-1:16+i;gpu_free[i]=i;host_free[i]=i;leases[i]=7+i;}
      // Resident current/selected IDs, including a timestamp already equal to
      // this epoch, must still structurally mark their GPU slots.
      directory[10]=6;tags[6]=10;directory[11]=7;tags[7]=11;times[11]=epoch;
      directory[12]=-2;directory[13]=-3;
      int free_counts[2]{available,8};
      int new_rows[8]{},promote_rows[8]{},plan[40],counters[8]{};
      int candidates[8],leased[8];int64_t prefix[3]{9,10,11};
      std::fill(plan,plan+40,-777);
      for(int i=0;i<8;++i){candidates[i]=available+i;leased[i]=31+i;}
      ring::Group group;
      State s{directory,tags,host_tags,times,leases,&epoch,gpu_free,host_free,
        free_counts,32,8,8};
      Workspace w{new_rows,promote_rows,plan,counters,reinterpret_cast<int*>(&group),
        prefix,candidates,leased,8};
      pin_row(10,true,s,w);pin_row(11,false,s,w);pin_row(11,false,s,w);
      if(fresh){pin_row(14,true,s,w);pin_row(14,true,s,w);}
      if(cpu_current){pin_row(12,true,s,w);pin_row(12,true,s,w);}
      if(cpu_selected){pin_row(13,false,s,w);pin_row(13,false,s,w);}
      assert((group.touched==std::vector<int>{6,7,7}));
      assert(times[10]==epoch && times[11]==epoch);
      assert(counters[NewCount]==int(fresh));
      assert(counters[PromoteCount]==int(cpu_current)+int(cpu_selected));
      if(cpu_current)assert(promote_rows[0]==-13);
      const int count=int(fresh)+int(cpu_current)+int(cpu_selected);
      const auto marks=group.touched;
      for(int lane=0;lane<256;++lane){threadIdx.x=lane;recent::Storage<256> scratch;metadata_finish_body(s,w,scratch);}
      const int used=min(count,available),victims=count-used;
      assert(epoch==42 && group.touched==marks && group.finish_calls==int(count!=0));
      assert(counters[PlanCount]==count && counters[FreeUsed]==used);
      assert(counters[VictimCount]==victims && counters[Candidates]==victims);
      assert(counters[LeaseCount]==0 && counters[PendingReserve]==0);
      assert(free_counts[0]==available-used && prefix[2]==victims);
      for(int i=0;i<8;++i){assert(leases[i]==7+i);assert(leased[i]==31+i);}
      // With no incoming records, no plan bytes or GPU/CPU mappings change.
      if(!count){for(int v:plan)assert(v==-777);assert(directory[10]==6 && directory[11]==7);}
      if(fresh)assert(plan[0]==14 && plan[16]==-1);
      if(cpu_current){int i=int(fresh);assert(plan[i]==12 && plan[16+i]==-2);}
      if(cpu_selected){int i=int(fresh)+int(cpu_current);assert(plan[i]==13 && plan[16+i]==1);}
    }
}
'''
    run_cpp(tmp_path,harness)


def test_native_producers_and_ready_payload_boundaries_are_not_rewritten():
    """The fast finish is downstream of original native producer launches."""
    parallel = cpp_function((KERNELS/'residency_parallel.cuh').read_text(),
        'flowkv_host_prepare_parallel')
    batched = cpp_function((KERNELS/'batched_prepare.cuh').read_text(),
        'flowkv_host_prepare_after_begin')
    for body in (parallel,batched):
        assert body.index('pin_rows<<<') < body.index('finish_metadata(')
    assert parallel.index('begin_current<<<') < parallel.index('pin_rows<<<')
    finish = cpp_function((KERNELS/'residency_finish.cuh').read_text(),'finish_metadata')
    low = finish.split('if (queries > 0 && queries < 8) {',1)[1].split('return;',1)[0]
    assert 'base::metadata_finish<<<1, 256, 0, stream>>>(s, w)' in low
    assert 'cudaGetLastError()' in low
    assert not any(word in low for word in ('publish','exchange','commit','epoch','memcpy','synchronize'))
