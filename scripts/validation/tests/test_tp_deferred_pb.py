"""Real entry routing and scheduling tests; CUDA memory ordering remains gated."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace as NS

import pytest
from test_low_query_residency_finish import cpp_function

ROOT=Path(__file__).resolve().parents[3]

@pytest.fixture(scope='module')
def actual_route_binary(tmp_path_factory):
    compiler=shutil.which('c++')
    if compiler is None:pytest.skip('A C++ compiler is required')
    source=(ROOT/'ulacache/csrc/cache/residency_fused.cuh').read_text()
    fields=source.split('enum FrameField {',1)[1].split('};',1)[0]
    functions='\n'.join(cpp_function(source,n) for n in ('defer_gpu_bookkeeping',
        'prepare_actual_misses_and_finish','finish_deferred_gpu_bookkeeping'))
    text=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#define __device__
#define __forceinline__ inline
#define __global__
struct Dim { int x=0; } threadIdx,blockIdx;
enum Counter { NewCount,PromoteCount,PlanCount };
enum FrameField { FIELDS };
struct State { int* free_counts; int host_slots; };
struct Work { int* counters; };
struct Frame {
 const int64_t* d;
 State state()const{return {reinterpret_cast<int*>(d[FFreeCounts]),int(d[FHostSlots])};}
 Work work()const{return {reinterpret_cast<int*>(d[FCounters])};}
};
int completions=0;
void finish_actual_miss_producers(const int64_t*,const int64_t*,const int* active,int q,int tiles){
 assert(q>0&&q<8&&tiles==4&&*active>=0&&*active<=q);++completions;
}
FUNCTIONS
int main(int argc,char** argv){
 int cpu=std::atoi(argv[1]),fresh=std::atoi(argv[2]),enabled=std::atoi(argv[3]),q=std::atoi(argv[4]);
 for(int active:{0,q/2,q}){
  int counts[2]{7,16-cpu},counters[8]{fresh,0,0};int64_t epoch=41,route=-999;
  int64_t d[FFrameWords]{};
  d[FFreeCounts]=reinterpret_cast<int64_t>(counts);d[FHostSlots]=16;
  d[FCounters]=reinterpret_cast<int64_t>(counters);d[FDeferredEpoch]=enabled?reinterpret_cast<int64_t>(&route):0;
  completions=0;prepare_actual_misses_and_finish(d,&epoch,&active,q,4);
  bool fast=cpu==0&&fresh==0&&enabled;
  assert(completions==(fast?0:1));assert(!enabled||route==(fast?epoch:0));
  // The last CPU row can disappear after the unchanged child commit. The
  // late dispatcher must honor the captured route, not recompute eligibility.
  counts[1]=16;counters[NewCount]=0;
  if(argc>5 && fast)counts[1]=15; // Illegal owner mutation must fail closed.
  finish_deferred_gpu_bookkeeping(d,&epoch,&active,q,4);
  assert(completions==1);
  // Epoch tags cannot authorize a stale call in the following transaction.
  ++epoch;finish_deferred_gpu_bookkeeping(d,&epoch,&active,q,4);assert(completions==1);
 }
}
'''.replace('FIELDS',fields).replace('FUNCTIONS',functions)
    # initializer_list is needed for the three controlled active shapes.
    text='#include <initializer_list>\n'+text
    root=tmp_path_factory.mktemp('deferred_actual_route');p=root/'route.cpp';p.write_text(text)
    binary=root/'route'
    subprocess.run([compiler,'-std=c++17',str(p),'-o',str(binary)],check=True,capture_output=True)
    return binary

@pytest.mark.parametrize('q',[1,3,6,7])
@pytest.mark.parametrize('cpu',[0,1])
@pytest.mark.parametrize('fresh',[0,1])
@pytest.mark.parametrize('enabled',[0,1])
def test_actual_early_late_entries_bind_epoch_route(actual_route_binary,q,cpu,fresh,enabled):
    subprocess.run([str(actual_route_binary),str(cpu),str(fresh),str(enabled),str(q)],
        check=True,capture_output=True,timeout=5)

def test_payload_event_precedes_late_work_and_all_work_precedes_publication(monkeypatch):
    import torch
    from ulacache.cache.prefetch import SharedIndexLifecycle
    from ulacache.cache.layers import LayerGroupPlan
    log=[]
    class Stream:
        def __init__(self,name):self.name=name
        def wait_stream(self,other):log.append((self.name,'join',other.name))
        def wait_event(self,event):log.append((self.name,'wait_payload'))
    class Event:
        def record(self,stream):log.append(('payload_event',stream.name))
    main,copy=Stream('main'),Stream('copy');current=[main]
    @contextmanager
    def on_stream(stream):
        old=current[0];current[0]=stream
        try:yield
        finally:current[0]=old
    monkeypatch.setattr(torch.cuda,'Stream',lambda **kw:copy)
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *a:current[0])
    monkeypatch.setattr(torch.cuda,'stream',on_stream)
    host=NS(deferred_epochs=object(),finish_deferred_group=lambda layer,**kw:(log.append(('late',layer,current[0].name)),True)[1])
    pool=NS(device='cuda:0',flowkv_host=host,
        flowkv_step_publication=NS(publish=lambda:log.append('publish')))
    # Final anchor has no followers: it still requires the last ordinary join.
    life=SharedIndexLifecycle(LayerGroupPlan.from_layer_anchors((0,0,2)),pool,prefetch=True)
    life._plan_index=lambda layer,ids:life._planned.add(life.groups.group(layer))
    life._pull_history=lambda layer:log.append(('pull',layer,current[0].name))
    ids=NS(shape=(3,2048));life.begin_forward()
    for layer in range(3):
        life.before_layer_prepare(layer)
        if layer in (0,2):life.on_index_ready(layer,ids)
        life.before_history_consume(layer);log.append(('writer_attention',layer))
        life.after_attention(layer)
    life.finish_forward()
    for layer in (0,2):
        pull=log.index(('pull',layer,'copy'));late=log.index(('late',layer,'copy'))
        assert log[pull+1]==('payload_event','copy') and pull+1<late
        assert late<log.index(('writer_attention',layer))
    assert log[-2:]==[('main','join','copy'),'publish']
    # Out-of-forward clear/free uses the same existing stream join, rather
    # than treating an early payload event as all-bookkeeping completion.
    life.join_idle_prefetch();assert log[-1]==('main','join','copy')

@pytest.mark.parametrize('q',[0,8,24])
def test_actual_host_method_retains_non_low_q_path(q):
    from ulacache.cache.host import FlowKVHostTier
    host=object.__new__(FlowKVHostTier);host.deferred_epochs=object()
    assert host.finish_deferred_group(0,queries=q) is False

def test_actual_begin_waits_all_lanes_before_clearing_peer_inputs(tmp_path):
    """Two real CPU threads expose the previous nonzero-lane race."""
    compiler=shutil.which('c++')
    if compiler is None:pytest.skip('A C++ compiler is required')
    source=(ROOT/'ulacache/csrc/cache/common.cuh').read_text()
    body=cpp_function(source,'begin_global_step')
    barrier='if constexpr (StepPublication) __syncthreads();'
    assert body.count(barrier)==1
    prelude=r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
using std::min;using std::max;
#define __global__
#define FLOW_RECORD_HOST_TIER 1
#define FLOW_ZERO_SENTINEL 1
constexpr int HASH=64;
struct Dim{int x;};thread_local Dim threadIdx;Dim blockDim{64};
std::atomic<bool> waiting{false},other_at_barrier{false};
std::mutex m;std::condition_variable cv;int arrivals=0,generation=0;
void __syncthreads(){
 if(threadIdx.x==32)other_at_barrier=true;
 std::unique_lock<std::mutex> l(m);int prior=generation;
 if(++arrivals==2){arrivals=0;++generation;cv.notify_all();}
 else cv.wait(l,[&]{return generation!=prior;});
}
namespace cuda{
enum{thread_scope_system,thread_scope_device,memory_order_acquire,memory_order_relaxed};
template<class T,int Scope>struct atomic_ref{
 T& v;atomic_ref(T&x):v(x){}
 T load(int){waiting=true;return __atomic_load_n(&v,__ATOMIC_ACQUIRE);}
 void store(T x,int){__atomic_store_n(&v,x,__ATOMIC_RELEASE);}
};}
int read_int(const void*p,int i,int){return ((const int*)p)[i];}
unsigned hash_row(int x){return unsigned(x)%HASH;}
int atomicCAS(int*p,int old,int n){__atomic_compare_exchange_n(p,&old,n,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST);return old;}
namespace sglang::flowkv_host { void begin_groups(const int64_t*,int,const void*,int,int){} }
template<bool FromNative=false,bool StepPublication=false>
BODY
int main(){
 int64_t step=9,flags[8];uint64_t peers[8];
 for(int i=0;i<8;++i){flags[i]=i==7?8:9;peers[i]=uint64_t(flags+i);}
 int active=0,keys[64],values[64],totals[4],hints[1],counts[64],hist[516];
 for(int&n:counts)n=7;
 int64_t cursors[4]{1,2,3,4};
 auto run=[&](int lane){threadIdx.x=lane;begin_global_step<false,true>(
   nullptr,4,nullptr,4,0,32,2,0,&step,&active,keys,values,cursors,totals,hints,counts,hist,
   nullptr,0,0,1,peers,8);};
 std::thread zero(run,0),one(run,32); // Representative lanes from different warps.
 while(!waiting.load()||!other_at_barrier.load())std::this_thread::yield();
 // The nonzero lane is held at its first actual CTA barrier. In the old
 // body that barrier was after the destructive count stores.
 assert(counts[32]==EXPECTED_BEFORE);
 __atomic_store_n(flags+7,int64_t(9),__ATOMIC_RELEASE);
 zero.join();one.join();assert(step==10);
 assert(counts[0]==0&&counts[32]==0);
}
'''
    for old in (False,True):
        code=prelude.replace('BODY',body.replace(barrier,'') if old else body)
        code=code.replace('EXPECTED_BEFORE','0' if old else '7')
        p=tmp_path/f'begin_{old}.cpp';p.write_text(code);binary=p.with_suffix('')
        subprocess.run([compiler,'-std=c++17','-pthread',str(p),'-o',str(binary)],
            check=True,capture_output=True,timeout=30)
        subprocess.run([str(binary)],check=True,capture_output=True,timeout=5)


def test_captured_fast_route_rejects_intervening_owner_mutation(actual_route_binary):
    p=subprocess.run([str(actual_route_binary),'0','0','1','3','mutated'],
        capture_output=True,timeout=5)
    assert p.returncode != 0

@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('policy',['pb','naive'])
def test_actual_constructor_keeps_route_storage_private_per_group(monkeypatch,enabled,policy):
    import torch
    import ulacache.cache.host as module
    import ulacache.eviction.lru as lru
    import ulacache.transfer.kernels.device_graph as graph
    import ulacache.compiler as compiler
    from ulacache.cache.layers import LayerGroupPlan
    from ulacache.cache_config import PrefetchPolicy
    class Peer:
        def __init__(self,tensor,*args,**kwargs):self.device_ptr=tensor.data_ptr();self.ptrs=tensor
    class Host:
        def __init__(self,*,layers,slots,row_bytes,device):
            self.storage=torch.empty((layers,slots,row_bytes),dtype=torch.uint8,device=device)
            self.ptrs=torch.tensor([self.storage.data_ptr()],dtype=torch.int64)
    monkeypatch.setattr(module,'FlowKVPeerShards',Peer)
    monkeypatch.setattr(lru,'create_presence_ring',lambda *args:NS(view=torch.zeros(11,dtype=torch.int64)))
    monkeypatch.setattr(graph,'device_graph_module',lambda **kwargs:NS(create=lambda *args:99))
    monkeypatch.setattr(compiler,'_recency_policy',None)
    monkeypatch.setenv('UDCACHE_RECENCY_POLICY',policy)
    groups=LayerGroupPlan.from_layer_anchors((0,1,1))
    pool=NS(device=torch.device('cpu'),_flowkv_global_rows=96,_flowkv_storage_size=8,
        allocation_page_size=64,flowkv_rank=0,flowkv_shards=torch.empty((3,8,656),dtype=torch.uint8),
        udcache_services=NS(config=NS(prefetch=PrefetchPolicy(enabled))),
        flowkv_layout=NS(layer_groups=groups,max_query_tokens=8,topk=32,layers=3,row_bytes=656),
        flowkv_arena=NS(pull_rows=torch.zeros((2,8,32),dtype=torch.int32),counts=torch.zeros((2,8),dtype=torch.int32)))
    tier=module.FlowKVHostTier(pool,4,None,host_pool_factory=Host,ipc_manager_factory=None)
    assert tier.fused_metadata.shape==(2,31)
    if enabled and policy=='pb':
        assert tier.deferred_epochs.tolist()==[0,0]
        assert tier.fused_metadata[:,-1].tolist()==[tier.deferred_epochs[g:g+1].data_ptr() for g in range(2)]
    else:
        assert tier.deferred_epochs is None and tier.fused_metadata[:,-1].tolist()==[0,0]
    # A follower shares its anchor's existing state; no second allocator/bank.
    assert tier.descriptors[1][6].item()==tier.descriptors[2][6].item()==1

def test_actual_dpa_frame_builder_keeps_deferred_route_disabled():
    """Execute the actual DPA registration-frame assignment with real tensors."""
    import ast
    import torch
    from ulacache.cache.directory import create_host_tier_state
    from ulacache.cache.host import create_host_tier_workspaces
    path=ROOT/'ulacache/cache/local_paged_cache.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativePagedResidency')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    node=next(n for n in init.body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Attribute) and t.attr=='registration_frames' for t in n.targets))
    self=NS(residency_states=tuple(create_host_tier_state(8,4,'cpu') for _ in range(2)),
        transfer_workspaces=create_host_tier_workspaces(97,2,8,'cpu'),
        maximum_query_count=8,maximum_selection_width=32,device=torch.device('cpu'))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),
        dict(self=self,torch=torch,capacity=NS(device_record_capacity=8,host_record_capacity=4),
             workspace_capacity=97))
    assert self.registration_frames.shape==(2,31)
    assert self.registration_frames[:,-1].tolist()==[0,0]
    for group,row in enumerate(self.registration_frames):
        assert row[:17].tolist()==[t.data_ptr() for t in
            (*self.residency_states[group],*self.transfer_workspaces[group])]
