"""Public TP clear must discard local occupancy while preserving IPC epochs."""
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')

from ulacache.cache.host import FlowKVHostTier
from ulacache.cache.directory import create_host_tier_state
from ulacache.cache.layers import LayerGroupPlan
from ulacache.cache.prefetch import SharedIndexLifecycle
from ulacache.cache.runtime import SlotManager
from ulacache.cache.tensor_parallel import NativeTensorParallelExecution
from ulacache.client import UnifiedMemoryCache
from ulacache.cache.attention_buffers import FlowKVC4Arena
from ulacache.storage.workspace import FlowKVLayout


def make_pool(monkeypatch, *, cpu_offload, prefetch=True, all_groups=False):
    owners = tuple(max(a for a in [0,1]+list(range(5,78,4)) if a <= layer)
                   for layer in range(78)) if all_groups else (0,1,1)
    groups = LayerGroupPlan.from_layer_anchors(owners)
    layout = FlowKVLayout(layers=len(owners),groups=groups.group_count,max_batch=1,
        pool_rows=512,row_bytes=656,disaggregation_mode='decode',topk=256,
        max_query_tokens=1,history_slots=256,page_size=64,
        native_segment_bytes=(656,),layer_anchors=owners)
    groups = layout.layer_groups
    arena = FlowKVC4Arena(layout,device=torch.device('cpu'),
        bytes_per_page=64*656,layer_groups=groups)
    for i,(name,tensor) in enumerate(vars(arena).items()):
        if isinstance(tensor,torch.Tensor):tensor.fill_(i+9)
    arena.cursors.fill_(layout.capacity)
    arena.step.fill_(12883)
    pool = NS(device=torch.device('cpu'),flowkv_arena=arena,flowkv_layout=layout,
        flowkv_step_publication=NS(ready=torch.tensor([12883])),flowkv_host=None)
    events = []
    lifecycle = object.__new__(SharedIndexLifecycle)
    lifecycle.pool,lifecycle.groups,lifecycle._planned = pool,groups,None
    lifecycle.prefetch,lifecycle._stream = prefetch,'existing_prefetch_stream'
    pool.flowkv_shared_index = lifecycle
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:
        NS(wait_stream=lambda stream:events.append(('join',stream))))
    if cpu_offload:
        tier = object.__new__(FlowKVHostTier)
        tier.pool,tier.groups = pool,groups
        tier.states = tuple(create_host_tier_state(4,4,'cpu') for _ in groups.anchors)
        for state in tier.states:
            state.epoch.fill_(12883);state.directory.fill_(4)
        tier.rings = tuple(range(groups.group_count))
        tier.ready = torch.full((groups.group_count,2),12883,dtype=torch.int64)
        tier.miss_ready = torch.full((groups.group_count,),12883,dtype=torch.int64)
        tier.ready_peers = NS(ptrs='original_peer_flags')
        tier._reserved_native_rows = torch.tensor([0,8])
        import ulacache.cache.kernels.cpu_residency as kernel
        monkeypatch.setattr(kernel,'drain_host_tier_readers',lambda state,ptrs,offset:
            events.append(('drain',offset)))
        monkeypatch.setattr(kernel,'clear_host_tier_ring',lambda ring:events.append(('ring',ring)))
        def padding(rows):
            assert rows is tier._reserved_native_rows
            assert bool((arena.cursors==0).all()) and bool((arena.tags==-1).all())
            events.append(('padding',))
        tier.allocate = padding
        pool.flowkv_host = tier
    return pool,events


def public_handle(pool,events):
    backend = object.__new__(NativeTensorParallelExecution);backend.pool=pool
    access = NS(require_idle=lambda operation:events.append(('public_idle',operation)),
        reset=lambda:events.append(('public_reset',)))
    cache = object.__new__(UnifiedMemoryCache)
    cache._execution_engine=backend
    cache._slot_manager=SlotManager(backend,None,access)
    return cache


@pytest.mark.parametrize('cpu_offload',[False,True])
@pytest.mark.parametrize('prefetch',[False,True])
def test_public_clear_resets_occupancy_without_changing_payload_epochs_or_pointers(monkeypatch,cpu_offload,prefetch):
    pool,events=make_pool(monkeypatch,cpu_offload=cpu_offload,prefetch=prefetch)
    arena=pool.flowkv_arena
    before={name:(tensor,tensor.data_ptr(),tensor.clone()) for name,tensor in vars(arena).items()
        if isinstance(tensor,torch.Tensor)}
    ready=pool.flowkv_step_publication.ready.clone()
    if cpu_offload:
        host=pool.flowkv_host;host_ready=host.ready.clone();miss_ready=host.miss_ready.clone()
    public_handle(pool,events).clear_registered_records()
    for name,(tensor,pointer,value) in before.items():
        assert getattr(arena,name) is tensor and tensor.data_ptr()==pointer
        if name in ('hints','tags'):assert bool((tensor==-1).all())
        elif name in ('pins','cursors'):assert bool((tensor==0).all())
        else:assert torch.equal(tensor,value),name
    assert torch.equal(pool.flowkv_step_publication.ready,ready)
    if prefetch:assert events[1]==('join','existing_prefetch_stream')
    if cpu_offload:
        assert torch.equal(host.ready,host_ready) and torch.equal(host.miss_ready,miss_ready)
        assert all(state.epoch.item()==12883 for state in host.states)
        assert [e for e in events if e[0]=='drain']==[('drain',2*g+1) for g in range(host.groups.group_count)]
        assert events[-2:]==[('padding',),('public_reset',)]
    assert arena.groups[1] is arena.groups[2]


def test_all21_group_views_observe_the_same_reset(monkeypatch):
    pool,_=make_pool(monkeypatch,cpu_offload=False,all_groups=True)
    NativeTensorParallelExecution.clear(NS(pool=pool))
    assert len({id(v) for v in pool.flowkv_arena.groups.values()})==21
    for state in pool.flowkv_arena.groups.values():
        assert state.cursors.tolist()==[0,0]
        assert bool((state.tags==-1).all()) and bool((state.pins==0).all())


@pytest.mark.parametrize('cpu_offload',[False,True])
@pytest.mark.parametrize('invalid',['in_forward','capturing','wrong_lifecycle'])
def test_unsafe_clear_fails_before_any_mutation(monkeypatch,cpu_offload,invalid):
    pool,events=make_pool(monkeypatch,cpu_offload=cpu_offload)
    before=pool.flowkv_arena.cursors.clone()
    if invalid=='in_forward':pool.flowkv_shared_index._planned=set()
    elif invalid=='capturing':
        pool.device=torch.device('cuda')
        monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
    else:pool.flowkv_shared_index.pool=object()
    with pytest.raises((RuntimeError,ValueError)):
        NativeTensorParallelExecution.clear(NS(pool=pool))
    assert torch.equal(pool.flowkv_arena.cursors,before) and events==[]
    if cpu_offload:assert all(bool((s.directory==4).all()) for s in pool.flowkv_host.states)


def test_arena_requires_explicit_retired_readers(monkeypatch):
    pool,_=make_pool(monkeypatch,cpu_offload=False)
    with pytest.raises(RuntimeError,match='Retire every resident reader'):
        pool.flowkv_arena.clear_local_residency(readers_retired=False)


@pytest.mark.parametrize('cpu_offload',[False,True])
def test_repeated_public_clear_is_idempotent(monkeypatch,cpu_offload):
    pool,events=make_pool(monkeypatch,cpu_offload=cpu_offload)
    cache=public_handle(pool,events)
    cache.clear_registered_records()
    before={name:(tensor.data_ptr(),tensor.clone()) for name,tensor in vars(pool.flowkv_arena).items()
        if isinstance(tensor,torch.Tensor)}
    cache.clear_registered_records()
    for name,(pointer,value) in before.items():
        tensor=getattr(pool.flowkv_arena,name)
        assert tensor.data_ptr()==pointer and torch.equal(tensor,value)
