"""Native callbacks must reuse the ordinary TP lifecycle and encoded writes."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from ulacache import (KeyValueRecordPlane,KeyValueStorageLayout,LayerReuseSpecification,
    MemoryTierCapacity,SparseAccessSpecification,UnifiedCacheSpecification)
from ulacache.cache.native_shared_tp import NativeSharedTensorParallelExecution,create_native_shared_tp_execution
from ulacache.cache.planning import create_paged_cache_execution_plan
from ulacache.storage.writers import NativeRecordWriteStaging


def specification(queries=8):
    return UnifiedCacheSpecification(KeyValueStorageLayout('native',(
        KeyValueRecordPlane('kv','mla.bfloat16.v1',(1024,)),),256),
        LayerReuseSpecification.from_layer_groups(2),SparseAccessSpecification(queries,2051),
        MemoryTierCapacity(4096,32768,1024,reserved_logical_record_count=32))


@pytest.mark.parametrize('queries',[1,8,9,64,192])
def test_factory_binds_original_tp_and_preserves_canonical_capacity(queries,monkeypatch):
    import ulacache
    import ulacache.compiler
    import ulacache.cache.tp_cache
    import ulacache.cache.prefetch
    observations={}
    class StopAfterBinding(Exception):pass
    def configure(pool,services,*,step_publication):
        observations.update(pool=pool,services=services,published=step_publication)
        pool.flowkv_prefetch_policy=services.config.prefetch
    def cache(spec,policy):
        observations.update(spec=spec,policy=policy)
        raise StopAfterBinding
    monkeypatch.setattr(ulacache,'UnifiedMemoryCache',cache)
    monkeypatch.setattr(ulacache.compiler,'configure_compiler',lambda compiler:None)
    monkeypatch.setattr(ulacache.cache.tp_cache,'configure_pool',configure)
    monkeypatch.setattr(ulacache.cache.prefetch,'SharedIndexLifecycle',lambda *a,**k:NS(in_forward=False))
    with pytest.raises(StopAfterBinding):
        create_native_shared_tp_execution(specification(queries),None,device='cpu',dtype=torch.bfloat16,
            group=NS(rank_in_group=0,cpu_group=None,device_group=None),degree=8,compiler=Mock(),
            host_factory=Mock(),ipc_factory=Mock(),memory_region=nullcontext,prefill_records=1048576)
    original=specification(queries)
    bound=observations['spec'];layout=observations['pool'].flowkv_layout
    assert observations['policy'].execution_options.record_storage_backend=='tensor_parallel'
    assert observations['published'] is True and observations['services'].config.prefetch.enabled
    assert bound.capacity_plan.device_record_capacity==original.capacity_plan.device_record_capacity
    assert bound.capacity_plan.host_record_capacity==original.capacity_plan.host_record_capacity
    assert layout.max_resident_queries==queries and layout.max_batch==queries
    observations['pool'].flowkv_shards=torch.empty(2,4096,1024,device='meta',dtype=torch.uint8)
    assert observations['pool']._flowkv_peer_offset(1)==4096*1024
    assert layout.history_slots==queries*2304*3
    assert bound.sparse_access.maximum_selected_record_count==2304
    assert bound.sparse_access.maximum_padded_query_count==1<<(queries-1).bit_length()


def fixture():
    spec=specification()
    plan=create_paged_cache_execution_plan(spec)
    def native(values,pages,indices):
        # Make the encoded staging visibly different from the incoming values.
        pages.view(torch.bfloat16).reshape(-1,1,512)[indices.long()]=values[0]+2
    staging=NativeRecordWriteStaging(spec,plan,'cpu',native)
    arena=NS(valid=torch.zeros(1,dtype=torch.int32),kernel=lambda:NS(
        globalize_current=lambda source,out,base:out.copy_(source.int())))
    pool=NS(flowkv_arena=arena,flowkv_shared_index=NS(in_forward=False),
        flowkv_layout=NS(layer_groups=NS(group=lambda layer:layer,anchors=(0,1)),topk=2304),
        _flowkv_global_current=torch.empty(8,dtype=torch.int32),_flowkv_owner_start=0,
        flowkv_step_publication=NS(begin=Mock()))
    cache=Mock()
    cache.access.return_value=NS(slot_ids=torch.arange(2304,dtype=torch.int32).reshape(1,-1))
    execution=NativeSharedTensorParallelExecution(cache,pool,staging,spec,2051,torch.bfloat16)
    execution.prefill_buffers=NS(begin_forward=Mock(),select=lambda layer,ids:ids)
    return execution,staging,cache,pool


def test_decode_preserves_native_selection_and_encoded_bytes_at_original_boundaries():
    execution,staging,cache,pool=fixture()
    ids=torch.tensor([257],dtype=torch.int64)
    values=torch.arange(512,dtype=torch.bfloat16).reshape(1,1,-1)
    execution.begin_forward(decode_like=True,active_query_counts=None,
        attention_data_parallel_rank=0,query_count=1,current_logical_record_ids=ids)
    staging.stage_native_records(0,ids,(values,None))
    execution.publish_after_native_write(0)
    cache.write.assert_not_called()
    selected=torch.arange(256,2307,dtype=torch.int32).reshape(1,-1)
    actual=execution.prepare_attention(0,selected)
    assert actual.shape==selected.shape and actual.tolist()==[list(range(2051))]
    padded=cache.prepare_sparse_attention.call_args.args[1].selected_logical_record_ids
    assert torch.equal(padded[:,:2051],selected) and torch.all(padded[:,2051:]==-1)
    written_ids,layer,encoded=cache.write.call_args.args
    assert written_ids is ids and layer==0 and torch.equal(encoded[0],values+2)
    assert len(encoded)==1 and staging.pending_logical_record_ids(0) is None
    calls=[call[0] for call in cache.mock_calls]
    assert calls.index('prepare_sparse_attention')<calls.index('access')<calls.index('write')
    execution.finish_attention(0)
    cache.finish_layer_attention.assert_called_once_with(0)


def test_prefill_publishes_before_reading_without_starting_decode_lru():
    execution,staging,cache,pool=fixture()
    ids=torch.tensor([257,258],dtype=torch.int64)
    values=torch.ones(2,1,512,dtype=torch.bfloat16)
    execution.begin_forward(decode_like=False,active_query_counts=None,
        attention_data_parallel_rank=0,query_count=2,current_logical_record_ids=ids,
        prefill_request_query_counts=[2])
    staging.stage_native_records(0,ids,(values,None))
    execution.publish_after_native_write(0)
    assert torch.equal(cache.write.call_args.args[2][0],values+2)
    cache.begin_model_forward.assert_not_called()
    pool.flowkv_step_publication.begin.assert_not_called()


def test_index_ready_prefetch_precedes_native_write_and_is_planned_only_once():
    execution,staging,cache,pool=fixture()
    ids=torch.tensor([257],dtype=torch.int64)
    selected=torch.arange(256,2307,dtype=torch.int32).reshape(1,-1)
    execution.begin_forward(decode_like=True,active_query_counts=None,
        attention_data_parallel_rank=0,query_count=1,current_logical_record_ids=ids)
    execution.prefetch_attention(0,selected)
    cache.prepare_sparse_attention.assert_called_once()
    cache.access.assert_not_called()
    cache.write.assert_not_called()
    assert staging.pending_logical_record_ids(0) is None
    staging.stage_native_records(0,ids,(torch.ones(1,1,512,dtype=torch.bfloat16),None))
    actual=execution.prepare_attention(0,selected)
    assert actual.shape==selected.shape
    cache.prepare_sparse_attention.assert_called_once()
    cache.access.assert_called_once()
    cache.write.assert_called_once()


def test_declared_encoded_write_uses_native_tp_writer_after_history_completion():
    execution,staging,cache,pool=fixture()
    ids=torch.tensor([257],dtype=torch.int64)
    selected=torch.arange(256,2307,dtype=torch.int32).reshape(1,-1)
    values=torch.ones(1,1,512,dtype=torch.bfloat16)
    execution.begin_forward(decode_like=True,active_query_counts=None,
        attention_data_parallel_rank=0,query_count=1,current_logical_record_ids=ids)
    assert execution.write_encoded_records(0,ids,values) is False
    execution.prefetch_attention(0,selected)
    assert execution.write_encoded_records(0,ids,values) is True
    assert staging.pending_logical_record_ids(0) is None
    assert cache.write.call_args.args[2][0] is values
    assert [call[0] for call in cache.mock_calls].index('access') < [call[0] for call in cache.mock_calls].index('write')
    execution.prepare_attention(0,selected)
    cache.access.assert_called_once()
    with pytest.raises(RuntimeError,match='Consume'):
        execution.write_encoded_records(0,ids,values)


def test_native_tail_pad_accepts_borrowed_remap_without_an_intermediate_copy():
    execution,staging,cache,pool=fixture()
    ids=torch.arange(257,265,dtype=torch.int64)
    selected=torch.arange(8*2051,dtype=torch.int32).reshape(8,-1)
    slots=torch.arange(8*2304,dtype=torch.int32).reshape(8,-1)
    cache.access.return_value=NS(slot_ids=slots)
    execution.begin_forward(decode_like=True,active_query_counts=None,
        attention_data_parallel_rank=0,query_count=8,current_logical_record_ids=ids)
    execution.prefetch_attention(0,selected)
    execution.write_encoded_records(0,ids,torch.ones(8,1,512,dtype=torch.bfloat16))
    result=execution.prepare_attention(0,selected)
    assert result.data_ptr()==slots.data_ptr() and result.stride(0)==2304
    assert torch.equal(result,slots[:,:2051])


def test_public_write_accepts_one_nope_segment_and_keeps_internal_native_writer_abi(monkeypatch):
    from ulacache import UnifiedMemoryCache
    from ulacache.cache.tensor_parallel import NativeTensorParallelExecution
    import ulacache.storage.writers

    backend=NativeTensorParallelExecution.__new__(NativeTensorParallelExecution)
    backend.pool=NS()
    cache=UnifiedMemoryCache.__new__(UnifiedMemoryCache)
    cache.cache_specification=specification()
    cache._execution_engine=backend
    writer=Mock()
    monkeypatch.setattr(ulacache.storage.writers,'write_native_record_segments',writer)
    ids=torch.tensor([257,258]);values=torch.randn(2,1,512,dtype=torch.bfloat16)
    cache.write(ids,0,(values,))
    native=writer.call_args.args[3]
    assert native[0] is values and native[1].shape==(2,1,0)
    assert native[1].dtype==values.dtype and native[1].device==values.device
