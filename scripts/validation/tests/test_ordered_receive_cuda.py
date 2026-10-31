"""Native ownership/byte gate for ordering only unpublished CPU receive slots."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import pytest
torch=pytest.importorskip('torch')


def permute_free_cpu_slots(state):
    """Shuffle the actual free extent; occupied slots may have arbitrary IDs."""
    count = int(state.free_counts[1])
    original = state.host_free[:count].clone()
    permutation = torch.randperm(count, device=state.host_free.device)
    state.host_free[:count].copy_(original[permutation])
    assert torch.equal(torch.sort(state.host_free[:count]).values,
                       torch.sort(original).values)


def test_fragmented_fixture_preserves_the_actual_nonzero_free_slot_set():
    # Receive ordering may allocate the low slots first. The remaining free
    # slots then start at64; replacing them with randperm(448) corrupts owners.
    state = SimpleNamespace(host_free=torch.arange(511, -1, -1, dtype=torch.int32),
        free_counts=torch.tensor([512, 448], dtype=torch.int32))
    before = state.host_free.clone()
    permute_free_cpu_slots(state)
    assert torch.equal(torch.sort(state.host_free[:448]).values,
                       torch.arange(64, 512, dtype=torch.int32))
    assert torch.equal(state.host_free[448:], before[448:])

def main(output):
    from ulacache.cache.local_cache import UnifiedLocalResidency
    from ulacache.cache.layers import LayerGroupPlan
    from ulacache.compiler import configure_compiler
    from ulacache.transfer.direct_receiver import DirectRecordReceiver
    from sglang.kernels.jit.utils import load_jit
    from sglang.srt.mem_cache.flowkv_host_pool import FlowKVHostRecordPool
    configure_compiler(load_jit)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    assert torch.cuda.mem_get_info()[0]>600_000_000
    groups=LayerGroupPlan.from_layer_anchors([0,0,2]);width=656
    engine=UnifiedLocalResidency(groups=groups,gpu_slots=512,host_slots=512,
        row_bytes=width,max_queries=2,topk=4,max_write_rows=256,device=device,
        host_pool_factory=FlowKVHostRecordPool,parallel_prepare=True,
        parallel_victims=True,async_followers=True,exchange_launch=(64,256),
        follower_exchange_launch=(64,256),follower_copy_mode='layer')
    # The native engine and ownership tables are real; this thin facade supplies
    # only the two framework-independent fields consumed by the receiver.
    facade=SimpleNamespace(_execution_engine=engine,cache_specification=
        SimpleNamespace(storage_layout=SimpleNamespace(records_per_page=64)))
    receiver=DirectRecordReceiver(facade)
    engine.host.storage.fill_(0xCD);engine.gpu.fill_(0xCD)
    ids=lambda first,last:torch.arange(first,last,dtype=torch.int64,device=device)
    prefix=ids(64,128);incoming=ids(128,640);hold=ids(640,768)
    engine.allocate_receive_rows(prefix)
    before_prefix=engine.directory[:,prefix].clone()
    # Any permutation of free entries is a valid native free-stack state.
    for group,state in enumerate(engine.states):
        torch.manual_seed(20261027+group)
        permute_free_cpu_slots(state)
    engine.allocate_receive_rows(incoming)
    before=engine.directory[:,incoming].clone()
    free_before=[s.free_counts.clone() for s in engine.states]
    gpu_before=engine.gpu.clone();cpu_before=engine.host.storage.clone()
    plan=receiver.reserve('abort-room',incoming[:495],prefix_records=64)
    after=torch.tensor(plan.locations,device=device)
    assert torch.equal(engine.directory[:,incoming[:495]],after)
    assert torch.equal(engine.directory[:,incoming[495:]],before[:,495:])
    assert torch.equal(engine.directory[:,prefix],before_prefix)
    assert torch.equal(engine.gpu,gpu_before) and torch.equal(engine.host.storage,cpu_before)
    del gpu_before,cpu_before
    for group,state in enumerate(engine.states):
        old=before[group,:495];new=after[group];hot=old>=0
        assert torch.equal(new[hot],old[hot])
        assert torch.equal(torch.sort(old[~hot]).values,torch.sort(new[~hot]).values)
        assert bool((torch.diff(-new[~hot]-2)>0).all())
        assert torch.equal(state.free_counts,free_before[group])
    engine.allocate_receive_rows(hold)
    second=receiver.reserve('commit-room',hold)
    extra=torch.cat((ids(1,64),ids(768,1024)))
    engine.allocate_unbound(extra,prefer_gpu=True)
    logical=ids(1,1024)

    def payload(rows,layer):
        return ((rows.cpu()[:,None]+13*layer+7*torch.arange(width)[None,:])%251).to(torch.uint8)

    def write_rows(rows):
        directory=engine.directory[:,rows].cpu()
        for layer in range(groups.layer_count):
            loc=directory[groups.group(layer)];hot=loc>=0;values=payload(rows,layer)
            engine.gpu[layer].index_copy_(0,loc[hot].to(device),values[hot].to(device))
            engine.host.storage[layer].index_copy_(0,-loc[~hot]-2,values[~hot])

    def verify_rows(rows):
        directory=engine.directory[:,rows].cpu()
        for layer in range(groups.layer_count):
            loc=directory[groups.group(layer)];hot=loc>=0;actual=torch.empty((len(rows),width),dtype=torch.uint8)
            actual[hot]=engine.gpu[layer][loc[hot].to(device)].cpu()
            actual[~hot]=engine.host.storage[layer][-loc[~hot]-2]
            assert torch.equal(actual,payload(rows,layer)),(layer,rows.cpu().tolist())

    write_rows(logical);torch.cuda.synchronize()
    # Every bound slot, including old records and the unused partial-page tail,
    # has the exact inverse owner after the metadata-only permutation.
    for group,state in enumerate(engine.states):
        loc=engine.directory[group,logical];hot=loc>=0
        assert torch.equal(state.gpu_tags[loc[hot]],logical[hot])
        assert torch.equal(state.host_tags[-loc[~hot]-2],logical[~hot])
    committed=torch.cat((prefix,extra));current=ids(1,3);engine.active.fill_(2)
    def forward(step):
        selected=committed[(torch.arange(8,device=device)*17+step*7)%len(committed)].to(torch.int32).reshape(2,4)
        engine.begin_forward()
        for layer in range(groups.layer_count):
            if layer in groups.anchors:engine.plan(layer,selected,current)
            mapped=engine.attention_rows(layer,2)
            actual=engine.gpu[layer][mapped.long()].cpu().reshape(8,width)
            assert torch.equal(actual,payload(selected.reshape(-1).long(),layer)),(step,layer)
            engine.after_attention(layer)
    started=time.monotonic()
    for step in range(300):
        forward(step)
        if step%50==0:
            engine._drain_transfers();verify_rows(incoming[:495]);verify_rows(hold)
    assert torch.equal(engine.directory[:,incoming[:495]],after)
    pages=sorted(plan.pages) if hasattr(plan,'pages') else sorted(set((plan.logical_rows//64).tolist()))
    try:receiver.logical_pages_freed(pages,transport_quiesced=False)
    except RuntimeError:pass
    else:raise AssertionError('Undrained abort retired the receive')
    receiver.logical_pages_freed(pages,transport_quiesced=True);engine.release(incoming)
    assert bool((engine.directory[:,incoming]==-1).all())
    engine.allocate_receive_rows(incoming)
    reused=receiver.reserve('reused-room',incoming[:495],prefix_records=64)
    write_rows(incoming);receiver.complete('reused-room');receiver.complete('commit-room')
    assert not receiver.metadata_inventory()['pending_rooms']
    assert all(int(state.leases.sum())==0 for state in engine.states)
    committed=logical
    for step in range(300,428):forward(step)
    engine._drain_transfers();verify_rows(logical)
    result=dict(passed=True,metadata_only_ordering=True,physical_slot_sets_preserved=True,
        gpu_destinations_preserved=True,old_prefix_and_unused_tail_preserved=True,
        all_inverse_owners_verified=True,held_receives_verified_over_epochs=300,
        postcommit_replacement_epochs=128,drained_abort_and_reuse=True,
        full_physical_pressure=True,row_bytes=width,async_followers=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        elapsed_seconds=time.monotonic()-started,rdma_tested=False,whole_model_acceptance=False)
    assert result['peak_allocated_bytes']<1_000_000_000
    output.open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
