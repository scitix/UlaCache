"""Bounded staging arithmetic and real metadata lease APIs, no large KV allocation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

torch=pytest.importorskip('torch')
from ulacache.transfer.receive_window import PinnedRecordReceiveWindow,ReceivePageAllocator
from ulacache.transfer.owner_index_receive import OwnerIndexRecordReceiveWindow

PATH=Path(__file__).resolve().parents[3]/'scripts/validation/tests/fixtures/hy4_pd/receive_budget.py'
spec=importlib.util.spec_from_file_location('pd_cpu_formula',PATH)
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def metadata_window(topology,rank=0,S=76032):
    cls=OwnerIndexRecordReceiveWindow if topology=='tp' else PinnedRecordReceiveWindow
    w=object.__new__(cls)
    w.page_size=64;w.record_capacity=S;w.pages=ReceivePageAllocator(S//64)
    w._pending={};w._sidecars={};w._registration_frozen=False
    w.engine=NS(directory=NS(device=torch.device('cpu'),shape=(21,1048576//8 if topology=='tp' else 1048576)))
    if topology=='tp':
        w.degree=8;w.rank=rank
        w._index_banks={
            'target':dict(degree=8,rank=rank,storage=NS(storage=[NS(nbytes=S*132)]),destinations=[NS(nbytes=(1048576//8)*132)]),
            'draft':dict(degree=1,rank=0,storage=NS(storage=[NS(nbytes=S*132)]),destinations=[NS(nbytes=1048576*132)])}
    return w


@pytest.mark.parametrize('topology',['tp','dpa'])
def test_full75973_cold_logical_IDs_far_exceed_bounded_staging_and_reuse(topology):
    w=metadata_window(topology)
    pages=torch.arange(8192,8192+1188)
    assert pages[0]*64>w.record_capacity
    w.reserve('full_cold',pages,record_count=75973)
    item=w._pending['full_cold']
    assert item['source_rows'].numel()==75973
    assert item['source_rows'].min()==0 and item['source_rows'].max()==75972
    assert max(w.transfer_pages('full_cold'))==1187
    assert w.available_records==0
    with pytest.raises(BufferError):w.reserve('small_suffix',torch.tensor([10000]),record_count=64)
    w.logical_pages_freed(pages.tolist())
    assert w.available_records==76032
    w.reserve('small_suffix',torch.tensor([10000]),record_count=64)
    with pytest.raises(BufferError):w.reserve('evicted_session_full_cold',pages,record_count=75973)
    w.logical_pages_freed([10000])
    w.reserve('evicted_session_full_cold',pages,record_count=75973)
    assert w._pending['evicted_session_full_cold']['source_rows'].max()<76032


def test_TP_publishes_owner_rows_but_staging_lease_counts_global_records():
    counts=[]
    for rank in range(8):
        w=metadata_window('tp',rank)
        w.reserve('cold',torch.arange(8192,8192+1188),record_count=75973)
        owned,source=w._target_record_indices(w._pending['cold'])
        counts.append(owned.numel())
        assert source.numel()==owned.numel() and source.max()<w.record_capacity
    assert sum(counts)==75973 and max(counts)-min(counts)==1


def test_undersized_unsegmented_window_fails_before_launch_not_as_capacity_endpoint():
    with pytest.raises(ValueError,match='cannot admit'):
        f.require_full_cold_capacity(75968,75973)
    assert f.require_full_cold_capacity(76032,75973)['required_page_aligned_records']==76032
    w=metadata_window('dpa',S=75968)
    with pytest.raises(BufferError):w.reserve('too_large',torch.arange(8192,8192+1188),record_count=75973)


@pytest.mark.parametrize('topology,staging_coefficient',[('tp',54728),('dpa',51824)])
def test_source_derived_buffers_fit_known_terms_without_target_only_sizing(topology,staging_coefficient):
    p=f.cpu_components(topology=topology,host_rows=0,staging_records=76032,
        target_layers=78,target_index_layers=21,pending_records=76032)
    staging=sum(p[k] for k in ['target_KV_receive','replicated_draft_KV_receive',
        'target_Index_receive','independent_draft_Index_receive'])
    assert staging==staging_coefficient*76032
    bound=f.host_row_upper_bound(topology=topology,staging_records=76032,
        target_layers=78,target_index_layers=21,pending_records=76032)
    H=bound['host_rows_upper_bound_from_known_terms']
    assert H<2501504
    assert f.cpu_components(topology=topology,host_rows=H,staging_records=76032,
        target_layers=78,target_index_layers=21,pending_records=76032)['known_tensor_peak_bound']<=128000000000
    assert not bound['actual_CPU_budget_passed'] and not bound['launch_geometry_selected']


@pytest.mark.parametrize('topology',['tp','dpa'])
def test_concurrent_small_suffixes_use_physical_staging_not_logical_IDs(topology):
    w=metadata_window(topology)
    A,B,C=12000,12100,12200
    a=w.reserve('A',torch.tensor([A]),record_count=7)
    b=w.reserve('B',torch.tensor([B]),record_count=19)
    assert a.pages==(0,) and b.pages==(1,)
    assert a.pages!=b.pages and max(w._pending['B']['source_rows']).item()<128
    with pytest.raises(RuntimeError,match='before transport drains'):
        w.pages.release(a,transport_quiesced=False)
    w.logical_pages_freed([A])
    c=w.reserve('C',torch.tensor([C]),record_count=64)
    assert c.pages==a.pages and c.generation>a.generation
    assert w._pending['B']['lease']==b
    w.logical_pages_freed([B,C])
    assert w.available_records==76032
