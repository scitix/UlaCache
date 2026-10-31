"""CPU routing tests. CUDA kernels and byte visibility require the GPU oracle."""
import sys
from types import ModuleType, SimpleNamespace as N
from unittest.mock import Mock

import pytest

from ulacache.cache.attention_access import NativeSparseCacheExecution
from ulacache.cache.local_paged_cache import NativePagedResidency
from ulacache.client import UnifiedMemoryCache
from ulacache.storage.writers import NativeRecordWriteStaging
from ulacache.storage.buffers import RecordPlaneStorage, RecordStorage
from ulacache.transfer.cpu_gpu import NativePagedRecordTransfer


class Tensor:
    def __init__(self, pointer=100, count=3, dtype='int64', device='cuda:0'):
        self.pointer, self.count, self.dtype, self.device = pointer, count, dtype, device
        self.shape = (count,)
    def data_ptr(self): return self.pointer
    def numel(self): return self.count
    def __getitem__(self, key): return self
    def flatten(self, start): return self
    def int(self): return self
    def contiguous(self): return self
    def reshape(self, shape): return self
    def fill_(self, value): return self
    def copy_(self, source): return self


def execution_fixture(*, capacity=True, enabled=True):
    events = []
    selected = Tensor(count=3); selected.shape = (3, 2048)
    current = Tensor()
    pending = {0: current, 1: current}
    cache = N(supports_local_native_current_publication=capacity,
        active_query_count_tensor=Tensor(), begin_model_forward=lambda: events.append('begin'),
        prepare_sparse_attention=lambda *args: events.append('commit'),
        get_resident_attention_inputs=lambda *args: N(device_record_indices=selected))
    writer = N(pending_logical_record_ids=lambda layer: pending.get(layer),
        publish_staged_records=lambda cache, layer, **kw: events.append(('publish', layer, kw)))
    e = NativeSparseCacheExecution.__new__(NativeSparseCacheExecution)
    e.local_current_publication = enabled
    e.memory_cache, e.writer_staging = cache, writer
    e.execution_plan = N(selection_indices_by_layer=(0, 0), selection_producer_layers=(0,))
    e.current_write_records = e.no_current_write_records = Tensor()
    e._resident_prefill_layers = set()
    e.begin_forward(decode_like=True, active_query_counts=None, attention_data_parallel_rank=0,
                    query_count=3, publish_active_query_count=False)
    return e, selected, current, pending, events


def test_default_serving_path_retains_system_publication_without_opt_in():
    e, selected, _, _, events = execution_fixture(enabled=False)
    e.prepare_attention(0, selected)
    assert events == ['begin', 'commit', ('publish', 0, {})]
    assert not e._committed_current_publications


def test_anchor_commit_precedes_local_current_publication_and_follower_matches():
    e, selected, _, _, events = execution_fixture()
    e.prepare_attention(0, selected); e.prepare_attention(1, selected)
    assert events == ['begin', 'commit', ('publish', 0, dict(local_device_after_commit=True)),
                      ('publish', 1, dict(local_device_after_commit=True))]


@pytest.mark.parametrize('change', ['pointer', 'count', 'dtype', 'device'])
def test_follower_different_current_view_retains_system_publication(change):
    e, selected, _, pending, events = execution_fixture()
    e.prepare_attention(0, selected)
    values = dict(pointer=100, count=3, dtype='int64', device='cuda:0')
    values[change] = dict(pointer=200, count=2, dtype='int32', device='cuda:1')[change]
    pending[1] = Tensor(**values)
    e.prepare_attention(1, selected)
    assert events[-1] == ('publish', 1, {})


def test_next_begin_invalidates_prior_commit_qualification():
    e, selected, _, _, events = execution_fixture()
    e.prepare_attention(0, selected)
    e.begin_forward(decode_like=True, active_query_counts=None, attention_data_parallel_rank=0,
                    query_count=3, publish_active_query_count=False)
    e.prepare_attention(1, selected)
    assert events[-1] == ('publish', 1, {})


def test_last_consumer_and_clear_discard_the_qualification():
    e, selected, _, _, _ = execution_fixture()
    e.execution_plan.last_consumer_layers_by_selection = (1,)
    e.memory_cache.finish_layer_attention = Mock()
    e.prepare_attention(0, selected)
    e.finish_attention(0)
    assert e._committed_current_publications
    e.finish_attention(1)
    assert not e._committed_current_publications
    e.prepare_attention(0, selected)
    e.memory_cache.clear_registered_records = Mock()
    e.memory_cache.register_logical_records = Mock()
    e.reserved_logical_records = Tensor()
    e.clear_records()
    assert not e._committed_current_publications


def test_commit_exception_never_qualifies_or_publishes():
    e, selected, _, _, events = execution_fixture()
    e.memory_cache.prepare_sparse_attention = Mock(side_effect=ValueError('commit failed'))
    with pytest.raises(ValueError, match='commit failed'): e.prepare_attention(0, selected)
    assert events == ['begin'] and not e._committed_current_publications


def test_unsupported_backend_and_prefill_retain_original_calls():
    e, selected, _, _, events = execution_fixture(capacity=False)
    e.prepare_attention(0, selected)
    assert events[-1] == ('publish', 0, {})
    e.decode_like = False; e.publish_after_native_write(1)
    assert events[-1] == ('publish', 1, {})


def test_multi_plane_or_legacy_storage_keeps_its_original_signature():
    backend = NativePagedResidency.__new__(NativePagedResidency)
    backend.record_storage = N(write_native_record_source=Mock())
    backend.execution_plan = N(residency_domain_indices_by_layer=(0,))
    backend.residency_states = [N(directory=Tensor())]
    backend.before_layer_prepare = Mock()
    assert backend.supports_local_native_current_publication is False
    backend.write_native_paged_records(0, Tensor(), Tensor(), Tensor())
    assert not backend.record_storage.write_native_record_source.call_args.kwargs


@pytest.mark.parametrize('local', [False, True])
def test_real_staging_client_native_backend_storage_transfer_callthrough(local, monkeypatch):
    # The ordinary mapper is eagerly imported by the existing method, but must
    # never execute on either fused route. This is an explicit GPU terminal.
    mapper = ModuleType('ulacache.storage.kernels.paged_write')
    mapper.map_native_write_locations = Mock(side_effect=AssertionError('ordinary mapper used'))
    monkeypatch.setitem(sys.modules, mapper.__name__, mapper)
    transfer = NativePagedRecordTransfer.__new__(NativePagedRecordTransfer)
    transfer._publish_module = N(publish=Mock())
    transfer._local_current_publish_module = N(publish=Mock())
    storage = RecordPlaneStorage.__new__(RecordPlaneStorage)
    storage.fuse_native_record_publication = True
    storage.capacity_plan = N(maximum_write_record_count=32)
    storage.execution_plan = N(storage_bank_indices_by_layer=(0,))
    storage.device_record_pages = Tensor()
    storage.host_record_pool = N(storage=Tensor())
    storage.record_transfer = transfer
    composed = RecordStorage.__new__(RecordStorage)
    composed.plane_storage = [storage]
    backend = NativePagedResidency.__new__(NativePagedResidency)
    backend.record_storage = composed
    backend.execution_plan = N(residency_domain_indices_by_layer=(0,))
    backend.residency_states = [N(directory=Tensor())]
    backend.before_layer_prepare = Mock()
    cache = UnifiedMemoryCache.__new__(UnifiedMemoryCache)
    cache._execution_engine = backend
    staging = NativeRecordWriteStaging.__new__(NativeRecordWriteStaging)
    staging.execution_plan = N(storage_bank_indices_by_layer=(0,))
    staging.maximum_write_record_count = 32
    staging.native_record_pages = Tensor()
    staging.native_record_plane_pages = (staging.native_record_pages,)
    staging.source_record_indices = Tensor()
    staging._pending_writes = {}
    staging.reserve_native_write(0, Tensor())
    staging.publish_staged_records(cache, 0, local_device_after_commit=local)
    backend.before_layer_prepare.assert_called_once_with(0)
    selected = transfer._local_current_publish_module if local else transfer._publish_module
    other = transfer._publish_module if local else transfer._local_current_publish_module
    selected.publish.assert_called_once(); other.publish.assert_not_called()
    mapper.map_native_write_locations.assert_not_called()
    assert not staging._pending_writes
