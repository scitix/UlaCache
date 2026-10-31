"""Keep optional expiry separate from active-query publication and readers."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ulacache.cache.attention_access import NativeSparseCacheExecution
from ulacache.cache.local_paged_cache import NativePagedResidency
from ulacache.client import UnifiedMemoryCache


def test_optional_expiry_waits_for_the_first_reader_and_never_republishes_counts():
    calls = []
    active = Mock()
    execution = NativeSparseCacheExecution.__new__(NativeSparseCacheExecution)
    execution.memory_cache = SimpleNamespace(active_query_count_tensor=active,
        begin_model_forward=lambda: calls.append('begin'),
        prepare_next_residency_epochs=lambda: calls.append('expire'))
    execution._resident_prefill_layers = set()
    for decode_like in [False, True, True, False]:
        execution.begin_forward(decode_like=decode_like, active_query_counts=None,
            attention_data_parallel_rank=0, query_count=8,
            publish_active_query_count=False)
        before = len(calls)
        execution._prepare_forward_residency_once()
        execution._prepare_forward_residency_once()
        assert len(calls) - before == int(decode_like)
    assert calls == ['begin', 'begin', 'expire', 'begin', 'expire', 'begin']
    active.fill_.assert_not_called()
    active.copy_.assert_not_called()


@pytest.mark.parametrize('recorded_setup', [False, True])
def test_outer_and_recorded_setup_emit_one_current_transaction_at_the_first_reader(recorded_setup):
    calls = []
    active, current = Mock(), Mock()
    current.numel.return_value = 8
    execution = NativeSparseCacheExecution.__new__(NativeSparseCacheExecution)
    execution.memory_cache = SimpleNamespace(active_query_count_tensor=active,
        begin_model_forward=lambda: calls.append('host_begin'),
        prepare_forward_current_records=Mock(side_effect=lambda ids: calls.append('device_current') or True),
        prepare_next_residency_epochs=Mock())
    execution._resident_prefill_layers = set()
    arguments = dict(decode_like=True, active_query_counts=None,
        attention_data_parallel_rank=0, query_count=8, current_logical_record_ids=current)
    execution.begin_forward(**arguments)
    if recorded_setup:
        execution.begin_forward(**arguments, publish_active_query_count=False)
    assert 'device_current' not in calls
    class Selected:
        shape = (8, 2048)
        def flatten(self, *args): return self
        def int(self): return self
        def contiguous(self): return self
        def reshape(self, *args): return self
        def __getitem__(self, key): return self
    selected = Selected()
    execution.writer_staging = SimpleNamespace(pending_logical_record_ids=lambda layer: current,
        publish_staged_records=Mock())
    execution.execution_plan = SimpleNamespace(selection_indices_by_layer=(0, 0),
        selection_producer_layers=(0,))
    execution.no_current_write_records = selected
    execution.memory_cache.prepare_sparse_attention = lambda *args: calls.append('plan')
    execution.memory_cache.get_resident_attention_inputs = lambda *args: SimpleNamespace(device_record_indices=selected)
    execution.prepare_attention(0, selected)
    execution.prepare_attention(1, selected)
    assert calls.count('device_current') == 1
    assert calls.index('device_current') < calls.index('plan')
    execution.memory_cache.prepare_forward_current_records.assert_called_once_with(current)
    execution.memory_cache.prepare_next_residency_epochs.assert_not_called()
    active.fill_.assert_called_once_with(8)


def test_other_execution_engines_keep_their_existing_lifecycle():
    cache = UnifiedMemoryCache.__new__(UnifiedMemoryCache)
    cache._execution_engine = SimpleNamespace()
    assert cache.prepare_next_residency_epochs() is None


def test_expiry_requires_previous_readers_to_retire_even_when_disabled():
    engine = NativePagedResidency.__new__(NativePagedResidency)
    engine._pending_transfers = set()
    engine._live_residency_domains = {0}
    engine.execution_options = SimpleNamespace(expire_residency_in_parallel=False)
    with pytest.raises(RuntimeError, match='Finish every record reader'):
        engine.prepare_next_residency_epochs()


@pytest.mark.parametrize('mismatch', ['pointer', 'shape', 'dtype', 'missing'])
def test_first_eager_reader_rejects_unbound_common_ids_before_device_mutation(mismatch):
    def tensor(pointer=100, shape=(2,), dtype='int64'):
        return SimpleNamespace(data_ptr=lambda: pointer, shape=shape, dtype=dtype)
    common = tensor()
    actual = dict(pointer=tensor(pointer=200), shape=tensor(shape=(1,)),
        dtype=tensor(dtype='int32'), missing=None)[mismatch]
    execution = NativeSparseCacheExecution.__new__(NativeSparseCacheExecution)
    execution._residency_preparation_pending = True
    execution._pending_forward_current_records = common
    execution.memory_cache = SimpleNamespace(prepare_forward_current_records=Mock(),
        prepare_next_residency_epochs=Mock())
    execution._prepare_forward_residency_once(actual)
    execution._prepare_forward_residency_once(common)
    execution.memory_cache.prepare_forward_current_records.assert_not_called()
    execution.memory_cache.prepare_next_residency_epochs.assert_called_once_with()
