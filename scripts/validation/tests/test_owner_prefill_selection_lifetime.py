"""Shared prefill selections survive their consumers and stay bank-local."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from ulacache import (
    KeyValueRecordPlane, KeyValueStorageLayout, LayerReuseSpecification,
    MemoryTierCapacity, SparseAccessSpecification, UnifiedCacheSpecification,
)
from ulacache.cache.owner_cache import RecordOwnerPagedResidency
from ulacache.cache.planning import create_owner_paged_execution_plan


def backend(monkeypatch):
    specification = UnifiedCacheSpecification(
        KeyValueStorageLayout('native', (KeyValueRecordPlane('kv','test.native',(8,)),),64),
        LayerReuseSpecification((0,0,2),(0,0,2),(0,0,0),(0,0,2)),
        SparseAccessSpecification(2,4), MemoryTierCapacity(64,64,8))
    value = RecordOwnerPagedResidency.__new__(RecordOwnerPagedResidency)
    value.execution_plan = create_owner_paged_execution_plan(specification)
    value.maximum_query_count, value.selection_width = 2,4
    value.no_current_records = MagicMock()
    value.exported_records = MagicMock()
    value._selection_records = {}
    value._forward_current_records = None
    value._step_reader_retirement_layer = value.execution_plan.step_reader_retirement_layer
    value._completed_step_selections = set()
    value._batch_reader_retirement = value._step_reader_retirement_layer is not None
    value._materialized_selections = set()
    value._prefill_remapped_selections = {}
    value.owner_residency = SimpleNamespace(active_query_count=2,begin_forward=Mock())
    value.publication = SimpleNamespace(wait_for_all_readers=Mock())
    value._prepare_exported_records = Mock()
    value._gather_records = Mock()
    value.finish_prefill_record_chunk = Mock()
    outputs = {0:object(),1:object()}
    class SelectionRows:
        def __getitem__(self,key):return outputs[key[0]]
    value.remapped_selections = SelectionRows()
    export = Mock()
    monkeypatch.setitem(sys.modules,'ulacache.cache.kernels.owner_lookup',
        SimpleNamespace(export_consumer_records=export))
    return value,export,outputs


def test_prefill_follower_reuses_selection_and_retires_after_the_last_consumer(monkeypatch):
    value,export,outputs = backend(monkeypatch)
    records = SimpleNamespace(shape=(2,4))
    assert value.plan_prefill(0,records) is outputs[0]
    value.after_attention(0)
    value.finish_prefill_record_chunk.assert_not_called()
    assert value.plan_prefill(1,records) is outputs[0]
    assert export.call_count==value._prepare_exported_records.call_count==value._gather_records.call_count==1
    value.after_attention(1)
    value.finish_prefill_record_chunk.assert_called_once_with(1)
    assert value.plan_prefill(2,records) is outputs[1]
    assert export.call_count==value._prepare_exported_records.call_count==value._gather_records.call_count==2


def test_prefill_follower_cannot_create_a_selection_before_its_producer(monkeypatch):
    value,export,_ = backend(monkeypatch)
    with pytest.raises(RuntimeError,match='first consumer'):
        value.plan_prefill(1,SimpleNamespace(shape=(2,4)))
    export.assert_not_called()


def test_next_forward_plans_the_shared_prefill_selection_again(monkeypatch):
    value,export,_ = backend(monkeypatch)
    value.plan_prefill(0,SimpleNamespace(shape=(2,4)))
    value.after_attention(0)
    value.plan_prefill(1,SimpleNamespace(shape=(2,4)))
    value.after_attention(1)
    value.begin_forward()
    value.plan_prefill(0,SimpleNamespace(shape=(2,4)))
    assert export.call_count==value._gather_records.call_count==2
