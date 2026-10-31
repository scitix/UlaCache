"""Owner topology and distributed reader lifetime contracts without CUDA."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ulacache import (
    LayerReuseSpecification, RecordOwnerBindings, UnifiedMemoryCache,
)
from ulacache.cache.owner_cache import RecordOwnerPagedResidency
from ulacache.cache.planning import create_owner_paged_execution_plan
from test_paged_execution_plan import specification
from test_standalone_contracts import Tensor


def owner_specification():
    return replace(specification((0, 0, 0, 0), (0, 0, 0, 0)),
        layer_reuse=LayerReuseSpecification((0, 1, 0, 1), (0, 0, 0, 0),
            (0, 0, 0, 0), (0, 0, 0, 0)))


def test_owner_service_selects_disjoint_storage_and_borrowed_attention_views():
    config = owner_specification()
    services = RecordOwnerBindings('cuda:0', object(), Mock(), Mock(), Mock(), 256, 64, True)
    backend = Mock()
    first, second = object(), object()
    backend.gpu = (first, second, first, second)
    with patch('ulacache.cache.create.create_record_owner_execution_engine', return_value=backend) as owner:
        with patch('ulacache.cache.create.create_rank_local_execution_engine') as local:
            cache = UnifiedMemoryCache(config, services)
    local.assert_not_called()
    plan = owner.call_args.args[1]
    assert plan.storage_bank_indices_by_layer == (0, 1, 0, 1)
    assert plan.selection_producer_layers == (0, 1)
    assert cache.get_device_record_planes(0) == cache.get_device_record_planes(2) == (first,)
    assert cache.get_device_record_planes(1) == cache.get_device_record_planes(3) == (second,)


def test_independent_banks_publish_fresh_writes_and_batch_the_final_reader_retirement():
    runtime = RecordOwnerPagedResidency.__new__(RecordOwnerPagedResidency)
    runtime.execution_plan = create_owner_paged_execution_plan(owner_specification())
    runtime.maximum_query_count, runtime.selection_width = 2, 8
    runtime.exported_records = Mock()
    runtime.exported_records.__getitem__ = Mock(return_value=Mock())
    runtime.owner_residency = Mock()
    runtime.publication = Mock()
    runtime._selection_records, runtime._materialized_selections = {}, set()
    runtime._forward_current_records = None
    runtime._step_reader_retirement_layer = runtime.execution_plan.step_reader_retirement_layer
    runtime._completed_step_selections = set()
    runtime._batch_reader_retirement = True
    runtime.owner_residency.residency_states = (object(), object())
    runtime.owner_residency._live_residency_domains = {0, 1}
    runtime.remapped_selections = Mock()
    runtime.remapped_selections.__getitem__ = Mock(side_effect=lambda index: ('attention', index[0]))
    events = []
    runtime._prepare_exported_records = lambda domain, count, **options: events.append(('place', domain))
    runtime._map_owner_records = lambda records: records
    runtime.owner_residency.write.side_effect = lambda layer, *args: events.append(('write', layer))
    runtime._gather_records = lambda layer, *args: events.append(('publish-and-read', layer))
    def finish_domain(domain):
        runtime.owner_residency._live_residency_domains.remove(domain)
        events.append(('unlease', domain))
    runtime.owner_residency.finish_record_residency.side_effect = finish_domain
    runtime.publication.finish_readers.side_effect = lambda domain: events.append(('retire', domain))
    runtime.publication.finish_all_readers.side_effect = lambda: events.append('retire-step')
    kernels = SimpleNamespace(export_consumer_records=Mock())
    selected, current = Tensor(2, 8), Tensor(2)
    with patch.dict('sys.modules', {'ulacache.cache.kernels.owner_lookup': kernels}):
        for layer in (0, 1):
            runtime.plan(layer, selected, current)
            runtime.write(layer, current, object())
            runtime.attention_rows(layer, 2)
            runtime.after_attention(layer)
        assert not runtime.publication.finish_readers.called
        assert runtime.attention_rows(2, 2) == ('attention', 0)
        assert runtime.attention_rows(3, 2) == ('attention', 1)
        runtime.after_attention(2)
        runtime.publication.finish_all_readers.assert_not_called()
        runtime.after_attention(3)
    assert events == [('place', 0), ('write', 0), ('publish-and-read', 0),
        ('place', 1), ('write', 1), ('publish-and-read', 1),
        ('unlease', 0), ('unlease', 1), 'retire-step']


def test_reclamation_checks_local_readers_before_enqueuing_remote_wait():
    runtime = RecordOwnerPagedResidency.__new__(RecordOwnerPagedResidency)
    runtime.owner_residency, runtime.publication = Mock(), Mock()
    runtime._map_owner_records = Mock()
    runtime.owner_residency._require_retired_readers.side_effect = RuntimeError('local reader still active')
    with pytest.raises(RuntimeError, match='local reader still active'):
        runtime.release(object())
    runtime.publication.wait_for_all_readers.assert_not_called()
    runtime._map_owner_records.assert_not_called()
    runtime.owner_residency.release.assert_not_called()
