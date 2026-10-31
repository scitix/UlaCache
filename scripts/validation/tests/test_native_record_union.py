"""Reuse native placement and opaque swaps without another residency planner."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ulacache import MemoryTierCapacity
from ulacache.cache.local_paged_cache import NativePagedResidency
from ulacache.cache_config import CacheExecutionOptions


def test_padded_residency_input_capacity_is_separate_from_unique_gpu_capacity():
    capacity = MemoryTierCapacity(64, 128, 8, maximum_residency_input_record_count=1024)
    assert capacity.logical_record_capacity == 192
    assert capacity.maximum_residency_input_record_count > capacity.device_record_capacity
    assert MemoryTierCapacity(64, 128, 8).maximum_residency_input_record_count == 0


def test_native_union_keeps_bank_exchange_commit_and_domain_reader_lifetimes():
    runtime = NativePagedResidency.__new__(NativePagedResidency)
    runtime._pending_transfers = set()
    runtime.execution_options = CacheExecutionOptions()
    runtime.residency_states = (object(), object())
    runtime.transfer_workspaces = (object(), object())
    runtime.prefill_current_record = object()
    runtime.prefill_active_query_count = object()
    runtime._live_residency_domains = set()
    events = []
    runtime.record_storage = SimpleNamespace(exchange_residency_plan=lambda domain, workspace:
        events.append(('exchange', domain)))
    native = SimpleNamespace(prepare_host_tier=Mock(side_effect=lambda *arguments, **options:
        events.append(('plan', arguments[4]))),
        commit_host_tier=Mock(side_effect=lambda state, workspace: events.append(('commit', state))),
        release_host_tier_leases=Mock(side_effect=lambda state, workspace: events.append(('release', state))))
    records = Mock()
    with patch.dict('sys.modules', {'ulacache.cache.kernels.cpu_residency': native}):
        runtime.prepare_record_residency(0, records)
        native.prepare_host_tier.assert_called_once_with(records.reshape.return_value,
            runtime.prefill_current_record, runtime.prefill_active_query_count, 0,
            runtime.residency_states[0], runtime.transfer_workspaces[0],
            parallel=False, parallel_victims=False, current_already_prepared=False)
        assert events[:3] == [('plan', runtime.residency_states[0]),
            ('exchange', 0), ('commit', runtime.residency_states[0])]
        runtime.prepare_record_residency(1, records)
        assert runtime._live_residency_domains == {0, 1}
        with pytest.raises(RuntimeError, match='Finish the current record union'):
            runtime.prepare_record_residency(0, records)
        assert native.prepare_host_tier.call_count == 2
        runtime.finish_record_residency(0)
        assert runtime._live_residency_domains == {1}
        runtime.prepare_record_residency(0, records)
        assert runtime._live_residency_domains == {0, 1}
        runtime.finish_record_residency(1)
        runtime.finish_record_residency(0)
        native.release_host_tier_leases.assert_not_called()
        with pytest.raises(KeyError):
            runtime.finish_record_residency(0)
    assert not runtime._live_residency_domains
