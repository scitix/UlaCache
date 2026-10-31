"""Shared payload storage and reindex lifetimes before allocating device memory."""

import pytest
from dataclasses import replace
from unittest.mock import Mock, patch

from ulacache import (
    KeyValueRecordPlane, KeyValueStorageLayout, LayerReuseSpecification,
    MemoryTierCapacity, SparseAccessSpecification, UnifiedCacheSpecification,
    CacheExecutionOptions, LocalCacheBindings, UnifiedMemoryCache,
)
from ulacache.cache.planning import create_paged_cache_execution_plan


def specification(storage_owners, selection_owners):
    return UnifiedCacheSpecification(
        KeyValueStorageLayout('history', (KeyValueRecordPlane('kv', 'native.fp4.v1',
            (256, 32), 256, 'segment_major'),), 64),
        LayerReuseSpecification(storage_owners, storage_owners, selection_owners, storage_owners),
        SparseAccessSpecification(3, 512), MemoryTierCapacity(1024, 2048, 512))


def test_reindex_changes_selection_without_duplicating_payload_banks():
    selections = tuple(4 * (layer // 4) for layer in range(20))
    plan = create_paged_cache_execution_plan(specification((0,) * 20, selections))
    assert plan.storage_producer_layers == (0,)
    assert plan.storage_bank_indices_by_layer == (0,) * 20
    assert plan.selection_producer_layers == (0, 4, 8, 12, 16)
    assert plan.last_consumer_layers_by_selection == (3, 7, 11, 15, 19)
    assert plan.selection_indices_by_layer == tuple(layer // 4 for layer in range(20))


def test_independent_banks_allow_interleaved_consumer_lifetimes():
    plan = create_paged_cache_execution_plan(specification((0, 1, 0, 1), (0, 1, 0, 1)))
    assert plan.storage_producer_layers == (0, 1)
    assert plan.storage_bank_indices_by_layer == (0, 1, 0, 1)
    assert plan.last_consumer_layers_by_selection == (2, 3)


def test_shared_topk_coordinates_independent_per_layer_payloads():
    config = specification(tuple(range(6)), (0, 0, 0, 3, 3, 3))
    config = replace(config, layer_reuse=LayerReuseSpecification.from_shared_sparse_selection((0, 0, 0, 3, 3, 3)))
    plan = create_paged_cache_execution_plan(config)
    assert plan.storage_producer_layers == tuple(range(6))
    assert plan.residency_producer_layers == (0, 3)
    assert plan.residency_domain_indices_by_layer == (0, 0, 0, 1, 1, 1)
    assert plan.storage_bank_ranges_by_residency_domain == (((0, 3),), ((3, 6),))


def test_interleaved_independent_payloads_keep_separate_residency_directories():
    config = specification(tuple(range(4)), (0, 1, 0, 1))
    config = replace(config, layer_reuse=LayerReuseSpecification.from_shared_sparse_selection((0, 1, 0, 1)))
    plan = create_paged_cache_execution_plan(config)
    assert plan.residency_domain_indices_by_layer == (0, 1, 0, 1)
    assert plan.storage_bank_ranges_by_residency_domain == (((0, 1), (2, 3)), ((1, 2), (3, 4)))
    assert plan.last_consumer_layers_by_selection == (2, 3)


def test_payload_alias_cannot_have_two_placement_directories():
    config = specification((0, 0), (0, 1))
    config = replace(config, layer_reuse=LayerReuseSpecification((0, 0), (0, 0), (0, 1), (0, 1)))
    with pytest.raises(NotImplementedError, match='shared payload bank'):
        create_paged_cache_execution_plan(config)


def test_shared_selection_requires_a_common_placement_directory():
    with pytest.raises(NotImplementedError, match='shared selection'):
        create_paged_cache_execution_plan(specification((0, 1), (0, 0)))


def test_overlapping_selections_on_one_bank_need_a_different_lease_implementation():
    with pytest.raises(NotImplementedError, match='additional lease storage'):
        create_paged_cache_execution_plan(specification((0, 0, 0), (0, 1, 0)))


def test_native_paged_runtime_preserves_page_order_and_shared_bank_views():
    config = specification((0, 0, 0), (0, 0, 2))
    native_bank = object()
    backend = Mock(gpu=(native_bank,) * 3)
    services = LocalCacheBindings('cuda:0', Mock(), Mock(),
        CacheExecutionOptions(record_storage_backend='native_paged'))
    with patch('ulacache.cache.create.create_rank_local_execution_engine', return_value=backend) as factory:
        cache = UnifiedMemoryCache(config, services)
    plan = factory.call_args.args[1]
    assert plan.storage_producer_layers == (0,) and plan.selection_producer_layers == (0, 2)
    inputs = cache.get_resident_attention_inputs(2, 3)
    assert inputs.record_planes == (native_bank,)
    assert inputs.records_per_page == 64 and inputs.record_layout == 'segment_major'
    logical_records, packed_pages, source_records = object(), object(), object()
    cache.write_native_paged_records(0, logical_records, packed_pages, source_records)
    backend.write_native_paged_records.assert_called_once_with(0, logical_records, packed_pages, source_records)
    chunk = cache.prepare_prefill_record_chunk(1, logical_records)
    assert chunk.record_planes == (native_bank,)
    backend.prepare_prefill_record_chunk.assert_called_once_with(1, logical_records)
    cache.finish_prefill_record_chunk(1)
    backend.finish_prefill_record_chunk.assert_called_once_with(1)
    backend.after_attention.assert_not_called()
    cache.close()


def test_paged_reindex_cannot_overwrite_an_unfinished_selection():
    from ulacache.cache.local_paged_cache import NativePagedResidency
    runtime = NativePagedResidency.__new__(NativePagedResidency)
    runtime._pending_transfers = set()
    runtime._joined_banks = set()
    runtime.execution_plan = create_paged_cache_execution_plan(specification((0,) * 4, (0, 0, 2, 2)))
    runtime._planned_selections = {0}
    runtime._live_residency_domains = {0}
    with pytest.raises(RuntimeError, match='still has readers'):
        runtime._start_selection(2)
    with pytest.raises(RuntimeError, match='Finish every record reader'):
        runtime.begin_forward()
    runtime._live_residency_domains.clear()
    assert runtime._start_selection(2) == (1, 0)
    runtime.begin_forward()
    assert not runtime._planned_selections


def test_unknown_backend_fails_before_loading_device_code():
    services = LocalCacheBindings('cuda:0', Mock(), Mock(),
        replace(CacheExecutionOptions(), record_storage_backend='unknown'))
    with patch('ulacache.cache.create.create_rank_local_execution_engine') as factory:
        with pytest.raises(ValueError, match='Unknown record storage backend'):
            UnifiedMemoryCache(specification((0,), (0,)), services)
    factory.assert_not_called()
