"""Separate native planes share ownership without merging their byte layouts."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import pytest

from test_paged_execution_plan import specification
from ulacache import KeyValueRecordPlane, CacheExecutionOptions, LocalCacheBindings, UnifiedMemoryCache
from ulacache.storage.buffers import RecordStorage
from ulacache.cache.local_paged_cache import NativePagedResidency
from ulacache.cache.planning import create_paged_cache_execution_plan


def separate_key_value_specification():
    original = specification((0, 1, 0), (0, 1, 0))
    planes = (KeyValueRecordPlane('key', 'native.key.bfloat16.v1', (256,), 256),
        KeyValueRecordPlane('value', 'native.value.fp8.v1', (128, 16), 128, 'segment_major'))
    return replace(original, storage_layout=replace(original.storage_layout, record_planes=planes))


def create_storage(plane_storage):
    configuration = separate_key_value_specification()
    plan = create_paged_cache_execution_plan(configuration)
    with patch('ulacache.storage.buffers.RecordPlaneStorage', side_effect=plane_storage) as factory:
        storage = RecordStorage(configuration, object(), plan)
    return configuration, plan, storage, factory


def test_single_plane_uses_the_same_storage_owner_and_preserves_fused_publication():
    configuration = specification((0, 1, 0), (0, 1, 0))
    plan = create_paged_cache_execution_plan(configuration)
    bank, pages, records, indices, directory = [object() for _ in range(5)]
    plane = Mock(get_device_record_bank=Mock(return_value=bank))
    with patch('ulacache.storage.buffers.RecordPlaneStorage', return_value=plane):
        storage = RecordStorage(configuration, object(), plan)
    assert storage.get_device_record_bank(0) is bank
    assert storage.get_device_record_planes(0) == (bank,)
    storage.write_native_record_source(0, records, pages, indices, directory,
                                      local_device_after_commit=True)
    plane.write_native_record_source.assert_called_once_with(
        0, records, pages, indices, directory, local_device_after_commit=True)
    workspace = object()
    storage.exchange_record_bank(1, workspace)
    plane.exchange_record_bank.assert_called_once_with(1, workspace)


def test_each_native_plane_retains_its_geometry_and_shared_bank_identity():
    key_banks, value_banks = [object(), object()], [object(), object()]
    keys = Mock(get_device_record_bank=lambda layer: key_banks[(0, 1, 0)[layer]])
    values = Mock(get_device_record_bank=lambda layer: value_banks[(0, 1, 0)[layer]])
    configuration, plan, storage, factory = create_storage([keys, values])
    for call, plane in zip(factory.call_args_list, configuration.storage_layout.record_planes):
        assert call.args[0].storage_layout.record_planes == (plane,)
        assert call.args[0].capacity_plan is configuration.capacity_plan
        assert call.args[2] is plan
    assert storage.get_device_record_planes(0) == storage.get_device_record_planes(2) == (key_banks[0], value_banks[0])
    assert storage.get_device_record_planes(1) == (key_banks[1], value_banks[1])
    backend = SimpleNamespace(device_record_planes_by_layer=tuple(storage.get_device_record_planes(layer) for layer in range(3)),
        attention_rows=Mock(return_value=object()))
    services = LocalCacheBindings('cuda:0', Mock(), Mock(), CacheExecutionOptions(record_storage_backend='native_paged'))
    with patch('ulacache.cache.create.create_rank_local_execution_engine', return_value=backend):
        cache = UnifiedMemoryCache(configuration, services)
    planes = cache.get_device_record_planes(0)
    assert planes is cache.get_device_record_planes(0)
    assert all(first is second for first, second in zip(planes, cache.get_device_record_planes(2)))
    assert cache.get_resident_attention_inputs(2, 3).record_planes == (key_banks[0], value_banks[0])


def test_placement_commits_only_after_all_native_planes_finish_copying(monkeypatch):
    events = []
    keys = Mock(exchange_residency_plan=lambda *_: events.append('key'))
    values = Mock(exchange_residency_plan=Mock(side_effect=RuntimeError('value transfer failed')))
    configuration, plan, storage, _ = create_storage([keys, values])
    runtime = NativePagedResidency.__new__(NativePagedResidency)
    runtime.record_storage = storage
    runtime.transfer_workspaces = (object(),)
    runtime.residency_states = (object(),)
    runtime._live_residency_domains = set()
    monkeypatch.setitem(sys.modules, 'ulacache.cache.kernels.cpu_residency', SimpleNamespace(
        commit_host_tier=lambda *_: events.append('commit')))
    with pytest.raises(RuntimeError, match='value transfer failed'):
        runtime._exchange_and_commit(0)
    assert events == ['key'] and not runtime._live_residency_domains
    values.exchange_residency_plan.side_effect = lambda *_: events.append('value')
    runtime._exchange_and_commit(0)
    assert events == ['key', 'key', 'value', 'commit']
    assert runtime._live_residency_domains == {0}


def test_native_publication_requires_all_planes_and_preserves_segment_order():
    keys, values = Mock(), Mock()
    _, _, storage, _ = create_storage([keys, values])
    records, indices, directory = object(), object(), object()
    key_pages, value_pages = object(), object()
    with pytest.raises(ValueError, match='every declared record plane'):
        storage.write_native_record_source(0, records, (key_pages,), indices, directory)
    keys.write_native_record_source.assert_not_called()
    storage.write_native_record_source(0, records, (key_pages, value_pages), indices, directory)
    keys.write_native_record_source.assert_called_once_with(0, records, key_pages, indices, directory)
    values.write_native_record_source.assert_called_once_with(0, records, value_pages, indices, directory)
    key_bytes, value_bytes, value_scales = object(), object(), object()
    storage.write_native_record_segments(0, records, (key_bytes, value_bytes, value_scales), directory)
    keys.write_native_record_segments.assert_called_once_with(0, records, (key_bytes,), directory)
    values.write_native_record_segments.assert_called_once_with(0, records, (value_bytes, value_scales), directory)


def test_failed_second_native_allocation_releases_the_first_pinned_plane():
    keys = Mock()
    configuration = separate_key_value_specification()
    plan = create_paged_cache_execution_plan(configuration)
    with patch('ulacache.storage.buffers.RecordPlaneStorage', side_effect=[keys, MemoryError('host allocation failed')]):
        with pytest.raises(MemoryError, match='host allocation failed'):
            RecordStorage(configuration, object(), plan)
    keys.close.assert_called_once_with()


@pytest.mark.parametrize('enabled,expected', [
    ([], False), ([True], True), ([False], False),
    ([True, True], True), ([True, False], False),
])
def test_fused_publication_requires_support_from_every_native_plane(enabled, expected):
    storage = RecordStorage.__new__(RecordStorage)
    storage.plane_storage = [SimpleNamespace(fuse_native_record_publication=value)
                             for value in enabled]
    assert storage.fuse_native_record_publication is expected


@pytest.mark.parametrize('multiple_planes', [False, True])
def test_prefill_reader_borrows_each_composed_plane_without_new_storage(multiple_planes):
    import torch
    from ulacache.cache.prefill.buffers import NativePrefillRecordReader

    configuration = (separate_key_value_specification() if multiple_planes
                     else specification((0, 1, 0), (0, 1, 0)))
    plan = create_paged_cache_execution_plan(configuration)
    planes = [SimpleNamespace(device_record_pages=torch.empty((2, 3, 64), dtype=torch.uint8))
              for _ in configuration.storage_layout.record_planes]
    storage = RecordStorage.__new__(RecordStorage)
    storage.plane_storage = planes
    backend = SimpleNamespace(record_storage=storage, execution_plan=plan)
    windows = tuple(torch.empty((2, 3, 64), dtype=torch.uint8) for _ in planes)
    with patch('ulacache.transfer.gpu_peer.NativePagedPeerRecordTransfer'):
        reader = NativePrefillRecordReader(backend, configuration, windows)
    assert reader.storages == tuple(planes)
    assert all(actual is original for actual, original in zip(reader.storages, planes))
