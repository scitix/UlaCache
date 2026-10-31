"""Position ownership enters the real public factory before lifetime binding."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ulacache import (
    CacheExecutionOptions, KeyValueRecordPlane, KeyValueStorageLayout,
    LayerReuseSpecification, MemoryTierCapacity, PositionShardBindings,
    PositionShardLayout, SparseAccessSpecification, UnifiedCacheSpecification, init,
)
from ulacache.cache.create import validate_position_configuration


def specification():
    return UnifiedCacheSpecification(
        KeyValueStorageLayout('history',
            (KeyValueRecordPlane('kv', 'mla.scaled656.fp8.v1', (528, 128)),), 64),
        LayerReuseSpecification.from_shared_sparse_selection((0, 0)),
        SparseAccessSpecification(2, 2048),
        MemoryTierCapacity(4096, 4096, 8192, reserved_logical_record_count=64,
            maximum_prefill_union_record_count=4096))


def bindings(**updates):
    return PositionShardBindings(device='cuda:0',
        allocate_host_record_storage=lambda **kwargs: None,
        compile_device_module=lambda **kwargs: None,
        position_layout=PositionShardLayout(0, 8, 4096, 4096), **updates)


def test_real_init_installs_one_position_engine_for_all_public_lifetimes(monkeypatch):
    import ulacache.cache.position_residency as residency
    import ulacache.cache.create as create
    local = object()
    owned = SimpleNamespace(gpu=[object(), object()])
    observed = []
    def wrap(engine, layout):
        observed.append((engine, layout))
        return owned
    monkeypatch.setattr(create, 'create_rank_local_execution_engine', lambda *args: local)
    monkeypatch.setattr(residency, 'PositionShardedResidency', wrap)
    policy = bindings()
    cache = init(specification(), policy)
    assert observed == [(local, policy.position_layout)]
    assert cache._execution_engine is owned
    assert cache._slot_manager.backend is owned
    assert cache._access_engine.backend is owned
    assert cache.get_device_record_planes(0) == (owned.gpu[0],)


def test_mismatched_physical_capacity_fails_before_any_device_factory(monkeypatch):
    import ulacache.cache.create as create
    def forbidden(*args):
        raise AssertionError('Invalid position ownership reached allocation')
    monkeypatch.setattr(create, 'create_rank_local_execution_engine', forbidden)
    spec = specification()
    spec = replace(spec, capacity_plan=replace(spec.capacity_plan, host_record_capacity=8192))
    with pytest.raises(ValueError, match='exclusive'):
        init(spec, bindings())


def test_unsupported_storage_and_reserved_page_are_explicit():
    with pytest.raises(NotImplementedError, match='contiguous'):
        validate_position_configuration(specification(),
            bindings(execution_options=CacheExecutionOptions(record_storage_backend='native_paged')))
    spec = specification()
    spec = replace(spec, capacity_plan=replace(spec.capacity_plan, reserved_logical_record_count=0))
    with pytest.raises(ValueError, match='zero page'):
        validate_position_configuration(spec, bindings())
