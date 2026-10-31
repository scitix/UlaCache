"""Startup group/prefetch wiring; CPU-only contracts, not CUDA acceptance."""

from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock, patch

import pytest

from ulacache import (
    CacheExecutionOptions, KeyValueRecordPlane, LayerReuseSpecification,
    LocalCacheBindings, PrefetchPolicy, RecordOwnerBindings, init,
)
from ulacache.cache_config import CacheOptions
from ulacache.cache.runtime import AccessEngine, SlotManager
from test_standalone_contracts import create_cache_specification


def grouped_specification():
    groups = ((0, 1, 2), (3, 4, 5))
    sharing = LayerReuseSpecification.from_layer_groups(6,
        index_groups=groups, selection_groups=groups)
    return replace(create_cache_specification(), layer_reuse=sharing)


def test_index_storage_selection_and_kv_groups_are_independent():
    sharing = LayerReuseSpecification.from_layer_groups(6,
        key_value_groups=((3, 4),), index_groups=((0, 1, 2), (3, 4)),
        selection_groups=((0, 1), (3, 4)))
    assert sharing.key_value_storage_owners == (0, 1, 2, 3, 3, 5)
    assert sharing.index_storage_owners == (0, 0, 0, 3, 3, 5)
    assert sharing.sparse_selection_owners == (0, 0, 2, 3, 3, 5)
    assert sharing.residency_plan_anchors == sharing.sparse_selection_owners
    with pytest.raises(FrozenInstanceError):
        sharing.index_storage_owners = tuple(range(6))


def test_explicit_residency_groups_and_unsorted_layers_use_canonical_producers():
    sharing = LayerReuseSpecification.from_layer_groups(4,
        index_groups=((2, 0, 1),), selection_groups=((1, 0),),
        residency_groups=((2, 0, 1),))
    assert sharing.index_storage_owners == sharing.residency_plan_anchors == (0, 0, 0, 3)
    assert sharing.sparse_selection_owners == (0, 0, 2, 3)


@pytest.mark.parametrize('count', [0, -1, True, 2.5])
def test_layer_count_must_be_a_positive_integer(count):
    with pytest.raises(ValueError, match='Layer count'):
        LayerReuseSpecification.from_layer_groups(count)


@pytest.mark.parametrize('field', ['key_value_groups', 'index_groups', 'selection_groups', 'residency_groups'])
@pytest.mark.parametrize('groups', [((0, 1), (1, 2)), ((0, 0),), ((-1, 2),), ((0, 3),), ((),)])
def test_invalid_or_overlapping_sharing_groups_are_rejected(field, groups):
    with pytest.raises(ValueError):
        LayerReuseSpecification.from_layer_groups(3, **{field: groups})


@pytest.mark.parametrize('backend, completion', [('contiguous_records', 'group'), ('native_paged', 'layer'), ('tensor_parallel', 'layer')])
def test_auto_prefetch_completion_is_lowered_once_at_initialization(backend, completion):
    policy = PrefetchPolicy(enabled=True)
    options = CacheExecutionOptions(record_storage_backend=backend, prefetch=policy)
    assert options.copy_follower_records_asynchronously
    assert options.follower_copy_mode == completion
    assert options.prefetch is policy
    with pytest.raises(FrozenInstanceError):
        policy.enabled = False
    with pytest.raises(FrozenInstanceError):
        options.follower_copy_mode = 'group'


def test_disabled_policy_is_explicit_and_legacy_options_are_preserved_when_unset():
    legacy = CacheExecutionOptions(copy_follower_records_asynchronously=True, follower_copy_mode='layer')
    assert legacy.prefetch is None and legacy.copy_follower_records_asynchronously
    disabled = replace(legacy, prefetch=PrefetchPolicy(enabled=False, completion='layer'))
    assert not disabled.copy_follower_records_asynchronously
    assert disabled.follower_copy_mode == 'group'


@pytest.mark.parametrize('arguments', [dict(enabled=1), dict(completion='batch')])
def test_invalid_prefetch_policies_fail_without_device_dependencies(arguments):
    with pytest.raises(ValueError):
        PrefetchPolicy(**arguments)
    with pytest.raises(TypeError, match='PrefetchPolicy'):
        CacheExecutionOptions(prefetch=True)


@pytest.mark.parametrize('backend', ['contiguous_records', 'native_paged'])
def test_group_and_prefetch_configuration_reaches_the_actual_backend_constructor(backend):
    spec = grouped_specification()
    options = CacheExecutionOptions(record_storage_backend=backend, prefetch=PrefetchPolicy(enabled=True))
    bindings = LocalCacheBindings('cuda:0', Mock(), Mock(), options)
    engine = Mock(gpu=tuple(object() for _ in range(6)))
    target = ('ulacache.cache.local_cache.UnifiedLocalResidency' if backend == 'contiguous_records'
        else 'ulacache.cache.local_paged_cache.NativePagedResidency')
    with patch('ulacache.compiler.configure_compiler'):
        with patch(target, return_value=engine) as constructor:
            cache = init(spec, bindings)
    assert isinstance(cache._slot_manager, SlotManager)
    assert isinstance(cache._access_engine, AccessEngine)
    assert cache._slot_manager.backend is cache._access_engine.backend is engine
    assert cache._access_engine.layer_sharing is spec.layer_reuse
    assert cache._access_engine.producers == (0, 0, 0, 3, 3, 3)
    assert cache._access_engine.execution_options is options
    if backend == 'contiguous_records':
        call = constructor.call_args.kwargs
        assert call['async_followers'] and call['follower_copy_mode'] == 'group'
        assert call['groups'].followers == ((1, 2), (4, 5))
    else:
        actual_spec, actual_bindings, actual_plan = constructor.call_args.args
        assert actual_spec is spec and actual_bindings is bindings
        assert actual_bindings.execution_options.follower_copy_mode == 'layer'
        assert actual_plan.selection_producer_layers == (0, 3)
        assert actual_plan.residency_domain_indices_by_layer == (0, 0, 0, 1, 1, 1)
    replacement = replace(bindings, execution_options=replace(options, prefetch=PrefetchPolicy(enabled=False)))
    assert not replacement.execution_options.copy_follower_records_asynchronously
    assert cache._access_engine.execution_options.copy_follower_records_asynchronously
    cache.close()
    assert cache._access_engine.backend is cache._slot_manager.backend is None
    assert cache._access_engine.record_planes == ()


def test_sharing_index_storage_alone_does_not_enable_unknown_future_topk_prefetch():
    spec = replace(create_cache_specification(), layer_reuse=LayerReuseSpecification.from_layer_groups(3,
        index_groups=((0, 1, 2),)))
    bindings = LocalCacheBindings('cuda:0', Mock(), Mock(),
        CacheExecutionOptions(prefetch=PrefetchPolicy(enabled=True)))
    with patch('ulacache.cache.create.create_rank_local_execution_engine') as factory:
        with pytest.raises(ValueError, match='TopK selection group'):
            init(spec, bindings)
    factory.assert_not_called()


@pytest.mark.parametrize('unsupported', ['owner', 'group_completion', 'kv_alias', 'multiple_planes', 'launch'])
def test_unsupported_async_layouts_fail_before_a_device_backend_is_created(unsupported):
    spec = grouped_specification()
    options = CacheExecutionOptions.native_paged(prefetch=PrefetchPolicy(enabled=True))
    if unsupported == 'group_completion':
        options = replace(options, prefetch=PrefetchPolicy(enabled=True, completion='group'))
    elif unsupported == 'kv_alias':
        spec = replace(spec, layer_reuse=replace(spec.layer_reuse,
            key_value_storage_owners=(0, 0, 2, 3, 4, 5)))
    elif unsupported == 'multiple_planes':
        plane = KeyValueRecordPlane('value', 'test.value.v1', (128,))
        spec = replace(spec, storage_layout=replace(spec.storage_layout,
            record_planes=(*spec.storage_layout.record_planes, plane)))
    elif unsupported == 'launch':
        options = replace(options, follower_exchange_launch_dimensions=(64, 256))
    bindings = LocalCacheBindings('cuda:0', Mock(), Mock(), options)
    if unsupported == 'owner':
        bindings = RecordOwnerBindings('cuda:0', object(), Mock(), Mock(), Mock(),
            256, 64, True, execution_options=options)
    with patch('ulacache.cache.create.create_rank_local_execution_engine') as local:
        with patch('ulacache.cache.create.create_record_owner_execution_engine') as owner:
            with pytest.raises((ValueError, NotImplementedError)):
                init(spec, bindings)
    local.assert_not_called()
    owner.assert_not_called()


def test_native_tp_preserves_the_policy_until_native_lifecycle_construction():
    enabled = PrefetchPolicy(enabled=True)
    disabled = PrefetchPolicy(enabled=False)
    assert CacheOptions(prefetch=enabled).prefetch is enabled
    assert CacheOptions(prefetch=disabled).prefetch is disabled
    assert not bool(CacheOptions(prefetch=disabled).prefetch)
    assert CacheOptions(prefetch=True).prefetch is True
    assert CacheOptions(prefetch=False).prefetch is False
    for completion in ('layer', 'group'):
        policy = PrefetchPolicy(enabled=True, completion=completion)
        assert CacheOptions(prefetch=policy).prefetch is policy


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('completion', ['auto', 'group', 'layer'])
def test_native_tp_policy_survives_json_roundtrip(enabled, completion):
    from dataclasses import asdict
    import json

    options = CacheOptions(prefetch=PrefetchPolicy(enabled, completion))
    restored = CacheOptions(**json.loads(json.dumps(asdict(options))))
    assert restored == options
    assert bool(restored.prefetch) is enabled
    assert restored.prefetch.completion == completion
