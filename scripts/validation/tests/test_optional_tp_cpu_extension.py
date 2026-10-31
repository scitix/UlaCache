"""Check native TP startup gating without allocating GPU or pinned storage."""

import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ulacache.cache_config import CacheOptions, PrefetchPolicy
from ulacache.storage.workspace import PlainPrefillLayout


def configure_pool():
    # Execute the real startup body with resource factories replaced below.
    # This is a host-side construction check, not CUDA kernel validation.
    from dataclasses import replace

    path = Path(__file__).resolve().parents[3] / 'ulacache/cache/tp_cache.py'
    node = next(node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == 'configure_pool')
    namespace = {'replace': replace, 'torch': sys.modules['torch']}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['configure_pool']


@pytest.mark.parametrize('cpu_offload', [False, True])
@pytest.mark.parametrize('cpu_policy', ['index', 'request'])
@pytest.mark.parametrize('prefill_window_rows', [0, 512])
@pytest.mark.parametrize('local_eviction_policy', ['age_lru', 'circular'])
@pytest.mark.parametrize('completion', ['group', 'layer'])
@pytest.mark.parametrize('prefetch_enabled', [False, True])
def test_owner_storage_and_host_planning_follow_the_cpu_extension(
        monkeypatch, cpu_offload, cpu_policy, prefill_window_rows, local_eviction_policy,
        completion, prefetch_enabled):
    allocations = []
    def tensor(shape, **kwargs):
        allocations.append(shape)
        result = Mock(shape=shape, device='cuda:0')
        result.unbind.return_value = [object(), object()]
        return result

    torch = SimpleNamespace(zeros=tensor, empty=tensor, arange=Mock(),
        uint8='uint8', int32='int32', int64='int64',
        cuda=SimpleNamespace(get_device_properties=lambda device:
            SimpleNamespace(multi_processor_count=132, max_threads_per_multi_processor=2048),
            Stream=Mock(return_value=object()), Event=Mock(side_effect=lambda: object())))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    kernel = Mock()
    record_factory = Mock(return_value=kernel)
    arena = SimpleNamespace(kernel=lambda: kernel)
    host = Mock()
    host.ready_peers.ptrs.cpu.return_value.tolist.return_value = []
    host.miss_ready_peers.ptrs.cpu.return_value.tolist.return_value = []
    host_factory, request_factory = Mock(return_value=host), Mock(return_value=Mock())
    publication = Mock()
    for name, module in {
        'ulacache.storage.kernels.records': SimpleNamespace(flowkv_record_module=record_factory),
        'ulacache.eviction.kernels.local_lru': SimpleNamespace(flowkv_record_policy_module=record_factory),
        'ulacache.cache.attention_buffers': SimpleNamespace(FlowKVC4Arena=Mock(return_value=arena)),
        'ulacache.transfer.ipc': SimpleNamespace(FlowKVPeerShards=Mock(), FlowKVStepPublication=publication),
        'ulacache.cache.host': SimpleNamespace(SharedWorkspaceHostTier=host_factory),
        'ulacache.cache.request_storage': SimpleNamespace(FlowKVRequestTier=request_factory),
        'ulacache.cache.request_access': SimpleNamespace(RequestAddressRuntime=Mock()),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    layout = PlainPrefillLayout(layers=2, groups=1, max_batch=1, pool_rows=1536,
        row_bytes=656, disaggregation_mode='null', topk=256, max_query_tokens=1,
        history_slots=256, page_size=64, native_segment_bytes=(656,),
        layer_anchors=(0, 0), max_staging_tokens=64, prefill_window_rows=prefill_window_rows)
    pool = SimpleNamespace(flowkv_layout=layout, device='cuda:0', allocation_page_size=64)
    options = CacheOptions(cpu_offload=cpu_offload, cpu_slots=64, cpu_policy=cpu_policy,
        local_eviction_policy=local_eviction_policy,
        prefetch=PrefetchPolicy(prefetch_enabled, completion))
    services = SimpleNamespace(config=options, degree=8, namespaces=1,
        group=SimpleNamespace(rank_in_group=0, cpu_group=object()),
        memory_region=nullcontext, ipc_manager_factory=Mock(),
        host_pool_factory=Mock(), host_memory_budget_bytes=Mock(), window_initializer=None)
    configure = configure_pool()
    if cpu_offload and cpu_policy == 'request' and prefill_window_rows:
        with pytest.raises(NotImplementedError, match='Compact Plain prefill'):
            configure(pool, services, step_publication=True)
        assert not allocations
        host_factory.assert_not_called()
        request_factory.assert_not_called()
        return
    configure(pool, services, step_publication=True)
    # Execute the same lifecycle construction as the frozen native factory.
    # All storage constructors above are CPU recorders; no device is initialized.
    from ulacache.cache.prefetch import SharedIndexLifecycle

    assert pool.flowkv_prefetch is prefetch_enabled
    assert pool.flowkv_prefetch_policy is options.prefetch
    lifecycle = SharedIndexLifecycle(layout.layer_groups, pool, prefetch=pool.flowkv_prefetch)
    assert lifecycle.prefetch is prefetch_enabled and lifecycle.completion == completion
    if prefetch_enabled:
        torch.cuda.Stream.assert_called_once_with(device=pool.device)
    else:
        torch.cuda.Stream.assert_not_called()
        torch.cuda.Event.assert_not_called()

    # CPU capacity settings cannot shrink the owner GPU bank when disabled.
    assert pool.flowkv_shards.shape == (2, 128 if cpu_offload else 192, 656)
    assert pool._flowkv_storage_size == 8 and pool._flowkv_namespaces == 1
    compiled = record_factory.call_args.kwargs
    if local_eviction_policy != 'circular' or prefill_window_rows:
        assert compiled['local_eviction_policy'] == local_eviction_policy
    else:
        assert 'local_eviction_policy' not in compiled
    assert compiled['host_tier'] is (cpu_offload and cpu_policy == 'index')
    assert compiled.get('request_tier', False) is (cpu_offload and cpu_policy == 'request')
    if cpu_offload and cpu_policy == 'index':
        host_factory.assert_called_once()
        request_factory.assert_not_called()
        assert pool.flowkv_host is host
        assert publication.call_args.kwargs['host_groups'] is host.fused_metadata
    elif cpu_offload:
        request_factory.assert_called_once()
        host_factory.assert_not_called()
        assert pool.flowkv_host is None
    else:
        host_factory.assert_not_called()
        request_factory.assert_not_called()
        assert pool.flowkv_host is None and pool.flowkv_request is None
        assert publication.call_args.kwargs['host_groups'] is None


@pytest.mark.parametrize('cpu_offload', [False, True])
def test_jit_omits_host_residency_without_removing_gpu_cache_policy(monkeypatch, cpu_offload):
    from ulacache.eviction.kernels import local_lru

    monkeypatch.setattr(local_lru, 'load_jit', lambda *args, **kwargs: kwargs)
    compiled = local_lru.flowkv_record_policy_module.__wrapped__(topk=256,
        max_queries=1, page_size=64, bytes_per_page=64 * 656, peers=8,
        slots_per_rank=192, row_bytes=656, native_segment_bytes=(656,),
        step_publication=True, compact_prefill=True,
        host_tier=cpu_offload)
    assert ('-DFLOW_RECORD_HOST_TIER=1' in compiled['extra_cuda_cflags']) is cpu_offload
    assert ('host_publish' in dict(compiled['cuda_wrappers'])) is cpu_offload
    assert '-DFLOW_PARALLEL_CIRCULAR=0' in compiled['extra_cuda_cflags']
    assert 'publish_step' in dict(compiled['cuda_wrappers'])
