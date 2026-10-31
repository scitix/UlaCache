"""Small CUDA completion gate, invoked explicitly by the reviewed GPU runner.

This tests real stream/event ordering and public TP adoption with bounded copy
payloads. It does not replace the native writer/attention numerical fixtures.
Importing this module neither imports Torch nor initializes CUDA.
"""


def run_prefetch_completion_fixture(device, output_directory=None):
    import hashlib
    import inspect
    import json
    import os
    from pathlib import Path
    import socket
    from types import SimpleNamespace

    import torch

    from ulacache import (CacheExecutionOptions, KeyValueRecordPlane,
        KeyValueStorageLayout, LayerReuseSpecification, MemoryTierCapacity,
        PrefetchPolicy, SparseAccessSpecification, TensorParallelCacheBindings,
        UnifiedCacheSpecification, init)
    from ulacache.cache.layers import LayerGroupPlan
    from ulacache.cache.prefetch import SharedIndexLifecycle

    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('This fixture requires an explicitly selected CUDA device')
    torch.cuda.set_device(device)
    allocated_before = torch.cuda.memory_allocated(device)
    groups = LayerGroupPlan.from_layer_anchors((0, 0, 0, 3, 3, 5))
    layers, maximum_rows, row_bytes = groups.layer_count, 8, 656
    rows = []
    for completion in ('group', 'layer'):
        # All event, stream, tensor and public-cache construction precedes
        # capture. Replays vary the device active count and complete payloads.
        backing = torch.zeros((layers, 64, row_bytes), dtype=torch.uint8, device=device)
        source = torch.empty((layers, maximum_rows, row_bytes), dtype=torch.uint8, device=device)
        observed = torch.empty_like(source)
        active = torch.empty((), dtype=torch.int32, device=device)
        row_ids = torch.arange(maximum_rows, device=device)
        pool = SimpleNamespace(device=device, flowkv_layout=SimpleNamespace(
            layers=layers, layer_groups=groups, row_bytes=row_bytes, page_size=64,
            max_query_tokens=maximum_rows, topk=1, max_staging_tokens=maximum_rows),
            flowkv_arena=SimpleNamespace(storage=backing,
                valid=torch.zeros(1, dtype=torch.int32, device=device)),
            flowkv_shards=backing, flowkv_host=None, flowkv_request=None,
            _flowkv_storage_size=8, _flowkv_namespaces=1,
            flowkv_step_publication=SimpleNamespace(publish=lambda: None),
            flowkv_prefetch_policy=PrefetchPolicy(True, completion))
        lifecycle = SharedIndexLifecycle(groups, pool, prefetch=True)
        pool.flowkv_shared_index = lifecycle
        transfer = lifecycle._stream
        events = tuple(event for event in (
            *lifecycle._anchor_ready, *lifecycle._followers_ready,
            *lifecycle._layer_ready) if event is not None)
        assert len({id(event) for event in events}) == len(events)
        if completion == 'layer':
            assert sum(event is not None for event in lifecycle._layer_ready) == 3

        def plan(layer, selected):
            group = groups.group(layer)
            assert group not in lifecycle._planned
            lifecycle._planned.add(group)

        def pull(layer):
            # CUDA work is intentionally tiny; the actual production lifecycle
            # determines stream, record and wait placement around these copies.
            backing[layer, :maximum_rows].copy_(
                torch.where((row_ids < active)[:, None], source[layer], 0))

        lifecycle._plan_index, lifecycle._pull_history = plan, pull
        spec = UnifiedCacheSpecification(
            KeyValueStorageLayout('history', (KeyValueRecordPlane(
                'kv', 'mla.scaled656.fp8.v1', (528, 128)),), 64),
            LayerReuseSpecification.from_shared_sparse_selection((0, 0, 0, 3, 3, 5)),
            SparseAccessSpecification(maximum_rows, 1),
            MemoryTierCapacity(64, 0, maximum_rows))
        cache = init(spec, TensorParallelCacheBindings(pool, owns_storage=False,
            execution_options=CacheExecutionOptions(record_storage_backend='tensor_parallel',
                prefetch=PrefetchPolicy(True, completion))))
        assert cache._execution_engine.pool is pool
        assert pool.flowkv_shared_index is lifecycle and lifecycle._stream is transfer
        compute = torch.cuda.Stream(device=device)
        selected = torch.empty((maximum_rows, 1), dtype=torch.int32, device=device)
        compute.wait_stream(torch.cuda.current_stream(device))

        def forward():
            cache.begin_model_forward()
            for layer in range(layers):
                cache.prepare_layer_execution(layer)
                if layer in groups.anchors:
                    lifecycle.on_index_ready(layer, selected)
                lifecycle.before_history_consume(layer)
                observed[layer].copy_(backing[layer, :maximum_rows])
                cache.finish_layer_attention(layer)

        template = torch.arange(source.numel(), dtype=torch.int32, device='cpu').reshape(source.shape)
        cases = []

        def execute(active_rows, sequence, graph=None):
            expected = ((template * 7 + sequence * 19) % 251).to(torch.uint8)
            expected[:, active_rows:] = 0
            with torch.cuda.stream(compute):
                source.copy_(expected)
                active.fill_(active_rows)
                observed.fill_(255)
                if graph is None:
                    forward()
                else:
                    graph.replay()
            compute.synchronize()
            torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
            assert lifecycle._stream is transfer
            assert events == tuple(event for event in (
                *lifecycle._anchor_ready, *lifecycle._followers_ready,
                *lifecycle._layer_ready) if event is not None)
            cases.append(dict(active_rows=active_rows, mode='graph' if graph else 'eager',
                complete_payload_bytes_compared=expected.numel()))

        # Vary input values as well as active rows; stale copies cannot pass.
        for sequence, count in enumerate((1, 8, 0, 3)):
            execute(count, sequence)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=compute):
            forward()
        for sequence, count in enumerate((8, 1, 3, 0, 2), start=10):
            execute(count, sequence, graph)
        torch.cuda.synchronize(device)
        cache.close()
        assert pool.flowkv_shared_index is lifecycle and lifecycle._stream is transfer
        rows.append(dict(completion=completion, passed=True, cases=cases,
            one_shared_lifecycle=True, one_transfer_stream=True,
            all_events_created_before_adoption_and_capture=True,
            distinct_follower_events=3 if completion == 'layer' else 0))
        del graph

    root = Path(inspect.getfile(SharedIndexLifecycle)).resolve().parents[2]
    files = ('ulacache/cache/prefetch.py', 'ulacache/cache/tensor_parallel.py',
             'ulacache/cache_config.py', 'ulacache/cache/tp_cache.py')
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    result = dict(complete=True, passed=True, cases=rows, pid=os.getpid(),
        start_ticks=int(stat[19]), hostname=socket.gethostname(),
        device=str(device), device_uuid=str(torch.cuda.get_device_properties(device).uuid),
        allocated_before=allocated_before, allocated_after=torch.cuda.memory_allocated(device),
        peak_allocated=torch.cuda.max_memory_allocated(device),
        production_lifecycle_with_bounded_copy_payloads=True,
        native_attention_numerics_tested=False,
        fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_files={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in files})
    if output_directory is not None:
        path = Path(output_directory) / 'PREFETCH_COMPLETION.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(result, stream, indent=2)
    return result
