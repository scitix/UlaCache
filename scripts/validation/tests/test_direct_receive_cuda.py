"""Real CUDA allocator/lease test; no model or RDMA bandwidth result."""

import argparse, json, time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def main(output, backend="local"):
    from ulacache.cache.local_cache import UnifiedLocalResidency
    from ulacache.cache.layers import LayerGroupPlan
    from ulacache.cache.kernels.local_cache import (
        acquire_receive_locations,
        register_receive_rows,
        release_receive_locations,
    )
    from sglang.srt.mem_cache.flowkv_host_pool import FlowKVHostRecordPool
    from sglang.kernels.jit.utils import load_jit
    from ulacache.compiler import configure_compiler

    assert torch.cuda.is_available()
    configure_compiler(load_jit)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    groups = LayerGroupPlan.from_layer_anchors([0, 0, 2])
    if backend == "native-paged":
        from ulacache import (
            CacheExecutionOptions,
            KeyValueRecordPlane,
            KeyValueStorageLayout,
            LayerReuseSpecification,
            LocalCacheBindings,
            MemoryTierCapacity,
            SparseAccessSpecification,
            UnifiedCacheSpecification,
            init,
        )
        from ulacache.transfer.direct_receiver import DirectRecordReceiver

        spec = UnifiedCacheSpecification(
            KeyValueStorageLayout(
                "test", (KeyValueRecordPlane("kv", "test.bytes.v1", (64,)),), 64
            ),
            LayerReuseSpecification.from_shared_sparse_selection([0, 0, 2]),
            SparseAccessSpecification(2, 4),
            MemoryTierCapacity(512, 512, 256),
        )
        options = CacheExecutionOptions(
            record_storage_backend="native_paged",
            prepare_residency_in_parallel=True,
            select_victims_in_parallel=True,
            copy_follower_records_asynchronously=True,
            follower_copy_mode="layer",
            exchange_launch_dimensions=(64, 256),
            follower_exchange_launch_dimensions=(64, 256),
        )
        handle = init(
            spec, LocalCacheBindings(device, FlowKVHostRecordPool, load_jit, options)
        )
        cache = handle._execution_engine
        storage = DirectRecordReceiver(handle).engine
    else:
        cache = UnifiedLocalResidency(
            groups=groups,
            gpu_slots=512,
            host_slots=512,
            row_bytes=64,
            max_queries=2,
            topk=4,
            max_write_rows=256,
            device=device,
            host_pool_factory=FlowKVHostRecordPool,
            parallel_prepare=True,
            parallel_victims=True,
            async_followers=True,
            exchange_launch=(64, 256),
            follower_exchange_launch=(64, 256),
            follower_copy_mode="layer",
        )
        storage = cache
    first = torch.arange(1, 129, dtype=torch.int64, device=device)
    cache.allocate_receive_rows(first)
    initial = acquire_receive_locations(first, storage.frames)
    assert bool((initial <= -2).all())
    # The free stack may yield an extent in either direction. The receiver
    # orders its unpublished CPU bindings before advertising wire addresses.
    assert bool((torch.diff(-initial - 2, dim=1).abs() == 1).all())
    for s in storage.states:
        assert int(s.leases.sum()) == 0
    release_receive_locations(first, storage.frames, initial, transport_quiesced=True)
    incoming = torch.arange(129, 641, dtype=torch.int64, device=device)
    cache.allocate_receive_rows(incoming)
    locations = acquire_receive_locations(incoming, storage.frames)
    assert ((locations >= 0).sum(dim=1) == 128).all().item()
    for g, s in enumerate(storage.states):
        assert int(s.leases.sum()) == 128
        gpu = locations[g][locations[g] >= 0].long()
        assert bool((storage.rings[g].gpu_bucket[gpu] == -1).all())
    remainder = torch.arange(641, 1024, dtype=torch.int64, device=device)
    register_receive_rows(remainder, storage.frames, prefer_gpu=True)
    # Initialize every bound row independently of the planner under test.
    logical = torch.arange(1, 1024, dtype=torch.int64, device=device)
    directory = storage.directory.index_select(1, logical).cpu()
    for layer in range(groups.layer_count):
        loc = directory[groups.group(layer)]
        hot = loc >= 0
        payload = (
            ((logical.cpu() + 13 * layer) % 251)
            .to(torch.uint8)[:, None]
            .expand(-1, 64)
            .contiguous()
        )
        storage.gpu[layer].index_copy_(0, loc[hot].to(device), payload[hot].to(device))
        storage.host.storage[layer].index_copy_(0, -loc[~hot] - 2, payload[~hot])
    torch.cuda.synchronize()
    committed = list(range(1, 129)) + list(range(641, 1024))
    current = torch.tensor([641, 642], dtype=torch.int64, device=device)
    cache.active.fill_(2)

    def forward(step):
        ids = [committed[(step * 8 + j) % len(committed)] for j in range(8)]
        selected = torch.tensor(ids, dtype=torch.int32, device=device).reshape(2, 4)
        cache.begin_forward()
        for layer in range(groups.layer_count):
            if layer in groups.anchors:
                cache.plan(layer, selected, current)
            mapped = cache.attention_rows(layer, 2)
            values = storage.gpu[layer][mapped.long()].cpu()
            expected = (
                ((selected.cpu().long() + 13 * layer) % 251)
                .to(torch.uint8)[:, :, None]
                .expand(-1, -1, 64)
            )
            assert torch.equal(values, expected), (layer, step)
            cache.after_attention(layer)

    started = time.monotonic()
    for step in range(300):
        forward(step)
    cache._drain_transfers()
    torch.cuda.synchronize()
    assert torch.equal(storage.directory.index_select(1, incoming), locations)
    for s in storage.states:
        assert int(s.leases.sum()) == 128
    # A failed/unknown remote transfer cannot authorize releasing destinations.
    try:
        release_receive_locations(
            incoming, storage.frames, locations, transport_quiesced=False
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Nonquiescent release was accepted")
    release_receive_locations(
        incoming, storage.frames, locations, transport_quiesced=True
    )
    torch.cuda.synchronize()
    for s in storage.states:
        assert int(s.leases.sum()) == 0
    for step in range(300, 428):
        forward(step)
    cache._drain_transfers()
    torch.cuda.synchronize()
    assert bool(
        (
            (storage.directory.index_select(1, incoming) != locations)
            & (locations >= 0)
        ).any()
    )
    result = dict(
        passed=True,
        backend=backend,
        device=torch.cuda.get_device_name(),
        cpu_ordered_allocation=True,
        gpu_fallback_rows_per_group=128,
        groups=groups.group_count,
        shared_layers=3,
        pinned_destination_checks_over_epochs=300,
        postrelease_forward_epochs=128,
        bytes_verified_during_native_replacement=True,
        async_followers=True,
        parallel_victims=True,
        quiescent_release_required=True,
        elapsed_seconds=time.monotonic() - started,
        rdma_tested=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    output.open("x").write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["local", "native-paged"], default="local")
    args = parser.parse_args()
    main(args.output, args.backend)
