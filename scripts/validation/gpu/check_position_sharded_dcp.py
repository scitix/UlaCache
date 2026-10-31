"""Eight-GPU payload and native FlashMLA DCP checks, without a model load.

This is NOT a whole-model overhead qualification. The supervisor performs the
sole known-empty GPU check before torchrun. No worker has a kill timer.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")

    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.layers.attention.dsa.dsa_dcp import DcpPageTableCache
    from sglang.srt.layers.dcp.comm import dcp_a2a_lse_reduce
    from sglang.srt.mem_cache.flowkv_host_pool import FlowKVHostRecordPool
    from ulacache import CacheExecutionOptions, SparseAttentionAccessPlan
    from ulacache import PositionShardLayout
    from ulacache.transfer.position_prefill import PositionPrefillMaterializer
    from sglang.srt.ulacache.context_parallel_attention import run_position_prefill_attention
    from ulacache.cache.prefill.reader import prepare_read_plan
    from ulacache.transfer.prefill_pipeline import allocate_native_prefill_windows, LayerPrefillPipeline
    from sglang.srt.ulacache.context_parallel_adapter import exchange_prefill_queries
    from sglang.srt.ulacache.standalone_cache_binding import create_native_latent_attention_cache

    degree, queries, topk, heads = dist.get_world_size(), 2, 2048, 128
    assert degree == 8
    device = torch.device("cuda", rank)
    group = GroupCoordinator([list(range(degree))], rank, "nccl", use_pynccl=True,
        use_mscclpp=False, use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False, use_hpu_communicator=False,
        use_xpu_communicator=False, use_npu_communicator=False,
        group_name="ulacache-position-dcp-validation")
    layout = PositionShardLayout(rank, degree, 4096, 4096)
    cache = create_native_latent_attention_cache(
        layer_selection_owners=(0, 0), bytes_per_record=656, records_per_page=64,
        device_record_capacity=4096, host_record_capacity=4096,
        maximum_padded_query_count=queries, maximum_selected_records=topk,
        maximum_write_record_count=8192, maximum_prefill_union_record_count=4096,
        device=device, allocate_host_record_storage=FlowKVHostRecordPool,
        compile_device_module=load_jit, position_layout=layout,
        execution_options=CacheExecutionOptions(prepare_residency_in_parallel=True,
            select_victims_in_parallel=True, copy_follower_records_asynchronously=True,
            exchange_launch_dimensions=(64, 256),
            follower_exchange_launch_dimensions=(64, 256)))
    engine = cache._execution_engine
    print(f"rank{rank}: cache constructed", flush=True)
    engine.local_engine.collect_stats = True
    all_ids = torch.arange(layout.allocation_page_size, layout.global_records - 1024,
        device=device, dtype=torch.int32)
    cache.register_logical_records(all_ids, prefer_device=False)
    torch.manual_seed(92181)
    encoded = []
    for layer in range(2):
        nope = torch.randn((all_ids.numel(), 512), device=device, dtype=torch.bfloat16) * .1
        rope = torch.randn((all_ids.numel(), 64), device=device, dtype=torch.bfloat16) * .1
        n, r = quantize_k_cache_separate(nope, rope)
        assert n.dtype == r.dtype == torch.uint8
        payload = torch.cat((n.reshape(-1, 528), r.reshape(-1, 128)), dim=-1)
        full = torch.zeros((layout.global_records, 656), device=device, dtype=torch.uint8)
        full[all_ids.long()] = payload
        encoded.append(full)
        cache.write_native_record_segments(layer, all_ids, (n, r))
    torch.cuda.synchronize()

    # The same read-only owner callbacks also support PCP assembly. Validate
    # both exact encoded reconstruction and that it leaves residency untouched.
    windows = allocate_native_prefill_windows(device)
    before = engine.directory.clone()
    plan = prepare_read_plan(all_ids, all_ids[:0])
    def assemble(view):
        work = dist.all_reduce(view, op=dist.ReduceOp.SUM,
            group=group.device_group, async_op=True)
        work.wait()
    materializer = PositionPrefillMaterializer(engine, assemble)
    materializer.begin()
    for layer in range(2):
        materializer.gather_history(layer, plan, windows[layer])
        torch.testing.assert_close(windows[layer].view(-1, 656)[:all_ids.numel()],
            encoded[layer][all_ids.long()], rtol=0, atol=0)
    torch.testing.assert_close(before, engine.directory, rtol=0, atol=0)
    fresh_ids = all_ids[-1024:]
    pipeline_plan = prepare_read_plan(all_ids, fresh_ids)
    pipeline = LayerPrefillPipeline(windows, gather_history=materializer.gather_history,
        install_current=materializer.install_current)
    materializer.begin()
    pipeline.begin(pipeline_plan, layer_count=2, record_count=pipeline_plan.record_count,
        keepalive_tensors=pipeline_plan.keepalive_tensors)
    for lid in range(2):
        pipeline.prepare(lid)
        values = encoded[lid][fresh_ids.long()]
        materializer.record_current(lid, fresh_ids, values[:, :528], values[:, 528:])
        actual = pipeline.read(lid).view(-1, 656)[:all_ids.numel()]
        torch.testing.assert_close(actual, encoded[lid][all_ids.long()], rtol=0, atol=0)
        pipeline.after_attention(lid)
    pipeline.finish()
    materializer.finish()
    torch.testing.assert_close(before, engine.directory, rtol=0, atol=0)
    del before
    print(f"rank{rank}: exact history reconstruction passed", flush=True)

    selected = torch.full((queries, topk), -1, device=device, dtype=torch.int32)
    # Parallel registration does not promise logical-ID allocation order.
    # Bind a guaranteed CPU-miss selection to the actual canonical directory.
    host_local = torch.nonzero(engine.directory[0] <= -2).flatten()[:384]
    assert host_local.numel() == 384
    host_global = (host_local * degree + rank).to(torch.int32).contiguous()
    host_by_owner = torch.empty(degree * 384, device=device, dtype=torch.int32)
    dist.all_gather_into_tensor(host_by_owner, host_global, group=group.device_group)
    host_by_owner = host_by_owner.reshape(degree, 384)
    selected[0, :1536] = host_by_owner[:, :192].flatten()
    selected[1, :1536] = host_by_owner[:, 192:].flatten()
    current = torch.tensor([6000, 6001], device=device, dtype=torch.int32)
    cache.active_query_count_tensor.fill_(queries)
    q = torch.randn((queries, heads, 576), device=device, dtype=torch.bfloat16) * .1
    layer = SimpleNamespace(tp_q_head_num=heads, head_dim=576, layer_id=0)
    checks = []

    def backend(degree_, pool):
        b = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        b.dcp_size, b.dcp_rank = degree_, rank
        b._dcp_page_table_cache = DcpPageTableCache()
        b.flashmla_kv_num_q_heads = heads
        b.dsa_kv_cache_store_fp8 = True
        b.physical_page_size, b.kv_cache_dim = 64, 656
        b.flashmla_kv_format, b.dsa_index_topk = "V32", topk
        b.token_to_kv_pool, b.device = pool, device
        b._arange_buf = torch.arange(queries + 1, device=device, dtype=torch.int32)
        lengths = torch.full((queries,), layout.global_records, device=device, dtype=torch.int32)
        b.forward_metadata = SimpleNamespace(dsa_cache_seqlens_int32=lengths,
            flashmla_metadata=b._compute_flashmla_metadata(lengths, 1))
        return b

    reference = backend(1, SimpleNamespace())
    native = backend(degree, SimpleNamespace())
    own = backend(degree, SimpleNamespace(flowkv_position_sharded=True,
        flowkv_local_attention_counts=lambda _layer, rows: engine.selected_counts[0, :rows],
        flowkv_after_attention=cache.finish_layer_attention))
    native_kv = [data[rank::degree].contiguous().view(torch.float8_e4m3fn) for data in encoded]

    def read(b, kv, ids, lid):
        layer.layer_id = lid
        return b._forward_flashmla_kv(q, kv, 512, 192**-.5, layer,
            b.forward_metadata, ids)

    def merge(out, lse):
        return dcp_a2a_lse_reduce(out.reshape(queries, heads, 512).contiguous(),
            lse.reshape(queries, heads).contiguous(), group,
            is_lse_base_on_e=True, comm_backend="a2a")

    def check(actual, expected, name):
        error = (actual.float() - expected.float())
        relative = (error.norm() / expected.float().norm().clamp_min(1e-12)).item()
        assert torch.isfinite(actual).all() and relative < .012, (name, rank, relative)
        checks.append(dict(name=name, relative_l2=relative, max_abs=error.abs().max().item()))

    # Prefill retains TP projection weights and transposes query heads into
    # token shards. Exercise tails shorter than CP and a ragged multi-row case
    # against complete unsharded native attention on the same encoded history.
    communication_allocations = []
    for tokens in (1, 9, 1024):
        full_query = torch.randn((tokens, heads, 576), device=device, dtype=torch.bfloat16) * .1
        full_rows = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
        for row in range(tokens):
            full_rows[row, :1536] = torch.arange(512 + 64 * (row % 16), 2048 + 64 * (row % 16),
                device=device, dtype=torch.int32)
        window_rows = plan.map_native_rows(full_rows)
        lengths = torch.full((tokens,), 1536, device=device, dtype=torch.int32)
        full_layer = SimpleNamespace(tp_q_head_num=heads, head_dim=576, layer_id=0)
        for lid in range(2):
            full_layer.layer_id = lid
            def attend(query, rows, local_lengths):
                metadata = SimpleNamespace(dsa_cache_seqlens_int32=local_lengths,
                    flashmla_metadata=native._compute_flashmla_metadata(local_lengths, 1))
                out = native._forward_flashmla_kv(query,
                    windows[lid].view(torch.float8_e4m3fn), 512, 192**-.5,
                    full_layer, metadata, rows, dcp_partial=False)
                return out.reshape(query.shape[0], heads, 512)
            metadata = SimpleNamespace(dsa_cache_seqlens_int32=lengths,
                flashmla_metadata=reference._compute_flashmla_metadata(lengths, 1))
            expected = reference._forward_flashmla_kv(full_query,
                encoded[lid].view(torch.float8_e4m3fn), 512, 192**-.5,
                full_layer, metadata, full_rows).reshape(tokens, heads, 512)
            local_heads = heads // degree
            for transport in ('native_eager', 'native_dcp'):
                torch.cuda.synchronize()
                free_before = torch.cuda.mem_get_info()[0]
                exchange = (group.all_to_all_single if transport == 'native_eager' else
                    lambda output, input: exchange_prefill_queries(group, output, input))
                actual = run_position_prefill_attention(
                    full_query[:, rank * local_heads:(rank + 1) * local_heads], window_rows, lengths,
                    degree=degree, rank=rank, value_dimension=512,
                    exchange=exchange, attend=attend)
                torch.cuda.synchronize()
                communication_allocations.append(dict(tokens=tokens, layer=lid,
                    transport=transport, free_bytes_before=free_before,
                    free_bytes_after=torch.cuda.mem_get_info()[0]))
                check(actual, expected[:, rank * local_heads:(rank + 1) * local_heads],
                    f"context-prefill-{transport}-{tokens}-{lid}")
    del windows
    print(f"rank{rank}: ragged context-parallel prefill passed", flush=True)

    for step in range(3):
        cache.begin_model_forward()
        cache.prepare_sparse_attention(0, SparseAttentionAccessPlan(selected, current))
        local_rows = cache.get_resident_attention_inputs(0, queries).device_record_indices
        counts = engine.selected_counts[0].clone()
        expected_counts = ((selected > 0) & (selected % degree == rank)).sum(-1)
        torch.testing.assert_close(counts, expected_counts.to(torch.int32), rtol=0, atol=0)
        for lid in range(2):
            cache.prepare_layer_execution(lid)
            actual_payload = engine.gpu[lid][local_rows.clamp_min(0).long()]
            local_ids = engine.selected[0].to(torch.int64) * degree + rank
            expected_payload = encoded[lid][local_ids.clamp_min(0)]
            mask = local_rows >= 0
            torch.testing.assert_close(actual_payload[mask], expected_payload[mask], rtol=0, atol=0)
            expected = read(reference, encoded[lid].view(torch.float8_e4m3fn), selected, lid)
            expected = expected.reshape(queries, heads, 512)[
                :, rank * (heads // degree):(rank + 1) * (heads // degree)]
            native_out, native_lse = read(native, native_kv[lid], selected, lid)
            check(merge(native_out, native_lse), expected, f"native-dcp-{step}-{lid}")
            out, lse = read(own, engine.gpu[lid].view(torch.float8_e4m3fn), local_rows, lid)
            check(merge(out, lse), expected, f"ulacache-cp-{step}-{lid}")
        cache.finish_model_forward()
        selected[:, :1536].add_(8)

    # A rank with no local KV must contribute zero output and -inf LSE,
    # including physical CUDA-Graph padding rows.
    selected.fill_(-1)
    selected[0, :64] = torch.arange(512, 1024, 8, device=device, dtype=torch.int32)
    cache.active_query_count_tensor.fill_(1)
    cache.begin_model_forward()
    cache.prepare_sparse_attention(0, SparseAttentionAccessPlan(selected, current))
    rows = cache.get_resident_attention_inputs(0, queries).device_record_indices
    out, lse = read(own, engine.gpu[0].view(torch.float8_e4m3fn), rows, 0)
    assert torch.count_nonzero(out[1]) == 0 and torch.isneginf(lse[1]).all()
    if rank:
        assert torch.count_nonzero(out[0]) == 0 and torch.isneginf(lse[0]).all()
    cache.finish_layer_attention(1)
    cache.finish_model_forward()
    torch.cuda.synchronize()
    residency_counts = engine.local_engine.stats.cpu().tolist()
    assert sum(row[2] for row in residency_counts) > 0, residency_counts

    # Use native communicator capture scopes, as the model runner does.
    # Reuse the same stable selected/current pointers while their contents and
    # active query count change; replay must not retain an eager CPU decision.
    selected.fill_(-1)
    selected[0, :1536] = torch.arange(512, 2048, device=device, dtype=torch.int32)
    selected[1, :1536] = torch.arange(4096, 5632, device=device, dtype=torch.int32)
    cache.active_query_count_tensor.fill_(queries)
    engine.local_engine.collect_stats = False

    def cache_step():
        cache.begin_model_forward()
        cache.prepare_sparse_attention(0, SparseAttentionAccessPlan(selected, current))
        rows = cache.get_resident_attention_inputs(0, queries).device_record_indices
        outputs = []
        for lid in range(2):
            cache.prepare_layer_execution(lid)
            out, lse = read(own, engine.gpu[lid].view(torch.float8_e4m3fn), rows, lid)
            outputs.append(merge(out, lse))
        cache.finish_model_forward()
        return outputs

    def native_step():
        outputs = []
        for lid in range(2):
            out, lse = read(native, native_kv[lid], selected, lid)
            outputs.append(merge(out, lse))
        return outputs

    graphs = {}
    captured = {}
    for name, step in (("native_dcp", native_step), ("ulacache_cp", cache_step)):
        with group.graph_capture() as context:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=context.stream):
                outputs = step()
        torch.cuda.synchronize()
        graphs[name], captured[name] = graph, outputs
    for replay in range(3):
        selected[:, :1536].add_(8)
        for name, graph in graphs.items():
            graph.replay()
            for lid, actual in enumerate(captured[name]):
                expected = read(reference, encoded[lid].view(torch.float8_e4m3fn), selected, lid)
                expected = expected.reshape(queries, heads, 512)[
                    :, rank * (heads // degree):(rank + 1) * (heads // degree)]
                check(actual, expected, f"graph-{name}-{replay}-{lid}")
    cache.active_query_count_tensor.fill_(1)
    graphs['ulacache_cp'].replay()
    for output in captured['ulacache_cp']:
        assert torch.count_nonzero(output[1]) == 0
    cache.active_query_count_tensor.fill_(queries)
    timing = {name: [] for name in graphs}
    for repeat in range(3):
        for name in (list(graphs) if repeat % 2 == 0 else list(reversed(graphs))):
            dist.barrier()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                graphs[name].replay()
            stop.record()
            stop.synchronize()
            timing[name].append(start.elapsed_time(stop) / 50)
    print(f"rank{rank}: Graph replay and component timing passed", flush=True)
    result = dict(passed=True, rank=rank, world_size=degree, checks=checks,
        prefetched_full_history_bytes_exact=True, prefill_preserves_residency=True,
        host_expansion_exercised=True, actual_CPU_miss_counts=residency_counts,
        empty_owner_and_padded_query_checked=True, model_loaded=False,
        cuda_graph_dynamic_selections_checked=True, cuda_graph_dynamic_active_queries_checked=True,
        context_prefill_ragged_query_rows=[1, 9, 1024],
        native_query_heads=heads,
        two_window_actual_stream_pipeline_checked=True,
        prefill_transport_allocation_samples=communication_allocations,
        two_layer_component_graph_milliseconds=timing,
        whole_model_overhead_qualified=False, inventory=engine.memory_inventory())
    (args.out / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    cache.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
