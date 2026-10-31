"""Actual CUDA byte oracle for two full-size windows and read-only history."""
from pathlib import Path
import argparse
import gc
import json

import pytest


def check_current_specialization(width):
    """Changing extend lengths must reuse code and preserve historical bytes."""
    import torch
    from ulacache.cache.kernels.read_only_prefill import _current, install_current

    device = torch.device('cuda:0')
    counts = [1, 2, 7, 16, 31, 64, 65, 127, 257, 2048, 8191, 8192]
    cut = 512 if width == 576 else 528
    hashes = set()
    for count in counts:
        current = torch.arange(count, dtype=torch.int32, device=device) + 100
        rows = torch.cat((torch.full((17,), 3, dtype=torch.int32, device=device),
                          current, current.new_tensor([0, -1])))
        fresh = torch.cat((torch.full((17,), -1, dtype=torch.int32, device=device),
                           torch.arange(count, dtype=torch.int32, device=device),
                           current.new_tensor([-1, -1])))
        values = ((torch.arange(count, device=device)[:, None] * 31
                   + torch.arange(width, device=device)[None, :]) % 251).to(torch.uint8)
        # Even the one-row case keeps the same physical writer strides.
        nope = torch.empty((count, cut), dtype=torch.uint8, device=device)
        rope = torch.empty((count, width - cut), dtype=torch.uint8, device=device)
        nope.copy_(values[:, :cut])
        rope.copy_(values[:, cut:])
        window = torch.full((rows.numel(), width), 77, dtype=torch.uint8, device=device)
        expected = window.clone()
        expected[17:17 + count] = values
        install_current(rows, fresh, current, nope, rope, window, row_bytes=width)
        torch.cuda.synchronize()
        assert torch.equal(window, expected), (width, count)
        kernel = _current.warmup(
            rows, fresh, current, nope, rope, window, rows.numel(), count,
            nope.stride(0), rope.stride(0), cut // 8, width, 8, 128,
            grid=(min(64, (rows.numel() + 7) // 8),), num_warps=8)
        hashes.add(kernel.hash)
    assert len(hashes) == 1, 'Extend lengths must not create new CUDA modules'
    return dict(row_bytes=width, counts=counts, byte_equal=True,
                compiled_variants=len(hashes))


@pytest.mark.parametrize('width', [576, 656])
def test_current_prefill_reuses_compiled_kernel(width):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    assert check_current_specialization(width)['compiled_variants'] == 1



def check(width):
    import torch
    from sglang.kernels.jit.utils import load_jit
    from sglang.srt.mem_cache.flowkv_host_pool import FlowKVHostRecordPool
    from ulacache.compiler import configure_compiler
    from ulacache.cache.layers import LayerGroupPlan
    from ulacache.cache.local_cache import UnifiedLocalResidency
    from ulacache.cache.prefill.reader import UnifiedPrefillMaterializer, prepare_read_plan
    from ulacache.transfer.prefill_pipeline import LayerPrefillPipeline, allocate_native_prefill_windows

    configure_compiler(load_jit)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    groups=LayerGroupPlan.from_layer_anchors([0,0,2,3])
    engine=UnifiedLocalResidency(groups=groups,gpu_slots=256,host_slots=1024,
        row_bytes=width,max_queries=2,topk=4,max_write_rows=64,
        device=device,host_pool_factory=FlowKVHostRecordPool,
        prefill_exchange_launch=(64,256))
    assert engine.frames.shape[1]==31 and engine.host.ptrs.dtype==torch.uint64
    rows=torch.arange(1,1250,dtype=torch.int64,device=device)
    engine.allocate_unbound(rows,prefer_gpu=False)

    def payload(ids,layer):
        return ((ids[:,None]*7+layer*13+torch.arange(width,device=device)[None,:]*11)%251).to(torch.uint8)

    for layer in range(groups.layer_count):
        locations=engine.directory[groups.group(layer),rows]
        gpu=locations>=0
        data=payload(rows,layer)
        engine.gpu[layer].index_copy_(0,locations[gpu],data[gpu])
        engine.host.storage[layer].index_copy_(0,(-locations[~gpu]-2).cpu(),data[~gpu].cpu())
    torch.cuda.synchronize()
    current=rows[-7:].contiguous()
    full=torch.cat((rows[:-7].flip(0),rows[:50],rows.new_tensor([0,-1])))
    plan=prepare_read_plan(full,current)
    assert plan.record_count==1249 and plan.record_count>engine.gpu_slots
    snapshots=[(tensor,tensor.clone()) for state in engine.states for tensor in state]
    snapshots += [(ring.gpu_bucket,ring.gpu_bucket.clone()) for ring in engine.rings]
    gpu_before,host_before=engine.gpu.clone(),engine.host.storage.clone()
    materializer=UnifiedPrefillMaterializer(engine)
    windows=allocate_native_prefill_windows(device,row_bytes=width)
    assert all(w.nbytes==1048576*width for w in windows)
    pipeline=LayerPrefillPipeline(windows,gather_history=materializer.gather_history,
                                 install_current=materializer.install_current,row_bytes=width)

    def begin():
        materializer.begin()
        pipeline.begin(plan,layer_count=groups.layer_count,record_count=plan.record_count,
                       keepalive_tensors=plan.keepalive_tensors)

    def consume(layer):
        pipeline.prepare(layer)
        data=payload(current,layer)
        cut=512 if width==576 else 528
        materializer.record_current(layer,current,data[:,:cut].contiguous(),data[:,cut:].contiguous())
        window=pipeline.read(layer)
        actual=window.reshape(-1,width)[:plan.record_count]
        assert torch.equal(actual,payload(plan.record_ids.long(),layer)),(width,layer)
        assert pipeline.read(layer).data_ptr()==window.data_ptr()
        pipeline.after_attention(layer)

    begin()
    for layer in range(groups.layer_count):consume(layer)
    pipeline.finish();materializer.finish()
    # Retire a partial forward, then reuse both windows through another wrap.
    begin();consume(0);pipeline.abort();materializer.finish()
    begin()
    for layer in range(groups.layer_count):consume(layer)
    pipeline.finish();materializer.finish();torch.cuda.synchronize()
    for original,before in snapshots:assert torch.equal(original,before)
    assert torch.equal(engine.gpu,gpu_before) and torch.equal(engine.host.storage,host_before)
    engine.release(rows);torch.cuda.synchronize()
    assert bool((engine.directory[:,rows]==-1).all())
    result=dict(passed=True,row_bytes=width,history_records=plan.record_count,
        gpu_records=engine.gpu_slots,two_window_bytes=[w.nbytes for w in windows],
        pointer_table_dtype=str(engine.host.ptrs.dtype),layers=groups.layer_count,
        shared_index_groups=groups.group_count,canonical_payload_and_metadata_unchanged=True,
        abort_and_reuse=True,release=True,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    engine.close()
    return result


@pytest.mark.parametrize('width', [576, 656])
def test_read_only_prefill_windows_preserve_canonical_records(width):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    result = check(width)
    assert result['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results=[]
    for width in [576,656]:
        results.append(dict(check(width), current_specialization=check_current_specialization(width)))
        gc.collect()
    value=dict(passed=True,cases=results,scope='Actual library CUDA history/current copies and event lifetimes; no model accuracy or timing claim')
    args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
