"""Preserve native selections, fresh writes and bounded plan lifetimes."""
from types import SimpleNamespace

import pytest
import torch

from ulacache.cache.layers import LayerGroupPlan
from ulacache.cache.prefill.plans import (
    NativePrefillReadPlans, PrefillSequenceBudget, prefill_request_groups,
)
from ulacache.cache.prefill.window import prepare_compact_prefill_record_read_plan, run_compact_prefill_records


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    source = torch.arange(4 * 64 * 3, dtype=torch.int32).reshape(4, 64, 1, 3).bfloat16()
    current = torch.tensor([9, 3], dtype=torch.int32)
    layer = 0
    reads, prepared, outputs = [], [], []

    def fill(peers, needed, staging, fresh, destination, offset, *unused):
        values = source[layer, needed.long()].clone()
        # Cold backing deliberately lacks newly written values.
        for position in range(needed.numel()):
            if needed[position] in current:
                values[position].fill_(-1)
            if fresh[position] >= 0:
                values[position].copy_(source[layer, current[fresh[position]].long()])
        destination[:needed.numel()].copy_(values)
        reads.append((layer, needed.clone()))

    pool = SimpleNamespace(flowkv_layout=SimpleNamespace(prefill_window_rows=32, compression_ratio=1),
        flowkv_arena=SimpleNamespace(prefill=torch.empty((32, 1, 3), dtype=torch.bfloat16),
            staging=None, kernel=lambda: SimpleNamespace(fill_compact=fill)),
        flowkv_host=None, flowkv_request=None, _flowkv_mode='prefill',
        _flowkv_namespaces=1, _flowkv_storage_size=1, _flowkv_current_rows=current,
        flowkv_peers=SimpleNamespace(device_ptr=0), flowkv_rank=0,
        _flowkv_owner_start=0, start_layer=0, _flowkv_peer_offset=lambda local: local)

    def prepare(rows):
        result = prepare_compact_prefill_record_read_plan(pool, rows)
        prepared.append(result)
        return result

    def read(local, rows, consumer, *, output, read_plan):
        nonlocal layer
        layer = local
        return run_compact_prefill_records(pool, local, rows, consumer,
            output_buffer=output, read_plan=read_plan)

    def consume(window, indices):
        result = window[indices.long()].clone()
        outputs.append(result)
        return result

    cache = SimpleNamespace(prepare_native_prefill_read_plan=prepare, read_native_prefill_records=read)
    plans = NativePrefillReadPlans(cache, LayerGroupPlan.from_layer_anchors((0, 0, 2, 2)), 32)
    return SimpleNamespace(plans=plans, source=source, current=current, prepared=prepared,
        consume=consume, reads=reads, outputs=outputs)


def test_dense_plan_reuses_metadata_but_reads_each_layers_fresh_payload(runtime, monkeypatch):
    rows = torch.tensor([5, 3, 19, 9, 5, 2], dtype=torch.int32)
    monkeypatch.setattr(torch, 'cat', lambda *a, **k: pytest.fail('One admitted group must return directly'))
    for layer in range(4):
        result = runtime.plans.read(layer, rows, runtime.consume, request_lengths=(2, 0, 4))
        assert result is runtime.outputs[-1]
        assert torch.equal(result, runtime.source[layer, rows.long()])
    assert len(runtime.prepared) == 1
    assert [layer for layer, _ in runtime.reads] == [0, 1, 2, 3]


def test_small_topk_reads_only_its_union_and_reuses_one_shared_group(runtime):
    rows = torch.arange(1, 25, dtype=torch.int32)
    previous = None
    for layer in range(4):
        indices = torch.tensor([[8, 2, 8, -1]], dtype=torch.int32) + (1 if layer >= 2 else 0)
        if layer >= 2:
            indices[0, -1] = -1
        result, remap = runtime.plans.read_selected(layer, rows, indices, runtime.current,
            runtime.consume, request_lengths=(8, 16))
        valid = indices >= 0
        assert torch.equal(result[remap[valid].long()], runtime.source[layer, rows[indices[valid]].long()])
        assert torch.equal(remap < 0, ~valid)
        assert runtime.reads[-1][1].numel() == 2
        if layer % 2:
            assert runtime.plans.sparse is previous
        elif previous is not None:
            assert runtime.plans.sparse is not previous
        previous = runtime.plans.sparse
    assert runtime.plans.dense is None
    assert len(runtime.prepared) == 0


def test_large_selection_keeps_dense_native_order_and_next_forward_replans(runtime):
    rows = torch.tensor([3, 9, 2, 11], dtype=torch.int32)
    indices = torch.zeros((2, 4), dtype=torch.int32)
    result, remap = runtime.plans.read_selected(0, rows, indices, runtime.current,
        runtime.consume, request_lengths=(4,))
    assert remap is indices
    assert torch.equal(result, runtime.source[0, rows.long()])
    first = runtime.plans.dense
    runtime.plans.reset()
    runtime.current.copy_(runtime.current.flip(0))
    result = runtime.plans.read(0, rows, runtime.consume)
    assert runtime.plans.dense is not first
    assert torch.equal(result, runtime.source[0, rows.long()])
    assert len(runtime.prepared) == 2


def test_invalid_request_partition_and_full_history_overflow_are_rejected(runtime):
    for rows, lengths, message in ((torch.arange(1, 34), (33,), 'fixed buffer'),
            (torch.arange(1, 5), (2, 3), 'disagree'),
            (torch.arange(1, 5), (-1, 5), 'nonnegative')):
        with pytest.raises(ValueError, match=message):
            runtime.plans.read(0, rows, runtime.consume, request_lengths=lengths)
    assert not runtime.reads and not runtime.prepared


def test_admission_charges_full_histories_independently_of_new_token_chunks():
    budget = PrefillSequenceBudget(1048576)
    budget.reserve(900000, 8192)
    assert budget.used == 908192
    assert not budget.can_reserve(150000, 64)
    with pytest.raises(RuntimeError, match='did not fit'):
        budget.reserve(150000, 64)
    with pytest.raises(ValueError, match='exceeds'):
        budget.can_reserve(1048576, 1)
    assert list(prefill_request_groups((2, 0, 3, 6, 1), 5)) == [(0, 5), (5, 5), (10, 1), (11, 1)]
