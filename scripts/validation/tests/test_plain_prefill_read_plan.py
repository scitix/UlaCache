"""Reuse only prefix metadata; independent layers still read their own bytes."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

import ulacache.cache.prefill.window as plain_prefill


def test_one_read_plan_preserves_independent_payloads_and_fresh_writes_across_layers(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    selected = torch.tensor([19, 2, 19, 7, 14, 1, 11, 5, 13, 4], dtype=torch.int32)
    current = torch.tensor([19, 11], dtype=torch.int32)
    source = torch.arange(78 * 32 * 3, dtype=torch.float32).to(torch.bfloat16).reshape(78, 32, 1, 3)
    window = torch.empty((4, 1, 3), dtype=source.dtype)
    active_layer = 0
    fill_count = 0
    planning_calls = []
    original_union = plain_prefill._shared_cpu_union

    def shared_union(*arguments):
        planning_calls.append(True)
        return original_union(*arguments)

    monkeypatch.setattr(plain_prefill, '_shared_cpu_union', shared_union)

    def fill(peer_table, needed, staging, fresh, output, *unused):
        nonlocal fill_count
        fill_count += 1
        cold_records = source[active_layer].clone()
        cold_records[current.long()] = -1
        values = cold_records.index_select(0, needed.long())
        staged_records = source[active_layer].index_select(0, current.long())
        for index in range(needed.numel()):
            if fresh[index] >= 0:
                values[index].copy_(staged_records[int(fresh[index])])
        output[:needed.numel()].copy_(values)

    pool = SimpleNamespace(flowkv_layout=SimpleNamespace(prefill_window_rows=4, compression_ratio=1),
        flowkv_arena=SimpleNamespace(prefill=window, staging=None,
            kernel=lambda: SimpleNamespace(fill_compact=fill)),
        flowkv_host=None, flowkv_request=None, _flowkv_mode='prefill',
        _flowkv_namespaces=1, _flowkv_storage_size=1, _flowkv_current_rows=current,
        flowkv_peers=SimpleNamespace(device_ptr=0), flowkv_rank=0,
        _flowkv_owner_start=0, start_layer=0, _flowkv_peer_offset=lambda layer: layer)
    read_plan = plain_prefill.prepare_compact_prefill_record_read_plan(pool, selected)
    assert len(read_plan.windows) == len(planning_calls) == 3
    assert all(tensor.dtype == torch.int32 for item in read_plan.windows
        for tensor in vars(item).values())
    assert all(tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        for item in read_plan.windows for tensor in vars(item).values())
    retained_bytes = sum(tensor.untyped_storage().nbytes()
        for item in read_plan.windows for tensor in vars(item).values())
    assert retained_bytes <= selected.numel() * 12

    def consume(pages, indices, output):
        torch.index_select(pages, 0, indices.long(), out=output)

    for active_layer in range(78):
        destination = torch.empty((selected.numel(), 1, 3), dtype=source.dtype)
        plain_prefill.run_compact_prefill_records(pool, active_layer, selected, consume,
            output_buffer=destination, read_plan=read_plan)
        expected = source[active_layer].index_select(0, selected.long())
        assert torch.equal(destination.view(torch.uint8), expected.view(torch.uint8))
    assert len(planning_calls) == 3
    assert fill_count == 78 * 3
