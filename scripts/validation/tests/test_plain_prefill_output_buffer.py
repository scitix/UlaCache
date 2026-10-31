"""Bounded native prefix output preserves bytes and releases each chunk."""

from types import SimpleNamespace
import weakref

import pytest

torch = pytest.importorskip('torch')

from ulacache.cache.prefill.window import run_compact_prefill_records


@pytest.mark.parametrize('use_output_buffer', [False, True])
def test_ordered_prefix_across_windows_retains_only_required_output(monkeypatch, use_output_buffer):
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    source = torch.arange(96, dtype=torch.bfloat16).reshape(32, 1, 3)
    selected = torch.tensor([19, 2, 19, 7, 14, 1, 11, 5, 13, 4], dtype=torch.int32)
    window = torch.empty((4, 1, 3), dtype=source.dtype)
    released_chunks = []

    def fill(peer_table, needed, staging, fresh, output, *unused):
        output[:needed.numel()].copy_(source.index_select(0, needed.long()))

    arena = SimpleNamespace(prefill=window, staging=None,
        kernel=lambda: SimpleNamespace(fill_compact=fill))
    pool = SimpleNamespace(flowkv_layout=SimpleNamespace(prefill_window_rows=4, compression_ratio=1),
        flowkv_arena=arena, flowkv_host=None, flowkv_request=None,
        _flowkv_mode='prefill', _flowkv_namespaces=1, _flowkv_storage_size=1,
        _flowkv_current_rows=torch.empty(0, dtype=torch.int32),
        flowkv_peers=SimpleNamespace(device_ptr=0), flowkv_rank=0,
        _flowkv_owner_start=0, start_layer=0, layer_num=1,
        _flowkv_peer_offset=lambda layer: 0)

    def consume(pages, records, output_buffer=None):
        if use_output_buffer:
            assert all(reference() is None for reference in released_chunks)
        result = torch.index_select(pages, 0, records.long(), out=output_buffer)
        if use_output_buffer:
            assert result.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr()
        released_chunks.append(weakref.ref(result))
        return result

    expected = source.index_select(0, selected.long())
    destination = torch.empty_like(expected) if use_output_buffer else None
    if use_output_buffer:
        def reject_concatenation(*unused, **options):
            raise AssertionError('Streaming prefix output must not concatenate chunks')
        monkeypatch.setattr(torch, 'cat', reject_concatenation)
    result = run_compact_prefill_records(pool, 0, selected, consume,
        output_buffer=destination)
    assert torch.equal(result.view(torch.uint8), expected.view(torch.uint8))
    if use_output_buffer:
        assert result is destination
        assert all(reference() is None for reference in released_chunks)
