"""Encoded TP batches retain native writer routing and byte inputs on CPU.

Kernel recorders exercise dispatch and publication state only. They do not
establish CUDA kernel correctness or cross-rank ownership.
"""
from types import SimpleNamespace as NS

import pytest

from test_module_boundaries import DEVICE_AND_FRAMEWORK, isolated

torch = pytest.importorskip('torch')

from ulacache.storage.writers import write_native_record_segments


def fixture(*, mode='decode', compact=False, row_bytes=656,
            dtype=torch.float8_e4m3fn, host=False, request=False):
    calls = []
    backing = torch.zeros((2, 32, row_bytes), dtype=torch.uint8)
    groups = {i: NS(cursors=object(), totals=object(), hints=object()) for i in range(2)}
    arena = NS(storage=torch.zeros((2, 20, row_bytes), dtype=torch.uint8),
               staging=torch.zeros((4, row_bytes), dtype=torch.uint8),
               prefill=torch.zeros((32, row_bytes), dtype=torch.uint8),
               groups=groups, active=object())

    def writer(name):
        def write(*args):
            calls.append((name, args))
            latent, rope, ids, _, resident, current = args[:6]
            encoded = torch.cat((latent.reshape(ids.numel(), -1).view(torch.uint8),
                                 rope.reshape(ids.numel(), -1).view(torch.uint8)), dim=1)
            layer = args[7] // 100
            for row, destination in enumerate(ids.tolist()):
                current[row] = destination
                if destination > 0:
                    backing[layer, destination].copy_(encoded[row])
                    at = 16 + row if name == 'decode' else row if name == 'staged' else destination
                    resident[at].copy_(encoded[row])
        return write

    kernel = NS(write_decode=writer('decode'), write_prefill_staged=writer('staged'),
                write_prefill=writer('prefill'), invalidate=lambda *args: calls.append(('invalidate', args)))
    arena.kernel = lambda: kernel
    pool = NS(dtype=dtype, start_layer=7, flowkv_layout=NS(layers=2,
        native_segment_bytes=(row_bytes,), row_bytes=row_bytes,
        max_staging_tokens=4, capacity=16),
        flowkv_arena=arena, flowkv_plain_compact_prefill=compact,
        flowkv_staging_target_rows=torch.empty(4, dtype=torch.int32),
        _flowkv_mode=mode, flowkv_host=NS(descriptors=[object(), object()]) if host else None,
        flowkv_request=NS(page_map=object()) if request else None,
        flowkv_peers=NS(device_ptr=123), flowkv_rank=0, _flowkv_namespaces=1,
        _flowkv_owner_start=0, _flowkv_peer_offset=lambda layer: layer * 100,
        _flowkv_current_rows=object(), _flowkv_prefill_layer=19,
        _flowkv_storage_size=1, _flowkv_global_rows=32)
    return pool, backing, calls, kernel


def encoded(rows, *, widths=(528, 128), dtype=torch.uint8):
    return tuple(torch.arange(rows * width, dtype=torch.int32).remainder(251)
                 .to(dtype).reshape(rows, 1, width) for width in widths)


@pytest.mark.parametrize('mode,compact,expected', [
    ('decode', False, 'decode'), ('prefill', False, 'prefill'), ('prefill', True, 'staged'),
])
@pytest.mark.parametrize('tier', ['device', 'host', 'request'])
def test_batch_uses_original_writer_buffers_and_descriptor(mode, compact, expected, tier):
    pool, backing, calls, _ = fixture(mode=mode, compact=compact,
                                     host=tier == 'host', request=tier == 'request')
    ids = torch.tensor([2, 9, 5], dtype=torch.int64)
    segments = encoded(3)
    write_native_record_segments(pool, 1, ids, segments)
    name, args = calls[0]
    assert name == expected
    assert args[0] is segments[0] and args[1] is segments[1] and args[2] is ids
    assert args[3] == 123
    expected_buffer = (pool.flowkv_arena.staging if compact else pool.flowkv_arena.prefill
                       if mode == 'prefill' else pool.flowkv_arena.storage[1])
    assert args[4].data_ptr() == expected_buffer.data_ptr()
    assert args[5].data_ptr() == pool.flowkv_staging_target_rows.data_ptr()
    assert args[6:11] == (0, 100, 0, False, 16)
    assert args[11] is pool.flowkv_arena.groups[1].cursors
    assert args[12] is pool.flowkv_arena.groups[1].totals
    assert args[13] is pool.flowkv_arena.active
    expected_descriptor = (() if tier == 'device' else
        (pool.flowkv_host.descriptors[1],) if tier == 'host' else (pool.flowkv_request.page_map,))
    assert args[14:] == expected_descriptor
    assert pool._flowkv_current_rows is args[5] and pool._flowkv_prefill_layer is None
    expected_bytes = torch.cat(tuple(value.reshape(3, -1) for value in segments), dim=1)
    assert torch.equal(backing[1, ids], expected_bytes)
    assert [name for name, _ in calls] == ([expected, 'invalidate'] if compact else [expected])
    if compact:
        assert torch.equal(calls[1][1][0], ids.int())
        assert calls[1][1][-1] is pool.flowkv_arena.groups[1].hints


@pytest.mark.parametrize('dtype,row_bytes', [(torch.float8_e4m3fn, 576), (torch.bfloat16, 1152)])
def test_original_unscaled_formats_keep_their_typed_bytes(dtype, row_bytes):
    pool, backing, calls, _ = fixture(row_bytes=row_bytes, dtype=dtype)
    ids = torch.tensor([2, 5], dtype=torch.int32)
    # Byte views avoid performing floating point arithmetic in this transport test.
    segments = tuple(torch.arange(2 * width * dtype.itemsize, dtype=torch.int32)
        .remainder(251).to(torch.uint8).reshape(2, 1, width * dtype.itemsize).view(dtype)
        for width in (512, 64))
    write_native_record_segments(pool, 0, ids, segments)
    expected = torch.cat(tuple(value.view(torch.uint8).reshape(2, -1) for value in segments), dim=1)
    assert torch.equal(backing[0, ids], expected)
    assert calls[0][1][0] is segments[0]


def test_multiple_prefill_chunks_then_decode_keep_completed_backing():
    pool, backing, calls, _ = fixture(mode='prefill', compact=True)
    expected = backing.clone()
    for ids in (torch.tensor([3, 7, 10], dtype=torch.int32),
                torch.tensor([4, 8], dtype=torch.int32)):
        segments = encoded(ids.numel())
        write_native_record_segments(pool, 0, ids, segments)
        expected[0, ids] = torch.cat(tuple(value.reshape(ids.numel(), -1) for value in segments), dim=1)
    pool._flowkv_mode = 'decode'
    pool.flowkv_plain_compact_prefill = False
    ids = torch.tensor([15], dtype=torch.int32)
    segments = encoded(1)
    write_native_record_segments(pool, 0, ids, segments)
    expected[0, ids] = torch.cat(tuple(value.reshape(1, -1) for value in segments), dim=1)
    assert torch.equal(backing, expected)
    assert [name for name, _ in calls] == ['staged', 'invalidate', 'staged', 'invalidate', 'decode']
    assert torch.equal(pool._flowkv_current_rows, ids)


@pytest.mark.parametrize('same_allocation', [False, True])
def test_forward_invalidation_is_reused_only_for_the_bound_native_write_ids(same_allocation):
    pool, backing, calls, _ = fixture(mode='prefill', compact=True)
    ids = torch.tensor([3, 7], dtype=torch.int32)
    pool._ulacache_prefill_invalidated_rows = ids.view(-1) if same_allocation else ids.clone()
    for layer in range(2):
        write_native_record_segments(pool, layer, ids, encoded(2))
        assert torch.equal(backing[layer, ids], torch.cat([x.reshape(2, -1) for x in encoded(2)], dim=1))
    assert sum(name == 'invalidate' for name, _ in calls) == (0 if same_allocation else 2)


def test_failed_kernel_keeps_prior_publication_state():
    pool, _, calls, kernel = fixture(mode='prefill', compact=True)
    before = pool._flowkv_current_rows
    def fail(*args): raise RuntimeError('native writer failed')
    kernel.write_prefill_staged = fail
    with pytest.raises(RuntimeError, match='native writer failed'):
        write_native_record_segments(pool, 0, torch.tensor([2]), encoded(1))
    assert pool._flowkv_current_rows is before and pool._flowkv_prefill_layer == 19
    assert not calls


@pytest.mark.parametrize('case', ['mode', 'capacity', 'ids', 'segments', 'row_bytes', 'layout'])
def test_invalid_contract_is_rejected_before_writer(case):
    pool, _, calls, _ = fixture()
    ids, segments = torch.tensor([2, 3]), encoded(2)
    if case == 'mode': pool._flowkv_mode = 'none'
    elif case == 'capacity': ids, segments = torch.arange(5), encoded(5)
    elif case == 'ids': ids = ids.float()
    elif case == 'segments': segments = (segments[0],)
    elif case == 'row_bytes': pool.flowkv_layout.row_bytes = 576
    else: pool.flowkv_layout.native_segment_bytes = (528, 128)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        write_native_record_segments(pool, 0, ids, segments)
    assert not calls


def test_writer_module_import_does_not_load_device_or_framework():
    process = isolated('''
from ulacache.storage.writers import write_native_record_segments
assert callable(write_native_record_segments)
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
''', DEVICE_AND_FRAMEWORK)
    assert process.returncode == 0, process.stdout + process.stderr
