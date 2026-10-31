"""Byte-exact file publication through the public TP API on CPU stand-ins.

These tests exercise real file IO, owner filtering and physical routing. They
do not claim CUDA stream, device-kernel or physical GPFS bandwidth acceptance.
"""

from dataclasses import replace
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ulacache import FileRecordSource, KeyValueRecordPlane
from test_tensor_parallel_client import adopted_cache


def file_cache(monkeypatch):
    cache, pool, spec, _, _ = adopted_cache(monkeypatch)
    pool.flowkv_rank = 0
    pool._flowkv_global_rows = 8 * (64 + 128)
    pool._flowkv_owner_start = 0
    pool.flowkv_host.host = SimpleNamespace(
        storage=torch.zeros((2, 128, 656), dtype=torch.uint8))
    pool.flowkv_host.directory = torch.full((1, 192), -1, dtype=torch.int64)
    pool.flowkv_arena.invalidate = Mock()
    return cache, pool, spec.storage_layout.record_planes[0]


def native_file(tmp_path, plane, rows):
    payload = torch.arange(rows * 656, dtype=torch.int64).remainder(256).to(
        torch.uint8).reshape(rows, 656)
    path = tmp_path / 'native-records.bin'
    path.write_bytes(bytes(payload.flatten().tolist()))
    return FileRecordSource(path, plane, rows,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()), payload


def test_file_write_exceeds_chunk_limit_and_keeps_owner_cpu_gpu_routes(monkeypatch, tmp_path):
    cache, pool, plane = file_cache(monkeypatch)
    ids = torch.tensor([8, 9, 16, 24, 32, 40, 48, 56, 64, 72, 80], dtype=torch.int32)
    locations = [3, -2, 0, -4, 5, -7, 6, -8, 7, -9]
    pool.flowkv_host.directory[0, 1:11] = torch.tensor(locations)
    source, payload = native_file(tmp_path, plane, len(ids))
    cache.write(ids, 0, source)
    positions = torch.where(ids.remainder(8) == 0)[0].tolist()
    for position, location in zip(positions, locations):
        storage, row = (pool.flowkv_shards, location) if location >= 0 else (
            pool.flowkv_host.host.storage, -location - 2)
        assert torch.equal(storage[0, row], payload[position])
    assert not torch.any(pool.flowkv_shards[1])
    assert not torch.any(pool.flowkv_host.host.storage[1])
    pool.flowkv_host.allocate.assert_not_called()
    pool.flowkv_host.release.assert_not_called()
    pool.flowkv_arena.invalidate.assert_called_once()
    assert pool._flowkv_prefill_layer is None


def test_snapshot_owner_file_pairs_with_the_same_global_allocated_ids(monkeypatch, tmp_path):
    cache, pool, plane = file_cache(monkeypatch)
    pool.flowkv_rank = 3
    ids = torch.tensor([11, 19, 27], dtype=torch.int64)
    pool.flowkv_host.directory[0, 1:4] = torch.tensor([5, -2, 7])
    source, payload = native_file(tmp_path, plane, 3)
    cache.write(ids, 1, source)
    assert torch.equal(pool.flowkv_shards[1, 5], payload[0])
    assert torch.equal(pool.flowkv_host.host.storage[1, 0], payload[1])
    assert torch.equal(pool.flowkv_shards[1, 7], payload[2])


def test_legacy_snapshot_encoding_uses_an_explicit_native_alias(monkeypatch, tmp_path):
    cache, pool, _ = file_cache(monkeypatch)
    ids = torch.tensor([8], dtype=torch.int32)
    pool.flowkv_host.directory[0, 1] = 3
    plane = KeyValueRecordPlane('legacy_mla_000', 'native-scaled-fp8-v32', (656,))
    source, payload = native_file(tmp_path, plane, 1)
    cache.write(ids, 0, source)
    assert torch.equal(pool.flowkv_shards[0, 3], payload[0])


@pytest.mark.parametrize('field,value', [
    ('record_count', 2),
    ('record_plane', KeyValueRecordPlane('kv', 'unknown.same.width', (656,))),
    ('record_plane', KeyValueRecordPlane('kv', 'mla.raw576.fp8.v1', (656,))),
])
def test_file_mismatch_is_rejected_before_any_backing_write(monkeypatch, tmp_path, field, value):
    cache, pool, plane = file_cache(monkeypatch)
    source, _ = native_file(tmp_path, plane, 1)
    changes = {field: value}
    if field == 'record_count':
        changes['file_bytes'] = 2 * 656
    with pytest.raises(ValueError, match='logical write ID|complete native cache record'):
        cache.write(torch.tensor([8], dtype=torch.int32), 0, replace(source, **changes))
    pool.flowkv_arena.invalidate.assert_not_called()
    assert not torch.any(pool.flowkv_shards)


@pytest.mark.parametrize('ids', [
    torch.tensor([8.0]), torch.tensor([0]), torch.tensor([1536]),
    torch.tensor([8, 8]),
])
def test_invalid_file_ids_do_not_change_backing(monkeypatch, tmp_path, ids):
    cache, pool, plane = file_cache(monkeypatch)
    source, _ = native_file(tmp_path, plane, len(ids))
    with pytest.raises(ValueError, match='native device integers|unique allocated'):
        cache.write(ids, 0, source)
    pool.flowkv_arena.invalidate.assert_not_called()
    assert not torch.any(pool.flowkv_shards)


def test_file_write_requires_idle_reader_lifetime(monkeypatch, tmp_path):
    cache, pool, plane = file_cache(monkeypatch)
    source, _ = native_file(tmp_path, plane, 1)
    cache.begin_model_forward()
    with pytest.raises(RuntimeError, match='Finish every TP reader'):
        cache.write(torch.tensor([8]), 0, source)
    pool.flowkv_arena.invalidate.assert_not_called()


def test_truncated_file_fails_before_entering_the_storage_boundary(monkeypatch, tmp_path):
    cache, pool, plane = file_cache(monkeypatch)
    source, _ = native_file(tmp_path, plane, 1)
    source.path.write_bytes(b'\x01')
    with pytest.raises(ValueError, match='truncated or extended'):
        cache.write(torch.tensor([8]), 0, source)
    pool.flowkv_arena.invalidate.assert_not_called()
    assert not torch.any(pool.flowkv_shards)


def test_file_source_rejects_packed_source_indices(monkeypatch, tmp_path):
    cache, pool, plane = file_cache(monkeypatch)
    source, _ = native_file(tmp_path, plane, 1)
    with pytest.raises(ValueError, match='complete source range'):
        cache.write(torch.tensor([8]), 0, source, source_indices=torch.tensor([0]))
    pool.flowkv_arena.invalidate.assert_not_called()


def test_file_partial_checksum_failure_invalidates_without_freeing_ids(monkeypatch, tmp_path):
    cache, pool, plane = file_cache(monkeypatch)
    ids = torch.arange(8, 8 * 8, 8, dtype=torch.int32)
    pool.flowkv_host.directory[0, 1:8] = torch.arange(1, 8)
    source, payload = native_file(tmp_path, plane, len(ids))
    source = replace(source, sha256='0' * 64)
    with pytest.raises(ValueError, match='checksum mismatch'):
        cache.write(ids, 0, source)
    # Earlier bounded chunks can have completed. No transactional claim is
    # made; allocation stays owned and all old resident views are invalidated.
    assert torch.equal(pool.flowkv_shards[0, 1:5], payload[:4])
    assert not torch.any(pool.flowkv_shards[0, 5:8])
    pool.flowkv_arena.invalidate.assert_called_once()
    pool.flowkv_host.release.assert_not_called()


def test_unallocated_route_is_rejected_and_invalidated(monkeypatch, tmp_path):
    cache, pool, plane = file_cache(monkeypatch)
    source, _ = native_file(tmp_path, plane, 1)
    with pytest.raises(ValueError, match='unallocated'):
        cache.write(torch.tensor([8]), 0, source)
    assert not torch.any(pool.flowkv_shards)
    pool.flowkv_arena.invalidate.assert_called_once()
