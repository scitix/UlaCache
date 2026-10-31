"""Snapshot restore keeps replicated Index ownership separate from MLA owners."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_canonical_plane_reader import make_snapshot
from ulacache.storage.layout import KeyValueRecordPlane
from ulacache.tools.snapshot.cuda import (
    CudaRequestSnapshot,
    FastCudaRequestSnapshot,
    PoolSnapshotBindings,
)
from ulacache.tools.snapshot.prefix import NativeByteBanks


@pytest.mark.parametrize("public_api", [False, True])
@pytest.mark.parametrize("replicated", [False, True])
def test_index_restore_uses_its_own_ownership(tmp_path, public_api, replicated):
    reader, request, payload = make_snapshot(tmp_path / "bank", rows=33)
    owners, rank = 8, 1
    rows = torch.arange(64, 97, dtype=torch.int64)
    local = rows[rank::owners] // owners
    gpu = torch.zeros((2, 64, 656), dtype=torch.uint8)
    index = {0: torch.zeros((4, 64 * 132), dtype=torch.uint8)}
    plane = KeyValueRecordPlane("key_value", "mla.scaled656.fp8.v1", (528, 128))

    class PublicCache:
        cache_specification = SimpleNamespace(
            storage_layout=SimpleNamespace(record_planes=(plane,))
        )

        def write(self, ids, layer, source):
            buffer = torch.empty((7, 656), dtype=torch.uint8)
            with source.open() as stream:
                while (chunk := stream.readinto(buffer)) is not None:
                    selected = ids[chunk.start : chunk.start + len(chunk.data)]
                    mask = selected % owners == rank
                    gpu[layer, selected[mask] // owners] = chunk.data[mask]
                    chunk.release()

    bindings = PoolSnapshotBindings(
        replace(reader.layout, owners=owners),
        rank,
        gpu,
        index,
        (0, 1),
        quiesce=lambda: None,
        public_cache=PublicCache() if public_api else None,
        index_owner_count=1 if replicated else owners,
    )
    snapshot = (FastCudaRequestSnapshot if public_api else CudaRequestSnapshot)(
        bindings
    )
    snapshot._quiesce = lambda: None
    result = snapshot.restore_canonical_request(reader.path, request.key, rows)
    expected_index = torch.frombuffer(
        bytearray(payload["index_000"]), dtype=torch.uint8
    ).reshape(33, 132)
    banks = NativeByteBanks(gpu, index, 656)
    restored_rows = rows if replicated else local
    actual_index = torch.frombuffer(
        bytearray(banks.read("index_000", restored_rows)), dtype=torch.uint8
    ).reshape(-1, 132)
    assert torch.equal(
        actual_index, expected_index if replicated else expected_index[rank::owners]
    )
    for layer in range(2):
        expected_mla = torch.frombuffer(
            bytearray(payload[f"mla_{layer:03d}"]), dtype=torch.uint8
        ).reshape(33, 656)
        assert torch.equal(gpu[layer, local], expected_mla[rank::owners])
    assert result["restored_bytes"] == len(local) * 2 * 656 + len(restored_rows) * 132
