"""Complete native record copies through a rank-local residency directory.

Native request retirement owns the quiescence boundary. Logical row IDs never
index the physical GPU bank directly, including when CPU records are present.
"""

from __future__ import annotations


def copy_registered_records_to_cpu(cache, logical_rows, *, chunk_records):
    import torch

    if chunk_records < 1:
        raise ValueError("Copy chunk capacity must be positive")
    engine = cache._execution_engine
    engine._drain_transfers()
    rows = logical_rows.to(engine.directory.device, dtype=torch.int64).reshape(-1)
    locations = engine.directory.index_select(1, rows).cpu()
    if (
        (locations == -1).any()
        or (locations >= engine.gpu_slots).any()
        or (locations < -engine.host_slots - 1).any()
    ):
        raise ValueError("Record copy requires allocated logical identities")
    result = []
    for layer in range(engine.groups.layer_count):
        group = engine.groups.group(layer)
        chunks = []
        for start in range(0, rows.numel(), chunk_records):
            loc = locations[group, start : start + chunk_records]
            output = torch.empty((loc.numel(), engine.row_bytes), dtype=torch.uint8)
            for cold in (False, True):
                positions = ((loc <= -2) if cold else (loc >= 0)).nonzero().flatten()
                if not positions.numel():
                    continue
                selected = loc.index_select(0, positions)
                storage = engine.host.storage[layer] if cold else engine.gpu[layer]
                indices = (-selected - 2 if cold else selected).to(storage.device)
                payload = storage.index_select(0, indices).cpu()
                output.index_copy_(0, positions, payload)
            chunks.append(output)
        result.append(chunks)
    return result


def load_registered_record_copy(cache, saved, logical_rows, *, chunk_records):
    import torch

    if chunk_records < 1:
        raise ValueError("Copy chunk capacity must be positive")
    engine = cache._execution_engine
    engine._drain_transfers()
    rows = logical_rows.to(engine.directory.device, dtype=torch.int64).reshape(-1)
    locations = engine.directory.index_select(1, rows).cpu()
    if (
        (locations == -1).any()
        or (locations >= engine.gpu_slots).any()
        or (locations < -engine.host_slots - 1).any()
    ):
        raise ValueError("Record restore requires allocated logical identities")
    if len(saved) != engine.groups.layer_count:
        raise ValueError("Record copy layer count differs from the cache")
    # Validate the full source before mutating any destination layer.
    for chunks in saved:
        if len(chunks) != (rows.numel() + chunk_records - 1) // chunk_records:
            raise ValueError("Record copy chunk count differs from the native request")
        for index, data in enumerate(chunks):
            count = min(chunk_records, rows.numel() - index * chunk_records)
            if (
                data.device.type != "cpu"
                or data.dtype != torch.uint8
                or data.shape != (count, engine.row_bytes)
            ):
                raise ValueError(
                    "Record restore requires complete native CPU byte records"
                )
    for layer, chunks in enumerate(saved):
        group = engine.groups.group(layer)
        for index, data in enumerate(chunks):
            loc = locations[
                group, index * chunk_records : index * chunk_records + len(data)
            ]
            for cold in (False, True):
                positions = ((loc <= -2) if cold else (loc >= 0)).nonzero().flatten()
                if not positions.numel():
                    continue
                selected = loc.index_select(0, positions)
                storage = engine.host.storage[layer] if cold else engine.gpu[layer]
                indices = (-selected - 2 if cold else selected).to(storage.device)
                payload = data.index_select(0, positions).to(storage.device)
                storage.index_copy_(0, indices, payload)
