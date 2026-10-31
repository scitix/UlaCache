"""CPU residency oracles for active-history preloads; never GPU timings."""

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json


def replay(initial, selected, current, output, workers=4):
    import numpy as np
    import torch

    ptr = ct.c_void_p
    lib = ct.CDLL(str(Path(output).parent / "cpu_reference.so"))
    lib.make_replay.argtypes = [ct.c_int, ct.c_int, ptr, ptr]
    lib.make_replay.restype = ptr
    lib.replay_step.argtypes = [ptr, ptr, ptr, ct.c_int, ct.c_int, ptr, ct.c_int, ptr]
    lib.delete_replay.argtypes = [ptr]
    counts = np.empty((2, 512, 21, 13), dtype=np.int64)
    offsets = np.zeros(1, dtype=np.int32)

    def group_replay(group):
        state = initial["groups"][group]["state"]
        seed = state["gpu_tags"].numpy().copy()
        ages = torch.where(
            state["gpu_tags"] >= 0,
            state["access_time"][state["gpu_tags"].clamp_min(0).long()],
            -1,
        )
        order = np.argsort(ages.numpy(), kind="stable").astype(np.int32)
        context = lib.make_replay(
            state["directory"].numel(), len(seed), seed.ctypes.data, order.ctypes.data
        )
        try:
            for frame in range(512):
                raw = selected[frame, group]
                writers = current[frame]
                assert raw.dtype == torch.int32 and raw.is_contiguous()
                assert writers.dtype == torch.int64 and writers.is_contiguous()
                pointers = np.array([raw.data_ptr()], dtype=np.uintp)
                result = np.empty((2, 13), dtype=np.int64)
                lib.replay_step(
                    context,
                    pointers.ctypes.data,
                    offsets.ctypes.data,
                    1,
                    raw.numel(),
                    writers.data_ptr(),
                    writers.numel(),
                    result.ctypes.data,
                )
                counts[:, frame, group] = result
        finally:
            lib.delete_replay(context)
        return group

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, group in enumerate(pool.map(group_replay, range(21)), 1):
            (Path(output) / "CPU_PROGRESS.json").write_text(
                json.dumps(dict(groups_complete=number, last_group=group)) + "\n"
            )
    assert not counts[:, :, :, 10].any()
    np.savez_compressed(
        Path(output) / "CPU_COUNTS.npz",
        native_linked_list=counts[0],
        hisparse=counts[1],
    )
    rows = []
    for name, array in zip(("native_linked_list", "hisparse"), counts):
        total = array[256:].sum(axis=(0, 1), dtype=np.int64)
        rows.append(
            dict(
                method=name,
                total_counts=total.tolist(),
                unique_historical_hit_rate=float(total[5] / total[4]),
            )
        )
    (Path(output) / "CPU_HIT_RATES.json").write_text(
        json.dumps(
            dict(
                passed=True,
                complete=True,
                rows=rows,
                frames=512,
                warmup=256,
                seed="Active-history preload, real empty slots, no inactive GPU tags",
                scope="CPU policy replay; not GPU latency. Each policy evolves independently.",
            ),
            indent=2,
        )
        + "\n"
    )
    return counts


def replay_per_request(initial, selected, current, output, *, capture, workers=8):
    """Independent per-request native-LRU oracle with order-preserving UID compaction."""
    import numpy as np
    import torch

    ptr = ct.c_void_p
    lib = ct.CDLL(str(Path(output).parent / "cpu_reference.so"))
    lib.make_replay.argtypes = [ct.c_int, ct.c_int, ptr, ptr]
    lib.make_replay.restype = ptr
    lib.replay_step.argtypes = [ptr, ptr, ptr, ct.c_int, ct.c_int, ptr, ct.c_int, ptr]
    lib.delete_replay.argtypes = [ptr]
    frames, groups, queries, topk = selected.shape
    requests = queries // 3
    if frames != 512 or groups != 21 or queries != 3 * requests:
        raise ValueError("Require the complete MTP2 trajectory")
    counts = np.zeros((frames, groups, 13), dtype=np.int64)
    offsets = np.zeros(1, dtype=np.int32)
    from seeding import histories

    history = histories(str(capture))[:requests]
    mappings = []
    for request, ids in enumerate(history):
        writers = current[:, 3 * request : 3 * request + 3].flatten()
        namespace = torch.unique(torch.cat((ids.long(), writers)), sorted=True)
        lower = int(namespace[0])
        table = torch.full((int(namespace[-1]) - lower + 1,), -1, dtype=torch.int32)
        table[namespace - lower] = torch.arange(
            1, len(namespace) + 1, dtype=torch.int32
        )
        mappings.append((lower, table, len(namespace) + 1))

    def request_group(pair):
        group, request = pair
        raw = selected[:, group, 3 * request : 3 * request + 3].contiguous()
        writers = current[:, 3 * request : 3 * request + 3].contiguous()
        seed = initial["groups"][group]["state"]["gpu_tags"][
            request * 8192 : (request + 1) * 8192
        ].contiguous()
        lower, table, logical = mappings[request]
        compact_raw = table[raw.long() - lower].contiguous().numpy()
        compact_writers = table[writers - lower].long().contiguous().numpy()
        compact_seed = table[seed.long() - lower].contiguous().numpy()
        if min(compact_raw.min(), compact_writers.min(), compact_seed.min()) <= 0:
            raise ValueError("A selected UID is absent from its native request history")
        order = np.arange(8192, dtype=np.int32)
        handle = lib.make_replay(
            logical, 8192, compact_seed.ctypes.data, order.ctypes.data
        )
        result = np.empty((frames, 13), dtype=np.int64)
        try:
            for frame in range(frames):
                pointers = np.array([compact_raw[frame].ctypes.data], dtype=np.uintp)
                both = np.empty((2, 13), dtype=np.int64)
                lib.replay_step(
                    handle,
                    pointers.ctypes.data,
                    offsets.ctypes.data,
                    1,
                    3 * topk,
                    compact_writers[frame].ctypes.data,
                    3,
                    both.ctypes.data,
                )
                result[frame] = both[1]
        finally:
            lib.delete_replay(handle)
        return group, result

    pairs = [(group, request) for group in range(groups) for request in range(requests)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, (group, values) in enumerate(pool.map(request_group, pairs), 1):
            counts[:, group] += values
            if completed % requests == 0:
                (Path(output) / "REQUEST_ORACLE_PROGRESS.json").write_text(
                    json.dumps(
                        dict(
                            request_group_pairs_complete=completed,
                            total_pairs=len(pairs),
                        )
                    )
                    + "\n"
                )
    if counts[:, :, 10].any():
        raise RuntimeError("Invalid per-request oracle bindings")
    np.savez_compressed(
        Path(output) / "PER_REQUEST_CPU_COUNTS.npz", hisparse_request=counts
    )
    return counts
