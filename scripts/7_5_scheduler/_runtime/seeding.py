"""Active-history preload with real empty slots and no inactive GPU residency."""

from functools import lru_cache
from pathlib import Path
import json


@lru_cache(maxsize=1)
def histories(capture):
    import torch

    result = []
    for rank in range(8):
        folder = Path(capture) / f"mtp2/traces/cohort0001/rank{rank}"
        start = json.loads((folder / "START.json").read_text())
        source = torch.load(
            start["initial_state"]["path"],
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        logical = source["groups"][0]["state"]["directory"].numel()
        table, sizes = source["native"]["req_to_token"], source["native"]["seq_lens"]
        for request in range(16):
            ids = table[request, : int(sizes[request])].int().clone() + rank * logical
            assert bool((ids > 0).all()) and ids.unique().numel() == ids.numel()
            result.append(ids)
    return tuple(result)


def repack(initial, capture, gpu_slots, requests, *, per_request=False):
    import torch

    history = histories(str(capture))[:requests]
    limit = 8192 if per_request else gpu_slots // requests
    selected = [ids[-min(ids.numel(), limit) :] for ids in history]
    hot = torch.cat(selected).long()
    assert hot.numel() <= gpu_slots and hot.unique().numel() == hot.numel()
    receipts = []
    for group, record in enumerate(initial["groups"]):
        state = record["state"]
        live = torch.cat(
            [
                state["gpu_tags"][state["gpu_tags"] >= 0],
                state["host_tags"][state["host_tags"] >= 0],
            ]
        ).long()
        assert bool((state["directory"][hot] != -1).all())
        chosen = torch.zeros(state["directory"].numel(), dtype=torch.bool)
        chosen[hot] = True
        cold = live[~chosen[live]]
        assert cold.numel() <= state["host_tags"].numel()
        state["directory"].fill_(-1)
        state["gpu_tags"].fill_(-1)
        state["gpu_tags"][: hot.numel()].copy_(hot.int())
        state["host_tags"].fill_(-1)
        state["host_tags"][: cold.numel()].copy_(cold.int())
        state["directory"][hot] = torch.arange(hot.numel())
        state["directory"][cold] = -torch.arange(cold.numel()) - 2
        free_gpu = gpu_slots - hot.numel()
        free_cpu = state["host_tags"].numel() - cold.numel()
        state["gpu_free"][:free_gpu] = torch.arange(
            hot.numel(), gpu_slots, dtype=torch.int32
        ).flip(0)
        state["host_free"][:free_cpu] = torch.arange(
            cold.numel(), state["host_tags"].numel(), dtype=torch.int32
        )
        state["free_counts"].copy_(
            torch.tensor([free_gpu, free_cpu], dtype=torch.int32)
        )
        # Prefill ends at the current epoch. Preserve token order within each
        # request through deterministic physical-slot ties; do not invent trace ages.
        state["access_time"][hot] = state["epoch"]
        assert torch.equal(state["directory"][hot], torch.arange(hot.numel()))
        receipts.append(
            dict(
                group=group,
                active_requests=requests,
                initial_history_tokens=sum(ids.numel() for ids in history),
                initial_hot_tokens=hot.numel(),
                free_gpu_slots=free_gpu,
                live_ids=live.numel(),
                per_request=per_request,
                selected_per_request=[ids.numel() for ids in selected],
                fully_preloaded=all(
                    len(a) == len(b) for a, b in zip(selected, history)
                ),
                rule="Most recent min(history length, capacity / BS) UIDs per active request; inactive requests remain CPU-cold.",
            )
        )
    return receipts


def rebuild_ring(state, ring):
    import numpy as np
    import torch

    tags = state.gpu_tags.cpu().numpy()
    times = state.access_time.cpu().numpy()
    epoch = int(state.epoch)
    R = ring.recent_count
    G = len(tags)
    slots = np.flatnonzero(tags >= 0)
    age = times[tags[slots]]
    assert np.all((age >= 0) & (age <= epoch))
    bid = np.where(epoch - age >= R, R, age % R)
    bits = np.zeros(tuple(ring.bits.shape), dtype=np.uint32)
    np.bitwise_or.at(
        bits,
        (bid, slots // 32),
        np.left_shift(np.uint32(1), (slots % 32).astype(np.uint32)),
    )
    summary = np.zeros(tuple(ring.summary.shape), dtype=np.uint32)
    previous = bits
    for offset, length in zip(ring.offsets, ring.lengths):
        b, j = np.nonzero(previous)
        level = np.zeros((R + 1, length), dtype=np.uint32)
        np.bitwise_or.at(
            level, (b, j // 32), np.left_shift(np.uint32(1), (j % 32).astype(np.uint32))
        )
        summary[:, offset : offset + length] = level
        previous = level
    summary[:, -1] = np.bincount(bid, minlength=R + 1).astype(np.uint32)
    labels = np.full(R, -1, dtype=np.int64)
    recent = bid < R
    labels[bid[recent]] = age[recent]
    inverse = np.full(G, -1, dtype=np.int32)
    inverse[slots] = np.where(bid == R, -2, bid)
    for name, data in [
        ("bits", bits),
        ("summary", summary),
        ("bucket_epochs", labels),
        ("gpu_bucket", inverse),
    ]:
        getattr(ring, name).copy_(torch.from_numpy(data))
