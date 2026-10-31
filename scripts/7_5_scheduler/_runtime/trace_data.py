"""Compose independent native DP-rank traces into one larger shared-pool load.

This is a trace-driven construction, not a native high-batch model capture.
Each source UID receives a fixed rank offset, preserving within-rank identity.
Source ages are preserved relative to a common epoch before capacity resizing.
"""

from pathlib import Path
import hashlib
import json


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(capture, mtp, requests, gpu_slots, output):
    import torch
    from ulacache.cache.kernels.cpu_residency import create_host_tier_state

    assert requests in (16, 32, 64, 128)
    factor = 1 if mtp == 0 else 3
    rank_count = requests // 16
    sources = []
    initials = []
    events = []
    source_slots = None
    original = torch.empty((512, 21, requests * factor, 2048), dtype=torch.int32)
    current = torch.empty((512, requests * factor), dtype=torch.int64)
    for rank in range(rank_count):
        directory = Path(capture) / f"mtp{mtp}/traces/cohort0001/rank{rank}"
        start = json.loads((directory / "START.json").read_text())
        done = json.loads((directory / "COMPLETE.json").read_text())
        audit = json.loads((directory / "NATIVE_SOURCE_BINDING_AUDIT.json").read_text())
        assert done["complete"] and done["target_forwards"] == 512
        assert audit["passed"] and audit["complete_sha256"] == digest(
            directory / "COMPLETE.json"
        )
        assert audit["start_sha256"] == digest(directory / "START.json")
        assert json.loads((directory / "COVERAGE_AUDIT.json").read_text())["passed"]
        assert (
            digest(start["initial_state"]["path"]) == start["initial_state"]["sha256"]
        )
        initial = torch.load(
            start["initial_state"]["path"],
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        logical = initial["groups"][0]["state"]["directory"].numel()
        source_slots = logical if source_slots is None else source_slots
        assert logical == source_slots
        offset = rank * logical
        initials.append(initial)
        frame_cursor = 0
        for ref in done["chunks"]:
            assert digest(ref["path"]) == ref["sha256"]
            data = torch.load(
                ref["path"], map_location="cpu", weights_only=True, mmap=True
            )
            assert all(m["queries_per_request"] == factor for m in data["metadata"])
            assert torch.equal(
                data["request_slots"],
                initial["native"]["request_slots"][None, :].expand_as(
                    data["request_slots"]
                ),
            )
            assert bool((data["current"] == data["current"][:, :1, :]).all())
            assert bool((data["selected"] > 0).all()) and bool(
                (data["current"] > 0).all()
            )
            n = data["selected"].shape[0]
            original[
                frame_cursor : frame_cursor + n,
                :,
                rank * 16 * factor : (rank + 1) * 16 * factor,
            ].copy_(data["selected"] + offset)
            current[
                frame_cursor : frame_cursor + n,
                rank * 16 * factor : (rank + 1) * 16 * factor,
            ].copy_(data["current"][:, 0].long() + offset)
            frame_cursor += n
        assert frame_cursor == 512
        assert digest(done["events"]["path"]) == done["events"]["sha256"]
        source_events = torch.load(
            done["events"]["path"], map_location="cpu", weights_only=True
        ).tolist()
        assert all(kind == 1 for _, kind, _ in source_events)
        events.extend((frame, kind, uid + offset) for frame, kind, uid in source_events)
        sources.append(
            dict(
                rank=rank,
                path=str(directory),
                uid_offset=offset,
                start_sha256=digest(directory / "START.json"),
                complete_sha256=digest(directory / "COMPLETE.json"),
                binding_audit_sha256=digest(
                    directory / "NATIVE_SOURCE_BINDING_AUDIT.json"
                ),
                initial_sha256=start["initial_state"]["sha256"],
                events_sha256=done["events"]["sha256"],
                chunks=done["chunks"],
            )
        )

    total_slots = 8 * source_slots
    assert gpu_slots < total_slots
    groups = []
    seed_receipts = []
    for group in range(21):
        inputs = [initial["groups"][group]["state"] for initial in initials]
        assert all(not bool(s["leases"].any()) for s in inputs)
        epoch = max(int(s["epoch"]) for s in inputs)
        hot = torch.cat(
            [
                s["gpu_tags"][s["gpu_tags"] > 0] + rank * source_slots
                for rank, s in enumerate(inputs)
            ]
        )
        cold = torch.cat(
            [
                s["host_tags"][s["host_tags"] > 0] + rank * source_slots
                for rank, s in enumerate(inputs)
            ]
        )
        live = torch.cat([hot, cold]).long()
        assert live.numel() >= gpu_slots and live.unique().numel() == live.numel()
        gpu = live[:gpu_slots]
        cpu = live[gpu_slots:]
        state = create_host_tier_state(
            gpu_slots, total_slots - gpu_slots, "cpu"
        )._asdict()
        state["epoch"].fill_(epoch)
        for rank, source in enumerate(inputs):
            times = source["access_time"]
            adjusted = torch.where(
                times >= 0, times + epoch - int(source["epoch"]), times
            )
            state["access_time"][rank * source_slots : (rank + 1) * source_slots] = (
                adjusted
            )
        state["gpu_tags"].copy_(gpu.int())
        state["host_tags"][: cpu.numel()] = cpu.int()
        state["directory"][gpu] = torch.arange(gpu_slots)
        state["directory"][cpu] = -torch.arange(cpu.numel()) - 2
        free = state["host_tags"].numel() - cpu.numel()
        state["host_free"][:free] = torch.arange(
            cpu.numel(), state["host_tags"].numel(), dtype=torch.int32
        )
        state["free_counts"].copy_(torch.tensor([0, free], dtype=torch.int32))
        assert int(state["access_time"][live].min()) >= 0
        assert int(state["access_time"][live].max()) <= epoch
        groups.append(dict(state=state))
        seed_receipts.append(
            dict(
                group=group,
                common_epoch=epoch,
                source_epochs=[int(s["epoch"]) for s in inputs],
                live_ids=live.numel(),
                original_hot_across_ranks=hot.numel(),
                gpu_slots=gpu_slots,
                cpu_slots=total_slots - gpu_slots,
            )
        )
    receipt = dict(
        trace_driven_composition=True,
        native_high_batch_capture=False,
        mtp_steps=mtp,
        requests=requests,
        queries=requests * factor,
        source_ranks=rank_count,
        source_slots_per_rank=source_slots,
        logical_slots=total_slots,
        namespace_ranks=8,
        fixed_total_capacity_across_request_counts=True,
        sources=sources,
        groups=seed_receipts,
        transformation="Concatenate complete16-request native rank trajectories at equal relative frame indices. Add rank*source_logical_slots to every selected/current/event/live UID. Normalize timestamps to a common epoch without changing ages. Concatenate registered hot IDs across ranks, then resize one shared pool before the common active-history preload. The total logical capacity is fixed at8source namespaces for every request count. All methods receive exactly this input and UID namespace.",
        semantics="One GPU hot pool and one exclusive CPU cold pool per Index group, shared across all32/64/128requests. This constructs a high-load replay from native DP8trajectories; it is not a model-throughput experiment.",
    )
    receipt["recipe_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()
    ).hexdigest()
    (Path(output) / "COMPOSED_SOURCE.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return dict(
        initial=dict(groups=groups),
        original=original,
        currents=current,
        events=sorted(events),
        G=gpu_slots,
        H=total_slots - gpu_slots,
        W=max(requests * factor * 2049, 1024),
        bound=requests * factor,
        receipt=receipt,
    )
