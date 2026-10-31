"""Validate fresh native Graph bindings before any scheduler-policy replay."""

from pathlib import Path
import hashlib
import json


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(".writing")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate(root, manifest):
    import torch
    from ulacache.telemetry.trace_coverage import validate_bucket_rotation

    root = Path(root)
    reports = []
    for rank in range(8):
        directory = root / f"mtp2/traces/cohort0001/rank{rank}"
        start_path, done_path = directory / "START.json", directory / "COMPLETE.json"
        start, done = (
            json.loads(start_path.read_bytes()),
            json.loads(done_path.read_bytes()),
        )
        if (
            not done["complete"]
            or done["target_forwards"] != 512
            or not start["native_frame_binding_required"]
        ):
            raise ValueError("Require a complete native-Graph-bound trace")
        if digest(start["initial_state"]["path"]) != start["initial_state"]["sha256"]:
            raise ValueError("Native initial state changed")
        initial = torch.load(
            start["initial_state"]["path"],
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        slots = initial["native"]["request_slots"]
        expiry = []
        cursor = 0
        bindings = []
        epochs = []
        binding_storage = []
        for reference in done["chunks"]:
            if digest(reference["path"]) != reference["sha256"]:
                raise ValueError("Native trace chunk changed")
            data = torch.load(
                reference["path"], map_location="cpu", weights_only=True, mmap=True
            )
            count = data["selected"].shape[0]
            epochs.append(data["epochs"])
            if (
                data["first_frame"] != cursor
                or data["selected"].shape[1:] != (21, 48, 2048)
                or not bool((data["selected"] > 0).all())
                or not bool((data["current"] > 0).all())
                or not bool((data["seen"] == 48).all())
                or not torch.equal(
                    data["request_slots"],
                    slots[None, :].expand_as(data["request_slots"]),
                )
                or not bool((data["current"] == data["current"][:, :1, :]).all())
            ):
                raise ValueError(
                    "Trace shape, native request identity or group coverage differs"
                )
            metadata = data["metadata"]
            if [item["frame"] for item in metadata] != list(
                range(cursor, cursor + count)
            ):
                raise ValueError("Native sampled frame ordinals skipped or repeated")
            chunk_bindings = [item["native_replay_binding"] for item in metadata]
            shared_binding = len(chunk_bindings) > 1 and all(
                value is chunk_bindings[0] for value in chunk_bindings
            )
            binding_storage.append(
                "shared_chunk_snapshot" if shared_binding else "per_frame"
            )
            for item in metadata:
                binding = item["native_replay_binding"]
                if (
                    item["requests"] != 16
                    or item["queries"] != 48
                    or item["queries_per_request"] != 3
                    or not binding["native_replay_tensor_binding_verified"]
                    or binding["raw_requests"] != 16
                    or binding["raw_queries"] != 48
                    or binding["verify_width"] != 3
                    or len(binding["captured"]["layers"]) != 21
                ):
                    raise ValueError(
                        "A sampled frame lacks the declared actual native replay"
                    )
                if not shared_binding:
                    bindings.append(binding["native_replay_sequence"])
            if shared_binding:
                bindings.append(chunk_bindings[-1]["native_replay_sequence"])
            expiry.append(data["expiry"])
            cursor += count
        # load_batch also runs in EAGLE preplanning, so its sequence is not a
        # target-forward clock. GPU residency epochs are recorded every frame
        # and independently prove that no cache transaction was skipped.
        clock = torch.cat(epochs)
        if (
            cursor != 512
            or clock.shape != (512, 21)
            or not bool((clock[1:] == clock[:-1] + 1).all())
            or any(b <= a for a, b in zip(bindings, bindings[1:]))
        ):
            raise ValueError("Native target replay skipped or repeated a frame")
        observed = torch.cat(expiry)
        groups = [
            validate_bucket_rotation(observed[:, g].tolist(), start["recent_counts"][g])
            for g in range(21)
        ]
        events = done["events"]
        if digest(events["path"]) != events["sha256"]:
            raise ValueError("Native lifetime events changed")
        values = torch.load(events["path"], map_location="cpu", weights_only=True)
        if values.numel() and not bool((values[:, 1] == 1).all()):
            raise ValueError(
                "The replay requires explicit support for these native release events"
            )
        write(
            directory / "COVERAGE_AUDIT.json",
            dict(passed=True, groups=groups, frames=512, queries=48),
        )
        write(
            directory / "NATIVE_SOURCE_BINDING_AUDIT.json",
            dict(
                passed=True,
                start_sha256=digest(start_path),
                complete_sha256=digest(done_path),
                model=manifest["model"],
                checkpoint_manifest_sha256=manifest.get("checkpoint_manifest_sha256")
                or manifest.get("checkpoint_config_sha256"),
                actual_native_graph_bound=True,
                native_replay_frames=len(bindings),
                target_frames=cursor,
                native_frame_clock="GPU residency epochs across all21 groups",
                binding_storage=binding_storage,
                load_sequence_is_not_target_forward_count=True,
            ),
        )
        reports.append(
            dict(
                rank=rank,
                frames=cursor,
                groups=21,
                first_native_replay=bindings[0],
                last_native_replay=bindings[-1],
            )
        )
    result = dict(
        manifest, trace_validation=dict(passed=True, ranks=reports), passed=True
    )
    write(root / "VALIDATED_CAPTURE.json", result)
    return result
