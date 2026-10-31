"""Partial reruns must keep GPU assignment and leave completed methods alone."""

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace


def test_linked_only_runs_just_requested_points_on_assigned_devices(
    tmp_path, monkeypatch
):
    replay = importlib.import_module("scripts.7_5_scheduler.run")
    validator = ModuleType("validate_capture")
    validator.validate = lambda root, manifest: manifest
    monkeypatch.setitem(sys.modules, "validate_capture", validator)
    monkeypatch.setattr(
        replay, "trace_manifest", lambda args: {"model": "GLM-5.3-NVFP4"}
    )
    seen = []

    class Process:
        def __init__(self, command, **kwargs):
            if command[0] == "g++":
                return
            script = replay.Path(command[1])
            assert script.name == "linked_probes.py"
            requests = int(command[command.index("--requests") + 1])
            seen.append((requests, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            (script.parent / "COMPLETE.json").write_text(
                json.dumps({"passed": True, "requests": requests, "probes": [{}]})
            )

        def wait(self):
            return 0

    monkeypatch.setattr(replay.subprocess, "Popen", Process)
    out = tmp_path / "output"
    replay.replay(
        SimpleNamespace(
            capture_root=tmp_path,
            manifest=None,
            gpus="2,5",
            batch_sizes="1,8",
            linked_only=True,
            out=out,
        )
    )
    assert sorted(seen) == [(1, "2"), (8, "5")]
    complete = json.loads((out / "COMPLETE.json").read_text())
    assert complete["passed"] and not complete["full_matrix"]
    assert complete["batches"] == [1, 8]
    assert not list(out.glob("request_r*"))
