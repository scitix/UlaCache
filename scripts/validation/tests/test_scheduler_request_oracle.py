"""A sparse native UID namespace must preserve the per-request LRU counts."""

from importlib import import_module, util
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    util.find_spec("torch") is None or shutil.which("g++") is None,
    reason="The independent native policy oracle needs optional Torch and a C++ compiler",
)


def test_order_preserving_uid_compaction_matches_known_hot_history(
    tmp_path, monkeypatch
):
    import torch

    root = Path(__file__).resolve().parents[3]
    runtime = root / "scripts/7_5_scheduler/_runtime"
    monkeypatch.syspath_prepend(str(runtime))
    seeding = import_module("seeding")
    reference = import_module("reference")
    offset = 1000000
    history = torch.arange(offset + 1, offset + 8193, dtype=torch.int32)
    monkeypatch.setattr(seeding, "histories", lambda _: (history,))
    compiled = subprocess.Popen(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            str(runtime / "reference.cpp"),
            "-o",
            str(tmp_path / "cpu_reference.so"),
        ]
    )
    assert compiled.wait() == 0
    output = tmp_path / "case"
    output.mkdir()
    initial = dict(
        groups=[dict(state=dict(gpu_tags=history.clone())) for _ in range(21)]
    )
    selected = (
        torch.tensor([offset + 4, offset + 5], dtype=torch.int32)
        .expand(512, 21, 3, 2)
        .contiguous()
    )
    current = (torch.arange(512 * 3, dtype=torch.int64) + offset + 20000).reshape(
        512, 3
    )
    counts = reference.replay_per_request(
        initial, selected, current, output, capture=tmp_path, workers=2
    )
    assert counts.shape == (512, 21, 13)
    # Two historical IDs remain protected; three fresh writers enter each step.
    assert (counts[:, :, 4] == 2).all()
    assert (counts[:, :, 5] == 2).all()
    assert (counts[:, :, 6] == 3).all()
    assert (counts[:, :, 7] == 0).all()
    assert (counts[:, :, 8] == 5).all()
    assert (counts[:, :, 9] == 2).all()
    assert not counts[:, :, 10].any()
