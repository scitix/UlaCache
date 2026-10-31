"""Exercise the direct CLI and request protocol without a model or cluster."""

import ast
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import subprocess
from threading import Thread
from types import SimpleNamespace

import pytest

from scripts import decode, fill_kv

ROOT = Path(__file__).resolve().parents[3]


def test_decode_uses_actual_per_rank_capacity_when_argument_is_omitted():
    report = {
        "loads": [
            {"dp_rank": 1, "max_total_num_tokens": 3000},
            {"dp_rank": 0, "max_total_num_tokens": 4000},
        ]
    }
    assert decode.resolve_rank_capacities(report, 2) == [4000, 3000]
    assert decode.resolve_rank_capacities(report, 2, 2000) == [2000, 2000]
    with pytest.raises(ValueError, match="exceeds"):
        decode.resolve_rank_capacities(report, 2, 3500)
    with pytest.raises(ValueError, match="missing"):
        decode.resolve_rank_capacities({"loads": report["loads"][:1]}, 2)


def test_large_uneven_cohort_routes_within_real_kv_and_request_limits():
    lengths = [70000, 78000, 86000, 95000, 100000, 110000, 120000, 129000]
    requests = [{"key": str(i), "input_ids": range(lengths[i % 8])} for i in range(236)]
    capacity = 3135168
    unbounded = decode.route_requests(requests, [2**60] * 8, 32, 2048)
    assert max(unbounded["reserved_tokens"]) > capacity
    actual = decode.route_requests(requests, [capacity] * 8, 32, 2048)
    assert [r["key"] for r in actual["route"]] == [r["key"] for r in requests]
    assert max(actual["reserved_tokens"]) <= capacity
    assert all(
        sum(r["attn_dp_rank"] == rank for r in actual["route"]) <= 32
        for rank in range(8)
    )
    with pytest.raises(ValueError, match="actual KV"):
        decode.route_requests(requests, [capacity] * 8, 1, 2048)


def test_hisparse_admission_uses_logical_capacity_not_gpu_copies():
    report = {"loads": [
        {"dp_rank": 0, "max_total_num_tokens": 681024,
         "logical_token_capacity": 2500992},
        {"dp_rank": 1, "max_total_num_tokens": 681024,
         "logical_token_capacity": 2500992},
    ]}
    assert decode.resolve_rank_capacities(report, 2) == [2500992, 2500992]
    assert decode.resolve_rank_capacities(report, 2, 2000000) == [2000000, 2000000]
    with pytest.raises(ValueError, match="exceeds"):
        decode.resolve_rank_capacities(report, 2, 2500993)
    requests = [{"key": str(i), "input_ids": range(100000)} for i in range(40)]
    result = decode.route_requests(requests, [2500992, 2500992], 32, 2048)
    assert len(result["route"]) == 40


def test_saved_capacity_balanced_routing_checks_real_slots_and_request_order():
    requests = [{"key": str(i), "input_ids": range(n)}
                for i, n in enumerate([3000, 3000, 1000, 1000])]
    assignment = [{"key": str(i), "attn_dp_rank": rank}
                  for i, rank in enumerate([0, 1, 0, 1])]
    result = decode.route_requests(requests, [6000, 6000], 2, 256, assignment)
    assert [row["attn_dp_rank"] for row in result["route"]] == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="actual KV/request capacity"):
        decode.route_requests(requests, [4000, 4000], 2, 256, assignment)
    with pytest.raises(ValueError, match="request order"):
        decode.route_requests(requests[::-1], [6000, 6000], 2, 256, assignment)
    with pytest.raises(ValueError, match="exact request cohort"):
        decode.route_requests(requests, [6000, 6000], 2, 256, assignment[:-1])


def test_saved_bank_keeps_its_declared_checkpoint_identity(tmp_path, monkeypatch):
    native = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if not native:
        pytest.skip("Set HISPARSE_NATIVE_SOURCE_ROOT to the patched official checkout")
    path = Path(native) / "python/sglang/srt/ulacache/entry.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "benchmark_model_fingerprint"
    )
    namespace = {"hashlib": hashlib, "os": os, "re": __import__("re")}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    resolve = namespace[function.name]
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.delenv("ULACACHE_BENCH_MODEL_FINGERPRINT", raising=False)
    assert resolve(tmp_path) == hashlib.sha256(b"{}").hexdigest()
    monkeypatch.setenv("ULACACHE_BENCH_MODEL_FINGERPRINT", "a" * 64)
    assert resolve(tmp_path) == "a" * 64
    pinned = "vendor/model@" + "b" * 40 + ":fp8:kv-fp8_e4m3"
    monkeypatch.setenv("ULACACHE_BENCH_MODEL_FINGERPRINT", pinned)
    assert resolve(tmp_path) == pinned
    monkeypatch.setenv("ULACACHE_BENCH_MODEL_FINGERPRINT", "invalid")
    with pytest.raises(ValueError, match="lowercase SHA256"):
        resolve(tmp_path)


def test_existing_bank_identity_is_checked_before_model_start(tmp_path):
    native = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if not native:
        pytest.skip("Set HISPARSE_NATIVE_SOURCE_ROOT to the patched official checkout")
    path = Path(native) / "python/sglang/srt/ulacache/entry.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "validate_snapshot_bank_identity")
    namespace = {"Path": Path, "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
         namespace)
    check = namespace[function.name]
    check(tmp_path, "a" * 64)  # A separate producer may populate an empty bank.
    record = tmp_path / "prefix" / "SNAPSHOT.json"
    record.parent.mkdir()
    record.write_text(json.dumps({"complete": True, "model_fingerprint": "a" * 64}))
    check(tmp_path, "a" * 64)
    with pytest.raises(ValueError, match="ULACACHE_BENCH_MODEL_FINGERPRINT"):
        check(tmp_path, "b" * 64)


@pytest.mark.parametrize("mode", ["tp", "dpa"])
@pytest.mark.parametrize("spec", ["on", "off"])
def test_launchers_contain_their_resolved_native_configuration(tmp_path, mode, spec):
    for script in sorted((ROOT / "scripts").glob("7_*/start_*.sh")):
        if mode == "dpa" and script == ROOT / "scripts/7_6_indexer/start_ulacache.sh":
            continue  # This ablation compares TP Index ownership only.
        logs = tmp_path / script.parent.name / script.stem
        result = subprocess.run(
            [
                "bash",
                str(script),
                "--model",
                str(tmp_path / "model"),
                "--mode",
                mode,
                "--spec",
                spec,
                "--log-dir",
                str(logs),
                "--kv-dir",
                str(tmp_path / "kv"),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        command = shlex.split(result.stdout.splitlines()[-1])
        assert command[:3] == ["python", "-m", "sglang.srt.ulacache.entry"]
        assert command[command.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
        assert ("--speculative-algorithm" in command) == (spec == "on")
        assert command[command.index("--attn-dp-size") + 1] == (
            "8" if mode == "dpa" else "1"
        )
        if script.stem == "start_ulacache":
            config = json.loads((logs / "ulacache_config.json").read_text())
            assert config["adapter"] == ("native_tp" if mode == "tp" else "native_dpa")


@pytest.mark.parametrize("index_mode", ["replicated", "distributed"])
def test_index_ablation_disables_host_capacity_and_sizes_actual_gpu_pool(
    tmp_path, index_mode
):
    logs = tmp_path / index_mode
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/7_6_indexer/start_ulacache.sh"),
            "--model",
            "/models/glm",
            "--index-mode",
            index_mode,
            "--log-dir",
            str(logs),
            "--kv-dir",
            str(tmp_path / "kv"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    command = shlex.split(result.stdout.splitlines()[-1])
    config = json.loads((logs / "ulacache_config.json").read_text())
    assert config["options"]["cpu_offload"] is False
    assert config["options"]["cpu_slots"] == 0
    assert config["options"]["index_sharding"] == (index_mode == "distributed")
    assert "--max-total-tokens" not in command
    assert command[command.index("--max-running-requests") + 1] == "64"
    assert command[command.index("--cuda-graph-max-bs-decode") + 1] == "64"


def test_decode_window_uses_only_simultaneous_decode():
    points = [
        dict(seconds=0, tokens=1),
        dict(seconds=2, tokens=101),
        dict(seconds=4, tokens=201),
    ]
    records = [dict(start=0, observations=points), dict(start=1, observations=points)]
    result = decode.common_window(records, 1, 201)
    assert result["seconds"] == 3
    assert result["tokens_per_second"] == 100
    assert result["mean_step_ms"] == 20
    records[1]["start"] = 5
    with pytest.raises(ValueError, match="No common"):
        decode.common_window(records, 1, 201)


def test_fill_client_uses_native_requests_and_reuses_committed_records(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, value):
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.reply(
                {
                    "loads": [
                        {"num_running_reqs": 0, "num_waiting_reqs": 0, "queues": {}}
                    ]
                }
            )

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/generate":
                self.reply({})
                return
            assert request["sampling_params"]["max_new_tokens"] == 1
            key = request["sampling_params"]["custom_params"]["udcache_snapshot_key"]
            seen.append(key)
            directory = bank / hashlib.sha256(key.encode()).hexdigest()
            (directory / "target").mkdir(parents=True)
            (directory / "target/COMMITTED.json").write_text("{}")
            value = dict(
                complete=True,
                key=key,
                input_ids_sha256="digest",
                draft=None,
                first_output_token=7,
                source_manifest_sha256="0" * 64,
            )
            (directory / "SNAPSHOT.json").write_text(json.dumps(value))
            self.reply({"meta_info": {"completion_tokens": 1}, "output_ids": [7]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    data = tmp_path / "requests.jsonl"
    data.write_text(json.dumps({"key": "a", "input_ids": [1, 2, 3]}) + "\n")
    args = SimpleNamespace(
        url=f"http://127.0.0.1:{server.server_port}",
        dataset=data,
        kv_dir=bank,
        out=tmp_path / "first",
        limit=None,
        batch_size=1,
        attn_dp_size=1,
    )
    try:
        fill_kv.main(args)
        args.out = tmp_path / "second"
        fill_kv.main(args)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert seen == ["a"]
    assert json.loads((args.out / "COMPLETE.json").read_text())["saved"] == 1
