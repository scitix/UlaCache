"""Launcher-only tracing cannot silently run an expensive empty capture."""

import json
from pathlib import Path
import pytest
from scripts.decode import validate_sparse_trace_startup


def fixture(tmp_path):
    trace, geometry = tmp_path / "traces", tmp_path / "geometry"
    trace.mkdir()
    geometry.mkdir()
    environment = dict(
        UDCACHE_SPARSE_TRACE_DIR=str(trace),
        UDCACHE_NATIVE_GEOMETRY_DIR=str(geometry),
        UDCACHE_SPARSE_TRACE_REQUESTS="16",
        UDCACHE_SPARSE_TRACE_QUERY_ROWS="48",
        UDCACHE_SPARSE_TRACE_FRAMES="512",
    )
    for rank in range(8):
        (trace / f"observer.rank{rank}.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    pid=1000 + rank,
                    rank=rank,
                    groups=21,
                    requests=16,
                    query_rows=48,
                    frames=512,
                    extra_gpu_bytes=1024,
                )
            )
        )
        (geometry / f"CAPTURE.rank{rank}.json").write_text(
            json.dumps(
                dict(
                    pid=1000 + rank,
                    buckets={
                        "16:3": dict(
                            size=16,
                            verify_width=3,
                            layers={str(i): {} for i in range(21)},
                        )
                    },
                )
            )
        )
    return environment


def test_formal_serving_does_not_require_a_trace_observer():
    assert validate_sparse_trace_startup({}) is None


def test_every_worker_requires_an_actual_observer_and_matching_graph(tmp_path):
    environment = fixture(tmp_path)
    result = validate_sparse_trace_startup(environment)
    assert (
        result["passed"]
        and len(result["observers"]) == 8
        and result["before_prefix_restore"]
    )
    (Path(environment["UDCACHE_SPARSE_TRACE_DIR"]) / "observer.rank7.json").unlink()
    with pytest.raises(ValueError, match="missing on rank7"):
        validate_sparse_trace_startup(environment)


def test_a_launcher_receipt_cannot_substitute_for_a_native_worker(tmp_path):
    environment = fixture(tmp_path)
    p = Path(environment["UDCACHE_NATIVE_GEOMETRY_DIR"]) / "CAPTURE.rank0.json"
    value = json.loads(p.read_text())
    value["pid"] = 99
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differs on rank0"):
        validate_sparse_trace_startup(environment)
