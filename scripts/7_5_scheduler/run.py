"""Replay fresh NVFP4 scheduler traces and render Figure11 previews."""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from threading import Lock
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
LOADS = (1, 2, 4, 8, 16, 32, 64, 128)
METHODS = ("pb_pin_only", "hisparse_request", "hisparse", "native_linked_list")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def capture_manifest(path):
    value = json.loads(path.read_bytes())
    if (
        value.get("model") != "GLM-5.3-NVFP4"
        or not value.get("passed")
        or value.get("MTP") != [2, 1, 3]
        or value.get("frames") != 512
        or value.get("groups") != 21
        or value.get("topk") != 2048
        or not (
            value.get("checkpoint_manifest_sha256")
            or value.get("checkpoint_config_sha256")
        )
    ):
        raise ValueError(
            "Require a completed, checkpoint-bound GLM-5.3 NVFP4 MTP2 capture"
        )
    return value


def trace_manifest(args):
    if args.manifest is not None:
        return capture_manifest(args.manifest)
    root = args.capture_root
    observers = [
        json.loads((root / f"traces/observer.rank{rank}.json").read_bytes())
        for rank in range(8)
    ]
    identity = observers[0].get("checkpoint_config_sha256")
    if not identity or any(
        row.get("checkpoint_config_sha256") != identity for row in observers
    ):
        raise ValueError(
            "The trace must contain matching checkpoint identities on all eight ranks"
        )
    alias = root / "mtp2"
    alias.mkdir(exist_ok=True)
    if not (alias / "traces").exists():
        (alias / "traces").symlink_to("../traces", target_is_directory=True)
    return dict(
        passed=True,
        model="GLM-5.3-NVFP4",
        MTP=[2, 1, 3],
        frames=512,
        groups=21,
        topk=2048,
        checkpoint_config_sha256=identity,
    )


def assemble(root, manifest):
    rows = []
    for requests in LOADS:
        case, linked = root / f"request_r{requests}", root / f"linked_r{requests}"
        paths = [
            case / "COMPLETE.json",
            case / "REPETITIONS.json",
            case / "HIT_RATES.json",
            case / "CPU_HIT_RATES.json",
            linked / "COMPLETE.json",
        ]
        if not (case / "READY_TO_COLLECT.json").exists() or not all(
            path.exists() for path in paths
        ):
            continue
        complete, repetitions, hits, cpu_hits, probes = [
            json.loads(p.read_bytes()) for p in paths
        ]
        if not complete.get("passed") or not probes.get("passed"):
            raise ValueError("Failed replay cases cannot enter a preview")
        for method in METHODS:
            if method == "native_linked_list":
                values = probes["repetition_means"]["linked_list"]
                hit = next(v for v in cpu_hits["rows"] if v["method"] == method)
            else:
                values = [v["mean_ms"][method] for v in repetitions]
                hit = next(v for v in hits["rows"] if v["method"] == method)
            if len(values) != 3:
                raise ValueError("Require three reset replay or probe means")
            rows.append(
                dict(
                    requests=requests,
                    method=method,
                    latency_ms=statistics.median(values),
                    min_ms=min(values),
                    max_ms=max(values),
                    unique_historical_hit_rate=hit["unique_historical_hit_rate"],
                    hit_rate_source="CPU policy replay"
                    if method == "native_linked_list"
                    else "separate GPU observer",
                )
            )
    if rows:
        with (root / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save(
            root / "PREVIEW_MANIFEST.json",
            dict(
                manifest,
                comparison_sha256=digest(root / "comparison.csv"),
                accepted_rows=len(rows),
                full_matrix_complete=len(rows) == 32,
                model_throughput_claim=False,
                timing_unit="GPU metadata ms across21 Index groups; not TPOT",
            ),
        )
    return rows


def preview(args):
    manifest = capture_manifest(
        args.manifest or args.comparison.parent / "PREVIEW_MANIFEST.json"
    )
    if digest(args.comparison) != manifest["comparison_sha256"]:
        raise ValueError("Preview CSV differs from its bound measurements")
    with args.comparison.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    identities = [(r["method"], int(r["requests"])) for r in rows]
    if not rows or len(set(identities)) != len(rows):
        raise ValueError("Require unique measured scheduler points")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    # Match the manuscript's single-column Figure 11, including its original
    # palette, marker shapes, two-row legend and stacked panel geometry.
    styles = {
        "hisparse_request": ("HiSparse", "#D55E00", "^", "-.", 3.0),
        "native_linked_list": ("Linked-list LRU", "#8064A2", "s", "--", 4.2),
        "hisparse": ("HiSparse (shared pool)", "#009E73", "D", ":", 3.6),
        "pb_pin_only": ("PB-LRU", "#0072B2", "o", "-", 2.6),
    }
    configuration = {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.labelsize": 6.7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(configuration):
        width, height = 234, 222  # PDF points; exactly one manuscript column.
        fig = plt.figure(figsize=(width / 72, height / 72))
        axes = [
            fig.add_axes(
                [
                    44.46 / width,
                    (height - bottom) / height,
                    186.03 / width,
                    62.21 / height,
                ]
            )
            for bottom in (92.72, 181.72)
        ]
        handles = []
        for method, (label, color, marker, line, size) in styles.items():
            handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    marker=marker,
                    linestyle=line,
                    lw=1,
                    ms=size,
                    markerfacecolor="white"
                    if method in ("native_linked_list", "hisparse")
                    else color,
                    markeredgewidth=0.65,
                    label=label,
                )
            )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.55, 0.995),
            ncol=2,
            fontsize=6.4,
            columnspacing=1.0,
            handlelength=1.6,
            handletextpad=0.4,
            borderaxespad=0,
            labelspacing=0.25,
        )
        for order, method in enumerate(
            ("native_linked_list", "hisparse", "hisparse_request", "pb_pin_only")
        ):
            _, color, marker, line, size = styles[method]
            selected = sorted(
                (r for r in rows if r["method"] == method),
                key=lambda r: int(r["requests"]),
            )
            if not selected:
                continue
            x = [LOADS.index(int(r["requests"])) for r in selected]
            values = [float(r["latency_ms"]) for r in selected]
            style = dict(
                color=color,
                marker=marker,
                linestyle=line,
                lw=1,
                ms=size,
                markerfacecolor="white"
                if method in ("native_linked_list", "hisparse")
                else color,
                markeredgewidth=0.65,
                zorder=3 + order,
            )
            axes[0].errorbar(
                x,
                values,
                yerr=[
                    [v - float(r["min_ms"]) for v, r in zip(values, selected)],
                    [float(r["max_ms"]) - v for v, r in zip(values, selected)],
                ],
                capsize=1.4,
                elinewidth=0.65,
                **style,
            )
            axes[1].plot(
                x,
                [100 * float(r["unique_historical_hit_rate"]) for r in selected],
                **style,
            )
        for ax in axes:
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#D9DDE2", linewidth=0.45)
            ax.tick_params(length=2, width=0.5, pad=1.5)
            ax.set_xlim(-0.35, 7.35)
            ax.set_xticks(range(8), [str(v) for v in LOADS])
            for spine in ax.spines.values():
                spine.set_color("#8C949E")
        axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
        axes[0].set(yscale="log", ylim=(0.45, 16000))
        axes[0].yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axes[0].yaxis.set_minor_locator(NullLocator())
        axes[0].set_ylabel("Scheduling (ms/forward)", labelpad=3)
        axes[1].set_ylim(86, 101.1)
        axes[1].yaxis.set_major_locator(FixedLocator([88, 92, 96, 100]))
        axes[1].set_ylabel("GPU hit rate (%)", labelpad=3)
        for baseline, label in (
            (103, "(a) Scheduling time"),
            (202, "(b) KV hit rate"),
            (217, "Batch size"),
        ):
            fig.text(
                138 / width,
                (height - baseline) / height,
                label,
                ha="center",
                va="baseline",
                fontsize=6.8,
            )
        for extension in ("png", "pdf"):
            fig.savefig(args.out / f"figure11_nvfp4_preview.{extension}", dpi=300)
        plt.close(fig)
    shutil.copyfile(args.comparison, args.out / "comparison.csv")
    save(args.out / "MANIFEST.json", manifest)
    print(
        json.dumps(
            dict(
                preview=str(args.out),
                measured_points=len(rows),
                manuscript_overwritten=False,
            )
        )
    )


def replay(args):
    manifest = trace_manifest(args)
    sys.path.insert(0, str(HERE / "_runtime"))
    from validate_capture import validate

    manifest = validate(args.capture_root, manifest)
    devices = args.gpus.split(",")
    batches = tuple(int(value) for value in args.batch_sizes.split(","))
    if (
        not batches
        or len(set(batches)) != len(batches)
        or any(value not in LOADS for value in batches)
    ):
        raise ValueError(f"Select distinct batch sizes from {LOADS}")
    if len(set(devices)) != len(devices) or not devices:
        raise ValueError("Supply distinct explicitly assigned GPUs")
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        HERE / "_runtime",
        args.out / "_runtime",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(HERE / "pb_config.json", args.out / "PB_CONFIG.json")
    files = {
        str(p.relative_to(args.out)): digest(p)
        for p in args.out.rglob("*")
        if p.is_file()
    }
    save(
        args.out / "SOURCE.json",
        dict(files=files, capture=manifest),
    )
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        UDCACHE_RECENCY_POLICY="pb",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    compiler = subprocess.Popen(
        [
            "g++",
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            str(args.out / "_runtime/reference.cpp"),
            "-o",
            str(args.out / "cpu_reference.so"),
        ]
    )
    if compiler.wait():
        raise RuntimeError("CPU correctness oracle compilation failed")
    assignments = [list(batches[i :: len(devices)]) for i in range(len(devices))]
    assembly_lock = Lock()

    def worker(item):
        device, batches = item
        child_env = dict(env, CUDA_VISIBLE_DEVICES=device)
        for requests in batches:
            if not args.linked_only:
                case = args.out / f"request_r{requests}"
                case.mkdir()
                shutil.copyfile(args.out / "_runtime/replay.py", case / "replay.py")
                command = [
                    sys.executable,
                    str(case / "replay.py"),
                    "--capture-root",
                    str(args.capture_root),
                    "--requests",
                    str(requests),
                    "--backend",
                    "hisparse_request",
                    "--repeats",
                    "3",
                ]
                with (case / "run.log").open("xb", buffering=0) as log:
                    child = subprocess.Popen(
                        command, env=child_env, stdout=log, stderr=subprocess.STDOUT
                    )
                    save(
                        case / "PROCESS.json",
                        dict(pid=child.pid, device=device, command=command),
                    )
                    if child.wait():
                        raise RuntimeError(
                            f"Scheduler replay failed at BS{requests}; inspect {case}"
                        )
            linked = args.out / f"linked_r{requests}"
            linked.mkdir()
            shutil.copyfile(
                args.out / "_runtime/linked_probes.py", linked / "linked_probes.py"
            )
            command = [
                sys.executable,
                str(linked / "linked_probes.py"),
                "--capture-root",
                str(args.capture_root),
                "--requests",
                str(requests),
                "--repeats",
                "3",
            ]
            with (linked / "run.log").open("xb", buffering=0) as log:
                child = subprocess.Popen(
                    command, env=child_env, stdout=log, stderr=subprocess.STDOUT
                )
                if child.wait():
                    raise RuntimeError(
                        f"Linked-LRU probe failed at BS{requests}; inspect {linked}"
                    )
            if args.linked_only:
                receipt = json.loads((linked / "COMPLETE.json").read_text())
                if not receipt.get("passed") or receipt["requests"] != requests:
                    raise RuntimeError(f"Incomplete linked-LRU probe at BS{requests}")
                rows = receipt["probes"]
            else:
                save(
                    case / "READY_TO_COLLECT.json", dict(passed=True, requests=requests)
                )
                with assembly_lock:
                    rows = assemble(args.out, manifest)
            save(
                args.out / f"GPU_{device}_PROGRESS.json",
                dict(
                    device=device,
                    completed_batch=requests,
                    collected_rows=len(rows),
                    observed_unix=time.time(),
                ),
            )

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            list(pool.map(worker, zip(devices, assignments)))
        if args.linked_only:
            receipts = {
                str(batch): json.loads(
                    (args.out / f"linked_r{batch}/COMPLETE.json").read_text()
                )
                for batch in batches
            }
            save(
                args.out / "COMPLETE.json",
                dict(
                    passed=True,
                    scope="matched linked-LRU probes",
                    batches=batches,
                    full_matrix=False,
                    probes=receipts,
                ),
            )
        else:
            rows = assemble(args.out, manifest)
            if len(rows) != 4 * len(batches):
                raise RuntimeError("Selected scheduler replay points are incomplete")
            save(
                args.out / "COMPLETE.json",
                dict(
                    passed=True,
                    rows=len(rows),
                    batches=batches,
                    full_matrix=set(batches) == set(LOADS),
                    model="GLM-5.3-NVFP4",
                    MTP=[2, 1, 3],
                ),
            )

    except BaseException as error:
        save(
            args.out / "HOLD.json",
            dict(reason=repr(error), signals_sent=False, automatic_retry=False),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "replay", help="Run the GPU metadata and hit-rate matrix on fresh traces"
    )
    run.add_argument("--capture-root", "--trace-dir", type=Path, required=True)
    run.add_argument(
        "--manifest",
        type=Path,
        help="Optional retained capture metadata; new captures carry their own identity",
    )
    run.add_argument(
        "--gpus", required=True, help="Assigned devices, for example0,1,2,3,4,5,6,7"
    )
    run.add_argument(
        "--batch-sizes",
        default=",".join(map(str, LOADS)),
        help="Comma-separated subset of the measured batch grid",
    )
    run.add_argument(
        "--linked-only",
        action="store_true",
        help="Run matched linked-LRU probes without repeating the three-method replay",
    )
    run.add_argument("--out", type=Path, required=True)
    plot = commands.add_parser(
        "preview",
        help="Plot completed fresh NVFP4 replay points without changing the paper",
    )
    plot.add_argument("--comparison", type=Path, required=True)
    plot.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to the CSV directory PREVIEW_MANIFEST.json",
    )
    plot.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"replay": replay, "preview": preview}[args.command](args)


if __name__ == "__main__":
    main()
