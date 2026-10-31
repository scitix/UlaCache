"""Standalone saved-KV decode benchmark. No experiment plan or cluster runtime.

Use fill_kv.py once, then start the selected method with --phase decode.
TTFT/prefill and disk restoration stay outside the common decode window.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import statistics
import csv
from urllib.error import URLError
import uuid
from urllib.request import Request, urlopen


def validate_sparse_trace_startup(environment, *, ranks=8):
    """Require actual worker observers before restoring a large diagnostic cohort."""
    if not environment.get("UDCACHE_SPARSE_TRACE_DIR"):
        return None
    root = Path(environment["UDCACHE_SPARSE_TRACE_DIR"])
    geometry_root = Path(environment["UDCACHE_NATIVE_GEOMETRY_DIR"])
    requests = int(environment["UDCACHE_SPARSE_TRACE_REQUESTS"])
    queries = int(environment["UDCACHE_SPARSE_TRACE_QUERY_ROWS"])
    frames = int(environment["UDCACHE_SPARSE_TRACE_FRAMES"])
    receipts = []
    for rank in range(ranks):
        observer, geometry = (
            root / f"observer.rank{rank}.json",
            geometry_root / f"CAPTURE.rank{rank}.json",
        )
        if not observer.exists() or not geometry.exists():
            raise ValueError(
                f"Sparse trace observer or native Graph receipt missing on rank{rank}"
            )
        actual, captured = (
            json.loads(observer.read_bytes()),
            json.loads(geometry.read_bytes()),
        )
        if (
            not actual.get("passed")
            or actual["rank"] != rank
            or actual["groups"] != 21
            or actual["requests"] != requests
            or actual["query_rows"] != queries
            or actual["frames"] != frames
            or actual["extra_gpu_bytes"] <= 0
            or actual["pid"] != captured["pid"]
            or not any(
                v["size"] >= requests
                and v["verify_width"] == 3
                and len(v["layers"]) == 21
                for v in captured["buckets"].values()
            )
        ):
            raise ValueError(
                f"Sparse trace worker or captured query geometry differs on rank{rank}"
            )
        receipts.append(actual)
    return dict(passed=True, observers=receipts, before_prefix_restore=True)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def idle(url, *, timeout=30, expected_ranks=None):
    """Wait for native cleanup after its final streaming response, outside timing."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urlopen(url + "/v1/loads", timeout=min(10, timeout)) as response:
                value = json.load(response)
        except (URLError, TimeoutError) as error:
            value = {"unavailable": repr(error)}
        rows = value.get("loads", [])
        # HTTP readiness can precede the first scheduler load report. A partial
        # report is not evidence of an idle scheduler; wait for queue counters.
        complete = expected_ranks is None or (
            len(rows) == expected_ranks
            and {row.get("dp_rank") for row in rows} == set(range(expected_ranks))
        )
        if (
            rows
            and complete
            and all(
                {"num_running_reqs", "num_waiting_reqs", "queues"} <= row.keys()
                and not row["num_running_reqs"]
                and not row["num_waiting_reqs"]
                and not any(row["queues"].values())
                for row in rows
            )
        ):
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Native request cleanup did not drain: {value}")
        time.sleep(0.05)


def common_window(records, first, last):
    series = [
        [(r["start"] + p["seconds"], p["tokens"]) for p in r["observations"]]
        for r in records
    ]
    begin = max(next(t for t, n in points if n >= first) for points in series)
    end = min(max(t for t, n in points if n <= last) for points in series)
    if end <= begin:
        raise ValueError("No common full-cohort decode window")

    def count(points, when):
        for (a, na), (b, nb) in zip(points, points[1:]):
            if a <= when <= b and b > a:
                return na + (nb - na) * (when - a) / (b - a)
        raise ValueError("Decode boundary lacks an SSE bracket")

    advance = [count(points, end) - count(points, begin) for points in series]
    return dict(
        seconds=end - begin,
        output_tokens=sum(advance),
        tokens_per_second=sum(advance) / (end - begin),
        mean_step_ms=statistics.mean((end - begin) * 1000 / n for n in advance),
        scope="Client common full-cohort SSE decode window; startup, disk restore, first token and drain excluded",
    )


def latest_pause_acknowledgements(directory):
    """Return the last actually processed native pause for each global rank."""
    rows = {}
    for path in directory.glob("pause_ack.*.json"):
        row = json.loads(path.read_text())
        rank = row["rank"]
        if rank not in rows or row["sequence"] > rows[rank]["sequence"]:
            rows[rank] = row
    return rows


def pause_acknowledgements_ready(rows, previous, world_size):
    return all(
        rank in rows
        and rows[rank]["engine_paused"]
        and rows[rank]["sequence"] > previous.get(rank, 0)
        for rank in range(world_size)
    )


def resolve_rank_capacities(loads, rank_count, override=None):
    """Read every rank's actual logical capacity before admitting a cohort."""
    capacities = {}
    for row in loads.get("loads", []):
        rank = row.get("dp_rank")
        capacity = row.get("logical_token_capacity") or row.get("max_total_num_tokens")
        if (
            type(rank) is not int
            or not 0 <= rank < rank_count
            or rank in capacities
            or type(capacity) is not int
            or capacity <= 0
        ):
            raise ValueError("Native load report has invalid or duplicate KV capacity")
        capacities[rank] = capacity
    if set(capacities) != set(range(rank_count)):
        raise ValueError("Native load report is missing attention-DP ranks")
    if override is not None and (
        type(override) is not int
        or override <= 0
        or any(override > capacity for capacity in capacities.values())
    ):
        raise ValueError("Declared KV capacity exceeds the actual native allocation")
    return [override or capacities[rank] for rank in range(rank_count)]


def route_requests(requests, capacities, max_local_bs, output_tokens, assignment=None):
    """Preserve request order, falling back when the next rank lacks KV slots."""
    used, counts = [0] * len(capacities), [0] * len(capacities)
    routes, next_rank = [], 0
    if assignment is not None and len(assignment) != len(requests):
        raise ValueError("Saved routing must cover this exact request cohort")
    for index, row in enumerate(requests):
        count = len(row["input_ids"])
        reserved = ((count + output_tokens + 452 + 63) // 64) * 64
        candidates = [
            (next_rank + step) % len(capacities) for step in range(len(capacities))
        ]
        if assignment is not None:
            saved = assignment[index]
            rank = saved.get("attn_dp_rank")
            if (
                saved.get("key") != row["key"]
                or type(rank) is not int
                or not 0 <= rank < len(capacities)
            ):
                raise ValueError(
                    "Saved routing has a different request order or invalid rank"
                )
            candidates = [rank]
        rank = next(
            (
                rank
                for rank in candidates
                if used[rank] + reserved <= capacities[rank]
                and counts[rank] < max_local_bs
            ),
            None,
        )
        if rank is None:
            raise ValueError("Requested batch exceeds the actual KV/request capacity")
        routes.append(dict(key=row["key"], attn_dp_rank=rank, input_tokens=count))
        used[rank] += reserved
        counts[rank] += 1
        next_rank = (rank + 1) % len(capacities)
    return dict(route=routes, rank_capacities=capacities, reserved_tokens=used)


def run_point(args):
    args.out.mkdir(parents=True, exist_ok=False)
    raw = args.dataset.read_bytes()
    all_requests = [json.loads(line) for line in raw.splitlines() if line]
    if len(all_requests) < args.batch_size:
        raise ValueError("The dataset has fewer requests than the requested batch")
    manifest = {"requests_sha256": hashlib.sha256(raw).hexdigest()}
    replay = None
    if getattr(args, "replay", None):
        from ulacache.tools.fixed_output import ReplayTrace

        replay = ReplayTrace(args.replay)
        if (
            replay.data["dataset_sha256"] != manifest["requests_sha256"]
            or replay.data["output_tokens"] != args.output_tokens
        ):
            raise ValueError("Replay trace and benchmark input/output budgets differ")
    url = args.url.rstrip("/")
    loads = idle(url, expected_ranks=args.attn_dp_size)
    capacities = resolve_rank_capacities(loads, args.attn_dp_size, args.kv_slots)
    assignment = None
    routing_file = getattr(args, "routing_file", None)
    if routing_file is not None:
        routing = json.loads(routing_file.read_bytes())
        if routing.get("requests_sha256") != manifest["requests_sha256"]:
            raise ValueError("Saved routing belongs to a different dataset")
        assignment = routing["routes"].get(str(args.batch_size))
    admission = route_requests(
        all_requests[: args.batch_size],
        capacities,
        args.max_local_bs,
        args.output_tokens,
        assignment=assignment,
    )
    save(args.out / "routing.json", admission)
    selected = all_requests[: args.batch_size]
    routes = admission["route"][: args.batch_size]
    assert [r["key"] for r in routes] == [r["key"] for r in selected]
    counts = [0] * len(admission["rank_capacities"])
    for route in routes:
        counts[route["attn_dp_rank"]] += 1

    def post(path, payload):
        request = Request(
            url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            return response.read()

    idle(url)
    post("/flush_cache", {})
    cohort = uuid.uuid4().hex
    previous_pause = {
        rank: row["sequence"]
        for rank, row in latest_pause_acknowledgements(args.admission_receipts).items()
    }
    control = None
    if getattr(args, "native_scheduler_receipt", None):
        from sglang.srt.ulacache.native_scheduler_admission import NativeSchedulerControl

        control = NativeSchedulerControl(args.native_scheduler_receipt)
        try:
            save(args.out / "NATIVE_PAUSE.json", control.pause())
        except BaseException:
            control.close()
            raise
    else:
        post("/pause_generation", dict(mode="in_place"))
    paused = True
    try:
        deadline = time.monotonic() + 180
        while True:
            pause_acks = latest_pause_acknowledgements(args.admission_receipts)
            if pause_acknowledgements_ready(
                pause_acks, previous_pause, args.world_size
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "All native ranks did not acknowledge execution pause"
                )
            time.sleep(0.05)
        save(
            args.out / "PAUSED.json",
            dict(global_ranks=args.world_size, acknowledgements=pause_acks),
        )
        if args.require_graph_padding_reset:
            if not all(
                row.get("graph_padding_reset") is True for row in pause_acks.values()
            ):
                raise RuntimeError(
                    "Every native rank must acknowledge normalized Graph padding"
                )
            resets = {}
            for path in args.admission_receipts.glob("graph_padding_reset.*.json"):
                row = json.loads(path.read_text())
                rank = row["rank"]
                if (
                    rank in pause_acks
                    and row["pause_sequence"] == pause_acks[rank]["sequence"]
                ):
                    if (
                        rank in resets
                        or row.get("reset") is not True
                        or not row.get("buffers")
                    ):
                        raise RuntimeError(
                            "Missing or duplicate actual Graph padding reset proof"
                        )
                    resets[rank] = row
            if set(resets) != set(range(args.world_size)):
                raise RuntimeError("Graph padding reset receipts are incomplete")
            save(
                args.out / "GRAPH_PADDING_RESET_VERIFIED.json",
                dict(
                    ranks=resets,
                    boundary="All ranks drained and paused before restored-cohort admission",
                ),
            )
    except BaseException as error:
        save(
            args.out / "FAILED.json",
            dict(error=repr(error), phase="pause_acknowledgement"),
        )
        try:
            post("/continue_generation", {})
        finally:
            if control is not None:
                control.close()
        raise
    pool = ThreadPoolExecutor(max_workers=args.batch_size)

    def generate(index):
        row, route = selected[index], routes[index]
        rank = route["attn_dp_rank"]
        params = dict(
            udcache_snapshot_key=row["key"],
            udcache_snapshot_batch=args.batch_size,
            udcache_snapshot_local_batch=counts[rank],
            udcache_snapshot_dataset_index=index,
            udcache_snapshot_cohort=cohort,
        )
        if replay is not None:
            params.update(
                ulacache_replay_round=args.repeat, ulacache_replay_sha256=replay.sha256
            )
        payload = dict(
            input_ids=row["input_ids"],
            rid=cohort + "-" + row["key"],
            stream=True,
            routed_dp_rank=rank,
            sampling_params=dict(
                temperature=0,
                max_new_tokens=args.output_tokens,
                ignore_eos=True,
                skip_special_tokens=False,
                custom_params=params,
            ),
        )
        query = Request(
            url + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start, observations, output_ids, final = time.perf_counter(), [], [], None
        with urlopen(query, timeout=14400) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    break
                final = json.loads(raw)
                if "error" in final:
                    raise RuntimeError(str(final))
                output_ids.extend(final.get("output_ids", []))
                observations.append(
                    dict(
                        seconds=time.perf_counter() - start,
                        tokens=final["meta_info"]["completion_tokens"],
                    )
                )
        record = dict(
            key=row["key"],
            dataset_index=index,
            routed_dp_rank=rank,
            start=start,
            observations=observations,
            output_ids=output_ids,
            meta_info=final["meta_info"] if final else None,
        )
        save(args.out / f"request_{index:05d}.json", record)
        if (
            final is None
            or final["meta_info"]["completion_tokens"] != args.output_tokens
            or len(output_ids) != args.output_tokens
        ):
            raise RuntimeError(
                "Generation did not produce the exact fixed output budget"
            )
        if final["meta_info"].get("num_retractions", 0):
            raise RuntimeError("Retraction invalidates the full-cohort decode window")
        if replay is not None:
            episode = replay.episode(
                args.attn_dp_size, args.repeat, args.batch_size, rank
            )
            if output_ids != episode["expected_output_ids"][row["key"]]:
                raise RuntimeError("Output token IDs differ from the recorded baseline")
            for name, expected in episode["expected_meta"][row["key"]].items():
                if final["meta_info"].get(name) != expected:
                    raise RuntimeError(f"Recorded MTP counter differs: {name}")
        return record

    futures = [pool.submit(generate, index) for index in range(args.batch_size)]
    try:
        deadline = time.monotonic() + 180
        while True:
            if any(f.done() and f.exception() for f in futures):
                raise RuntimeError(
                    "A saved-prefix request failed before full admission"
                )
            receipts = []
            for path in args.admission_receipts.glob("admission.*.json"):
                row = json.loads(path.read_text())
                if row["request_cohort"] == cohort:
                    receipts.append(row)
            by_rank = {r["attn_dp_rank"]: r for r in receipts}
            ready = True
            for rank, count in enumerate(counts):
                if not count:
                    continue
                expected = sorted(r["key"] for r in routes if r["attn_dp_rank"] == rank)
                ready &= (
                    rank in by_rank
                    and by_rank[rank]["native_waiting_count"] == count
                    and by_rank[rank]["waiting_keys"] == expected
                )
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "All routed saved-prefix requests did not enter the paused cohort"
                )
            time.sleep(0.1)
        save(
            args.out / "ADMITTED.json",
            dict(
                global_bs=args.batch_size,
                per_rank_counts=counts,
                routes=routes,
                admission_receipts=receipts,
                original_native_pause=True,
            ),
        )
        post("/continue_generation", {})
        paused = False
        save(
            args.out / "PROGRESS.json",
            dict(
                phase="decode",
                batch=args.batch_size,
                output_tokens=args.output_tokens,
                first=args.first,
                last=args.last,
            ),
        )
        records = [f.result() for f in futures]
        idle(url)
        if replay is not None:
            receipts = [
                json.loads(path.read_bytes())
                for path in args.admission_receipts.glob("replay.*.json")
            ]
            receipts = [r for r in receipts if r["request_cohort"] == cohort]
            expected_receipts = (
                8 if args.attn_dp_size == 1 else sum(c > 0 for c in counts)
            )
            if len(receipts) != expected_receipts or not all(
                r["complete"]
                and r["trace_sha256"] == replay.sha256
                and r["frames"]
                == r["expected_frames"]
                == r["model_calls"]
                == r["sampler_calls"]
                for r in receipts
            ):
                raise RuntimeError(
                    "Recorded MTP forward/sampler receipts are incomplete"
                )
            save(
                args.out / "REPLAY_VERIFIED.json",
                dict(complete=True, receipts=receipts),
            )
        correct = sum(r["meta_info"].get("spec_num_correct_drafts", 0) for r in records)
        proposed = sum(
            r["meta_info"].get("spec_num_proposed_drafts", 0) for r in records
        )
        result = dict(
            complete=True,
            batch=args.batch_size,
            output_tokens=args.output_tokens,
            request_cohort=cohort,
            decode=common_window(records, args.first, args.last),
            requests_sha256=manifest["requests_sha256"],
            per_rank_counts=counts,
            spec_num_correct_drafts=correct,
            spec_num_proposed_drafts=proposed,
            spec_accept_rate=correct / proposed if proposed else None,
            fixed_output_count_verified=True,
            temperature=0,
            ignore_eos=True,
            fixed_output_ids_verified=replay is not None,
            fixed_mtp_trajectory_verified=replay is not None and replay.speculative,
            replay_sha256=replay.sha256 if replay is not None else None,
            graph_padding_reset_verified=args.require_graph_padding_reset,
            natural_correctness_claim=False,
            model_prefill_in_measured_run="must be zero in matching server restore receipts",
            repetitions=1,
            scope="Client full-cohort steady decode only; restore, initial output, first64 and final64 tokens excluded",
        )
        save(args.out / "metrics.json", result)
        print(json.dumps(dict(batch=args.batch_size, **result["decode"])), flush=True)
    except BaseException as error:
        save(args.out / "FAILED.json", dict(error=repr(error)))
        if paused:
            post("/continue_generation", {})
            paused = False
        raise
    finally:
        try:
            if paused:
                post("/continue_generation", {})
        finally:
            pool.shutdown(wait=True)
            if control is not None:
                control.close()


def main():
    parser = argparse.ArgumentParser(
        description="Measure decode over saved native KV; start a server with --phase decode first"
    )
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--dataset", type=Path, required=True, help="JSONL with key and input_ids"
    )
    parser.add_argument(
        "--receipts",
        type=Path,
        required=True,
        help="Server log directory's restore subdirectory",
    )
    parser.add_argument("--attn-dp-size", type=int, choices=(1, 8), default=8)
    parser.add_argument(
        "--batch-sizes", default="16", help="Comma-separated concurrent request counts"
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--replay",
        type=Path,
        help="Optional recorded output IDs and MTP workload; requires the same server trace",
    )
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument(
        "--routing-file",
        type=Path,
        help="Optional dataset-bound per-batch rank assignments; every assignment is capacity checked",
    )
    parser.add_argument(
        "--kv-slots",
        type=int,
        help="Optional lower limit per rank; defaults to actual server KV capacity",
    )
    parser.add_argument(
        "--max-local-bs", type=int, help="Request slots per attention-DP rank"
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Optional scheduler trace directory; verify worker hooks before KV IO",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.trace_dir:
        validate_sparse_trace_startup(
            dict(
                UDCACHE_SPARSE_TRACE_DIR=str(args.trace_dir / "traces"),
                UDCACHE_NATIVE_GEOMETRY_DIR=str(args.trace_dir / "native_geometry"),
                UDCACHE_SPARSE_TRACE_REQUESTS="16",
                UDCACHE_SPARSE_TRACE_QUERY_ROWS="48",
                UDCACHE_SPARSE_TRACE_FRAMES="512",
            )
        )
    batches = [int(n) for n in args.batch_sizes.split(",")]
    if args.rounds < 1 or not batches or min(batches) < 1 or args.output_tokens <= 128:
        parser.error(
            "Positive batches/rounds and more than128 output tokens are required"
        )
    args.max_local_bs = args.max_local_bs or (64 if args.attn_dp_size == 1 else 32)
    args.world_size = 8
    args.admission_receipts = args.receipts
    args.native_scheduler_receipt = (
        args.receipts / "scheduler_control.rank0.json"
        if args.attn_dp_size == 1
        else None
    )
    args.require_graph_padding_reset = True
    args.first, args.last = 64, args.output_tokens - 64
    root = args.out
    root.mkdir(parents=True, exist_ok=False)
    save(
        root / "options.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    rows = []
    for repeat in range(1, args.rounds + 1):
        args.repeat = repeat
        for batch in batches:
            args.batch_size = batch
            args.out = root / f"bs{batch:04d}-r{repeat}"
            run_point(args)
            result = json.loads((args.out / "metrics.json").read_bytes())
            rows.append(
                dict(
                    batch=batch,
                    repeat=repeat,
                    output_tokens_per_second=result["decode"]["tokens_per_second"],
                    tpot_ms=result["decode"]["mean_step_ms"],
                    draft_accept_rate=result["spec_accept_rate"],
                )
            )
            with (root / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    save(root / "COMPLETE.json", dict(complete=True, points=rows))


if __name__ == "__main__":
    main()
