"""Replay whole natural-generation conversations against one existing service.

This client never launches, signals, reloads or aborts a model. A failed point
stops new turns and drains all client requests before native idle verification.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import uuid

import math


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def token_sha(ids):
    return sha(json.dumps(ids, separators=(",", ":")).encode())


def percentile(values, fraction):
    values = sorted(values)
    return values[int((len(values) - 1) * fraction)] if values else None


class SSEDecoder:
    """Parse complete SSE events across arbitrary TCP boundaries and CRLF."""

    def __init__(self):
        self.buffer = b""
        self.data = []

    def _line(self, line):
        line = line.rstrip(b"\r")
        if not line:
            result = b"\n".join(self.data) if self.data else None
            self.data = []
            return result
        if line.startswith(b"data:"):
            self.data.append(line[5:].removeprefix(b" "))
        return None

    def feed(self, chunk):
        self.buffer += chunk
        results = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            result = self._line(line)
            if result is not None:
                results.append(result)
        return results

    def finish(self):
        results = []
        if self.buffer:
            result = self._line(self.buffer)
            self.buffer = b""
            if result is not None:
                results.append(result)
        result = self._line(b"")
        if result is not None:
            results.append(result)
        return results


class StreamAccumulator:
    def __init__(self, *, output_mode="incremental"):
        if output_mode not in ("incremental", "cumulative"):
            raise ValueError("Explicit native output mode required")
        self.mode = output_mode
        self.ids, self.observations, self.final = [], [], None

    def add(self, event, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Client token timestamp must be finite and nonnegative")
        if not isinstance(event, dict) or "error" in event:
            raise ValueError(("Native generation error", event))
        chunk = event.get("output_ids", [])
        if not isinstance(chunk, list) or any(
            type(t) is not int or t < 0 for t in chunk
        ):
            raise ValueError("Native output IDs must be nonnegative integers")
        before = len(self.ids)
        meta = event.get("meta_info", {})
        if self.mode == "incremental":
            self.ids.extend(chunk)
        elif not chunk and meta.get("completion_tokens") == before:
            # Native final bookkeeping may carry metadata without output IDs.
            pass
        else:
            if chunk[:before] != self.ids or len(chunk) < before:
                raise ValueError(
                    "Cumulative native outputs changed their committed prefix"
                )
            self.ids = list(chunk)
        if meta.get("completion_tokens") != len(self.ids):
            raise ValueError(
                (
                    "Native output mode/count mismatch",
                    self.mode,
                    meta.get("completion_tokens"),
                    len(self.ids),
                )
            )
        self.final = event
        count = len(self.ids) - before
        if count:
            self.observations.append(
                dict(seconds=elapsed, tokens=len(self.ids), tokens_in_chunk=count)
            )
        return count

    def metrics(self):
        if not self.observations:
            raise ValueError("No native output token arrived")
        times = [v["seconds"] for v in self.observations]
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError("Client token timestamps are not monotonic")
        gaps = [b - a for a, b in zip(times, times[1:])]
        span = times[-1] - times[0]
        return dict(
            ttft_seconds=times[0],
            last_token_seconds=times[-1],
            tpot_seconds=span / (len(self.ids) - 1) if len(self.ids) > 1 else None,
            decode_arrival_span_seconds=span,
            observed_token_bursts=len(times),
            first_burst_tokens=self.observations[0]["tokens_in_chunk"],
            max_burst_gap_seconds=max(gaps) if gaps else None,
            p95_burst_gap_seconds=percentile(gaps, 0.95),
        )


def native_counters(meta, *, spec_on=True, speculative_steps=2):
    names = ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts")
    if not spec_on:
        return dict.fromkeys(names)
    value = {n: meta[n] for n in names}
    if any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError(("Invalid native MTP counters", value))
    if (
        value["spec_verify_ct"] <= 0
        or value["spec_num_proposed_drafts"]
        != speculative_steps * value["spec_verify_ct"]
    ):
        raise ValueError(
            ("Native MTP counters disagree with configured draft steps", value)
        )
    if value["spec_num_correct_drafts"] > value["spec_num_proposed_drafts"]:
        raise ValueError("Native accepted drafts exceed proposals")
    return value


def aggregate(records):
    tokens = sum(r["output_tokens"] for r in records)
    verify = sum(r.get("spec_verify_ct") or 0 for r in records)
    correct = sum(r.get("spec_num_correct_drafts") or 0 for r in records)
    proposed = sum(r.get("spec_num_proposed_drafts") or 0 for r in records)
    return dict(
        requests=len(records),
        output_tokens=tokens,
        spec_verify_ct=verify,
        spec_num_correct_drafts=correct,
        spec_num_proposed_drafts=proposed,
        native_token_weighted_accept_length=tokens / verify if verify else None,
        emitted_after_initial_token_per_verify=(tokens - len(records)) / verify
        if verify
        else None,
        draft_accept_rate=correct / proposed if proposed else None,
    )


def summarize(records, elapsed_seconds):
    value = dict(
        requests=len(records),
        elapsed_seconds=elapsed_seconds,
        output_tokens=sum(r["output_tokens"] for r in records),
    )
    value["output_tokens_per_second"] = value["output_tokens"] / elapsed_seconds
    for name in ("ttft", "tpot", "total"):
        values = [
            r[name + "_seconds"]
            for r in records
            if r.get(name + "_seconds") is not None
        ]
        value[name + "_mean_seconds"] = sum(values) / len(values) if values else None
        value[name + "_p50_seconds"] = percentile(values, 0.5)
        value[name + "_p95_seconds"] = percentile(values, 0.95)
    denominator = sum(r["output_tokens"] - 1 for r in records if r["output_tokens"] > 1)
    value["tpot_token_weighted_seconds"] = (
        sum(r["decode_arrival_span_seconds"] for r in records if r["output_tokens"] > 1)
        / denominator
        if denominator
        else None
    )
    clean = [r for r in records if not r.get("num_retractions", 0)]
    context = sum(r["input_tokens"] for r in clean)
    reused = sum(r["cached_prefix_tokens"] for r in clean)
    value["prefix_cache"] = dict(
        unretracted_requests=len(clean),
        retracted_requests=len(records) - len(clean),
        context_tokens=context,
        cached_prefix_tokens=reused,
        token_weighted_hit_rate=reused / context if context else None,
        retraction_semantics="Retracted requests retain raw native reports but are excluded from original-prompt hit ratio",
    )
    value["acceptance"] = aggregate(records)
    return value


def valid_route(meta, plan, rank, *, allow_absent=False):
    dp, tp = plan.get("attention_dp"), plan.get("attention_tp")
    if type(dp) is not int or type(tp) is not int or (dp, tp) not in ((1, 8), (8, 1)):
        return False
    if type(rank) is not int or not 0 <= rank < dp:
        return False
    value = meta.get("dp_rank")
    return (
        (allow_absent and value is None)
        or (dp == 1 and value is None)
        or (type(value) is int and value == rank)
    )


async def generate(
    session,
    plan,
    ids,
    budget,
    rid,
    namespace,
    rank,
    failure_path,
    *,
    on_first_token=None,
):
    if type(budget) is not int or budget < 1 or not ids:
        raise ValueError(
            "Generation requires a nonempty prompt and positive output budget"
        )
    if type(rank) is not int or not valid_route({}, plan, rank, allow_absent=True):
        raise ValueError("Request rank differs from the declared attention topology")
    payload = dict(
        input_ids=list(ids),
        rid=rid,
        extra_key=namespace,
        routed_dp_rank=rank,
        stream=True,
        sampling_params=dict(
            temperature=0,
            max_new_tokens=budget,
            ignore_eos=True,
            skip_special_tokens=False,
            stream_interval=1,
        ),
    )
    started = time.perf_counter()
    decoder = SSEDecoder()
    state = StreamAccumulator(output_mode=plan.get("stream_output_mode", "incremental"))
    done = False
    try:
        async with session.post(
            plan["url"].rstrip("/") + "/generate", json=payload
        ) as response:
            if response.status != 200:
                raise RuntimeError((response.status, (await response.text())[:2000]))
            async for chunk in response.content.iter_any():
                for data in decoder.feed(chunk):
                    if data == b"[DONE]":
                        done = True
                        break
                    count = state.add(json.loads(data), time.perf_counter() - started)
                    if (
                        count
                        and len(state.observations) == 1
                        and on_first_token is not None
                    ):
                        on_first_token()
                if done:
                    break
            if not done:
                for data in decoder.finish():
                    if data == b"[DONE]":
                        done = True
                        continue
                    count = state.add(json.loads(data), time.perf_counter() - started)
                    if (
                        count
                        and len(state.observations) == 1
                        and on_first_token is not None
                    ):
                        on_first_token()
        if state.final is None or len(state.ids) != budget:
            raise ValueError(
                (
                    "Native response ended before requested output budget",
                    len(state.ids),
                    budget,
                )
            )
        meta = state.final["meta_info"]
        if (
            meta.get("id") != rid
            or meta.get("prompt_tokens") != len(ids)
            or not valid_route(meta, plan, rank)
            or meta.get("completion_tokens") != budget
        ):
            raise ValueError(("Native response request/rank/count differs", meta))
        if meta.get("finish_reason", {}).get("type") != "length":
            raise ValueError(("Native output did not reach its length budget", meta))
        retractions = meta.get("num_retractions", 0)
        cached = meta.get("cached_tokens", 0)
        if type(retractions) is not int or retractions < 0:
            raise ValueError("Native retraction count must be a nonnegative integer")
        upper = len(ids) + (budget - 1 if retractions else 0)
        if type(cached) is not int or not 0 <= cached <= upper:
            raise ValueError(("Invalid native cached-token report", cached, upper))
        return dict(
            output_ids=state.ids,
            meta_info=meta,
            raw_dp_rank=meta.get("dp_rank"),
            resolved_attention_dp_rank=rank,
            dp_rank_resolution="declared_attention_topology_and_native_response",
            total_seconds=time.perf_counter() - started,
            observations=state.observations,
            **state.metrics(),
            **native_counters(
                meta,
                spec_on=plan.get("spec_on", True),
                speculative_steps=plan.get("speculative_steps", 2),
            ),
        )
    except BaseException as error:
        detail = dict(
            error=repr(error),
            request_id=rid,
            input_tokens=len(ids),
            expected_output_tokens=budget,
            received_output_tokens=len(state.ids),
            partial_output_ids=state.ids,
            final_event=state.final,
            observations=state.observations,
            elapsed_seconds=time.perf_counter() - started,
            automatic_model_signals=False,
            automatic_native_abort=False,
        )
        save(failure_path, detail)
        raise


def input_statistics(rows):
    turns = sum(len(r["turns"]) for r in rows)
    first = sum(len(r["initial_input_ids"]) for r in rows)
    later = sum(len(t["append_input_ids"]) for r in rows for t in r["turns"][1:])
    outputs = sum(t["output_tokens"] for r in rows for t in r["turns"])
    return dict(
        episodes=len(rows),
        turns=turns,
        first_input_mean=first / len(rows),
        later_new_input_mean=later / (turns - len(rows)) if turns > len(rows) else None,
        new_input_mean_including_first=(first + later) / turns,
        output_mean=outputs / turns,
    )


class RecentTTFT:
    """Sticky failure from the latest32 completed requests, including first turns."""

    def __init__(self, limit=120.0, window=32):
        self.limit, self.values, self.failed = limit, deque(maxlen=window), False

    def add(self, seconds):
        self.values.append(seconds)
        if len(self.values) == self.values.maxlen and self.mean > self.limit:
            self.failed = True

    @property
    def mean(self):
        return sum(self.values) / len(self.values) if self.values else None


def native_idle(snapshot, dp):
    rows = snapshot.get("loads", [])
    if len(rows) != dp or {r.get("dp_rank") for r in rows} != set(range(dp)):
        return False
    scalars = (
        "num_running_reqs",
        "num_waiting_reqs",
        "num_waiting_uncached_tokens",
        "num_used_tokens",
        "num_total_tokens",
        "num_active_tokens",
        "num_prealloc_ready_tokens",
    )
    queues = ("waiting", "grammar", "paused", "retracted", "prealloc_ready")
    return (
        all(type(r.get(k)) is int and r[k] == 0 for r in rows for k in scalars)
        and all(r.get("disaggregation", {}).get("mode") == "null" for r in rows)
        and all(
            type(r.get("queues", {}).get(k)) is int and r["queues"][k] == 0
            for r in rows
            for k in queues
        )
    )


def read_workload(path, expected_sha):
    raw = path.read_bytes()
    if expected_sha is not None and hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Workload checksum differs from the explicit plan")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if not rows or len({row["key"] for row in rows}) != len(rows):
        raise ValueError("Require uniquely identified complete conversations")
    for row in rows:
        if not row.get("whole_original_episode") or row.get("teacher_history_used"):
            raise ValueError(
                "Require whole observed conversations and natural new history"
            )
        if (
            not row["initial_input_ids"]
            or not row["turns"]
            or row["turns"][0]["append_input_ids"]
        ):
            raise ValueError("Invalid first-turn framing")
        if any(
            type(t["output_tokens"]) is not int or t["output_tokens"] < 2
            for t in row["turns"]
        ):
            raise ValueError("Output budgets must preserve the selected source lengths")
    return rows


async def read_idle(session, url, dp, *, maximum_wait_seconds=120, poll_seconds=0.25):
    deadline = time.perf_counter() + maximum_wait_seconds
    while True:
        async with session.get(url.rstrip("/") + "/v1/loads") as response:
            if response.status != 200:
                raise ValueError("Cannot verify native service idle state")
            value = await response.json()
        if native_idle(value, dp):
            return value
        if time.perf_counter() >= deadline:
            raise ValueError(
                (
                    "Service is not drained on all native DP ranks; HOLD without abortion",
                    value,
                )
            )
        await asyncio.sleep(poll_seconds)


async def replay(session, plan, rows, out, *, generate_request=generate):
    concurrency, instances = plan["concurrency"], plan["instances"]
    if (
        any(type(x) is not int or x < 1 for x in (concurrency, instances))
        or instances < concurrency
    ):
        raise ValueError("Instances must cover all positive concurrent user slots")
    if (
        plan["attention_dp"] not in (1, 8)
        or plan["attention_tp"] != 8 // plan["attention_dp"]
    ):
        raise ValueError("Require declared TP8 or DPA8 topology")
    run_key = uuid.uuid4().hex
    source_assigned = [rows[i % len(rows)] for i in range(instances)]
    smoke_turns = plan.get("functional_smoke_turns_per_instance")
    if smoke_turns is not None and (type(smoke_turns) is not int or smoke_turns < 1):
        raise ValueError("Functional smoke requires an explicit positive turn limit")
    assigned = (
        [dict(row, turns=row["turns"][:smoke_turns]) for row in source_assigned]
        if smoke_turns is not None
        else source_assigned
    )
    total_turns = sum(len(row["turns"]) for row in assigned)
    recent, records, errors = (
        RecentTTFT(
            limit=plan.get("ttft_limit", 120.0), window=plan.get("failure_window", 32)
        ),
        [],
        [],
    )
    completed_instances, active = 0, 0
    started = time.perf_counter()
    save(out / "WORKLOAD_STATISTICS.json", input_statistics(assigned))
    save(out / "SOURCE_WORKLOAD_STATISTICS.json", input_statistics(source_assigned))

    def progress():
        recent_stats = (
            summarize(records[-32:], max(time.perf_counter() - started, 1.0e-9))
            if records
            else {}
        )
        return dict(
            completed_turns=len(records),
            total_turns=total_turns,
            purpose="functional_smoke"
            if smoke_turns is not None
            else "whole_conversation_benchmark",
            full_original_conversation_replay_requested=smoke_turns is None,
            full_original_conversations_replayed=smoke_turns is None
            and completed_instances == instances,
            completed_instances=completed_instances,
            instances=instances,
            active_requests=active,
            recent_completed_ttft_mean_seconds=recent.mean,
            recent_completed_requests=len(recent.values),
            failed_ttft120=recent.failed,
            errors=errors,
            elapsed_seconds=time.perf_counter() - started,
            recent_completed_statistics={
                k: v
                for k, v in recent_stats.items()
                if k == "requests" or k.startswith(("ttft_", "tpot_", "total_"))
            },
        )

    async def user_slot(slot):
        nonlocal completed_instances, active
        # Preserve the paper's per-user conversation order and DP ownership.
        # Faster users must not consume another user's next conversation.
        for instance in range(slot, instances, concurrency):
            if recent.failed or errors:
                return
            row = assigned[instance]
            namespace = f"{run_key}:instance{instance}"
            history = list(row["initial_input_ids"])
            rank = slot % plan["attention_dp"]
            for turn, source in enumerate(row["turns"]):
                if recent.failed or errors:
                    return
                if turn:
                    history.extend(source["append_input_ids"])
                rid = f"{run_key}-i{instance}-t{turn}"
                directory = (
                    out / "requests" / f"instance{instance:05d}" / f"turn{turn:05d}"
                )
                active += 1
                try:
                    result = await generate_request(
                        session,
                        plan,
                        history,
                        source["output_tokens"],
                        rid,
                        namespace,
                        rank,
                        directory / "FAILURE.json",
                    )
                    save(directory / "RESPONSE.json", result)
                    meta = result["meta_info"]
                    record = {
                        k: v
                        for k, v in result.items()
                        if k not in ("output_ids", "meta_info", "observations")
                    }
                    record.update(
                        instance=instance,
                        user_slot=slot,
                        source_key=row["key"],
                        turn=turn,
                        request_id=rid,
                        attention_dp_rank=rank,
                        input_tokens=len(history),
                        appended_input_tokens=len(row["initial_input_ids"])
                        if not turn
                        else len(source["append_input_ids"]),
                        input_sha256=token_sha(history),
                        output_tokens=len(result["output_ids"]),
                        output_sha256=token_sha(result["output_ids"]),
                        cached_prefix_tokens=meta.get("cached_tokens", 0),
                        num_retractions=meta.get("num_retractions", 0),
                        response_path=str(
                            (directory / "RESPONSE.json").relative_to(out)
                        ),
                    )
                    records.append(record)
                    recent.add(record["ttft_seconds"])
                    history.extend(result["output_ids"])
                    save(directory / "METRICS.json", record)
                except Exception as error:
                    errors.append(dict(request_id=rid, error=repr(error)))
                    return
                finally:
                    active -= 1
                    save(out / "PROGRESS.json", progress())
            completed_instances += 1

    async def reporter(tasks):
        while any(not task.done() for task in tasks):
            save(out / "PROGRESS.json", progress())
            await asyncio.wait(tasks, timeout=5, return_when=asyncio.ALL_COMPLETED)

    tasks = [asyncio.create_task(user_slot(slot)) for slot in range(concurrency)]
    report = asyncio.create_task(reporter(tasks))
    # No wait_for, task cancellation or native request abortion at a deadline.
    await asyncio.gather(*tasks)
    await report
    value = dict(
        progress(),
        records=records,
        protocol=dict(
            episode_assignment="static_user_stride",
            attention_dp_assignment="user_slot_mod_attention_dp",
            natural_outputs_enter_history=True,
            source_assistant_ids_replayed=False,
            private_instance_prefix_namespace=True,
            output_budgets="source visible output lengths",
            failure="latest32 completed request mean TTFT >120 seconds; sticky",
            native_abort_sent=False,
            model_signals_sent=False,
        ),
    )
    value["summary"] = (
        summarize(records, time.perf_counter() - started) if records else None
    )
    save(out / "RESULTS.json", value)
    return value


async def execute(args):
    import aiohttp

    plan = dict(
        url=args.url,
        concurrency=args.users,
        instances=args.instances or 2 * args.users,
        attention_dp=args.attention_dp,
        attention_tp=8 // args.attention_dp,
        spec_on=args.spec == "on",
        speculative_steps=2,
        stream_output_mode=args.stream_output_mode,
        ttft_limit=args.ttft_limit,
        failure_window=args.failure_window,
        functional_smoke_turns_per_instance=args.max_turns,
    )
    rows = read_workload(args.traces, args.dataset_sha256)
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "options.json", plan)
    # No overall HTTP deadline: unknown in-flight work must remain owned.
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        save(
            args.out / "INITIAL_IDLE.json",
            await read_idle(session, plan["url"], plan["attention_dp"]),
        )
        try:
            result = await replay(session, plan, rows, args.out)
            idle = await read_idle(session, plan["url"], plan["attention_dp"])
            save(args.out / "FINAL_IDLE.json", idle)
        except Exception as error:
            save(
                args.out / "HOLD.json",
                dict(
                    error=repr(error),
                    safe_to_reuse_service=False,
                    automatic_abort=False,
                    automatic_signals=False,
                ),
            )
            raise
    clean = not result["errors"] and not result["active_requests"]
    passed = (
        clean
        and not result["failed_ttft120"]
        and result["completed_instances"] == plan["instances"]
    )
    terminal = (
        "FUNCTIONAL_SMOKE_COMPLETE.json"
        if passed and result["purpose"] == "functional_smoke"
        else "COMPLETE.json"
        if passed
        else "FAILED_TTFT120.json"
        if clean and result["failed_ttft120"]
        else "HOLD.json"
    )
    save(
        args.out / terminal,
        dict(
            passed=passed,
            safe_to_reuse_service=clean,
            completed_turns=result["completed_turns"],
            total_turns=result["total_turns"],
            native_all_rank_idle_verified=True,
            failure_is_capacity_proof=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--traces",
        "--dataset",
        type=Path,
        required=True,
        help="Whole-conversation JSONL",
    )
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument(
        "--instances", type=int, help="Conversations to replay; default2x users"
    )
    parser.add_argument("--attention-dp", type=int, choices=(1, 8), default=8)
    parser.add_argument("--spec", choices=("on", "off"), default="on")
    parser.add_argument(
        "--stream-output-mode",
        choices=("incremental", "cumulative"),
        default="incremental",
    )
    parser.add_argument("--ttft-limit", type=float, default=120)
    parser.add_argument("--failure-window", type=int, default=32)
    parser.add_argument(
        "--max-turns", type=int, help="Optional explicitly labeled short smoke test"
    )
    parser.add_argument("--dataset-sha256", help="Optional dataset integrity check")
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
