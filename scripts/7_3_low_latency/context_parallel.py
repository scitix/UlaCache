"""Compare all three complete CP rounds; retain acceptance and timing separately."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import math


def fixed_batch_window(before, after, batch, *, minimum_steps=200):
    if type(batch) is not int or batch < 1 or len(before) != 6 or len(after) != 6:
        raise ValueError("Invalid native decode moment geometry")
    if not all(type(v) in (int, float) and math.isfinite(v) for v in [*before, *after]):
        raise ValueError("Native decode moments must be finite numbers")
    delta = [b - a for a, b in zip(before, after)]
    steps, sum_batch, duration_us, sum_square, batch_time, tokens = delta
    if steps < minimum_steps or steps != int(steps) or min(delta) < 0:
        raise ValueError("Counter reset or insufficient complete native decode steps")
    if sum_batch != steps * batch or sum_square != steps * batch * batch:
        raise ValueError("Actual native batch changed inside the measurement window")
    if duration_us <= 0 or tokens <= 0 or batch_time != duration_us * batch:
        raise ValueError("Invalid native fixed-batch time or output counters")
    return dict(
        steps=int(steps),
        actual_batch=batch,
        batch_variance=0,
        duration_us=duration_us,
        generated_tokens=int(tokens),
        mean_step_ms=duration_us / steps / 1000,
        generated_tokens_per_step_per_request=tokens / steps / batch,
        effective_tpot_ms=batch * duration_us / tokens / 1000,
        decode_tokens_per_second=tokens * 1.0e6 / duration_us,
        native_moment_delta=delta,
    )


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


BATCHES = (1, 2, 4, 8, 16)
GEOMETRY = (
    "attention_dp",
    "attention_tp",
    "allocation_page_size",
    "spec_on",
    "speculative_steps",
    "input_tokens",
    "output_tokens",
    "selected_input_token_sha256",
)


def read_bound(path, expected=None):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(("Result checksum changed", str(path)))
    return json.loads(raw), dict(path=str(path), sha256=digest)


def serving_source_hashes(plan):
    root = Path(plan["service_out"]).parent
    return {
        role: hashlib.sha256((root / role / "SOURCE.json").read_bytes()).hexdigest()
        for role in ("library", "framework", "runtime")
    }


def read_arm(root, kind):
    plan, plan_binding = read_bound(root / "ACTUAL_PLAN.json")
    terminal, terminal_binding = read_bound(root / "COMPLETE.json")
    startup, startup_binding = read_bound(root / "STARTUP_ACCEPTANCE.json")
    accepted_kinds = ("ulacache", "ulacache") if kind == "ulacache" else (kind,)
    if plan["variant"] not in accepted_kinds or terminal.get("passed") is not True:
        raise ValueError("Require the complete native or UlaCache arm")
    policies = startup.get("policies", [])
    if len(policies) != 8 or {p["rank"] for p in policies} != set(range(8)):
        raise ValueError("Require all eight actual native startup policies")
    if any(
        not p.get("passed")
        or p.get("kind") not in accepted_kinds
        or not p.get("fp8_kv")
        or p.get("dcp_size") != 8
        or p.get("MTP") != [2, 1, 3]
        or p.get("max_running_requests") != 64
        or p.get("cgbs") != 64
        or p.get("simulated_acceptance")
        for p in policies
    ):
        raise ValueError("Native startup differs from the declared CP protocol")
    expected = {(r, b) for r in (1, 2, 3) for b in BATCHES}
    retained = {}
    for case in terminal["cases"]:
        key = (case["repeat"], case["requested_batch"])
        if (
            key not in expected
            or key in retained
            or not case.get("fixed_batch_qualified")
        ):
            raise ValueError("Missing, duplicate or unqualified CP round")
        retained[key] = case
    if retained.keys() != expected:
        raise ValueError("All five batches and all three complete rounds are required")
    cases = []
    for (repeat, batch), item in sorted(retained.items()):
        previous = item.get("retained_previous_epoch")
        path = (
            Path(previous["path"])
            if previous
            else root / f"round{repeat}" / f"B{batch:02d}" / "COMPLETE.json"
        )
        if previous:
            prior, _ = read_bound(Path(previous["plan_path"]), previous["plan_sha256"])
            if (
                any(prior.get(k) != plan.get(k) for k in GEOMETRY)
                or prior["variant"] not in accepted_kinds
            ):
                raise ValueError("Preempted-epoch adoption changed the protocol")
            if serving_source_hashes(prior) != serving_source_hashes(plan):
                raise ValueError(
                    "Preempted-epoch adoption crossed a serving-code change"
                )
        value, binding = read_bound(path, previous["sha256"] if previous else None)
        if not value.get("requests_completed") or not value.get(
            "fixed_batch_qualified"
        ):
            raise ValueError("Incomplete requests cannot enter an overhead mean")
        if (
            value["requested_batch"],
            value["input_tokens"],
            value["natural_output_tokens_per_request"],
        ) != (batch, 131072, 4096):
            raise ValueError("CP prompt, natural output budget or actual batch differs")
        window = value["fixed_window"]
        left = window["before"]["loads"]["loads"][0]["decode_moments"]
        right = window["after"]["loads"]["loads"][0]["decode_moments"]
        actual = fixed_batch_window(left, right, batch)
        if actual["duration_us"] < 10.0e6 or window["observed_wall_seconds"] < 10:
            raise ValueError(
                "Native and client measurement windows must both exceed10 seconds"
            )
        for name in (
            "mean_step_ms",
            "effective_tpot_ms",
            "generated_tokens_per_step_per_request",
        ):
            if actual[name] != window[name]:
                raise ValueError(
                    "Reported fixed-batch timing differs from native counters"
                )
        mtp = value["MTP_full_requests"]
        if mtp["output_tokens"] != batch * 4096 or mtp["spec_verify_ct"] <= 0:
            raise ValueError("Missing actual native generation/verification counters")
        if (
            mtp["spec_num_proposed_drafts"] != 2 * mtp["spec_verify_ct"]
            or not 0
            <= mtp["spec_num_correct_drafts"]
            <= mtp["spec_num_proposed_drafts"]
        ):
            raise ValueError(
                "Actual MTP counters differ from the declared native draft geometry"
            )
        cases.append(
            dict(
                repeat=repeat,
                batch=batch,
                binding=binding,
                adopted_previous_epoch=previous is not None,
                request_tpot_ms=value["full_request_TPOT_mean_ms"],
                request_ttft_seconds=value["full_request_TTFT_mean_seconds"],
                step_ms=actual["mean_step_ms"],
                fixed_tpot_ms=actual["effective_tpot_ms"],
                output_tokens=mtp["output_tokens"],
                verify_steps=mtp["spec_verify_ct"],
                accepted_drafts=mtp["spec_num_correct_drafts"],
                proposed_drafts=mtp["spec_num_proposed_drafts"],
            )
        )
    return dict(
        kind=kind,
        plan=plan,
        startup_policies=policies,
        cases=cases,
        bindings=[plan_binding, terminal_binding, startup_binding],
    )


def summarize_arm(arm, batch):
    values = [v for v in arm["cases"] if v["batch"] == batch]
    if len(values) != 3:
        raise ValueError("Three actual repeats are required for each batch")
    result = {
        name: statistics.mean(v[name] for v in values)
        for name in (
            "request_tpot_ms",
            "request_ttft_seconds",
            "step_ms",
            "fixed_tpot_ms",
        )
    }
    result["request_tpot_range_ms"] = [
        min(v["request_tpot_ms"] for v in values),
        max(v["request_tpot_ms"] for v in values),
    ]
    result["native_accept_length"] = sum(v["output_tokens"] for v in values) / sum(
        v["verify_steps"] for v in values
    )
    result["draft_accept_rate"] = sum(v["accepted_drafts"] for v in values) / sum(
        v["proposed_drafts"] for v in values
    )
    return result


def compare(native, ulacache):
    if any(native["plan"].get(k) != ulacache["plan"].get(k) for k in GEOMETRY):
        raise ValueError("The paired workload or parallel/MTP geometry differs")
    rows = []
    for batch in BATCHES:
        a, b = summarize_arm(native, batch), summarize_arm(ulacache, batch)
        overhead = {
            name: 100 * (b[name] / a[name] - 1)
            for name in ("request_tpot_ms", "step_ms", "fixed_tpot_ms")
        }
        rows.append(
            dict(
                batch=batch,
                native=a,
                ulacache=b,
                overhead_percent=overhead,
                all_timing_metrics_within5percent=all(
                    v <= 5 for v in overhead.values()
                ),
            )
        )
    mean_metrics = {
        name: statistics.mean(r["overhead_percent"][name] for r in rows)
        for name in ("request_tpot_ms", "step_ms", "fixed_tpot_ms")
    }
    mean_overhead = mean_metrics["request_tpot_ms"]
    return dict(
        batches=rows,
        mean_overhead_percent=mean_metrics,
        average_all_timing_metrics_within5percent=all(
            v <= 5 for v in mean_metrics.values()
        ),
        mean_request_tpot_overhead_percent=mean_overhead,
        average_overhead_within5percent=mean_overhead <= 5,
        acceptance_rule="Arithmetic mean across BS1/2/4/8/16 of three-round full-request TPOT overhead; threshold5percent",
        all_batches_within5percent=all(
            r["all_timing_metrics_within5percent"] for r in rows
        ),
        scope="Three full natural-generation repeats; raw values and all native counter proofs retained",
        native_bindings=native["bindings"],
        ulacache_bindings=ulacache["bindings"],
        native_cases=native["cases"],
        ulacache_cases=ulacache["cases"],
        full_request_mean_and_fixed_native_window_are_separate=True,
        equality_of_generated_token_ids_required=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--ulacache-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = compare(
        read_arm(args.native_results, "native"), read_arm(args.ulacache_results, "ulacache")
    )
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "COMPARISON.json", value)
    rows = []
    for row in value["batches"]:
        item = dict(batch=row["batch"])
        for arm in ("native", "ulacache"):
            item.update(
                {
                    arm + "_" + k: v
                    for k, v in row[arm].items()
                    if not isinstance(v, list)
                }
            )
        item.update(
            {"overhead_percent_" + k: v for k, v in row["overhead_percent"].items()}
        )
        rows.append(item)
    with (args.out / "COMPARISON.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            dict(
                output=str(args.out),
                mean_request_tpot_overhead_percent=value[
                    "mean_request_tpot_overhead_percent"
                ],
                average_overhead_within5percent=value[
                    "average_overhead_within5percent"
                ],
            )
        )
    )


if __name__ == "__main__":
    main()
