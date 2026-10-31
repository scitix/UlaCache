"""Check natural generation and incremental KV through two running PD servers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import secrets
import time
import uuid


def save(path, value):
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


async def post(session, url, payload):
    async with session.post(url + "/generate", json=payload) as response:
        value = await response.json()
        if response.status != 200 or "error" in value:
            raise RuntimeError((response.status, value))
        return value


async def idle(session, urls, ranks):
    for url, expected in zip(urls, ranks):
        async with session.get(url + "/v1/loads") as response:
            if response.status != 200:
                return False
            rows = (await response.json()).get("loads", [])
        if len(rows) != expected or {row["dp_rank"] for row in rows} != set(
            range(expected)
        ):
            return False
        for row in rows:
            queued = row.get("disaggregation", {})
            if (
                row["num_running_reqs"]
                or row["num_waiting_reqs"]
                or any(row.get("queues", {}).values())
                or any(
                    value
                    for key, value in queued.items()
                    if key.endswith("_queue_reqs")
                )
            ):
                return False
    return True


async def wait_idle(session, args, seconds):
    deadline = time.monotonic() + seconds
    while True:
        try:
            if await idle(
                session,
                (args.prefill_url, args.decode_url),
                (args.prefill_dp, args.decode_dp),
            ):
                return
        except (OSError, asyncio.TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("PD services did not become idle; no server signal sent")
        await asyncio.sleep(0.5)


def audit_files(args):
    return [
        path
        for root in (args.send_audit, args.receive_audit)
        for path in root.glob("direct-*.rank*.jsonl")
    ]


def read_audits(args, offsets, rooms):
    rows = []
    for path in audit_files(args):
        with path.open("rb") as stream:
            stream.seek(offsets.get(path, 0))
            for line in stream:
                value = json.loads(line)
                if value.get("room") in rooms:
                    rows.append(
                        dict(
                            value,
                            kind=path.name.split(".rank", 1)[0],
                            rank=int(path.stem.rsplit("rank", 1)[1]),
                        )
                    )
    return rows


def validate_audits(records, audits, bytes_per_token):
    verified = []
    for record in records:
        room, rank = record["room"], record["decode_rank"]
        receive = [
            r
            for r in audits
            if r["kind"] == "direct-receive" and r["room"] == room and r["rank"] == rank
        ]
        send = [r for r in audits if r["kind"] == "direct-send" and r["room"] == room]
        if len(receive) != 1 or not send:
            raise LookupError("Missing native completion evidence for a PD request")
        rx = receive[0]
        if (
            not rx["transport_quiesced"]
            or rx["payload_staging_bytes"] != 0
            or rx["prefix_records"] + rx["received_records"] != record["input_tokens"]
            or sum(s["records"] for s in send) != rx["received_records"]
            or any(s["status"] != 0 for s in send)
        ):
            raise ValueError("Native PD byte counts or retirement evidence disagree")
        transferred = sum(s["bytes"] for s in send)
        if (
            bytes_per_token is not None
            and transferred != bytes_per_token * rx["received_records"]
        ):
            raise ValueError(
                "Transferred KV bytes disagree with the declared native format"
            )
        if record["turn"] and not 0 < rx["received_records"] < record["input_tokens"]:
            raise ValueError("The follow-up request did not use incremental transfer")
        verified.append(
            dict(
                rank=rank,
                turn=record["turn"],
                room=room,
                input_tokens=record["input_tokens"],
                prefix_tokens=rx["prefix_records"],
                received_tokens=rx["received_records"],
                kv_bytes=transferred,
            )
        )
    return verified


async def main(args):
    import aiohttp

    args.out.mkdir(parents=True, exist_ok=False)
    requests = []
    with args.dataset.open() as stream:
        for line in stream:
            row = json.loads(line)
            ids = row["input_ids"]
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("Dataset must contain nonempty tokenized input_ids")
            requests.append(ids)
            if len(requests) == args.prefill_dp:
                break
    if len(requests) != args.prefill_dp:
        raise ValueError("Provide at least one dataset request per prefill DP rank")
    save(
        args.out / "OPTIONS.json",
        dict(
            vars(args),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input_lengths=list(map(len, requests)),
        ),
    )
    offsets = {path: path.stat().st_size for path in audit_files(args)}
    namespace, records, owned = uuid.uuid4().hex, [], set()
    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await wait_idle(session, args, args.timeout)

        async def user(rank, initial):
            history = list(initial)
            for turn in range(args.turns):
                room = (
                    secrets.randbelow((2**63 - 1 - rank) // args.prefill_dp) + 1
                ) * args.prefill_dp + rank
                rid = f"pd-check-{namespace}-{rank}-{turn}"
                owned.add(rid)
                base = dict(
                    input_ids=history,
                    rid=rid,
                    routed_dp_rank=rank,
                    extra_key=f"pd-check-{namespace}-{rank}",
                    bootstrap_host=args.bootstrap_host,
                    bootstrap_port=args.bootstrap_port,
                    bootstrap_room=room,
                    sampling_params=dict(
                        temperature=0,
                        max_new_tokens=args.output_tokens,
                        ignore_eos=True,
                    ),
                )
                started = time.monotonic()
                prefill = asyncio.create_task(
                    post(session, args.prefill_url, dict(base, stream=False))
                )
                output, final, first = [], None, None
                try:
                    payload = dict(
                        base,
                        stream=True,
                        routed_dp_rank=rank % args.decode_dp,
                        disagg_prefill_dp_rank=rank,
                    )
                    async with session.post(
                        args.decode_url + "/generate", json=payload
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError((response.status, await response.text()))
                        async for line in response.content:
                            if (
                                not line.startswith(b"data:")
                                or line[5:].strip() == b"[DONE]"
                            ):
                                continue
                            final = json.loads(line[5:])
                            if "error" in final:
                                raise RuntimeError(final)
                            tokens = final.get("output_ids", [])
                            if tokens and first is None:
                                first = time.monotonic() - started
                            output.extend(tokens)
                            if final["meta_info"]["completion_tokens"] != len(output):
                                raise ValueError(
                                    "Server did not send incremental token chunks"
                                )
                    upstream = await prefill
                except BaseException:
                    prefill.cancel()
                    await asyncio.gather(prefill, return_exceptions=True)
                    raise
                if (
                    final is None
                    or len(output) != args.output_tokens
                    or len(upstream["output_ids"]) != 1
                ):
                    raise ValueError(
                        "PD did not finish the requested natural output count"
                    )
                meta = final["meta_info"]
                counters = {
                    key: int(meta.get(key, 0))
                    for key in (
                        "spec_verify_ct",
                        "spec_num_proposed_drafts",
                        "spec_num_correct_drafts",
                    )
                }
                if args.spec == "on" and not (
                    counters["spec_verify_ct"] > 0
                    and counters["spec_num_proposed_drafts"]
                    == 2 * counters["spec_verify_ct"]
                    and 0
                    < counters["spec_num_correct_drafts"]
                    <= counters["spec_num_proposed_drafts"]
                ):
                    raise ValueError(
                        "Native MTP counters did not establish natural draft acceptance"
                    )
                record = dict(
                    rank=rank,
                    decode_rank=rank % args.decode_dp,
                    turn=turn,
                    room=room,
                    input_tokens=len(history),
                    output_tokens=len(output),
                    ttft_seconds=first,
                    total_seconds=time.monotonic() - started,
                    prefill_meta=upstream["meta_info"],
                    decode_meta=meta,
                    **counters,
                )
                save(
                    args.out / f"rank{rank}-turn{turn}.json",
                    dict(record, output_ids=output),
                )
                records.append(record)
                print(
                    json.dumps(
                        dict(
                            rank=rank,
                            turn=turn,
                            completed=len(records),
                            total=args.prefill_dp * args.turns,
                        )
                    ),
                    flush=True,
                )
                history = history + output + initial[-args.append_tokens :]
                owned.remove(rid)

        tasks = [
            asyncio.create_task(user(rank, ids)) for rank, ids in enumerate(requests)
        ]
        try:
            await asyncio.gather(*tasks)
            await wait_idle(session, args, 120)
        except BaseException as error:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            aborts = []
            for rid in sorted(owned):
                for url in (args.prefill_url, args.decode_url):
                    try:
                        async with session.post(
                            url + "/abort_request", json=dict(rid=rid, abort_all=False)
                        ) as response:
                            aborts.append(
                                dict(rid=rid, url=url, status=response.status)
                            )
                    except Exception as abort_error:
                        aborts.append(dict(rid=rid, url=url, error=repr(abort_error)))
            save(
                args.out / "HOLD.json",
                dict(
                    error=repr(error),
                    completed=len(records),
                    aborts=aborts,
                    server_signals_sent=False,
                ),
            )
            raise
    deadline = time.monotonic() + 30
    while True:
        audits = read_audits(args, offsets, {record["room"] for record in records})
        try:
            verified = validate_audits(records, audits, args.kv_bytes_per_token)
            break
        except LookupError as error:
            if time.monotonic() >= deadline:
                save(args.out / "AUDITS.json", audits)
                save(
                    args.out / "HOLD.json",
                    dict(error=repr(error), server_signals_sent=False),
                )
                raise
            await asyncio.sleep(0.5)
    save(args.out / "AUDITS.json", audits)
    proposed = sum(r["spec_num_proposed_drafts"] for r in records)
    accepted = sum(r["spec_num_correct_drafts"] for r in records)
    verifies = sum(r["spec_verify_ct"] for r in records)
    save(
        args.out / "COMPLETE.json",
        dict(
            passed=True,
            requests=len(records),
            incremental_transfer_verified=True,
            payload_staging_bytes=0,
            rows=verified,
            draft_accept_rate=accepted / proposed if proposed else None,
            mean_accept_length=sum(r["output_tokens"] for r in records) / verifies
            if verifies
            else None,
            forced_output_ids=False,
            transfer_timing_benchmark=False,
            byte_count_scope="target and draft KV; Index uses native state transfer",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--bootstrap-host", required=True)
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--prefill-dp", type=int, default=8)
    parser.add_argument("--decode-dp", type=int, default=8)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--send-audit", type=Path, required=True)
    parser.add_argument("--receive-audit", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--append-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--kv-bytes-per-token", type=int, default=None)
    parser.add_argument("--spec", choices=("on", "off"), default="on")
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.prefill_dp, args.decode_dp, args.append_tokens, args.output_tokens) < 1
        or args.turns < 2
    ):
        parser.error("Use positive capacities and at least two turns")
    asyncio.run(main(args))
