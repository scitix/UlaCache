"""Fill a portable native KV bank; run a baseline server with --phase fill first.

Example: python fill_kv.py --url http://localhost:30000 --dataset requests.jsonl
         --kv-dir results/kv --out results/fill --attn-dp-size 8 --batch-size 8
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def idle(url, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = json.load(urlopen(url + "/v1/loads", timeout=10))
        rows = value.get("loads", [])
        if rows and all(
            not r.get("num_running_reqs", 1)
            and not r.get("num_waiting_reqs", 1)
            and not any(r.get("queues", {}).values())
            for r in rows
        ):
            return
        time.sleep(0.1)
    raise TimeoutError("Service still has active requests")


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    requests = [
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if line.strip()
    ]
    if not requests or any(
        not row.get("key") or not row.get("input_ids") for row in requests
    ):
        raise ValueError("Dataset rows require key and input_ids")
    if args.limit is not None:
        requests = requests[: args.limit]
    started = time.monotonic()

    def capture(item):
        index, row = item
        path = (
            args.kv_dir
            / hashlib.sha256(row["key"].encode()).hexdigest()
            / "SNAPSHOT.json"
        )
        if path.exists():
            receipt = json.loads(path.read_text())
            assert receipt["complete"] and receipt["key"] == row["key"]
            assert receipt["input_ids_sha256"] == row.get(
                "input_ids_sha256", receipt["input_ids_sha256"]
            )
            assert (path.parent / "target/COMMITTED.json").is_file()
            if receipt.get("draft") is not None:
                assert (path.parent / "draft/COMMITTED.json").is_file()
                assert (
                    hashlib.sha256(
                        (path.parent / "continuation.pt").read_bytes()
                    ).hexdigest()
                    == receipt["continuation_sha256"]
                )
            return receipt
        payload = dict(
            input_ids=row["input_ids"],
            stream=False,
            rid="prefix-" + row["key"],
            routed_dp_rank=index % args.attn_dp_size,
            sampling_params=dict(
                temperature=0,
                max_new_tokens=1,
                ignore_eos=True,
                custom_params=dict(udcache_snapshot_key=row["key"]),
            ),
        )
        query = Request(
            args.url + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(query, timeout=3600) as response:
            result = json.load(response)
        if result["meta_info"]["completion_tokens"] != 1 or result["meta_info"].get(
            "num_retractions", 0
        ):
            raise RuntimeError("Native prefix seed did not finish exactly once")
        receipt = json.loads(path.read_text())
        assert receipt["complete"] and receipt["input_ids_sha256"] == row.get(
            "input_ids_sha256", receipt["input_ids_sha256"]
        )
        assert receipt["first_output_token"] == result["output_ids"][0]
        save(
            args.out / (hashlib.sha256(row["key"].encode()).hexdigest() + ".json"),
            dict(key=row["key"], snapshot=str(path), response=result),
        )
        return receipt

    completed = []
    cohort_size = args.batch_size or args.attn_dp_size
    for begin in range(0, len(requests), cohort_size):
        idle(args.url)
        flush = Request(
            args.url + "/flush_cache",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(flush, timeout=60) as response:
            response.read()
        subset = list(enumerate(requests[begin : begin + cohort_size], start=begin))
        with ThreadPoolExecutor(max_workers=cohort_size) as executor:
            completed.extend(executor.map(capture, subset))
        save(
            args.out / "PROGRESS.json",
            dict(
                complete=False,
                saved=len(completed),
                total=len(requests),
                elapsed_seconds=time.monotonic() - started,
                recent_keys=[r["key"] for r in completed[-cohort_size:]],
            ),
        )
        print(json.dumps(dict(saved=len(completed), total=len(requests))), flush=True)
    idle(args.url)
    value = dict(
        complete=True,
        saved=len(completed),
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        snapshots=completed,
        seconds=time.monotonic() - started,
        source_manifest_sha256s=sorted(
            {row["source_manifest_sha256"] for row in completed}
        ),
        scope="Once-only native full prefill and seed; no formal decode timing",
    )
    save(args.out / "COMPLETE.json", value)
    save(args.kv_dir / "CATALOG.json", value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL rows with key and tokenized input_ids",
    )
    parser.add_argument(
        "--kv-dir",
        type=Path,
        required=True,
        help="Same KV directory used by the server in --phase fill",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--attn-dp-size", type=int, choices=(1, 8), default=8)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    main(parser.parse_args())
