# Paper experiments

Each numbered directory contains three complete server launch scripts:
`start_baseline.sh`, `start_hisparse.sh`, and `start_ulacache.sh`.
Their command lines, memory limits, TP/DPA settings, speculation settings and
cache optimizations are written in those files. They do not load experiment
plans or call a cluster scheduler.

Write generated logs, KV snapshots, measurements and plots to an output
directory outside the checkout, such as `/results/` in the examples below.
The repository includes running scripts and configuration examples. Measurement
tables, plots, recorded replay traces and individual test runs are not included.

All launchers accept `--model`, `--draft-model`, `--mode tp|dpa`,
`--spec on|off`, `--port`, `--log-dir`, and `--kv-dir`. `--dry-run` prints the
resolved command. Additional native SGLang arguments follow `--`.
Defaults describe the GLM-5.3 NVFP4 / eight-H200 experiment. Set `CACHE_DIR`
to a shared compilation-cache directory when using multiple machines.
All launchers explicitly select the Python cache-tree backend used by the
recorded experiments, keeping prefix-cache behavior independent of an image's
Rust-extension defaults.
The throughput, low-latency and capacity launchers also accept `--platform b300` for the
paper's HY4 Preview FP8 configuration; see their dataset and capacity settings
in [throughput](7_2_throughput/README.md#hy4-preview-fp8-on-b300) and
[low latency](7_3_low_latency/README.md#hy4-preview-fp8-on-b300).

The paper's platform budgets are per GPU, in decimal GB. Count shared
allocations once when checking the actual target and draft inventories.

| Platform | Checkpoint | KV format | Model + GPU KV | CPU KV | UlaCache TP request limit |
| --- | --- | --- | ---: | ---: | ---: |
| 8 × H200 | GLM-5.3 NVFP4 | FP8 E4M3 | 133 GB | 128 GB | 64 |
| 8 × B300 | HY4 Preview FP8 | FP8 E4M3 | 272 GB | 192 GB | 192 |

The recorded B300 low-latency comparison uses a 270.5 GB GPU target instead
of the 272 GB throughput target. Preserve this experiment-specific setting
and the original request buffers; see the low-latency directory. HiCache CPU
backups and HiSparse GPU copies do not add unique active KV capacity.

| Directory | Experiment | Request or analysis command |
| --- | --- | --- |
| `7_1_online` | Multi-turn TTFT, TPOT and end-to-end latency | `7_1_online/run.py` |
| `7_2_throughput` | Output throughput | `fill_kv.py`, then `decode.py` |
| `7_3_low_latency` | Low-batch TPOT | `decode.py --batch-sizes 1,2,4,8 --rounds 3` |
| `7_4_capacity` | Unique GPU/CPU KV capacity | `7_4_capacity/run.py` |
| `7_5_scheduler` | GPU scheduler time and hit rate | `7_5_scheduler/run.py replay` and `preview` |
| `7_6_indexer` | GPU-only replicated/distributed Index | `decode.py` with 128K inputs and 4096 outputs |

The experiments use separate protocols; a smaller measurement batch does not
change the server's request limit, local buffers, or KV allocation.
In H200 DPA, the saved-KV offline launchers use two 8192-record prefill windows;
the online launcher uses two 1,048,576-record windows for complete histories.
The backend applies the declared window size to allocation, admission and
history reconstruction. These windows count toward the same GPU budget and
must not silently replace one another when changing experiment scripts.

| Experiment | Input and output | Measurement |
| --- | --- | --- |
| Throughput | Frozen Multi-LexSum selection; 2048 naturally generated output tokens | One formal round; common decode interval excludes the first and final 64 tokens |
| Low latency | The same KV bank; B300 replays recorded output IDs and, for SPEC ON, the recorded MTP trajectory | BS 1/2/4/8, three formal rounds per BS on one model load; no discarded warmup round |
| Online | Whole Finding the Right Fit conversations, 256K–512K sequence lengths; original per-turn output budgets | Sequential turns within each conversation; independent prefix per instance; latest 32 completed requests' mean TTFT above 120 s fails the point |
| Indexer | 128K input, 4096 output tokens; all KV and Index keys on GPU | Replicated and distributed Index ownership under the same GPU memory budget |
| Scheduler | Recorded GLM-5.3 NVFP4 DPA/SPEC ON access trace | GPU metadata time and hit rate; excludes KV transfer time |

Keep the source commit, resolved launch command, dataset identity, startup
memory inventory, and raw per-round records with every result. Compare only
matching model, topology, speculation and measurement settings. An incomplete
or preempted point is not a pass.

For a code-release regression check, repeat all low-latency points. For
throughput and online serving, first verify batch/user count 16 and each
method's highest passing paper point. Use that method's recorded limit rather
than another method's limit. Full curve reproduction uses the original grid;
these release checks do not replace it. Report code errors and preemptions
separately from the paper's admission or TTFT-limit failures.
The online experiment's [release-check grid](7_1_online/README.md) distinguishes
the highest plotted point from HiSparse DPA's 72-user source result. The
[throughput grid](7_2_throughput/README.md) lists each platform's original
endpoints. Use those experiment-specific grids and launchers rather than
transferring request limits, prefill windows or memory settings between figures.

## Fill KV, then measure decode

Run these commands from the repository root. The dataset is JSONL with a
unique `key` and tokenized `input_ids` per request. Dataset preparation lives
in `datasets/`.

First launch the baseline to produce a shared, native target/draft KV bank:

```bash
bash scripts/7_2_throughput/start_baseline.sh \
  --model /models/GLM-5.3-NVFP4 --mode dpa --spec on \
  --phase fill --kv-dir /results/kv --log-dir /results/fill-server
```

In another terminal:

```bash
python scripts/fill_kv.py --url http://localhost:30000 \
  --dataset /data/requests.jsonl --attn-dp-size 8 --batch-size 8 \
  --kv-dir /results/kv --out /results/fill
```

After filling, stop that server and launch the method being measured. The
same KV bank is used for all methods; separate machines can run in parallel.

For an existing bank, preserve the producer's model identity by exporting
`ULACACHE_BENCH_MODEL_FINGERPRINT` with its SHA256 identifier or its explicit
`model@<40-character revision>:weight-format:kv-format` identifier. Verify
that the bank was generated by the same checkpoint before using it. Fresh banks
default to the checkpoint's `config.json` SHA256; changing an identity does not
convert KV from another checkpoint.
Existing bank identities are checked before model loading, so a missing or
incorrect override fails before allocating the model and cache.
Offline launchers reuse the committed bank's metadata and skip repeated full
payload hashing. Set `UDCACHE_SNAPSHOT_VERIFY_PAYLOAD=1` to recheck payload
hashes. Neither choice includes bank loading in measured decode time.

```bash
bash scripts/7_2_throughput/start_ulacache.sh \
  --model /models/GLM-5.3-NVFP4 --mode dpa --spec on \
  --phase decode --kv-dir /results/kv --log-dir /results/ulacache-server

python scripts/decode.py --url http://localhost:30000 \
  --dataset /data/requests.jsonl --attn-dp-size 8 \
  --receipts /results/ulacache-server/restore --batch-sizes 16,237 \
  --kv-slots 3162112 --max-local-bs 32 --output-tokens 2048 \
  --out /results/ulacache-throughput
```

Choose batch sizes for the actual deployment. The client reads each rank's
actual logical capacity from `/v1/loads` before admitting requests. An optional
`--kv-slots` can impose a lower per-rank limit; it cannot exceed that allocation.
The same capacity is recorded in `audit/target.rankN.json`
(`pool_fields.logical_allocator_size`). The patched server publishes this as
`logical_token_capacity` in `/v1/loads`; HiSparse's physical GPU buffer size
does not limit its logical CPU-backed history. With TP,
use `--attn-dp-size 1` and the shared logical capacity. The client preserves
request order and uses round-robin routing with capacity fallback.

Decode timing uses the common interval during which the whole batch is
producing output, excluding KV restore, prefill, and the first/final 64 tokens.
Outputs are generated naturally with a fixed count. Each run saves raw request
records, per-point `metrics.json`, and `results.csv`. It stops on a failed point
and drains its requests without aborting the server.

For low-batch latency, use the matching `7_3_low_latency` launcher and
`--batch-sizes 1,2,4,8 --rounds 3`; B300 additionally requires its recorded
`--replay` file on both the server and client. For the Index experiment, use a dataset of
128K requests and `--output-tokens 4096`; both server configurations must have
CPU expansion disabled and differ only in Index ownership. A label alone does
not change the implementation.

## Online conversations

Start the selected `7_1_online/start_*.sh` server, then:

```bash
python scripts/7_1_online/run.py --url http://localhost:30000 \
  --traces /data/conversations.jsonl --users 64 --instances 128 \
  --attention-dp 8 --spec on --out /results/online
```

The input contains whole conversations with `initial_input_ids` and `turns`.
Each turn has `append_input_ids` and its original `output_tokens` budget.
Generated output enters the next turn's history. Each instance receives an
independent prefix namespace. Defaults stop new admission when the latest32
completed requests have mean TTFT above120seconds, then drain current requests.

## Capacity

```bash
python scripts/7_4_capacity/run.py --mode dpa \
  --input baseline=/results/baseline-server/audit \
  --input hisparse=/results/hisparse-server/audit \
  --input ulacache=/results/ulacache-server/audit --out /results/capacity
```

This reads actual startup inventories. Replicated buffers and HiCache backups
do not add unique logical history capacity.

## Scheduler preview

Use `7_5_scheduler/start_ulacache.sh --mode dpa --spec on --phase decode`
with `--trace-dir /results/trace`. Send a128-request,4096-output-token cohort
using `decode.py`, also passing `--trace-dir /results/trace`. The server skips
128 target forwards and records512 forwards,21 Index groups and48 query rows
per rank. Trace buffers are diagnostic allocations.

```bash
python scripts/7_5_scheduler/run.py replay --trace-dir /results/trace \
  --gpus 0,1,2,3,4,5,6,7 --out /results/scheduler
python scripts/7_5_scheduler/run.py preview \
  --comparison /results/scheduler/comparison.csv --out /results/preview
```

Scheduler time measures GPU metadata work, not model TPOT. Native per-request
HiSparse and the shared-pool adaptation are labeled separately. Preview output
never overwrites manuscript figures.

`7_3_low_latency/context_parallel.py` analyzes the separate paired CP regression;
it is not the GPU-only Index experiment.

When reusing a server allocation, drain its client and send `SIGTERM` to the
server's main PID. The native shutdown request lets each rank unregister its
host KV before exiting. The launchers set `SGLANG_SCHEDULER_EXIT_TIMEOUT_SECS=0`
so a slow release waits to finish; start the next model after the old workers
have exited and the GPUs are free.
