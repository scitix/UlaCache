# Online conversations: TTFT, TPOT and end-to-end latency

Start the selected method using `start_baseline.sh`, `start_hisparse.sh`, or
`start_ulacache.sh`. Each file contains all server settings and accepts direct CLI
arguments; use `--help` or `--dry-run` to inspect them.

These launchers use a 524,288-token context for the complete long conversations.
Their DPA/SPEC ON defaults follow the paper's online deployment, independently
of the offline decode scripts:

| Method | Global MRQ | Global CGBS | GPU memory fraction |
| --- | ---: | ---: | ---: |
| GPU + HiCache | 512 | 512 | 0.93 |
| HiSparse | 128 | 512 | 0.94 |
| UlaCache | 512 | 512 | 0.94 |

HiSparse uses a 32,768-token device buffer per request. UlaCache declares 609,152
GPU records and 2,501,504 CPU records per rank. Validate the actual startup
inventories before comparing performance on another model or GPU.

TP/SPEC ON uses MRQ/CGBS 64/64, two 1,048,576-record prefill windows,
2,500,416 CPU records per owner, distributed target and draft Index,
and the paper's circular local replacement policy. The logical token cap is
23,128,000; native allocation rounds it to whole 512-token owner superpages.
The prefill windows are separate from the decode residency buffer and are
included in the GPU budget. Offline decode launchers have their own settings;
do not copy their 8192-record prefill window into this experiment.

For a release check, use the paper's 16-user point and each method's recorded
highest passing point. These user counts do not change MRQ, CGBS or buffer sizes:

| Configuration | Release-check users | Highest plotted users |
| --- | --- | ---: |
| GPU + HiCache, TP | 8, then 16 to check the recorded failure | 8 |
| HiSparse, TP | 8, then 16 to check the recorded failure | 8 |
| UlaCache, TP | 16, 64 | 64 |
| GPU + HiCache, DPA | 16, 32 | 32 |
| HiSparse, DPA | 16, 64, 72 | 64 |
| UlaCache, DPA | 16, 88 | 88 |

The manuscript cites 72 users as HiSparse DPA's highest passing source point;
72 is absent from its selected plotting grid. Check it separately rather than
substituting the plotted 64-user point. Stop an upward sweep after a TTFT-limit
failure. A client or engine error requires diagnosis and does not validate that
failure boundary.

Run from the repository root:

```bash
bash scripts/7_1_online/start_ulacache.sh \
  --model /models/GLM-5.3-NVFP4 --mode dpa --spec on \
  --log-dir /results/ulacache-online-server
```

In another terminal:

```bash
python scripts/7_1_online/run.py --url http://localhost:30000 \
  --traces /data/conversations.jsonl \
  --dataset-sha256 f00e851f42ea7c4f55edcb9fd52bc81883dbdc052ae2ac10549d797a5beb204d \
  --users 16 --instances 32 \
  --attention-dp 8 --spec on --failure-window 32 --ttft-limit 120 \
  --out /results/ulacache-online-16
```

Run each method with its own launcher and output directory. Use the same whole
conversations and output-length budgets; generated tokens enter the next turn's
history. Each conversation instance has an independent prefix namespace.
User slot `u` runs instances `u, u + users, u + 2 * users, ...` in that order
and stays on attention-DP rank `u % attention_dp`. This preserves the paper's
conversation assignment even when some users finish earlier than others.
Outputs include per-request records, progress, and TTFT/TPOT/end-to-end summaries.
Use `max(32, 2 * users)` conversation instances for these paper release checks.
The SHA256 above identifies the prepared 42-conversation paper trace, not the
upstream dataset archive. A different selection must be reported separately.

For a second load point, wait for the previous client to drain before starting
another client. `FAILED_TTFT120.json` marks the rolling TTFT limit; `HOLD.json`
records a client or service error and does not establish a capacity limit.
See [dataset preparation and protocol](../README.md#online-conversations).
