# Output throughput over a filled KV bank

Start the selected method using `start_baseline.sh`, `start_hisparse.sh`, or
`start_ulacache.sh`. Each file contains all server settings and accepts direct CLI
arguments; use `--help` or `--dry-run` to inspect them.

Run from the repository root, using the tokenized MultiLexSum dataset and an
existing native KV bank:

```bash
bash scripts/7_2_throughput/start_ulacache.sh \
  --model /models/GLM-5.3-NVFP4 --mode dpa --spec on --phase decode \
  --kv-dir /results/kv --log-dir /results/ulacache-server
```

In another terminal:

```bash
python scripts/decode.py --url http://localhost:30000 \
  --dataset /data/requests.jsonl --attn-dp-size 8 \
  --receipts /results/ulacache-server/restore --batch-sizes 16,237 \
  --max-local-bs 32 --output-tokens 2048 --rounds 1 \
  --out /results/ulacache-throughput
```

Run baseline and HiSparse with their own launcher and log/output directories.
Use each method's recorded paper endpoint below. For TP use `--mode tp`,
`--attn-dp-size 1` and the TP request bound. The client checks each rank's
actual capacity before admission.

The H200 protocol uses the paper's ordered 256 Multi-LexSum inputs (mean
98,124.7 tokens), 2048 natural output tokens and one formal round per point.
Snapshot restoration is excluded from the common decode interval. Record the
source commit and actual startup inventory with each result. The following
table lists the manuscript's endpoint batches.

| H200 configuration | Baseline | HiSparse | UlaCache |
| --- | ---: | ---: | ---: |
| TP, SPEC OFF | 14 | 25 | 64 |
| TP, SPEC ON | 13 | 25 | 64 |
| DPA, SPEC OFF | 71 | 197 | 240 |
| DPA, SPEC ON | 63 | 197 | 237 |

These are reference endpoints, not claims about a newly installed server. If
the current allocation cannot admit the paper endpoint, record that difference;
do not silently lower the batch and label it the same point. Likewise, the
H200 baseline TP endpoints are below 16: record the 16-request admission
failure and compare throughput at the original endpoint.

`results.csv` contains output tokens/s and decode latency for the common
steady batch interval. KV loading and prefill are excluded. See the
[KV-bank preparation commands](../README.md#fill-kv-then-measure-decode) before
the first run. No cluster coordinator is required.

## HY4 Preview FP8 on B300

The same three launchers accept `--platform b300`. Use the HY4 checkpoint
exported for the selected TP or DPA topology, and pass the independent MTP
checkpoint with `--draft-model` when speculation is enabled.

```bash
bash scripts/7_2_throughput/start_ulacache.sh --platform b300 \
  --model /models/Hy4-preview-FP8-Sharded-DPA8 \
  --draft-model /models/Hy4-preview-FP8-MTP \
  --mode dpa --spec on --phase decode \
  --kv-dir /results/hy4-multilexsum-kv --log-dir /results/hy4-ulacache

python scripts/decode.py --url http://localhost:30000 \
  --dataset /data/hy4-multilexsum/requests.jsonl --attn-dp-size 8 \
  --receipts /results/hy4-ulacache/restore --batch-sizes 16,460 \
  --routing-file /path/to/hy4_multilexsum_dpa460_routing.json \
  --max-local-bs 128 --output-tokens 2048 --rounds 1 \
  --out /results/hy4-ulacache-throughput
```

The paper selection contains 512 Multi-LexSum requests, tokenized with HY4:
mean input 100,165 tokens, context bound 163,840. The frozen requests SHA256 is
`e8e2e1b29a5571b3302ab247d919bc50e4d2d79317c3b398c8c3505b2ffc9648`.
Use this selection for comparisons with the published B300 curve. LongBench
256K is a separate workload.

Each B300 point uses one formal round and 2048 naturally generated output
tokens. Saved-KV restoration is outside the decode timing window. Include
runtime and CUDA Graph allocations when checking the GPU budget, and retain
the component inventory separately.

| Setting | TP | DPA |
| --- | --- | --- |
| UlaCache MRQ / Graph maximum | 192 / 192 | 1024 global / 128 local |
| Baseline MRQ / Graph, SPEC off | 192 / 192 | 1024 global / 128 local |
| Baseline MRQ / Graph, SPEC on | 328 / 320 | 1024 global / 128 local |
| HiSparse MRQ / Graph | 328 / 320 | 1024 global / 128 local |
| UlaCache history multiplier | 3 | Shared rank-local storage |
| HiSparse device buffer / TopK | 32768 / 2048 | 32768 / 2048 |

These are the source experiment's request and capture bounds; changing the
measurement batch does not shrink them. All methods use FP8 KV and native
FlashMLA KV attention. All three B300 launchers enable DeepGEMM PDL and retain
the source experiment's TRT-LLM MoE PDL limit of 8192 tokens.
The SPEC ON capture grids include the final October 27 supplemental runs:
TP HiSparse/UlaCache add batch 37; DPA HiSparse adds 25/34/37/43/49/63/69,
and DPA UlaCache also adds 58. These are ordinary native CUDA Graph capture sizes,
with the original MRQ and maximum capture batch retained.
SPEC uses EAGLE with two draft steps, topk 1 and three
verification tokens. UlaCache and HiSparse allocate approximately 192 GB of CPU
KV per GPU. GPU-only throughput uses native GPU KV. The intended total
model-plus-cache GPU budget is 272 GB/card, with a 1 GB comparison tolerance.
Check the actual startup inventory after changing the model export or code;
the launch constants do not establish that the new allocation matches it.

Published endpoint batches are TP 30/29 (baseline off/on), 37 (HiSparse) and
192 (UlaCache); DPA 199/195, 289 and 470/460 respectively. Recheck admission on
the running service before using an endpoint. DPA routing may need a balanced
assignment close to capacity; a failed greedy assignment alone does not show
that the KV pool is full. The supplied routing file preserves the paper's
balanced 460-request assignment and applies only to that batch. The client
checks its dataset SHA256, request order and current per-rank capacities before
sending requests; other batches keep their usual routing.
