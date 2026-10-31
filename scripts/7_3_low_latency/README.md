# Low-batch decode latency and CP regression

Start the selected method using `start_baseline.sh`, `start_hisparse.sh`, or
`start_ulacache.sh`. Each file contains all server settings and accepts direct CLI
arguments; use `--help` or `--dry-run` to inspect them.

Run from the repository root with a prepared native KV bank:

```bash
bash scripts/7_3_low_latency/start_ulacache.sh \
  --model /models/GLM-5.3-NVFP4 --mode tp --spec on --phase decode \
  --kv-dir /results/kv --log-dir /results/ulacache-low-server
```

In another terminal:

```bash
python scripts/decode.py --url http://localhost:30000 \
  --dataset /data/requests.jsonl --attn-dp-size 1 \
  --receipts /results/ulacache-low-server/restore --batch-sizes 1,2,4,8 \
  --max-local-bs 64 --output-tokens 2048 --rounds 3 \
  --out /results/ulacache-low
```

Use the same input bank, batch grid, speculation setting and round count for
all three launchers. For DPA change the server to `--mode dpa` and the client
to `--attn-dp-size 8 --max-local-bs 32`. Each server is loaded once; all three
formal rounds are retained. There is no discarded warmup round.

The manuscript's H200 low-batch table contains one formal round per point;
the B300 table contains three. The commands above use three rounds for the
release regression check. Keep those repetitions separate from the original
paper records when comparing means.

Compare mean TPOT at the same batch: `100 * (method / baseline - 1)`.
Raw request records, per-round `metrics.json` and `results.csv` are saved in
the output directory. The separate `context_parallel.py` command analyzes a
native/UlaCache CP pair; it is not part of this TP/DPA sweep.

See [KV-bank preparation](../README.md#fill-kv-then-measure-decode) for the
first run. No cluster coordinator is required.

## HY4 Preview FP8 on B300

The recorded replay files are distributed separately. Supply their paths in
the commands below, or omit `--replay` for naturally generated outputs.

Use `--platform b300` and the HY4 checkpoint exported for the selected topology.
The B300 SPEC ON comparison fixes the full baseline output and MTP acceptance
trajectory, including the same synthetic rejected proposals for all methods.
It measures cache overhead with real target, draft and sampler execution; the
forced acceptance rate is not a natural model-accuracy result.

```bash
bash scripts/7_3_low_latency/start_ulacache.sh --platform b300 \
  --model /models/Hy4-preview-FP8-Sharded-DPA8 \
  --draft-model /models/Hy4-preview-FP8-MTP --mode dpa --spec on \
  --phase decode --kv-dir /results/hy4-kv --log-dir /results/hy4-low \
  --replay /path/to/hy4_lowbs_replay.json.gz

python scripts/decode.py --url http://localhost:30000 \
  --dataset /data/hy4-multilexsum/requests.jsonl --attn-dp-size 8 \
  --receipts /results/hy4-low/restore --batch-sizes 1,2,4,8 \
  --max-local-bs 128 --output-tokens 2048 --rounds 3 \
  --replay /path/to/hy4_lowbs_replay.json.gz --out /results/hy4-low-decode
```

The three launchers retain TP UlaCache R192/r3, TP baseline/HiSparse MRQ328,
and DPA global MRQ1024/local128. Low-BS Graph capture uses 1/2/4/8 without
shrinking those cache capacities. The historical low-BS memory target is
270.5 GB/card, distinct from the nominal 272 GB throughput configuration;
verify actual startup inventories within the paper's 1 GB tolerance.
The B300 offline baseline disables the prefix cache, as in the recorded
launch commands. This setting belongs to saved-KV decode experiments; the
online conversation baseline keeps its HiCache and prefix cache enabled.
The client checks exact output IDs, native speculation counters, and complete
draft/sampler execution receipts for every measured cohort.

For SPEC OFF, use `--spec off` on the launcher and pass
`--replay /path/to/hy4_lowbs_plain_replay.json.gz` to both commands.
This fixes the recorded native output IDs while retaining every target forward
and native sampler call; there is no speculative acceptance override.
