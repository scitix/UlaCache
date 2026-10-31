# GPU-only Index ownership ablation

Compare two UlaCache TP deployments with `start_ulacache.sh --index-mode replicated`
and `start_ulacache.sh --index-mode distributed`. Both keep MLA KV owner-striped,
disable CPU expansion, and retain MRQ/CGBS 64/64 and the same resident buffers.
Only Index storage and scoring ownership change. For SPEC ON, both use the same
71 GB pool budget, reserving space for the measured target and draft weights
within the paper's approximately 133 GB model-plus-cache budget per GPU.
Check the actual target/draft storage inventory on all eight ranks before
measurement, with shared allocations counted once and a 1 GB tolerance.
No CPU-derived token limit is imposed.

Snapshot restoration follows the same ownership: replicated Index restores
all Index records on every rank, while MLA KV remains owner-striped. Keep this
distinction when preparing a shared KV bank or reading startup inventories.

Fill a shared KV bank with `start_baseline.sh --phase fill` and
`scripts/fill_kv.py`, using a tokenized dataset of 128K-token requests. Then run
each variant on a separate machine, or stop the previous server before starting
the next one:

```bash
bash scripts/7_6_indexer/start_ulacache.sh --model /models/GLM-5.3-NVFP4 \
  --index-mode replicated --spec on --phase decode \
  --kv-dir /results/kv --log-dir /results/replicated

python scripts/decode.py --url http://localhost:30000 \
  --dataset /data/requests-128k.jsonl --attn-dp-size 1 \
  --receipts /results/replicated/restore --batch-sizes 1,2,4,8,16,32 \
  --max-local-bs 64 --output-tokens 4096 --rounds 3 \
  --out /results/replicated-decode
```

Repeat with `--index-mode distributed`, its own log/output directories, and the
same dataset, batch sizes and generation parameters. For capacity comparison:

```bash
python scripts/7_4_capacity/run.py --mode tp \
  --input replicated=/results/replicated/audit \
  --input distributed=/results/distributed/audit --out /results/index-capacity
```

Each `target.rankN.json` records actual Index ownership and allocated bytes in
`index_distribution`. Both deployments must report zero CPU KV capacity. Compare
their `usable_history_slots` and matching-batch `results.csv`; omit batch sizes
that exceed either deployment's measured capacity.

Use three formal rounds for each point. Decode timing excludes snapshot I/O
and the first and last 64 output tokens. Report each arm's measured capacity
alongside its matching-batch TPOT; a capacity difference must not be hidden by
comparing different batch sizes. Store per-round records and plots in your
selected output directory.

`start_baseline.sh` and `start_hisparse.sh` retain the complete reference launch
configurations. They are separate cache designs, not the Index-ownership
ablation pair. All commands accept direct parameters and require no cluster
coordinator or generated experiment plan.
