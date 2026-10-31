# GPU scheduler time and historical KV hit rate

Start the selected method using `start_baseline.sh`, `start_hisparse.sh`, or
`start_ulacache.sh`. Each file contains all server settings and accepts direct CLI
arguments; use `--help` or `--dry-run` to inspect them.

See [the experiment commands](../README.md) for the matching request and analysis
commands. No experiment-plan generation or cluster coordinator is required.

To rerun only missing linked-LRU comparison points against the same captured
trace, use a new output directory:

```bash
python scripts/7_5_scheduler/run.py replay --trace-dir /results/trace \
  --gpus 0,1,2,3 --batch-sizes 1,2,4,8 --linked-only \
  --out /results/linked-low-batch
```

This produces matched-state probe results without repeating the three-method
replay. Its completion record explicitly distinguishes a subset from the full
experiment matrix.

## GLM-5.3 NVFP4 / H200 protocol

The matrix contains eight batch sizes and four scheduler comparisons. Use
FP8 KV, 21 shared Index groups, page size 64, a 1,048,576-record shared GPU
pool and MTP with 2 steps / top-k 1 / 3 verification rows. Native HiSparse
retains its per-request 8,192-record buffer.

Capture 512 target forwards after 128 skipped forwards. Main-method timing
uses three replay rounds and the final 256 frames. Linked-list timing uses
three matched-state probes at frames 256, 384 and 511, with three resets per
probe. Its hit-rate curve comes from a separate CPU policy replay; the other
hit rates come from separate GPU observers. Times cover GPU metadata work
across all 21 groups, excluding KV payload movement, model execution, input
copies and DP communication.

Generate the paper-style Figure 11 preview from your measured comparison:

```bash
python scripts/7_5_scheduler/run.py preview \
  --comparison /results/scheduler/comparison.csv --out /results/scheduler-preview
```

The preview uses stacked panels, 234 × 222 pt, and the manuscript's palette,
markers, legend order and axis styling. Generated CSV, PDF and PNG files stay
in the selected output directory.
