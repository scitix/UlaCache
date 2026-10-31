# Unique GPU/CPU KV capacity

Start the selected method using `start_baseline.sh`, `start_hisparse.sh`, or
`start_ulacache.sh`. Each file contains all server settings and accepts direct CLI
arguments; use `--help` or `--dry-run` to inspect them.

After each server reports ready, retain its `audit/target.rank*.json` files.
Capacity comes from the actual allocated pools; this step sends no requests:

```bash
python scripts/7_4_capacity/run.py --mode dpa \
  --input baseline=/results/baseline-server/audit \
  --input hisparse=/results/hisparse-server/audit \
  --input ulacache=/results/ulacache-server/audit --out /results/capacity
```

Use `--mode tp` for TP inventories. The output includes `capacity.csv` and
`capacity.json`, with logical history capacity and unique GPU/CPU record slots.
HiCache backups and HiSparse hot buffers do not add unique history capacity.
Keep model, speculation, memory budget and request/Graph geometry in the
comparison record; changing them changes the available KV capacity.
These launchers use the throughput experiment's allocation, including its
separate H200/B300 settings. Pass `--platform b300` and the matching HY4
TP or DPA checkpoint for B300; its GPU budget is 272 GB, not the 270.5 GB
low-latency allocation.

The manuscript's H200 reference capacities use the 133 GB GPU / 128 GB CPU
budget. Compare these with the actual inventories from your deployment:

| Mode | SPEC | Baseline | HiSparse | UlaCache |
| --- | --- | ---: | ---: | ---: |
| TP8 | ON | 1,302,720 | 2,500,992 | 23,152,128 |
| DPA8 | ON | 6,876,160 | 20,007,936 | 25,296,896 |
| TP8 | OFF | 1,371,392 | 2,500,992 | 26,159,104 |
| DPA8 | OFF | 7,415,296 | 20,007,936 | 26,088,960 |

Values are unique history token slots across the eight-GPU deployment.
The report excludes native reserved pages; its GPU and CPU components sum
to the allocator's usable capacity. It does not add backup or staging copies.
