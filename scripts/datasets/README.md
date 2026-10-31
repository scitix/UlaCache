# Workload preparation

These scripts prepare workloads from public datasets using an external selection
manifest. Experimental selection manifests, routing files and recorded replay
traces are not included in this source release.

## Finding the Right Fit

```bash
python scripts/datasets/prepare_rightfit.py \
  --selection /path/to/rightfit_selection.json \
  --tokenizer /path/to/tokenizer.json \
  --cache-dir /path/to/dataset-cache \
  --out /path/to/rightfit-output
```

The script verifies the selected public records and tokenizer against the
provided manifest. Install the `artifact` extra first. `--records-root` can
point to a previously extracted record cache. The output directory must not
already exist.

## Multi-LexSum

```bash
python scripts/datasets/prepare_multilexsum.py \
  --selection /path/to/multilexsum_selection.json \
  --tokenizer /path/to/tokenizer-directory \
  --cache /path/to/dataset-cache \
  --out /path/to/multilexsum-output
```

Use a manifest matching the tokenizer and intended workload. The script
verifies source files and selected records before writing requests.

Generated datasets, KV banks and replay traces should stay outside the checkout.
