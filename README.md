# UlaCache

Source repository: https://github.com/scitix/UlaCache

UlaCache pools GPU and CPU memory for sparse-attention KV caches. It supports
prefetching, distributed Index storage, and prefix-cache reuse through SGLang,
with TP, DPA, and context-parallel execution on NVIDIA H200 and B300 GPUs.

## Installation

Use Linux with NVIDIA GPUs and the public SGLang v0.5.22 image:

```bash
docker run --gpus all --ipc=host --rm -it \
  -v "$PWD":/workspace/ulacache -w /workspace/ulacache \
  lmsysorg/sglang:v0.5.22 bash
```

Inside the container:

```bash
python scripts/validation/install.py --verify-cuda
```

The installer applies `patches/sglang.patch`, installs UlaCache, and verifies
installed imports and CUDA availability. The patch targets SGLang commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`.
Use `--sglang-root /path/to/sglang` to select a different checkout of that commit.

## Usage

Choose a model configuration from [scripts/configs/models](scripts/configs/models)
and adjust its memory and request limits for your deployment:

```bash
export ULACACHE_CONFIGURATION=/path/to/configuration.json
python -m sglang.srt.ulacache.entry \
  --model-path /path/to/model --tp-size 8 --kv-cache-dtype fp8_e4m3
```

The cache API follows `init`, `alloc`, `write`, `access`, and `free`.
SGLang retains responsibility for request scheduling, prefix matching,
attention, and sampling.

## Experiments

[scripts/README.md](scripts/README.md) describes the launch commands and
measurement protocols. Each numbered directory provides separate launchers
for UlaCache, HiSparse, and the baseline:

| Directory | Experiment |
| --- | --- |
| `scripts/7_1_online` | Multi-turn serving |
| `scripts/7_2_throughput` | Decode throughput |
| `scripts/7_3_low_latency` | Low-batch latency and context parallelism |
| `scripts/7_4_capacity` | KV capacity |
| `scripts/7_5_scheduler` | Scheduler ablation |
| `scripts/7_6_indexer` | Distributed Index ablation |

Workload manifests, replay traces, and experiment results are distributed
separately. See [dataset preparation](scripts/datasets/README.md) for input formats.

## Source layout

```text
ulacache/
  cache/       Allocation, residency, and cache lifecycle
  storage/     KV layouts and memory pools
  eviction/    Replacement policies
  transfer/    Prefetch and GPU/CPU transfers
  indexer/     Distributed Index storage and scoring
  flashag/     GPU communication
  csrc/        C++ and CUDA kernels
scripts/       Launchers, benchmarks, and validation
patches/       Framework integration patches
```

## Validation

```bash
python -m pip install -e '.[test,artifact]'
python -m pytest -q
python scripts/validation/check.py
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party attribution.
