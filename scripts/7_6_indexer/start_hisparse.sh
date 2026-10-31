#!/usr/bin/env bash
# GLM-5.3 NVFP4 / 8 x H200. All model and cache options are declared here.
set -euo pipefail
# Let native ShutdownReq release registered host KV before process exit.
export SGLANG_SCHEDULER_EXIT_TIMEOUT_SECS=0
# The prepared paper bank is committed; metadata checks stay enabled.
export UDCACHE_SNAPSHOT_VERIFY_PAYLOAD=${UDCACHE_SNAPSHOT_VERIFY_PAYLOAD:-0}
MODEL_PATH=""; DRAFT_MODEL_PATH=""; MODE=tp; SPEC=on; PHASE=serve
PORT=30000; HOST=0.0.0.0; LOG_DIR=results/server; KV_DIR=results/kv
DRY_RUN=0
while (($#)); do
  case "$1" in
    --model) MODEL_PATH=$2; shift 2;;
    --draft-model) DRAFT_MODEL_PATH=$2; shift 2;;
    --mode) MODE=$2; shift 2;;
    --spec) SPEC=$2; shift 2;;
    --phase) PHASE=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --host) HOST=$2; shift 2;;
    --log-dir) LOG_DIR=$2; shift 2;;
    --kv-dir) KV_DIR=$2; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --help) echo "Usage: $0 --model PATH [--draft-model PATH] [--mode tp|dpa] [--spec on|off] [--phase serve|fill|decode] [--kv-dir PATH] [--log-dir PATH] [--port N] [--dry-run] [-- EXTRA_SGLANG_ARGS]"; exit 0;;
    --) shift; break;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$MODEL_PATH" ]] || { echo '--model is required' >&2; exit 2; }
[[ "$PHASE" == serve || "$PHASE" == fill || "$PHASE" == decode ]] || exit 2
mkdir -p "$LOG_DIR" "$KV_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd); KV_DIR=$(cd "$KV_DIR" && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Match the Python cache-tree backend in the recorded paper deployments.
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
# Persistent compilation caches; no cluster address or queue coordinator.
CACHE_DIR=${CACHE_DIR:-${HOME}/.cache/ulacache}
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch" TRITON_CACHE_DIR="$CACHE_DIR/triton"
export SGLANG_JIT_CACHE_DIR="$CACHE_DIR/sglang-jit" CUDA_CACHE_PATH="$CACHE_DIR/cuda"
export SGLANG_DG_CACHE_DIR="$CACHE_DIR/deepgemm"
export FLASHINFER_WORKSPACE_BASE="$CACHE_DIR/flashinfer"
export UDCACHE_MATRIX_AUDIT_DIR="$LOG_DIR/audit"
METHOD=hisparse
args=(
  --served-model-name glm-5.3-nvfp4
  --trust-remote-code
  --tp-size 8
  --ep-size 1
  --pp-size 1
  --dp-size 1
  --attn-cp-size 1
  --moe-runner-backend marlin
  --dtype bfloat16
  --kv-cache-dtype fp8_e4m3
  --dsa-prefill-backend flashmla_kv
  --dsa-decode-backend flashmla_kv
  --dsa-paged-mqa-logits-backend deepgemm
  --cuda-graph-backend-decode full
  --cuda-graph-backend-prefill disabled
  --context-length 163840
  --chunked-prefill-size 8192
  --max-prefill-tokens 16384
  --mem-fraction-static 0.93
  --random-seed 0
  --stream-interval 1
  --prefill-decode-interval 0
  --disable-overlap-schedule
  --skip-server-warmup
  --incremental-streaming-output
  --moe-dense-tp-size 8
  --model-loader-extra-config '{"enable_multithread_load": false}'
  --weight-loader-disable-mmap
  --enable-metrics
  --page-size 64
  --fp4-gemm-backend marlin
  --quantization modelopt_mixed
  --disable-shared-experts-fusion
  --enable-hisparse
  --disable-radix-cache
  --watchdog-timeout 86400
)
case "$MODE:$SPEC" in
  tp:on)
    MAX_REQUESTS=64
    args+=(
    --max-running-requests 64
    --cuda-graph-max-bs-decode 64
    --attn-dp-size 1
    --max-total-tokens 1145984
    --speculative-algorithm EAGLE
    --speculative-num-steps 2
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 3
    --hisparse-config '{"top_k": 2048, "device_buffer_size": 16384, "host_to_device_ratio": 2.1823969619122083}'
    )
    ;;
  tp:off)
    MAX_REQUESTS=64
    args+=(
    --max-running-requests 64
    --cuda-graph-max-bs-decode 64
    --attn-dp-size 1
    --max-total-tokens 1292352
    --hisparse-config '{"top_k": 2048, "device_buffer_size": 19968, "host_to_device_ratio": 1.935225077997326}'
    )
    ;;
  dpa:on)
    MAX_REQUESTS=256
    args+=(
    --max-running-requests 256
    --cuda-graph-max-bs-decode 256
    --attn-dp-size 8
    --max-total-tokens 681024
    --speculative-algorithm EAGLE
    --speculative-num-steps 2
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 3
    --hisparse-config '{"top_k": 2048, "device_buffer_size": 19904, "host_to_device_ratio": 3.6723992106005077}'
    )
    ;;
  dpa:off)
    MAX_REQUESTS=256
    args+=(
    --max-running-requests 256
    --cuda-graph-max-bs-decode 256
    --attn-dp-size 8
    --max-total-tokens 831616
    --hisparse-config '{"top_k": 2048, "device_buffer_size": 25664, "host_to_device_ratio": 3.0073880252424194}'
    )
    ;;
  *) echo "Use --mode tp|dpa and --spec on|off" >&2; exit 2;;
esac
export SGLANG_DEEPGEMM_STANDARD_LAYOUT=auto
export SGLANG_DSA_FUSE_TOPK=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=false
export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
export SGLANG_OPT_USE_ONLINE_COMPRESS=false
export UDCACHE_CPU_CACHE_CAP_BYTES=128000000000
export UDCACHE_MATRIX_CACHE_KIND="hisparse"

# Optional saved-prefix IO. This JSON contains values, never another script/plan.
export ULACACHE_BENCH_PHASE="$PHASE" ULACACHE_BENCH_KV_DIR="$KV_DIR"
export ULACACHE_BENCH_RECEIPTS="$LOG_DIR/restore" ULACACHE_BENCH_SPEC="$SPEC"
export ULACACHE_BENCH_MODE="$MODE" ULACACHE_BENCH_MODEL="$MODEL_PATH"
export ULACACHE_BENCH_MAX_REQUESTS="$MAX_REQUESTS"
command=(python -m sglang.srt.ulacache.entry "${args[@]}"
  --model-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH" --host "$HOST" --port "$PORT"
  --skip-tokenizer-init --enable-metrics --watchdog-timeout 86400)
if [[ "$SPEC" == on ]]; then
  command+=(--speculative-draft-model-path "${DRAFT_MODEL_PATH:-$MODEL_PATH}")
fi
command+=("$@")
printf 'MODE=%s SPEC=%s PHASE=%s METHOD=%s\n' "$MODE" "$SPEC" "$PHASE" "$METHOD"
printf '%q ' "${command[@]}"; printf '\n'
if ((DRY_RUN)); then exit 0; fi
exec "${command[@]}"
