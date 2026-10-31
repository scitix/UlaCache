#!/usr/bin/env bash
# Paper GLM-5.3 NVFP4 / H200 or HY4 Preview FP8 / B300. All model and cache options are declared here.
set -euo pipefail
# Let native ShutdownReq release registered host KV before process exit.
export SGLANG_SCHEDULER_EXIT_TIMEOUT_SECS=0
# The prepared paper bank is committed; metadata checks stay enabled.
export UDCACHE_SNAPSHOT_VERIFY_PAYLOAD=${UDCACHE_SNAPSHOT_VERIFY_PAYLOAD:-0}
MODEL_PATH=""; DRAFT_MODEL_PATH=""; MODE=tp; SPEC=on; PHASE=serve; PLATFORM=h200; REPLAY_TRACE=""
PORT=30000; HOST=0.0.0.0; LOG_DIR=results/server; KV_DIR=results/kv
DRY_RUN=0
while (($#)); do
  case "$1" in
    --replay) REPLAY_TRACE=$2; shift 2;;
    --platform) PLATFORM=$2; shift 2;;
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
    --help) echo "Usage: $0 --model PATH [--platform h200|b300] [--replay FILE] [--draft-model PATH] [--mode tp|dpa] [--spec on|off] [--phase serve|fill|decode] [--kv-dir PATH] [--log-dir PATH] [--port N] [--dry-run] [-- EXTRA_SGLANG_ARGS]"; exit 0;;
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
METHOD=ulacache
case "$PLATFORM" in
h200)
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
    --max-total-tokens 23152576
    --speculative-algorithm EAGLE
    --speculative-num-steps 2
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 3
    --radix-eviction-policy lru
    )
    ;;
  tp:off)
    MAX_REQUESTS=64
    args+=(
    --max-running-requests 64
    --cuda-graph-max-bs-decode 64
    --attn-dp-size 1
    --max-total-tokens 26159552
    --radix-eviction-policy lru
    )
    ;;
  dpa:on)
    MAX_REQUESTS=256
    args+=(
    --max-running-requests 256
    --cuda-graph-max-bs-decode 256
    --attn-dp-size 8
    --max-total-tokens 3162112
    --speculative-algorithm EAGLE
    --speculative-num-steps 2
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 3
    )
    export UDCACHE_DPA_GPU_SLOTS=660672
    export UDCACHE_DPA_CPU_SLOTS=2501504
    export UDCACHE_DPA_CACHE_BUDGET_BYTES=47288817688
    ;;
  dpa:off)
    MAX_REQUESTS=256
    args+=(
    --max-running-requests 256
    --cuda-graph-max-bs-decode 256
    --attn-dp-size 8
    --max-total-tokens 3261120
    )
    export UDCACHE_DPA_GPU_SLOTS=759680
    export UDCACHE_DPA_CPU_SLOTS=2501504
    export UDCACHE_DPA_CACHE_BUDGET_BYTES=50245682048
    ;;
  *) echo "Use --mode tp|dpa and --spec on|off" >&2; exit 2;;
esac
export SGLANG_DEEPGEMM_STANDARD_LAYOUT=auto
export SGLANG_DSA_FUSE_TOPK=1
export SGLANG_FLOWKV_CACHE_CAPACITY_MULTIPLIER=3.0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=false
export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
export SGLANG_OPT_USE_ONLINE_COMPRESS=false
export UDCACHE_CPU_CACHE_CAP_BYTES=128000000000
export UDCACHE_DPA_ASYNC_FOLLOWERS=1
export UDCACHE_DPA_EXCHANGE_BLOCKS=64
export UDCACHE_DPA_EXCHANGE_THREADS=256
export UDCACHE_DPA_FOLLOWER_COPY_MODE=layer
export UDCACHE_DPA_FOLLOWER_EXCHANGE_BLOCKS=64
export UDCACHE_DPA_FOLLOWER_EXCHANGE_THREADS=256
export UDCACHE_DPA_PARALLEL_PREPARE=1
export UDCACHE_DPA_PARALLEL_VICTIMS=1
export UDCACHE_DPA_PREFILL_EXCHANGE_BLOCKS=64
export UDCACHE_DPA_PREFILL_EXCHANGE_THREADS=256
export UDCACHE_DPA_PREFILL_UNION_ROWS=8192
export UDCACHE_INDEX_COMPACT_SCORES=1
export UDCACHE_PLAIN_PREFILL_WINDOW_ROWS=8192
export UDCACHE_TP_HOST_EXCHANGE_BLOCKS=64
export UDCACHE_TP_HOST_EXCHANGE_THREADS=256
export UDCACHE_TP_PARALLEL_FINISH=1
export UDCACHE_TP_PARALLEL_VICTIMS=1
export UDCACHE_MATRIX_CACHE_KIND="${MODE}_udcache"
export ULACACHE_CONFIGURATION="$LOG_DIR/ulacache_config.json"
if [[ "$MODE" == tp ]]; then
  cat > "$ULACACHE_CONFIGURATION" <<'JSON'
{"schema_version":1,"framework_release":"v0.5.22","adapter":"native_tp","options":{"native_kv_format":"scaled656","cpu_offload":true,"cpu_slots":2500416,"prefill_window_rows":8192,"prefetch":{"enabled":true,"completion":"layer"},"host_exchange_blocks":64,"host_exchange_threads":256,"local_eviction_policy":"age_lru","cache_capacity_multiplier":3.0}}
JSON
else
  cat > "$ULACACHE_CONFIGURATION" <<JSON
{"schema_version":1,"adapter":"native_dpa","capacity":{"device_token_capacity_per_owner":${UDCACHE_DPA_GPU_SLOTS},"host_token_capacity_per_owner":2501504},"options":{"device_cache_budget_bytes":${UDCACHE_DPA_CACHE_BUDGET_BYTES},"host_cache_budget_bytes":128000000000,"parallel_prepare":true,"parallel_victims":true,"asynchronous_followers":true}}
JSON
fi

  ;;
b300)
  if [[ "$PHASE" == "decode" && -z "$REPLAY_TRACE" ]]; then
    echo "B300 paper low-BS decoding requires --replay FILE" >&2; exit 2
  fi
  # HY4 Preview FP8 / 8 x B300, Multi-LexSum, paper low-BS experiment.
  export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
  export SGLANG_DEEPGEMM_PDL=1 SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS=8192
  args=(
    --served-model-name hy4-preview-fp8 --trust-remote-code
    --tp-size 8 --ep-size 1 --pp-size 1 --dp-size 1 --attn-cp-size 1
    --moe-a2a-backend none --dtype bfloat16 --quantization mxfp8
    --moe-runner-backend flashinfer_trtllm --kv-cache-dtype fp8_e4m3
    --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
    --page-size 64 --context-length 163840
    --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
    --chunked-prefill-size 2048 --max-prefill-tokens 2048
    --mem-fraction-static 0.98 --random-seed 0 --stream-interval 1
    --disable-overlap-schedule --skip-server-warmup
    --incremental-streaming-output --enable-metrics --load-format sharded_state
    --disable-flashinfer-autotune --watchdog-timeout 86400
  )
  export UDCACHE_CPU_CACHE_CAP_BYTES=192000000000
  export UDCACHE_RECENCY_POLICY=pb
  export SGLANG_DISABLE_HISPARSE_PREFETCH=0
  export SGLANG_DEBUG_HISPARSE_SKIP_IO=0
  case "$MODE:$SPEC" in
    tp:on)
      MAX_REQUESTS=192
      args+=(
        --max-running-requests 192
        --cuda-graph-bs-decode 1 2 4 8
        --cuda-graph-max-bs-decode 8
        --attn-dp-size 1
        --speculative-algorithm EAGLE
        --speculative-num-steps 2
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 3
        --speculative-moe-runner-backend flashinfer_trtllm
        --speculative-moe-a2a-backend none
        --speculative-draft-model-quantization mxfp8
        --speculative-draft-load-format auto
        --disable-radix-cache
      )
      export ULACACHE_CONFIGURATION="$LOG_DIR/ulacache_config.json"
      cat > "$ULACACHE_CONFIGURATION" <<'JSON'
{"schema_version":1,"framework_release":"v0.5.22","adapter":"native_tp","options":{"native_kv_format":"scaled656","cpu_offload":true,"cpu_slots":3751488,"prefill_window_rows":65536,"low_query_pull_blocks":64,"prefetch":{"enabled":true,"completion":"layer"},"host_exchange_blocks":64,"host_exchange_threads":256,"local_eviction_policy":"age_lru","cache_capacity_multiplier":3.0,"device_cache_budget_bytes":160828237632}}
JSON
      ;;
    tp:off)
      MAX_REQUESTS=192
      args+=(
        --max-running-requests 192
        --cuda-graph-bs-decode 1 2 4 8
        --cuda-graph-max-bs-decode 8
        --attn-dp-size 1
        --disable-radix-cache
      )
      export ULACACHE_CONFIGURATION="$LOG_DIR/ulacache_config.json"
      cat > "$ULACACHE_CONFIGURATION" <<'JSON'
{"schema_version":1,"framework_release":"v0.5.22","adapter":"native_tp","options":{"native_kv_format":"scaled656","cpu_offload":true,"cpu_slots":3751488,"prefill_window_rows":65536,"low_query_pull_blocks":64,"prefetch":{"enabled":true,"completion":"layer"},"host_exchange_blocks":64,"host_exchange_threads":256,"local_eviction_policy":"age_lru","cache_capacity_multiplier":3.0,"device_cache_budget_bytes":165028237632}}
JSON
      ;;
    dpa:on)
      MAX_REQUESTS=1024
      args+=(
        --max-running-requests 1024
        --cuda-graph-bs-decode 1 2 4 8
        --cuda-graph-max-bs-decode 8
        --attn-dp-size 8
        --speculative-algorithm EAGLE
        --speculative-num-steps 2
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 3
        --speculative-moe-runner-backend flashinfer_trtllm
        --speculative-moe-a2a-backend none
        --speculative-draft-model-quantization mxfp8
        --speculative-draft-load-format auto
        --disable-radix-cache
      )
      export ULACACHE_CONFIGURATION="$LOG_DIR/ulacache_config.json"
      cat > "$ULACACHE_CONFIGURATION" <<'JSON'
{"schema_version":1,"framework_release":"v0.5.22","adapter":"native_dpa","capacity":{"device_token_capacity_per_owner":2175808,"host_token_capacity_per_owner":3752320},"options":{"maximum_write_record_count":8192,"maximum_running_requests":128,"device_cache_budget_bytes":128000000000,"host_cache_budget_bytes":192000000000,"exchange_blocks":64,"exchange_threads":256,"local_current_publication":true}}
JSON
      ;;
    dpa:off)
      MAX_REQUESTS=1024
      args+=(
        --max-running-requests 1024
        --cuda-graph-bs-decode 1 2 4 8
        --cuda-graph-max-bs-decode 8
        --attn-dp-size 8
        --disable-radix-cache
      )
      export ULACACHE_CONFIGURATION="$LOG_DIR/ulacache_config.json"
      cat > "$ULACACHE_CONFIGURATION" <<'JSON'
{"schema_version":1,"framework_release":"v0.5.22","adapter":"native_dpa","capacity":{"device_token_capacity_per_owner":2294208,"host_token_capacity_per_owner":3752320},"options":{"maximum_write_record_count":8192,"maximum_running_requests":128,"device_cache_budget_bytes":128000000000,"host_cache_budget_bytes":192000000000,"exchange_blocks":64,"exchange_threads":256,"local_current_publication":true}}
JSON
      ;;
    *) echo "Use --mode tp|dpa and --spec on|off" >&2; exit 2;;
  esac
  export UDCACHE_MATRIX_CACHE_KIND="${MODE}_udcache"
  ;;
*) echo "Use --platform h200|b300" >&2; exit 2;;
esac

if [[ -n "$REPLAY_TRACE" ]]; then
  export ULACACHE_BENCH_REPLAY_TRACE="$REPLAY_TRACE"
fi

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
