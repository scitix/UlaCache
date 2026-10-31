"""JIT entry points for interleaved whole-record C4 transport."""

from ulacache.compiler import cache_once, load_jit


@cache_once
def flowkv_record_module(
    *,
    topk,
    max_queries,
    page_size,
    bytes_per_page,
    peers,
    slots_per_rank,
    row_bytes,
    native_segment_bytes,
    step_publication=False,
    host_tier=False,
    request_tier=False,
    gpu_slots_per_rank=None,
):
    segments = tuple(native_segment_bytes)
    if not segments or any(width <= 0 or width % 8 for width in segments):
        raise ValueError("Native record segments must contain aligned eight-byte words")
    if sum(segments) != row_bytes or bytes_per_page < page_size * row_bytes:
        raise ValueError("Native page geometry does not match the complete KV record")
    packed_page = segments == (row_bytes - 8, 8)
    if step_publication and segments != (row_bytes,):
        raise NotImplementedError("Step publication requires ordinary MLA records")
    definitions = {
        "FLOW_TOPK": topk,
        "FLOW_MAX_QUERIES": max_queries,
        "FLOW_ROW_BYTES": row_bytes,
        "FLOW_RECORD_PEERS": peers,
        "FLOW_RECORD_SLOTS": slots_per_rank,
        "FLOW_C4_PAGE_SIZE": page_size,
        "FLOW_C4_BYTES_PER_PAGE": bytes_per_page,
        "FLOW_ZERO_SENTINEL": 1,
        "GLOBAL_SMALL_HIST_BLOCKS": 64,
        "GLOBAL_HIST_BLOCKS": 128,
        "GLOBAL_ALLOC_ITEMS": 4,
        "FLOW_CIRCULAR_MAX_QUERIES": 64,
        "FLOW_PARALLEL_CIRCULAR": 1,
        "FLOW_DIRECT_CURRENT_LOOKUP": 1,
        "FLOW_PDL": 1,
        "FLOW_C4_SKIP_EMPTY_SCAN": 1,
    }
    wrappers = [
        ("begin", "sglang::flowkv_begin"),
        ("begin_native", "sglang::flowkv_record_begin"),
        ("invalidate", "sglang::flowkv_invalidate"),
        ("plan", "sglang::flowkv_plan<false>"),
        ("allocate", "sglang::flowkv_plan<true>"),
        ("plan_native", "sglang::flowkv_record_plan_native"),
        ("globalize_current", "sglang::flowkv_record_globalize_current"),
        ("globalize", "sglang::flowkv_record_globalize"),
        ("globalize_bound", "sglang::flowkv_record_globalize_bound"),
        ("fill_native", "sglang::flowkv_record_fill_native"),
        ("seed_native", "sglang::flowkv_record_seed_native"),
        ("publish", "sglang::flowkv_record_publish"),
        ("gather", "sglang::flowkv_record_gather"),
        ("pull", "sglang::flowkv_record_pull_cached<true, true>"),
        ("prefetch", "sglang::flowkv_record_pull_cached<true, false>"),
        ("install_current", "sglang::flowkv_record_pull_cached<false, true>"),
    ]
    if step_publication:
        definitions["FLOW_RECORD_STEP_PUBLICATION"] = 1
        wrappers.extend(
            [
                ("begin_published", "sglang::flowkv_record_begin_published"),
                ("publish_step", "sglang::flowkv_record_publish_step"),
                ("validate_step_peer", "sglang::flowkv_record_validate_step_peer"),
            ]
        )
    if not packed_page:
        definitions["FLOW_RECORD_NATIVE_LAYOUT"] = 1
        definitions["FLOW_RECORD_PAGE_SEGMENT_BYTES"] = ",".join(map(str, segments))
        wrappers = [
            (
                name,
                function.replace(
                    "flowkv_record_pull_cached", "flowkv_record_pull_native_cached"
                ),
            )
            for name, function in wrappers
        ]
        wrappers.extend(
            [
                (
                    "follower",
                    "sglang::flowkv_record_pull_native_cached<true, false, false>",
                ),
                ("write_decode", "sglang::flowkv_record_write_plain<true>"),
                ("write_prefill", "sglang::flowkv_record_write_plain<false>"),
            ]
        )
    if host_tier:
        if packed_page or not step_publication:
            raise ValueError(
                "CPU residency requires owner-striped ordinary MLA records"
            )
        definitions["FLOW_RECORD_HOST_TIER"] = 1
        wrappers.extend(
            [
                ("host_publish", "sglang::flowkv_record_host_publish"),
                (
                    "finish_deferred_group",
                    "sglang::flowkv_finish_deferred_gpu_bookkeeping",
                ),
                (
                    "host_prefill_read_done",
                    "sglang::flowkv_record_host_prefill_read_done",
                ),
            ]
        )
    if request_tier:
        if host_tier or packed_page or not step_publication:
            raise ValueError(
                "Request residency requires ordinary published MLA without Index CPU manager"
            )
        if (
            not isinstance(gpu_slots_per_rank, int)
            or gpu_slots_per_rank <= 0
            or gpu_slots_per_rank % 64
            or slots_per_rank % 64
        ):
            raise ValueError(
                "Request page map requires page64 GPU and logical capacity"
            )
        definitions["FLOW_RECORD_REQUEST_TIER"] = 1
        definitions["FLOW_RECORD_GPU_SLOTS"] = gpu_slots_per_rank
        wrappers.append(
            ("retire_request_pages", "sglang::flowkv_record_retire_request_pages")
        )
    elif gpu_slots_per_rank is not None:
        raise ValueError("Physical GPU capacity is only valid for request residency")
    return load_jit(
        "flowkv_record_pool",
        *(str(v).replace(",", "_") for v in definitions.values()),
        cuda_files=["csrc/cache/attention_cache.cuh", "csrc/storage/records.cuh"],
        cuda_wrappers=wrappers,
        # NVCC parses macro commas; Ninja removes one shell escape level.
        extra_cuda_cflags=[
            f"-D{k}=" + str(v).replace(",", r"\\,") for k, v in definitions.items()
        ],
    )
