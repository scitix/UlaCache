# SPDX-License-Identifier: Apache-2.0
"""Shared allocation geometry for record storage and its memory budget."""

from ulacache.storage.capacity import flowkv_cpu_pool_geometry

from sglang.srt.mem_cache.flowkv_adapter import FlowKVPackedMLASpec
from sglang.srt.mem_cache.flowkv_layout import layout_for_packed_mla




def validate_record_config(kvc, *, compression_ratio=4, supports_prefetch=False):
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.flowkv_v0522_context import (
        get_disagg,
        get_exec,
        get_memory,
        get_parallel,
        get_spec,
    )

    if get_memory().enable_flowkv_prefetch and not supports_prefetch:
        raise NotImplementedError(
            "This record adapter has no shared Index prefetch lifecycle"
        )
    if kvc.device != "cuda":
        raise ValueError(f"Record peer storage requires CUDA, received {kvc.device}")
    import os
    owner_query = os.environ.get("UDCACHE_OWNER_QUERY_PREFILL") == "1"
    if owner_query and compression_ratio != 1:
        raise NotImplementedError("Owner-query pilot requires native Plain KV")
    cp_prefill = get_memory().enable_flowkv_cp_prefill
    if cp_prefill:
        parallel = get_parallel()
        if (
            compression_ratio != 4
            or parallel.attn_dp_size != 1
            or parallel.attn_cp_size != parallel.tp_size
            or parallel.attn_cp_size < 2
            or parallel.pp_size != 1
            or parallel.moe_ep_size != 1
            or get_memory().enable_flowkv_cpu_offload
        ):
            raise NotImplementedError(
                "FlowKV CP prefill requires GPU packed C4, full CP within one "
                "shared TP namespace, PP1 and EP1"
            )
        page = kvc.pool_page_size // compression_ratio
        rows = get_memory().flowkv_prefill_window_rows
        if rows < page or rows % page:
            raise ValueError("FlowKV prefill window rows must be positive and native-page aligned")
    if get_memory().enable_flowkv_fused_topk and (
        get_exec().kernel.dsa_topk_backend != "sgl-kernel"
        or not envs.SGLANG_OPT_USE_TOPK_V2.get()
    ):
        raise ValueError("Fused FlowKV lookup requires native SGL TopK v2")
    if get_spec().speculative_algorithm is not None and compression_ratio == 1 and supports_prefetch:
        from ulacache.speculative import LinearVerifyConfig

        spec = get_spec()
        LinearVerifyConfig(
            algorithm=spec.speculative_algorithm,
            steps=spec.speculative_num_steps,
            topk=spec.speculative_eagle_topk,
            verify_tokens=spec.speculative_num_draft_tokens,
            adaptive=spec.speculative_adaptive,
        ).validate()
    unsupported = {
        "HiCache": get_memory().enable_hierarchical_cache,
        "HiSparse": get_memory().enable_hisparse,
        "unified-memory allocator": get_memory().enable_unified_memory,
        "PD disaggregation": get_disagg().disaggregation_mode != "null",
        "speculative decoding": (get_spec().speculative_algorithm is not None
            and not (compression_ratio == 1 and supports_prefetch)),
        "prefill CP": get_parallel().attn_cp_size > 1 and not cp_prefill and not owner_query,
        "decode CP": get_parallel().attn_dcp_size > 1,
        "prefill CUDA Graph": get_exec().graph.cuda_graph_config.prefill.backend
        != "disabled",
    }
    for feature, selected in unsupported.items():
        if selected:
            raise NotImplementedError(
                f"The record KV adapter for {feature} has not been integrated"
            )
    from sglang.srt.mem_cache.flowkv_record_storage import storage_geometry

    degree, _, _ = storage_geometry()
    if degree < 1 or (kvc.pool_page_size // compression_ratio) % degree:
        raise ValueError("Storage peers must divide the native compressed page size")


def record_layout(kvc, c4_size):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        _num_dsv4_physical_kv_pages,
        collect_sources_by_ratio,
    )

    stage = range(kvc.layer_info.start_layer, kvc.layer_info.end_layer)
    sources = collect_sources_by_ratio(
        kvc.model_config.compress_ratios,
        kvc.model_config.hf_config.kv_source_layer_ids,
        stage,
    )[4]
    page = kvc.pool_page_size // 4
    rows = _num_dsv4_physical_kv_pages(c4_size, page, kvc.pool_page_size) * page
    return layout_for_packed_mla(
        kvc,
        rows,
        FlowKVPackedMLASpec(
            compression_ratio=4,
            indexed_layer_ids=sources,
            topk=kvc.model_config.hf_config.index_topk,
            row_bytes=584,
            native_segment_bytes=(576, 8),
            index_layer_anchors=tuple(sources),
            page_storage_alignment=576,
        ),
    )


def plain_record_spec(kvc):
    """Describe native MLA row bytes and native Index-sharing roles."""
    from v0522_owner_tp import validate_native_context
    validate_native_context(kvc)
    import os
    import torch
    from sglang.srt.configs.model_config import (
        dsa_layer_skips_topk,
        get_dsa_mtp_topk_width,
    )
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.kv_cache_configurator import (
        calculate_mla_kv_cache_dim,
    )
    from sglang.srt.mem_cache.flowkv_v0522_context import get_exec, get_memory, get_parallel, get_spec

    if not kvc.use_mla_backend or kvc.is_hybrid_swa or kvc.is_draft_worker:
        raise NotImplementedError("Plain record storage requires a target MLA cache")
    if kvc.kv_cache_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise NotImplementedError("Raw MLA records require BF16 or FP8 E4M3 KV")
    native_dim = calculate_mla_kv_cache_dim(
        model_config=kvc.model_config, kv_cache_dtype=kvc.kv_cache_dtype
    )
    scaled_fp8 = kvc.kv_cache_dtype == torch.float8_e4m3fn and native_dim != (
        kvc.model_config.kv_lora_rank + kvc.model_config.qk_rope_head_dim
    )
    parallel = get_parallel()
    spec = get_spec()
    native_bf16_linear = (
        os.environ.get("ULACACHE_EXPERIMENTAL_NATIVE_BF16_MTP") == "1"
        and kvc.kv_cache_dtype == torch.bfloat16
        and native_dim == kvc.model_config.kv_lora_rank + kvc.model_config.qk_rope_head_dim
        and get_exec().kernel.dsa_prefill_backend == "flashmla_sparse"
        and get_exec().kernel.dsa_decode_backend == "flashmla_sparse"
        and parallel.attn_dp_size == parallel.attn_cp_size == parallel.attn_dcp_size == 1
        and parallel.attn_tp_size == parallel.tp_size == 8
        and spec.speculative_algorithm == "EAGLE"
        and spec.speculative_num_steps == 2
        and spec.speculative_eagle_topk == 1
        and spec.speculative_num_draft_tokens == 3
    )
    # This explicit experimental binding retains native BF16 writers/readers.
    # Its mandatory eight-owner fixture verifies causal three-query attention,
    # CPU residency and graph replay before a complete-model experiment starts.
    native_raw_linear = (not scaled_fp8 and native_dim == 576
        and kvc.kv_cache_dtype == torch.float8_e4m3fn
        and get_exec().kernel.dsa_prefill_backend == "trtllm"
        and get_exec().kernel.dsa_decode_backend == "trtllm")
    if spec.speculative_algorithm is not None and not native_raw_linear:
        raise NotImplementedError("This version-pinned GLM binding requires raw576 TRTLLM target records")
    prefill = get_exec().kernel.dsa_prefill_backend
    decode = get_exec().kernel.dsa_decode_backend
    if scaled_fp8:
        # These native readers consume the same per-token FP8 values, FP32
        # block scales and BF16 RoPE bytes produced by the native writer.
        if decode != "flashmla_kv" or prefill not in (
            "flashmla_kv", "flashmla_sparse", "flashmla_sparse_q8"
        ):
            raise NotImplementedError(
                "Scaled FP8 MLA records require native FlashMLA KV decode "
                "and a compatible FlashMLA prefill reader"
            )
        if kvc.model_config.qk_rope_head_dim not in (0, 64):
            raise NotImplementedError(
                "The native scaled FP8 MLA writer requires zero or 64 RoPE dimensions"
            )
    else:
        if decode not in ("trtllm", "flashmla_sparse") or prefill not in (
            "trtllm", "flashmla_sparse"
        ):
            raise NotImplementedError(
                "Plain records require native TRTLLM/FlashMLA sparse decode and prefill"
            )
        if kvc.kv_cache_dtype == torch.float8_e4m3fn and (
            decode != "trtllm" or prefill != "trtllm"
        ):
            raise NotImplementedError("Raw FP8 MLA records require TRTLLM in both phases")
    if get_memory().enable_flowkv_fused_topk:
        raise NotImplementedError(
            "Plain shared-Index uses the native Index return interface"
        )
    if (
        not envs.SGLANG_DSA_FUSE_TOPK.get()
        or get_exec().kernel.dsa_topk_backend not in ("sgl-kernel", "flashinfer")
    ):
        raise NotImplementedError(
            "Plain record storage requires native physical TopK output"
        )
    if get_exec().overlap.enable_two_batch_overlap:
        raise NotImplementedError("Shared Index propagation has no TBO lifecycle")
    layers = tuple(range(kvc.layer_info.start_layer, kvc.layer_info.end_layer))
    owners = []
    for layer in layers:
        if not dsa_layer_skips_topk(kvc.model_config.hf_config, layer):
            anchor = layer
        elif not owners:
            raise ValueError("The first local KV layer must own its Index")
        owners.append(anchor)
    row_bytes = kvc.kv_cache_dtype.itemsize * native_dim
    return FlowKVPackedMLASpec(
        compression_ratio=1,
        indexed_layer_ids=layers,
        # The native helper also describes the main Index output width:
        # index_topk plus every unpooled tail column, without truncation.
        topk=get_dsa_mtp_topk_width(kvc.model_config.hf_config),
        row_bytes=row_bytes,
        native_segment_bytes=(row_bytes,),
        index_layer_anchors=tuple(owners),
    )


def plain_record_layout(kvc, size):
    validate_record_config(kvc, compression_ratio=1, supports_prefetch=True)
    return layout_for_packed_mla(kvc, size + kvc.pool_page_size, plain_record_spec(kvc))
