    def _init_cp_index_scorer(self, model_runner):
        from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
            DSAPagedMQALogitsBackend,
        )

        native_score = DSAPagedMQALogitsBackend.resolve(
            get_exec().kernel.dsa_paged_mqa_logits_backend
        )
        from sglang.srt.mem_cache.flowkv_owner_index import FlowKVOwnerIndexCache

        if not (
            _is_cuda
            and native_score.is_deepgemm()
            and self.physical_page_size == 64
            and self.dsa_index_kpool == 1
            and self.use_fused_topk
            and self.dsa_topk_backend.should_use_topk_v2()
            and self.hisparse_coordinator is None
            and get_parallel().attn_dp_size == 1
            and (get_parallel().attn_cp_size == 1 or os.environ.get("UDCACHE_OWNER_QUERY_PREFILL") == "1")
            and not get_exec().overlap.enable_two_batch_overlap
            and not get_disagg().enable_pdmux
            and isinstance(self.token_to_kv_pool.index_key_cache, FlowKVOwnerIndexCache)
        ):
            raise ValueError(
                "CP Index requires native DeepGEMM, page64, fused PAGED TopK v2, "
                "owner-bound FlowKV Index, replicated single-token queries "
                "and serial Index execution"
            )
        from sglang.srt.distributed.device_communicators.flash_ag_index import (
            FlashAGIndexOwner,
        )
        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.layers.attention.dsa.cp_index_scorer import CPIndexScorer
        from ulacache.speculative import is_cache_step, index_query_metadata

        self._cp_cache_step = lambda mode: is_cache_step(mode) or (
            model_runner.is_draft_worker and mode.is_draft_extend_v2()
        )
        self._cp_query_metadata = lambda mode, metadata: index_query_metadata(
            mode, metadata, max_queries=model_runner.token_to_kv_pool.flowkv_layout.resident_query_capacity,
        )

        group = get_tp_group()
        # CP scores only the admitted MRQ * verify-width queries.  Kernel
        # launch padding is a separate bound and does not enlarge this domain.
        max_queries = model_runner.token_to_kv_pool.flowkv_layout.resident_query_capacity
        max_page_width = (self.req_to_token.shape[1] + 63) // 64
        device = torch.device("cuda", model_runner.gpu_id)
        if group.index_score_comm is None:
            group.index_score_comm = FlashAGIndexOwner(
                group, max_queries=max_queries,
                max_page_width=max_page_width, device=device,
            )
        else:
            group.index_score_comm.validate_bounds(max_queries, max_page_width)
        self.cp_index_scorer = CPIndexScorer(
            rank=group.rank_in_group, degree=group.world_size,
            max_queries=max_queries, max_page_width=max_page_width,
            group=group.index_score_comm, device=device,
            owner_page_map=(model_runner.token_to_kv_pool.flowkv_request.page_map
                if getattr(model_runner.token_to_kv_pool, "flowkv_request", None) is not None else None),
            owner_gpu_pages=(model_runner.token_to_kv_pool.flowkv_request.gpu_page_count
                if getattr(model_runner.token_to_kv_pool, "flowkv_request", None) is not None else 0),
        )
