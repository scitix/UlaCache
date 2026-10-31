def _build_dsa_kv_pool(
        self,
        *,
        max_total_num_tokens: int,
        max_running_requests: int,
        dsa_pool_class: type,
    ):
        from sglang.srt.layers.cp.utils import get_glm_dsa_cp_layer_shard_info

        (
            dsa_cp_layer_shard_rank,
            dsa_cp_layer_shard_size,
        ) = get_glm_dsa_cp_layer_shard_info(self)
        pool_kwargs = {}
        if get_memory().enable_hisparse and not self.is_draft_worker:
            reject_out_of_tree_path(
                current_platform, subsystem="the HiSparse DSA KV pool"
            )
            PoolCls = HiSparseDSATokenToKVPool
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            pool_kwargs["host_to_device_ratio"] = (
                parse_hisparse_config().host_to_device_ratio
            )
        elif dsa_cp_layer_shard_rank is not None:
            reject_out_of_tree_path(
                current_platform, subsystem="the CP layer-split DSA KV pool"
            )
            # DSA cache layer split: shard KV/indexer layers across CP ranks.
            from sglang.srt.mem_cache.dsa_cache_layer_split import (
                LayerSplitDSATokenToKVPool,
            )

            PoolCls = LayerSplitDSATokenToKVPool
            pool_kwargs["layer_shard_rank"] = dsa_cp_layer_shard_rank
            pool_kwargs["layer_shard_size"] = dsa_cp_layer_shard_size
        else:
            PoolCls = dsa_pool_class
            from v0522_owner_tp import configuration
            if configuration().enabled and not self.is_draft_worker:
                from sglang.srt.mem_cache.flowkv_plain_pool import FlowKVPlainTokenToKVPool
                from sglang.srt.mem_cache.flowkv_config import plain_record_layout
                PoolCls = FlowKVPlainTokenToKVPool
                pool_kwargs['flowkv_layout'] = plain_record_layout(self, max_total_num_tokens)
        from v0522_owner_tp import configuration
        if self.is_draft_worker and configuration().enabled and configuration().owner_index_draft:
            from tensor_parallel_draft_index import create_owner_index_draft_pool_class
            PoolCls = create_owner_index_draft_pool_class(
                PoolCls, self.token_to_kv_pool_allocator.get_kvcache())
        if self.loc_space_scale > 1:
            pool_kwargs["dcp_replicated"] = True
        if _should_elide_dsa_index_k(is_draft_worker=self.is_draft_worker):
            pool_kwargs["skip_topk_layers"] = [
                dsa_layer_skips_topk(self.model_config.hf_config, layer_id)
                for layer_id in range(
                    self.layer_info.start_layer, self.layer_info.end_layer
                )
            ]
        token_to_kv_pool = PoolCls(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            kv_cache_dim=calculate_mla_kv_cache_dim(
                model_config=self.model_config,
                kv_cache_dtype=self.kv_cache_dtype,
            ),
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
            index_kpool=get_dsa_index_kpool(self.model_config.hf_config),
            index_kpool_compress=get_dsa_index_kpool_compress(
                self.model_config.hf_config
            ),
            tail_extra_slots=(max_speculative_num_draft_tokens() or 0),
            max_running_requests=max_running_requests,
            **pool_kwargs,
        )
        return token_to_kv_pool
