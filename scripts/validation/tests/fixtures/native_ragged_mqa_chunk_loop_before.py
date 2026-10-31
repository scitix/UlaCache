# Exact native ragged chunk loop from dsa_indexer.py SHA256 01b2b70b2b32155d85302a3bb1289afdd12222d220b87b1ff5e1e70f285ff6a4
# Only wrapped in a function for standalone CPU/GPU lifecycle probes.
def native_ragged_chunk_loop():
    start = 0
    while start < q_offset:
        end = min(start + max_rows, q_offset)

        with self._with_real_sm_count():
            if _is_hip:
                from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

                kv, scale = kv_fp8
                # clean_logits=False: topk transform handles masking (see above)
                logits_chunk = fp8_mqa_logits(
                    q_fp8[start:end],
                    kv,
                    scale,
                    weights[start:end],
                    ks[start:end],
                    ke[start:end],
                    clean_logits=False,
                )
            elif _is_xpu:
                logits_chunk = sgl_fp8_mqa_logits(
                    q_fp8[start:end],
                    kv_fp8,
                    weights[start:end],
                    ks[start:end],
                    ke[start:end],
                    clean_logits=False,
                )
            else:
                q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(
                    q_fp8[start:end], weights[start:end]
                )
                logits_chunk = resolve_fp8_mqa_logits_fn()(
                    q_padded,
                    kv_fp8,
                    w_padded,
                    ks[start:end],
                    ke[start:end],
                    clean_logits=False,
                )

        lengths_chunk = seq_lens_expanded[start:end]
        self._mask_init_and_local_tokens(logits_chunk, lengths_chunk, ks[start:end])

        # RAGGED: use global offset; PAGED: construct local cu_seqlens_q per chunk
        if global_topk_offset is not None:
            # RAGGED path
            topk_offset_chunk = global_topk_offset[start:end]
            cu_seqlens_q_chunk = None
            batch_idx_chunk = None
        else:
            # PAGED path: treat each token as a length-1 sequence
            topk_offset_chunk = None
            cu_seqlens_q_chunk = cu_seqlens_q_full[start:end]
            batch_idx_chunk = token_to_batch_idx[start:end]

        raw_topk_chunk = metadata.topk_transform(
            logits_chunk,
            self.index_topk,
            ks=ks[start:end],
            cu_seqlens_q=cu_seqlens_q_chunk,
            ke_offset=lengths_chunk,
            batch_idx_list=batch_idx_chunk,
            topk_indices_offset_override=topk_offset_chunk,
        )
        topk_result[start:end] = raw_topk_chunk
        start = end

    return topk_result
