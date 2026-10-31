# Extracted verbatim from framework 95c31eac dsa_indexer.py:1326-1377.
def _chunked_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    w: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    mqa_schedule_metadata: torch.Tensor,
    max_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    batch_size, chunk_next_n = q.shape[:2]
    if batch_size == 0:
        return torch.empty((0, max_len), dtype=torch.float32, device=q.device)
    if batch_size <= self.sm_count:
        return resolve_fp8_paged_mqa_logits_fn()(
            q,
            kv_cache,
            w,
            context_lens,
            block_table,
            mqa_schedule_metadata,
            max_len,
            clean_logits=clean_logits,
        )
    logits_chunks = []
    # CP partitions may have no local keys. Its prepared schedule uses
    # positive work bounds while scoring retains the actual zero
    # lengths. Preserve those bounds when rebuilding chunk schedules.
    schedule_context_lens = (
        metadata.cp_index_plan.bucket.schedule_lengths
        if metadata.cp_index_scorer is not None
        else context_lens
    )
    for start in range(0, batch_size, self.sm_count):
        end = min(start + self.sm_count, batch_size)
        chunk_context_lens = context_lens[start:end]
        chunk_schedule_metadata = resolve_paged_mqa_logits_metadata_fn()(
            schedule_context_lens[start:end], blocksize, self.sm_count
        )
        logits_chunks.append(
            resolve_fp8_paged_mqa_logits_fn()(
                q[start:end],
                kv_cache,
                w[start * chunk_next_n : end * chunk_next_n],
                chunk_context_lens,
                block_table[start:end],
                chunk_schedule_metadata,
                max_len,
                clean_logits=clean_logits,
            )
        )
    return torch.cat(logits_chunks, dim=0)
