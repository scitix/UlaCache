"""Distributed native FP8 Index storage and scoring with original math services."""

from ulacache.indexer.ownership import NativeIndexOwnership


class NativeFP8IndexRuntime:
    """Keep each encoded record once and retain native logical TopK inputs.

    The integration supplies original paged/ragged FP8 scoring callbacks and
    scheduling. Cold Index records are token-striped across the owner group.
    Decode packs this owner's requested keys into native page64 scratch, then
    reuses the CP scorer's private score collective. Eager prefill already
    receives complete reconstructed keys, so its default uses the original
    native ragged callback directly. Column-striped scoring remains explicit.
    """

    def __init__(
        self,
        ownership,
        storage,
        *,
        maximum_queries,
        maximum_columns,
        score_group,
        cpu_group,
        ipc_manager_factory,
        schedule_builder,
        prefill_query_chunk_count=None,
        ragged_score_mode="native",
    ):
        if ragged_score_mode not in ("native", "distributed"):
            raise ValueError(
                "Native ragged Index scoring mode must be native or distributed"
            )
        import torch
        import triton
        import torch.distributed as distributed
        from ulacache.indexer.scorer import CPIndexScorer
        from ulacache.transfer.ipc import FlowKVPeerShards
        from ulacache.cache.reader_lifetime import DistributedResidencyPublication

        if (
            not isinstance(ownership, NativeIndexOwnership)
            or ownership.records_per_native_page != 64
            or ownership.bytes_per_record != 132
        ):
            raise ValueError("Bind the complete native FP8 page64 Index geometry")
        if (
            storage.dtype != torch.uint8
            or storage.ndim != 3
            or storage.shape[1:] != (ownership.native_page_count, 8448)
            or not storage.is_contiguous()
            or storage.device.type != "cuda"
        ):
            raise ValueError(
                "Bind contiguous native encoded Index banks before capture"
            )
        if (
            type(maximum_columns) is not int
            or maximum_columns < 1
            or type(maximum_queries) is not int
            or maximum_queries < 1
        ):
            raise ValueError("Declare native Index query and logical-column bounds")
        self.ownership, self.storage = ownership, storage
        self.ragged_score_mode = ragged_score_mode
        self.device = storage.device
        self.rank = distributed.get_rank(cpu_group)
        self.degree = distributed.get_world_size(cpu_group)
        if (
            self.degree != ownership.owner_count
            or score_group.world_size != self.degree
        ):
            raise ValueError(
                "Cold Index owners and score collective must use the same rank order"
            )
        self.maximum_queries, self.maximum_columns = maximum_queries, maximum_columns
        self.prefill_query_chunk_count = (
            maximum_queries
            if prefill_query_chunk_count is None
            else prefill_query_chunk_count
        )
        if (
            type(self.prefill_query_chunk_count) is not int
            or not 1 <= self.prefill_query_chunk_count <= maximum_queries
        ):
            raise ValueError(
                "Prefill query chunks must fit the declared Index collective arena"
            )
        self.maximum_page_width = triton.cdiv(maximum_columns, 64)
        self.local_page_stride = (
            triton.cdiv(triton.cdiv(self.maximum_page_width, self.degree), 4) * 4
        )
        self.schedule_builder, self.score_group, self.cpu_group = (
            schedule_builder,
            score_group,
            cpu_group,
        )
        self.scorer = CPIndexScorer(
            rank=self.rank,
            degree=self.degree,
            max_queries=maximum_queries,
            max_page_width=self.maximum_page_width,
            group=score_group,
            device=self.device,
        )
        self.native_pages = torch.empty(
            (maximum_queries * self.local_page_stride, 8448),
            dtype=torch.uint8,
            device=self.device,
        )
        rows = torch.arange(maximum_queries, dtype=torch.int32, device=self.device)
        columns = torch.arange(
            self.maximum_page_width, dtype=torch.int32, device=self.device
        )
        self.virtual_page_table = (
            rows[:, None] * (self.local_page_stride * self.degree)
            + (columns[None, :] // self.degree) * self.degree
        )
        self.dense_scores = torch.empty(
            (maximum_queries, maximum_columns), dtype=torch.float32, device=self.device
        )
        self.ragged_ranges = torch.empty(
            (4, maximum_queries), dtype=torch.int32, device=self.device
        )
        self.peers = FlowKVPeerShards(
            storage, cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        self.prefix_publication = DistributedResidencyPublication(
            storage.shape[0],
            self.device,
            cpu_group,
            ipc_manager_factory=ipc_manager_factory,
        )
        self._plan = None
        self._epoch = None
        self._closed = False

    def begin_forward(self):
        self._plan = None
        self._epoch = object()

    def before_native_write(self, bank):
        """Retire remote prefix readers before native storage may be reused."""
        self.prefix_publication.wait_for_readers(bank)

    def _bucket(self, queries, page_width):
        import torch

        key = (queries, page_width)
        bucket = self.scorer.buckets.get(key)
        if bucket is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm each native Index query/page shape before Graph capture"
                )
            bucket = self.scorer.register_bucket(*key)
        return bucket

    def score_paged(self, bank, q, weights, lengths, page_table, *, native_score):
        from ulacache.indexer.kernels.fp8_storage import (
            gather_local_native_fp8_pages,
            restore_native_fp8_scores,
        )

        queries, page_width = page_table.shape
        if (
            queries > self.maximum_queries
            or page_width > self.maximum_page_width
            or q.shape[0] != queries
            or weights.shape[0] != queries
        ):
            raise ValueError(
                "Native paged Index exceeds its declared query/page bounds: "
                f"pages={tuple(page_table.shape)}, q={tuple(q.shape)}, weights={tuple(weights.shape)}, "
                f"maximum_queries={self.maximum_queries}, maximum_page_width={self.maximum_page_width}"
            )
        if self._epoch is None:
            raise RuntimeError("Begin the native Index forward before scoring")
        bucket = self._bucket(queries, page_width)
        if self._plan is None:
            self._plan = self.scorer.prepare_forward(
                page_table=self.virtual_page_table[:queries, :page_width],
                lengths=lengths,
                schedule_builder=self.schedule_builder,
                epoch=self._epoch,
            )
        elif self._plan.bucket is not bucket:
            raise ValueError(
                "Native Index anchors must share one forward query/page shape"
            )
        gather_local_native_fp8_pages(
            page_table,
            lengths,
            self.storage,
            self.native_pages,
            owner_rank=self.rank,
            owner_count=self.degree,
            bank_byte_offset=bank * self.storage.stride(0),
            destination_page_stride=self.local_page_stride,
            local_page_count=bucket.local_pages.shape[1],
        )
        native_cache = self.native_pages.view(-1, 64, 1, 132)
        layout = self.scorer.score(
            q=q,
            kv_cache=native_cache,
            weights=weights,
            plan=self._plan,
            epoch=self._epoch,
            score_fn=native_score,
        )
        # Native TopK consumes contiguous logical columns and owns causal/tail
        # selection. This layout conversion never changes a valid FP32 score.
        logical_columns = min(page_width * 64, self.maximum_columns)
        if layout.logical_columns != logical_columns:
            layout.logical_columns = logical_columns
        output = self.dense_scores.reshape(-1)[: queries * logical_columns].view(
            queries, logical_columns
        )
        return restore_native_fp8_scores(layout, lengths, output)

    def gather_prefix(self, bank, page_table, keys, scales):
        from ulacache.indexer.kernels.fp8_storage import gather_native_fp8_prefix

        self.prefix_publication.begin_selection(bank)
        gather_native_fp8_prefix(
            self.peers.ptrs,
            page_table,
            keys,
            scales,
            owner_count=self.degree,
            bank_byte_offset=bank * self.storage.stride(0),
        )
        self.prefix_publication.finish_readers(bank)

    def gather_ragged_prefix(
        self, bank, page_table, lengths, keys, scales, *, maximum_length
    ):
        from ulacache.indexer.kernels.fp8_storage import gather_native_fp8_ragged_prefix

        self.prefix_publication.begin_selection(bank)
        gather_native_fp8_ragged_prefix(
            self.peers.ptrs,
            page_table,
            lengths,
            keys,
            scales,
            maximum_length=maximum_length,
            owner_count=self.degree,
            bank_byte_offset=bank * self.storage.stride(0),
        )
        self.prefix_publication.finish_readers(bank)

    def score_ragged(
        self, q, keys, scales, weights, starts, ends, *, native_score, clean_logits
    ):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError("Native ragged Index scoring remains eager-only")
        queries, columns = q.shape[0], keys.shape[0]
        if columns > self.ownership.logical_record_capacity:
            raise ValueError(
                "Native ragged Index exceeds its admitted logical history namespace"
            )
        if self.ragged_score_mode == "native" and queries and columns:
            # Prefix reconstruction has already supplied every encoded key.
            # Preserve the native output stride and math without splitting it
            # into small launches and gathering the same complete score matrix.
            # Cold Index storage and paged decode remain disjoint/distributed.
            return native_score(
                q, keys, scales, weights, starts, ends, clean_logits=clean_logits
            )
        import triton
        from ulacache.indexer.score_layout import RankMajorScoreLayout
        from ulacache.indexer.kernels.fp8_storage import (
            restore_native_fp8_scores,
            native_fp8_ragged_ranges,
        )

        output = torch.empty(
            (queries, columns), dtype=torch.float32, device=self.device
        )
        if not queries or not columns:
            return output
        # Native ragged metadata can concatenate several requests. Bound both
        # axes of each collective without enlarging decode's retained arena.
        for column_start in range(0, columns, self.maximum_columns):
            column_stop = min(column_start + self.maximum_columns, columns)
            column_count = column_stop - column_start
            page_width = triton.cdiv(column_count, 64)
            prototype = self._bucket(
                min(queries, self.prefill_query_chunk_count), page_width
            )
            local_columns = prototype.local_columns
            local_keys = torch.zeros(
                (local_columns, keys.shape[1]), dtype=keys.dtype, device=self.device
            )
            local_scales = torch.zeros(
                local_columns, dtype=scales.dtype, device=self.device
            )
            owned_keys = keys[column_start:column_stop][self.rank :: self.degree]
            owned_scales = scales[column_start:column_stop][self.rank :: self.degree]
            local_keys[: owned_keys.shape[0]].copy_(owned_keys)
            local_scales[: owned_scales.numel()].copy_(owned_scales.reshape(-1))
            for start in range(0, queries, self.prefill_query_chunk_count):
                stop = min(start + self.prefill_query_chunk_count, queries)
                count = stop - start
                bucket = self._bucket(count, page_width)
                chunk_starts, chunk_ends, local_starts, local_ends = (
                    native_fp8_ragged_ranges(
                        starts,
                        ends,
                        self.ragged_ranges,
                        query_start=start,
                        query_count=count,
                        column_start=column_start,
                        column_count=column_count,
                        owner_rank=self.rank,
                        owner_count=self.degree,
                    )
                )
                local = native_score(
                    q[start:stop].contiguous(),
                    local_keys,
                    local_scales,
                    weights[start:stop].contiguous(),
                    local_starts,
                    local_ends,
                    clean_logits=clean_logits,
                )
                if (
                    local.shape != (count, local_columns)
                    or local.dtype != torch.float32
                ):
                    raise ValueError(
                        "Native FP8 scorer must preserve its FP32 score ABI"
                    )
                # Native ragged kernels can expose padded row strides.
                self.score_group.all_gather_into_tensor(
                    bucket.gathered, local.contiguous()
                )
                layout = RankMajorScoreLayout(
                    bucket.gathered.view(self.degree, count, local_columns),
                    column_count,
                    self.degree,
                    score_page_size=1,
                )
                restore_native_fp8_scores(
                    layout,
                    chunk_ends,
                    output[start:stop, column_start:column_stop],
                    starts=chunk_starts,
                )

        return output

    def get_device_storage_inventory(self):
        import torch

        tensors = [
            self.storage,
            self.native_pages,
            self.virtual_page_table,
            self.dense_scores,
            self.ragged_ranges,
            self.scorer.local_pages_workspace,
            self.scorer.local_lengths_workspace,
            self.scorer.schedule_lengths_workspace,
            self.peers.ptrs,
        ]
        for name in ("compact_prefix_workspace", "compact_packs_workspace"):
            value = getattr(self.scorer, name)
            if value is not None:
                tensors.append(value)
        for name in ("slab", "gathered_workspace", "_peer_table"):
            value = getattr(self.score_group, name, None)
            if value is not None:
                tensors.append(value)
        tensors.extend(getattr(self.score_group, "_counters", {}).values())

        def retain(value):
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    retain(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    retain(item)

        for plan in [self._plan, *self.scorer.captured_plans]:
            if plan is not None:
                retain(plan.lengths)
                retain(plan.schedule)
        result = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for t in tensors
        }
        result.update(self.prefix_publication.get_device_storage_inventory())
        return result

    def close(self):
        import torch
        import torch.distributed as distributed

        if self._closed:
            return
        self.prefix_publication.wait_for_all_readers()
        torch.cuda.synchronize(self.device)
        distributed.barrier(group=self.cpu_group)
        self.peers.close()
        self.prefix_publication.close()
        self.score_group.close()
        self._closed = True
