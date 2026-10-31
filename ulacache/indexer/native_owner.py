"""Native Index scoring on disjoint owners with independent consumer queries."""

from ulacache.indexer.ownership import NativeIndexOwnership, NativeIndexQueryLayout


class NativeOwnerIndexScorer:
    """Same-host CUDA IPC scoring with an explicitly bound native math tile.

    Calls are serialized on the caller's CUDA stream. All owners execute the
    same physical query/column shapes and phase sequence, including padded
    idle rows. Queries, weights, request IDs, lengths and candidate masks may
    differ between consumers. Returned original-order scores are borrowed
    until the next call; the caller retains its unchanged native top-k.
    """

    def __init__(
        self,
        query_layout,
        *,
        device,
        cpu_group,
        ipc_manager_factory,
        compile_device_module,
        native_score_tile,
        isolate_storage=True,
        specialize_prefill_shapes=False,
        reuse_equal_decode_queries=True,
        reuse_equal_prefill_queries=True,
        minimum_prefill_reuse_cells=2097152,
    ):
        from contextlib import nullcontext
        import torch
        import torch.distributed as distributed
        from ulacache.compiler import configure_compiler

        if not isinstance(query_layout, NativeIndexQueryLayout):
            raise TypeError("Bind explicit native Index query/storage bounds")
        if native_score_tile is None:
            raise ValueError("Bind the original native score tile before capture")
        if type(specialize_prefill_shapes) is not bool:
            raise TypeError("Declare an explicit prefill shape specialization policy")
        if type(reuse_equal_decode_queries) is not bool:
            raise TypeError("Declare an explicit equal-query reuse policy")
        if type(reuse_equal_prefill_queries) is not bool:
            raise TypeError("Declare an explicit equal-prefill-query reuse policy")
        if (
            type(minimum_prefill_reuse_cells) is not int
            or minimum_prefill_reuse_cells < 0
        ):
            raise ValueError(
                "Declare a nonnegative integer prefill reuse work threshold"
            )
        self._minimum_prefill_reuse_cells = minimum_prefill_reuse_cells
        self._reuse_equal_prefill_queries = reuse_equal_prefill_queries
        self._reuse_equal_decode_queries = reuse_equal_decode_queries
        self._specialize_prefill_shapes = specialize_prefill_shapes
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "meta"):
            raise ValueError(
                "Native owner Index requires CUDA; meta is allocation-only"
            )
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Construct native Index communication before Graph capture"
                )
            if torch.cuda.get_device_capability(self.device)[0] != 9:
                raise NotImplementedError(
                    "Native owner Index transport is scoped to Hopper"
                )
        self._multiprocessors = (
            torch.cuda.get_device_properties(self.device).multi_processor_count
            if self.device.type == "cuda"
            else 0
        )
        self.rank = distributed.get_rank(cpu_group)
        self.degree = distributed.get_world_size(cpu_group)
        if self.degree != 8:
            raise NotImplementedError(
                "Native owner Index transport is scoped to eight ranks"
            )
        peers = [None] * self.degree
        query_contract = (
            query_layout,
            reuse_equal_decode_queries,
            reuse_equal_prefill_queries,
            minimum_prefill_reuse_cells,
        )
        distributed.all_gather_object(peers, query_contract, group=cpu_group)
        if any(peer != query_contract for peer in peers):
            raise ValueError(
                "Native Index owners disagree on query bounds or reuse policy"
            )
        self.layout = query_layout
        self.cpu_group = cpu_group
        self.native_score_tile = native_score_tile
        configure_compiler(compile_device_module)
        # Communication exports outlive weight-loading temporaries. Give their
        # CUDA allocations a separate pool so a small export cannot keep the
        # unused tail of a large, previously cached model allocation alive.
        # The pool uses the default cudaMalloc-backed allocator required by IPC.
        if type(isolate_storage) is not bool:
            raise TypeError("Declare an explicit Index allocation-pool policy")
        self._allocation_pool = (
            torch.cuda.MemPool()
            if self.device.type == "cuda" and isolate_storage
            else None
        )
        context = (
            torch.cuda.use_mem_pool(self._allocation_pool, device=self.device)
            if self._allocation_pool is not None
            else nullcontext()
        )
        with context:
            self._allocate_storage(ipc_manager_factory)
        self._closed = False

    def _allocate_storage(self, ipc_manager_factory):
        import torch
        from ulacache.transfer.ipc import FlowKVPeerShards
        from ulacache.cache.reader_lifetime import DistributedResidencyPublication

        query_layout = self.layout
        arena_bytes = query_layout.maximum_export_bytes
        local_width = (query_layout.maximum_columns + self.degree - 1) // self.degree
        output_width = (query_layout.maximum_columns + 7) // 8 * 8
        self.exported = torch.empty(arena_bytes, dtype=torch.uint8, device=self.device)
        self.collected = torch.empty(
            self.degree * arena_bytes, dtype=torch.uint8, device=self.device
        )
        self.local_scores = torch.empty(
            self.degree * query_layout.maximum_queries * local_width,
            dtype=torch.float32,
            device=self.device,
        )
        self.native_output = torch.empty(
            query_layout.maximum_queries * output_width,
            dtype=torch.float32,
            device=self.device,
        )
        self.publication = DistributedResidencyPublication(
            1, self.device, self.cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        self.query_peers = FlowKVPeerShards(
            self.exported, self.cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        self.score_peers = FlowKVPeerShards(
            self.local_scores, self.cpu_group, ipc_manager_factory=ipc_manager_factory
        )

    def _exchange_queries(
        self,
        q,
        weights,
        requests,
        lengths,
        *,
        logical_width,
        candidate_mask,
        align_output_rows,
    ):
        import torch

        if self.device.type != "cuda":
            raise RuntimeError("Meta Index scorers are allocation-only")
        import triton
        import ulacache.indexer.kernels.scoring as kernels

        if self._closed:
            raise RuntimeError("Native owner Index scorer is closed")
        if q.ndim != 3 or q.shape[1:] != (
            self.layout.query_heads,
            self.layout.head_dimension,
        ):
            raise ValueError(
                "Native query heads/dimension disagree with the bound service"
            )
        rows = q.shape[0]
        used_bytes = self.layout.export_bytes(
            rows, logical_width, has_mask=candidate_mask is not None
        )
        if (
            q.dtype != torch.bfloat16
            or weights.dtype != torch.bfloat16
            or weights.shape != (rows, self.layout.query_heads)
            or requests.shape != (rows,)
            or lengths.shape != (rows,)
            or requests.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)
            or not requests.is_contiguous()
            or not lengths.is_contiguous()
        ):
            raise ValueError(
                "Native Index needs BF16 queries/head weights and integer metadata"
            )
        tensors = [q, weights, requests, lengths]
        if candidate_mask is not None:
            if (
                candidate_mask.dtype != torch.bool
                or candidate_mask.ndim != 2
                or candidate_mask.shape[0] != rows
                or candidate_mask.shape[1] < logical_width
                or candidate_mask.stride(1) != 1
            ):
                raise ValueError(
                    "Native candidate mask must cover original logical columns"
                )
            tensors.append(candidate_mask)
        if any(tensor.device != self.device for tensor in tensors):
            raise ValueError("Native owner Index tensors must share the bound device")
        local_width = (logical_width + self.degree - 1) // self.degree
        output_width = (
            (logical_width + 7) // 8 * 8 if align_output_rows else logical_width
        )
        output = self.native_output[: rows * output_width].view(rows, output_width)
        if not rows:
            return rows, local_width, output, self.layout.offsets(rows), None
        offsets = self.layout.offsets(rows)
        mask_bytes = (
            rows * ((logical_width + 7) // 8) if candidate_mask is not None else 0
        )
        blocks = max(triton.cdiv(q.numel(), 256), triton.cdiv(mask_bytes, 256))
        # Eager prefill lengths change on every mixed decode/prefill step.
        # Keep those lengths/offsets as kernel arguments while retaining the
        # original static decode specialization. Both entries share one body.
        runtime_prefill = not align_output_rows and not self._specialize_prefill_shapes
        export = (
            kernels.export_queries_runtime
            if runtime_prefill
            else kernels.export_queries
        )
        collect = (
            kernels.collect_queries_runtime
            if runtime_prefill
            else kernels.collect_queries
        )
        export[(blocks,)](
            q,
            weights,
            requests,
            lengths,
            candidate_mask,
            self.exported,
            rows,
            self.layout.query_heads,
            self.layout.head_dimension,
            logical_width,
            *q.stride(),
            *weights.stride(),
            candidate_mask.stride(0) if candidate_mask is not None else 0,
            *offsets,
            candidate_mask is not None,
            256,
            triton.next_power_of_2(rows),
        )
        self.publication.begin_selection(0)
        if (
            align_output_rows
            and self._reuse_equal_decode_queries
            and candidate_mask is None
        ):
            kernels.collect_queries_words[
                (triton.cdiv(self.degree * used_bytes, 1024),)
            ](
                self.query_peers.ptrs,
                self.collected,
                used_bytes,
                self.degree,
                *offsets,
                256,
            )
        else:
            collect[(triton.cdiv(self.degree * used_bytes, 256),)](
                self.query_peers.ptrs,
                self.collected,
                used_bytes,
                self.degree,
                *offsets,
                256,
            )
        flags = None
        # Alias unused bytes, without another allocation or a reduced query
        # bound. Full-arena calls retain the original independent-row path.
        flags_start = self.degree * used_bytes
        flags_end = flags_start + self.degree * rows
        reuse_queries = (
            self._reuse_equal_decode_queries
            if align_output_rows
            else self._reuse_equal_prefill_queries
            and rows * logical_width >= self._minimum_prefill_reuse_cells
        )
        if reuse_queries and flags_end <= self.collected.numel():
            flags = self.collected[flags_start:flags_end]
            identify = (
                kernels.identify_equal_queries
                if align_output_rows
                else kernels.identify_equal_queries_runtime
            )
            identify[(self.degree, rows)](
                self.collected,
                flags,
                rows,
                self.layout.query_heads,
                self.layout.head_dimension,
                logical_width,
                self.degree,
                *offsets,
                candidate_mask is not None,
                triton.next_power_of_2(
                    self.layout.query_heads * self.layout.head_dimension // 2
                    + self.layout.query_heads // 2
                    + 4
                ),
                1024,
                num_warps=4,
            )
        return rows, local_width, output, offsets, flags

    def _finish_scores(
        self,
        lengths,
        output,
        *,
        logical_width,
        local_width,
        prefill=False,
        reuse_flags=None,
    ):
        import triton
        import ulacache.indexer.kernels.scoring as kernels

        rows, output_width = output.shape
        if rows:
            self.publication.publish_residency(0)
            restore = (
                kernels.restore_peer_scores_runtime
                if prefill and not self._specialize_prefill_shapes
                else kernels.restore_peer_scores
            )
            if reuse_flags is None:
                restore[(rows, triton.cdiv(output_width, 256))](
                    self.score_peers.ptrs,
                    lengths,
                    output,
                    rows,
                    logical_width,
                    local_width,
                    output_width,
                    self.degree,
                    self.rank,
                    256,
                )
            else:
                restore_reuse = (
                    kernels.restore_peer_scores_reuse_runtime
                    if prefill
                    else kernels.restore_peer_scores_reuse
                )
                restore_reuse[(rows, triton.cdiv(output_width, 256))](
                    self.score_peers.ptrs,
                    lengths,
                    output,
                    reuse_flags,
                    rows,
                    logical_width,
                    local_width,
                    output_width,
                    self.degree,
                    self.rank,
                    256,
                )
            self.publication.finish_readers(0)
        return output[:, :logical_width]

    def score(
        self,
        q,
        weights,
        requests,
        lengths,
        request_table,
        native_index,
        *,
        ownership,
        logical_width,
        compression_ratio,
        candidate_mask=None,
    ):
        import torch

        if self.device.type != "cuda":
            raise RuntimeError("Meta Index scorers are allocation-only")
        import triton
        import ulacache.indexer.kernels.scoring as kernels

        if (
            not isinstance(ownership, NativeIndexOwnership)
            or ownership.owner_count != self.degree
        ):
            raise ValueError("Native Index storage and score ownership disagree")
        if (
            type(compression_ratio) is not int
            or compression_ratio < 1
            or request_table.ndim != 2
            or request_table.dtype != torch.int32
            or request_table.stride(1) != 1
            or logical_width * compression_ratio > request_table.shape[1]
            or native_index.ndim != 2
            or native_index.dtype != torch.uint8
            or not native_index.is_contiguous()
            or native_index.shape
            != (
                ownership.native_page_count,
                ownership.records_per_native_page * ownership.bytes_per_record,
            )
            or request_table.device != self.device
            or native_index.device != self.device
        ):
            raise ValueError(
                "Native Index/request storage disagrees with the declared geometry"
            )
        rows, local_width, output, offsets, flags = self._exchange_queries(
            q,
            weights,
            requests,
            lengths,
            logical_width=logical_width,
            candidate_mask=candidate_mask,
            align_output_rows=True,
        )
        if not rows:
            return output[:, :logical_width]
        workers = max(
            1,
            min(
                triton.cdiv(local_width, 64),
                triton.cdiv(self._multiprocessors * 4, self.degree * rows),
            ),
        )
        score = (
            kernels.score_local_index_reuse
            if flags is not None
            else kernels.score_local_index
        )
        arguments = (self.collected, request_table, native_index, self.local_scores)
        if flags is not None:
            arguments += (flags,)
        score[(self.degree * rows, workers)](
            *arguments,
            rows,
            self.layout.query_heads,
            self.layout.head_dimension,
            logical_width,
            local_width,
            self.degree,
            self.rank,
            *offsets,
            request_table.stride(0),
            ownership.records_per_native_page,
            native_index.stride(0),
            ownership.logical_record_capacity,
            ownership.reserved_record_count,
            compression_ratio,
            candidate_mask is not None,
            self.native_score_tile,
            64,
            num_warps=4,
        )
        return self._finish_scores(
            lengths,
            output,
            logical_width=logical_width,
            local_width=local_width,
            reuse_flags=flags,
        )

    def score_prefill(
        self,
        q,
        weights,
        requests,
        lengths,
        local_keys,
        *,
        logical_width,
        native_score_fn,
        candidate_mask=None,
    ):
        """Preserve the original Torch prefill math and contiguous top-k input.

        The integration supplies one shared request per call and dequantizes
        only this owner's logical column stripe using the native decoder.
        The original query chunk bound also bounds the gathered-consumer
        native math because each owner scores only one eighth of its keys.
        Prefill transients belong to the native math service and must be
        charged separately from this scorer's persistent inventory. The bound
        native math must score each query row independently. Prefill equality
        checks are skipped below the measured work threshold, avoiding their
        eager synchronization cost for small batches.
        """
        import torch

        if self.device.type != "cuda":
            raise RuntimeError("Meta Index scorers are allocation-only")
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError("Native Torch prefill Index is eager-only")
        if not callable(native_score_fn):
            raise ValueError("Bind the original native Torch prefill score function")
        expected_columns = max(
            0, (logical_width - self.rank + self.degree - 1) // self.degree
        )
        if (
            weights.dtype != torch.bfloat16
            or local_keys.dtype != torch.bfloat16
            or local_keys.device != self.device
            or local_keys.shape != (expected_columns, self.layout.head_dimension)
        ):
            raise ValueError(
                "Native prefill needs BF16 weights and this owner's native key stripe"
            )
        import triton
        import ulacache.indexer.kernels.scoring as kernels

        rows, local_width, output, offsets, flags = self._exchange_queries(
            q,
            weights,
            requests,
            lengths,
            logical_width=logical_width,
            candidate_mask=candidate_mask,
            align_output_rows=False,
        )
        if not rows:
            return output
        # Eager prefill may choose the original native GEMM's input batch.
        # Require exact equality of every exported input before compacting;
        # mixed batches preserve the original independent-consumer call.
        canonical = flags is not None and bool(torch.all(flags).item())
        math_rows = rows if canonical else self.degree * rows
        if expected_columns:
            weights_offset, requests_offset, _, _ = offsets
            queries = (
                self.collected[: self.degree * weights_offset]
                .view(torch.bfloat16)
                .view(
                    self.degree * rows,
                    self.layout.query_heads,
                    self.layout.head_dimension,
                )
            )
            head_weights = (
                self.collected[
                    self.degree * weights_offset : self.degree * requests_offset
                ]
                .view(torch.bfloat16)
                .view(self.degree * rows, self.layout.query_heads)
            )
            native_scores = native_score_fn(
                queries[:math_rows], local_keys, head_weights[:math_rows]
            )
            if (
                native_scores.dtype != torch.float32
                or native_scores.device != self.device
                or native_scores.shape != (math_rows, expected_columns)
                or native_scores.stride(1) != 1
            ):
                raise ValueError(
                    "Native prefill math must return FP32 scores in original row order"
                )
            copy = (
                kernels.copy_canonical_prefill_scores_runtime
                if canonical
                else kernels.copy_prefill_scores
                if self._specialize_prefill_shapes
                else kernels.copy_prefill_scores_runtime
            )
            copy[(triton.cdiv(math_rows * local_width, 256),)](
                native_scores,
                self.collected,
                requests,
                self.local_scores,
                rows,
                logical_width,
                local_width,
                expected_columns,
                native_scores.stride(0),
                self.degree,
                self.rank,
                *offsets[1:],
                candidate_mask is not None,
                256,
            )
        return self._finish_scores(
            lengths,
            output,
            logical_width=logical_width,
            local_width=local_width,
            prefill=True,
            reuse_flags=flags if canonical else None,
        )

    def device_storage_inventory(self):
        return (
            self.exported,
            self.collected,
            self.local_scores,
            self.native_output,
            self.query_peers.ptrs,
            self.score_peers.ptrs,
            self.publication.epochs,
            self.publication.flags,
            self.publication.peer_flags.ptrs,
        )

    def close(self):
        import torch
        import torch.distributed as distributed

        if self._closed:
            return
        self.publication.wait_for_all_readers()
        torch.cuda.synchronize(self.device)
        distributed.barrier(group=self.cpu_group)
        self.query_peers.close()
        self.score_peers.close()
        distributed.barrier(group=self.cpu_group)
        self.publication.close()
        self._closed = True
