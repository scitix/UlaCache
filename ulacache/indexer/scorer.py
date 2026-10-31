"""Native paged Index scoring with one rank-major score all-gather.

The attention backend owns the forward epoch and the private communication
group. Query heads, quantized inputs, native Index pages and TopK stay native.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl

from ulacache.indexer.score_layout import RankMajorScoreLayout
from ulacache.indexer.plan import CPScorePlan


@triton.jit
def _make_local_page_table(
    ORIGINAL,
    LENGTHS,
    LOCAL,
    LOCAL_LENGTHS,
    SCHEDULE_LENGTHS,
    ORIGINAL_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    LOCAL_WIDTH: tl.constexpr,
    RANK: tl.constexpr,
    DEGREE: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGE_MAP=None,
    REQUEST_MAP: tl.constexpr = False,
    GPU_PAGES: tl.constexpr = 0,
    COMPACT_PREFIX=None,
    COMPACT_PACKS=None,
    QUERIES: tl.constexpr = 0,
    COMPACT: tl.constexpr = False,
    PREFIX_BLOCK: tl.constexpr = 1,
):
    row = tl.program_id(0)
    local_col = tl.arange(0, BLOCK)
    # Each allocation superpage contains 64 * DEGREE native token slots.
    # Token striping leaves one complete native page64 on each owner.
    logical_col = local_col * DEGREE
    length = tl.load(LENGTHS + row)
    live = (logical_col < WIDTH) & (logical_col * 64 + RANK < length)
    physical_page = tl.load(
        ORIGINAL + row * ORIGINAL_STRIDE + logical_col, live, other=0
    )
    if REQUEST_MAP:
        physical_page = tl.load(PAGE_MAP + physical_page // DEGREE, live, other=0)
        tl.device_assert(
            (~live) | ((physical_page >= 0) & (physical_page < GPU_PAGES)),
            "request CP Index page is not GPU resident",
        )
        physical_page *= DEGREE
    tl.store(
        LOCAL + row * LOCAL_WIDTH + local_col,
        physical_page // DEGREE,
        local_col < LOCAL_WIDTH,
    )
    local_length = tl.maximum(0, (length + DEGREE - 1 - RANK) // DEGREE)
    tl.store(LOCAL_LENGTHS + row, local_length)
    # Native scheduling needs positive work bounds; logits and TopK retain
    # the actual lengths, including empty local partitions.
    tl.store(SCHEDULE_LENGTHS + row, tl.maximum(local_length, 1))
    if COMPACT and row == 0:
        q = tl.arange(0, PREFIX_BLOCK)
        global_lengths = tl.maximum(tl.load(LENGTHS + q, q < QUERIES, other=0), 0)
        widths = tl.cdiv(tl.cdiv(global_lengths, DEGREE), 4) * 4
        dense = tl.sum(widths).to(tl.int64) * 100 >= QUERIES * LOCAL_WIDTH * 64 * 80
        widths = tl.where(dense & (q < QUERIES), LOCAL_WIDTH * 64, widths)
        tl.store(COMPACT_PREFIX, 0)
        tl.store(COMPACT_PREFIX + q + 1, tl.cumsum(widths), q < QUERIES)
        tl.store(COMPACT_PACKS, tl.sum(widths) // 4)


@dataclass(frozen=True)
class CPScoreBucket:
    queries: int
    page_width: int
    local_pages: torch.Tensor
    local_lengths: torch.Tensor
    schedule_lengths: torch.Tensor
    gathered: torch.Tensor
    compact_prefix: torch.Tensor | None = None
    compact_packs: torch.Tensor | None = None

    @property
    def local_columns(self) -> int:
        return self.local_pages.shape[1] * 64


class CPIndexScorer:
    """Serial scorer workspace shared by native Index layers in one forward.

    Register physical Graph buckets before capture. Prepare once for the exact
    native metadata epoch; all Index anchors consume that plan on the same
    stream. The caller must finish TopK before the next score overwrites it.
    """

    def __init__(
        self,
        *,
        rank,
        degree,
        max_queries,
        max_page_width,
        group,
        device,
        owner_page_map=None,
        owner_gpu_pages=0,
    ):
        for value in (degree, max_queries, max_page_width):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("CP score bounds must be positive integers")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or degree & (degree - 1)
            or not 0 <= rank < degree
        ):
            raise ValueError("Invalid CP rank or non-power-of-two degree")
        self.rank = rank
        self.degree = degree
        self.max_queries = max_queries
        self.max_page_width = max_page_width
        requested_device = torch.device(device)
        self.device = group.device
        self.group = group
        self.owner_gpu_pages = owner_gpu_pages
        self.owner_page_map = owner_page_map
        if owner_page_map is not None and (
            owner_page_map.dtype != torch.int32
            or owner_page_map.device != self.device
            or owner_page_map.ndim != 1
            or not owner_page_map.is_contiguous()
            or owner_gpu_pages <= 0
        ):
            raise ValueError("Request CP requires a stable local int32 owner-page map")
        if (
            group.rank_in_group != rank
            or group.world_size != degree
            or requested_device.type != self.device.type
            or (
                requested_device.index is not None
                and requested_device.index != self.device.index
            )
            or group.max_queries != max_queries
            or group.max_page_width != max_page_width
        ):
            raise ValueError("CP scorer and private Index owner geometry disagree")
        self.gathered_workspace = group.gathered_workspace
        self.compact_scores = group.compact_scores
        self.compact_prefix_workspace = (
            torch.empty(
                max_queries + 1,
                dtype=torch.int32,
                device=self.device,
            )
            if self.compact_scores
            else None
        )
        self.compact_packs_workspace = (
            torch.empty(
                1,
                dtype=torch.int64,
                device=self.device,
            )
            if self.compact_scores
            else None
        )
        max_local_width = triton.cdiv(triton.cdiv(max_page_width, degree), 4) * 4
        self.local_pages_workspace = torch.empty(
            max_queries * max_local_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.local_lengths_workspace = torch.empty(
            max_queries,
            dtype=torch.int32,
            device=self.device,
        )
        self.schedule_lengths_workspace = torch.empty(
            max_queries,
            dtype=torch.int32,
            device=self.device,
        )
        self.buckets = {}
        self.captured_plans = []

    def register_bucket(self, queries: int, page_width: int) -> CPScoreBucket:
        if (
            not isinstance(queries, int)
            or isinstance(queries, bool)
            or not isinstance(page_width, int)
            or isinstance(page_width, bool)
            or not 1 <= queries <= self.max_queries
            or not 1 <= page_width <= self.max_page_width
        ):
            raise ValueError("Invalid CP physical Graph query/page shape")
        key = (queries, page_width)
        if key not in self.buckets:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Register CP Graph buckets before capture")
            local_width = triton.cdiv(triton.cdiv(page_width, self.degree), 4) * 4
            self.buckets[key] = CPScoreBucket(
                queries,
                page_width,
                self.local_pages_workspace[: queries * local_width].view(
                    queries, local_width
                ),
                self.local_lengths_workspace[:queries].view(queries, 1),
                self.schedule_lengths_workspace[:queries].view(queries, 1),
                self.gathered_workspace[
                    : self.degree * queries * local_width * 64
                ].view(self.degree * queries, local_width * 64),
                self.compact_prefix_workspace[: queries + 1]
                if self.compact_scores
                else None,
                self.compact_packs_workspace,
            )
        return self.buckets[key]

    def prepare_forward(
        self,
        *,
        page_table: torch.Tensor,
        lengths: torch.Tensor,
        schedule_builder: Callable,
        epoch: object,
    ) -> CPScorePlan:
        if page_table.ndim != 2:
            raise ValueError("CP requires a native two-dimensional page table")
        queries = page_table.shape[0]
        if (
            page_table.dtype != torch.int32
            or page_table.stride(1) != 1
            or lengths.shape != (queries,)
            or lengths.dtype != torch.int32
            or not lengths.is_contiguous()
            or page_table.device != self.device
            or lengths.device != self.device
        ):
            raise ValueError("CP requires original native int32 page table and lengths")
        bucket = self.buckets.get((queries, page_table.shape[1]))
        if bucket is None or epoch is None:
            raise RuntimeError(
                "CP requires a registered bucket and native forward epoch"
            )
        local_width = bucket.local_pages.shape[1]
        _make_local_page_table[(queries,)](
            page_table,
            lengths,
            bucket.local_pages,
            bucket.local_lengths,
            bucket.schedule_lengths,
            page_table.stride(0),
            bucket.page_width,
            local_width,
            self.rank,
            self.degree,
            triton.next_power_of_2(local_width),
            self.owner_page_map,
            self.owner_page_map is not None,
            self.owner_gpu_pages,
            bucket.compact_prefix,
            bucket.compact_packs,
            queries,
            self.compact_scores,
            triton.next_power_of_2(queries),
        )
        capturing = torch.cuda.is_current_stream_capturing()
        # Native metadata owns cp_index_plan. Eager plans must borrow that
        # metadata, otherwise the back-reference retains old GPU page tables
        # until cyclic GC. Captured plans intentionally retain device lifetimes.
        plan = CPScorePlan(
            bucket,
            lengths,
            schedule_builder(bucket.schedule_lengths),
            epoch,
            retain_epoch=capturing,
        )
        if capturing:
            self.captured_plans.append(plan)
        return plan

    def score(
        self,
        *,
        q,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        plan: CPScorePlan,
        epoch: object,
        score_fn: Callable,
        init_tokens: int = 0,
        recent_tokens: int = 0,
    ) -> RankMajorScoreLayout:
        if init_tokens < 0 or recent_tokens < 0:
            raise ValueError(
                "Native init/recent token policy counts must be nonnegative"
            )
        tensors = q if isinstance(q, tuple) else (q,)
        if len(tensors) not in (1, 2) or not all(
            isinstance(tensor, torch.Tensor) for tensor in tensors
        ):
            raise ValueError(
                "Native Q must be a tensor or quantized tensor/scale tuple"
            )
        primary = tensors[0]
        bucket = plan.bucket
        if (
            primary.ndim != 4
            or primary.shape[:2] != (bucket.queries, 1)
            or weights.shape != (bucket.queries, primary.shape[2])
        ):
            raise ValueError(
                "CP supports native single-token decode Q and all query heads"
            )
        if (
            plan.epoch is not epoch
            or epoch is None
            or self.buckets.get((bucket.queries, bucket.page_width)) is not bucket
        ):
            raise RuntimeError("CP plan does not belong to this scorer/forward epoch")
        if any(
            tensor.device != self.device for tensor in (*tensors, kv_cache, weights)
        ):
            raise ValueError("Native score inputs and CP workspace must share a device")
        local = score_fn(
            q,
            kv_cache,
            weights,
            bucket.local_lengths,
            bucket.local_pages,
            plan.schedule,
            bucket.local_columns,
            False,
        )
        if (
            local.shape != (bucket.queries, bucket.local_columns)
            or local.dtype != torch.float32
            or local.device != self.device
            or not local.is_contiguous()
        ):
            raise RuntimeError(
                "Native scorer must return contiguous FP32 local score rows"
            )
        if self.compact_scores:
            self.group.all_gather_compact(
                bucket.gathered, local, bucket.compact_prefix, bucket.compact_packs
            )
        else:
            self.group.all_gather_into_tensor(bucket.gathered, local)
        return RankMajorScoreLayout(
            gathered=bucket.gathered.view(
                self.degree, bucket.queries, bucket.local_columns
            ),
            logical_columns=bucket.page_width * 64,
            degree=self.degree,
            score_page_size=1,
            init_tokens=init_tokens,
            recent_tokens=recent_tokens,
            compact_query_offsets=bucket.compact_prefix,
            compact_input_packs=bucket.compact_packs,
        )
