"""Explicit experimental score storage, independent of logical TopK columns.

This is not a Tensor facade. Its consumer must opt into the storage layout;
native logical scores continue through their original Tensor-only path.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RankMajorScoreLayout:
    gathered: Any
    logical_columns: int
    degree: int
    score_page_size: int = 64
    init_tokens: int = 0
    recent_tokens: int = 0
    compact_query_offsets: Any = None
    compact_input_packs: Any = None

    def __post_init__(self):
        if (self.compact_query_offsets is None) != (self.compact_input_packs is None):
            raise ValueError("Compact rank-major scores require both GPU descriptors")
        if self.compact_query_offsets is not None:
            import torch

            prefix, packs = self.compact_query_offsets, self.compact_input_packs
            if (
                prefix.shape != (self.queries + 1,)
                or prefix.dtype != torch.int32
                or packs.shape != (1,)
                or packs.dtype != torch.int64
                or prefix.device != self.gathered.device
                or packs.device != self.gathered.device
                or not prefix.is_contiguous()
                or not packs.is_contiguous()
                or self.score_page_size != 1
            ):
                raise ValueError("Invalid compact token-striped score descriptors")
        if self.degree < 1 or self.degree & (self.degree - 1):
            raise ValueError("Rank-major degree must be a power of two")
        if (
            self.score_page_size != 1
            and self.score_page_size < 4
            or self.score_page_size & (self.score_page_size - 1)
        ):
            raise ValueError(
                "Rank-major score ownership unit must be 1 or a power of two >=4"
            )
        if self.gathered.ndim != 3 or self.gathered.shape[0] != self.degree:
            raise ValueError("Rank-major scores require explicit [P,Q,L] storage")
        if not self.gathered.is_contiguous() or self.gathered.data_ptr() % 16:
            raise ValueError(
                "Rank-major gathered storage must be contiguous and 16B aligned"
            )
        if self.queries <= 0 or self.local_columns % self.score_page_size:
            raise ValueError("Invalid rank-major query/local-page shape")
        if not 0 < self.logical_columns <= self.degree * self.local_columns:
            raise ValueError("Invalid original logical score width")
        if self.init_tokens < 0 or self.recent_tokens < 0:
            raise ValueError("Native mask counts must be nonnegative")

    @property
    def queries(self):
        return self.gathered.shape[1]

    @property
    def local_columns(self):
        return self.gathered.shape[2]

    def set_native_mask(self, *, lengths, init_tokens, recent_tokens, row_starts):
        """Record the original decode policy for the fused input reader.

        No score writes are required. Validity and tie keys still use logical
        columns and original lengths. Extend windows require a separate ABI.
        """
        if row_starts is not None or lengths.shape != (self.queries,):
            raise RuntimeError(
                "Rank-major prototype accepts original decode mask rows only"
            )
        if init_tokens < 0 or recent_tokens < 0:
            raise ValueError("Native mask counts must be nonnegative")
        self.init_tokens = init_tokens
        self.recent_tokens = recent_tokens
        return self
