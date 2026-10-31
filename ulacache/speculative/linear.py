# SPDX-License-Identifier: Apache-2.0
"""Contracts for native linear MTP target verification at configured lengths.

The framework retains drafting, sampling, acceptance, and allocation. UDCache
consumes its expanded causal query rows through the existing decode arena.
No second scheduler or speculative KV identity space is introduced here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearVerifyConfig:
    algorithm: str
    steps: int
    topk: int
    verify_tokens: int
    adaptive: bool = False

    def validate(self) -> None:
        if (
            not isinstance(self.algorithm, str)
            or self.algorithm.upper() != "EAGLE"
            or self.adaptive
            or any(
                type(value) is not int
                for value in (self.steps, self.topk, self.verify_tokens)
            )
            or self.steps < 0
            or self.topk != 1
            or not 1 <= self.verify_tokens <= self.steps + 1
        ):
            raise NotImplementedError(
                "UDCache MTP requires a native EAGLE linear chain, topk=1, "
                "nonnegative draft steps, 1 <= verify tokens <= steps + 1, "
                "and no adaptive speculation"
            )


def is_cache_step(mode) -> bool:
    """Select existing publication/planning for decode, idle, or linear verify.

    This does not mutate the native forward mode: attention metadata, sampling,
    and rejection still observe TARGET_VERIFY. Startup must validate the linear
    configuration before exposing this path to a framework worker.
    """
    return mode.is_decode_or_idle() or mode.is_target_verify()


def draft_kv_cell_bytes(*, row_bytes: int, layers: int) -> int:
    """Charge complete native draft records in the shared logical slot space.

    The draft does not own UDCache's striped target storage. Inferring its cost
    by dividing the target's per-token cost by target layers undercounts it.
    Native Index bytes remain separately charged by the framework configurator.
    """
    if any(type(value) is not int or value < 1 for value in (row_bytes, layers)):
        raise ValueError("Native draft record bytes and layer count must be positive")
    return row_bytes * layers


def index_query_metadata(mode, metadata, *, max_queries=None):
    """Return original per-query page rows and causal lengths without copying.

    Native verification already repeats each request's page row and expands its
    lengths, e.g. history100 with four query rows -> lengths101,102,103,104.
    Using cache_seqlens_int32 here would expose later speculative slots to the
    first query. No device values are read on CPU or new tensors allocated.
    """
    table = metadata.real_page_table
    draft_extend = getattr(mode, "is_draft_extend_v2", None)
    expanded = mode.is_target_verify() or (callable(draft_extend) and draft_extend())
    lengths = (
        metadata.dsa_seqlens_expanded if expanded else metadata.cache_seqlens_int32
    )
    if len(table.shape) != 2 or lengths.shape != (table.shape[0],):
        raise ValueError("Distributed Index requires one causal length per query row")
    if max_queries is not None and table.shape[0] > max_queries:
        raise ValueError("Native query rows exceed the resident KV capacity bound")
    return table, lengths
