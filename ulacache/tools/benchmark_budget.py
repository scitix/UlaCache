# SPDX-License-Identifier: Apache-2.0
"""Scalar benchmark sizing from actual allocator capacity; no device imports."""


def restored_cohort_metadata_tokens(*, requests: int, page_size: int) -> int:
    """Budget native metadata admission for one saved next-token per request.

    The pinned native prefill adder bills a page even for a one-token cache
    hit. One extra page avoids its strict remaining-budget boundary. This
    budget is for metadata preparation only; no prefill forward is permitted.
    """
    if any(type(value) is not int or value < 1 for value in (requests, page_size)):
        raise ValueError("Restored cohort and page size must be positive integers")
    return (requests + 1) * page_size


def hisparse_buffer_rows(
    *,
    physical_rows: int,
    requests: int,
    prefill_rows: int,
    page_size: int = 64,
    topk: int = 2048,
    rank_limit: int = 32768,
) -> int:
    """Maximize the stock request buffer while retaining a full prefill reserve.

    Native HiSparse holds a complete request's prefill until CPU backup ends.
    Completed requests retain their buffer plus one current-token page. Native
    LRU ranks use int16. This function only chooses a native startup parameter;
    it does not change an allocator, scheduler, transfer or kernel.
    """
    values = (physical_rows, requests, prefill_rows, page_size, topk, rank_limit)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("HiSparse capacity inputs must be positive integers")
    reserve = (prefill_rows + page_size - 1) // page_size * page_size
    rows = min(
        (physical_rows - reserve) // (requests * page_size) * page_size - page_size,
        rank_limit // page_size * page_size,
    )
    if rows < topk:
        raise ValueError(
            "Physical HiSparse pool cannot retain TopK and the prefill reserve"
        )
    return rows
