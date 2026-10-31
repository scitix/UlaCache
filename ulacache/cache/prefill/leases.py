"""Temporary physical-page leases for prefill readers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrefillPage:
    physical_page: int
    references: int = 0


class PrefillPageLeases:
    def __init__(self):
        self.pages = {}
        self.requests = {}

    def acquire(self, request, logical_to_physical):
        """Acquire each logical page at most once for a request.

        Validate the entire update before mutating so a stale physical alias
        cannot leave a partially acquired request behind.
        """
        items = dict(logical_to_physical)
        for logical, physical in items.items():
            if type(logical) is not int or type(physical) is not int:
                raise ValueError("Page identities must be integers")
            if logical <= 0 or physical <= 0:
                raise ValueError("Page zero is reserved for padding")
            existing = self.pages.get(logical)
            if existing is not None and existing.physical_page != physical:
                raise ValueError("A live prefix page changed physical ownership")
        owned = self.requests.setdefault(request, set())
        acquired = []
        for logical, physical in items.items():
            if logical in owned:
                continue
            page = self.pages.setdefault(logical, PrefillPage(physical))
            page.references += 1
            owned.add(logical)
            acquired.append(logical)
        return acquired

    def release(self, request, logical_pages=None):
        """Return physical pages whose last prefill reader just finished."""
        owned = self.requests.get(request)
        if owned is None:
            return {}
        selected = set(owned) if logical_pages is None else set(logical_pages)
        if not selected.issubset(owned):
            raise ValueError("A request cannot release another reader's pages")
        freed = {}
        for logical in selected:
            page = self.pages[logical]
            page.references -= 1
            owned.remove(logical)
            if page.references == 0:
                freed[logical] = page.physical_page
                del self.pages[logical]
        if not owned:
            del self.requests[request]
        return freed

    def assert_reclaimable(self, logical_pages):
        if any(page in self.pages for page in logical_pages):
            raise ValueError("A logical prefix page still has active prefill readers")


def prefill_physical_tokens(
    *,
    prefix_tokens,
    held_prefix_tokens,
    extend_tokens,
    hot_tokens,
    scratch_tokens,
    page_size,
):
    """Physical GPU demand, independent of reclaimable logical host pages."""
    if page_size <= 0 or any(
        x < 0
        for x in (
            prefix_tokens,
            held_prefix_tokens,
            extend_tokens,
            hot_tokens,
            scratch_tokens,
        )
    ):
        raise ValueError("Prefill memory geometry must be nonnegative")

    def aligned(n):
        return -(-n // page_size) * page_size

    return (
        aligned(max(prefix_tokens - held_prefix_tokens, 0))
        + aligned(extend_tokens)
        + hot_tokens
        + scratch_tokens
        + page_size
    )
