"""Execution and transfer options for the public cache client."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PrefetchPolicy:
    """Startup-only asynchronous layer prefetch configuration.

    Auto completion uses per-layer events for native pages and ordinary TP,
    and group events for contiguous local paths. Sharing is declared in the
    cache specification; this policy never invents a future TopK selection.
    """

    enabled: bool = False
    completion: Literal["auto", "group", "layer"] = "auto"

    def __post_init__(self):
        if type(self.enabled) is not bool or self.completion not in (
            "auto",
            "group",
            "layer",
        ):
            raise ValueError(
                "Prefetch requires a boolean enabled flag and auto/group/layer completion"
            )

    def __bool__(self):
        return self.enabled


@dataclass(frozen=True)
class CacheExecutionOptions:
    """Preserve the baseline's launch dimensions and transfer ordering."""

    expire_residency_in_parallel: bool = True
    prepare_residency_in_parallel: bool = False
    prepare_current_records_in_parallel: bool = False
    fuse_native_record_publication: bool = False
    select_victims_in_parallel: bool = False
    copy_follower_records_asynchronously: bool = False
    exchange_launch_dimensions: tuple[int, int] = (4, 1024)
    follower_exchange_launch_dimensions: tuple[int, int] | None = None
    follower_copy_mode: str = "group"
    prefill_exchange_launch_dimensions: tuple[int, int] | None = None
    prefill_union_method: str = "sort"
    record_storage_backend: str = "contiguous_records"
    reuse_attention_workspaces: bool = False
    prefetch: PrefetchPolicy | None = None

    @classmethod
    def native_paged(
        cls,
        *,
        prepare_in_parallel=False,
        prefill_union_method="device",
        reuse_attention_workspaces=False,
        prefetch=None,
    ):
        """Keep native paged preparation and publication options consistent."""
        return cls(
            record_storage_backend="native_paged",
            prefill_union_method=prefill_union_method,
            prepare_residency_in_parallel=prepare_in_parallel,
            select_victims_in_parallel=prepare_in_parallel,
            prepare_current_records_in_parallel=prepare_in_parallel,
            fuse_native_record_publication=prepare_in_parallel,
            reuse_attention_workspaces=reuse_attention_workspaces,
            prefetch=prefetch,
        )

    def __post_init__(self):
        if self.prefetch is not None:
            if not isinstance(self.prefetch, PrefetchPolicy):
                raise TypeError("Bind a PrefetchPolicy at initialization")
            completion = self.prefetch.completion
            if completion == "auto":
                completion = (
                    "layer"
                    if self.record_storage_backend
                    in ("native_paged", "tensor_parallel")
                    else "group"
                )
            object.__setattr__(
                self, "copy_follower_records_asynchronously", self.prefetch.enabled
            )
            object.__setattr__(
                self,
                "follower_copy_mode",
                completion if self.prefetch.enabled else "group",
            )
        if type(self.reuse_attention_workspaces) is not bool:
            raise ValueError("Attention workspace reuse must be boolean")
        if (
            self.reuse_attention_workspaces
            and self.record_storage_backend != "native_paged"
        ):
            raise ValueError("Attention workspace reuse requires native-paged storage")
        if type(self.fuse_native_record_publication) is not bool:
            raise ValueError("Fused native publication must be boolean")
        if (
            self.fuse_native_record_publication
            and self.record_storage_backend != "native_paged"
        ):
            raise ValueError("Fused native publication requires native-paged storage")
        if type(self.prepare_current_records_in_parallel) is not bool:
            raise ValueError("Parallel current preparation must be boolean")
        if self.prepare_current_records_in_parallel and (
            not self.prepare_residency_in_parallel
            or self.record_storage_backend not in ("native_paged", "contiguous_records")
        ):
            raise ValueError(
                "Parallel current preparation requires supported parallel residency"
            )


@dataclass(frozen=True)
class CacheOptions:
    """Native TP options; CPU residency and its replacement policy are opt-in.

    With ``cpu_offload=False``, CPU sizing and host launch settings do not
    construct a host tier. ``local_eviction_policy`` independently controls
    the GPU attention-buffer cache, including when all owner KV is on GPU.
    """

    cpu_offload: bool = False
    cpu_slots: int = 0
    cpu_ratio: float = 2.0
    cpu_policy: Literal["index", "request"] = "index"
    prefetch: bool | PrefetchPolicy = False
    fused_topk: bool = False
    local_eviction_policy: Literal["circular", "age_lru"] = "age_lru"
    local_lru_histogram_blocks: int | None = None
    host_exchange_blocks: int = 4
    host_exchange_threads: int = 1024
    host_parallel_finish: bool = False
    host_parallel_victims: bool = False
    low_query_pull_blocks: int | None = None

    def __post_init__(self):
        if isinstance(self.prefetch, dict):
            object.__setattr__(self, "prefetch", PrefetchPolicy(**self.prefetch))
        if (
            not isinstance(self.prefetch, PrefetchPolicy)
            and type(self.prefetch) is not bool
        ):
            raise ValueError("Native TP prefetch must be boolean or a PrefetchPolicy")
        if type(self.cpu_offload) is not bool:
            raise ValueError("CPU offload must be boolean")
        if self.cpu_policy not in ("index", "request"):
            raise ValueError("CPU residency policy must be 'index' or 'request'")
        if self.cpu_slots < 0 or self.cpu_ratio <= 0:
            raise ValueError("CPU slot count must be nonnegative and ratio positive")
        if self.local_eviction_policy not in ("circular", "age_lru"):
            raise ValueError("Local eviction policy must be circular or age_lru")
        if self.local_eviction_policy != "circular" and self.fused_topk:
            raise NotImplementedError("Local LRU requires standalone native lookup")
        blocks = self.local_lru_histogram_blocks
        if blocks is not None and (
            type(blocks) is not int or not 32 <= blocks <= 2048 or blocks & (blocks - 1)
        ):
            raise ValueError(
                "Local LRU histogram blocks must be a power of two in32..2048"
            )
        if (
            type(self.host_exchange_blocks) is not int
            or not 0 < self.host_exchange_blocks < 2**31
            or type(self.host_exchange_threads) is not int
            or not 0 < self.host_exchange_threads <= 1024
            or self.host_exchange_threads % 32
        ):
            raise ValueError(
                "Host exchange needs positive blocks and 32..1024 warp-aligned threads"
            )
        if (
            type(self.host_parallel_finish) is not bool
            or type(self.host_parallel_victims) is not bool
        ):
            raise ValueError("Host parallel finish and victim options must be boolean")
        cap = self.low_query_pull_blocks
        if cap is not None:
            if type(cap) is not int or not 16 <= cap <= 512 or cap & (cap - 1):
                raise ValueError(
                    "Low-query pull blocks must be a power of two in16..512"
                )
            if not self.prefetch:
                raise NotImplementedError(
                    "Low-query pull tuning requires native prefetch"
                )
        if self.host_parallel_victims and not self.host_parallel_finish:
            raise ValueError("Parallel host victims require parallel host finish")
