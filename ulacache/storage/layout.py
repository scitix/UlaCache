"""Native KV encoding, page geometry and record ownership."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Literal


@dataclass(frozen=True)
class KeyValueRecordPlane:
    """One transported plane; segments retain the native reader/writer ABI.

    Encoding is an opaque, versioned name supplied by the adapter. Equal byte
    widths alone do not make two encodings interchangeable. Scales and zero
    points must be included in the segments or in another explicit plane.
    """

    name: str
    native_encoding: str
    bytes_per_segment: tuple[int, ...]
    page_alignment_bytes: int = 1
    page_memory_order: Literal["record_major", "segment_major"] = "record_major"

    def __post_init__(self):
        object.__setattr__(self, "bytes_per_segment", tuple(self.bytes_per_segment))
        if not self.name or not self.native_encoding:
            raise ValueError("A record plane needs a name and native encoding")
        if (
            not self.bytes_per_segment
            or any(type(n) is not int or n < 1 for n in self.bytes_per_segment)
            or type(self.page_alignment_bytes) is not int
            or self.page_alignment_bytes < 1
        ):
            raise ValueError(
                "Native segments and page alignment must be positive integers"
            )
        if self.page_memory_order not in ("record_major", "segment_major"):
            raise ValueError("Unknown native page layout")

    @property
    def bytes_per_record(self) -> int:
        return sum(self.bytes_per_segment)

    def bytes_per_page(self, records_per_page: int) -> int:
        if type(records_per_page) is not int or records_per_page < 1:
            raise ValueError("Native page size must be a positive integer")
        size = records_per_page * self.bytes_per_record
        return (
            (size + self.page_alignment_bytes - 1)
            // self.page_alignment_bytes
            * self.page_alignment_bytes
        )

    def calculate_record_byte_offset(
        self, record: int, byte: int, records_per_page: int
    ) -> int:
        """Reference address calculation, including native page tail padding."""
        page_bytes = self.bytes_per_page(records_per_page)
        if record < 0 or not 0 <= byte < self.bytes_per_record:
            raise IndexError("Native record byte outside geometry")
        page, row = divmod(record, records_per_page)
        offset = page * page_bytes
        if self.page_memory_order == "record_major":
            return offset + row * self.bytes_per_record + byte
        for width in self.bytes_per_segment:
            if byte < width:
                return offset + row * width + byte
            byte -= width
            offset += records_per_page * width
        raise AssertionError("A valid byte must belong to a segment")


@dataclass(frozen=True)
class KeyValueStorageLayout:
    """A homogeneous cache tier; separate tiers may have different geometry.

    A canonical logical ID identifies one stored record, including for a
    compressed tier. ``tokens_per_record`` describes its native compression
    unit; it is never used to approximate the model's selection or causality.
    Bounded window caches and recurrent states stay outside this sparse tier.
    """

    name: str
    record_planes: tuple[KeyValueRecordPlane, ...]
    records_per_page: int
    record_kind: Literal["token", "compressed"] = "token"
    tokens_per_record: int = 1

    def __post_init__(self):
        object.__setattr__(self, "record_planes", tuple(self.record_planes))
        if not self.name or not self.record_planes:
            raise ValueError("A storage tier needs a name and record planes")
        if len({p.name for p in self.record_planes}) != len(self.record_planes):
            raise ValueError("Record plane names must be unique within a tier")
        if self.record_kind not in ("token", "compressed"):
            raise ValueError("Unknown sparse record kind")
        if type(self.tokens_per_record) is not int or self.tokens_per_record < 1:
            raise ValueError("Tokens per native record must be positive")
        if self.record_kind == "token" and self.tokens_per_record != 1:
            raise ValueError("Uncompressed token records represent one token")
        for plane in self.record_planes:
            plane.bytes_per_page(self.records_per_page)

    @property
    def bytes_per_record(self) -> int:
        return sum(p.bytes_per_record for p in self.record_planes)

    def calculate_storage_bytes(self, records: int) -> int:
        """Include each plane's native page padding, even for a partial page."""
        if type(records) is not int or records < 0:
            raise ValueError("Record count must be nonnegative")
        pages = (records + self.records_per_page - 1) // self.records_per_page
        return pages * sum(
            p.bytes_per_page(self.records_per_page) for p in self.record_planes
        )


def ceil_div(a, b):
    return (a + b - 1) // b


def history_capacity(max_queries, topk, multiplier, page_size):
    """Page-align the requested resident history without inflating its ratio."""
    if min(max_queries, topk, page_size) < 1:
        raise ValueError("Cache geometry must be positive")
    if not isfinite(multiplier) or multiplier < 1:
        raise ValueError(
            "FlowKV cache capacity multiplier must be finite and at least one"
        )
    slots = ceil_div(ceil(max_queries * topk * multiplier), page_size) * page_size
    if slots >= 2**31:
        raise ValueError("FlowKV resident slots must fit signed int32 indices")
    return slots


@dataclass(frozen=True)
class NativePageLayout:
    page_size: int
    segment_bytes: tuple[int, ...]
    page_alignment: int

    def __post_init__(self):
        object.__setattr__(self, "segment_bytes", tuple(self.segment_bytes))
        if self.page_size < 1 or self.page_alignment < 1:
            raise ValueError("Native page dimensions must be positive")
        if not self.segment_bytes or any(width < 1 for width in self.segment_bytes):
            raise ValueError("Native page segments must be positive byte widths")
        if any(width % 8 for width in self.segment_bytes) or self.bytes_per_page % 8:
            raise ValueError(
                "Record transport requires aligned eight-byte page segments"
            )

    @property
    def row_bytes(self):
        return sum(self.segment_bytes)

    @property
    def bytes_per_page(self):
        return (
            ceil_div(self.page_size * self.row_bytes, self.page_alignment)
            * self.page_alignment
        )

    def byte_offset(self, row, byte):
        if row < 0 or not 0 <= byte < self.row_bytes:
            raise IndexError("Native row/byte outside geometry")
        page, offset = divmod(row, self.page_size)
        row_prefix = page_prefix = 0
        for width in self.segment_bytes:
            if byte < row_prefix + width:
                return (
                    page * self.bytes_per_page
                    + page_prefix
                    + offset * width
                    + byte
                    - row_prefix
                )
            row_prefix += width
            page_prefix += self.page_size * width
        raise AssertionError("A valid byte must belong to a native segment")


@dataclass(frozen=True)
class InterleavedRecordLayout:
    peers: int
    slots_per_rank: int
    native_page: NativePageLayout
    layers: int = 1

    def __post_init__(self):
        if min(self.peers, self.slots_per_rank, self.layers) < 1:
            raise ValueError("Peer count, local slots and layers must be positive")
        if self.global_slots >= 2**31:
            raise ValueError("The inherited cache planner uses signed int32 slot IDs")
        if self.global_slots % self.page_size:
            raise ValueError("Global capacity must be native-page aligned")

    @property
    def global_slots(self):
        return self.peers * self.slots_per_rank

    @property
    def row_bytes(self):
        return self.native_page.row_bytes

    @property
    def page_size(self):
        return self.native_page.page_size

    @property
    def layer_bytes(self):
        return self.slots_per_rank * self.row_bytes

    @property
    def physical_bytes_per_rank(self):
        return self.layers * self.layer_bytes

    @property
    def bytes_per_page(self):
        return self.native_page.bytes_per_page

    def locate(self, gid):
        if not 0 <= gid < self.global_slots:
            raise IndexError("Global KV slot outside capacity")
        return gid // self.slots_per_rank, gid % self.slots_per_rank

    def global_id(self, rank, local_slot):
        if not 0 <= rank < self.peers or not 0 <= local_slot < self.slots_per_rank:
            raise IndexError("Physical KV address outside capacity")
        return rank * self.slots_per_rank + local_slot

    def native_to_global(self, native_row, namespace=0, namespace_rows=None):
        """Compose SGLang's page mapping with position-based storage ownership.

        Native compressed page size must be a multiple of N: the page offset
        then preserves sequence-relative compressed position modulo N. The
        global physical slot number itself is NOT reduced modulo N.
        """
        if native_row <= 0:
            return native_row
        if self.page_size % self.peers:
            raise ValueError("Native pages must preserve relative position modulo N")
        if namespace_rows is None:
            namespace_rows = self.global_slots
        logical = namespace_global_id(native_row, namespace, namespace_rows)
        owner, local = logical % self.peers, logical // self.peers
        return self.global_id(owner, local)

    def native_byte_offset(self, row, byte):
        return self.native_page.byte_offset(row, byte)

    def cache_bytes(self, max_queries, topk=512, multiplier=2):
        slots = history_capacity(max_queries, topk, multiplier, self.page_size)
        rows = ceil_div(slots + max_queries, self.page_size) * self.page_size
        return {
            "history_slots": slots,
            "rows_per_layer": rows,
            "data_bytes": self.layers * rows // self.page_size * self.bytes_per_page,
            "dense_hint_bytes": self.layers * self.global_slots * 4,
        }

    def misses_remote_payload(self, gids, consumer_rank):
        """Effective payload after dedup; excludes protocol/alignment overhead."""
        return sum(
            self.row_bytes for gid in set(gids) if self.locate(gid)[0] != consumer_rank
        )


def namespace_global_id(native_row, namespace, namespace_rows):
    """Bridge existing disjoint DPA request allocators to the shared directory.

    A coordinated global allocator may supply IDs directly. This bridge keeps
    existing per-owner allocation disjoint without moving records on TP/DP
    execution changes. All CP helpers must use the actual request namespace.
    """
    if native_row <= 0:
        return native_row
    if not 0 <= native_row < namespace_rows or namespace < 0:
        raise IndexError("Native row outside its request-owner namespace")
    return namespace * namespace_rows + native_row


def native_mla_encodings_match(left, right):
    """Compare opaque encodings, with one explicitly audited legacy alias.

    Both scaled656 labels mean512 FP8 latent bytes, four FP32 scales and64
    BF16 RoPE values, in that order. No width-based inference or conversion is
    performed; other encoding labels match only themselves.
    """
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return False
    return left == right or {left, right} == {
        "mla.scaled656.fp8.v1",
        "native-scaled-fp8-v32",
    }


@dataclass(frozen=True)
class NativeMLARecord:
    encoding: str
    latent_width: int = 512
    rope_width: int = 64
    quantization_block: int = 128

    def __post_init__(self):
        if self.encoding not in ("raw576", "scaled656"):
            raise ValueError("Select an explicit raw576 or scaled656 native MLA record")
        if (self.latent_width, self.rope_width, self.quantization_block) != (
            512,
            64,
            128,
        ):
            raise NotImplementedError(
                "This binding supports native512 latent,64 RoPE,block128"
            )

    @property
    def segment_bytes(self):
        if self.encoding == "raw576":
            return (self.latent_width, self.rope_width)
        return (
            self.latent_width + self.latent_width // self.quantization_block * 4,
            self.rope_width * 2,
        )

    @property
    def row_bytes(self):
        return sum(self.segment_bytes)

    @property
    def native_encoding(self):
        return f"mla.{self.encoding}.fp8.v1"

    def validate_backends(self, prefill, decode):
        if self.encoding == "raw576":
            if (prefill, decode) != ("trtllm", "trtllm"):
                raise NotImplementedError(
                    "Raw576 requires native TRTLLM in both phases"
                )
        elif decode != "flashmla_kv" or prefill not in (
            "flashmla_sparse",
            "flashmla_kv",
        ):
            raise NotImplementedError(
                "Scaled656 requires native FlashMLA KV decode and compatible prefill"
            )


@dataclass(frozen=True)
class FlowKVPackedMLASpec:
    """Model-provided geometry for a packed, indexed MLA cache tier.

    A packed cache (such as a FlashMLA tier) has a physical KV row that is
    different from ordinary DSA's latent row.  The FlowKV planner does not
    need to know the model family or a tier's marketing name: it only needs
    the rows that use this representation, the index width, and its packing
    geometry.  A model-specific cache adapter constructs this descriptor.
    """

    compression_ratio: int
    indexed_layer_ids: Sequence[int]
    # Complete native selected-output width, including any appended tail.
    topk: int
    row_bytes: int
    native_segment_bytes: tuple[int, ...]
    index_layer_anchors: tuple[int, ...]
    # Some packed formats pad every storage page for a kernel ABI.  This is a
    # property of the packed representation, not of FlowKV's planner.
    page_storage_alignment: int = 1

    def __post_init__(self):
        if self.topk <= 0 or self.topk % 256:
            raise NotImplementedError(
                f"FlowKV planner requires an aligned native selected width: "
                f"received {self.topk}, expected a positive multiple of 256. "
                "Native Index outputs and appended tail columns are preserved."
            )

    def local_layer_anchors(self, start_layer: int, end_layer: int) -> tuple[int, ...]:
        if len(self.index_layer_anchors) != len(self.indexed_layer_ids):
            raise ValueError("Every native KV layer must declare an Index anchor")
        local_ids = tuple(
            layer
            for layer in self.indexed_layer_ids
            if start_layer <= layer < end_layer
        )
        local = {layer: i for i, layer in enumerate(local_ids)}
        owners = dict(zip(self.indexed_layer_ids, self.index_layer_anchors))
        if any(owners[layer] not in local for layer in local_ids):
            raise ValueError("A shared Index group crosses the local KV stage")
        return tuple(local[owners[layer]] for layer in local_ids)

    def local_layer_count(self, start_layer: int, end_layer: int) -> int:
        if start_layer < 0 or end_layer <= start_layer:
            raise ValueError("Invalid packed-MLA layer range")
        if self.page_storage_alignment <= 0:
            raise ValueError("Packed-MLA page storage alignment must be positive")
        return sum(
            start_layer <= layer_id < end_layer for layer_id in self.indexed_layer_ids
        )


@dataclass(frozen=True)
class PositionShardLayout:
    rank: int
    degree: int
    device_records: int
    host_records: int
    page_size: int = 64

    def __post_init__(self):
        values = (
            self.rank,
            self.degree,
            self.device_records,
            self.host_records,
            self.page_size,
        )
        if any(type(value) is not int for value in values):
            raise ValueError("Position-shard geometry must use integers")
        if (
            self.degree < 2
            or self.degree & (self.degree - 1)
            or not 0 <= self.rank < self.degree
        ):
            raise ValueError("CP requires a power-of-two owner group and a valid rank")
        if (
            self.page_size != 64
            or self.device_records < self.page_size
            or self.host_records < 0
        ):
            raise ValueError("CP requires native page64 and a reserved GPU page")
        if self.device_records % self.page_size or self.host_records % self.page_size:
            raise ValueError(
                "Both exclusive storage tiers must contain complete native pages"
            )
        if self.global_records >= 2**31:
            raise ValueError("The native global slot namespace must fit signed int32")

    @property
    def local_records(self):
        return self.device_records + self.host_records

    @property
    def global_records(self):
        return self.degree * self.local_records

    @property
    def allocation_page_size(self):
        return self.degree * self.page_size

    @property
    def usable_records(self):
        return self.global_records - self.allocation_page_size

    def owner(self, global_slot):
        if type(global_slot) is not int or not 0 <= global_slot < self.global_records:
            raise ValueError("Global slot is outside this allocator namespace")
        return global_slot % self.degree

    def local_slot(self, global_slot):
        """Non-local and native-zero rows are sentinels, never remote reads."""
        owner = self.owner(global_slot)
        local = global_slot // self.degree
        return local if owner == self.rank and local > 0 else -1

    def local_length(self, length, *, start=0):
        if type(length) is not int or type(start) is not int or min(length, start) < 0:
            raise ValueError("Sequence frontiers must be nonnegative integers")

        def before(end):
            return end // self.degree + int(end % self.degree > self.rank)

        return before(start + length) - before(start)
