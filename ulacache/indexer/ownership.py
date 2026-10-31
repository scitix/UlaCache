"""Token-striped ownership for a native Index's shared logical namespace."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeIndexQueryLayout:
    """Byte bounds for BF16 query exports, native metadata and packed masks."""

    maximum_queries: int
    maximum_columns: int
    query_heads: int
    head_dimension: int

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (
                self.maximum_queries,
                self.maximum_columns,
                self.query_heads,
                self.head_dimension,
            )
        ):
            raise ValueError("Native Index query bounds must be positive integers")
        if (
            self.query_heads < 4
            or self.query_heads & (self.query_heads - 1)
            or self.head_dimension % 2
        ):
            raise ValueError(
                "Declare at least four power-of-two heads and an even head dimension"
            )

    def offsets(self, queries):
        if type(queries) is not int or not 0 <= queries <= self.maximum_queries:
            raise ValueError("Native Index queries exceed their startup bound")
        weights = queries * self.query_heads * self.head_dimension * 2
        requests = weights + queries * self.query_heads * 2
        lengths = requests + queries * 8
        masks = lengths + queries * 8
        return weights, requests, lengths, masks

    def export_bytes(self, queries, columns, *, has_mask):
        if type(columns) is not int or not 0 < columns <= self.maximum_columns:
            raise ValueError("Native Index columns exceed their startup bound")
        return self.offsets(queries)[-1] + (
            queries * ((columns + 7) // 8) if has_mask else 0
        )

    @property
    def maximum_export_bytes(self):
        return self.export_bytes(
            self.maximum_queries, self.maximum_columns, has_mask=True
        )

    def storage_bytes(self, degree):
        """All scorer tensors, peer tables and publication storage, counted once."""
        if type(degree) is not int or degree < 2 or degree & (degree - 1):
            raise ValueError("Index communication degree must be a power of two")
        local_columns = (self.maximum_columns + degree - 1) // degree
        native_columns = (self.maximum_columns + 7) // 8 * 8
        return (
            (degree + 1) * self.maximum_export_bytes
            + 4 * degree * self.maximum_queries * local_columns
            + 4 * self.maximum_queries * native_columns
            + 24 * degree
            + 32
        )


@dataclass(frozen=True)
class NativeIndexOwnership:
    """Preserve logical IDs while storing each native Index record once.

    The serving allocator keeps its original IDs. Consecutive records are
    striped over owners; local physical storage rounds to native pages. The
    reserved prefix is allocator padding and is never a live Index record.
    Each owner reserves the same physical shape, including an uneven tail.
    """

    logical_record_capacity: int
    reserved_record_count: int
    owner_count: int
    records_per_native_page: int
    native_padding_pages: int
    bytes_per_record: int

    def __post_init__(self):
        values = (
            self.logical_record_capacity,
            self.reserved_record_count,
            self.owner_count,
            self.records_per_native_page,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("Index ownership dimensions must be positive integers")
        if self.owner_count < 2 or self.owner_count & (self.owner_count - 1):
            raise ValueError("Native Index owners require a power-of-two group")
        if self.records_per_native_page & (self.records_per_native_page - 1):
            raise ValueError("Native Index page size must be a power of two")
        if (
            not self.owner_count
            <= self.reserved_record_count
            < self.logical_record_capacity
        ):
            raise ValueError("Reserve allocator padding on every Index owner")
        if self.reserved_record_count % self.owner_count:
            raise ValueError("Allocator padding must cover complete owner stripes")
        if self.logical_record_capacity > 2**31:
            raise ValueError("Native Index record IDs must fit signed int32")
        if type(self.native_padding_pages) is not int or self.native_padding_pages < 0:
            raise ValueError("Declare the native allocator padding-page charge")
        if type(self.bytes_per_record) is not int or self.bytes_per_record < 1:
            raise ValueError("Declare the complete encoded native Index record bytes")

    @property
    def local_record_capacity(self):
        return (self.logical_record_capacity + self.owner_count - 1) // self.owner_count

    @property
    def local_reserved_record_count(self):
        return self.reserved_record_count // self.owner_count

    @property
    def native_page_count(self):
        return (
            self.local_record_capacity + self.records_per_native_page - 1
        ) // self.records_per_native_page + self.native_padding_pages

    @property
    def storage_bytes_per_bank(self):
        return (
            self.native_page_count
            * self.records_per_native_page
            * self.bytes_per_record
        )

    def live_record_count(self, owner_rank):
        """Unique live records on this owner, excluding the allocator prefix."""
        self.validate_rank(owner_rank)
        total = max(
            0, (self.logical_record_capacity - 1 - owner_rank) // self.owner_count + 1
        )
        return total - self.local_reserved_record_count

    def validate_rank(self, owner_rank):
        if type(owner_rank) is not int or not 0 <= owner_rank < self.owner_count:
            raise ValueError("Invalid native Index owner rank")

    def owner_and_local_record(self, logical_record):
        """CPU admission/debug translation; device paths use the same striping."""
        if (
            type(logical_record) is not int
            or not self.reserved_record_count
            <= logical_record
            < self.logical_record_capacity
        ):
            raise ValueError(
                "Index record must be live in the shared allocator namespace"
            )
        return logical_record % self.owner_count, logical_record // self.owner_count

    def logical_record(self, owner_rank, local_record):
        self.validate_rank(owner_rank)
        if (
            type(local_record) is not int
            or local_record < self.local_reserved_record_count
        ):
            raise ValueError("Local Index record is allocator padding")
        logical = local_record * self.owner_count + owner_rank
        if logical >= self.logical_record_capacity:
            raise ValueError(
                "Local Index record exceeds the shared allocator namespace"
            )
        return logical


class NativeIndexWriteMapping:
    """Bound native write IDs before calling a masked native writer service.

    Construction allocates the reusable ID workspace before Graph capture.
    Unowned records and positive allocator padding map to -1. Native slot zero
    remains replicated once per write batch so padded queries retain a guard
    record without duplicate copy destinations. Live write IDs must be unique.
    The integration must bind a writer which skips negative locations;
    an unmasked native scatter cannot consume this mapping. Quantization,
    RoPE and packed bytes stay native. Guard zero never adds live history.
    """

    def __init__(
        self,
        ownership,
        *,
        owner_rank,
        maximum_write_records,
        device,
        native_writer_supports_negative_locations,
    ):
        if not isinstance(ownership, NativeIndexOwnership):
            raise TypeError("Bind explicit native Index ownership")
        ownership.validate_rank(owner_rank)
        if type(maximum_write_records) is not int or maximum_write_records < 1:
            raise ValueError("Declare a positive native Index write bound")
        if native_writer_supports_negative_locations is not True:
            raise ValueError("Native Index scatter must skip negative mapped locations")
        import torch

        self.ownership = ownership
        self.owner_rank = owner_rank
        self.locations = torch.empty(
            maximum_write_records, dtype=torch.int64, device=device
        )

    def prepare(self, logical_record_ids):
        from ulacache.indexer.kernels.ownership import map_index_write_records

        count = logical_record_ids.numel()
        if count > self.locations.numel():
            raise ValueError("Native Index write exceeds its startup bound")
        output = self.locations[:count].view(logical_record_ids.shape)
        map_index_write_records(
            logical_record_ids,
            output,
            ownership=self.ownership,
            owner_rank=self.owner_rank,
        )
        return output

    def device_storage_inventory(self):
        return (self.locations,)
