"""Translate consumer logical records into owner-local residency inputs."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["Count"])
def _map_owner_records(
    LogicalRecords,
    OwnerRecords,
    Count,
    OwnerRank: tl.constexpr,
    OwnerCount: tl.constexpr,
    OwnerLogicalCapacity: tl.constexpr,
    BlockSize: tl.constexpr,
):
    index = tl.program_id(0) * BlockSize + tl.arange(0, BlockSize)
    logical_record = tl.load(LogicalRecords + index, index < Count, other=-1)
    valid = (index < Count) & (logical_record > 0)
    tl.device_assert(
        (~valid) | (logical_record < OwnerCount * OwnerLogicalCapacity),
        "Native record exceeds the shared logical namespace",
    )
    local_record = logical_record // OwnerCount
    owned = valid & (logical_record % OwnerCount == OwnerRank) & (local_record > 0)
    tl.store(
        OwnerRecords + index,
        tl.where(owned, local_record, -1).to(tl.int32),
        index < Count,
    )


def map_owner_records(
    logical_record_ids,
    owner_local_record_ids,
    *,
    owner_rank,
    owner_count,
    owner_logical_capacity,
):
    """Map allocator/writer IDs; unowned records and padding become negative."""
    if logical_record_ids.numel() != owner_local_record_ids.numel():
        raise ValueError("Owner mapping must preserve the native record count")
    if logical_record_ids.numel():
        _map_owner_records[(triton.cdiv(logical_record_ids.numel(), 256),)](
            logical_record_ids,
            owner_local_record_ids,
            logical_record_ids.numel(),
            owner_rank,
            owner_count,
            owner_logical_capacity,
            256,
        )


@triton.jit
def _export_consumer_records(
    SelectedRecords,
    CurrentRecords,
    ActiveQueries,
    ExportedRecords,
    QueryCount: tl.constexpr,
    SelectionWidth: tl.constexpr,
    CurrentCount: tl.constexpr,
    ExportedQueryCapacity: tl.constexpr,
    BlockSize: tl.constexpr,
):
    index = tl.program_id(0) * BlockSize + tl.arange(0, BlockSize)
    width = SelectionWidth + 1
    query, column = index // width, index % width
    active = tl.load(ActiveQueries)
    selected = tl.load(
        SelectedRecords + query * SelectionWidth + column,
        (query < QueryCount) & (query < active) & (column < SelectionWidth),
        other=-1,
    )
    current = tl.load(
        CurrentRecords + query,
        (query < CurrentCount) & (query < active) & (column == SelectionWidth),
        other=-1,
    )
    logical_record = tl.where(column < SelectionWidth, selected, current)
    tl.store(
        ExportedRecords + index,
        tl.where(logical_record > 0, logical_record, -1).to(tl.int32),
        index < QueryCount * width,
    )


def export_consumer_records(
    selected_logical_records,
    current_logical_records,
    active_query_count,
    exported_record_ids,
):
    """Publish into a fixed contiguous int32[query_capacity, selected_width+1].

    This copy precedes the selection-ready publication. Each peer masks its
    own inactive queries; owners need no CPU-visible query counts.
    Only the current query-count prefix is published. All peers collect that
    same padded prefix, rather than the entire startup allocation capacity.
    """
    query_count, width = selected_logical_records.shape
    query_capacity = exported_record_ids.shape[0]
    if query_count > query_capacity or exported_record_ids.shape[1] != width + 1:
        raise ValueError("Consumer export exceeds its startup query or selection shape")
    if query_count:
        _export_consumer_records[(triton.cdiv(query_count * (width + 1), 256),)](
            selected_logical_records,
            current_logical_records,
            active_query_count,
            exported_record_ids,
            query_count,
            width,
            current_logical_records.numel(),
            query_capacity,
            256,
        )


@triton.jit(do_not_specialize=["RecordsPerConsumer"])
def _collect_owner_records(
    ConsumerPeerPointers,
    OwnerRecords,
    OwnerRank: tl.constexpr,
    OwnerCount: tl.constexpr,
    RecordsPerConsumer,
    OwnerLogicalCapacity: tl.constexpr,
    BlockSize: tl.constexpr,
):
    index = tl.program_id(0) * BlockSize + tl.arange(0, BlockSize)
    count = OwnerCount * RecordsPerConsumer
    consumer, position = index // RecordsPerConsumer, index % RecordsPerConsumer
    source = tl.load(ConsumerPeerPointers + consumer, index < count, other=0).to(
        tl.pointer_type(tl.int32)
    )
    logical_record = tl.load(
        source + position, index < count, other=-1, cache_modifier=".cg"
    )
    valid = (index < count) & (logical_record > 0)
    tl.device_assert(
        (~valid) | (logical_record < OwnerCount * OwnerLogicalCapacity),
        "Consumer record exceeds the shared logical namespace",
    )
    local_record = logical_record // OwnerCount
    owned = valid & (logical_record % OwnerCount == OwnerRank) & (local_record > 0)
    tl.store(
        OwnerRecords + index,
        tl.where(owned, local_record, -1).to(tl.int32),
        index < count,
    )


def collect_owner_records(
    consumer_peer_pointers,
    owner_local_record_ids,
    *,
    owner_rank,
    owner_count,
    records_per_consumer,
    owner_logical_capacity,
):
    """Read published peers into a padded owner-local input, retaining repeats.

    The existing native residency planner deduplicates positive IDs. Native
    reserved allocation pages cover owner-local logical zero; it remains the
    local planner's sentinel. This function does not register or promote data.
    """
    count = owner_count * records_per_consumer
    if owner_local_record_ids.numel() != count:
        raise ValueError("Owner residency input must cover every consumer export")
    if count:
        _collect_owner_records[(triton.cdiv(count, 256),)](
            consumer_peer_pointers,
            owner_local_record_ids,
            owner_rank,
            owner_count,
            records_per_consumer,
            owner_logical_capacity,
            256,
        )


@triton.jit(do_not_specialize=["RecordCount", "SelectionWidth"])
def _read_owner_locations(
    SelectedRecords,
    ActiveQueries,
    DirectoryPeerPointers,
    SourceOwners,
    SourceRecords,
    DestinationRecords,
    RecordCount,
    SelectionWidth,
    OwnerCount: tl.constexpr,
    OwnerLogicalCapacity: tl.constexpr,
    BlockSize: tl.constexpr,
):
    index = tl.program_id(0) * BlockSize + tl.arange(0, BlockSize)
    logical_record = tl.load(SelectedRecords + index, index < RecordCount, other=-1)
    active = tl.load(ActiveQueries)
    owner, local_record = logical_record % OwnerCount, logical_record // OwnerCount
    valid = (
        (index < RecordCount)
        & (index // SelectionWidth < active)
        & (logical_record > 0)
        & (local_record > 0)
    )
    tl.device_assert(
        (~valid) | (logical_record < OwnerCount * OwnerLogicalCapacity),
        "Selected record exceeds the shared logical namespace",
    )
    directory = tl.load(DirectoryPeerPointers + owner, valid, other=0).to(
        tl.pointer_type(tl.int64)
    )
    location = tl.load(directory + local_record, valid, other=-1, cache_modifier=".cg")
    tl.device_assert(
        (~valid) | (location >= 0), "Owner did not promote a selected record"
    )
    resident = valid & (location >= 0)
    tl.store(
        SourceOwners + index,
        tl.where(resident, owner, -1).to(tl.int32),
        index < RecordCount,
    )
    tl.store(
        SourceRecords + index,
        tl.where(resident, location, -1).to(tl.int32),
        index < RecordCount,
    )
    tl.store(
        DestinationRecords + index,
        tl.where(resident, index, -1).to(tl.int32),
        index < RecordCount,
    )


def read_owner_locations(
    selected_logical_records,
    active_query_count,
    directory_peer_pointers,
    source_owner_ranks,
    source_record_indices,
    destination_record_indices,
    *,
    owner_count,
    owner_logical_capacity,
):
    """Resolve published owner GPU directories before native peer-page gather.

    Locations remain borrowed through the reader-retired phase. Destination
    records preserve selection order and duplicates, with negative padding.
    They are ordinal rows in a separate local native attention bank.
    """
    if selected_logical_records.numel():
        _read_owner_locations[(triton.cdiv(selected_logical_records.numel(), 256),)](
            selected_logical_records,
            active_query_count,
            directory_peer_pointers,
            source_owner_ranks,
            source_record_indices,
            destination_record_indices,
            selected_logical_records.numel(),
            selected_logical_records.shape[1],
            owner_count,
            owner_logical_capacity,
            256,
        )
