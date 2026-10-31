"""Opaque native FP8 Index ownership and logical score layout operations."""

import triton
import triton.language as tl


@triton.jit
def owned_index_location(location, rank: tl.constexpr, degree: tl.constexpr):
    valid = (location >= 0) & ((location % degree == rank) | (location == 0))
    return tl.where(valid, location // degree, -1)


@triton.jit
def _ragged_ranges(
    STARTS,
    ENDS,
    OUTPUT,
    QUERY_START: tl.constexpr,
    QUERIES: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    COLUMN_START: tl.constexpr,
    COLUMNS: tl.constexpr,
    RANK: tl.constexpr,
    DEGREE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(STARTS + QUERY_START + row, row < QUERIES, other=0)
    end = tl.load(ENDS + QUERY_START + row, row < QUERIES, other=0)
    start = tl.minimum(tl.maximum(start - COLUMN_START, 0), COLUMNS)
    end = tl.minimum(tl.maximum(end - COLUMN_START, 0), COLUMNS)
    local_start = tl.maximum((start + DEGREE - 1 - RANK) // DEGREE, 0)
    local_end = tl.maximum((end + DEGREE - 1 - RANK) // DEGREE, 0)
    tl.store(OUTPUT + row, start, row < QUERIES)
    tl.store(OUTPUT + OUTPUT_STRIDE + row, end, row < QUERIES)
    tl.store(OUTPUT + 2 * OUTPUT_STRIDE + row, local_start, row < QUERIES)
    tl.store(OUTPUT + 3 * OUTPUT_STRIDE + row, local_end, row < QUERIES)


def native_fp8_ragged_ranges(
    starts,
    ends,
    output,
    *,
    query_start,
    query_count,
    column_start,
    column_count,
    owner_rank,
    owner_count,
):
    """Project native bounds without changing the original scorer's math."""
    import torch

    if (
        starts.ndim != 1
        or ends.shape != starts.shape
        or starts.dtype != torch.int32
        or ends.dtype != torch.int32
        or not starts.is_contiguous()
        or not ends.is_contiguous()
        or output.dtype != torch.int32
        or not output.is_contiguous()
        or output.ndim != 2
        or output.shape[0] != 4
        or not 0 < query_count <= output.shape[1]
        or query_start < 0
        or query_start + query_count > starts.numel()
        or column_start < 0
        or column_count < 1
        or not 0 <= owner_rank < owner_count
    ):
        raise ValueError(
            "Bind native int32 ragged bounds within the retained query arena"
        )
    _ragged_ranges[(triton.cdiv(query_count, 256),)](
        starts,
        ends,
        output,
        query_start,
        query_count,
        output.shape[1],
        column_start,
        column_count,
        owner_rank,
        owner_count,
        256,
    )
    return tuple(output[index, :query_count] for index in range(4))


@triton.jit
def _restore_scores(
    SOURCE,
    PREFIX,
    STARTS,
    ENDS,
    OUTPUT,
    QUERIES: tl.constexpr,
    COLUMNS: tl.constexpr,
    LOCAL_COLUMNS: tl.constexpr,
    DEGREE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    COMPACT: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    rank = column % DEGREE
    local = column // DEGREE
    end = tl.load(ENDS + row)
    start = tl.load(STARTS + row) if HAS_STARTS else 0
    if COMPACT:
        begin = tl.load(PREFIX + row)
        finish = tl.load(PREFIX + row + 1)
        rank_stride = tl.load(PREFIX + QUERIES)
        offset = rank * rank_stride + begin + local
        present = local < finish - begin
    else:
        offset = (rank * QUERIES + row) * LOCAL_COLUMNS + local
        present = local < LOCAL_COLUMNS
    valid = (column < COLUMNS) & (column >= start) & (column < end) & present
    value = tl.load(SOURCE + offset, valid, other=-float("inf"))
    tl.store(OUTPUT + row * OUTPUT_STRIDE + column, value, column < COLUMNS)


def restore_native_fp8_scores(layout, lengths, output, *, starts=None):
    import torch

    if (
        output.dtype != torch.float32
        or output.shape != (layout.queries, layout.logical_columns)
        or output.stride(1) != 1
        or output.device != layout.gathered.device
        or lengths.shape != (layout.queries,)
        or lengths.device != output.device
    ):
        raise ValueError(
            "Restore native FP32 score rows within the original query/column layout"
        )
    compact = layout.compact_query_offsets is not None
    _restore_scores[(layout.queries, triton.cdiv(layout.logical_columns, 256))](
        layout.gathered,
        layout.compact_query_offsets,
        starts,
        lengths,
        output,
        layout.queries,
        layout.logical_columns,
        layout.local_columns,
        layout.degree,
        output.stride(0),
        compact,
        starts is not None,
        256,
    )
    return output


@triton.jit
def _gather_local_pages(
    PAGES,
    LENGTHS,
    SOURCE,
    DESTINATION,
    PAGE_TABLE_STRIDE: tl.constexpr,
    PAGE_TABLE_COLUMNS: tl.constexpr,
    BANK_BYTE_OFFSET: tl.constexpr,
    DESTINATION_PAGE_STRIDE: tl.constexpr,
    RANK: tl.constexpr,
    DEGREE: tl.constexpr,
    PAGE_BYTES: tl.constexpr = 8448,
):
    query = tl.program_id(0).to(tl.int64)
    destination_page = tl.program_id(1)
    local_column = destination_page * 64 + tl.arange(0, 64)
    dimension = tl.arange(0, 128)
    logical_column = local_column * DEGREE + RANK
    length = tl.load(LENGTHS + query)
    live = (logical_column < length) & (logical_column // 64 < PAGE_TABLE_COLUMNS)
    physical_page = tl.load(
        PAGES + query * PAGE_TABLE_STRIDE + logical_column // 64, live, other=0
    )
    native_record = physical_page.to(tl.int64) * 64 + logical_column % 64
    owner_record = native_record // DEGREE
    source_page, source_slot = owner_record // 64, owner_record % 64
    key_offset = (
        BANK_BYTE_OFFSET
        + source_page[:, None] * PAGE_BYTES
        + source_slot[:, None] * 128
        + dimension[None, :]
    )
    keys = tl.load(SOURCE + key_offset, live[:, None], other=0)
    destination_base = (query * DESTINATION_PAGE_STRIDE + destination_page) * PAGE_BYTES
    tl.store(
        DESTINATION
        + destination_base
        + tl.arange(0, 64)[:, None] * 128
        + dimension[None, :],
        keys,
    )
    source_scale = (
        SOURCE + BANK_BYTE_OFFSET + source_page * PAGE_BYTES + 8192 + source_slot * 4
    ).to(tl.pointer_type(tl.float32))
    scales = tl.load(source_scale, live, other=0.0)
    destination_scale = (DESTINATION + destination_base + 8192).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(destination_scale + tl.arange(0, 64), scales)


def gather_local_native_fp8_pages(
    page_table,
    lengths,
    owner_storage,
    destination,
    *,
    owner_rank,
    owner_count,
    bank_byte_offset,
    destination_page_stride,
    local_page_count,
):
    if owner_count > 64 or 64 % owner_count:
        raise NotImplementedError(
            "Native FP8 Index pages must contain complete owner stripes"
        )
    _gather_local_pages[(page_table.shape[0], local_page_count)](
        page_table,
        lengths,
        owner_storage,
        destination,
        page_table.stride(0),
        page_table.shape[1],
        bank_byte_offset,
        destination_page_stride,
        owner_rank,
        owner_count,
        num_warps=4,
    )


@triton.jit
def _gather_prefix(
    PEERS,
    PAGES,
    KEYS,
    SCALES,
    RECORDS: tl.constexpr,
    BANK_BYTE_OFFSET: tl.constexpr,
    DEGREE: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    PAGE_BYTES: tl.constexpr = 8448,
):
    record = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
    dimension = tl.arange(0, 128)
    live = record < RECORDS
    page = tl.load(PAGES + record // 64, live, other=0)
    logical = page.to(tl.int64) * 64 + record % 64
    rank, local = logical % DEGREE, logical // DEGREE
    base = tl.load(PEERS + rank).to(tl.pointer_type(tl.uint8))
    source_page, source_slot = local // 64, local % 64
    source_key = (
        base[:, None]
        + BANK_BYTE_OFFSET
        + source_page[:, None] * PAGE_BYTES
        + source_slot[:, None] * 128
        + dimension[None, :]
    )
    keys = tl.load(source_key, live[:, None], other=0)
    tl.store(
        KEYS + record[:, None] * KEY_STRIDE + dimension[None, :], keys, live[:, None]
    )
    source_scale = (
        base + BANK_BYTE_OFFSET + source_page * PAGE_BYTES + 8192 + source_slot * 4
    ).to(tl.pointer_type(tl.float32))
    scale = tl.load(source_scale, live, other=0.0)
    tl.store(SCALES + record, scale, live)


def gather_native_fp8_prefix(
    peer_pointers, page_table, keys, scales, *, owner_count, bank_byte_offset
):
    import torch

    if page_table.ndim == 2:
        if page_table.shape[0] != 1:
            raise NotImplementedError(
                "Native FP8 prefix reads bind one request per call"
            )
        page_table = page_table[0]
    if keys.ndim != 2 or keys.shape[1] != 128 or keys.stride(1) != 1:
        raise ValueError("Bind the original native FP8 Index key rows")
    if scales.numel() != keys.shape[0] or not scales.is_contiguous():
        raise ValueError("Bind one native FP32 scale per Index record")
    _gather_prefix[(triton.cdiv(keys.shape[0], 16),)](
        peer_pointers,
        page_table,
        keys.view(torch.uint8),
        scales,
        keys.shape[0],
        bank_byte_offset,
        owner_count,
        keys.stride(0),
        num_warps=4,
    )


@triton.jit
def _gather_ragged_prefix(
    PEERS,
    PAGES,
    LENGTHS,
    OFFSETS,
    KEYS,
    SCALES,
    PAGE_STRIDE: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    BANK_BYTE_OFFSET: tl.constexpr,
    DEGREE: tl.constexpr,
):
    request = tl.program_id(0).to(tl.int64)
    record = tl.program_id(1).to(tl.int64) * 16 + tl.arange(0, 16)
    dimension = tl.arange(0, 128)
    length = tl.load(LENGTHS + request)
    begin = tl.load(OFFSETS + request - 1, request > 0, other=0)
    live = record < length
    page = tl.load(PAGES + request * PAGE_STRIDE + record // 64, live, other=0)
    logical = page.to(tl.int64) * 64 + record % 64
    owner, local = logical % DEGREE, logical // DEGREE
    base = tl.load(PEERS + owner).to(tl.pointer_type(tl.uint8))
    source_page, source_slot = local // 64, local % 64
    key = tl.load(
        base[:, None]
        + BANK_BYTE_OFFSET
        + source_page[:, None] * 8448
        + source_slot[:, None] * 128
        + dimension[None, :],
        live[:, None],
        other=0,
    )
    tl.store(
        KEYS + (begin + record)[:, None] * KEY_STRIDE + dimension[None, :],
        key,
        live[:, None],
    )
    source_scale = (
        base + BANK_BYTE_OFFSET + source_page * 8448 + 8192 + source_slot * 4
    ).to(tl.pointer_type(tl.float32))
    scale = tl.load(source_scale, live, other=0.0)
    tl.store(SCALES + begin + record, scale, live)


def gather_native_fp8_ragged_prefix(
    peer_pointers,
    page_table,
    lengths,
    keys,
    scales,
    *,
    maximum_length,
    owner_count,
    bank_byte_offset,
):
    import torch

    if (
        page_table.ndim != 2
        or lengths.shape != (page_table.shape[0],)
        or not lengths.is_contiguous()
        or keys.shape[1:] != (128,)
        or keys.stride(1) != 1
        or scales.numel() != keys.shape[0]
        or scales.dtype != torch.float32
        or not scales.is_contiguous()
        or maximum_length > page_table.shape[1] * 64
    ):
        raise ValueError(
            "Bind native request lengths, page tables and flattened FP8 prefix outputs"
        )
    if keys.shape[0]:
        offsets = lengths.cumsum(0, dtype=torch.int64)
        _gather_ragged_prefix[(page_table.shape[0], triton.cdiv(maximum_length, 16))](
            peer_pointers,
            page_table,
            lengths,
            offsets,
            keys.view(torch.uint8),
            scales,
            page_table.stride(0),
            keys.stride(0),
            bank_byte_offset,
            owner_count,
            num_warps=4,
        )


@triton.jit
def _store_quantized(
    LOCATIONS,
    KEYS,
    SCALES,
    DESTINATION,
    COUNT: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    RANK: tl.constexpr,
    DEGREE: tl.constexpr,
):
    record = tl.program_id(0).to(tl.int64)
    dimension = tl.arange(0, 128)
    global_location = tl.load(LOCATIONS + record).to(tl.int64)
    location = owned_index_location(global_location, RANK, DEGREE)
    valid = location >= 0
    safe = tl.maximum(location, 0)
    key = tl.load(KEYS + record * KEY_STRIDE + dimension)
    tl.store(DESTINATION + safe // 64 * 8448 + safe % 64 * 128 + dimension, key, valid)
    scale = tl.load(SCALES + record)
    output_scale = (DESTINATION + safe // 64 * 8448 + 8192 + safe % 64 * 4).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(output_scale, scale, valid)


def store_owned_native_fp8_index(
    locations, keys, scales, destination, *, owner_rank, owner_count
):
    import torch

    if (
        keys.shape != (locations.numel(), 128)
        or not scales.is_contiguous()
        or scales.numel() != locations.numel()
        or scales.dtype != torch.float32
        or destination.dtype != torch.uint8
        or not destination.is_contiguous()
    ):
        raise ValueError("Preserve native FP8 key bytes, FP32 scales and packed pages")
    if locations.numel():
        _store_quantized[(locations.numel(),)](
            locations,
            keys.view(torch.uint8),
            scales,
            destination,
            locations.numel(),
            keys.stride(0),
            owner_rank,
            owner_count,
            num_warps=4,
        )
