"""Exchange independent native queries and score token-striped Index storage.

NativeScoreTile is a caller-supplied Triton service. It retains the serving
framework's encoding, dot products and rounding. This module owns placement,
communication layout, visibility and reconstruction into original columns.
"""

import triton
import triton.language as tl


@triton.jit
def _export_queries_body(
    Q,
    Weights,
    Requests,
    Lengths,
    Mask,
    Arena,
    Rows,
    Heads: tl.constexpr,
    Dimension: tl.constexpr,
    Width,
    QStride0: tl.constexpr,
    QStride1: tl.constexpr,
    QStride2: tl.constexpr,
    WStride0: tl.constexpr,
    WStride1: tl.constexpr,
    MaskStride: tl.constexpr,
    WeightsOffset,
    RequestsOffset,
    LengthsOffset,
    MasksOffset,
    HasMask: tl.constexpr,
    Block: tl.constexpr,
    MetadataBlock: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * Block + tl.arange(0, Block)
    qrow, element = offset // (Heads * Dimension), offset % (Heads * Dimension)
    head, channel = element // Dimension, element % Dimension
    query = tl.load(
        Q + qrow * QStride0 + head * QStride1 + channel * QStride2, qrow < Rows, 0
    )
    qtarget = Arena.to(tl.pointer_type(tl.bfloat16))
    tl.store(qtarget + offset, query, offset < Rows * Heads * Dimension)
    wrow, wh = offset // Heads, offset % Heads
    weight = tl.load(Weights + wrow * WStride0 + wh * WStride1, wrow < Rows, 0)
    wtarget = (Arena + WeightsOffset).to(tl.pointer_type(tl.bfloat16))
    tl.store(wtarget + offset, weight.to(tl.bfloat16), offset < Rows * Heads)
    if tl.program_id(0) == 0:
        row = tl.arange(0, MetadataBlock)
        req = tl.load(Requests + row, row < Rows, 0).to(tl.int64)
        length = tl.load(Lengths + row, row < Rows, 0).to(tl.int64)
        tl.device_assert(
            (row >= Rows) | ((length >= 0) & (length <= Width)),
            "Native Index length exceeds its score bound",
        )
        tl.store(
            (Arena + RequestsOffset).to(tl.pointer_type(tl.int64)) + row,
            req,
            row < Rows,
        )
        tl.store(
            (Arena + LengthsOffset).to(tl.pointer_type(tl.int64)) + row,
            length,
            row < Rows,
        )
    if HasMask:
        mask_bytes = tl.cdiv(Width, 8)
        bit = tl.arange(0, 8)
        row = offset // mask_bytes
        column = (offset % mask_bytes)[:, None] * 8 + bit[None, :]
        value = tl.load(
            Mask + row[:, None] * MaskStride + column,
            (row[:, None] < Rows) & (column < Width),
            False,
        )
        packed = tl.sum(value.to(tl.uint32) << bit[None, :], axis=1).to(tl.uint8)
        tl.store(Arena + MasksOffset + offset, packed, offset < Rows * mask_bytes)


@triton.jit
def export_queries(
    Q,
    Weights,
    Requests,
    Lengths,
    Mask,
    Arena,
    Rows: tl.constexpr,
    Heads: tl.constexpr,
    Dimension: tl.constexpr,
    Width: tl.constexpr,
    QStride0: tl.constexpr,
    QStride1: tl.constexpr,
    QStride2: tl.constexpr,
    WStride0: tl.constexpr,
    WStride1: tl.constexpr,
    MaskStride: tl.constexpr,
    WeightsOffset: tl.constexpr,
    RequestsOffset: tl.constexpr,
    LengthsOffset: tl.constexpr,
    MasksOffset: tl.constexpr,
    HasMask: tl.constexpr,
    Block: tl.constexpr,
    MetadataBlock: tl.constexpr,
):
    _export_queries_body(
        Q,
        Weights,
        Requests,
        Lengths,
        Mask,
        Arena,
        Rows,
        Heads,
        Dimension,
        Width,
        QStride0,
        QStride1,
        QStride2,
        WStride0,
        WStride1,
        MaskStride,
        WeightsOffset,
        RequestsOffset,
        LengthsOffset,
        MasksOffset,
        HasMask,
        Block,
        MetadataBlock,
    )


@triton.jit
def export_queries_runtime(
    Q,
    Weights,
    Requests,
    Lengths,
    Mask,
    Arena,
    Rows,
    Heads: tl.constexpr,
    Dimension: tl.constexpr,
    Width,
    QStride0: tl.constexpr,
    QStride1: tl.constexpr,
    QStride2: tl.constexpr,
    WStride0: tl.constexpr,
    WStride1: tl.constexpr,
    MaskStride: tl.constexpr,
    WeightsOffset,
    RequestsOffset,
    LengthsOffset,
    MasksOffset,
    HasMask: tl.constexpr,
    Block: tl.constexpr,
    MetadataBlock: tl.constexpr,
):
    _export_queries_body(
        Q,
        Weights,
        Requests,
        Lengths,
        Mask,
        Arena,
        Rows,
        Heads,
        Dimension,
        Width,
        QStride0,
        QStride1,
        QStride2,
        WStride0,
        WStride1,
        MaskStride,
        WeightsOffset,
        RequestsOffset,
        LengthsOffset,
        MasksOffset,
        HasMask,
        Block,
        MetadataBlock,
    )


@triton.jit
def _collect_queries_body(
    PeerPointers,
    Collected,
    UsedBytes,
    Degree: tl.constexpr,
    WeightsOffset,
    RequestsOffset,
    LengthsOffset,
    MasksOffset,
    Block: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * Block + tl.arange(0, Block)
    valid = offset < Degree * UsedBytes
    consumer, byte = offset // UsedBytes, offset % UsedBytes
    source = tl.load(PeerPointers + consumer, valid, 0).to(tl.pointer_type(tl.uint8))
    value = tl.load(source + byte, valid, 0, cache_modifier=".cg")
    start = tl.where(
        byte < WeightsOffset,
        0,
        tl.where(
            byte < RequestsOffset,
            WeightsOffset,
            tl.where(
                byte < LengthsOffset,
                RequestsOffset,
                tl.where(byte < MasksOffset, LengthsOffset, MasksOffset),
            ),
        ),
    )
    width = tl.where(
        byte < WeightsOffset,
        WeightsOffset,
        tl.where(
            byte < RequestsOffset,
            RequestsOffset - WeightsOffset,
            tl.where(
                byte < LengthsOffset,
                LengthsOffset - RequestsOffset,
                tl.where(
                    byte < MasksOffset,
                    MasksOffset - LengthsOffset,
                    UsedBytes - MasksOffset,
                ),
            ),
        ),
    )
    destination = Degree * start + consumer.to(tl.int64) * width + byte - start
    tl.store(Collected + destination, value, valid)


@triton.jit
def collect_queries(
    PeerPointers,
    Collected,
    UsedBytes: tl.constexpr,
    Degree: tl.constexpr,
    WeightsOffset: tl.constexpr,
    RequestsOffset: tl.constexpr,
    LengthsOffset: tl.constexpr,
    MasksOffset: tl.constexpr,
    Block: tl.constexpr,
):
    _collect_queries_body(
        PeerPointers,
        Collected,
        UsedBytes,
        Degree,
        WeightsOffset,
        RequestsOffset,
        LengthsOffset,
        MasksOffset,
        Block,
    )


@triton.jit
def collect_queries_runtime(
    PeerPointers,
    Collected,
    UsedBytes,
    Degree: tl.constexpr,
    WeightsOffset,
    RequestsOffset,
    LengthsOffset,
    MasksOffset,
    Block: tl.constexpr,
):
    _collect_queries_body(
        PeerPointers,
        Collected,
        UsedBytes,
        Degree,
        WeightsOffset,
        RequestsOffset,
        LengthsOffset,
        MasksOffset,
        Block,
    )


@triton.jit
def collect_queries_words(
    PeerPointers,
    Collected,
    UsedBytes: tl.constexpr,
    Degree: tl.constexpr,
    WeightsOffset: tl.constexpr,
    RequestsOffset: tl.constexpr,
    LengthsOffset: tl.constexpr,
    MasksOffset: tl.constexpr,
    Block: tl.constexpr,
):
    # The unmasked export consists entirely of aligned BF16/int64 sections.
    # Copy their exact bits in words instead of issuing byte-wide peer loads.
    offset = tl.program_id(0).to(tl.int64) * Block + tl.arange(0, Block)
    used = UsedBytes // 4
    valid = offset < Degree * used
    consumer, word = offset // used, offset % used
    source = tl.load(PeerPointers + consumer, valid, 0).to(tl.pointer_type(tl.uint32))
    value = tl.load(source + word, valid, 0, cache_modifier=".cg")
    weights, requests, lengths = (
        WeightsOffset // 4,
        RequestsOffset // 4,
        LengthsOffset // 4,
    )
    start = tl.where(
        word < weights,
        0,
        tl.where(word < requests, weights, tl.where(word < lengths, requests, lengths)),
    )
    width = tl.where(
        word < weights,
        weights,
        tl.where(
            word < requests,
            requests - weights,
            tl.where(word < lengths, lengths - requests, used - lengths),
        ),
    )
    destination = Degree * start + consumer.to(tl.int64) * width + word - start
    tl.store(Collected.to(tl.pointer_type(tl.uint32)) + destination, value, valid)


@triton.jit
def identify_equal_queries(
    Collected,
    Flags,
    Rows: tl.constexpr,
    Heads: tl.constexpr,
    Dimension: tl.constexpr,
    Width: tl.constexpr,
    Degree: tl.constexpr,
    WeightsOffset: tl.constexpr,
    RequestsOffset: tl.constexpr,
    LengthsOffset: tl.constexpr,
    MasksOffset: tl.constexpr,
    HasMask: tl.constexpr,
    Block: tl.constexpr,
    MaskBlock: tl.constexpr,
):
    consumer, row = tl.program_id(0), tl.program_id(1)
    equal = tl.full((), True, tl.int1)
    if consumer != 0:
        # Compare bit patterns, including metadata and candidate masks. A
        # matching row can therefore use consumer zero's unchanged score tile.
        qwords: tl.constexpr = Heads * Dimension // 2
        wwords: tl.constexpr = Heads // 2
        word = tl.arange(0, Block)
        in_q = word < qwords
        in_w = (word >= qwords) & (word < qwords + wwords)
        in_r = (word >= qwords + wwords) & (word < qwords + wwords + 2)
        reference = tl.where(
            in_q,
            row * qwords + word,
            tl.where(
                in_w,
                Degree * (WeightsOffset // 4) + row * wwords + word - qwords,
                tl.where(
                    in_r,
                    Degree * (RequestsOffset // 4) + row * 2 + word - qwords - wwords,
                    Degree * (LengthsOffset // 4)
                    + row * 2
                    + word
                    - qwords
                    - wwords
                    - 2,
                ),
            ),
        )
        stride = tl.where(in_q, Rows * qwords, tl.where(in_w, Rows * wwords, Rows * 2))
        words = Collected.to(tl.pointer_type(tl.uint32))
        valid = word < qwords + wwords + 4
        original = tl.load(words + reference, valid, 0)
        other = tl.load(words + reference + consumer * stride, valid, 0)
        equal = tl.sum((original != other).to(tl.int32), axis=0) == 0
        if HasMask:
            mask_bytes: tl.constexpr = tl.cdiv(Width, 8)
            for start in range(tl.cdiv(mask_bytes, MaskBlock)):
                byte = start * MaskBlock + tl.arange(0, MaskBlock)
                reference_byte = Degree * MasksOffset + row * mask_bytes + byte
                original_byte = tl.load(
                    Collected + reference_byte, byte < mask_bytes, 0
                )
                other_byte = tl.load(
                    Collected + reference_byte + consumer * Rows * mask_bytes,
                    byte < mask_bytes,
                    0,
                )
                equal = equal & (
                    tl.sum((original_byte != other_byte).to(tl.int32), axis=0) == 0
                )
    tl.store(Flags + consumer * Rows + row, equal.to(tl.uint8))


@triton.jit
def score_local_index(
    Collected,
    RequestTable,
    NativeIndex,
    Scores,
    Rows: tl.constexpr,
    Heads: tl.constexpr,
    Dimension: tl.constexpr,
    Width: tl.constexpr,
    LocalWidth: tl.constexpr,
    Degree: tl.constexpr,
    OwnerRank: tl.constexpr,
    WeightsOffset: tl.constexpr,
    RequestsOffset: tl.constexpr,
    LengthsOffset: tl.constexpr,
    MasksOffset: tl.constexpr,
    RequestStride: tl.constexpr,
    NativePageSize: tl.constexpr,
    NativeRowStride: tl.constexpr,
    LogicalCapacity: tl.constexpr,
    Reserved: tl.constexpr,
    Ratio: tl.constexpr,
    HasMask: tl.constexpr,
    NativeScoreTile: tl.constexpr,
    Block: tl.constexpr,
):
    consumer_query = tl.program_id(0)
    q = Collected.to(tl.pointer_type(tl.bfloat16))
    weights = (Collected + Degree * WeightsOffset).to(tl.pointer_type(tl.bfloat16))
    request = tl.load(
        (Collected + Degree * RequestsOffset).to(tl.pointer_type(tl.int64))
        + consumer_query
    )
    length = tl.minimum(
        tl.load(
            (Collected + Degree * LengthsOffset).to(tl.pointer_type(tl.int64))
            + consumer_query
        ),
        Width,
    )
    local_visible = tl.maximum(0, tl.cdiv(length - OwnerRank, Degree))
    for tile in range(
        tl.program_id(1), tl.cdiv(local_visible, Block), tl.num_programs(1)
    ):
        local_column = tile * Block + tl.arange(0, Block)
        column = OwnerRank + Degree * local_column
        valid = column < length
        if HasMask:
            packed = tl.load(
                Collected
                + Degree * MasksOffset
                + consumer_query.to(tl.int64) * tl.cdiv(Width, 8)
                + column // 8,
                valid,
                0,
            )
            valid = valid & (((packed >> (column % 8)) & 1) != 0)
        record = (
            tl.load(
                RequestTable + request * RequestStride + column.to(tl.int64) * Ratio,
                valid,
                0,
            ).to(tl.int64)
            // Ratio
        )
        tl.device_assert(
            (~valid)
            | (record == 0)
            | (
                (record >= Reserved)
                & (record < LogicalCapacity)
                & (record % Degree == OwnerRank)
            ),
            "Native Index stripe does not match the allocator",
        )
        logit = NativeScoreTile(
            q,
            weights,
            NativeIndex,
            consumer_query,
            record // Degree,
            valid,
            NativePageSize,
            NativeRowStride,
            Heads * Dimension,
            Dimension,
            Heads,
            Heads,
            Dimension // 2,
        )
        tl.store(
            Scores + consumer_query.to(tl.int64) * LocalWidth + local_column,
            logit,
            local_column < LocalWidth,
        )


@triton.jit
def score_local_index_reuse(
    Collected,
    RequestTable,
    NativeIndex,
    Scores,
    Flags,
    Rows: tl.constexpr,
    Heads: tl.constexpr,
    Dimension: tl.constexpr,
    Width: tl.constexpr,
    LocalWidth: tl.constexpr,
    Degree: tl.constexpr,
    OwnerRank: tl.constexpr,
    WeightsOffset: tl.constexpr,
    RequestsOffset: tl.constexpr,
    LengthsOffset: tl.constexpr,
    MasksOffset: tl.constexpr,
    RequestStride: tl.constexpr,
    NativePageSize: tl.constexpr,
    NativeRowStride: tl.constexpr,
    LogicalCapacity: tl.constexpr,
    Reserved: tl.constexpr,
    Ratio: tl.constexpr,
    HasMask: tl.constexpr,
    NativeScoreTile: tl.constexpr,
    Block: tl.constexpr,
):
    group, row = tl.program_id(0) // Rows, tl.program_id(0) % Rows
    consumer = tl.arange(0, Degree)
    unique = (consumer == 0) | (tl.load(Flags + consumer * Rows + row) == 0)
    ordinal = tl.cumsum(unique.to(tl.int32), axis=0) - 1
    count = tl.sum(unique.to(tl.int32), axis=0)
    target = group % count
    selected = tl.sum(tl.where(unique & (ordinal == target), consumer, 0), axis=0)
    consumer_query = selected * Rows + row
    # Every program retains useful work. Fully independent inputs reproduce
    # the original consumer/tile assignment; equal inputs share its workers.
    worker = (group // count) * tl.num_programs(1) + tl.program_id(1)
    workers = tl.cdiv(Degree - target, count) * tl.num_programs(1)
    q = Collected.to(tl.pointer_type(tl.bfloat16))
    weights = (Collected + Degree * WeightsOffset).to(tl.pointer_type(tl.bfloat16))
    request = tl.load(
        (Collected + Degree * RequestsOffset).to(tl.pointer_type(tl.int64))
        + consumer_query
    )
    length = tl.minimum(
        tl.load(
            (Collected + Degree * LengthsOffset).to(tl.pointer_type(tl.int64))
            + consumer_query
        ),
        Width,
    )
    local_visible = tl.maximum(0, tl.cdiv(length - OwnerRank, Degree))
    for tile in range(worker, tl.cdiv(local_visible, Block), workers):
        local_column = tile * Block + tl.arange(0, Block)
        column = OwnerRank + Degree * local_column
        valid = column < length
        if HasMask:
            packed = tl.load(
                Collected
                + Degree * MasksOffset
                + consumer_query.to(tl.int64) * tl.cdiv(Width, 8)
                + column // 8,
                valid,
                0,
            )
            valid = valid & (((packed >> (column % 8)) & 1) != 0)
        record = (
            tl.load(
                RequestTable + request * RequestStride + column.to(tl.int64) * Ratio,
                valid,
                0,
            ).to(tl.int64)
            // Ratio
        )
        tl.device_assert(
            (~valid)
            | (record == 0)
            | (
                (record >= Reserved)
                & (record < LogicalCapacity)
                & (record % Degree == OwnerRank)
            ),
            "Native Index stripe does not match the allocator",
        )
        logit = NativeScoreTile(
            q,
            weights,
            NativeIndex,
            consumer_query,
            record // Degree,
            valid,
            NativePageSize,
            NativeRowStride,
            Heads * Dimension,
            Dimension,
            Heads,
            Heads,
            Dimension // 2,
        )
        tl.store(
            Scores + consumer_query.to(tl.int64) * LocalWidth + local_column,
            logit,
            local_column < LocalWidth,
        )


@triton.jit
def _copy_prefill_scores_body(
    NativeScores,
    Collected,
    ExpectedRequest,
    Scores,
    Rows,
    Width,
    LocalWidth,
    NativeWidth,
    NativeStride,
    Degree: tl.constexpr,
    OwnerRank: tl.constexpr,
    RequestsOffset,
    LengthsOffset,
    MasksOffset,
    HasMask: tl.constexpr,
    Block: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * Block + tl.arange(0, Block)
    row, local_column = offset // LocalWidth, offset % LocalWidth
    in_row = row < Degree * Rows
    request = tl.load(
        (Collected + Degree * RequestsOffset).to(tl.pointer_type(tl.int64)) + row,
        in_row,
        0,
    )
    expected = tl.load(ExpectedRequest)
    tl.device_assert(
        (~in_row) | (request == expected),
        "Native prefill Index keys belong to one shared request",
    )
    length = tl.load(
        (Collected + Degree * LengthsOffset).to(tl.pointer_type(tl.int64)) + row,
        in_row,
        0,
    )
    column = OwnerRank + Degree * local_column
    valid = (
        in_row
        & (local_column < NativeWidth)
        & (column < Width)
        & (column < length)
        & (request == expected)
    )
    if HasMask:
        packed = tl.load(
            Collected + Degree * MasksOffset + row * tl.cdiv(Width, 8) + column // 8,
            valid,
            0,
        )
        valid = valid & (((packed >> (column % 8)) & 1) != 0)
    score = tl.load(
        NativeScores + row * NativeStride + local_column, valid, -float("inf")
    )
    tl.store(Scores + offset, score, in_row)


@triton.jit
def copy_prefill_scores(
    NativeScores,
    Collected,
    ExpectedRequest,
    Scores,
    Rows: tl.constexpr,
    Width: tl.constexpr,
    LocalWidth: tl.constexpr,
    NativeWidth: tl.constexpr,
    NativeStride: tl.constexpr,
    Degree: tl.constexpr,
    OwnerRank: tl.constexpr,
    RequestsOffset: tl.constexpr,
    LengthsOffset: tl.constexpr,
    MasksOffset: tl.constexpr,
    HasMask: tl.constexpr,
    Block: tl.constexpr,
):
    _copy_prefill_scores_body(
        NativeScores,
        Collected,
        ExpectedRequest,
        Scores,
        Rows,
        Width,
        LocalWidth,
        NativeWidth,
        NativeStride,
        Degree,
        OwnerRank,
        RequestsOffset,
        LengthsOffset,
        MasksOffset,
        HasMask,
        Block,
    )


@triton.jit
def copy_prefill_scores_runtime(
    NativeScores,
    Collected,
    ExpectedRequest,
    Scores,
    Rows,
    Width,
    LocalWidth,
    NativeWidth,
    NativeStride,
    Degree: tl.constexpr,
    OwnerRank: tl.constexpr,
    RequestsOffset,
    LengthsOffset,
    MasksOffset,
    HasMask: tl.constexpr,
    Block: tl.constexpr,
):
    _copy_prefill_scores_body(
        NativeScores,
        Collected,
        ExpectedRequest,
        Scores,
        Rows,
        Width,
        LocalWidth,
        NativeWidth,
        NativeStride,
        Degree,
        OwnerRank,
        RequestsOffset,
        LengthsOffset,
        MasksOffset,
        HasMask,
        Block,
    )


@triton.jit
def _restore_peer_scores_body(
    PeerPointers,
    Lengths,
    Output,
    Rows,
    Width,
    LocalWidth,
    OutputStride,
    Degree: tl.constexpr,
    Consumer: tl.constexpr,
    Block: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * Block + tl.arange(0, Block)
    length = tl.load(Lengths + row)
    valid = (column < Width) & (column < length)
    owner, local_column = column % Degree, column // Degree
    source = tl.load(PeerPointers + owner, valid, 0).to(tl.pointer_type(tl.float32))
    offset = (Consumer * Rows + row).to(tl.int64) * LocalWidth + local_column
    score = tl.load(source + offset, valid, -float("inf"), cache_modifier=".cg")
    tl.store(
        Output + row.to(tl.int64) * OutputStride + column, score, column < OutputStride
    )


@triton.jit
def restore_peer_scores(
    PeerPointers,
    Lengths,
    Output,
    Rows: tl.constexpr,
    Width: tl.constexpr,
    LocalWidth: tl.constexpr,
    OutputStride: tl.constexpr,
    Degree: tl.constexpr,
    Consumer: tl.constexpr,
    Block: tl.constexpr,
):
    _restore_peer_scores_body(
        PeerPointers,
        Lengths,
        Output,
        Rows,
        Width,
        LocalWidth,
        OutputStride,
        Degree,
        Consumer,
        Block,
    )


@triton.jit
def restore_peer_scores_runtime(
    PeerPointers,
    Lengths,
    Output,
    Rows,
    Width,
    LocalWidth,
    OutputStride,
    Degree: tl.constexpr,
    Consumer: tl.constexpr,
    Block: tl.constexpr,
):
    _restore_peer_scores_body(
        PeerPointers,
        Lengths,
        Output,
        Rows,
        Width,
        LocalWidth,
        OutputStride,
        Degree,
        Consumer,
        Block,
    )


@triton.jit
def restore_peer_scores_reuse(
    PeerPointers,
    Lengths,
    Output,
    Flags,
    Rows: tl.constexpr,
    Width: tl.constexpr,
    LocalWidth: tl.constexpr,
    OutputStride: tl.constexpr,
    Degree: tl.constexpr,
    Consumer: tl.constexpr,
    Block: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * Block + tl.arange(0, Block)
    length = tl.load(Lengths + row)
    valid = (column < Width) & (column < length)
    owner, local_column = column % Degree, column // Degree
    source = tl.load(PeerPointers + owner, valid, 0).to(tl.pointer_type(tl.float32))
    duplicate = tl.load(Flags + Consumer * Rows + row) != 0
    consumer = tl.where(duplicate, 0, Consumer)
    offset = (consumer * Rows + row).to(tl.int64) * LocalWidth + local_column
    score = tl.load(source + offset, valid, -float("inf"), cache_modifier=".cg")
    tl.store(
        Output + row.to(tl.int64) * OutputStride + column, score, column < OutputStride
    )


@triton.jit
def identify_equal_queries_runtime(
    Collected,
    Flags,
    Rows,
    Heads: tl.constexpr,
    Dimension: tl.constexpr,
    Width,
    Degree: tl.constexpr,
    WeightsOffset,
    RequestsOffset,
    LengthsOffset,
    MasksOffset,
    HasMask: tl.constexpr,
    Block: tl.constexpr,
    MaskBlock: tl.constexpr,
):
    consumer, row = tl.program_id(0), tl.program_id(1)
    equal = tl.full((), True, tl.int1)
    if consumer != 0:
        # Compare bit patterns, including metadata and candidate masks. A
        # matching row can therefore use consumer zero's unchanged score tile.
        qwords: tl.constexpr = Heads * Dimension // 2
        wwords: tl.constexpr = Heads // 2
        word = tl.arange(0, Block)
        in_q = word < qwords
        in_w = (word >= qwords) & (word < qwords + wwords)
        in_r = (word >= qwords + wwords) & (word < qwords + wwords + 2)
        reference = tl.where(
            in_q,
            row * qwords + word,
            tl.where(
                in_w,
                Degree * (WeightsOffset // 4) + row * wwords + word - qwords,
                tl.where(
                    in_r,
                    Degree * (RequestsOffset // 4) + row * 2 + word - qwords - wwords,
                    Degree * (LengthsOffset // 4)
                    + row * 2
                    + word
                    - qwords
                    - wwords
                    - 2,
                ),
            ),
        )
        stride = tl.where(in_q, Rows * qwords, tl.where(in_w, Rows * wwords, Rows * 2))
        words = Collected.to(tl.pointer_type(tl.uint32))
        valid = word < qwords + wwords + 4
        original = tl.load(words + reference, valid, 0)
        other = tl.load(words + reference + consumer * stride, valid, 0)
        equal = tl.sum((original != other).to(tl.int32), axis=0) == 0
        if HasMask:
            mask_bytes = tl.cdiv(Width, 8)
            for start in range(tl.cdiv(mask_bytes, MaskBlock)):
                byte = start * MaskBlock + tl.arange(0, MaskBlock)
                reference_byte = Degree * MasksOffset + row * mask_bytes + byte
                original_byte = tl.load(
                    Collected + reference_byte, byte < mask_bytes, 0
                )
                other_byte = tl.load(
                    Collected + reference_byte + consumer * Rows * mask_bytes,
                    byte < mask_bytes,
                    0,
                )
                equal = equal & (
                    tl.sum((original_byte != other_byte).to(tl.int32), axis=0) == 0
                )
    tl.store(Flags + consumer * Rows + row, equal.to(tl.uint8))


@triton.jit
def copy_canonical_prefill_scores_runtime(
    NativeScores,
    Collected,
    ExpectedRequest,
    Scores,
    Rows,
    Width,
    LocalWidth,
    NativeWidth,
    NativeStride,
    Degree: tl.constexpr,
    OwnerRank: tl.constexpr,
    RequestsOffset,
    LengthsOffset,
    MasksOffset,
    HasMask: tl.constexpr,
    Block: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * Block + tl.arange(0, Block)
    row, local_column = offset // LocalWidth, offset % LocalWidth
    in_row = row < Rows
    request = tl.load(
        (Collected + Degree * RequestsOffset).to(tl.pointer_type(tl.int64)) + row,
        in_row,
        0,
    )
    expected = tl.load(ExpectedRequest)
    tl.device_assert(
        (~in_row) | (request == expected),
        "Native prefill Index keys belong to one shared request",
    )
    length = tl.load(
        (Collected + Degree * LengthsOffset).to(tl.pointer_type(tl.int64)) + row,
        in_row,
        0,
    )
    column = OwnerRank + Degree * local_column
    valid = (
        in_row
        & (local_column < NativeWidth)
        & (column < Width)
        & (column < length)
        & (request == expected)
    )
    if HasMask:
        packed = tl.load(
            Collected + Degree * MasksOffset + row * tl.cdiv(Width, 8) + column // 8,
            valid,
            0,
        )
        valid = valid & (((packed >> (column % 8)) & 1) != 0)
    score = tl.load(
        NativeScores + row * NativeStride + local_column, valid, -float("inf")
    )
    tl.store(Scores + offset, score, in_row)


@triton.jit
def restore_peer_scores_reuse_runtime(
    PeerPointers,
    Lengths,
    Output,
    Flags,
    Rows,
    Width,
    LocalWidth,
    OutputStride,
    Degree: tl.constexpr,
    Consumer: tl.constexpr,
    Block: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * Block + tl.arange(0, Block)
    length = tl.load(Lengths + row)
    valid = (column < Width) & (column < length)
    owner, local_column = column % Degree, column // Degree
    source = tl.load(PeerPointers + owner, valid, 0).to(tl.pointer_type(tl.float32))
    duplicate = tl.load(Flags + Consumer * Rows + row) != 0
    consumer = tl.where(duplicate, 0, Consumer)
    offset = (consumer * Rows + row).to(tl.int64) * LocalWidth + local_column
    score = tl.load(source + offset, valid, -float("inf"), cache_modifier=".cg")
    tl.store(
        Output + row.to(tl.int64) * OutputStride + column, score, column < OutputStride
    )
