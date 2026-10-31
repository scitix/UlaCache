"""Publish registered complete records directly to their reserved destinations.

The caller owns transport completion, directory stability and the final stream
visibility boundary. Host pointer tables are provided by the registered host
allocator; ordinary CPU addresses are never assumed to be device accessible.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["Count"])
def _publish_records(
    SourceRows,
    Locations,
    LayerGroups,
    SourcePointers,
    HostPointers,
    GPU,
    Count,
    SourceLayerBytes: tl.constexpr,
    HostLayerBytes: tl.constexpr,
    GPULayerBytes: tl.constexpr,
    RowBytes: tl.constexpr,
    Block: tl.constexpr,
):
    record, layer = tl.program_id(0), tl.program_id(1)
    source_row = tl.load(SourceRows + record).to(tl.int64)
    group = tl.load(LayerGroups + layer)
    location = tl.load(Locations + group * Count + record).to(tl.int64)
    source = tl.load(SourcePointers).to(tl.pointer_type(tl.uint8))
    source += layer.to(tl.int64) * SourceLayerBytes + source_row * RowBytes
    if location >= 0:
        target = GPU + layer.to(tl.int64) * GPULayerBytes + location * RowBytes
    else:
        host = tl.load(HostPointers).to(tl.pointer_type(tl.uint8))
        target = host + layer.to(tl.int64) * HostLayerBytes + (-location - 2) * RowBytes
    columns = tl.arange(0, Block)
    payload = tl.load(source + columns, columns < RowBytes, other=0)
    tl.store(target + columns, payload, columns < RowBytes)


@triton.jit
def _publish_sidecar(
    SourceRows,
    DestinationRows,
    SourcePointers,
    Destination,
    SourceLayer: tl.constexpr,
    SourceLayerBytes: tl.constexpr,
    RowBytes: tl.constexpr,
    Block: tl.constexpr,
):
    record = tl.program_id(0)
    source_row = tl.load(SourceRows + record).to(tl.int64)
    destination_row = tl.load(DestinationRows + record).to(tl.int64)
    source = tl.load(SourcePointers).to(tl.pointer_type(tl.uint8))
    source += SourceLayer * SourceLayerBytes + source_row * RowBytes
    columns = tl.arange(0, Block)
    payload = tl.load(source + columns, columns < RowBytes, other=0)
    tl.store(
        Destination + destination_row * RowBytes + columns, payload, columns < RowBytes
    )


def publish_registered_records(
    source_rows, locations, layer_groups, source_storage, host_storage, gpu
):
    """Copy one bounded chunk, using an already validated residency snapshot."""
    if source_rows.numel():
        width = gpu.shape[-1]
        _publish_records[(source_rows.numel(), gpu.shape[0])](
            source_rows,
            locations,
            layer_groups,
            source_storage.ptrs,
            host_storage.ptrs,
            gpu,
            source_rows.numel(),
            source_storage.storage.stride(0),
            host_storage.storage.stride(0),
            gpu.stride(0),
            width,
            triton.next_power_of_2(width),
            num_warps=4,
        )


def publish_registered_sidecar(
    source_rows, destination_rows, source_storage, source_layer, destination
):
    """Publish a complete canonical sidecar under the same receive lease."""
    if source_rows.numel():
        width = destination.shape[-1]
        _publish_sidecar[(source_rows.numel(),)](
            source_rows,
            destination_rows,
            source_storage.ptrs,
            destination,
            source_layer,
            source_storage.storage.stride(0),
            width,
            triton.next_power_of_2(width),
            num_warps=4,
        )
