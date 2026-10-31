"""Ordinary tensor copies of native Index bytes, without re-quantization.

The caller owns registered staging, native logical allocation and remote-write
retirement. Device work uses the current stream; that stream must finish before
the common target/draft/Index receive lease is released. This module imports no
device runtime until publication is explicitly requested.
"""

from ulacache.transfer.index_pages import (
    KEY_BYTES,
    SCALE_BYTES,
    validate_index_record_plan,
)


def publish_index_record_tensors(
    source, destination, plan, *, transport_quiesced, copy_chunk_records=4096
):
    """Publish selected key/scale bytes to an existing canonical native bank.

    Source storage must be contiguous CPU uint8. Destination storage must be
    contiguous CPU or CUDA uint8. Copy only planned bytes, preserving all other
    records, partial-page tails and scale payload bits. All coordinates and
    bounds are validated before the first destination write. This does not
    establish a native RDMA, peer-drain or GPU correctness acceptance.
    """
    import torch

    if transport_quiesced is not True:
        raise RuntimeError("Index publication requires retired remote writes")
    if (
        not isinstance(source, torch.Tensor)
        or not isinstance(destination, torch.Tensor)
        or source.dtype != torch.uint8
        or destination.dtype != torch.uint8
        or source.device.type != "cpu"
        or destination.device.type not in ("cpu", "cuda")
        or not source.is_contiguous()
        or not destination.is_contiguous()
    ):
        raise ValueError(
            "Index publication requires contiguous native CPU receive and byte banks"
        )
    if (
        source.device == destination.device
        and source.untyped_storage().data_ptr()
        == destination.untyped_storage().data_ptr()
    ):
        raise ValueError("Receive staging and canonical Index storage must not alias")
    if type(copy_chunk_records) is not int or copy_chunk_records < 1:
        raise ValueError("Index copy chunk must contain a positive number of records")
    plans = validate_index_record_plan(plan, source.nbytes, destination.nbytes)
    if not plans:
        return 0
    width = KEY_BYTES + SCALE_BYTES
    count = min(copy_chunk_records, len(plans))
    staging = torch.empty(
        (count, width),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=destination.device.type == "cuda",
    )
    payload = torch.empty((count, width), dtype=torch.uint8, device=destination.device)
    column = torch.arange(width, dtype=torch.int64)
    key_columns = column < KEY_BYTES
    source_flat, destination_flat = source.reshape(-1), destination.reshape(-1)
    for start in range(0, len(plans), count):
        chunk = plans[start : start + count]
        source_key = torch.tensor([item.source_key_byte for item in chunk])
        source_scale = torch.tensor([item.source_scale_byte for item in chunk])
        destination_key = torch.tensor([item.destination_key_byte for item in chunk])
        destination_scale = torch.tensor(
            [item.destination_scale_byte for item in chunk]
        )
        source_indices = torch.where(
            key_columns,
            source_key[:, None] + column,
            source_scale[:, None] + column - KEY_BYTES,
        )
        destination_indices = torch.where(
            key_columns,
            destination_key[:, None] + column,
            destination_scale[:, None] + column - KEY_BYTES,
        )
        temporary = staging[: len(chunk)].reshape(-1)
        torch.index_select(source_flat, 0, source_indices.reshape(-1), out=temporary)
        # A blocking host copy retains the temporary until its H2D transfer has
        # finished. The caller still owns the final device scatter/lease fence.
        device_payload = payload[: len(chunk)].reshape(-1)
        device_payload.copy_(temporary)
        destination_flat.index_copy_(
            0, destination_indices.reshape(-1).to(destination.device), device_payload
        )
    return len(plans)
