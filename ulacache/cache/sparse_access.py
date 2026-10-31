"""CUDA address-only adaptation of native packed sparse regions."""

from ulacache.cache.sparse_regions import PackedRegionLayout


def _validate_tensors(packed, output, *, output_dtype):
    import torch

    if (
        packed.ndim != 2
        or packed.dtype != torch.int64
        or packed.stride(1) != 1
        or not packed.is_cuda
        or output.device != packed.device
        or output.dtype != output_dtype
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Native packed regions require CUDA int64 rows and contiguous same-device outputs"
        )


def expand_packed_region_records(
    packed, counts, active_rows, output, *, layout=PackedRegionLayout()
):
    import torch
    import triton
    from ulacache.cache.kernels.sparse_regions import expand_regions_kernel

    _validate_tensors(packed, output, output_dtype=torch.int32)
    if (
        packed.ndim != 2
        or packed.shape[1] != layout.region_slots_per_query
        or output.shape != (packed.shape[0], layout.selected_token_width)
        or counts.shape != (packed.shape[0],)
        or counts.dtype != torch.int32
        or not counts.is_contiguous()
        or counts.device != packed.device
        or active_rows.numel() != 1
        or active_rows.dtype != torch.int32
        or active_rows.device != packed.device
    ):
        raise ValueError(
            "Native region and output token dimensions must match the declared layout"
        )
    expand_regions_kernel[
        (packed.shape[0], triton.cdiv(layout.selected_token_width, 256))
    ](
        packed,
        counts,
        active_rows,
        output,
        packed.stride(0),
        TOKENS=layout.tokens_per_region,
        REGIONS=layout.region_slots_per_query,
        BLOCK=256,
    )
    return output


def remap_packed_region_records(
    packed, resident_records, output, *, layout=PackedRegionLayout()
):
    import torch
    import triton
    from ulacache.cache.kernels.sparse_regions import remap_regions_kernel

    _validate_tensors(packed, output, output_dtype=torch.int64)
    if (
        packed.shape != output.shape
        or packed.shape[1] != layout.region_slots_per_query
        or resident_records.shape != (packed.shape[0], layout.selected_token_width)
        or resident_records.dtype != torch.int32
        or resident_records.device != packed.device
        or not resident_records.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Native region remapping requires complete contiguous resident token rows"
        )
    remap_regions_kernel[
        (packed.shape[0], triton.cdiv(layout.region_slots_per_query, 256))
    ](
        packed,
        resident_records,
        output,
        packed.stride(0),
        TOKENS=layout.tokens_per_region,
        REGIONS=layout.region_slots_per_query,
        BLOCK=256,
    )
    return output
