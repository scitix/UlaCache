"""Batch independent native FP32 row-block scale multiplications."""


def aligned_fp32_scales(values, scales, block_size):
    import torch

    if (
        len(block_size) != 2
        or block_size[0] != 1
        or block_size[1] <= 0
        or values.ndim != 2
        or scales.ndim != 2
        or scales.dtype != torch.float32
    ):
        return False
    rows, columns = values.shape
    width = block_size[1]
    return (
        rows > 0
        and columns > 0
        and columns % width == 0
        and tuple(scales.shape) == (rows, columns // width)
        and scales.device == values.device
    )


def dequantize_aligned_rows(values, scales, block_size):
    import torch

    if not aligned_fp32_scales(values, scales, block_size):
        raise ValueError("Expected aligned row blocks with FP32 native scales")
    rows, columns = values.shape
    width = block_size[1]
    result = values.to(torch.float32)
    result.view(rows, columns // width, width).mul_(scales.unsqueeze(-1))
    return result
