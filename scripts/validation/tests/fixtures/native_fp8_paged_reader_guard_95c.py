# Actual frozen95c native dequant_k_cache.py SHA9410e820956e9a1e683d06723024aae244f0e5c5658c1aab282ee5b5d69676d5
# Execute its original pre-kernel shape/view/dtype contract on CPU.
def dequantize_k_cache_paged(quant_k_cache: torch.Tensor, page_table_1_flattened: torch.Tensor, group_size: int=128) -> torch.Tensor:
    """
    De-quantize the k-cache with paged layout
    Args:
        quant_k_cache: [total_num_tokens, 1, dim_quant] or [num_blocks, block_size, 1, dim_quant], the quantized k-cache in paged layout
        page_table_1_flattened: [num_tokens], the flattened page_table_1 with the page indices in each requests concatenated together
    Returns:
        output: [num_tokens, 1, dim_nope + dim_rope], the de-quantized k-cache
    """
    dim_quant = quant_k_cache.shape[-1]
    (dim_nope, dim_rope) = _infer_dsa_dims(dim_quant)
    quant_k_cache = quant_k_cache.view((-1, dim_quant))
    num_tokens = page_table_1_flattened.shape[0]
    assert quant_k_cache.dtype == torch.float8_e4m3fn
    return quant_k_cache
