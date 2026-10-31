# Extracted from immutable f198 native HiSparseDSATokenToKVPool.
# Full parent file SHA256: 0c6680c5389c5d065cb9bd6956058ff6d3935495657b034b4bca74839633beda
import torch

def transfer_values_on_device(self, dst_indices: torch.Tensor, src_indices: torch.Tensor) -> None:
    """Copy target MLA KV between physical slots, without remapping IDs."""
    if src_indices.numel() == 0:
        return
    config = {'bytes_per_tile': 128, 'byte_tiles': (self.bytes_per_token + 127) // 128, 'num_warps': 4}
    for start in range(0, src_indices.numel(), 32):
        src = src_indices[start:start + 32].contiguous()
        dst = dst_indices[start:start + 32].contiguous()
        copy_all_layer_kv_cache_func(self.data_ptrs, self.data_strides, dst, src, src.numel(), 1 << (src.numel() - 1).bit_length(), config)
