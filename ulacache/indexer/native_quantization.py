"""Compile the packaged native owner encoders through an explicit service."""

from pathlib import Path


def load_owner_index_module(kind, arguments, *, request_map, compile_device_module):
    """Keep owner kernel bodies in the library and native compiler policy intact."""
    entries = {
        "norm_rope": (
            "indexer_k.cuh",
            "dsa_owner_k_indexer_norm_rope_store",
            "FusedKIndexerNormRopeStoreKernel",
            "forward",
            "forward_owner",
        ),
        "quantized": (
            "fused_store_index_cache.cuh",
            "fused_store_owner_index_k_cache",
            "FusedStoreCacheIndexerKernel",
            "store",
            "run_owner",
        ),
    }
    if kind not in entries or type(request_map) is not bool:
        raise ValueError("Declare a supported owner encoder and explicit page mapping")
    if not callable(compile_device_module):
        raise TypeError("Bind the native CUDA compiler before owner Index capture")
    filename, name, kernel, export, method = entries[kind]
    native = Path(__file__).resolve().parents[1] / "csrc"
    header = native / "owner_index" / filename
    if request_map:
        method += "_mapped"
    return compile_device_module(
        name,
        *arguments,
        cuda_files=[str(header.resolve(strict=True))],
        cuda_wrappers=[(export, f"{kernel}<{arguments}>::{method}")],
        extra_include_paths=[str(native / "include")],
    )
