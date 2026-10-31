"""Native FP8 sink wiring preserves padding, schedules and caller-owned tensors."""

import ast
import os
from pathlib import Path
import sys
import textwrap
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def native_sink(monkeypatch):
    root = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if root is None:
        pytest.skip("Requires the patched native checkout")
    path = Path(root) / "python/sglang/srt/layers/attention/dsa_backend.py"
    source = path.read_text()
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekSparseAttnBackend"
    )
    method = next(
        (
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_forward_flashmla_kv_native_sink"
        ),
        None,
    )
    if method is None:
        pytest.skip("The installed patch still uses the runtime sink adapter")
    namespace = {}
    exec(
        compile(
            textwrap.dedent(ast.get_source_segment(source, method)), str(path), "exec"
        ),
        namespace,
    )
    calls = []

    def flash(**kwargs):
        calls.append(kwargs)
        return kwargs["q"][..., : kwargs["head_dim_v"]].clone(), None

    module = ModuleType("sgl_kernel.flash_mla")
    module.FlashMLASchedMeta = NS
    module.flash_mla_with_kvcache = flash
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return namespace["_forward_flashmla_kv_native_sink"], calls


@pytest.mark.parametrize("padded_heads", [2, 4])
def test_sink_padding_keeps_query_and_kv_bytes(native_sink, padded_heads):
    method, calls = native_sink
    query = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    kv = torch.arange(64 * 656).to(torch.uint8)
    original_query, original_kv = query.clone(), kv.clone()
    sink = torch.tensor([3.0, 7.0])
    metadata = NS(
        flashmla_metadata=NS(num_splits=torch.zeros(4), flashmla_metadata="tiles")
    )
    backend = NS(
        dsa_kv_cache_store_fp8=True,
        kv_cache_dim=656,
        physical_page_size=64,
        flashmla_kv_num_q_heads=padded_heads,
        _sink_pad_cache={},
        flashmla_kv_format="v32",
    )
    output = method(
        backend,
        query,
        kv,
        2,
        0.5,
        NS(tp_q_head_num=2, head_dim=4),
        metadata,
        torch.zeros((3, 2), dtype=torch.int32),
        sink,
    )
    assert torch.equal(output, query[:, None, :, :2])
    assert torch.equal(query, original_query) and torch.equal(kv, original_kv)
    assert calls[0]["k_cache"].data_ptr() == kv.data_ptr()
    assert torch.equal(calls[0]["attn_sink"][:2], sink)
    if padded_heads > 2:
        assert not torch.count_nonzero(calls[0]["q"][:, :, 2:])
        assert not torch.count_nonzero(calls[0]["attn_sink"][2:])


def test_actual_query_schedule_does_not_mutate_outer_metadata(native_sink):
    method, calls = native_sink
    metadata = NS(
        flashmla_metadata=NS(num_splits=torch.zeros(6), flashmla_metadata="outer"),
        dsa_cache_seqlens_int32=torch.tensor(
            [64, 128, 192, 256, 320], dtype=torch.int32
        ),
    )
    schedule = NS(num_splits=torch.zeros(4), flashmla_metadata="narrowed")
    compute = Mock(return_value=schedule)
    backend = NS(
        dsa_kv_cache_store_fp8=True,
        kv_cache_dim=656,
        physical_page_size=64,
        flashmla_kv_num_q_heads=2,
        _sink_pad_cache={},
        flashmla_kv_format="v32",
        _compute_flashmla_metadata=compute,
    )
    args = (
        backend,
        torch.ones((3, 2, 4)),
        torch.zeros(64 * 656),
        2,
        0.5,
        NS(tp_q_head_num=2, head_dim=4),
        metadata,
        torch.zeros((3, 2)),
        torch.ones(2),
    )
    method(*args)
    method(*args)
    assert compute.call_count == 1
    assert compute.call_args.kwargs["cache_seqlens"].tolist() == [64, 128, 192]
    assert metadata.flashmla_metadata.flashmla_metadata == "outer"
    assert calls[-1]["tile_scheduler_metadata"].tile_scheduler_metadata == "narrowed"


@pytest.mark.parametrize(
    "override,expected",
    [
        ({}, True),
        ({"tp_size": 4}, False),
        ({"dcp_size": 2}, False),
        ({"page_size": 128}, False),
        ({"kv_cache_dtype": "bf16"}, False),
        ({"dsa_prefill_backend": "flashmla_sparse"}, False),
    ],
)
def test_fp8_sink_argument_policy_keeps_its_supported_geometry(
    monkeypatch, override, expected
):
    root = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if root is None:
        pytest.skip("Requires the patched native checkout")
    path = Path(root) / "python/sglang/srt/arg_groups/overrides.py"
    source = path.read_text()
    method = next(
        (
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_hy4_scaled656_sink_supported"
        ),
        None,
    )
    if method is None:
        pytest.skip("The installed patch still uses runtime argument adaptation")
    namespace = {}
    exec(compile(ast.get_source_segment(source, method), str(path), "exec"), namespace)
    module = ModuleType("sglang.srt.layers.attention.dsa_backend")
    module.DeepseekSparseAttnBackend = NS(supports_scaled_fp8_sink=True)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    view = dict(
        tp_size=8,
        dcp_size=1,
        page_size=64,
        kv_cache_dtype="fp8_e4m3",
        dsa_prefill_backend="flashmla_kv",
        dsa_decode_backend="flashmla_kv",
    )
    view.update(override)
    config = NS(
        architectures=["HYV4ForCausalLM"], kv_lora_rank=512, qk_rope_head_dim=64
    )
    assert namespace["_hy4_scaled656_sink_supported"](NS(**view), config) is expected
