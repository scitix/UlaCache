"""Exercise the real native transfer method on CPU, without claiming CUDA coverage."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


FIXTURES = Path(__file__).parent / "fixtures"


def native_method(which, copy):
    namespace = {"copy_all_layer_kv_cache_func": copy}
    path = FIXTURES / f"native_hisparse_device_copy_{which}.py"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace["transfer_values_on_device"]


@pytest.mark.parametrize("count", [0, 1, 31, 32, 33, 256, 288, 384])
def test_native_copy_keeps_all_rows_and_bounds_compiler_tile(count):
    # Seventy-eight separate native layer buffers; inspect every encoded byte.
    generator = torch.Generator().manual_seed(163 + count)
    data = torch.randint(256, (78, 1024, 656), dtype=torch.uint8, generator=generator)
    before = data.clone()
    source = torch.arange(count, dtype=torch.int64).flip(0)
    target = torch.arange(512, 512 + count, dtype=torch.int64)
    calls = []

    def copy(_ptrs, _strides, dst, src, rows, upper, config):
        calls.append((rows, upper))
        assert rows == src.numel() == dst.numel()
        assert upper <= 32
        assert config == {"bytes_per_tile": 128, "byte_tiles": 6, "num_warps": 4}
        data[:, dst] = data[:, src].clone()

    pool = SimpleNamespace(bytes_per_token=656, data_ptrs=None, data_strides=None)
    native_method("after", copy)(pool, target, source)
    assert sum(rows for rows, _ in calls) == count
    assert torch.equal(data[:, target], before[:, source])
    unchanged = torch.ones(1024, dtype=torch.bool)
    unchanged[target] = False
    assert torch.equal(data[:, unchanged], before[:, unchanged])


def test_native_self_copy_and_tail_do_not_change_payload():
    calls = []

    def copy(_ptrs, _strides, dst, src, rows, upper, _config):
        assert torch.equal(dst, src)
        calls.append((rows, upper))

    pool = SimpleNamespace(bytes_per_token=656, data_ptrs=None, data_strides=None)
    indices = torch.tensor([0, 17, 1023] * 11, dtype=torch.int64)
    native_method("after", copy)(pool, indices, indices)
    assert calls == [(32, 32), (1, 1)]


def test_original_large_verify_exposes_spilling_256_row_variant():
    calls = []

    def copy(_ptrs, _strides, _dst, _src, rows, upper, _config):
        calls.append((rows, upper))

    pool = SimpleNamespace(bytes_per_token=656, data_ptrs=None, data_strides=None)
    indices = torch.arange(96 * 3, dtype=torch.int64)
    native_method("before", copy)(pool, indices + 512, indices)
    assert calls == [(256, 256), (32, 32)]
