"""Check the remap coverage floor and preserve untuned larger-query behavior."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from ulacache.bindings import CacheOptions


def load_pure_grid_function():
    # The sizing expression has no device dependency. Load only that function
    # so this port regression also runs without optional torch/CUDA packages.
    path = Path(__file__).resolve().parents[3] / 'ulacache/cache/tp_cache.py'
    node = next(n for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == 'pull_grid_blocks')
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
        str(path), 'exec'), namespace)
    return namespace['pull_grid_blocks']


@pytest.mark.parametrize('queries', [1, 2, 4, 8])
@pytest.mark.parametrize('topk', [128, 2048, 4096])
def test_low_query_grid_keeps_every_remap_column_covered(queries, topk):
    size = load_pure_grid_function()
    pool = SimpleNamespace(flowkv_layout=SimpleNamespace(topk=topk),
        _flowkv_pull_grid_budget=1056,
        udcache_services=SimpleNamespace(config=CacheOptions(
            prefetch=True, low_query_pull_blocks=16)))
    capped = size(pool, queries)
    assert capped * 128 >= topk
    pool.udcache_services.config = CacheOptions(prefetch=True)
    assert capped <= size(pool, queries)


@pytest.mark.parametrize('queries', [9, 12, 16, 32, 48, 64, 96, 128])
def test_larger_query_grids_are_identical_to_the_existing_policy(queries):
    size = load_pure_grid_function()
    pool = SimpleNamespace(flowkv_layout=SimpleNamespace(topk=2048),
        _flowkv_pull_grid_budget=1056,
        udcache_services=SimpleNamespace(config=CacheOptions(prefetch=True)))
    original = size(pool, queries)
    pool.udcache_services.config = CacheOptions(prefetch=True, low_query_pull_blocks=16)
    assert size(pool, queries) == original


@pytest.mark.parametrize('cap', [True, 0, -16, 15, 24, 1024, 32.0])
def test_invalid_grid_configuration_is_rejected_before_device_setup(cap):
    with pytest.raises(ValueError, match='Low-query pull blocks'):
        CacheOptions(prefetch=True, low_query_pull_blocks=cap)


def test_grid_tuning_requires_the_validated_prefetch_route():
    with pytest.raises(NotImplementedError, match='requires native prefetch'):
        CacheOptions(low_query_pull_blocks=16)
