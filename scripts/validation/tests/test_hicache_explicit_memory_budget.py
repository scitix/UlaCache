"""Explicit HiCache sizing reserves one shared startup memory snapshot."""
import ast
import __future__
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]


def patched_file(relative):
    """Read the exact patched native source used by the installation check."""
    import os
    source = os.environ.get('HISPARSE_NATIVE_SOURCE_ROOT')
    if not source:
        pytest.skip('Set HISPARSE_NATIVE_SOURCE_ROOT to the exact patched official source')
    return (Path(source) / relative).read_text()


def functions(source, names):
    tree = ast.parse(source)
    return ast.Module(body=[node for node in tree.body
                           if isinstance(node, ast.FunctionDef) and node.name in names],
                      type_ignores=[])


@pytest.mark.parametrize('distributed', [False, True])
def test_explicit_pools_share_one_budget_without_resizing(distributed):
    events = []
    memory = SimpleNamespace(hicache_host_memory_fraction=None, hicache_ratio=2.7188)
    state = ContextVar('host-memory-test', default=None)

    class Tensor:
        def __init__(self, values): self.value = values[0]
        def item(self): return self.value

    def sample(**kwargs):
        events.append('sample')
        return 8 * 200_000_000_000 if events.count('sample') == 1 else 8_000_000_000

    def reduce(value, **kwargs):
        assert kwargs == dict(op='min', group='world-cpu')
        events.append('collective')
        value.value = 180_000_000_000

    namespace = dict(contextmanager=contextmanager, _host_memory_budget=state,
        available_host_memory_bytes=sample, HICACHE_HOST_MEMORY_RESERVE_BYTES=0,
        ranks_per_host=lambda: 8,
        torch=SimpleNamespace(int64='int64', tensor=lambda values, **kwargs: Tensor(values),
            distributed=SimpleNamespace(is_initialized=lambda: distributed,
                all_reduce=reduce, ReduceOp=SimpleNamespace(MIN='min'))),
        get_memory=lambda: memory,
        get_parallel=lambda: SimpleNamespace(world_group=SimpleNamespace(cpu_group='world-cpu')))
    native = patched_file('python/sglang/srt/mem_cache/pool_host/base.py')
    exec(compile(functions(native, {'host_memory_budget_scope', 'host_memory_budget_bytes'}),
                 'host_budget', 'exec', flags=__future__.annotations.compiler_flag), namespace)
    sizing = patched_file('python/sglang/srt/mem_cache/hicache_auto_size.py')
    exec(compile(functions(sizing, {'auto_size_hicache'}), 'hicache_sizing', 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    expected = (180 if distributed else 200) * 1_000_000_000
    with namespace['auto_size_hicache'](None, None, enabled=True):
        assert namespace['host_memory_budget_bytes'](121_000_000_000) == expected
        assert namespace['host_memory_budget_bytes'](6_000_000_000) == expected - 121_000_000_000
        assert state.get() == expected - 127_000_000_000
    assert state.get() is None
    assert events == ['sample', *(['collective'] if distributed else [])]
    assert memory.hicache_ratio == 2.7188
