"""Registered host KV must retire before the scheduler process is reaped."""
import ast
import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


def source_file(relative):
    root = os.environ.get('HISPARSE_NATIVE_SOURCE_ROOT')
    if root is None:
        pytest.skip('Set HISPARSE_NATIVE_SOURCE_ROOT to the patched checkout')
    return Path(root)/'python/sglang/srt'/relative


def test_registered_host_resources_close_once_after_gpu_readers(monkeypatch):
    path = source_file('mem_cache/flowkv_forward.py')
    function = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef)
                    and n.name == 'release_native_cache_host_resources')
    events = []
    monkeypatch.setitem(sys.modules, 'torch', NS(cuda=NS(
        synchronize=lambda device: events.append(('sync', device)))))
    cache = NS(close=lambda: events.append('cache'))
    host = NS(close=lambda: events.append('host'))
    pool = NS(device='cuda:0', memory_cache=cache, ulacache=cache, flowkv_host=host)
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    close = namespace[function.name]
    close(pool)
    close(pool)
    assert events == [('sync', 'cuda:0'), 'cache', 'host']
    assert pool.flowkv_host is None
    close(NS())
    assert len(events) == 3


@pytest.mark.parametrize('timeout', [0, 15])
@pytest.mark.parametrize('tracked_handles', [False, True])
def test_shutdown_zero_waits_for_workers_before_reaping(timeout, tracked_handles):
    path = source_file('managers/tokenizer_manager.py')
    tree = ast.parse(path.read_text())
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.AsyncFunctionDef)
                    and n.name == 'sigterm_watchdog')
    live_method = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef)
                       and n.name == '_live_scheduler_processes')
    calls = []
    worker = NS(pid=123, is_running=lambda: bool(collect()), status=lambda: 'running')
    remaining = [[worker], [worker], [], []]
    clock = iter([0, 100, 200, 300])

    def collect():
        value = remaining.pop(0) if remaining else []
        calls.append(('collect', bool(value)))
        return value

    def reap(*args, **kwargs):
        calls.append(('reap', calls[-1][1]))

    namespace = dict(asyncio=asyncio, os=os, sys=sys, logger=Mock(),
        psutil=NS(STATUS_ZOMBIE='zombie', NoSuchProcess=ProcessLookupError),
        get_bool_env_var=lambda _: False, ServerStatus=NS(UnHealthy='bad'),
        ShutdownReq=object, _SCHEDULER_EXIT_TIMEOUT_SECS=timeout,
        # Titles can remain "python". Owned multiprocessing handles must
        # still delay reaping until that worker has released its host KV.
        collect_scheduler_processes=(lambda: []) if tracked_handles else collect,
        kill_process_tree=reap,
        time=NS(monotonic=lambda: next(clock), sleep=lambda _: None))
    exec(compile(ast.Module(body=[function, live_method], type_ignores=[]), str(path), 'exec'), namespace)
    manager = NS(gracefully_exit=True, rid_to_state={}, server_status='healthy',
        _subprocess_watchdog=Mock(), _dispatch_to_scheduler=Mock(),
        _server_stop_hook=Mock(), asyncio_tasks=[],
        _scheduler_processes=(worker,) if tracked_handles else None)
    manager._live_scheduler_processes = lambda: namespace['_live_scheduler_processes'](manager)
    asyncio.run(namespace[function.name](manager))
    reaped_while_worker_alive = next(value for kind, value in calls if kind == 'reap')
    assert reaped_while_worker_alive is (timeout != 0)
    manager._subprocess_watchdog.stop.assert_called_once()
    manager._server_stop_hook.assert_called_once()


@pytest.mark.parametrize('dp_workers', [None, [11, 12]])
def test_launcher_tracks_workers_without_waiting_for_dp_router(dp_workers):
    path = source_file('entrypoints/engine.py')
    tree = ast.parse(path.read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
        and any(isinstance(child, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == '_scheduler_processes'
                        for t in child.targets) for child in ast.walk(n)))
    # Pick the smallest containing block, rather than executing startup.
    candidates = [n for n in ast.walk(assignment) if isinstance(n, ast.If)
        and any(isinstance(child, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == '_scheduler_processes'
                        for t in child.targets) for child in ast.walk(n))]
    assignment = min(candidates, key=lambda n: n.end_lineno - n.lineno)
    manager = NS()
    namespace = dict(tokenizer_manager=manager,
        scheduler_procs=[NS(pid=p) for p in ([99, 11, 12] if dp_workers else [11, 12])],
        scheduler_init_result=NS(scheduler_infos=[{'workers': dp_workers}] if dp_workers else [{}]),
        SCHEDULER_PIDS_ARG='workers', psutil=NS(Process=lambda pid: pid))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), 'exec'), namespace)
    assert manager._scheduler_processes == (11, 12)
