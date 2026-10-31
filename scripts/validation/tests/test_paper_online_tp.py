"""The online TP launcher preserves the recorded experiment, not decode defaults."""
import json
from pathlib import Path
import subprocess


def test_offline_dpa_restores_paper_capacity_without_online_windows(tmp_path):
    root = Path(__file__).resolve().parents[3]
    for experiment in ('7_2_throughput', '7_3_low_latency',
                       '7_4_capacity', '7_5_scheduler'):
        logs = tmp_path / experiment
        result = subprocess.run([
            'bash', str(root/'scripts'/experiment/'start_ulacache.sh'),
            '--model', '/models/glm', '--mode', 'dpa', '--spec', 'on',
            '--log-dir', str(logs), '--kv-dir', str(tmp_path/'kv'), '--dry-run',
        ], check=True, capture_output=True, text=True)
        config = json.loads((logs/'ulacache_config.json').read_text())
        assert config['capacity']['device_token_capacity_per_owner'] == 660672
        assert config['capacity']['host_token_capacity_per_owner'] == 2501504
        assert '--max-total-tokens 3162112' in result.stdout
        assert '--max-running-requests 256' in result.stdout


def test_hisparse_tp_keeps_all_request_buffers_with_bounded_spare_pages(tmp_path):
    import shlex
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([
        'bash', str(root/'scripts/7_1_online/start_hisparse.sh'),
        '--model', '/models/glm', '--mode', 'tp', '--spec', 'on',
        '--log-dir', str(tmp_path/'logs'), '--kv-dir', str(tmp_path/'kv'),
        '--dry-run',
    ], check=True, capture_output=True, text=True)
    command = shlex.split(result.stdout.splitlines()[-1])
    physical = int(command[command.index('--max-total-tokens')+1])
    config = json.loads(command[command.index('--hisparse-config')+1])
    # 64 native request slots retain the 16K hot buffer and 1K MTP scratch.
    assert config['device_buffer_size'] == 16384
    assert physical >= 64 * (16384 + 64 + 1024)
    assert round(physical * config['host_to_device_ratio']) == 2500992


def test_online_tp_configuration_matches_recorded_paper(tmp_path):
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([
        'bash', str(root/'scripts/7_1_online/start_ulacache.sh'),
        '--model', '/models/glm', '--mode', 'tp', '--spec', 'on',
        '--log-dir', str(tmp_path/'logs'), '--kv-dir', str(tmp_path/'kv'),
        '--dry-run',
    ], check=True, capture_output=True, text=True)
    options = json.loads((tmp_path/'logs/ulacache_config.json').read_text())['options']
    assert options['prefill_window_rows'] == 1048576
    assert options['cpu_slots'] == 2500416
    assert options['owner_index_draft'] is True
    assert options['local_eviction_policy'] == 'circular'
    assert options['cache_capacity_multiplier'] == 3.0
    assert '--max-running-requests 64' in result.stdout
    assert '--cuda-graph-max-bs-decode 64' in result.stdout
    assert '--max-total-tokens 23128000' in result.stdout
    assert '--mem-fraction-static 0.94' in result.stdout
    assert '--chunked-prefill-size 8192' in result.stdout


def test_full_history_native_reader_does_not_split_topk(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ulacache.cache.tensor_parallel import NativeTensorParallelExecution
    plan = SimpleNamespace(map_native_rows=Mock(return_value='mapped'))
    backend = SimpleNamespace(plan=plan, read=Mock(return_value='window'),
                              after_attention=Mock())
    execution = NativeTensorParallelExecution.__new__(NativeTensorParallelExecution)
    execution.pool = SimpleNamespace(_tp_full_prefill=backend)
    native = Mock(return_value='output')
    assert execution.attend_native_prefill_records(
        4, 'queries', 'ids', 'lengths', native) == 'output'
    native.assert_called_once_with('queries', 'mapped', 'window', 'lengths')
    plan.map_native_rows.assert_called_once_with('ids')
    backend.read.assert_called_once_with(4)
    backend.after_attention.assert_called_once_with(4)


def test_hisparse_prefix_startup_installs_the_shared_prefill_workspace(monkeypatch):
    import ast
    import os
    import sys
    from types import ModuleType

    source = os.environ.get('HISPARSE_NATIVE_SOURCE_ROOT')
    if source is None:
        import pytest
        pytest.skip('Set HISPARSE_NATIVE_SOURCE_ROOT to the patched checkout')
    path = Path(source)/'python/sglang/srt/ulacache/entry.py'
    function = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'install_bindings')
    calls = []
    for name in ['v0522_fp8_loading', 'agent_hicache_shared_index_adapter',
                 'hisparse_prefill_relay_adapter', 'hisparse_prefix_adapter',
                 'hisparse_prefill_adapter']:
        module = ModuleType(name)
        module.install = lambda name=name: calls.append(name)
        monkeypatch.setitem(sys.modules, 'sglang.srt.ulacache.'+name, module)
    monkeypatch.setenv('UDCACHE_MATRIX_CACHE_KIND', 'hisparse')
    monkeypatch.setenv('ULACACHE_BENCH_PREFIX_CACHE', '1')
    for name in ['ULACACHE_CONFIGURATION', 'UDCACHE_SPARSE_TRACE_DIR',
                 'UDCACHE_MATRIX_AUDIT_DIR', 'ULACACHE_BENCH_PHASE']:
        monkeypatch.delenv(name, raising=False)
    namespace = dict(os=os)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    namespace['install_bindings']()
    assert calls[-2:] == ['hisparse_prefix_adapter', 'hisparse_prefill_adapter']


def test_legacy_binding_does_not_charge_native_pipeline_twice():
    import ast
    import os
    from types import SimpleNamespace
    import pytest

    source = os.environ.get('HISPARSE_NATIVE_SOURCE_ROOT')
    if source is None:
        pytest.skip('Set HISPARSE_NATIVE_SOURCE_ROOT to the patched checkout')
    path = Path(source)/'python/sglang/srt/ulacache/tensor_parallel_prefill.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == 'fixed')
    namespace = dict(original_fixed=lambda *_: 100,
                     full_history_lookup_upper_bytes=lambda _: 24)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    layout = SimpleNamespace(row_bytes=656, disaggregation_mode='null',
        prefill_window_rows=1048576, prefill_layers=2,
        prefill_buffer_bytes=688, prefill_lookup_bytes=80)
    configurator = SimpleNamespace(record_template=layout, cpu_enabled=True,
                                   request_policy=False)
    assert namespace['fixed'](configurator, 64) == 100
    layout.prefill_layers = 1
    assert namespace['fixed'](configurator, 64) == 100 + 688 + 24 - 80


def test_hisparse_workspace_enters_the_actual_native_forward(monkeypatch):
    import ast
    from contextlib import contextmanager
    import os
    import sys
    from types import ModuleType, SimpleNamespace
    import pytest

    source = os.environ.get('HISPARSE_NATIVE_SOURCE_ROOT')
    if source is None:
        pytest.skip('Set HISPARSE_NATIVE_SOURCE_ROOT to the patched checkout')
    path = Path(source)/'python/sglang/srt/mem_cache/flowkv_forward.py'
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef)
             and node.name in ('scoped_raw_model_forward', 'bind_native_model_forward')]
    pool = SimpleNamespace(hisparse_prefill_workspace=object())
    req_table = object()
    module = ModuleType('sglang.srt.model_executor.forward_context')
    module.get_token_to_kv_pool = lambda: pool
    module.get_req_to_token_pool = lambda: SimpleNamespace(req_to_token=req_table)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    events = []

    @contextmanager
    def scope(p, batch, table):
        assert p is pool and table is req_table
        events.append('enter')
        yield
        events.append('exit')

    namespace = dict(FlowKVSharedIndexAdapter=type('OtherPool', (), {}),
                     native_cache_adapter=lambda _: None, flowkv_forward=scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)

    def raw(*args, **kwargs):
        events.append('model')
        return 'output'

    decorated = namespace['scoped_raw_model_forward'](raw)
    assert decorated(object(), 'ids', 'positions', 'batch') == 'output'
    assert events == ['enter', 'model', 'exit']
    events.clear()
    for name in ['ULACACHE_CONFIGURATION', 'ULACACHE_GLM_FLASH_CONFIGURATION',
                 'ULACACHE_NATIVE_DSA_CONFIGURATION', 'ULACACHE_NATIVE_PAGED_CONFIGURATION']:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('ULACACHE_BENCH_PREFIX_CACHE', '1')
    model = SimpleNamespace(forward=raw)
    namespace['bind_native_model_forward'](model)
    assert model.forward('ids', 'positions', 'batch') == 'output'
    assert events == ['enter', 'model', 'exit']
