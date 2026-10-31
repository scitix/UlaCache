"""Check native request-buffer and full-prefill coexistence at capacity edges."""
import pytest

from ulacache.tools.benchmark_budget import hisparse_buffer_rows, restored_cohort_metadata_tokens


def test_saved_next_tokens_fit_page_billed_native_admission():
    for requests in (1, 2, 4, 8, 16, 32, 64):
        remaining = restored_cohort_metadata_tokens(requests=requests, page_size=512)
        for _ in range(requests):
            assert remaining > 512
            remaining -= 512
        assert remaining == 512
    for requests, page in ((0, 512), (64, 0), (True, 512)):
        with pytest.raises(ValueError):
            restored_cohort_metadata_tokens(requests=requests, page_size=page)


def test_full_prefill_reserve_and_next_buffer_page():
    capacity=262656
    rows=hisparse_buffer_rows(physical_rows=capacity,requests=8,prefill_rows=131072)
    assert rows==16384
    assert 8*(rows+64)+131072<=capacity
    assert 8*(rows+128)+131072>capacity
    with pytest.raises(ValueError,match="cannot retain TopK"):
        hisparse_buffer_rows(physical_rows=147904,requests=8,prefill_rows=131072)
    assert hisparse_buffer_rows(physical_rows=2097152,requests=8,prefill_rows=131072)==32768


@pytest.mark.parametrize('mode', ['tp', 'dpa'])
def test_capacity_report_excludes_reserved_gpu_pages(tmp_path, mode):
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / '7_4_capacity/run.py'
    spec = importlib.util.spec_from_file_location('paper_capacity', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for rank in range(8):
        if mode == 'tp':
            distribution = dict(kind='owner_striped_records', owner_count=8,
                allocator_records_per_page=512,
                gpu_cold_record_capacity_per_owner=128,
                cpu_cold_record_capacity_per_owner=256)
        else:
            distribution = dict(kind='native_sparse_cache_partitions',
                required_partitions={'history': dict(
                    device_record_capacity_per_owner=128,
                    host_record_capacity_per_owner=256,
                    reserved_logical_record_count=64)})
        value = dict(pool_fields=dict(logical_allocator_size=2560 if mode=='tp' else 320,
                                      native_request_slots=64),
                     record_distribution=distribution)
        (tmp_path / f'target.rank{rank}.json').write_text(json.dumps(value))
    value = module.summarize(tmp_path, mode, 8)
    assert value['unique_gpu_record_slots'] == 512
    assert value['unique_cpu_record_slots'] == 2048
    assert value['usable_history_slots'] == 2560
