"""Bounded Plain prefill accounting and optional compiler contracts."""
from dataclasses import fields, replace

import pytest

from ulacache.storage.workspace import PlainPrefillLayout as FlowKVLayout
from ulacache.storage.workspace import PlainPrefillLayout
from ulacache.storage.capacity import plain_prefill_pool_budget
from ulacache.storage.capacity import shared_host_workspace_bytes, shared_host_workspace_fixed_bytes, shared_host_workspace_expiry_bytes
import ulacache.eviction.kernels.local_lru as factory


def layout(**kwargs):
    values = dict(layers=78, groups=21, max_batch=64, pool_rows=1163520,
        row_bytes=656, disaggregation_mode='null', topk=2048,
        max_query_tokens=64, history_slots=64 * 2 * 2048,
        page_size=64, native_segment_bytes=(656,),
        layer_anchors=tuple(range(21)) + (20,) * 57,
        max_staging_tokens=256, prefill_window_rows=65536)
    values.update(kwargs)
    return FlowKVLayout(**values)


def test_pipeline_conversion_keeps_geometry_and_adds_two_prefill_windows():
    from ulacache.storage.workspace import PipelineTPLayout

    source = layout(max_query_tokens=128, max_resident_queries=96)
    converted = PipelineTPLayout.from_native(source)
    assert converted.resident_query_capacity == source.resident_query_capacity
    assert converted.rows_per_layer == source.rows_per_layer
    assert converted.total_prefill_buffer_bytes == 2 * source.prefill_buffer_bytes


def test_plain_prefill_scratch_has_no_logical_pool_slope():
    small = layout(pool_rows=131072)
    large = layout(pool_rows=8388608)
    assert small.prefill_buffer_bytes == large.prefill_buffer_bytes == 65536 * 656
    assert small.staging_buffer_bytes == large.staging_buffer_bytes == 256 * 656
    assert small.prefill_lookup_bytes == large.prefill_lookup_bytes
    old = replace(large, prefill_window_rows=0)
    assert old.prefill_buffer_bytes == large.pool_rows * 656
    assert old.staging_buffer_bytes == 0
    assert small.extra_bytes == sum((small.local_kv_buffer_bytes,
        small.prefill_buffer_bytes, small.staging_buffer_bytes,
        small.prefill_lookup_bytes, small.metadata_bytes))


def test_window_requires_complete_topk_and_explicit_staging():
    with pytest.raises(ValueError, match='complete TopK'):
        layout(prefill_window_rows=1024)
    with pytest.raises(ValueError, match='staging bound'):
        layout(max_staging_tokens=0)
    decode = layout(disaggregation_mode='decode')
    assert decode.prefill_buffer_bytes == decode.staging_buffer_bytes == 0


def test_resident_plain_window_does_not_limit_native_compute_chunk():
    ordinary = layout()
    values = {field.name: getattr(ordinary, field.name) for field in fields(ordinary)}
    values.update(max_query_tokens=256, max_resident_queries=192,
        history_slots=64 * 3 * 2048, max_staging_tokens=8192)
    speculative = PlainPrefillLayout(**values)
    assert speculative.resident_query_capacity == 192
    assert speculative.capacity == 64 * 3 * 2048
    assert speculative.prefill_buffer_bytes == ordinary.prefill_buffer_bytes
    assert speculative.staging_buffer_bytes == 8192 * 656


def test_native_plain_solver_charges_fixed_window_not_logical_scratch():
    compact = layout()
    budget = plain_prefill_pool_budget(compact)
    assert budget.logical_row_bytes == 0
    assert budget.fixed_bytes == compact.prefill_buffer_bytes + compact.staging_buffer_bytes + compact.prefill_lookup_bytes
    for rows in (131072, 1163520, 8388608):
        native = replace(compact, pool_rows=rows, prefill_window_rows=0)
        old = plain_prefill_pool_budget(native)
        assert old.fixed_bytes + old.logical_row_bytes * rows == native.prefill_buffer_bytes
        new = plain_prefill_pool_budget(replace(compact, pool_rows=rows))
        assert new.fixed_bytes == budget.fixed_bytes
    decode = plain_prefill_pool_budget(replace(compact, disaggregation_mode='decode'))
    assert decode.logical_row_bytes == decode.fixed_bytes == 0


def test_shared_workspace_static_dynamic_sum_matches_actual_allocation():
    for capacity, groups, gpu in ((97, 3, 64), (4194560, 21, 43904), (31, 1, 64)):
        assert (shared_host_workspace_fixed_bytes(capacity, groups)
            + shared_host_workspace_expiry_bytes(capacity, groups, gpu)
            == shared_host_workspace_bytes(capacity, groups, gpu))


def test_compact_compile_identity_and_original_assets(monkeypatch):
    monkeypatch.setattr(factory, 'load_jit', lambda *args, **kwargs: (args, kwargs))
    base = dict(topk=2048, max_queries=64, page_size=64, bytes_per_page=64 * 656,
        peers=8, slots_per_rank=4160, row_bytes=656, native_segment_bytes=(656,),
        step_publication=True, local_eviction_policy='age_lru')
    original = factory.flowkv_record_policy_module(**base)
    compact = factory.flowkv_record_policy_module(**base, compact_prefill=True)
    assert original[0] != compact[0]
    assert original[1]['cuda_files'] == compact[1]['cuda_files'][:-1]
    wrappers = dict(compact[1]['cuda_wrappers'])
    assert wrappers['write_prefill_staged'] == 'sglang::plain_prefill_write_staged'
    assert wrappers['fill_compact'] == 'sglang::plain_prefill_fill_compact'
    assert dict(original[1]['cuda_wrappers'])['write_decode'] == wrappers['write_decode']
