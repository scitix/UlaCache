"""Full native query bounds preserve exact, page-aligned resident KV budgets."""
from dataclasses import replace

import pytest

from ulacache.storage.workspace import ResidentQueryLayout


def layout():
    return ResidentQueryLayout(layers=78, groups=21, max_batch=64, pool_rows=1048576,
        row_bytes=656, disaggregation_mode='null', topk=2048, max_query_tokens=512,
        max_resident_queries=320, history_slots=320 * 2048, page_size=64,
        native_segment_bytes=(512, 16, 128),
        layer_anchors=(0,) + tuple(1 + 4*((layer-1)//4) for layer in range(1,78)))


def test_bs64_four_step_metadata_keeps_exact_resident_bytes():
    value = layout()
    assert value.compression_ratio == 1
    assert value.max_query_tokens == 512
    assert value.resident_query_capacity == 320
    assert value.rows_per_layer == 320 * 2048 + 320
    assert value.rows_per_layer % 64 == 0
    assert value.local_kv_buffer_bytes == 78 * (320 * 2048 + 320) * 656


def test_r328_linear_verify_uses_984_resident_rows_with_1024_metadata():
    value = replace(layout(), max_batch=328, max_query_tokens=1024,
                    max_resident_queries=984, history_slots=984 * 2048)
    assert value.resident_query_capacity == 984
    assert value.max_query_tokens == 1024
    assert value.rows_per_layer == 984 * 2048 + 1024
    assert value.local_kv_buffer_bytes == 78 * (984 * 2048 + 1024) * 656
    padded = replace(value, max_resident_queries=1024, history_slots=1024 * 2048)
    assert value.local_kv_buffer_bytes < padded.local_kv_buffer_bytes


@pytest.mark.parametrize('changes', [
    dict(max_query_tokens=4096), dict(max_query_tokens=320),
    dict(max_query_tokens=256), dict(max_query_tokens=True),
    dict(max_batch=513), dict(max_resident_queries=513),
    dict(history_slots=320 * 2048 - 64), dict(topk=2049),
    dict(native_segment_bytes=(512, 128)), dict(groups=79),
])
def test_expanded_bound_preserves_geometry_and_budget_checks(changes):
    with pytest.raises(ValueError):
        replace(layout(), **changes)
