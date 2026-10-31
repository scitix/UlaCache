"""Sampled history references retain conditional denominators and layer weights."""
from types import SimpleNamespace as NS

import pytest

from ulacache.cache.layers import LayerGroupPlan
from ulacache.telemetry.cache_hits import CacheHitSampler


class Counters:
    def __init__(self, rows):
        self.rows = rows

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


def observer(rows):
    sampler = CacheHitSampler.__new__(CacheHitSampler)
    sampler.counters = Counters(rows)
    sampler.samples_per_query, sampler.step_stride = 32, 16
    sampler.pool = NS(flowkv_layout=NS(layer_groups=LayerGroupPlan.from_layer_anchors((0, 1, 1, 1))))
    return sampler


def test_current_and_padding_do_not_inflate_either_hit_rate():
    sampler = observer([[100, 80, 20, 5, 15, 0, 200, 300, 10],
                        [100, 0, 100, 100, 0, 0, 900, 400, 10]])
    value = sampler.snapshot()
    assert value['total']['local_hit_rate'] == .4
    assert value['total']['global_gpu_hit_rate_after_local_miss'] == 105/120
    assert value['layer_weighted']['local_hit_rate'] == .2
    assert value['layer_weighted']['global_gpu_hit_rate_after_local_miss'] == 305/320
    assert value['total']['cpu_fraction_of_history_samples'] == 15/200


def test_empty_sample_has_no_fabricated_hit_rate():
    value = observer([[0]*9, [0]*9]).snapshot()
    assert value['total']['local_hit_rate'] is None
    assert value['total']['global_gpu_hit_rate_after_local_miss'] is None


def test_unallocated_references_are_visible_and_not_counted_as_gpu_hits():
    value = observer([[10, 2, 8, 3, 4, 1, 0, 0, 1], [0]*9]).snapshot()
    assert value['total']['global_gpu_hit_rate_after_local_miss'] == 3/8
    assert value['total']['unallocated_after_local_miss'] == 1
    with pytest.raises(AssertionError):
        observer([[10, 2, 8, 3, 4, 0, 0, 0, 1], [0]*9]).snapshot()
