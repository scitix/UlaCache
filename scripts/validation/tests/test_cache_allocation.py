"""Check maximum page allocation against an exhaustive independent oracle."""

import pytest

from ulacache import CacheMemoryBudget, CacheMemoryUsage, plan_cache_capacity


def measure_usage(device_tokens, host_tokens):
    # Every rank keeps native Index for the complete eight-owner namespace.
    # Host directories are page padded independently of cold GPU payloads.
    total = (device_tokens + host_tokens) * 8
    return CacheMemoryUsage(80 + 3 * total + 24 * device_tokens,
        48 + 9 * host_tokens + 8 * ((host_tokens + 7) // 8))


@pytest.mark.parametrize('device_limit,host_limit', [
    (400, 48), (640, 120), (900, 400), (2000, 100), (4000, 4000),
])
def test_maximum_total_then_device_capacity_matches_exhaustive_allocation(device_limit, host_limit):
    budget = CacheMemoryBudget(device_limit, host_limit)
    plan = plan_cache_capacity(memory_budget=budget, measure_memory_usage=measure_usage,
        minimum_device_token_capacity=4, maximum_token_capacity_per_owner=40,
        page_size=4, storage_owner_count=8)
    candidates = [(device + host, device, host)
        for device in range(4, 41, 4) for host in range(0, 41 - device, 4)
        if budget.contains(measure_usage(device, host))]
    _, device, host = max(candidates)
    assert (plan.device_token_capacity_per_owner, plan.host_token_capacity_per_owner) == (device, host)
    assert plan.memory_usage == measure_usage(device, host)
    assert plan.usable_history_token_capacity(8) == 8 * (device + host) - 8
    plan.validate_memory_budget(device_limit)


def test_native_headroom_can_reject_a_plan_within_its_paired_budget():
    plan = plan_cache_capacity(memory_budget=CacheMemoryBudget(900, 400),
        measure_memory_usage=measure_usage, minimum_device_token_capacity=4,
        maximum_token_capacity_per_owner=40, page_size=4, storage_owner_count=8)
    with pytest.raises(ValueError, match='native available GPU memory'):
        plan.validate_memory_budget(plan.memory_usage.device_bytes - 1)


def test_minimum_working_set_cannot_be_shrunk_to_report_more_host_history():
    with pytest.raises(ValueError, match='minimum device working set'):
        plan_cache_capacity(memory_budget=CacheMemoryBudget(100, 4000),
            measure_memory_usage=measure_usage, minimum_device_token_capacity=4,
            maximum_token_capacity_per_owner=40, page_size=4, storage_owner_count=8)


def test_request_address_bound_limits_allocation_even_with_unused_memory():
    plan = plan_cache_capacity(memory_budget=CacheMemoryBudget(10**9, 10**9),
        measure_memory_usage=measure_usage, minimum_device_token_capacity=4,
        maximum_token_capacity_per_owner=43, page_size=4, storage_owner_count=8)
    assert plan.device_token_capacity_per_owner == 40
    assert plan.host_token_capacity_per_owner == 0


def test_native_paged_backend_keeps_its_required_positive_host_pool():
    measured = []

    def native_usage(device, host):
        assert host >= 4
        measured.append((device, host))
        return measure_usage(device, host)

    budget = CacheMemoryBudget(2000, 200)
    plan = plan_cache_capacity(memory_budget=budget, measure_memory_usage=native_usage,
        minimum_device_token_capacity=4, minimum_host_token_capacity=4,
        maximum_token_capacity_per_owner=40, page_size=4, storage_owner_count=8)
    _, device, host = max((device + host, device, host)
        for device in range(4, 41, 4) for host in range(4, 41 - device, 4)
        if budget.contains(measure_usage(device, host)))
    assert (plan.device_token_capacity_per_owner, plan.host_token_capacity_per_owner) == (device, host)
    assert measured and plan.host_token_capacity_per_owner >= 4
