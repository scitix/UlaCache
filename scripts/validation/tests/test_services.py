"""Startup contracts must stand on their own without framework validation."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ulacache.bindings import BackendServices, CacheOptions


def services(*, degree=4, rank=0, namespaces=1, options=None):
    return BackendServices(
        degree=degree, group=SimpleNamespace(rank_in_group=rank, cpu_group=None),
        namespaces=namespaces, config=options or CacheOptions(),
        memory_region=nullcontext, ipc_manager_factory=lambda: None,
        host_pool_factory=lambda **kwargs: None,
        host_memory_budget_bytes=lambda requested: requested,
    )


def test_unknown_residency_policy_does_not_select_index_implicitly():
    with pytest.raises(ValueError, match="residency policy"):
        CacheOptions(cpu_offload=True, cpu_policy="unknown")


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_cpu_extension_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="CPU offload must be boolean"):
        CacheOptions(cpu_offload=value)


@pytest.mark.parametrize("options", [{"cpu_slots": -1}, {"cpu_ratio": 0}])
def test_invalid_host_geometry_fails_at_startup(options):
    with pytest.raises(ValueError, match="slot count"):
        CacheOptions(**options)


@pytest.mark.parametrize("parameters", [
    {"degree": 0}, {"namespaces": 0}, {"rank": -1}, {"rank": 4},
])
def test_invalid_owner_coordinates_fail_before_allocating(parameters):
    with pytest.raises(ValueError):
        services(**parameters)


def test_service_binding_does_not_execute_factories():
    def forbidden(*args, **kwargs):
        raise AssertionError("startup binding executed a resource factory")

    options = CacheOptions(cpu_offload=True, cpu_policy="request")
    bound = BackendServices(
        8, SimpleNamespace(rank_in_group=7, cpu_group=None), 1, options,
        forbidden, forbidden, forbidden, forbidden,
    )
    assert bound.config is options
    assert bound.group.rank_in_group == 7


def test_parallel_victims_require_an_enabled_finish_path():
    with pytest.raises(ValueError, match="require parallel host finish"):
        CacheOptions(host_parallel_victims=True)
