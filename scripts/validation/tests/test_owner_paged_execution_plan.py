"""Bank-local owner placement keeps native KV and selection aliases separate."""

from dataclasses import replace

from ulacache import LayerReuseSpecification
from ulacache.cache.planning import create_owner_paged_execution_plan, create_paged_cache_execution_plan
from test_paged_execution_plan import specification


def test_shared_native_selection_can_plan_independent_payload_banks():
    original = specification((0, 0, 0), (0, 0, 0))
    original = replace(original, layer_reuse=LayerReuseSpecification.from_shared_sparse_selection((0, 0, 0)))
    local = create_paged_cache_execution_plan(original)
    owner = create_owner_paged_execution_plan(original)
    assert local.storage_bank_ranges_by_residency_domain == (((0, 3),),)
    assert owner.storage_producer_layers == (0, 1, 2)
    assert owner.residency_domain_indices_by_layer == (0, 1, 2)
    assert owner.selection_producer_layers == (0, 1, 2)
    assert owner.last_consumer_layers_by_selection == (0, 1, 2)
    assert owner.step_reader_retirement_layer == 2
    assert original.layer_reuse.sparse_selection_owners == (0, 0, 0)


def test_shared_bank_reindex_keeps_all_five_native_selection_lifetimes():
    selections = tuple(layer // 4 * 4 for layer in range(20))
    config = specification((0,) * 20, selections)
    assert create_owner_paged_execution_plan(config) == create_paged_cache_execution_plan(config)
    plan = create_owner_paged_execution_plan(config)
    assert plan.storage_producer_layers == (0,)
    assert plan.selection_producer_layers == (0, 4, 8, 12, 16)
    assert plan.last_consumer_layers_by_selection == (3, 7, 11, 15, 19)
    assert plan.step_reader_retirement_layer is None


def test_interleaved_banks_keep_their_own_last_consumers_of_one_selection():
    config = specification((0, 0, 0, 0), (0, 0, 0, 0))
    config = replace(config, layer_reuse=LayerReuseSpecification((0, 1, 0, 1),
        (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)))
    plan = create_owner_paged_execution_plan(config)
    assert plan.storage_bank_indices_by_layer == (0, 1, 0, 1)
    assert plan.residency_domain_indices_by_layer == (0, 1, 0, 1)
    assert plan.selection_indices_by_layer == (0, 1, 0, 1)
    assert plan.last_consumer_layers_by_selection == (2, 3)
    assert plan.step_reader_retirement_layer == 3
