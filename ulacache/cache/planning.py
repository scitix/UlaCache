"""Startup validation and execution plans for native cache backends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from ulacache.cache.specification import UnifiedCacheSpecification
from ulacache.cache.layers import LayerReuseSpecification
from ulacache.cache.layers import LayerGroupPlan


@dataclass(frozen=True)
class PagedCacheExecutionPlan:
    storage_producer_layers: tuple[int, ...]
    storage_bank_indices_by_layer: tuple[int, ...]
    residency_producer_layers: tuple[int, ...]
    residency_domain_indices_by_layer: tuple[int, ...]
    storage_bank_ranges_by_residency_domain: tuple[tuple[tuple[int, int], ...], ...]
    selection_producer_layers: tuple[int, ...]
    selection_indices_by_layer: tuple[int, ...]
    last_consumer_layers_by_selection: tuple[int, ...]

    @property
    def step_reader_retirement_layer(self):
        """Batch retirement when each directory has one selection per forward."""
        domains = tuple(
            self.residency_domain_indices_by_layer[layer]
            for layer in self.selection_producer_layers
        )
        if len(domains) != len(set(domains)):
            return None
        return max(self.last_consumer_layers_by_selection)


def create_paged_attention_workspace_plan(
    execution_plan: PagedCacheExecutionPlan, *, reuse_retired_banks=False
):
    """Bound temporary bank views by their actual selection consumers."""
    from ulacache.storage.workspace import (
        AttentionWorkspaceLifetime,
        AttentionWorkspacePlan,
        create_attention_workspace_plan,
    )

    bank_count = len(execution_plan.storage_producer_layers)
    if not reuse_retired_banks:
        return AttentionWorkspacePlan(tuple(range(bank_count)), bank_count)

    consumers = {}
    for layer, bank_selection in enumerate(
        zip(
            execution_plan.storage_bank_indices_by_layer,
            execution_plan.selection_indices_by_layer,
        )
    ):
        consumers.setdefault(bank_selection, []).append(layer)
    lifetimes = tuple(
        AttentionWorkspaceLifetime(bank, layers[0], layers[-1])
        for (bank, selection), layers in consumers.items()
    )
    return create_attention_workspace_plan(bank_count, lifetimes)


def create_owner_paged_execution_plan(cache_specification: UnifiedCacheSpecification):
    """Lower shared native selections into independent owner payload banks.

    Each KV bank has its own directory and writer publication boundary. A
    selection shared across banks still contains the same logical IDs, while
    each bank plans its physical placement at its first consumer. Local ordinal
    attention rows preserve the original selection order across those banks.
    KV aliases stay in one bank; successive selections on that bank retain the
    existing lease-interval checks. No native Index or attention math changes.
    """
    reuse = cache_specification.layer_reuse
    first_consumers = {}
    selection_owners = []
    for layer, pair in enumerate(
        zip(reuse.key_value_storage_owners, reuse.sparse_selection_owners)
    ):
        first_consumers.setdefault(pair, layer)
        selection_owners.append(first_consumers[pair])
    lowered_reuse = LayerReuseSpecification(
        reuse.key_value_storage_owners,
        reuse.index_storage_owners,
        tuple(selection_owners),
        reuse.key_value_storage_owners,
    )
    return create_paged_cache_execution_plan(
        replace(cache_specification, layer_reuse=lowered_reuse)
    )


def create_paged_cache_execution_plan(cache_specification: UnifiedCacheSpecification):
    """One directory can coordinate several independent native payload banks.

    A directory may serve successive reindex selections. Their consumer intervals
    must not overlap because the existing planner has one lease list per directory.
    Other directories can execute between a producer and its last consumer.
    """
    layer_reuse = cache_specification.layer_reuse
    access = cache_specification.sparse_access
    capacity = cache_specification.capacity_plan
    if (
        access.selection_unit != "record"
        or access.selection_identifier_space != "logical"
    ):
        raise NotImplementedError(
            "Native selections must first name canonical logical records"
        )
    if capacity.device_record_capacity % 64 or capacity.host_record_capacity < 1:
        raise ValueError(
            "Paged residency requires 64-aligned device slots and positive host capacity"
        )
    if capacity.logical_record_capacity >= 2**31:
        raise ValueError("Paged residency logical IDs must fit positive int32")

    storage_producers = tuple(sorted(set(layer_reuse.key_value_storage_owners)))
    storage_banks = {layer: bank for bank, layer in enumerate(storage_producers)}
    residency_producers = tuple(sorted(set(layer_reuse.residency_plan_anchors)))
    residency_domains = {
        layer: domain for domain, layer in enumerate(residency_producers)
    }
    for layer, owner in enumerate(layer_reuse.key_value_storage_owners):
        if (
            layer_reuse.residency_plan_anchors[layer]
            != layer_reuse.residency_plan_anchors[owner]
        ):
            raise NotImplementedError(
                "A shared payload bank requires one residency directory"
            )
    bank_ranges_by_domain = []
    for producer in residency_producers:
        ranges = []
        for bank, layer in enumerate(storage_producers):
            if layer_reuse.residency_plan_anchors[layer] != producer:
                continue
            if ranges and ranges[-1][1] == bank:
                ranges[-1] = (ranges[-1][0], bank + 1)
            else:
                ranges.append((bank, bank + 1))
        bank_ranges_by_domain.append(tuple(ranges))
    selection_producers = tuple(sorted(set(layer_reuse.sparse_selection_owners)))
    selection_indices = {
        layer: index for index, layer in enumerate(selection_producers)
    }
    last_consumers = tuple(
        max(
            layer
            for layer, owner in enumerate(layer_reuse.sparse_selection_owners)
            if owner == producer
        )
        for producer in selection_producers
    )
    for layer, producer in enumerate(layer_reuse.sparse_selection_owners):
        if (
            layer_reuse.residency_plan_anchors[layer]
            != layer_reuse.residency_plan_anchors[producer]
        ):
            raise NotImplementedError(
                "A shared selection requires one residency directory"
            )
    for residency_producer in residency_producers:
        intervals = [
            (producer, last_consumer)
            for producer, last_consumer in zip(selection_producers, last_consumers)
            if layer_reuse.residency_plan_anchors[producer] == residency_producer
        ]
        if any(
            last >= following
            for (_, last), (following, _) in zip(intervals, intervals[1:])
        ):
            raise NotImplementedError(
                "Overlapping selections on one directory require additional lease storage"
            )
    return PagedCacheExecutionPlan(
        storage_producers,
        tuple(storage_banks[owner] for owner in layer_reuse.key_value_storage_owners),
        residency_producers,
        tuple(residency_domains[owner] for owner in layer_reuse.residency_plan_anchors),
        tuple(bank_ranges_by_domain),
        selection_producers,
        tuple(
            selection_indices[owner] for owner in layer_reuse.sparse_selection_owners
        ),
        last_consumers,
    )


def create_rank_local_layer_groups(
    cache_specification: UnifiedCacheSpecification,
) -> LayerGroupPlan:
    """Reject unsupported lowering before importing CUDA or allocating memory."""
    storage_layout, layer_reuse, sparse_access, capacity_plan = (
        cache_specification.storage_layout,
        cache_specification.layer_reuse,
        cache_specification.sparse_access,
        cache_specification.capacity_plan,
    )
    if (
        storage_layout.records_per_page != 64
        or storage_layout.record_kind != "token"
        or len(storage_layout.record_planes) != 1
    ):
        raise NotImplementedError(
            "Local v1 requires one uncompressed native plane with page64"
        )
    plane = storage_layout.record_planes[0]
    formats = {
        "mla.raw576.fp8.v1": (512, 64),
        "mla.scaled656.fp8.v1": (528, 128),
    }
    if (
        plane.native_encoding not in formats
        or plane.bytes_per_segment != formats[plane.native_encoding]
        or plane.page_alignment_bytes != 1
        or plane.page_memory_order != "record_major"
    ):
        raise NotImplementedError(
            "Local v1 supports native raw576/scaled656 FP8 encodings only"
        )
    if (
        sparse_access.selection_unit != "record"
        or sparse_access.selection_identifier_space != "logical"
        or sparse_access.maximum_selected_units % 256
    ):
        raise NotImplementedError(
            "Local v1 needs canonical record IDs with a 256-aligned width"
        )
    if layer_reuse.key_value_storage_owners != tuple(range(layer_reuse.layer_count)):
        raise NotImplementedError(
            "Local v1 does not lower cross-layer key/value storage aliases"
        )
    if layer_reuse.residency_plan_anchors != layer_reuse.sparse_selection_owners:
        raise NotImplementedError(
            "Local v1 residency groups must follow native selection reuse"
        )
    groups = LayerGroupPlan.from_layer_anchors(layer_reuse.residency_plan_anchors)
    if (
        capacity_plan.device_record_capacity % 64
        or capacity_plan.host_record_capacity < 1
    ):
        raise ValueError(
            "Local v1 requires page-aligned device capacity and positive host capacity"
        )
    if capacity_plan.logical_record_capacity >= 2**31:
        raise ValueError("Local v1 logical IDs must fit positive signed int32")
    return groups
