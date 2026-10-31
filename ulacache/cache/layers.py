"""Shared layer ownership and immutable Index-group geometry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerGroupPlan:
    layer_to_group: tuple[int, ...]
    anchors: tuple[int, ...]
    followers: tuple[tuple[int, ...], ...]

    @classmethod
    def from_layer_anchors(cls, owners):
        owners = tuple(owners)
        if not owners:
            raise ValueError("A record cache must contain a local layer")
        anchors = tuple(layer for layer, owner in enumerate(owners) if layer == owner)
        group_ids = {anchor: group for group, anchor in enumerate(anchors)}
        for layer, owner in enumerate(owners):
            if owner not in group_ids or owner > layer:
                raise ValueError("A local group must start at its explicit anchor")
            if layer > owner and owners[layer - 1] != owner:
                raise ValueError("Shared Index followers must be contiguous")
        mapping = tuple(group_ids[owner] for owner in owners)
        followers = tuple(
            tuple(
                layer
                for layer, group in enumerate(mapping)
                if group == index and layer != anchor
            )
            for index, anchor in enumerate(anchors)
        )
        return cls(mapping, anchors, followers)

    @property
    def layer_count(self):
        return len(self.layer_to_group)

    @property
    def group_count(self):
        return len(self.anchors)

    def group(self, layer):
        if not 0 <= layer < self.layer_count:
            raise IndexError("Layer outside the local record cache")
        return self.layer_to_group[layer]


@dataclass(frozen=True)
class LayerReuseSpecification:
    """Canonical owners in local execution order, one entry per consumer.

    Owners must be earlier producers (or the layer itself), not alias chains.
    The four relations are independent. Sharing indexer K does not imply
    sharing a TopK result; sharing TopK does not imply sharing key/value storage.
    Noncontiguous reuse is representable, but needs a backend with matching
    lifetime support. The initial CUDA backend only lowers contiguous groups.
    """

    key_value_storage_owners: tuple[int, ...]
    index_storage_owners: tuple[int, ...]
    sparse_selection_owners: tuple[int, ...]
    residency_plan_anchors: tuple[int, ...]

    def __post_init__(self):
        fields = (
            "key_value_storage_owners",
            "index_storage_owners",
            "sparse_selection_owners",
            "residency_plan_anchors",
        )
        for name in fields:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        count = len(self.key_value_storage_owners)
        if not count or any(len(getattr(self, name)) != count for name in fields):
            raise ValueError("Every local layer must declare all four reuse relations")
        for name in fields:
            owners = getattr(self, name)
            for layer, owner in enumerate(owners):
                if (
                    type(owner) is not int
                    or not 0 <= owner <= layer
                    or owners[owner] != owner
                ):
                    raise ValueError(
                        f"{name} must reference a canonical preceding producer"
                    )

    @property
    def layer_count(self) -> int:
        return len(self.key_value_storage_owners)

    @property
    def unique_key_value_storage_count(self) -> int:
        return len(set(self.key_value_storage_owners))

    @classmethod
    def from_shared_sparse_selection(cls, owners) -> "LayerReuseSpecification":
        """Ordinary DSA: independent KV, with shared Index and selection groups."""
        owners = tuple(owners)
        return cls(tuple(range(len(owners))), owners, owners, owners)

    @classmethod
    def from_layer_groups(
        cls,
        layer_count,
        *,
        key_value_groups=(),
        index_groups=(),
        selection_groups=(),
        residency_groups=None,
    ) -> "LayerReuseSpecification":
        """Declare sharing once at startup; unlisted layers remain independent.

        Each disjoint group uses its earliest layer as producer. Index storage
        and TopK selection sharing are explicit, independent declarations.
        Residency defaults to selection groups for ordinary independent KV.
        Backend lowering still checks native layout and consumer lifetimes.
        """
        if type(layer_count) is not int or layer_count < 1:
            raise ValueError("Layer count must be a positive integer")

        def owners(groups, name):
            result, seen = list(range(layer_count)), set()
            for group in groups:
                group = tuple(group)
                if not group or any(
                    type(layer) is not int or not 0 <= layer < layer_count
                    for layer in group
                ):
                    raise ValueError(f"{name} groups must contain valid layer numbers")
                if len(set(group)) != len(group) or seen.intersection(group):
                    raise ValueError(
                        f"{name} groups must be disjoint without repeated layers"
                    )
                seen.update(group)
                for layer in group:
                    result[layer] = min(group)
            return tuple(result)

        selections = owners(selection_groups, "Selection")
        return cls(
            owners(key_value_groups, "KV"),
            owners(index_groups, "Index"),
            selections,
            selections
            if residency_groups is None
            else owners(residency_groups, "Residency"),
        )
