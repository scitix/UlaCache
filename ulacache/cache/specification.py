"""A cache tier's startup contract and payload capacity accounting."""

from dataclasses import dataclass

from ulacache.cache.access import SparseAccessSpecification
from ulacache.cache.layers import LayerReuseSpecification
from ulacache.storage.layout import KeyValueStorageLayout


@dataclass(frozen=True)
class MemoryTierCapacity:
    """Exclusive CPU/GPU residency; reservations are inside logical capacity.

    Logical history capacity and GPU working-set capacity are different limits.
    A model may fit in GPU+CPU while its simultaneously selected union does not
    fit in GPU. Admission must respect both limits, including current writes.
    ``maximum_residency_input_record_count`` bounds padded planner inputs from
    all consumers; repeats and sentinels do not add unique resident records.
    """

    device_record_capacity: int
    host_record_capacity: int
    maximum_write_record_count: int
    reserved_logical_record_count: int = 0
    maximum_prefill_union_record_count: int = 0
    maximum_residency_input_record_count: int = 0
    prefill_workspace_record_capacity: int = 0

    def __post_init__(self):
        if any(
            type(n) is not int
            for n in (
                self.device_record_capacity,
                self.host_record_capacity,
                self.maximum_write_record_count,
                self.reserved_logical_record_count,
                self.maximum_prefill_union_record_count,
                self.maximum_residency_input_record_count,
                self.prefill_workspace_record_capacity,
            )
        ):
            raise ValueError("Cache capacities must be integers")
        if (
            self.device_record_capacity < 1
            or self.host_record_capacity < 0
            or self.maximum_write_record_count < 1
        ):
            raise ValueError("Invalid GPU, CPU or write capacity")
        if not 0 <= self.reserved_logical_record_count < self.logical_record_capacity:
            raise ValueError("Reservations must leave usable logical capacity")
        if self.prefill_workspace_record_capacity < 0:
            raise ValueError("Prefill workspace capacity must be nonnegative")
        prefill_capacity = max(
            self.device_record_capacity, self.prefill_workspace_record_capacity
        )
        if not 0 <= self.maximum_prefill_union_record_count <= prefill_capacity:
            raise ValueError(
                "Prefill union cannot exceed device capacity or its declared workspace"
            )
        if self.maximum_residency_input_record_count < 0:
            raise ValueError("Residency input capacity must be nonnegative")

    @property
    def logical_record_capacity(self) -> int:
        return self.device_record_capacity + self.host_record_capacity

    @property
    def usable_logical_record_capacity(self) -> int:
        return self.logical_record_capacity - self.reserved_logical_record_count


@dataclass(frozen=True)
class UnifiedCacheSpecification:
    """One homogeneous sparse tier. A model can construct several tier handles."""

    storage_layout: KeyValueStorageLayout
    layer_reuse: LayerReuseSpecification
    sparse_access: SparseAccessSpecification
    capacity_plan: MemoryTierCapacity

    def calculate_key_value_memory_budget(self) -> dict[str, int]:
        """Count KV aliases once; excludes Index, workspaces, draft and Graph."""
        count = self.layer_reuse.unique_key_value_storage_count
        return {
            "device_key_value_bytes": count
            * self.storage_layout.calculate_storage_bytes(
                self.capacity_plan.device_record_capacity
            ),
            "host_key_value_bytes": count
            * self.storage_layout.calculate_storage_bytes(
                self.capacity_plan.host_record_capacity
            ),
            "logical_record_capacity": self.capacity_plan.logical_record_capacity,
            "usable_logical_record_capacity": self.capacity_plan.usable_logical_record_capacity,
            "unique_key_value_storage_count": count,
        }

    @property
    def worst_case_query_record_count(self) -> int:
        """Disjoint selected/current records plus the resident zero sentinel."""
        return (
            self.sparse_access.maximum_padded_query_count
            * (self.sparse_access.maximum_selected_record_count + 1)
            + 1
        )
