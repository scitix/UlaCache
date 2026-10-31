"""Receive into existing native-paged banks and independent domain directories.

No persistent directory or KV data is copied. Only the selected publication
columns are gathered after the native reader-retirement boundary.
"""

from types import SimpleNamespace

from ulacache.transfer.receive_window import PinnedRecordReceiveWindow


class _DirectoryColumns:
    def __init__(self, states, logical_slots):
        import torch

        self.directories = tuple(state.directory for state in states)
        if not self.directories or any(
            directory.dtype != torch.int64
            or directory.shape != (logical_slots,)
            or not directory.is_contiguous()
            for directory in self.directories
        ):
            raise ValueError(
                "Native paged domains require contiguous int64 logical-row directories"
            )
        self.device = self.directories[0].device
        if any(directory.device != self.device for directory in self.directories):
            raise ValueError("Native paged directories must share their device")
        if len({directory.data_ptr() for directory in self.directories}) != len(
            self.directories
        ):
            raise ValueError(
                "Independent native residency domains cannot alias their directories"
            )
        self.shape = (len(self.directories), logical_slots)
        self.ndim = 2

    def index_select(self, dim, rows):
        import torch

        if dim != 1:
            raise ValueError(
                "Receive publication selects logical columns within every native domain"
            )
        return torch.stack(
            [directory.index_select(0, rows) for directory in self.directories]
        )


def native_record_receive_view(cache, *, device=None):
    """Borrow contiguous native-paged payloads without duplicating storage."""
    import torch
    from ulacache.cache.local_paged_cache import NativePagedResidency

    native = cache._execution_engine
    if not isinstance(native, NativePagedResidency):
        raise ValueError(
            "Native paged reception requires the actual NativePagedResidency engine"
        )
    plan, spec = native.execution_plan, cache.cache_specification
    layout, capacity = spec.storage_layout, spec.capacity_plan
    layers = spec.layer_reuse.layer_count
    if (
        plan.storage_producer_layers != tuple(range(layers))
        or len(layout.record_planes) != 1
        or layout.record_planes[0].page_memory_order != "record_major"
    ):
        raise NotImplementedError(
            "Native paged reception requires one independent record bank per layer"
        )
    width = sum(layout.record_planes[0].bytes_per_segment)
    alignment = getattr(layout.record_planes[0], "page_alignment_bytes", 1)
    if layout.records_per_page * width % alignment:
        raise NotImplementedError(
            "Direct record views cannot include page-tail padding"
        )
    storage = native.record_storage.plane_storage[0]
    gpu = storage.device_record_pages.view(layers, -1, width)
    host = storage.host_record_pool.storage.view(layers, -1, width)
    declared_device = gpu.device if device is None else torch.device(device)
    if (
        gpu.shape[1] != capacity.device_record_capacity
        or host.shape[1] != capacity.host_record_capacity
        or gpu.device.type != declared_device.type
        or (
            declared_device.index is not None
            and gpu.device.index != declared_device.index
        )
    ):
        raise ValueError(
            "Native paged record views must retain the actual physical capacities/device"
        )
    domains = tuple(plan.residency_domain_indices_by_layer)
    if len(domains) != layers or any(
        not 0 <= domain < len(native.residency_states) for domain in domains
    ):
        raise ValueError("Native layer-to-residency-domain mapping is incomplete")
    columns = _DirectoryColumns(native.residency_states, gpu.shape[1] + host.shape[1])
    if columns.device != gpu.device:
        raise ValueError(
            "Native directory and record views must belong to the same device"
        )
    engine = SimpleNamespace(
        gpu=gpu,
        host=SimpleNamespace(storage=host),
        gpu_slots=gpu.shape[1],
        host_slots=host.shape[1],
        row_bytes=width,
        directory=columns,
        groups=SimpleNamespace(
            layer_count=layers,
            group_count=len(native.residency_states),
            group=lambda layer: domains[layer],
        ),
        _drain_transfers=native._require_retired_readers,
        native_engine=native,
    )
    return engine


class NativePagedRecordReceiveWindow(PinnedRecordReceiveWindow):
    """A transport view over the approved native paged cache, never a new cache."""

    def __init__(self, cache, **kwargs):
        engine = native_record_receive_view(cache, device=kwargs["device"])
        self.native_engine = engine.native_engine
        super().__init__(cache, execution_engine=engine, **kwargs)

    def logical_pages_freed(self, page_ids):
        retired = set(int(page) for page in page_ids)
        for item in self._pending.values():
            if retired.intersection(item["logical_pages"]):
                item["retired_before_publication"] = True
        return super().logical_pages_freed(retired)

    def publish(self, key):
        if self._pending[key].get("retired_before_publication", False):
            raise RuntimeError(
                "A retired native receive transaction cannot publish into reused logical rows"
            )
        return super().publish(key)
