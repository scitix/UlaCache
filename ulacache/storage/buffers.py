"""Native record plane storage and pinned host pools."""

from __future__ import annotations

from dataclasses import replace


class RecordPlaneStorage:
    """Keep native payload/scale pages intact across both memory tiers.

    The injected host allocator allocates a native page as one byte row. Record
    capacities and page counts remain distinct. Writer sources are bounded
    native packed pages produced by the framework's original quantizer.
    """

    def __init__(self, cache_specification, cache_bindings, execution_plan):
        import torch

        from ulacache.compiler import configure_compiler
        from ulacache.transfer.cpu_gpu import NativePagedRecordTransfer

        configure_compiler(cache_bindings.compile_device_module)
        self.storage_layout = cache_specification.storage_layout
        self.capacity_plan = cache_specification.capacity_plan
        self.execution_plan = execution_plan
        plane = self.storage_layout.record_planes[0]
        self.records_per_page = self.storage_layout.records_per_page
        self.bytes_per_page = plane.bytes_per_page(self.records_per_page)
        device_pages = (
            self.capacity_plan.device_record_capacity + self.records_per_page - 1
        ) // self.records_per_page
        host_pages = (
            self.capacity_plan.host_record_capacity + self.records_per_page - 1
        ) // self.records_per_page
        storage_banks = len(execution_plan.storage_producer_layers)
        blocks, threads = cache_bindings.execution_options.exchange_launch_dimensions
        self.fuse_native_record_publication = getattr(
            cache_bindings.execution_options, "fuse_native_record_publication", False
        )
        self.record_transfer = NativePagedRecordTransfer(
            plane,
            self.records_per_page,
            launch_blocks=blocks,
            launch_threads=threads,
            fuse_publication=self.fuse_native_record_publication,
        )
        self.device_record_pages = torch.zeros(
            (storage_banks, device_pages, self.bytes_per_page),
            dtype=torch.uint8,
            device=cache_bindings.device,
        )
        self.device_write_record_indices = torch.empty(
            self.capacity_plan.maximum_write_record_count,
            dtype=torch.int32,
            device=cache_bindings.device,
        )
        self.host_write_record_indices = torch.empty_like(
            self.device_write_record_indices
        )
        self.host_record_pool = cache_bindings.allocate_host_record_storage(
            layers=storage_banks,
            slots=host_pages,
            row_bytes=self.bytes_per_page,
            device=cache_bindings.device,
        )
        self._closed = False

    def get_device_record_bank(self, layer_index):
        bank = self.execution_plan.storage_bank_indices_by_layer[layer_index]
        return self.device_record_pages[bank]

    def get_device_record_planes(self, layer_index):
        return (self.get_device_record_bank(layer_index),)

    def exchange_residency_plan(self, residency_domain_index, transfer_workspace):
        for start, stop in self.execution_plan.storage_bank_ranges_by_residency_domain[
            residency_domain_index
        ]:
            self.record_transfer.exchange(
                self.host_record_pool.storage[start:stop],
                self.device_record_pages[start:stop],
                transfer_workspace.plan,
                transfer_workspace.counters,
            )

    def exchange_record_bank(self, bank, transfer_workspace):
        self.record_transfer.exchange(
            self.host_record_pool.storage[bank : bank + 1],
            self.device_record_pages[bank : bank + 1],
            transfer_workspace.plan,
            transfer_workspace.counters,
        )

    def write_native_record_source(
        self,
        layer_index,
        logical_record_ids,
        source_record_pages,
        source_record_indices,
        residency_directory,
        *,
        local_device_after_commit=False,
    ):
        """Copy native quantized bytes without interpreting or rounding them."""
        from ulacache.storage.kernels.paged_write import map_native_write_locations

        count = logical_record_ids.numel()
        if count > self.capacity_plan.maximum_write_record_count:
            raise ValueError("Native writer exceeds its startup record capacity")
        if self.fuse_native_record_publication:
            bank = self.execution_plan.storage_bank_indices_by_layer[layer_index]
            self.record_transfer.publish(
                source_record_pages,
                self.device_record_pages[bank],
                self.host_record_pool.storage[bank],
                logical_record_ids,
                source_record_indices,
                residency_directory,
                local_device_after_commit=local_device_after_commit,
            )
            return
        if local_device_after_commit:
            raise ValueError(
                "Local current publication requires the fused native publisher"
            )
        device_records = self.device_write_record_indices[:count]
        host_records = self.host_write_record_indices[:count]
        map_native_write_locations(
            logical_record_ids, residency_directory, device_records, host_records
        )
        bank = self.execution_plan.storage_bank_indices_by_layer[layer_index]
        self.record_transfer.copy(
            source_record_pages,
            self.device_record_pages[bank],
            source_record_indices,
            device_records,
        )
        self.record_transfer.copy(
            source_record_pages,
            self.host_record_pool.storage[bank],
            source_record_indices,
            host_records,
        )

    def write_native_record_segments(
        self, layer_index, logical_record_ids, record_segments, residency_directory
    ):
        """Write the native quantizer's byte segments to their current tier."""
        import torch

        from ulacache.storage.kernels.paged_write import write_native_record_segment

        plane = self.storage_layout.record_planes[0]
        if len(record_segments) != len(plane.bytes_per_segment):
            raise ValueError("Native writer must supply every declared byte segment")
        bank = self.execution_plan.storage_bank_indices_by_layer[layer_index]
        prefix = 0
        for segment, width in zip(record_segments, plane.bytes_per_segment):
            source = segment.view(torch.uint8).reshape(
                logical_record_ids.numel(), width
            )
            write_native_record_segment(
                logical_record_ids,
                source,
                residency_directory,
                self.device_record_pages[bank],
                self.host_record_pool.ptrs,
                host_bank_offset=bank * self.host_record_pool.storage.stride(0),
                records_per_page=self.records_per_page,
                bytes_per_page=self.bytes_per_page,
                bytes_per_record=plane.bytes_per_record,
                segment_prefix=prefix,
                segment_major=plane.page_memory_order == "segment_major",
            )
            prefix += width

    def device_storage_inventory(self):
        tensors = (
            self.device_record_pages,
            self.device_write_record_indices,
            self.host_write_record_indices,
            self.host_record_pool.ptrs,
        )
        return {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in tensors
            if tensor.numel()
        }

    def host_storage_inventory(self):
        storage = self.host_record_pool.storage.untyped_storage()
        return {storage.data_ptr(): storage.nbytes()} if storage.nbytes() else {}

    def memory_inventory(self):
        return dict(
            gpu_kv_bytes=self.device_record_pages.nbytes,
            cpu_kv_bytes=self.host_record_pool.nbytes,
            physical_storage_banks=len(self.execution_plan.storage_producer_layers),
            records_per_page=self.records_per_page,
            bytes_per_page=self.bytes_per_page,
        )

    def close(self):
        if self._closed:
            return
        import torch

        torch.cuda.synchronize(self.device_record_pages.device)
        self.host_record_pool.close()
        self._closed = True


class RecordStorage:
    """Separate native K/V allocations, sharing logical IDs and placement.

    Each plane retains its own page padding, segment order and native byte
    transfer. All plane transfers precede the caller's single residency commit.
    This class performs no packing, quantization or attention computation.
    """

    def __init__(self, cache_specification, cache_bindings, execution_plan):
        self.storage_layout = cache_specification.storage_layout
        self.execution_plan = execution_plan
        self.plane_storage = []
        try:
            for plane in self.storage_layout.record_planes:
                specification = replace(
                    cache_specification,
                    storage_layout=replace(self.storage_layout, record_planes=(plane,)),
                )
                self.plane_storage.append(
                    RecordPlaneStorage(specification, cache_bindings, execution_plan)
                )
        except BaseException:
            self.close()
            raise

    @property
    def fuse_native_record_publication(self):
        return bool(self.plane_storage) and all(
            plane.fuse_native_record_publication is True for plane in self.plane_storage
        )

    def get_device_record_bank(self, layer_index):
        if len(self.plane_storage) != 1:
            raise ValueError("Use get_device_record_planes for a multi-plane layout")
        return self.plane_storage[0].get_device_record_bank(layer_index)

    def get_device_record_planes(self, layer_index):
        return tuple(
            storage.get_device_record_bank(layer_index)
            for storage in self.plane_storage
        )

    def exchange_residency_plan(self, residency_domain_index, transfer_workspace):
        for storage in self.plane_storage:
            storage.exchange_residency_plan(residency_domain_index, transfer_workspace)

    def exchange_record_bank(self, bank, transfer_workspace):
        for storage in self.plane_storage:
            storage.exchange_record_bank(bank, transfer_workspace)

    def write_native_record_source(
        self,
        layer_index,
        logical_record_ids,
        source_record_pages,
        source_record_indices,
        residency_directory,
        *,
        local_device_after_commit=False,
    ):
        if len(self.plane_storage) == 1 and not isinstance(
            source_record_pages, (tuple, list)
        ):
            source_record_pages = (source_record_pages,)
        if len(source_record_pages) != len(self.plane_storage):
            raise ValueError("Native writer must publish every declared record plane")
        options = (
            {"local_device_after_commit": True} if local_device_after_commit else {}
        )
        for storage, pages in zip(self.plane_storage, source_record_pages):
            storage.write_native_record_source(
                layer_index,
                logical_record_ids,
                pages,
                source_record_indices,
                residency_directory,
                **options,
            )

    def write_native_record_segments(
        self, layer_index, logical_record_ids, record_segments, residency_directory
    ):
        segment_counts = [
            len(plane.bytes_per_segment) for plane in self.storage_layout.record_planes
        ]
        if len(record_segments) != sum(segment_counts):
            raise ValueError("Native writer must supply every declared plane segment")
        start = 0
        for storage, count in zip(self.plane_storage, segment_counts):
            storage.write_native_record_segments(
                layer_index,
                logical_record_ids,
                record_segments[start : start + count],
                residency_directory,
            )
            start += count

    def device_storage_inventory(self):
        inventory = {}
        for storage in self.plane_storage:
            inventory.update(storage.device_storage_inventory())
        return inventory

    def host_storage_inventory(self):
        inventory = {}
        for storage in self.plane_storage:
            inventory.update(storage.host_storage_inventory())
        return inventory

    def memory_inventory(self):
        planes = [storage.memory_inventory() for storage in self.plane_storage]
        return dict(
            gpu_kv_bytes=sum(plane["gpu_kv_bytes"] for plane in planes),
            cpu_kv_bytes=sum(plane["cpu_kv_bytes"] for plane in planes),
            physical_storage_banks=len(self.execution_plan.storage_producer_layers),
            record_plane_count=len(planes),
            records_per_page=self.storage_layout.records_per_page,
            planes=planes,
        )

    def close(self):
        for storage in self.plane_storage:
            storage.close()


def validate_host_allocation_headroom(required_bytes, *, reserve_bytes=8 * 2**30):
    """Check aggregate pinned allocation against host and cgroup working space."""
    from pathlib import Path

    if required_bytes < 0 or reserve_bytes < 0:
        raise ValueError("Host allocation and reserve bytes must be nonnegative")
    meminfo = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    available = int(meminfo["MemAvailable"].split()[0]) * 1024
    cgroup = Path("/sys/fs/cgroup")
    if (cgroup / "memory.max").exists():
        maximum = (cgroup / "memory.max").read_text().strip()
        if maximum != "max":
            current = int((cgroup / "memory.current").read_text())
            stats = dict(
                line.split()
                for line in (cgroup / "memory.stat").read_text().splitlines()
            )
            # Clean unmapped file cache can be reclaimed even while active.
            # Recent checkpoint reads otherwise look like anonymous/pinned
            # working memory and incorrectly reject a large host cache.
            protected = sum(
                int(stats.get(name, 0))
                for name in (
                    "shmem",
                    "file_mapped",
                    "file_dirty",
                    "file_writeback",
                    "unevictable",
                )
            )
            reclaimable = min(current, max(0, int(stats.get("file", 0)) - protected))
            available = min(available, int(maximum) - max(0, current - reclaimable))
    if required_bytes + reserve_bytes > available:
        raise ValueError(
            "Aggregate pinned cache and reserve exceed available host/cgroup memory"
        )
    return dict(
        required_bytes=required_bytes,
        reserve_bytes=reserve_bytes,
        available_bytes=available,
    )


class PinnedHostRecordPool:
    """Allocate the common host-record port without serving-framework imports.

    NVIDIA UVA makes pinned CPU addresses visible to the owning CUDA device.
    Peer devices consume their own owner's transfer publication, rather than
    dereferencing another process's host pointers. Construct outside Graphs.
    """

    def __init__(self, *, layers, slots, row_bytes, device):
        import torch

        if any(
            type(value) is not int or value < 0 for value in (layers, slots, row_bytes)
        ):
            raise ValueError("Pinned byte-page dimensions must be nonnegative integers")
        # Native block readers can fetch a complete region containing a masked,
        # unwritten tail. Match the native zero-initialized GPU page contents.
        self.storage = torch.zeros(
            (layers, slots, row_bytes), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self.ptrs = torch.tensor(
            [self.storage[layer].data_ptr() for layer in range(layers)],
            dtype=torch.uint64,
            device=device,
        )
        self.nbytes = self.storage.numel()
        self._closed = False

    def close(self):
        if self._closed:
            return
        import torch

        torch.cuda.synchronize(self.ptrs.device)
        self.ptrs = None
        self.storage = None
        self._closed = True


def compose_selected_record_ids(
    request_records, request_rows, positions, sequence_lengths
):
    """Compose native request-relative positions with the live request table.

    Graph metadata may contain a dummy slot table used only for shape capture.
    The serving adapter must supply the same live request table and request rows
    as its native K/V gather, rather than that metadata placeholder.
    """
    import torch
    import triton

    from ulacache.storage.kernels.paired_records import compose_record_ids_kernel

    rows = positions.shape[0]
    if request_rows.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("Native request rows and lengths must match selection rows")
    if (
        not positions.is_cuda
        or positions.stride(1) != 1
        or request_records.stride(1) != 1
    ):
        raise ValueError(
            "CUDA request composition requires contiguous native row entries"
        )
    output = torch.empty(positions.shape, dtype=torch.int32, device=positions.device)
    if rows:
        compose_record_ids_kernel[(rows, triton.cdiv(positions.shape[1], 256))](
            request_records,
            request_rows,
            positions,
            sequence_lengths,
            output,
            positions.shape[1],
            request_records.shape[1],
            request_records.stride(0),
            positions.stride(0),
            BLOCK=256,
        )
    return output


def paired_record_views(pages, dtype, head_count, key_dimension, value_dimension):
    key_width = head_count * key_dimension
    value_width = head_count * value_dimension
    rows = pages.view(dtype).reshape(-1, key_width + value_width)
    return (
        rows[:, :key_width].view(-1, head_count, key_dimension),
        rows[:, key_width:].view(-1, head_count, value_dimension),
    )


def read_paired_records(
    pages, record_ids, outputs, dtype, head_count, key_dimension, value_dimension
):
    """Reconstruct bounded prefill chunks in their original logical order."""
    key, value = paired_record_views(
        pages, dtype, head_count, key_dimension, value_dimension
    )
    slots = record_ids.clamp_min(0).long()
    outputs[0].copy_(key.index_select(0, slots))
    outputs[1].copy_(value.index_select(0, slots))


def gather_selected_pairs(
    key,
    value,
    physical_ids,
    positions,
    sequence_lengths,
    row_offsets,
    output_key,
    output_value,
    *,
    padded_columns=0,
):
    """Gather resident pairs into a native compact or page-aligned workspace.

    Native selection positions, lengths and row offsets still control validity,
    packing order and duplicate selections. Only the source addresses change.
    Valid entries must form a prefix of each native selection row.
    """
    import triton

    from ulacache.storage.kernels.paired_records import gather_pairs_kernel

    if key.shape[1:] != value.shape[1:]:
        raise NotImplementedError(
            "Selected paired gathering requires symmetric K/V heads"
        )
    if physical_ids.shape != positions.shape or not physical_ids.is_contiguous():
        raise ValueError("Resident record IDs must match the native selection matrix")
    if not output_key.is_contiguous() or not output_value.is_contiguous():
        raise ValueError("Native packed attention workspaces must be contiguous")
    batch, selected = positions.shape
    heads, dimension = key.shape[1:]
    if (
        key.stride(1) != dimension
        or value.stride(1) != dimension
        or key.stride(2) != 1
        or value.stride(2) != 1
    ):
        raise ValueError("Paired K/V requires contiguous heads and dimensions")
    if padded_columns and padded_columns < selected:
        raise ValueError(
            "Page-aligned output rows must hold the complete selection width"
        )
    columns = padded_columns or selected
    gather_pairs_kernel[(batch, heads, triton.cdiv(columns, 16))](
        key,
        value,
        physical_ids,
        positions,
        sequence_lengths,
        row_offsets,
        output_key,
        output_value,
        selected,
        heads,
        dimension,
        key.stride(0),
        value.stride(0),
        positions.stride(0),
        columns,
        PADDED=bool(padded_columns),
        BLOCK_D=triton.next_power_of_2(dimension),
        num_warps=8,
    )
