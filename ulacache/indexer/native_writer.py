"""Preserve native Index quantization through bounded opaque page staging."""

from ulacache.indexer.ownership import NativeIndexOwnership, NativeIndexWriteMapping


class NativeIndexStagingWriter:
    """Let the original encoder write ordinals, then copy each owner's bytes.

    The native producer receives the returned pages and ordinal locations and
    keeps its original quantization/RoPE implementation. Commit reuses native
    page IO with masked owner destinations. Complete live writer replicas and
    unique live logical IDs are supplied by the serving integration. Scratch
    is serial on the caller's stream and cannot be reserved twice at once.
    """

    def __init__(
        self,
        ownership,
        record_plane,
        *,
        owner_rank,
        maximum_write_records,
        device,
        compile_device_module,
    ):
        import torch
        from ulacache.storage.layout import KeyValueRecordPlane
        from ulacache.compiler import configure_compiler
        from ulacache.transfer.cpu_gpu import NativePagedRecordTransfer

        if not isinstance(ownership, NativeIndexOwnership) or not isinstance(
            record_plane, KeyValueRecordPlane
        ):
            raise TypeError(
                "Bind explicit native Index ownership and encoded page traits"
            )
        ownership.validate_rank(owner_rank)
        if (
            record_plane.bytes_per_record != ownership.bytes_per_record
            or record_plane.bytes_per_page(ownership.records_per_native_page)
            != ownership.records_per_native_page * ownership.bytes_per_record
        ):
            raise ValueError(
                "Native Index staging must preserve complete encoded pages"
            )
        configure_compiler(compile_device_module)
        self.ownership = ownership
        self.mapping = NativeIndexWriteMapping(
            ownership,
            owner_rank=owner_rank,
            maximum_write_records=maximum_write_records,
            device=device,
            native_writer_supports_negative_locations=True,
        )
        self.transfer = NativePagedRecordTransfer(
            record_plane, ownership.records_per_native_page
        )
        pages = (
            maximum_write_records + ownership.records_per_native_page - 1
        ) // ownership.records_per_native_page
        self.staging = torch.empty(
            (pages, record_plane.bytes_per_page(ownership.records_per_native_page)),
            dtype=torch.uint8,
            device=device,
        )
        self.ordinals = torch.arange(
            maximum_write_records, dtype=torch.int64, device=device
        )
        self._pending_count = None

    def reserve(self, logical_records):
        """Borrow staging until commit; native encoder locations are all valid."""
        if self._pending_count is not None:
            raise RuntimeError(
                "Commit the current native Index writer before reserving again"
            )
        self.mapping.prepare(logical_records)
        self._pending_count = logical_records.numel()
        return self.staging, self.ordinals[: self._pending_count]

    def commit(self, native_owner_index):
        import torch

        if self._pending_count is None:
            raise RuntimeError("Reserve native Index staging before committing it")
        if (
            native_owner_index.device != self.staging.device
            or native_owner_index.dtype != torch.uint8
            or native_owner_index.shape
            != (self.ownership.native_page_count, self.staging.shape[1])
            or not native_owner_index.is_contiguous()
        ):
            raise ValueError(
                "Native Index commit destination disagrees with its owned page geometry"
            )
        count = self._pending_count
        self.transfer.copy(
            self.staging,
            native_owner_index,
            self.ordinals[:count],
            self.mapping.locations[:count],
        )
        self._pending_count = None

    @property
    def storage_bytes(self):
        return (
            self.staging.nbytes + self.ordinals.nbytes + self.mapping.locations.nbytes
        )

    def device_storage_inventory(self):
        return self.staging, self.ordinals, self.mapping.locations
