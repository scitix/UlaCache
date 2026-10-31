"""Native page/segment byte IO over the existing exclusive residency plans.

This transport preserves native packed payloads and scales. It does not perform
quantization, selection, residency planning, or attention. Host and device banks
use the same native page layout, including padding.
"""

from ulacache.storage.layout import KeyValueRecordPlane
from ulacache.compiler import cache_once, load_jit


@cache_once
def _compile_paged_record_transfer(
    records_per_page, bytes_per_page, segment_major, segment_widths
):
    parameters = ",".join(
        map(
            str, (records_per_page, bytes_per_page, int(segment_major), *segment_widths)
        )
    )
    return load_jit(
        "ulacache_paged_record_transfer",
        str(records_per_page),
        str(bytes_per_page),
        str(int(segment_major)),
        *map(str, segment_widths),
        cuda_files=["csrc/transfer/cpu_gpu.cuh"],
        cuda_wrappers=[
            ("exchange", f"ulacache::exchange_paged_records<{parameters}>"),
            ("copy", f"ulacache::copy_paged_records<{parameters}>"),
        ],
    )


@cache_once
def _compile_paged_record_publish(
    records_per_page, bytes_per_page, segment_major, segment_widths
):
    parameters = ",".join(
        map(
            str, (records_per_page, bytes_per_page, int(segment_major), *segment_widths)
        )
    )
    return load_jit(
        "ulacache_paged_record_publish_v1",
        str(records_per_page),
        str(bytes_per_page),
        str(int(segment_major)),
        *map(str, segment_widths),
        cuda_files=["csrc/storage/publish.cuh"],
        cuda_wrappers=[("publish", f"ulacache::publish_paged_records<{parameters}>")],
    )


@cache_once
def _compile_local_current_record_publish(
    records_per_page, bytes_per_page, segment_major, segment_widths
):
    parameters = ",".join(
        map(
            str, (records_per_page, bytes_per_page, int(segment_major), *segment_widths)
        )
    )
    return load_jit(
        "ulacache_local_current_record_publish_v1",
        str(records_per_page),
        str(bytes_per_page),
        str(int(segment_major)),
        *map(str, segment_widths),
        cuda_files=["csrc/storage/publish.cuh"],
        cuda_wrappers=[
            ("publish", f"ulacache::publish_local_current_records<{parameters}>")
        ],
    )


class NativePagedRecordTransfer:
    """Bind one plane's geometry and compile before CUDA Graph capture.

    ``exchange`` consumes the existing int32[5, capacity] residency plan and its
    eight counters. The caller commits ownership after every plane/bank moved.
    ``copy`` consumes unique indices in each domain and nonoverlapping input
    and output memory ranges; it is not an in-place device permutation.
    """

    def __init__(
        self,
        record_plane: KeyValueRecordPlane,
        records_per_page: int,
        *,
        launch_blocks=4,
        launch_threads=1024,
        fuse_publication=False,
    ):
        if launch_blocks < 1 or not 32 <= launch_threads <= 1024 or launch_threads % 32:
            raise ValueError(
                "Record transfer needs positive blocks and warp-aligned threads"
            )
        self.record_plane = record_plane
        self.records_per_page = records_per_page
        self.bytes_per_page = record_plane.bytes_per_page(records_per_page)
        self.launch_blocks, self.launch_threads = launch_blocks, launch_threads
        # record-major segments occupy one contiguous native record. Keep the
        # storage/quantization plane intact; specialize only byte transport.
        self.transfer_segment_widths = (
            (sum(record_plane.bytes_per_segment),)
            if record_plane.page_memory_order == "record_major"
            else record_plane.bytes_per_segment
        )
        self._device_module = _compile_paged_record_transfer(
            records_per_page,
            self.bytes_per_page,
            record_plane.page_memory_order == "segment_major",
            self.transfer_segment_widths,
        )
        if type(fuse_publication) is not bool:
            raise ValueError("Fused publication must be boolean")
        self._publish_module = None
        self._local_current_publish_module = None
        if fuse_publication:
            self._publish_module = _compile_paged_record_publish(
                records_per_page,
                self.bytes_per_page,
                record_plane.page_memory_order == "segment_major",
                record_plane.bytes_per_segment,
            )
            self._local_current_publish_module = _compile_local_current_record_publish(
                records_per_page,
                self.bytes_per_page,
                record_plane.page_memory_order == "segment_major",
                record_plane.bytes_per_segment,
            )

    def exchange(
        self,
        host_record_storage,
        device_record_storage,
        transfer_plan,
        transfer_counters,
    ):
        """Exchange uint8[banks, pages, page_bytes] before residency commit."""
        self._device_module.exchange(
            host_record_storage,
            device_record_storage,
            transfer_plan,
            transfer_counters,
            self.launch_blocks,
            self.launch_threads,
        )

    def copy(
        self,
        source_record_storage,
        destination_record_storage,
        source_record_indices,
        destination_record_indices,
    ):
        """Copy uint8[pages, page_bytes] using CUDA int32/int64 record indices."""
        self._device_module.copy(
            source_record_storage,
            destination_record_storage,
            source_record_indices,
            destination_record_indices,
            self.launch_blocks,
            self.launch_threads,
        )

    def publish(
        self,
        source_record_storage,
        device_record_storage,
        host_record_storage,
        logical_record_ids,
        source_record_indices,
        residency_directory,
        *,
        local_device_after_commit=False,
    ):
        """Publish native bytes directly through the unchanged exclusive directory."""
        if self._publish_module is None:
            raise RuntimeError(
                "Bind fused native publication before CUDA Graph capture"
            )
        module = (
            self._local_current_publish_module
            if local_device_after_commit
            else self._publish_module
        )
        if module is None:
            raise RuntimeError(
                "Bind local current publication before CUDA Graph capture"
            )
        module.publish(
            source_record_storage,
            device_record_storage,
            host_record_storage,
            logical_record_ids,
            source_record_indices,
            residency_directory,
        )
