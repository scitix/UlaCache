"""Gather opaque native records from already published owner GPU pages."""

from ulacache.storage.layout import KeyValueRecordPlane
from ulacache.compiler import cache_once, load_jit


@cache_once
def _compile_peer_record_gather(
    records_per_page, bytes_per_page, segment_major, segment_widths, vector_bytes
):
    parameters = ",".join(
        map(
            str,
            (
                records_per_page,
                bytes_per_page,
                int(segment_major),
                vector_bytes,
                *segment_widths,
            ),
        )
    )
    return load_jit(
        "ulacache_peer_paged_record_gather",
        str(records_per_page),
        str(bytes_per_page),
        str(int(segment_major)),
        str(vector_bytes),
        *map(str, segment_widths),
        cuda_files=["csrc/transfer/gpu_peer.cuh"],
        cuda_wrappers=[
            ("gather", f"ulacache::gather_peer_paged_records<{parameters}>")
        ],
    )


class NativePagedPeerRecordTransfer:
    """Copy published owner-local device records into native destination pages.

    Reuse the pointer table from ``FlowKVPeerShards``. Residency planning must
    promote CPU records and publish the selected owners' device locations
    before calling ``gather``. The caller retains those locations and IPC
    mappings until every reader finishes; this component does not synchronize
    owners, allocate storage, change residency, or implement model selection.

    Sources may repeat. Nonnegative destination indices must be unique and
    must not alias source storage. A negative owner, source or destination
    skips that entry, preserving destination bytes. Padding is never copied.
    """

    def __init__(
        self,
        record_plane: KeyValueRecordPlane,
        records_per_page: int,
        *,
        launch_blocks=None,
        launch_threads=256,
        vector_bytes=None,
    ):
        if (
            (launch_blocks is not None and launch_blocks < 1)
            or not 32 <= launch_threads <= 1024
            or launch_threads % 32
        ):
            raise ValueError(
                "Peer record gather needs positive blocks and warp-aligned threads"
            )
        if vector_bytes is None:
            vector_bytes = (
                16
                if all(width % 16 == 0 for width in record_plane.bytes_per_segment)
                else 8
            )
        if vector_bytes not in (8, 16):
            raise ValueError("Peer record vectors must be 8 or 16 bytes")
        self.record_plane = record_plane
        self.records_per_page = records_per_page
        self.bytes_per_page = record_plane.bytes_per_page(records_per_page)
        self.launch_blocks, self.launch_threads = launch_blocks, launch_threads
        self._device_module = _compile_peer_record_gather(
            records_per_page,
            self.bytes_per_page,
            record_plane.page_memory_order == "segment_major",
            record_plane.bytes_per_segment,
            vector_bytes,
        )

    def gather(
        self,
        source_peer_pointers,
        source_owner_ranks,
        source_record_indices,
        destination_record_pages,
        destination_record_indices,
        *,
        source_bank_byte_offset=0,
    ):
        """Read CUDA uint64 peer pointers and signed int32/int64 index vectors.

        All owners share this native plane geometry and bank byte offset.
        Source indices already identify physical records in the owners' device
        banks, rather than global logical IDs or CPU directory locations.
        The destination is contiguous uint8[pages, bytes_per_page].
        Automatic launches expose one warp per record, up to 1024 blocks.
        Narrow small selections use 512 blocks; wide or longer selections
        expose more CTAs. The launch is determined by the padded shape,
        without reading query counts from the device during Graph replay.
        Explicit startup grids remain available for reproducible comparisons.
        """
        blocks = self.launch_blocks
        if blocks is None:
            warps_per_block = self.launch_threads // 32
            count = source_record_indices.numel()
            needed = max(1, (count + warps_per_block - 1) // warps_per_block)
            ceiling = (
                512
                if self.record_plane.bytes_per_record <= 384 and count <= 12288
                else 1024
            )
            blocks = min(ceiling, 1 << (needed - 1).bit_length())
        self._device_module.gather(
            source_peer_pointers,
            source_owner_ranks,
            source_record_indices,
            destination_record_pages,
            destination_record_indices,
            source_bank_byte_offset,
            blocks,
            self.launch_threads,
        )
