"""Map native writer outputs to exclusive device/host record locations."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["Count"])
def _map_write_locations(
    LogicalRecords,
    Directory,
    DeviceRecords,
    HostRecords,
    Count,
    BlockSize: tl.constexpr,
):
    index = tl.program_id(0) * BlockSize + tl.arange(0, BlockSize)
    logical_record = tl.load(LogicalRecords + index, index < Count, other=-1)
    valid = (index < Count) & (logical_record > 0)
    location = tl.load(Directory + logical_record, valid, other=-1)
    tl.device_assert(
        (~valid) | (location != -1), "Native writer record is not registered"
    )
    tl.store(
        DeviceRecords + index,
        tl.where(valid & (location >= 0), location, -1).to(tl.int32),
        index < Count,
    )
    tl.store(
        HostRecords + index,
        tl.where(valid & (location <= -2), -location - 2, -1).to(tl.int32),
        index < Count,
    )


def map_native_write_locations(
    logical_record_ids, residency_directory, device_record_indices, host_record_indices
):
    if logical_record_ids.numel():
        _map_write_locations[(triton.cdiv(logical_record_ids.numel(), 256),)](
            logical_record_ids,
            residency_directory,
            device_record_indices,
            host_record_indices,
            logical_record_ids.numel(),
            256,
        )


@triton.jit
def _write_native_segment(
    LogicalRecords,
    Source,
    Directory,
    DevicePages,
    HostPointers,
    SourceStride: tl.constexpr,
    HostBankOffset: tl.constexpr,
    RecordsPerPage: tl.constexpr,
    PageBytes: tl.constexpr,
    RecordBytes: tl.constexpr,
    SegmentPrefix: tl.constexpr,
    SegmentBytes: tl.constexpr,
    SegmentMajor: tl.constexpr,
    BlockSize: tl.constexpr,
):
    index = tl.program_id(0)
    logical_record = tl.load(LogicalRecords + index)
    if logical_record > 0:
        location = tl.load(Directory + logical_record)
        tl.device_assert(location != -1, "Native writer record is not registered")
        record = tl.where(location >= 0, location, -location - 2)
        page, slot = record // RecordsPerPage, record % RecordsPerPage
        if SegmentMajor:
            offset = (
                page * PageBytes + SegmentPrefix * RecordsPerPage + slot * SegmentBytes
            )
        else:
            offset = page * PageBytes + slot * RecordBytes + SegmentPrefix
        if location >= 0:
            destination = DevicePages + offset
        else:
            host = tl.load(HostPointers).to(tl.pointer_type(tl.uint8))
            destination = host + HostBankOffset + offset
        columns = tl.arange(0, BlockSize)
        data = tl.load(
            Source + index * SourceStride + columns, columns < SegmentBytes, other=0
        )
        tl.store(destination + columns, data, columns < SegmentBytes)


def write_native_record_segment(
    logical_record_ids,
    segment_bytes,
    residency_directory,
    device_record_pages,
    host_record_pointers,
    *,
    host_bank_offset,
    records_per_page,
    bytes_per_page,
    bytes_per_record,
    segment_prefix,
    segment_major,
):
    if logical_record_ids.numel():
        width = segment_bytes.shape[1]
        _write_native_segment[(logical_record_ids.numel(),)](
            logical_record_ids,
            segment_bytes,
            residency_directory,
            device_record_pages,
            host_record_pointers,
            segment_bytes.stride(0),
            host_bank_offset,
            records_per_page,
            bytes_per_page,
            bytes_per_record,
            segment_prefix,
            width,
            segment_major,
            triton.next_power_of_2(width),
        )
