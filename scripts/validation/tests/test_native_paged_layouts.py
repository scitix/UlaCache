"""Native DSV4 reader offsets, including scale regions and page-tail padding."""

import pytest

from ulacache import KeyValueRecordPlane
from test_module_boundaries import DEVICE_AND_FRAMEWORK, isolated


@pytest.mark.parametrize('segments,page_records,alignment,expected_page_bytes,record,byte,expected_offset', [
    ((576, 8), 16, 576, 9792, 17, 577, 19017),
    ((512, 16), 16, 512, 8704, 15, 527, 8447),
    ((512, 16), 16, 512, 8704, 16, 0, 8704),
    ((256, 32), 64, 256, 18432, 63, 256, 18400),
    ((256, 32), 64, 256, 18432, 64, 0, 18432),
])
def test_native_payload_scale_offsets(segments, page_records, alignment,
                                     expected_page_bytes, record, byte, expected_offset):
    # Golden values follow SGLang's KVLayout/PagedKV native reader ABI.
    plane = KeyValueRecordPlane('key_value', 'native.test.v1', segments,
                               alignment, 'segment_major')
    assert plane.bytes_per_page(page_records) == expected_page_bytes
    assert plane.calculate_record_byte_offset(record, byte, page_records) == expected_offset


def test_paged_transfer_import_preserves_device_dependency_boundary():
    process = isolated('''
from ulacache.transfer.cpu_gpu import NativePagedRecordTransfer
assert NativePagedRecordTransfer.__name__ == 'NativePagedRecordTransfer'
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
''', DEVICE_AND_FRAMEWORK)
    assert process.returncode == 0, process.stdout + process.stderr
