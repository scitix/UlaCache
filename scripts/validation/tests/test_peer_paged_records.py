"""Peer-native byte transport retains explicit geometry and dependency ports."""

from unittest.mock import Mock, patch
from types import SimpleNamespace

import pytest

from ulacache import KeyValueRecordPlane
from ulacache.transfer.gpu_peer import NativePagedPeerRecordTransfer
from test_module_boundaries import DEVICE_AND_FRAMEWORK, isolated


@pytest.mark.parametrize('segments,page,alignment,order,page_bytes', [
    ((1024, 128), 64, 1, 'record_major', 73728),
    ((576, 8), 256, 576, 'segment_major', 149760),
    ((512, 16), 128, 512, 'segment_major', 67584),
    ((256, 32), 64, 256, 'segment_major', 18432),
    ((5, 1), 4, 32, 'segment_major', 32),
    ((16, 4, 3), 4, 64, 'record_major', 128),
])
def test_peer_transport_uses_the_complete_native_page_geometry(segments, page, alignment, order, page_bytes):
    plane = KeyValueRecordPlane('native', 'test.opaque.v1', segments, alignment, order)
    module = Mock()
    with patch('ulacache.transfer.gpu_peer._compile_peer_record_gather', return_value=module) as compile:
        transfer = NativePagedPeerRecordTransfer(plane, page)
    compile.assert_called_once_with(page, page_bytes, order == 'segment_major', segments, 16 if all(width % 16 == 0 for width in segments) else 8)
    peers, owners, source_records, destination_pages, destination_records = (object() for _ in range(5))
    source_records = SimpleNamespace(numel=lambda:7)
    transfer.gather(peers, owners, source_records, destination_pages, destination_records,
        source_bank_byte_offset=page_bytes * 7)
    module.gather.assert_called_once_with(peers, owners, source_records,
        destination_pages, destination_records, page_bytes * 7, 1, 256)


@pytest.mark.parametrize('record_count,explicit_blocks,expected_blocks', [
    (0,None,1), (512,None,64), (4096,None,512), (65536,None,1024), (4096,4,4),
])
def test_native_gather_parallelism_covers_record_prefixes_and_preserves_explicit_grids(
        record_count, explicit_blocks, expected_blocks):
    module = Mock()
    plane = KeyValueRecordPlane('native','test.opaque.v1',(576,8),576,'segment_major')
    with patch('ulacache.transfer.gpu_peer._compile_peer_record_gather', return_value=module):
        transfer = NativePagedPeerRecordTransfer(plane,256,launch_blocks=explicit_blocks)
    source_records = SimpleNamespace(numel=lambda:record_count)
    peers, owners, destination, destination_records = (object() for _ in range(4))
    transfer.gather(peers,owners,source_records,destination,destination_records)
    module.gather.assert_called_once_with(peers,owners,source_records,destination,
        destination_records,0,expected_blocks,256)


def test_peer_transport_import_needs_no_tensor_library_or_serving_framework():
    process = isolated('''
from ulacache.transfer.gpu_peer import NativePagedPeerRecordTransfer
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
''', DEVICE_AND_FRAMEWORK)
    assert process.returncode == 0, process.stdout + process.stderr
