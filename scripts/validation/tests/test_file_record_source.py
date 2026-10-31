"""Native file range, committed owner and partial-chunk CPU contracts."""
import hashlib

import pytest

from test_canonical_plane_reader import make_snapshot
from test_module_boundaries import DEVICE_AND_FRAMEWORK, isolated
from ulacache.transfer.file_io import FileRecordSource
from ulacache.storage.layout import KeyValueRecordPlane

torch = pytest.importorskip('torch')


def plane(encoding='native-scaled-fp8-v32'):
    return KeyValueRecordPlane('key_value', encoding, (528, 128))


def test_offset_range_preserves_complete_scales_rope_and_partial_chunk(tmp_path):
    payload = bytes((i * 13) % 251 for i in range(17 * 656))
    path = tmp_path / 'native-records.bin'
    prefix, suffix = b'opaque-header', b'file-trailer'
    path.write_bytes(prefix + payload + suffix)
    source = FileRecordSource(path, plane(), 17, offset_bytes=len(prefix),
        file_bytes=path.stat().st_size, sha256=hashlib.sha256(payload).hexdigest())
    buffer = torch.empty((8, 656), dtype=torch.uint8)
    chunks = []
    with source.open() as reader:
        while chunk := reader.readinto(buffer):
            chunks.append((chunk.start, chunk.data.shape[0], bytes(chunk.data.flatten().tolist())))
            assert chunk.data.data_ptr() == buffer.data_ptr()
            chunk.release()
    assert [(start, count) for start, count, _ in chunks] == [(0, 8), (8, 8), (16, 1)]
    assert b''.join(data for _, _, data in chunks) == payload
    assert reader.statistics()['payload_sha256_verified']
    assert path.read_bytes() == prefix + payload + suffix


def test_raw_file_hash_failure_prevents_final_chunk_return(tmp_path):
    path = tmp_path / 'native-records.bin'
    payload = bytes(range(256)) * 41  #16 complete656-byte records.
    path.write_bytes(payload)
    source = FileRecordSource(path, plane(), 16, sha256='0' * 64)
    buffer = torch.empty((9, 656), dtype=torch.uint8)
    with source.open() as reader:
        first = reader.readinto(buffer)
        assert first.data.shape[0] == 9
        first.release()
        with pytest.raises(ValueError, match='checksum mismatch'):
            reader.readinto(buffer)
        assert not reader.statistics()['eof_verified']
        assert reader.statistics()['closed']


def test_changed_file_trailer_is_detected_after_bounded_range(tmp_path):
    path = tmp_path / 'with-trailer.bin'
    path.write_bytes(b'x' * (3 * 656) + b'trailer')
    source = FileRecordSource(path, plane(), 3, file_bytes=path.stat().st_size,
                              verify_payload=False)
    with source.open() as reader:
        chunk = reader.readinto(torch.empty((2, 656), dtype=torch.uint8))
        chunk.release()
        with path.open('ab') as file:
            file.write(b'changed')
        with pytest.raises(ValueError, match='size changed'):
            reader.readinto(torch.empty((2, 656), dtype=torch.uint8))


def test_committed_multi_owner_source_remains_one_explicit_owner(tmp_path):
    snapshot, request, payload = make_snapshot(tmp_path / 'snapshot', owners=8, rows=23)
    with pytest.raises(ValueError, match='explicit source owner'):
        FileRecordSource.from_snapshot(snapshot, request.key, 'mla_000')
    source = FileRecordSource.from_snapshot(snapshot, request.key, 'mla_000', owner=3,
                                            record_plane=plane())
    assert source.record_count == request.owned_rows(3, 8) == 3
    with source.open() as reader:
        chunk = reader.readinto(torch.empty((4, 656), dtype=torch.uint8))
        expected = b''.join(payload['mla_000'][row * 656:(row + 1) * 656]
                            for row in (3, 11, 19))
        assert bytes(chunk.data.flatten().tolist()) == expected
        chunk.release()
        assert reader.readinto(torch.empty((4, 656), dtype=torch.uint8)) is None


def test_snapshot_encoding_is_not_inferred_from_equal_width(tmp_path):
    snapshot, request, _ = make_snapshot(tmp_path / 'snapshot')
    with pytest.raises(ValueError, match='encoding'):
        FileRecordSource.from_snapshot(snapshot, request.key, 'mla_000',
                                        record_plane=plane('different-encoding'))
    snapshot.verify_payload = False
    source = FileRecordSource.from_snapshot(snapshot, request.key, 'mla_000')
    assert not source.verify_payload
    assert source.sha256 is not None
    assert FileRecordSource.from_snapshot(snapshot, request.key, 'mla_000',
                                          verify_payload=True).verify_payload
    alias = FileRecordSource.from_snapshot(snapshot, request.key, 'mla_000',
                                            record_plane=plane('mla.scaled656.fp8.v1'))
    assert alias.record_plane.bytes_per_segment == (528, 128)


@pytest.mark.parametrize('change', [
    {'record_count': -1}, {'offset_bytes': -1}, {'file_bytes': 655},
    {'sha256': 'not-a-checksum'}, {'verify_payload': 1}, {'sha256': None},
])
def test_invalid_descriptor_is_rejected_before_open(tmp_path, change):
    args = dict(path=tmp_path / 'absent', record_plane=plane(), record_count=1,
                sha256='0' * 64)
    args.update(change)
    with pytest.raises(ValueError):
        FileRecordSource(**args)


def test_page_segmented_file_requires_an_explicit_conversion(tmp_path):
    layout = KeyValueRecordPlane('key_value', 'native.test', (528, 128),
                                 page_memory_order='segment_major')
    with pytest.raises(NotImplementedError, match='record-major'):
        FileRecordSource(tmp_path / 'absent', layout, 1, verify_payload=False)


def test_empty_range_has_verified_empty_hash(tmp_path):
    path = tmp_path / 'empty.bin'
    path.write_bytes(b'')
    with FileRecordSource(path, plane(), 0, sha256=hashlib.sha256(b'').hexdigest()).open() as reader:
        assert reader.readinto(None) is None
        assert reader.statistics()['payload_sha256_verified']


def test_descriptor_and_reader_import_without_optional_dependencies():
    process = isolated('''
from ulacache.transfer.file_io import FileRecordSource, FileRecordReader
from ulacache.storage.layout import KeyValueRecordPlane
source = FileRecordSource('/not-opened', KeyValueRecordPlane('kv', 'native.test', (656,)),
                          1, verify_payload=False)
assert source.record_count == 1
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
''', DEVICE_AND_FRAMEWORK)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize('owner', range(8))
@pytest.mark.parametrize('rows', [3, 23])
def test_owner_views_copy_only_complete_selected_records(tmp_path, owner, rows):
    payload = torch.arange(rows * 656).remainder(251).to(torch.uint8).reshape(rows, 656)
    path = tmp_path / 'striped-native.bin'
    original = payload.numpy().tobytes()
    path.write_bytes(original)
    source = FileRecordSource(path, plane(), rows, verify_payload=False).select_owner(owner, 8)
    assert source.record_count == len(range(owner, rows, 8))
    assert source.record_stride == 8
    chunks = []
    with source.open() as reader:
        buffer = torch.empty((2, 656), dtype=torch.uint8)
        while (chunk := reader.readinto(buffer)) is not None:
            chunks.append(chunk.data.clone())
            assert chunk.data.data_ptr() == buffer.data_ptr()
            chunk.release()
        actual = torch.cat(chunks) if chunks else buffer[:0]
        assert torch.equal(actual, payload[owner::8])
        assert reader.bytes_read == source.record_count * 656
    assert path.read_bytes() == original


def test_owner_view_never_weakens_hash_policy(tmp_path):
    source = FileRecordSource(tmp_path / 'absent', plane(), 9, sha256='0' * 64)
    with pytest.raises(ValueError, match='preverified'):
        source.select_owner(0, 8)
    from dataclasses import replace
    for stride in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='stride'):
            replace(source, record_stride=stride)
