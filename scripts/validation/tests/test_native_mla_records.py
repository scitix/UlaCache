"""Native MLA records retain the reader's payload/scale/RoPE byte regions."""
import pytest

from ulacache.storage.layout import NativeMLARecord, native_mla_encodings_match
from ulacache.storage.layout import NativePageLayout
from test_module_boundaries import DEVICE_AND_FRAMEWORK, isolated


@pytest.mark.parametrize('encoding,segments', [
    ('raw576', (512, 64)), ('scaled656', (528, 128)),
])
def test_native_page_segments_cover_every_byte(encoding, segments):
    record = NativeMLARecord(encoding)
    page = NativePageLayout(64, record.segment_bytes, 1)
    assert record.segment_bytes == segments
    assert record.row_bytes == sum(segments)
    assert page.bytes_per_page == 64 * record.row_bytes
    offsets = [page.byte_offset(row, byte)
               for row in range(64) for byte in range(record.row_bytes)]
    assert sorted(offsets) == list(range(page.bytes_per_page))


@pytest.mark.parametrize('encoding,prefill,decode', [
    ('raw576', 'trtllm', 'trtllm'),
    ('scaled656', 'flashmla_sparse', 'flashmla_kv'),
    ('scaled656', 'flashmla_kv', 'flashmla_kv'),
])
def test_explicit_reader_contract(encoding, prefill, decode):
    NativeMLARecord(encoding).validate_backends(prefill, decode)


def test_unsupported_reader_is_rejected():
    with pytest.raises(NotImplementedError):
        NativeMLARecord('scaled656').validate_backends('trtllm', 'trtllm')
    with pytest.raises(ValueError):
        NativeMLARecord('auto')


def test_only_explicit_native_encoding_alias_is_accepted():
    assert native_mla_encodings_match('mla.scaled656.fp8.v1', 'native-scaled-fp8-v32')
    assert native_mla_encodings_match('native-scaled-fp8-v32', 'mla.scaled656.fp8.v1')
    assert native_mla_encodings_match('mla.raw576.fp8.v1', 'mla.raw576.fp8.v1')
    assert not native_mla_encodings_match('mla.raw576.fp8.v1', 'native-scaled-fp8-v32')
    assert not native_mla_encodings_match('another656byteformat', 'mla.scaled656.fp8.v1')
    assert not native_mla_encodings_match(None, None)


def test_native_record_import_is_dependency_free():
    process = isolated('''
from ulacache.storage.layout import NativeMLARecord
assert NativeMLARecord('scaled656').row_bytes == 656
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
''', DEVICE_AND_FRAMEWORK)
    assert process.returncode == 0, process.stdout + process.stderr
