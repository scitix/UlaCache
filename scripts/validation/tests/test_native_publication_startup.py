"""Compile the optional publisher only when its native storage port opts in."""
from unittest.mock import Mock

import pytest

from ulacache import KeyValueRecordPlane
from ulacache.cache_config import CacheExecutionOptions
import ulacache.transfer.cpu_gpu as paged_records


def test_default_byte_transport_never_loads_the_optional_cuda_publisher(monkeypatch):
    original = Mock()
    publisher = Mock(side_effect=AssertionError('optional publisher loaded'))
    local_publisher = Mock(side_effect=AssertionError('optional local publisher loaded'))
    monkeypatch.setattr(paged_records, '_compile_paged_record_transfer', original)
    monkeypatch.setattr(paged_records, '_compile_paged_record_publish', publisher)
    monkeypatch.setattr(paged_records, '_compile_local_current_record_publish', local_publisher)
    transfer = paged_records.NativePagedRecordTransfer(
        KeyValueRecordPlane('kv', 'native.scaled_fp8.v1', (528, 128)), 64)
    assert original.call_count == 1
    with pytest.raises(RuntimeError, match='before CUDA Graph capture'):
        transfer.publish(None, None, None, None, None, None)
    publisher.assert_not_called()
    local_publisher.assert_not_called()


def test_opted_in_publisher_is_bound_before_any_recorded_write(monkeypatch):
    native = Mock()
    compiled = Mock(return_value=native)
    local_compiled = Mock()
    monkeypatch.setattr(paged_records, '_compile_paged_record_transfer', Mock())
    monkeypatch.setattr(paged_records, '_compile_paged_record_publish', compiled)
    monkeypatch.setattr(paged_records, '_compile_local_current_record_publish', local_compiled)
    transfer = paged_records.NativePagedRecordTransfer(
        KeyValueRecordPlane('kv', 'native.scaled_fp8.v1', (528, 128)), 64,
        fuse_publication=True)
    compiled.assert_called_once_with(64, 64 * 656, False, (528, 128))
    local_compiled.assert_called_once_with(64, 64 * 656, False, (528, 128))
    arguments = tuple(object() for _ in range(6))
    transfer.publish(*arguments)
    native.publish.assert_called_once_with(*arguments)


def test_fused_publication_is_explicitly_native_paged():
    assert not CacheExecutionOptions().fuse_native_record_publication
    with pytest.raises(ValueError, match='requires native-paged storage'):
        CacheExecutionOptions(fuse_native_record_publication=True)
