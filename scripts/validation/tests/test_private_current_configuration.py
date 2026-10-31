"""The new current contract is opt-in and cannot affect ordinary backends."""
import pytest
from ulacache.cache_config import CacheExecutionOptions


def test_existing_options_leave_current_preparation_serial():
    assert not CacheExecutionOptions().prepare_current_records_in_parallel


@pytest.mark.parametrize('arguments', [dict(), dict(record_storage_backend='native_paged'),
    dict(prepare_residency_in_parallel=True, record_storage_backend='unsupported')])
def test_parallel_current_requires_the_supported_native_transaction(arguments):
    with pytest.raises(ValueError, match='requires supported parallel residency'):
        CacheExecutionOptions(prepare_current_records_in_parallel=True, **arguments)


def test_native_parallel_transaction_can_explicitly_bind_private_currents():
    assert CacheExecutionOptions(record_storage_backend='native_paged',
        prepare_residency_in_parallel=True, prepare_current_records_in_parallel=True).prepare_current_records_in_parallel


def test_contiguous_parallel_transaction_can_explicitly_bind_currents():
    assert CacheExecutionOptions(prepare_residency_in_parallel=True,
        prepare_current_records_in_parallel=True).prepare_current_records_in_parallel
