"""Owner current preparation preserves the native ID and reader promises."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ulacache.cache.owner_cache import RecordOwnerPagedResidency


@pytest.mark.parametrize('eligible', [False, True])
def test_owner_maps_current_once_and_retains_native_fallback(eligible):
    torch = pytest.importorskip('torch')
    runtime = RecordOwnerPagedResidency.__new__(RecordOwnerPagedResidency)
    runtime.maximum_query_count = 4
    runtime._selection_records = {}
    runtime._forward_current_records = None
    current = torch.tensor([64, 65], dtype=torch.int64)
    mapped = torch.tensor([32, -1], dtype=torch.int32)
    runtime._map_owner_records = Mock(return_value=mapped)
    runtime.owner_residency = SimpleNamespace(
        prepare_forward_current_records=Mock(return_value=eligible))
    assert runtime.prepare_forward_current_records(current) is eligible
    runtime.owner_residency.prepare_forward_current_records.assert_called_once_with(mapped)
    assert runtime._forward_current_records is (current if eligible else None)


def test_owner_rejects_different_current_ids_before_publication_or_device_mutation():
    torch = pytest.importorskip('torch')
    runtime = RecordOwnerPagedResidency.__new__(RecordOwnerPagedResidency)
    runtime._forward_current_records = torch.tensor([64, 65], dtype=torch.int64)
    runtime._prepare_exported_records = Mock()
    with pytest.raises(ValueError, match='bound native current-ID tensor'):
        runtime.plan(0, object(), runtime._forward_current_records.clone())
    runtime._prepare_exported_records.assert_not_called()


def test_batched_owner_current_preparation_cannot_start_after_a_selection():
    torch = pytest.importorskip('torch')
    runtime = RecordOwnerPagedResidency.__new__(RecordOwnerPagedResidency)
    runtime.maximum_query_count = 4
    runtime._selection_records = {0: object()}
    runtime._map_owner_records = Mock()
    with pytest.raises(RuntimeError, match='before the first owner selection'):
        runtime.prepare_forward_current_records(torch.tensor([64], dtype=torch.int64))
    runtime._map_owner_records.assert_not_called()


def test_prepared_owner_union_reuses_native_transaction_after_publication():
    runtime = RecordOwnerPagedResidency.__new__(RecordOwnerPagedResidency)
    runtime.native_selections_replicated = True
    runtime.owner_rank, runtime.owner_count, runtime.owner_logical_capacity = 0, 2, 64
    runtime.exported_records, runtime.owner_input_records = [64, 65], [0, 0]
    runtime.selection_input_source_count = 1
    runtime.publication = Mock()
    runtime.owner_residency = Mock()
    kernels = SimpleNamespace(map_owner_records=Mock(), collect_owner_records=Mock())
    with patch.dict('sys.modules', {'ulacache.cache.kernels.owner_lookup': kernels}):
        runtime._prepare_exported_records(0, 2, current_already_prepared=True)
    runtime.publication.begin_selection.assert_called_once_with(0)
    runtime.owner_residency.prepare_record_residency.assert_called_once_with(
        0, [0, 0], current_already_prepared=True)
