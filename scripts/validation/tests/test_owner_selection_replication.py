"""Selection replicas share residency input while preserving publication order."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ulacache.cache.owner_cache import RecordOwnerPagedResidency


@pytest.mark.parametrize('replicated', [False, True])
def test_owner_prepares_the_declared_consumer_union_after_reader_retirement(replicated):
    runtime = RecordOwnerPagedResidency.__new__(RecordOwnerPagedResidency)
    runtime.owner_count, runtime.owner_rank, runtime.owner_logical_capacity = 8, 3, 128
    runtime.native_selections_replicated = replicated
    runtime.selection_input_source_count = 1 if replicated else 8
    runtime.exported_records = list(range(1024, 1056))
    runtime.owner_input_records = list(range(256))
    runtime.consumer_peers = SimpleNamespace(ptrs=object())
    events = []
    runtime.publication = SimpleNamespace(begin_selection=lambda domain: events.append(('retire-and-publish', domain)))
    runtime.owner_residency = SimpleNamespace(prepare_record_residency=lambda domain, records, **options:
        events.append(('plan', domain, len(records))))
    kernels = SimpleNamespace(map_owner_records=Mock(side_effect=lambda *args, **kwargs:
        events.append(('local-replicas', len(args[0])))),
        collect_owner_records=Mock(side_effect=lambda *args, **kwargs:
            events.append(('peer-union', len(args[1])))))
    with patch.dict('sys.modules', {'ulacache.cache.kernels.owner_lookup': kernels}):
        runtime._prepare_exported_records(2, 16)
    count = 16 if replicated else 128
    assert events == [('retire-and-publish', 2),
        ('local-replicas', 16) if replicated else ('peer-union', 128), ('plan', 2, count)]
    if replicated:
        kernels.collect_owner_records.assert_not_called()
        kernels.map_owner_records.assert_called_once_with(runtime.exported_records[:16],
            runtime.owner_input_records[:16], owner_rank=3, owner_count=8, owner_logical_capacity=128)
    else:
        kernels.map_owner_records.assert_not_called()
        kernels.collect_owner_records.assert_called_once_with(runtime.consumer_peers.ptrs,
            runtime.owner_input_records[:128], owner_rank=3, owner_count=8,
            records_per_consumer=16, owner_logical_capacity=128)
