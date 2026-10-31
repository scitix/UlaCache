"""Portable geometry/lifetime contracts; these do not execute CUDA kernels."""

from dataclasses import replace
from math import prod
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ulacache import (
    UnifiedCacheSpecification, LayerReuseSpecification, LocalCacheBindings, KeyValueRecordPlane, SparseAttentionAccessPlan,
    SparseAccessSpecification, KeyValueStorageLayout, MemoryTierCapacity, UnifiedMemoryCache, ForwardExecutionContext,
)
from ulacache.cache.planning import create_rank_local_layer_groups
from ulacache.cache.local_cache import UnifiedLocalResidency
from test_module_boundaries import DEVICE_AND_FRAMEWORK, isolated


def create_cache_specification():
    return UnifiedCacheSpecification(
        KeyValueStorageLayout("history", (KeyValueRecordPlane("kv", "mla.scaled656.fp8.v1", (528, 128)),), 64),
        LayerReuseSpecification.from_shared_sparse_selection((0, 0, 2)),
        SparseAccessSpecification(maximum_padded_query_count=2, maximum_selected_units=256),
        MemoryTierCapacity(1024, 2048, 512, reserved_logical_record_count=64, maximum_prefill_union_record_count=512),
    )


def test_native_page_geometry_distinguishes_interleaving_and_padding():
    # Small scale record_planes and FP4 payloads must be describable without CUDA's
    # eight-byte transport constraints leaking into the generic geometry.
    plane = KeyValueRecordPlane("packed", "test.fp4.v1", (5, 1), page_alignment_bytes=32)
    segmented = replace(plane, page_memory_order="segment_major")
    assert plane.bytes_per_page(4) == segmented.bytes_per_page(4) == 32
    assert plane.calculate_record_byte_offset(2, 5, 4) == 17
    assert segmented.calculate_record_byte_offset(2, 5, 4) == 22
    assert plane.calculate_record_byte_offset(4, 0, 4) == segmented.calculate_record_byte_offset(4, 0, 4) == 32
    for layout in (plane, segmented):
        offsets = {layout.calculate_record_byte_offset(row, byte, 4) for row in range(8) for byte in range(6)}
        assert len(offsets) == 48
        assert not offsets.intersection(set(range(24, 32)) | set(range(56, 64)))


def test_multiplane_budget_counts_native_padding_and_kv_aliases_once():
    storage = KeyValueStorageLayout("split", (
        KeyValueRecordPlane("k", "test.k", (8,), page_alignment_bytes=64),
        KeyValueRecordPlane("v", "test.v", (3,), page_alignment_bytes=32),
    ), records_per_page=4)
    reuse = LayerReuseSpecification(
        key_value_storage_owners=(0, 0, 2, 2), index_storage_owners=(0, 0, 0, 0),
        sparse_selection_owners=(0, 1, 1, 3), residency_plan_anchors=(0, 1, 1, 3),
    )
    config = UnifiedCacheSpecification(storage, reuse, SparseAccessSpecification(2, 16), MemoryTierCapacity(8, 5, 4))
    assert storage.calculate_storage_bytes(5) == 2 * (64 + 32)
    assert config.calculate_key_value_memory_budget() == {
        "device_key_value_bytes": 384, "host_key_value_bytes": 384,
        "logical_record_capacity": 13, "usable_logical_record_capacity": 13, "unique_key_value_storage_count": 2,
    }
    assert reuse.index_storage_owners != reuse.sparse_selection_owners


@pytest.mark.parametrize("owners", [(0, 0, 1), (1, 1, 2), (0, -1, 2)])
def test_reuse_rejects_alias_chains_and_future_or_negative_producers(owners):
    with pytest.raises(ValueError, match="canonical preceding"):
        LayerReuseSpecification.from_shared_sparse_selection(owners)


def test_block_and_compressed_descriptors_keep_distinct_units():
    access = SparseAccessSpecification(8, 512, selection_unit="block", records_per_selected_unit=4)
    compressed = KeyValueStorageLayout("compressed", (KeyValueRecordPlane("kv", "test.packed", (584,)),),
                             64, record_kind="compressed", tokens_per_record=4)
    assert access.maximum_selected_record_count == 2048
    assert compressed.tokens_per_record == 4
    with pytest.raises(NotImplementedError, match="canonical record"):
        create_rank_local_layer_groups(replace(create_cache_specification(), sparse_access=access))
    with pytest.raises(NotImplementedError, match="uncompressed"):
        create_rank_local_layer_groups(replace(create_cache_specification(), storage_layout=compressed))


@pytest.mark.parametrize("change, error, message", [
    ({"storage_layout": KeyValueStorageLayout("wrong", (KeyValueRecordPlane("kv", "other", (528, 128)),), 64)},
     NotImplementedError, "encodings"),
    ({"sparse_access": SparseAccessSpecification(2, 256, selection_identifier_space="candidate")},
     NotImplementedError, "canonical record"),
    ({"layer_reuse": LayerReuseSpecification((0, 0, 2), (0, 0, 2), (0, 0, 2), (0, 0, 2))},
     NotImplementedError, "storage aliases"),
    ({"layer_reuse": LayerReuseSpecification.from_shared_sparse_selection((0, 1, 0))},
     ValueError, "contiguous"),
    ({"capacity_plan": MemoryTierCapacity(1024, 2**31, 512, reserved_logical_record_count=64)},
     ValueError, "int32"),
])
def test_unsupported_specs_fail_before_device_or_factory_loading(change, error, message):
    services = LocalCacheBindings("cuda:0", Mock(), Mock())
    with patch("ulacache.cache.create.create_rank_local_execution_engine") as factory:
        with pytest.raises(error, match=message):
            UnifiedMemoryCache(replace(create_cache_specification(), **change), services)
        factory.assert_not_called()


def test_portable_api_and_support_check_do_not_import_framework_or_devices():
    process = isolated("""
import ulacache
import ulacache.bindings
from ulacache import UnifiedCacheSpecification, KeyValueRecordPlane, KeyValueStorageLayout, LayerReuseSpecification
from ulacache import SparseAccessSpecification, MemoryTierCapacity, UnifiedMemoryCache, LocalCacheBindings, RecordOwnerBindings
assert ulacache.bindings.RecordOwnerBindings is RecordOwnerBindings
from ulacache.cache.planning import create_rank_local_layer_groups
s = UnifiedCacheSpecification(KeyValueStorageLayout('history', (KeyValueRecordPlane('kv', 'mla.raw576.fp8.v1', (512, 64)),), 64),
    LayerReuseSpecification.from_shared_sparse_selection((0, 0)), SparseAccessSpecification(1, 256),
    MemoryTierCapacity(512, 1024, 128, reserved_logical_record_count=64))
assert create_rank_local_layer_groups(s).group_count == 1
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
""", DEVICE_AND_FRAMEWORK)
    assert process.returncode == 0, process.stdout + process.stderr


class Tensor:
    def __init__(self, *shape, dtype="int32", device="cuda:0", contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.contiguous = contiguous
        self.fill_ = Mock()
        self.copy_ = Mock()

    def numel(self):
        return prod(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return id(self)

    def stride(self):
        return tuple(prod(self.shape[index + 1:]) for index in range(len(self.shape)))


def engine():
    result = Mock()
    result.groups = create_rank_local_layer_groups(create_cache_specification())
    result.active = Tensor(1)
    result.gpu = (object(), object(), object())
    return result


@pytest.fixture
def cache_and_engine():
    backend = engine()
    with patch("ulacache.cache.create.create_rank_local_execution_engine", return_value=backend):
        cache = UnifiedMemoryCache(create_cache_specification(), LocalCacheBindings("cuda:0", Mock(), Mock()))
    yield cache, backend
    cache.close()


def test_runtime_passes_canonical_ids_without_replacing_native_selection(cache_and_engine):
    cache, backend = cache_and_engine
    allocated, selected, current = Tensor(128), Tensor(2, 256), Tensor(2)
    cache.register_logical_records(allocated)
    backend.allocate_unbound.assert_called_once_with(allocated, prefer_gpu=True)
    active = Tensor(1)
    cache.begin_model_forward(ForwardExecutionContext("decode", active, 2))
    backend.active.copy_.assert_called_once_with(active)
    access = SparseAttentionAccessPlan(selected, current)
    cache.prepare_sparse_attention(0, access)
    backend.plan.assert_called_once_with(0, selected, current)
    # A follower reads its own KV bank using the group's existing remap.
    view = cache.get_resident_attention_inputs(1, 2)
    assert view.record_planes == (backend.gpu[1],)
    assert view.device_record_indices is backend.attention_rows.return_value
    assert view.records_per_page == 64 and view.record_layout == "record_major"
    cache.finish_layer_attention(1)
    backend.after_attention.assert_called_once_with(1)
    cache.release_logical_records(allocated)
    backend.release.assert_called_once_with(allocated)


def test_runtime_native_writes_and_close_failures_preserve_ownership(cache_and_engine):
    cache, backend = cache_and_engine
    rows, segments = Tensor(32), (Tensor(32, 528), Tensor(32, 128))
    cache.write_native_record_segments(2, rows, segments)
    backend.write.assert_called_once_with(2, rows, *segments)
    backend.close.side_effect = RuntimeError("drain failed")
    with pytest.raises(RuntimeError, match="drain failed"):
        cache.close()
    assert cache.get_memory_inventory() is backend.memory_inventory.return_value
    backend.close.side_effect = None
    cache.close()
    cache.close()
    assert backend.close.call_count == 2


def test_eager_prefill_view_retains_layer_and_native_row_table(cache_and_engine):
    cache, backend = cache_and_engine
    rows = Tensor(64, 256)
    view = cache.prepare_prefill_attention_inputs(2, rows)
    backend.plan_prefill.assert_called_once_with(2, rows)
    assert view.record_planes == (backend.gpu[2],)
    assert view.device_record_indices is backend.plan_prefill.return_value


def test_real_engine_close_drains_before_host_unregistration_and_is_retryable():
    events = []
    cache = UnifiedLocalResidency.__new__(UnifiedLocalResidency)
    cache.gpu = SimpleNamespace(device="cuda:0")
    cache._drain_transfers = lambda: events.append("join")
    host = Mock()
    cache.host = host
    host.close.side_effect = [RuntimeError("unregister failed"), None]
    cuda = SimpleNamespace(is_current_stream_capturing=lambda: False,
                           synchronize=lambda device: events.append(("sync", device)))
    with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
        with pytest.raises(RuntimeError, match="unregister failed"):
            cache.close()
        assert not getattr(cache, "_closed", False)
        host.close.side_effect = lambda: events.append("unregister")
        cache.close()
        cache.close()
    assert events == ["join", ("sync", "cuda:0"), "join", ("sync", "cuda:0"), "unregister"]
