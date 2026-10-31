"""Native DPA file IO borrows the same composed storage used by decoding."""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ulacache.storage.buffers import RecordStorage


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if os.environ.get("ULACACHE_TEST_CUDA") == "1" else [])
)
def test_native_dpa_round_trip_uses_composed_gpu_and_host_planes(monkeypatch, device):
    import torch

    root = os.environ.get("HISPARSE_NATIVE_SOURCE_ROOT")
    if not root:
        pytest.skip("Set HISPARSE_NATIVE_SOURCE_ROOT to the patched framework")
    path = Path(root) / "python/sglang/srt/ulacache/v0522_dpa.py"
    spec = importlib.util.spec_from_file_location("_native_dpa_io_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    gpu = torch.zeros((2, 2, 64 * 656), dtype=torch.uint8, device=device)
    host = torch.zeros((2, 2, 64 * 656), dtype=torch.uint8)
    plane = SimpleNamespace(
        device_record_pages=gpu,
        host_record_pool=SimpleNamespace(storage=host),
        fuse_native_record_publication=True,
    )
    storage = RecordStorage.__new__(RecordStorage)
    storage.plane_storage = [plane]
    assert not hasattr(storage, "device_record_pages")
    assert storage.fuse_native_record_publication is True
    plan = SimpleNamespace(
        storage_producer_layers=(0, 1), residency_domain_indices_by_layer=(0, 0)
    )
    directory = torch.tensor([-1, 2, -4, 4, -7], dtype=torch.int64, device=device)
    retired = []
    engine = SimpleNamespace(
        record_storage=storage,
        execution_plan=plan,
        residency_states=(SimpleNamespace(directory=directory),),
        _require_retired_readers=lambda: retired.append(True),
    )
    pool = module.DPACacheHooks()
    pool.memory_cache = SimpleNamespace(_execution_engine=engine)
    pool.layer_num, pool.device, pool.cpu_offloading_chunk_size = 2, device, 2
    index_events = []
    pool.index_key_cache = SimpleNamespace(
        cpu_copy=lambda rows: rows.cpu().clone(),
        load_cpu_copy=lambda value, rows: index_events.append((value, rows.cpu())),
    )
    rows = torch.arange(1, 5, dtype=torch.int64, device=device)
    payloads = [
        torch.arange(4 * 656, dtype=torch.int64)
        .reshape(4, 656)
        .add(layer * 13)
        .remainder(251)
        .to(torch.uint8)
        for layer in range(2)
    ]
    pointers = gpu.data_ptr(), host.data_ptr()
    with pool._native_record_io(rows) as records:
        for layer, payload in enumerate(payloads):
            records.write(layer, 0, payload)
            assert torch.equal(records.read(layer, 0, 4), payload)
    saved = pool.get_cpu_copy(rows)
    assert all(
        tuple(chunk.shape) == (2, 1, 656) for chunks in saved["kv"] for chunk in chunks
    )
    gpu.zero_()
    host.zero_()
    pool.load_cpu_copy(saved, rows)
    with pool._native_record_io(rows) as records:
        for layer, payload in enumerate(payloads):
            assert torch.equal(records.read(layer, 0, 4), payload)
            assert torch.equal(gpu.view(2, 128, 656)[layer, 2].cpu(), payload[0])
            assert torch.equal(host.view(2, 128, 656)[layer, 2], payload[1])
    assert pointers == (gpu.data_ptr(), host.data_ptr())
    assert len(retired) == 8 and len(index_events) == 1
    storage.plane_storage.append(plane)
    with pytest.raises(ValueError, match="one complete record plane"):
        pool._native_record_io(rows)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if os.environ.get("ULACACHE_TEST_CUDA") == "1" else [])
)
def test_native_receive_publishes_into_the_same_composed_banks(monkeypatch, device):
    import torch
    from ulacache.cache.local_paged_cache import NativePagedResidency
    from ulacache.transfer.native_paged_receive import NativePagedRecordReceiveWindow

    if device == "cpu":
        original_empty = torch.empty

        def empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return original_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", empty)
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda device: SimpleNamespace(synchronize=lambda: None),
        )

    def host_storage(*, layers, slots, row_bytes, device):
        data = torch.zeros((layers, slots, row_bytes), dtype=torch.uint8)
        return SimpleNamespace(storage=data, nbytes=data.nbytes)

    gpu = torch.zeros((2, 2, 64 * 656), dtype=torch.uint8, device=device)
    host = host_storage(layers=2, slots=128, row_bytes=656, device=device)
    storage = RecordStorage.__new__(RecordStorage)
    storage.plane_storage = [
        SimpleNamespace(
            device_record_pages=gpu,
            host_record_pool=host,
            fuse_native_record_publication=True,
        )
    ]
    directories = [
        torch.full((256,), -1, dtype=torch.int64, device=device) for _ in range(2)
    ]
    for domain, directory in enumerate(directories):
        for row in range(64):
            directory[64 + row] = row if (row + domain) % 2 else -row - 2
    retired = []
    native = NativePagedResidency.__new__(NativePagedResidency)
    native.record_storage = storage
    native.execution_plan = SimpleNamespace(
        storage_producer_layers=(0, 1), residency_domain_indices_by_layer=(0, 1)
    )
    native.residency_states = tuple(SimpleNamespace(directory=d) for d in directories)
    native._require_retired_readers = lambda: retired.append(True)
    native.registration_frames = torch.empty((2, 1), dtype=torch.int64, device=device)
    native.presence_rings = (object(), object())
    native.maximum_query_count = 2
    native.maximum_selection_width = 4
    layout = SimpleNamespace(
        records_per_page=64,
        record_planes=(
            SimpleNamespace(page_memory_order="record_major", bytes_per_segment=(656,)),
        ),
    )
    cache = SimpleNamespace(
        _execution_engine=native,
        cache_specification=SimpleNamespace(
            storage_layout=layout,
            capacity_plan=SimpleNamespace(
                device_record_capacity=128, host_record_capacity=128,
                logical_record_capacity=256,
            ),
            layer_reuse=SimpleNamespace(layer_count=2),
        ),
    )
    from ulacache.transfer.direct_receiver import DirectRecordReceiver

    direct = DirectRecordReceiver(cache)
    assert direct.native_engine is native
    assert direct.engine.frames is native.registration_frames
    assert direct.engine.states is native.residency_states
    assert direct.engine.gpu.data_ptr() == gpu.data_ptr()
    assert direct.engine.host.storage.data_ptr() == host.storage.data_ptr()
    assert direct.additional_registration_regions() == ((host.storage.data_ptr(), host.storage.nbytes),)
    assert direct.device_registration_regions() == ((gpu.untyped_storage().data_ptr(), gpu.untyped_storage().nbytes()),)
    assert direct.metadata_inventory()["payload_staging_bytes"] == 0
    window = NativePagedRecordReceiveWindow(
        cache,
        device=device,
        allocate_host_record_storage=host_storage,
        record_capacity=64,
        copy_chunk_records=17,
    )
    assert window.engine.gpu.data_ptr() == gpu.data_ptr()
    assert window.engine.host.storage.data_ptr() == host.storage.data_ptr()
    window.reserve("request", torch.tensor([1]), record_count=64)
    payload = (
        torch.arange(2 * 64 * 656).reshape(2, 64, 656).remainder(251).to(torch.uint8)
    )
    window.storage.storage.copy_(payload)
    assert window.publish("request") == 64
    for layer in range(2):
        for row in range(64):
            location = int(directories[layer][64 + row])
            actual = (
                gpu.view(2, 128, 656)[layer, location].cpu()
                if location >= 0
                else host.storage[layer, -location - 2]
            )
            assert torch.equal(actual, payload[layer, row])
    assert retired == [True] and window.available_records == 64
