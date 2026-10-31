"""Publication dependencies stay optional and IPC retirement precedes unmap."""

from types import SimpleNamespace
from unittest.mock import patch

from ulacache.cache.reader_lifetime import DistributedResidencyPublication
from test_module_boundaries import DEVICE_AND_FRAMEWORK, isolated


def test_batch_retirement_passes_device_epochs_without_host_readback():
    calls = []
    runtime = DistributedResidencyPublication.__new__(DistributedResidencyPublication)
    runtime.epochs, runtime.flags = object(), object()
    runtime._device_module = SimpleNamespace(finish_all_readers=lambda *args: calls.append(args))
    runtime.finish_all_readers()
    assert calls == [(runtime.epochs, runtime.flags)]


def test_residency_publication_import_does_not_initialize_a_device_runtime():
    process = isolated('''
from ulacache.cache.reader_lifetime import DistributedResidencyPublication
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
''', DEVICE_AND_FRAMEWORK)
    assert process.returncode == 0, process.stdout + process.stderr


def test_collective_close_waits_for_readers_before_unmapping_exported_flags():
    events = []
    runtime = DistributedResidencyPublication.__new__(DistributedResidencyPublication)
    runtime.epochs = SimpleNamespace(device='cuda:0')
    runtime.cpu_group = object()
    runtime.peer_flags = SimpleNamespace(ptrs=object(), close=lambda: events.append('unmap'))
    runtime._device_module = SimpleNamespace(
        wait_for_all_readers=lambda epochs, pointers: events.append('wait_readers'))
    runtime._closed = False
    distributed = SimpleNamespace(barrier=lambda **arguments: events.append('barrier'))
    torch = SimpleNamespace(distributed=distributed, cuda=SimpleNamespace(
        synchronize=lambda device: events.append('synchronize')))
    with patch.dict('sys.modules', {'torch': torch, 'torch.distributed': distributed}):
        runtime.close()
        runtime.close()
    assert events == ['wait_readers', 'synchronize', 'barrier', 'unmap', 'barrier']
