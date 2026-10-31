"""Exercise dependency boundaries in clean interpreters, without device code.

The optional SGLang adapter check accepts a caller-provided file path. A plain
UlaCache checkout neither needs nor discovers an SGLang repository.
"""

import os
from pathlib import Path
import subprocess
import sys
import unittest


SOURCE = Path(__file__).resolve().parents[3]
ADAPTER_FILE = os.environ.get("UDCACHE_SGLANG_ADAPTER_FILE")


def isolated(code, blocked, *arguments):
    bootstrap = """
import importlib.abc
import sys

class BlockedDependency(ModuleNotFoundError):
    pass

class RejectDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise BlockedDependency('dependency boundary rejected ' + fullname)

blocked = BLOCKED
sys.path.insert(0, SOURCE)
sys.meta_path.insert(0, RejectDependency())
""".replace("BLOCKED", repr(tuple(blocked))).replace("SOURCE", repr(str(SOURCE)))
    return subprocess.run(
        [sys.executable, "-I", "-c", bootstrap + code, *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


DEVICE_AND_FRAMEWORK = ("torch", "triton", "sglang", "tvm", "tvm_ffi", "cupy", "cuda")


class ModuleBoundaries(unittest.TestCase):
    def test_public_api_has_one_package_and_no_import_redirect(self):
        process = isolated("""
import importlib.util
finders = tuple(sys.meta_path)
import ulacache
import ulacache.compiler as runtime
from ulacache import UlaCache, UnifiedCacheSpecification, init, ResidentView, LogicalSlotAllocator, PrefetchPolicy
from ulacache.client import UlaCache as Client, UnifiedMemoryCache
assert callable(UlaCache)
assert UlaCache is Client is UnifiedMemoryCache
assert UlaCache.__module__ == 'ulacache.client'
assert callable(init) and ResidentView.__module__ == 'ulacache.cache.access'
assert LogicalSlotAllocator.__module__ == 'ulacache.bindings'
assert PrefetchPolicy(enabled=True).completion == 'auto'
assert 'ulacache.cache.create' not in sys.modules
assert 'ulacache.cache.tp_cache' not in sys.modules
assert callable(UnifiedCacheSpecification)
assert tuple(sys.meta_path) == finders
assert importlib.util.find_spec('udcache') is None
compiler = lambda *args, **kwargs: None
runtime.configure_compiler(compiler)
assert runtime._compiler is compiler
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
""", DEVICE_AND_FRAMEWORK)
        self.check_process(process)

    def check_process(self, process):
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_root_and_optional_namespaces_do_not_load_core_or_device_modules(self):
        process = isolated("""
import ulacache
import ulacache.indexer
import ulacache.speculative
assert ulacache.__version__
assert ulacache.indexer.__all__
assert ulacache.speculative.__all__
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
""", (*DEVICE_AND_FRAMEWORK, "ulacache.cache", "ulacache.storage", "ulacache.transfer", "ulacache.compiler"))
        self.check_process(process)

    def test_public_descriptors_import_without_device_or_serving_dependencies(self):
        process = isolated("""
import importlib
modules = (
    'cache.access', 'cache.access', 'cache.specification', 'cache.runtime',
    'cache.layers', 'cache.layers', 'cache.request_pages',
    'cache.sparse_regions', 'cache.prefill.plans', 'storage.capacity', 'storage.capacity',
    'storage.layout', 'storage.layout', 'storage.workspace',
    'storage.workspace', 'storage.layout', 'storage.capacity',
    'storage.workspace', 'storage.capacity', 'storage.workspace',
)
for name in modules:
    importlib.import_module('ulacache.' + name)
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
""", (*DEVICE_AND_FRAMEWORK, "ulacache.cache.tp_cache", "ulacache.cache.owner_cache",
      "ulacache.transfer", "ulacache.compiler"))
        self.check_process(process)

    def test_independent_dpa_contracts_load_devices_only_when_constructed(self):
        process = isolated("""
from ulacache.cache.local_cache import UnifiedLocalResidency
from ulacache.tools.snapshot.restore import PreparedSequenceRestore, write_sequence_records
assert callable(PreparedSequenceRestore)
assert callable(write_sequence_records)
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
try:
    UnifiedLocalResidency(groups=None, gpu_slots=64, host_slots=64,
        row_bytes=656, max_queries=3, topk=2048, max_write_rows=8192,
        device='cuda', host_pool_factory=None)
except BlockedDependency as exc:
    assert str(exc) == 'dependency boundary rejected torch'
else:
    raise AssertionError('explicit DPA feature ignored its missing dependency')
""", DEVICE_AND_FRAMEWORK)
        self.check_process(process)

    @unittest.skipUnless(ADAPTER_FILE, "optional UDCACHE_SGLANG_ADAPTER_FILE not provided")
    def test_optional_sglang_adapter_abc_does_not_import_udcache(self):
        process = isolated("""
import importlib.util
spec = importlib.util.spec_from_file_location('native_adapter_boundary_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.FlowKVPackedMLAAdapter.__abstractmethods__
assert module.FlowKVSharedIndexAdapter.__abstractmethods__
assert module.is_flowkv_adapter(object()) is False
assert not any(name == 'ulacache' or name.startswith('ulacache.') for name in sys.modules)
try:
    module.FlowKVPackedMLASpec
except BlockedDependency as exc:
    assert 'ulacache' in str(exc)
else:
    raise AssertionError('enabled descriptor import should require UlaCache')
""", (*DEVICE_AND_FRAMEWORK, "ulacache"), str(Path(ADAPTER_FILE).resolve()))
        self.check_process(process)


if __name__ == "__main__":
    unittest.main()
