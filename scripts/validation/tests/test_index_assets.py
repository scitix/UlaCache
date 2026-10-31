"""Check the optional Index compiler boundary without a device or framework."""
from test_module_boundaries import DEVICE_AND_FRAMEWORK, isolated


def test_compact_topk_uses_packaged_sources_and_explicit_compiler():
    process = isolated('''
from pathlib import Path
from ulacache.indexer.topk import compact_topk_module
from ulacache.compiler import configure_compiler

args = ("true", "8", "0", "0", "0", "true")
flags = ("-DSGL_TOPK_V2_MAX_C8_OCC2=0",)
try:
    compact_topk_module(args, flags)
except RuntimeError as error:
    assert "configure_compiler" in str(error)
else:
    raise AssertionError("Index used an implicit compiler")

calls = []
def compiler(*args, **kwargs):
    for name in kwargs["cuda_files"]:
        path = Path(name)
        assert path.is_file() and "/ulacache/csrc/indexer/" in str(path)
    calls.append((args, kwargs))
    return kwargs

configure_compiler(compiler)
result = compact_topk_module(args, flags)
assert compact_topk_module(args, flags) is result
assert len(calls) == 1
assert result["cuda_files"][0].endswith("/csrc/indexer/topk_v2.cuh")
assert flags[0] in result["extra_cuda_cflags"]
assert result["cuda_wrappers"] == [("topk_transform_paged_rank_major",
    "TopKRankMajorKernel<true, 8, 0, 0, 0, true>::transform_paged")]
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
''', DEVICE_AND_FRAMEWORK)
    assert process.returncode == 0, process.stdout + process.stderr
