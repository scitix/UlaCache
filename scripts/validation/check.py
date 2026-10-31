"""Check source layout, local native includes and lazy imports without a GPU."""

from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
IMPORTS = r"""
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class BlockDevice(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','triton','sglang','cuda','cupy','tvm_ffi'):
            raise AssertionError('Unexpected eager import: '+fullname)
sys.meta_path.insert(0, BlockDevice())
import ulacache, ulacache.cache, ulacache.storage, ulacache.transfer, ulacache.indexer
import ulacache.flashag, ulacache.speculative, ulacache.tools
for name in ulacache.__all__:
    getattr(ulacache, name)
from ulacache import PositionShardLayout, PositionShardBindings
import ulacache.compiler as runtime
calls=[]
def compiler(*args, **kwargs):
    from pathlib import Path
    for key in ('cuda_files','cpp_files'):
        for p in kwargs.get(key,[]):
            assert Path(p).is_file(), p
    for p in kwargs.get('extra_include_paths',[]):
        assert Path(p).is_dir(), p
    calls.append((args, kwargs))
    return calls[-1]
runtime.configure_compiler(compiler)
from ulacache.indexer.topk import compact_topk_module
compact_topk_module(("float", "int", "32", "64", "8", "true"), ())
assert calls
print('Lazy imports and packaged compiler requests passed')
"""


def check(root):
    package = root / "ulacache"
    python = list(package.rglob("*.py"))
    for p in python:
        ast.parse(p.read_text(), filename=str(p))
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if name.startswith("ulakv") or name.startswith("sglang"):
                    raise ValueError(f"Library import boundary: {p}: {name}")
                if name.startswith("ulacache."):
                    target = root.joinpath(*name.split("."))
                    if (
                        not target.with_suffix(".py").is_file()
                        and not (target / "__init__.py").is_file()
                    ):
                        raise ValueError(f"Unresolved library module: {p}: {name}")
    experiments = list((root / "scripts").glob("7_*"))
    experiment_modules = [
        p for directory in experiments for p in directory.rglob("*.py")
    ]
    for p in experiment_modules:
        for node in ast.walk(ast.parse(p.read_text(), filename=str(p))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "ulacache."
            ):
                target = root.joinpath(*node.module.split("."))
                if (
                    not target.with_suffix(".py").is_file()
                    and not (target / "__init__.py").is_file()
                ):
                    raise ValueError(
                        f"Unresolved experiment library module: {p}: {node.module}"
                    )
    headers = [
        p
        for p in (package / "csrc").rglob("*")
        if p.suffix in (".cuh", ".h", ".hpp", ".cu")
    ]
    edges = 0
    for p in headers:
        for mode, name in re.findall(
            r'^\s*#\s*include\s*([<"])([^>\"]+)', p.read_text(), re.M
        ):
            if mode != '"' and not name.startswith("sgl_kernel/"):
                continue
            if not any(
                (d / name).is_file()
                for d in (p.parent, package, package / "csrc", package / "csrc/include")
            ):
                raise ValueError(f"Missing packaged include: {p}: {name}")
            edges += 1
    subprocess.run(
        [sys.executable, "-I", "-c", IMPORTS, str(root)], check=True, text=True
    )
    return dict(
        python_modules=len(python),
        experiment_modules=len(experiment_modules),
        native_headers=len(headers),
        local_include_edges=edges,
        lazy_imports=True,
    )


def check_framework_imports(root, framework_root):
    """Check the installed patch against this library, including moved symbols."""
    entry = framework_root / "python/sglang/srt/ulacache/entry.py"
    if not entry.is_file():
        raise ValueError(f"Not a patched SGLang checkout: missing {entry}")
    checked = 0
    for path in (framework_root / "python" / "sglang").rglob("*.py"):
        source = path.read_text()
        if "ulacache" not in source:
            continue
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if module.startswith("ulacache."):
                target = root.joinpath(*module.split("."))
            elif module.startswith("sglang.srt.ulacache"):
                target = (framework_root / "python").joinpath(*module.split("."))
            else:
                continue
            target = (
                target.with_suffix(".py")
                if target.with_suffix(".py").is_file()
                else target / "__init__.py"
            )
            if not target.is_file():
                raise ValueError(
                    f"Unresolved framework import: {path}:{node.lineno}: {node.module}"
                )
            declarations = ast.parse(target.read_text(), filename=str(target))
            names = set()
            for declaration in declarations.body:
                if isinstance(
                    declaration, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    names.add(declaration.name)
                elif isinstance(declaration, (ast.Import, ast.ImportFrom)):
                    names.update(
                        alias.asname or alias.name.split(".")[0]
                        for alias in declaration.names
                    )
                elif isinstance(declaration, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        declaration.targets
                        if isinstance(declaration, ast.Assign)
                        else [declaration.target]
                    )
                    names.update(
                        item.id
                        for target in targets
                        for item in ast.walk(target)
                        if isinstance(item, ast.Name)
                    )
            if "__getattr__" not in names:
                missing = [
                    alias.name
                    for alias in node.names
                    if alias.name != "*"
                    and alias.name not in names
                    and not (
                        target.name == "__init__.py"
                        and (
                            (target.parent / f"{alias.name}.py").is_file()
                            or (target.parent / alias.name / "__init__.py").is_file()
                        )
                    )
                ]
                if missing:
                    raise ValueError(
                        f"Unresolved framework symbols: {path}:{node.lineno}: {node.module}: {missing}"
                    )
            checked += 1
    if not checked:
        raise ValueError("The SGLang checkout contains no UlaCache import bindings")
    return checked


def check_framework_source_guards(framework_root):
    """Catch stale startup fingerprints before launching any model weights."""
    checked = 0
    source_root = framework_root / "python"
    for path in (source_root / "sglang/srt/ulacache").glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            try:
                values = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                continue
            for module, fingerprints in values.items():
                if not isinstance(module, str) or not module.startswith("sglang."):
                    continue
                accepted = (
                    fingerprints
                    if isinstance(fingerprints, (tuple, list))
                    else (fingerprints,)
                )
                if not all(
                    isinstance(value, str) and len(value) == 64 for value in accepted
                ):
                    continue
                target = source_root.joinpath(*module.split(".")).with_suffix(".py")
                if not target.is_file():
                    # Separate legacy framework entries may describe modules
                    # that are not present in the official release.
                    continue
                if hashlib.sha256(target.read_bytes()).hexdigest() not in accepted:
                    raise ValueError(
                        f"Stale framework source guard: {path.name}: {module}"
                    )
                checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument(
        "--sglang-root",
        type=Path,
        help="Check imports in an already patched SGLang checkout",
    )
    args = parser.parse_args()
    if args.wheel:
        with tempfile.TemporaryDirectory(prefix="ulacache-wheel-") as path:
            with zipfile.ZipFile(args.wheel) as wheel:
                wheel.extractall(path)
            expected = {
                source.relative_to(ROOT)
                for source in (ROOT / "ulacache").rglob("*.py")
            }
            packaged = {
                source.relative_to(path)
                for source in (Path(path) / "ulacache").rglob("*.py")
            }
            if packaged != expected:
                raise ValueError(
                    "Wheel/source module mismatch: "
                    f"missing={sorted(map(str, expected - packaged))}, "
                    f"extra={sorted(map(str, packaged - expected))}"
                )
            value = check(Path(path))
    else:
        value = check(ROOT)
    if args.sglang_root:
        value["framework_imports"] = check_framework_imports(ROOT, args.sglang_root)
        value["framework_source_guards"] = check_framework_source_guards(
            args.sglang_root
        )
    print(json.dumps(dict(passed=True, **value), indent=2))


if __name__ == "__main__":
    main()
