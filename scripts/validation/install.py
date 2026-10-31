"""Install UlaCache into the official SGLang v0.5.22 CUDA environment."""

from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.validation.apply_patch import BASE, ROOT, inspect_patch

UPSTREAM = "https://github.com/sgl-project/sglang.git"
IMAGE = "docker.io/lmsysorg/sglang:v0.5.22"


def pip_command():
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install"]
    if uv := shutil.which("uv"):
        return [uv, "pip", "install", "--python", sys.executable]
    raise RuntimeError("The serving interpreter needs pip or uv")


def installed_framework_root():
    code = "import importlib.util,json; s=importlib.util.find_spec('sglang'); print(json.dumps(list(s.submodule_search_locations) if s else []))"
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True
    )
    locations = json.loads(result.stdout)
    return Path(locations[0]).resolve().parents[1] if locations else None


def install(framework, *, verify_cuda=False):
    if sys.version_info < (3, 11):
        raise RuntimeError("Official SGLang v0.5.22 requires Python3.11 or newer")
    existing = installed_framework_root()
    framework = (framework or existing or ROOT.parent / "sglang-v0.5.22").resolve()
    if not framework.exists():
        framework.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v0.5.22",
                "--single-branch",
                UPSTREAM,
                str(framework),
            ],
            check=True,
        )
    state = inspect_patch(framework)
    if state["state"] == "ready_to_apply":
        subprocess.run(
            ["git", "apply", str(ROOT / "patches/sglang.patch")],
            cwd=framework,
            check=True,
        )
    # The official image already supplies its native CUDA dependencies. Reuse
    # that exact environment instead of resolving/replacing Torch or kernels.
    command = pip_command()
    targets = ["-e", str(ROOT)]
    reused = existing == framework
    if not reused:
        # A separate checkout needs the upstream build helpers. Do not resolve
        # or replace the image's Torch/CUDA packages in build isolation.
        missing = [
            requirement
            for module, requirement in (
                ("setuptools_rust", "setuptools-rust>=1.11"),
                ("setuptools_scm", "setuptools-scm>=8.0"),
            )
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            subprocess.run([*command, *missing], check=True)
        targets += ["-e", str(framework / "python")]
    subprocess.run(
        [*command, "--no-deps", "--no-build-isolation", *targets], check=True
    )
    code = r"""
import importlib, json, sys
from pathlib import Path
import torch, sglang, ulacache
root, framework = map(Path, sys.argv[1:3])
assert Path(ulacache.__file__).resolve().is_relative_to(root/'ulacache')
assert Path(sglang.__file__).resolve().is_relative_to(framework/'python/sglang')
for name in ('model_configuration','context_parallel_adapter','v0522_owner_tp',
             'native_tp_cache'):
    importlib.import_module('sglang.srt.ulacache.'+name)
if sys.argv[3] == '1':
    assert torch.cuda.is_available(), 'CUDA is unavailable in this container'
print(json.dumps(dict(ulacache=str(Path(ulacache.__file__).resolve()),
    sglang=str(Path(sglang.__file__).resolve()), sglang_version=sglang.__version__,
    torch_version=torch.__version__, torch_cuda=torch.version.cuda,
    cuda_verified=sys.argv[3]=='1')))
"""
    # Ignore a caller's PYTHONPATH to verify the actual editable installation.
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(ROOT),
            str(framework),
            str(int(verify_cuda)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    imported = json.loads(child.stdout.strip().splitlines()[-1])
    return dict(
        passed=True,
        official_image=IMAGE,
        official_source=UPSTREAM,
        existing_framework_install_reused=reused,
        upstream_commit=BASE,
        patch=inspect_patch(framework),
        imports=imported,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sglang-root",
        type=Path,
        help="Default: use the official image checkout; otherwise clone an exact-tag checkout",
    )
    parser.add_argument("--verify-cuda", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    value = install(args.sglang_root, verify_cuda=args.verify_cuda)
    rendered = json.dumps(value, indent=2) + "\n"
    if args.report:
        with args.report.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
