"""Apply the single UlaCache patch to the exact official SGLang v0.5.22 base."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE = "baa9befa381511cd4d56f83980951e5a45fffc6d"
ROOT = Path(__file__).resolve().parents[2]


def inspect_patch(root):
    root = Path(root).resolve(strict=True)
    patch = ROOT / "patches/sglang.patch"
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if head != BASE:
        raise ValueError(f"Expected official SGLang v0.5.22 at {BASE}; found {head}")

    def possible(reverse=False):
        cmd = ["git", "apply", "--check"]
        if reverse:
            cmd.append("--reverse")
        return subprocess.run(
            [*cmd, str(patch)], cwd=root, capture_output=True, text=True
        )

    # Full Git blob identities reject edits outside individual patch hunks.
    expected = []
    for block in re.split(r"(?m)^diff --git ", patch.read_text())[1:]:
        lines = block.splitlines()
        relative = lines[0].split(" b/", 1)[1]
        index = next(line for line in lines if line.startswith("index "))
        before, after = index.split()[1].split("..")
        file = root / relative
        if file.exists():
            raw = file.read_bytes()
            actual = hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\x00" + raw
            ).hexdigest()
        else:
            actual = "0" * 40
        expected.append((actual == before, actual == after))
    if not expected:
        raise ValueError("The patch contains no source identities")
    forward, reverse = possible(), possible(True)
    if all(old for old, _ in expected) and forward.returncode == 0:
        state = "ready_to_apply"
    elif all(new for _, new in expected) and reverse.returncode == 0:
        state = "already_patched"
    else:
        raise ValueError(
            "The patch is partially applied or affected files contain other changes.\n"
            + forward.stderr
            + reverse.stderr
        )
    return dict(
        state=state,
        base=BASE,
        patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = inspect_patch(args.sglang_root)
    if not args.check and result["state"] == "ready_to_apply":
        subprocess.run(
            ["git", "apply", str(ROOT / "patches/sglang.patch")],
            cwd=args.sglang_root,
            check=True,
        )
        result = inspect_patch(args.sglang_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
