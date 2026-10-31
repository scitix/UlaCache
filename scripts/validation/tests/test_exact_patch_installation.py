"""Patch installation requires exact whole files, including untouched hunk gaps."""
from pathlib import Path
import subprocess

import pytest

from scripts.validation import apply_patch


def git(root, *arguments):
    return subprocess.check_output(['git', *arguments], cwd=root, text=True)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    repo, package = tmp_path/'upstream', tmp_path/'package'
    repo.mkdir()
    (package/'patches').mkdir(parents=True)
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.name', 'Test')
    git(repo, 'config', 'user.email', 'test@example.org')
    for name in ('a.py', 'b.py'):
        (repo/name).write_text(''.join(f'value_{i} = {i}\n' for i in range(100)))
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'Base')
    monkeypatch.setattr(apply_patch, 'BASE', git(repo, 'rev-parse', 'HEAD').strip())
    monkeypatch.setattr(apply_patch, 'ROOT', package)
    for name in ('a.py', 'b.py'):
        p = repo/name
        p.write_text(p.read_text().replace('value_0 = 0', 'value_0 = 1'))
    patch = package/'patches/sglang.patch'
    patch.write_text(git(repo, 'diff', '--full-index'))
    git(repo, 'restore', '.')
    return repo, patch


def test_pristine_and_fully_patched_checkouts_are_recognized(checkout):
    repo, patch = checkout
    assert apply_patch.inspect_patch(repo)['state'] == 'ready_to_apply'
    git(repo, 'apply', str(patch))
    assert apply_patch.inspect_patch(repo)['state'] == 'already_patched'


def test_an_edit_outside_patch_context_is_rejected(checkout):
    repo, patch = checkout
    p = repo/'a.py'
    p.write_text(p.read_text().replace('value_99 = 99', 'value_99 = -1'))
    git(repo, 'apply', '--check', str(patch))  # Hunk-only checking misses this edit.
    with pytest.raises(ValueError, match='partially applied'):
        apply_patch.inspect_patch(repo)


def test_a_partial_application_is_rejected(checkout):
    repo, _ = checkout
    p = repo/'a.py'
    p.write_text(p.read_text().replace('value_0 = 0', 'value_0 = 1'))
    with pytest.raises(ValueError, match='partially applied'):
        apply_patch.inspect_patch(repo)


def test_framework_validation_rejects_an_empty_or_unpatched_directory(tmp_path):
    from scripts.validation.check import ROOT, check_framework_imports

    with pytest.raises(ValueError, match="Not a patched SGLang checkout"):
        check_framework_imports(ROOT, tmp_path)
    entry = tmp_path / "python/sglang/srt/ulacache/entry.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("# No actual import bindings\n")
    with pytest.raises(ValueError, match="no UlaCache import bindings"):
        check_framework_imports(ROOT, tmp_path)


def test_startup_guard_matches_the_installed_native_source(tmp_path):
    import hashlib
    from scripts.validation.check import check_framework_source_guards

    source = tmp_path / 'python/sglang/srt'
    adapters = source / 'ulacache'
    adapters.mkdir(parents=True)
    target = source / 'native.py'
    target.write_text('VALUE = 1\n')
    guard = adapters / 'binding.py'
    guard.write_text("PARENT_FILES = {'sglang.srt.native': '" + '0' * 64 + "'}\n")
    with pytest.raises(ValueError, match='Stale framework source guard'):
        check_framework_source_guards(tmp_path)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    guard.write_text("PARENT_FILES = {'sglang.srt.native': '" + digest + "'}\n")
    assert check_framework_source_guards(tmp_path) == 1
    target.write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='sglang.srt.native'):
        check_framework_source_guards(tmp_path)
