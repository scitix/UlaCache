"""Large pinned admission distinguishes clean file cache from working memory."""

from pathlib import Path

import pytest

from ulacache.storage.buffers import validate_host_allocation_headroom


def memory(monkeypatch, *, current, file_bytes, maximum=2000, available=2500, **stats):
    values = {
        '/proc/meminfo': f'MemAvailable: {available} kB\n',
        '/sys/fs/cgroup/memory.max': str(maximum * 1024),
        '/sys/fs/cgroup/memory.current': str(current * 1024),
        '/sys/fs/cgroup/memory.stat': '\n'.join(f'{key} {value * 1024}'
            for key, value in dict(file=file_bytes, **stats).items()),
    }
    original_read, original_exists = Path.read_text, Path.exists
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **kw:
        values[str(path)] if str(path) in values else original_read(path, *a, **kw))
    monkeypatch.setattr(Path, 'exists', lambda path:
        str(path) in values or original_exists(path))


def test_recent_clean_checkpoint_cache_is_reclaimable_even_when_active(monkeypatch):
    memory(monkeypatch, current=1900, file_bytes=1800, active_file=1800, inactive_file=0)
    result = validate_host_allocation_headroom(1700 * 1024, reserve_bytes=100 * 1024)
    assert result['available_bytes'] == 1900 * 1024


def test_anonymous_or_pinned_working_memory_cannot_be_reclaimed(monkeypatch):
    memory(monkeypatch, current=1900, file_bytes=100)
    with pytest.raises(ValueError, match='available host/cgroup memory'):
        validate_host_allocation_headroom(200 * 1024, reserve_bytes=1024)


def test_shared_mapped_dirty_writeback_and_locked_bytes_remain_charged(monkeypatch):
    memory(monkeypatch, current=1900, file_bytes=1000,
        shmem=400, file_mapped=200, file_dirty=100, file_writeback=50, unevictable=50)
    assert validate_host_allocation_headroom(300 * 1024, reserve_bytes=0)['available_bytes'] == 300 * 1024
    with pytest.raises(ValueError, match='available host/cgroup memory'):
        validate_host_allocation_headroom(300 * 1024, reserve_bytes=1024)


def test_physical_available_memory_still_bounds_reclaimable_cgroup_cache(monkeypatch):
    memory(monkeypatch, current=1900, file_bytes=1800, available=400)
    with pytest.raises(ValueError, match='available host/cgroup memory'):
        validate_host_allocation_headroom(401 * 1024, reserve_bytes=0)
