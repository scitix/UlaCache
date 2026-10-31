"""Freeze recency ABI and compiler identity before any captured execution."""
import ctypes
import importlib.util
from pathlib import Path

import pytest


def runtime():
    path=Path(__file__).resolve().parents[3]/'ulacache/compiler.py'
    spec=importlib.util.spec_from_file_location('recency_binding_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pb_keeps_original_module_identity_and_no_naive_flag(monkeypatch):
    monkeypatch.delenv('UDCACHE_RECENCY_POLICY',raising=False)
    module=runtime();calls=[]
    module.configure_compiler(lambda *args,**kwargs:calls.append((args,kwargs)))
    module.load_jit('record_writer',cuda_files=['csrc/cache/cpu_residency.cuh'])
    assert calls[0][0]==('record_writer',)
    assert not any('NAIVE' in flag for flag in calls[0][1]['extra_cuda_cflags'])


def test_every_naive_translation_unit_has_distinct_identity_and_flags(monkeypatch):
    monkeypatch.setenv('UDCACHE_RECENCY_POLICY','naive')
    module=runtime();calls=[]
    module.configure_compiler(lambda *args,**kwargs:calls.append((args,kwargs)))
    for name in ['record_writer','allocator','host_prepare','device_graph']:
        module.load_jit(name,cuda_files=['csrc/cache/cpu_residency.cuh'])
    assert [args[0] for args,_ in calls]==[n+'_naive_lru' for n in ['record_writer','allocator','host_prepare','device_graph']]
    for _,kwargs in calls:
        assert '-DUDCACHE_NAIVE_LRU=1' in kwargs['extra_cuda_cflags']
        assert '-DUDCACHE_NAIVE_LRU=1' in kwargs['extra_cflags']
    monkeypatch.setenv('UDCACHE_RECENCY_POLICY','pb')
    with pytest.raises(RuntimeError,match='frozen'):
        module.load_jit('record_writer')


def test_unknown_policy_is_rejected_before_compiler_binding(monkeypatch):
    monkeypatch.setenv('UDCACHE_RECENCY_POLICY','cpu_lru')
    module=runtime()
    with pytest.raises(ValueError,match='pb or naive'):
        module.configure_compiler(lambda *a,**k:None)
    assert module._compiler is None


def test_descriptor_layout_preserves_state_fields_and_native_work():
    from ulacache.eviction.lru import Group, NaiveGroup
    assert ctypes.sizeof(Group)==88 and ctypes.sizeof(NaiveGroup)==96
    for name,_ in Group._fields_:
        assert getattr(Group,name).offset==getattr(NaiveGroup,name).offset
    assert NaiveGroup.naive_histogram.offset==88
