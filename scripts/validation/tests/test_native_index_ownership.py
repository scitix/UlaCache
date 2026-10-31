"""Check native allocator padding, disjoint Index ownership and page tails."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from ulacache.indexer.ownership import NativeIndexOwnership, NativeIndexWriteMapping, NativeIndexQueryLayout


@pytest.mark.parametrize('degree', [2, 4, 8])
@pytest.mark.parametrize('ratio', [1, 2])
def test_native_index_records_partition_the_original_allocator_without_aliases(degree, ratio):
    reserved = 256 // ratio
    capacity = (8192 + 256) // ratio + 3
    ownership = NativeIndexOwnership(capacity, reserved, degree, 64, 1, 68)
    recovered = set()
    counts = [0] * degree
    for logical in range(reserved, capacity):
        owner, local = ownership.owner_and_local_record(logical)
        assert ownership.logical_record(owner, local) == logical
        assert local < ownership.native_page_count * 64
        assert (owner, local) not in recovered
        recovered.add((owner, local))
        counts[owner] += 1
    assert counts == [ownership.live_record_count(rank) for rank in range(degree)]
    assert sum(counts) == capacity - reserved
    assert max(counts) - min(counts) <= 1
    with pytest.raises(ValueError, match='namespace'):
        ownership.owner_and_local_record(capacity)
    with pytest.raises(ValueError, match='padding'):
        ownership.logical_record(0, ownership.local_reserved_record_count - 1)


def test_native_index_mapping_rejects_unmasked_scatter_before_loading_device_dependencies():
    ownership = NativeIndexOwnership(1024, 256, 8, 64, 1, 68)
    with pytest.raises(ValueError, match='skip negative'):
        NativeIndexWriteMapping(ownership, owner_rank=0, maximum_write_records=8,
            device='cuda:0', native_writer_supports_negative_locations=False)


def test_native_index_storage_charges_the_native_extra_page_without_adding_history():
    # The actual V4.1 namespace has one ratio1 bank and three ratio2 banks.
    # Its native FP4 page has64 records of68 bytes and an additional guard page.
    global_tokens = 208739328
    first = NativeIndexOwnership(global_tokens, 256, 8, 64, 1, 68)
    second = NativeIndexOwnership(global_tokens // 2, 128, 8, 64, 1, 68)
    assert first.native_page_count == 407695
    assert second.native_page_count == 203848
    assert sum(first.live_record_count(rank) for rank in range(8)) == global_tokens - 256
    assert 2 * sum(second.live_record_count(rank) for rank in range(8)) == global_tokens - 256
    with pytest.raises(ValueError, match='exceeds'):
        first.logical_record(0, first.native_page_count * 64 - 1)


def test_index_ownership_namespace_import_remains_dependency_free():
    source = str(Path(__file__).resolve().parents[3])
    program = '''import builtins,sys
sys.path.insert(0,sys.argv[1])
original=builtins.__import__
def restricted(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','triton','sglang'):raise AssertionError(name)
 return original(name,*args,**kwargs)
builtins.__import__=restricted
from ulacache.indexer import NativeIndexOwnership,NativeIndexWriteMapping,NativeFP8IndexRuntime
assert NativeIndexOwnership(1024,256,8,64,1,68).local_record_capacity==128
assert NativeFP8IndexRuntime.__name__ == 'NativeFP8IndexRuntime'
'''
    subprocess.run([sys.executable, '-I', '-c', program, source], check=True)


@pytest.mark.parametrize('columns', [1, 63, 64, 65, 135171])
def test_scorer_workspace_bound_matches_actual_meta_allocations(columns):
    torch = pytest.importorskip('torch')
    from ulacache.indexer.native_owner import NativeOwnerIndexScorer

    assert not torch.cuda.is_initialized()
    layout = NativeIndexQueryLayout(32, columns, 64, 128)

    class Peers:
        def __init__(self, *args, **kwargs):
            self.ptrs = torch.empty(8, dtype=torch.uint64)

    class Publication:
        def __init__(self, *args, **kwargs):
            self.epochs = torch.zeros(1, dtype=torch.int64, device='meta')
            self.flags = torch.zeros((1, 3), dtype=torch.int64, device='meta')
            self.peer_flags = Peers()

    def agree(peers, value, **kwargs):
        peers[:] = [value] * 8

    with patch.object(torch.distributed, 'get_rank', return_value=0), \
         patch.object(torch.distributed, 'get_world_size', return_value=8), \
         patch.object(torch.distributed, 'all_gather_object', side_effect=agree), \
         patch('ulacache.compiler.configure_compiler'), \
         patch('ulacache.transfer.ipc.FlowKVPeerShards', Peers), \
         patch('ulacache.cache.reader_lifetime.DistributedResidencyPublication', Publication):
        scorer = NativeOwnerIndexScorer(layout, device='meta', cpu_group=object(),
            ipc_manager_factory=object(), compile_device_module=object(), native_score_tile=object())
    storages = {tensor.untyped_storage()._cdata: tensor.untyped_storage().nbytes()
                for tensor in scorer.device_storage_inventory()}
    assert len(storages) == 9
    assert sum(storages.values()) == layout.storage_bytes(8)
    with pytest.raises(RuntimeError, match='allocation-only'):
        scorer.score(None, None, None, None, None, None, ownership=None,
                     logical_width=columns, compression_ratio=1)
    assert not torch.cuda.is_initialized()
