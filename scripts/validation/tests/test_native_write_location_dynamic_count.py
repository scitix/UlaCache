"""Changing prefill unions must not create a kernel for every record count."""
import pytest

torch=pytest.importorskip('torch')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA write-location oracle')
@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
def test_native_mapping_reuses_compilation_across_counts_and_preserves_tails(dtype):
    from ulacache.storage.kernels.paged_write import map_native_write_locations, _map_write_locations

    directory=torch.arange(100,device='cuda',dtype=torch.int64)
    directory[1::2]=-directory[1::2]-2
    ids=(torch.arange(16381,device='cuda',dtype=dtype)%99)+1
    ids[::13]=0
    ids[::17]=-1
    gpu=torch.empty(ids.numel()+17,device='cuda',dtype=torch.int32)
    host=torch.empty_like(gpu)
    location=directory[ids.clamp_min(0).long()]
    expected_gpu=torch.where((ids>0)&(location>=0),location,-1).int()
    expected_host=torch.where((ids>0)&(location<=-2),-location-2,-1).int()
    count_after_first=None
    for count in (0,1,3,16,255,256,257,2048,2051,4093,8192,16381):
        gpu.fill_(-777);host.fill_(-888)
        map_native_write_locations(ids[:count],directory,gpu[:count],host[:count])
        torch.cuda.synchronize()
        torch.testing.assert_close(gpu[:count],expected_gpu[:count],rtol=0,atol=0)
        torch.testing.assert_close(host[:count],expected_host[:count],rtol=0,atol=0)
        assert torch.all(gpu[count:]==-777) and torch.all(host[count:]==-888)
        if count:
            compiled=len(_map_write_locations.device_caches[torch.cuda.current_device()][0])
            if count_after_first is None:
                count_after_first=compiled
            assert compiled==count_after_first, 'A changing union length triggered another specialization'
