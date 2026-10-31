"""Compile native transport exports and verify opaque record bytes on CUDA."""

import argparse
import json
from pathlib import Path


def check_current_prefill():
    import torch
    from ulacache.cache.kernels.tensor_parallel_prefill import (
        _staged_current, install_staged_current,
    )
    from ulacache.storage.layout import NativePageLayout

    page = NativePageLayout(64, (528, 128), 1)

    def offsets(count):
        row = torch.arange(count, device='cuda')[:, None]
        byte = torch.arange(656, device='cuda')[None, :]
        return (row // 64 * page.bytes_per_page + torch.where(
            byte < 528, row % 64 * 528 + byte,
            64 * 528 + row % 64 * 128 + byte - 528))

    hashes = set()
    counts = [1, 2, 7, 16, 31, 64, 65, 127, 257, 2048, 8191, 8192]
    for count in counts:
        current = torch.arange(count, dtype=torch.int32, device='cuda') + 100
        rows = torch.cat((torch.full((17,), 3, dtype=torch.int32, device='cuda'), current))
        fresh = torch.cat((torch.full((17,), -1, dtype=torch.int32, device='cuda'),
                           torch.arange(count, dtype=torch.int32, device='cuda')))
        staging = torch.zeros(((count + 63) // 64 * page.bytes_per_page,),
                              dtype=torch.uint8, device='cuda')
        target = torch.full(((count + 17 + 63) // 64 * page.bytes_per_page,),
                            77, dtype=torch.uint8, device='cuda')
        expected = target.clone()
        values = ((torch.arange(count, device='cuda')[:, None] * 31
                   + torch.arange(656, device='cuda')[None, :]) % 251).to(torch.uint8)
        staging[offsets(count)] = values
        expected[offsets(count + 17)[17:]] = values
        install_staged_current(rows, fresh, current, staging, target, native_page=page)
        torch.cuda.synchronize()
        assert torch.equal(target, expected), count
        kernel = _staged_current.warmup(
            rows, fresh, current, staging, target, rows.numel(), count,
            656, 64, page.bytes_per_page, page.segment_bytes, 8, 128,
            grid=(min(64, (rows.numel() + 7) // 8),), num_warps=8)
        hashes.add(kernel.hash)
    assert len(hashes) == 1, 'Input lengths must not create new CUDA modules'
    return dict(counts=counts, byte_equal=True, compiled_variants=len(hashes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from sglang.kernels.jit.utils import load_jit
    from ulacache.compiler import configure_compiler
    from ulacache import KeyValueRecordPlane
    from ulacache.transfer.cpu_gpu import NativePagedRecordTransfer
    from ulacache.transfer.gpu_peer import NativePagedPeerRecordTransfer
    from ulacache.cache.reader_lifetime import _compile_residency_publication
    from ulacache.flashag.ipc import _module as compile_ipc

    configure_compiler(load_jit)
    torch.cuda.set_device(0)
    _compile_residency_publication()
    compile_ipc(4)
    results = []
    for segments, page, alignment, order in (
        ((528, 128), 64, 1, 'record_major'),
        ((576, 8), 16, 576, 'segment_major'),
        ((512, 16), 16, 512, 'segment_major'),
    ):
        plane = KeyValueRecordPlane('kv', 'test.native.bytes', segments, alignment, order)
        width = plane.bytes_per_page(page)
        transfer = NativePagedRecordTransfer(plane, page, launch_blocks=64,
                                             launch_threads=256, fuse_publication=True)
        peer = NativePagedPeerRecordTransfer(plane, page)
        host = torch.zeros((4, width), dtype=torch.uint8, pin_memory=True)
        source = torch.randint(0, 256, (4, width), dtype=torch.uint8, device='cuda')
        source_cpu = source.cpu()
        source_ids = [0, page - 1, page, 2 * page + 1]
        destination_ids = [3 * page, 1, 2 * page - 1, page + 2]
        src = torch.tensor(source_ids, dtype=torch.int32, device='cuda')
        dst = torch.tensor(destination_ids, dtype=torch.int64, device='cuda')

        def copy_reference(target, source_id, destination_id):
            prefix = 0
            for size in segments:
                def offset(record):
                    p, r = divmod(record, page)
                    return p * width + (prefix * page + r * size if order == 'segment_major'
                                        else r * sum(segments) + prefix)
                a, b = offset(source_id), offset(destination_id)
                target.view(-1)[b:b + size] = source_cpu.view(-1)[a:a + size]
                prefix += size

        expected = torch.zeros_like(source_cpu)
        for a, b in zip(source_ids, destination_ids):
            copy_reference(expected, a, b)
        for destination in (torch.zeros_like(source), host):
            transfer.copy(source, destination, src, dst)
            torch.cuda.synchronize()
            assert torch.equal(destination.cpu(), expected)

        device = torch.zeros_like(source)
        host.zero_()
        logical = torch.arange(1, 5, dtype=torch.int32, device='cuda')
        directory = torch.full((8,), -1, dtype=torch.int64, device='cuda')
        directory[logical.long()] = torch.tensor([destination_ids[0], -destination_ids[1] - 2,
                                                  destination_ids[2], -destination_ids[3] - 2],
                                                 dtype=torch.int64, device='cuda')
        transfer.publish(source, device, host, logical, src, directory)
        torch.cuda.synchronize()
        expected_device, expected_host = torch.zeros_like(source_cpu), torch.zeros_like(source_cpu)
        for i, (a, b) in enumerate(zip(source_ids, destination_ids)):
            copy_reference(expected_device if i % 2 == 0 else expected_host, a, b)
        assert torch.equal(device.cpu(), expected_device)
        assert torch.equal(host, expected_host)

        directory[logical.long()] = dst
        device.zero_()
        transfer.publish(source, device, host, logical, src, directory, local_device_after_commit=True)
        torch.cuda.synchronize()
        assert torch.equal(device.cpu(), expected)

        device.zero_()
        pointers = torch.tensor([source.data_ptr()], dtype=torch.uint64, device='cuda')
        owners = torch.zeros(4, dtype=torch.int32, device='cuda')
        peer.gather(pointers, owners, src, device, dst)
        torch.cuda.synchronize()
        assert torch.equal(device.cpu(), expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            transfer.copy(source, device, src, dst)
        for _ in range(4):
            source.random_(0, 256)
            source_cpu.copy_(source.cpu())
            expected.zero_()
            for a, b in zip(source_ids, destination_ids):
                copy_reference(expected, a, b)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(device.cpu(), expected)
        results.append(dict(segments=segments, records_per_page=page, page_bytes=width,
                            order=order, copy=True, publication=True, local_current=True,
                            peer=True, changed_input_graph_replays=4))
        print(json.dumps(results[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(complete=True, device=torch.cuda.get_device_name(),
        cases=results, native_publication_exports_compiled=True,
        compact_ipc_exports_compiled=True,
        current_prefill=check_current_prefill()), indent=2) + '\n')


if __name__ == '__main__':
    main()
