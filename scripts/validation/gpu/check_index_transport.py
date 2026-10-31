"""Bitwise H200/B300 FlashAG checks through the public standalone bindings."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def worker(rank, args):
    sys.modules["flashinfer_cubin"] = None
    sys.modules["flashinfer_jit_cache"] = None
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=args.init, rank=rank,
                            world_size=args.world, timeout=timedelta(seconds=180))
    from sglang.kernels.jit.utils import load_jit
    from ulacache.compiler import configure_compiler

    capability = torch.cuda.get_device_capability(rank)
    transport = {(9, 0): "h200", (10, 3): "b300"}.get(capability)
    if transport is None or args.transport not in ("auto", transport):
        raise ValueError(f"Unsupported transport/device: {args.transport}/{capability}")
    from sglang.kernels.ops.communication.all_reduce import IPCManager
    from ulacache.flashag import create_index_owner

    modes = ("dense", "compact")

    configure_compiler(load_jit)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    group = SimpleNamespace(world_size=args.world, rank_in_group=rank,
                            cpu_group=dist.group.WORLD, unique_name=f"{transport}-index-validation")
    for mode in modes:
        os.environ["UDCACHE_INDEX_COMPACT_SCORES"] = "1"
        os.environ["UDCACHE_INDEX_SCORE_LAYOUT"] = mode
        owner = create_index_owner(group, max_queries=max(*args.queries, args.graph_queries), max_page_width=2084,
                                   device=f"cuda:{rank}", ipc_manager_factory=IPCManager)
        owner_module = type(owner).__module__
        cols = owner.local_columns
        print(f"rank={rank} {transport} {mode} READY cols={cols}", flush=True)

        def gather(output, local, prefix, packs, _owner=owner, _mode=mode):
            if _mode == "dense":
                _owner.all_gather_into_tensor(output, local)
            else:
                _owner.all_gather_compact(output, local, prefix, packs)

        def data(q, peer, epoch):
            values = torch.arange(q * cols, dtype=torch.int32, device=rank)
            values = torch.bitwise_xor(values, (peer + 1) * 1048576 + epoch * 4096)
            values = values.reshape(q, cols)
            # Preserve NaN payloads and signed zero without FP arithmetic.
            values[:, 0] = -2147483648
            values[:, 1] = 2143289345 + peer
            return values

        def lengths(q, epoch):
            if epoch % 4 == 0:
                return [0] * q
            if epoch % 4 == 1:
                return [cols] * q
            return [4 * ((i * 127 + epoch * 19) % (cols // 4 + 1)) for i in range(q)]

        def descriptors(ls):
            prefix = [0]
            for length in ls:
                prefix.append(prefix[-1] + length)
            return (torch.tensor(prefix, dtype=torch.int32, device=rank),
                    torch.tensor([prefix[-1] // 4], dtype=torch.int64, device=rank))

        def check(output, q, ls, epoch):
            torch.cuda.synchronize()
            # Dense padding and compact trailing capacity are unspecified.
            for peer in range(args.world):
                expected = data(q, peer, epoch)
                actual = output.view(torch.int32)
                offset = peer * sum(ls)
                for query, length in enumerate(ls):
                    if mode == "dense":
                        got = actual[peer * q + query, :length]
                    else:
                        got = actual.reshape(-1)[offset:offset + length]
                        offset += length
                    if not torch.equal(got, expected[query, :length]):
                        raise AssertionError((rank, mode, q, epoch, peer, query, length))
            dist.barrier()

        for q in args.queries:
            local = torch.empty((q, cols), dtype=torch.float32, device=rank)
            output = owner.gathered_workspace[:q * cols * args.world].view(args.world * q, cols)
            for epoch in range(4):
                local.view(torch.int32).copy_(data(q, rank, epoch))
                ls = lengths(q, epoch)
                prefix, packs = descriptors(ls)
                dist.barrier()
                gather(output, local, prefix, packs)
                check(output, q, ls, epoch)
                rows.append(dict(mode=mode, kind="eager", q=q, epoch=epoch,
                                 live_elements_per_rank=sum(ls)))
            print(f"rank={rank} mode={mode} eager q={q} PASS", flush=True)

        q = args.graph_queries
        local = torch.empty((q, cols), dtype=torch.float32, device=rank)
        output = owner.gathered_workspace[:q * cols * args.world].view(args.world * q, cols)
        prefix, packs = descriptors(lengths(q, 1))
        local.view(torch.int32).copy_(data(q, rank, 1))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gather(output, local, prefix, packs)
        stream.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with owner.graph_capture(), torch.cuda.graph(graph, stream=stream):
            gather(output, local, prefix, packs)
        for epoch in range(64):
            local.view(torch.int32).copy_(data(q, rank, epoch))
            ls = lengths(q, epoch)
            p, n = descriptors(ls)
            prefix.copy_(p)
            packs.copy_(n)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            check(output, q, ls, epoch)
            rows.append(dict(mode=mode, kind="graph", q=q, epoch=epoch,
                             live_elements_per_rank=sum(ls)))
        torch.cuda.synchronize()
        del graph
        owner.close()
        print(f"rank={rank} mode={mode} Graph64 PASS", flush=True)
        del owner
    module = Path(sys.modules[owner_module].__file__)
    (outdir / f"rank{rank}.json").write_text(json.dumps(dict(
        complete=True, rank=rank, world=args.world, transport=transport, cuda=str(torch.version.cuda),
        device=torch.cuda.get_device_name(rank), module=str(module),
        module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(), rows=rows), indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=("auto", "h200", "b300"), default="auto")
    p.add_argument("--world", type=int, choices=(8,), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--queries", type=int, nargs="+", default=[1, 4, 24, 192, 384],
                   help="Eager query counts, including the deployment's largest verify batch")
    p.add_argument("--graph-queries", type=int, default=24,
                   help="Query count for the 64 CUDA Graph replays")
    args = p.parse_args()
    if min(*args.queries, args.graph_queries) < 1:
        p.error("Query counts must be positive")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args.init = f"tcp://127.0.0.1:{port}"
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=args.world, join=True)
    root = Path(args.out)
    receipts = [json.loads((root / f"rank{r}.json").read_text()) for r in range(args.world)]
    assert all(x["complete"] for x in receipts)
    assert len({x["module_sha256"] for x in receipts}) == 1
    (root / "COMPLETE.json").write_text(json.dumps(dict(
        complete=True, world=args.world, rank_cases=[len(x["rows"]) for x in receipts],
        queries=args.queries, graph_queries=args.graph_queries,
        module_sha256=receipts[0]["module_sha256"],
        scope="Bitwise live payload, empty/uneven/full GPU descriptors, direct compact output, "
              "CUDA Graph64 with slot reuse; no model or performance claim."), indent=2) + "\n")
