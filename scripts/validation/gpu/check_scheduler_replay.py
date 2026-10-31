"""Validate the three Figure11 metadata planners without a model or KV payload."""

from pathlib import Path
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/7_5_scheduler/_runtime"))


def main():
    import torch
    from sglang.kernels.jit.utils import load_jit
    from ulacache.compiler import configure_compiler
    from ulacache.cache.kernels.cpu_residency import (
        create_host_tier_state,
        create_host_tier_workspace,
        commit_host_tier,
        release_host_tier_leases,
    )
    from ulacache.cache.kernels.local_cache import (
        expire_recent_groups,
        remap_resident_rows,
    )
    from ulacache.eviction.lru import create_presence_ring
    from ulacache.eviction.kernels.launch_config import PBLaunch, prepare as prepare_pb
    from pb_helpers import rebuild_ring

    configure_compiler(load_jit)
    gpu_slots, host_slots, queries, topk = 1048576, 1048576, 3, 2048
    historical = (
        torch.cat((torch.arange(1, 1025), torch.arange(8193, 9217))).int().cuda()
    )
    selected = historical.repeat(queries, 1).contiguous()
    current = torch.tensor([20000, 20001, 20002], dtype=torch.int64, device="cuda")
    active = torch.tensor([queries], dtype=torch.int32, device="cuda")
    results = []
    for method in ("pb", "hisparse_shared", "hisparse_request"):
        source = create_host_tier_state(gpu_slots, host_slots, "cpu")
        source.epoch.fill_(1)
        hot = torch.arange(1, 8193)
        cold = torch.arange(8193, 30001)
        source.gpu_tags[: len(hot)] = hot.int()
        source.host_tags[: len(cold)] = cold.int()
        source.directory[hot] = torch.arange(len(hot))
        source.directory[cold] = -torch.arange(len(cold)) - 2
        source.access_time[torch.arange(1, 30001)] = 1
        source.free_counts[0] = gpu_slots - len(hot)
        source.free_counts[1] = host_slots - len(cold)
        source.gpu_free[: gpu_slots - len(hot)] = (
            torch.arange(len(hot), gpu_slots).flip(0).int()
        )
        source.host_free[: host_slots - len(cold)] = torch.arange(
            len(cold), host_slots
        ).int()
        state = type(source)(*(v.cuda() for v in source))
        workspace = create_host_tier_workspace(queries * (topk + 1), "cuda")
        ring = create_presence_ring(state, workspace)
        workspace = workspace._replace(histogram=ring.view)
        if method == "pb":
            rebuild_ring(state, ring)
            frame = torch.tensor(
                [
                    [v.data_ptr() for v in (*state, *workspace)]
                    + [
                        0,
                        0,
                        0,
                        0,
                        0,
                        gpu_slots,
                        host_slots,
                        queries * (topk + 1),
                        0,
                        1,
                        queries,
                        topk,
                        0,
                        0,
                    ]
                ],
                dtype=torch.int64,
                device="cuda",
            )
            expire_recent_groups(frame)
            prepare_pb(
                selected,
                current,
                active,
                0,
                state,
                workspace,
                config=PBLaunch(pin_blocks=512),
            )
        else:
            module = importlib.import_module(method)
            scratch = module.create_workspace(state, queries, topk)
            module.prepare(selected, current, active, 0, state, workspace, scratch)
        commit_host_tier(state, workspace)
        output = torch.empty_like(selected)
        remap_resident_rows(selected, active, state.directory, output)
        release_host_tier_leases(state, workspace)
        torch.cuda.synchronize()
        locations = state.directory[selected.long()]
        if (
            not torch.equal(output.long(), locations)
            or not bool((locations >= 0).all())
            or not torch.equal(state.gpu_tags[locations], selected)
            or not bool((state.directory[current] >= 0).all())
            or bool(state.leases.any())
        ):
            raise RuntimeError(("Invalid scheduler mapping", method))
        live = state.host_tags >= 0
        if not torch.equal(
            state.directory[state.host_tags[live].long()],
            -torch.arange(host_slots, device="cuda")[live] - 2,
        ):
            raise RuntimeError(("Invalid cold inverse mapping", method))
        results.append(
            dict(method=method, passed=True, promotions=int(workspace.counters[2]))
        )
    print(
        json.dumps(dict(passed=True, cases=results, GPU=torch.cuda.get_device_name(0)))
    )


if __name__ == "__main__":
    main()
