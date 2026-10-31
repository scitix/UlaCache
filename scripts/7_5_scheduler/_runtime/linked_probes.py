"""Matched-state single-step probes: plain one-thread linked LRU versus PB."""

from pathlib import Path
import argparse
import hashlib
import itertools
import json
import statistics
import sys


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    import torch
    import triton
    import triton.language as tl
    from sglang.kernels.jit.utils import load_jit
    from ulacache.compiler import configure_compiler
    from ulacache.cache.kernels.cpu_residency import (
        create_host_tier_state,
        create_host_tier_workspace,
        prepare_host_tier,
        commit_host_tier,
        release_host_tier_leases,
    )
    from ulacache.eviction.lru import create_presence_ring
    from ulacache.cache.kernels.local_cache import (
        register_unbound,
        expire_recent_groups,
        remap_resident_rows,
    )
    from ulacache.eviction.kernels.launch_config import PBLaunch, prepare as prepare_pb
    from ulacache.eviction.kernels.linked_list_lru import (
        create_workspace,
        prepare as prepare_linked,
        initialize_order,
    )

    configure_compiler(load_jit)
    torch.set_num_threads(2)
    out = Path(__file__).resolve().parent
    root = out.parent
    sys.path.insert(0, str(root / "_runtime"))
    from trace_data import load
    from seeding import rebuild_ring

    def save(n, d):
        (out / n).write_text(json.dumps(d, indent=2) + "\n")

    data = load(a.capture_root, 2, 128, 1048576, out)
    initial = data["initial"]
    original = data["original"]
    currents = data["currents"]
    G = data["G"]
    H = data["H"]
    W = data["W"]
    Q = a.requests * 3
    K = 2048
    groups = 21
    events = data["events"]
    bound = data["bound"]
    data_logical_slots = data["receipt"]["logical_slots"]
    del data
    original = original[:, :, :Q].contiguous()
    currents = currents[:, :Q].contiguous()
    W = max(Q * 2049, 1024)
    bound = Q
    from seeding import repack

    seed_receipts = repack(initial, a.capture_root, G, a.requests)
    save("PREFILL_SEED.json", dict(groups=seed_receipts, paired_seed_identical=True))
    assert (
        original.shape == (512, groups, Q, K)
        and G == 1048576
        and G + H == data_logical_slots
    )
    h = hashlib.sha256()
    for g, item in enumerate(initial["groups"]):
        for n, t in item["state"].items():
            h.update((str(g) + n + str(t.dtype) + str(tuple(t.shape))).encode())
            h.update(memoryview(t.numpy()).cast("B"))
    initial_sha = h.hexdigest()
    h = hashlib.sha256()
    h.update(memoryview(original.numpy()).cast("B"))
    h.update(memoryview(currents.numpy()).cast("B"))
    input_sha = h.hexdigest()
    trace = original.cuda()
    writers = currents.cuda()
    del original, currents
    selected = torch.empty((groups, Q, K), dtype=torch.int32, device="cuda")
    current = torch.empty(Q, dtype=torch.int64, device="cuda")
    active = torch.tensor([Q], dtype=torch.int32, device="cuda")
    config = PBLaunch(**json.loads((root / "PB_CONFIG.json").read_text())["selected"])
    names = ["linked_list", "pb_default", "pb_pin512"]
    allnames = ["reference", *names]
    arms = {}
    ring_fields = ["bits", "summary", "bucket_epochs", "gpu_bucket"]
    for name in allnames:
        states = [create_host_tier_state(G, H, "cuda") for _ in range(groups)]
        works = [create_host_tier_workspace(W, "cuda") for _ in states]
        rings = [create_presence_ring(s, w) for s, w in zip(states, works)]
        works = [w._replace(histogram=r.view) for w, r in zip(works, rings)]
        for g, (s, r) in enumerate(zip(states, rings)):
            for field, t in zip(s._fields, s):
                t.copy_(initial["groups"][g]["state"][field])
            if name != "linked_list":
                rebuild_ring(s, r)
        linked = (
            [create_workspace(s, Q, K) for s in states]
            if name == "linked_list"
            else None
        )
        frames = torch.tensor(
            [
                [t.data_ptr() for t in (*s, *w)]
                + [0, 0, 0, g, 0, G, H, W, 0, 1, bound, K, 0, 0]
                for g, (s, w) in enumerate(zip(states, works))
            ],
            dtype=torch.int64,
            device="cuda",
        )
        arms[name] = dict(
            states=states,
            works=works,
            rings=rings,
            linked=linked,
            frames=frames,
            output=torch.empty_like(selected),
        )

    # Retain the CPU canonical seed until compilation/capture finish.
    def iterate(name):
        arm = arms[name]
        if name != "linked_list":
            expire_recent_groups(arm["frames"])
        for g, (s, w) in enumerate(zip(arm["states"], arm["works"])):
            if name == "linked_list":
                prepare_linked(selected[g], current, active, 0, s, w, arm["linked"][g])
            elif name == "pb_default":
                prepare_host_tier(
                    selected[g],
                    current,
                    active,
                    0,
                    s,
                    w,
                    parallel=True,
                    parallel_victims=True,
                )
            else:
                prepare_pb(selected[g], current, active, 0, s, w, config=config)
            commit_host_tier(s, w)
            remap_resident_rows(selected[g], active, s.directory, arm["output"][g])
            release_host_tier_leases(s, w)

    @triton.jit
    def check(
        S,
        OUTPUT,
        F,
        C,
        E,
        WORDS: tl.constexpr,
        N: tl.constexpr,
        Q: tl.constexpr,
        B: tl.constexpr,
    ):
        g = tl.program_id(0)
        i = tl.program_id(1) * B + tl.arange(0, B)
        row = tl.load(S + g * N + i, i < N, other=-1)
        valid = (i < N) & (row > 0)
        slot = tl.load(OUTPUT + g * N + i, i < N, other=-1)
        dp = tl.load(F + g * WORDS).to(tl.pointer_type(tl.int64))
        tp = tl.load(F + g * WORDS + 1).to(tl.pointer_type(tl.int32))
        loc = tl.load(dp + row, valid, other=-1)
        actual = tl.load(tp + slot, valid & (slot >= 0), other=-1)
        tl.atomic_add(
            E,
            tl.sum(
                (valid & ((slot < 0) | (loc != slot) | (actual != row))).to(tl.int32)
            ),
        )
        if tl.program_id(1) == 0:
            j = tl.arange(0, 512)
            uid = tl.load(C + j, j < Q, other=-1)
            p = tl.load(dp + uid, j < Q, other=-1)
            v = tl.load(tp + p, (j < Q) & (p >= 0), other=-1)
            tl.atomic_add(E, tl.sum(((j < Q) & ((p < 0) | (v != uid))).to(tl.int32)))

    errors = torch.zeros(len(allnames), dtype=torch.int64, device="cuda")

    def validate(name, full=False):
        arm = arms[name]
        idx = allnames.index(name)
        check[(groups, triton.cdiv(Q * K, 256))](
            selected,
            arm["output"],
            arm["frames"],
            current,
            errors[idx : idx + 1],
            arm["frames"].shape[1],
            Q * K,
            Q,
            256,
        )
        if full:
            for s in arm["states"]:
                live = s.host_tags >= 0
                hot = s.gpu_tags >= 0
                assert torch.equal(
                    s.directory[s.gpu_tags[hot].long()],
                    torch.arange(G, device="cuda")[hot],
                )
                assert torch.equal(
                    s.directory[s.host_tags[live].long()],
                    -torch.arange(H, device="cuda")[live] - 2,
                )
                assert not bool(s.leases.any())
                free = s.gpu_free[: int(s.free_counts[0])]
                assert (
                    int((~hot).sum()) == free.numel()
                    and free.unique().numel() == free.numel()
                    and bool((s.gpu_tags[free.long()] < 0).all())
                )
            if name == "linked_list":
                slots = torch.arange(G, device="cuda")
                for x in arm["linked"]:
                    head, tail = map(int, x.ends)
                    assert int(x.previous[head]) == -1 and int(x.following[tail]) == -1
                    right = x.following >= 0
                    left = x.previous >= 0
                    assert torch.equal(
                        x.previous[x.following[right].long()], slots[right].int()
                    )
                    assert torch.equal(
                        x.following[x.previous[left].long()], slots[left].int()
                    )
        torch.cuda.synchronize()
        assert not bool(errors.any()), (name, errors.cpu().tolist())

    def bind(f):
        selected.copy_(trace[f])
        current.copy_(writers[f])

    def restore(name, link_seed=None, verify=False):
        for s, source in zip(arms[name]["states"], arms["reference"]["states"]):
            for t, v in zip(s, source):
                t.copy_(v)
            if verify:
                assert all(torch.equal(t, v) for t, v in zip(s, source))
        if name != "linked_list":
            for r, source in zip(arms[name]["rings"], arms["reference"]["rings"]):
                for field in ring_fields:
                    getattr(r, field).copy_(getattr(source, field))
        elif link_seed:
            for x, seed in zip(arms[name]["linked"], link_seed):
                for t, v in zip(x[4:7], seed):
                    t.copy_(v)

    lifecycle = {
        f: torch.tensor([v[2] for v in rows], dtype=torch.int32, device="cuda")
        for f, rows in itertools.groupby(events, key=lambda v: v[0])
    }
    bind(0)
    for name in allnames:
        iterate(name)
        validate(name)
    # Restore the original canonical reference after module warmup.
    seed = initial
    for g, (s, r) in enumerate(
        zip(arms["reference"]["states"], arms["reference"]["rings"])
    ):
        for field, t in zip(s._fields, s):
            t.copy_(seed["groups"][g]["state"][field])
        rebuild_ring(s, r)
    del seed
    for name in names:
        restore(name)
    for s, x in zip(arms["linked_list"]["states"], arms["linked_list"]["linked"]):
        initialize_order(s, x)
    graphs = {}
    for name in allnames:
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            iterate(name)
    # Capture executes once; reload the reference only, outside any measurement.
    seed = initial
    for g, (s, r) in enumerate(
        zip(arms["reference"]["states"], arms["reference"]["rings"])
    ):
        for field, t in zip(s._fields, s):
            t.copy_(seed["groups"][g]["state"][field])
        rebuild_ring(s, r)
    del seed, initial
    probes = []
    errors.zero_()
    save(
        "READY.json",
        dict(
            requests=a.requests,
            methods=names,
            frames=[256, 384, 511],
            repeats=a.repeats,
            initial_sha256=initial_sha,
            input_sha256=input_sha,
        ),
    )
    for f in range(512):
        if f - 1 in lifecycle:
            register_unbound(
                lifecycle[f - 1], arms["reference"]["frames"], prefer_gpu=False
            )
        bind(f)
        if f in [256, 384, 511]:
            ref = arms["reference"]
            h = hashlib.sha256()
            for g, s in enumerate(ref["states"]):
                for field, t in zip(s._fields, s):
                    h.update((str(g) + field).encode())
                    h.update(memoryview(t.cpu().numpy()).cast("B"))
            state_sha = h.hexdigest()
            for s, x in zip(ref["states"], arms["linked_list"]["linked"]):
                initialize_order(s, x)
            link_seed = [
                [t.clone() for t in x[4:7]] for x in arms["linked_list"]["linked"]
            ]
            samples = []
            if a.profile and f == 511:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
            for rep in range(a.repeats):
                for name in names:
                    restore(name, link_seed, verify=rep == 0)
                order = names[rep % 3 :] + names[: rep % 3]
                for name in order:
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                    validate(name, full=True)
                    plans = sum(int(w.counters[2]) for w in arms[name]["works"])
                    samples.append(
                        dict(repetition=rep, method=name, ms=ms, plan_count=plans)
                    )
            if a.profile and f == 511:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            probes.append(
                dict(
                    frame=f,
                    pre_state_sha256=state_sha,
                    State9_equal_before_each_method=True,
                    samples=samples,
                )
            )
            save("PROBES.json", probes)
        graphs["reference"].replay()
        validate("reference")
        if (f + 1) % 32 == 0:
            save(
                "PROGRESS.json",
                dict(reference_frames=f + 1, probes_complete=len(probes)),
            )
    means = {
        name: [
            statistics.mean(
                s["ms"]
                for p in probes
                for s in p["samples"]
                if s["method"] == name and s["repetition"] == rep
            )
            for rep in range(a.repeats)
        ]
        for name in names
    }
    result = dict(
        complete=True,
        passed=True,
        requests=a.requests,
        queries=Q,
        groups=groups,
        gpu_slots=G,
        cpu_slots=H,
        probe_frames=[256, 384, 511],
        repeats=a.repeats,
        methods=names,
        median_ms={n: statistics.median(v) for n, v in means.items()},
        repetition_means=means,
        probes=probes,
        source_sha256=digest(root / "SOURCE.json"),
        harness_sha256=digest(__file__),
        initial_sha256=initial_sha,
        input_sha256=input_sha,
        profiled=a.profile,
        LRU_grid=[1, 1, 1],
        LRU_block=[1, 1, 1],
        scope="Three matched-State9 single-step probes drawn from a PB reference trajectory over the retained MTP2 trace composition. Three resets per probe; not a full linked-LRU policy replay or hit-rate experiment. All21groups; includes GPU dedup, one-thread list maintenance/plan, common commit/remap/release. KV payload, input copies, model, DP communication and setup excluded.",
    )
    save("COMPLETE.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "probes"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument(
        "--requests", type=int, choices=[1, 2, 4, 8, 16, 32, 64, 128], required=True
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    main(p.parse_args())
