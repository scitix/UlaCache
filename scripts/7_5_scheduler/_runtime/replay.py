"""H200 low/high-BS metadata replay, separate hit observers and list probes."""

from pathlib import Path
import argparse
import hashlib
import itertools
import json
import statistics
import sys


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(a):
    import torch
    import triton
    import numpy as np
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

    configure_compiler(load_jit)
    torch.set_num_threads(2)
    out = Path(__file__).resolve().parent
    root = out.parent
    sys.path.insert(0, str(root / "_runtime"))
    from trace_data import load
    from seeding import rebuild_ring

    def save(n, d):
        (out / n).write_text(json.dumps(d, indent=2) + "\n")

    data = load(a.capture_root, 2, 128, a.gpu_slots, out)
    initial = data["initial"]
    original = data["original"]
    currents = data["currents"]
    events = data["events"]
    G = data["G"]
    H = data["H"]
    W = data["W"]
    bound = data["bound"]
    data_logical_slots = data["receipt"]["logical_slots"]
    del data
    original = original[:, :, : a.requests * 3].contiguous()
    currents = currents[:, : a.requests * 3].contiguous()
    W = max(a.requests * 3 * 2049, 1024)
    bound = a.requests * 3
    from seeding import repack
    import copy

    request_initial = copy.deepcopy(initial)
    request_receipts = repack(
        request_initial, a.capture_root, G, a.requests, per_request=True
    )
    seed_receipts = repack(initial, a.capture_root, G, a.requests)
    save(
        "PREFILL_SEED.json",
        dict(
            shared=seed_receipts,
            per_request=request_receipts,
            shared_policies_identical=True,
            slots_per_request=8192,
        ),
    )
    from reference import replay_per_request

    expected = replay_per_request(
        request_initial, original, currents, out, capture=a.capture_root
    )
    N, groups, Q, K = original.shape
    assert N == 512 and groups == 21 and Q == a.requests * 3 and K == 2048
    assert G == 1048576 and G + H == data_logical_slots
    from reference import replay as replay_cpu

    reference_counts = replay_cpu(initial, original, currents, out)
    trace = original
    trace_current = currents
    del original, currents
    selected = torch.empty((groups, Q, K), dtype=torch.int32, device="cuda")
    current = torch.empty(Q, dtype=torch.int64, device="cuda")
    active = torch.tensor([Q], dtype=torch.int32, device="cuda")
    config = PBLaunch(**json.loads((root / "PB_CONFIG.json").read_text())["selected"])
    methods = ["pb_pin_only", "hisparse_request", "hisparse"]
    arms = {}
    ring_fields = ["bits", "summary", "bucket_epochs", "gpu_bucket"]
    for name in methods:
        states = [create_host_tier_state(G, H, "cuda") for _ in range(groups)]
        works = [create_host_tier_workspace(W, "cuda") for _ in states]
        rings = [
            create_presence_ring(s, w, total_buckets=64 if name == "pb64" else 256)
            for s, w in zip(states, works)
        ]
        works = [w._replace(histogram=r.view) for w, r in zip(works, rings)]
        for g, (s, r) in enumerate(zip(states, rings)):
            source = request_initial if name.startswith("hisparse_request") else initial
            for field, t in zip(s._fields, s):
                t.copy_(source["groups"][g]["state"][field])
            if name.startswith("pb"):
                rebuild_ring(s, r)
        native = None
        prepare = None
        if name.startswith("hisparse_request"):
            from hisparse_request import create_workspace, prepare

            native = [
                create_workspace(
                    s,
                    Q,
                    K,
                    dedup_backend="triton" if name.endswith("triton") else "cub",
                )
                for s in states
            ]
        elif name == "hisparse":
            from hisparse_shared import create_workspace, prepare

            native = [create_workspace(s, Q, K) for s in states]
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
            native=native,
            prepare=prepare,
            frames=frames,
            output=torch.empty_like(selected),
        )
    del initial, request_initial

    def native_state(name, x):
        return [x.lru_slots]

    def snapshot(name, arm):
        return dict(
            states=[[t.clone() for t in s] for s in arm["states"]],
            rings=[[getattr(r, f).clone() for f in ring_fields] for r in arm["rings"]]
            if not arm["native"]
            else None,
            native=[[t.clone() for t in native_state(name, x)] for x in arm["native"]]
            if arm["native"]
            else None,
        )

    seeds = {name: snapshot(name, arm) for name, arm in arms.items()}

    def restore(name):
        arm = arms[name]
        seed = seeds[name]
        for s, old in zip(arm["states"], seed["states"]):
            for t, v in zip(s, old):
                t.copy_(v)
        if seed["rings"]:
            for r, old in zip(arm["rings"], seed["rings"]):
                for f, v in zip(ring_fields, old):
                    getattr(r, f).copy_(v)
        if seed["native"]:
            for x, old in zip(arm["native"], seed["native"]):
                for t, v in zip(native_state(name, x), old):
                    t.copy_(v)

    lifecycle = {
        f: torch.tensor([v[2] for v in rows], dtype=torch.int32, device="cuda")
        for f, rows in itertools.groupby(events, key=lambda x: x[0])
    }

    def iterate(name):
        arm = arms[name]
        if not arm["native"]:
            expire_recent_groups(arm["frames"])
        for g, (s, w) in enumerate(zip(arm["states"], arm["works"])):
            if arm["native"]:
                arm["prepare"](selected[g], current, active, 0, s, w, arm["native"][g])
            elif name in ("pb_pin_only", "pb64"):
                prepare_pb(selected[g], current, active, 0, s, w, config=config)
            else:
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

    errors = torch.zeros(len(methods), dtype=torch.int64, device="cuda")

    def validate(name):
        arm = arms[name]
        i = methods.index(name)
        check[(groups, triton.cdiv(Q * K, 256))](
            selected,
            arm["output"],
            arm["frames"],
            current,
            errors[i : i + 1],
            arm["frames"].shape[1],
            Q * K,
            Q,
            256,
        )

    selected.copy_(trace[0])
    current.copy_(trace_current[0])
    for name in methods:
        iterate(name)
        validate(name)
    torch.cuda.synchronize()
    assert not bool(errors.any()), ("warmup_residency", errors.cpu().tolist())
    graphs = {name: torch.cuda.CUDAGraph() for name in methods}
    for name in methods:
        restore(name)
        with torch.cuda.graph(graphs[name]):
            iterate(name)
    torch.cuda.synchronize()
    save(
        "READY.json",
        dict(
            methods=methods,
            requests=a.requests,
            queries=Q,
            topk=K,
            gpu_slots=G,
            cpu_slots=H,
            groups=groups,
            shared_policy_seed_identical=True,
            partitioned_seed_uses_own_capacity=True,
            pb_config=config._asdict(),
        ),
    )
    repetitions = []
    promotions = {name: [] for name in methods}
    for rep in range(a.repeats):
        for name in methods:
            restore(name)
        errors.zero_()
        pairs = {name: [] for name in methods}
        for f in range(512):
            if f - 1 in lifecycle:
                for arm in arms.values():
                    register_unbound(lifecycle[f - 1], arm["frames"], prefer_gpu=False)
            selected.copy_(trace[f])
            current.copy_(trace_current[f])
            for name in (
                methods[(f + rep) % len(methods) :]
                + methods[: (f + rep) % len(methods)]
            ):
                x, y = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                x.record()
                graphs[name].replay()
                y.record()
                if f >= 256:
                    pairs[name].append((x, y))
                validate(name)
            if rep == 0:
                for backend in ("hisparse_request",):
                    actual = np.array(
                        [int(w.counters[2]) for w in arms[backend]["works"]],
                        dtype=np.int64,
                    )
                    wanted = expected[f, :, 8] - expected[f, :, 9]
                    assert np.array_equal(actual, wanted), (
                        "CPU_policy_plan_mismatch",
                        backend,
                        a.requests,
                        f,
                        actual.tolist(),
                        wanted.tolist(),
                    )
                actual = np.array(
                    [int(w.counters[2]) for w in arms["hisparse"]["works"]],
                    dtype=np.int64,
                )
                assert np.array_equal(
                    actual, reference_counts[1, f, :, 8] - reference_counts[1, f, :, 9]
                ), ("shared_CPU_policy_plan_mismatch", a.requests, f)
            if rep == 0 and f >= 256:
                for name, arm in arms.items():
                    promotions[name].append(
                        sum(int(w.counters[2]) for w in arm["works"])
                    )
            if (f + 1) % 32 == 0:
                torch.cuda.synchronize()
                assert not bool(errors.any()), (
                    "residency",
                    rep,
                    f,
                    errors.cpu().tolist(),
                )
                save("PROGRESS.json", dict(repetition=rep, frames=f + 1))
        torch.cuda.synchronize()
        for name, arm in arms.items():
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
                assert int((~hot).sum()) == int(s.free_counts[0])
                free = s.gpu_free[: int(s.free_counts[0])]
                assert free.unique().numel() == free.numel() and bool(
                    (s.gpu_tags[free.long()] < 0).all()
                )
                assert not bool(s.leases.any())
        raw = {name: [x.elapsed_time(y) for x, y in p] for name, p in pairs.items()}
        repetitions.append(
            dict(
                repetition=rep,
                mean_ms={name: statistics.mean(v) for name, v in raw.items()},
                raw_ms=raw,
            )
        )
        save("REPETITIONS.json", repetitions)
    native_final_digests = {}
    for name in ("hisparse_request",):
        tags = hashlib.sha256()
        lru = hashlib.sha256()
        for s, x in zip(arms[name]["states"], arms[name]["native"]):
            tags.update(s.gpu_tags.cpu().numpy().tobytes())
            lru.update(x.lru_slots.cpu().numpy().tobytes())
        native_final_digests[name] = dict(tags=tags.hexdigest(), lru=lru.hexdigest())
    save(
        "NATIVE_POLICY_VALIDATION.json",
        dict(
            passed=True, final=native_final_digests, all512x21plan_counts_match_CPU=True
        ),
    )
    # A separate fourth pass observes residency before update. It is excluded
    # from all clean CUDA-event measurements above.
    from ulacache.eviction.kernels.trace_hits import (
        create_workspace as hit_workspace,
        observe,
        COLUMNS,
    )

    hit_methods = ["pb_pin_only", "hisparse_request", "hisparse"]
    hit_graphs = {}
    histories = {}
    hit_counts = {}
    hit_work = {}
    for name in hit_methods:
        hit_counts[name] = torch.empty(
            (groups, len(COLUMNS)), dtype=torch.int64, device="cuda"
        )
        hit_work[name] = [
            hit_workspace(s, Q, K, stats=hit_counts[name][g])
            for g, s in enumerate(arms[name]["states"])
        ]
        histories[name] = torch.empty(
            (512, groups, len(COLUMNS)), dtype=torch.int64, device="cuda"
        )
        hit_graphs[name] = torch.cuda.CUDAGraph()
        restore(name)
        with torch.cuda.graph(hit_graphs[name]):
            for g, s in enumerate(arms[name]["states"]):
                observe(selected[g], current, active, s, hit_work[name][g])
            iterate(name)
        restore(name)
    errors.zero_()
    for f in range(512):
        if f - 1 in lifecycle:
            for name in hit_methods:
                register_unbound(
                    lifecycle[f - 1], arms[name]["frames"], prefer_gpu=False
                )
        selected.copy_(trace[f])
        current.copy_(trace_current[f])
        for name in hit_methods:
            hit_graphs[name].replay()
            histories[name][f].copy_(hit_counts[name])
            validate(name)
            actual = torch.stack([w.counters[2] for w in arms[name]["works"]]).long()
            assert torch.equal(
                actual, hit_counts[name][:, 8] - hit_counts[name][:, 9]
            ), ("observer_plan", name, f)
        if (f + 1) % 32 == 0:
            torch.cuda.synchronize()
            assert not bool(errors.any())
            save("PROGRESS.json", dict(phase="separate_hit_observer", frames=f + 1))
    arrays = {name: value.cpu().numpy() for name, value in histories.items()}
    for name, value in arrays.items():
        assert not value[:, :, 10].any()
        target = (
            expected
            if name == "hisparse_request"
            else reference_counts[1]
            if name == "hisparse"
            else None
        )
        if target is not None:
            assert np.array_equal(value, target), ("all_CPU_hit_counters", name)
    np.savez_compressed(out / "GPU_HIT_COUNTS.npz", **arrays)
    hit_rows = []
    for name, value in arrays.items():
        total = value[256:].sum(axis=(0, 1), dtype=np.int64)
        hit_rows.append(
            dict(
                method=name,
                total_counts=dict(zip(COLUMNS, map(int, total))),
                unique_historical_hit_rate=float(total[5] / total[4]),
            )
        )
    save(
        "HIT_RATES.json",
        dict(
            passed=True,
            complete=True,
            requests=a.requests,
            columns=COLUMNS,
            rows=hit_rows,
            separate_from_clean_timing=True,
            all512x21_CPU_hisparse_counters_match=True,
            seed_family="active_history_prefill_v1",
            frames=512,
            warmup=256,
        ),
    )
    d = dict(
        passed=True,
        complete=True,
        mtp=2,
        requests=a.requests,
        queries=Q,
        gpu_slots=G,
        cpu_slots=H,
        groups=groups,
        topk=K,
        frames=512,
        warmup=256,
        repeats=a.repeats,
        backend=a.backend,
        methods=methods,
        median_ms={
            name: statistics.median(r["mean_ms"][name] for r in repetitions)
            for name in methods
        },
        promotions=promotions,
        source_sha256=digest(root / "SOURCE.json"),
        harness_sha256=digest(__file__),
        composition_sha256=digest(root / "_runtime/trace_data.py"),
        timing="GPU dedup, native policy, exclusive-cold plan, common commit/remap/release across21groups. Native registration, copies, allocation, validation, KV payload and model/DP execution excluded.",
        trace="Real native Attention-DP8MTP2 traces; use first2/4/8rank inputs from the full8rank namespace. Both arms start from identical128x8192request-prefix residency; inactive request partitions cannot lend slots to active requests in HiSparse. PB can use the complete1Mshared pool.256real warmup frames. Not native high-batch model captures.",
        paired_seed_identical=True,
        pb_total_buckets=64 if a.backend == "pb64" else 256,
        seed_family="128request_prefix8192",
        hisparse_request_cpu_plan_counts_match_all_512x21=True,
        fixed_max_requests=128,
        hot_slots_per_request=8192,
        trace_staged_on_CPU=True,
        native_device_body_unmodified=True,
        hisparse_dedup="CUB segmented radix sort and block-scan unique",
    )
    d.update(
        hardware="NVIDIA H200",
        request_loads_supported=[1, 2, 4, 8, 16, 32, 64, 128],
        trace="First active requests from the retained real DP8 MTP2 trace; not fresh per-BS captures. Shared pools preload active history up to equal per-request shares; free slots remain empty. Per-request HiSparse preloads the latest8192UIDs per active request.",
        shared_hisparse_CPU_plan_counts_match_all_512x21=True,
        GPU_hit_observers_separate_from_timing=True,
        native_CPU_hit_replay="CPU_COUNTS.npz; identical shared preload, independent512-frame trajectory",
        paired_seed_identical=False,
        shared_policy_seed_identical=True,
        seed_family="active_history_prefill_v1",
        prefill=seed_receipts[0],
        per_request_prefill=request_receipts[0],
    )
    if seed_receipts[0]["fully_preloaded"]:
        for name in ("pb_pin_only", "hisparse"):
            assert np.array_equal(arrays[name][:, :, 4], arrays[name][:, :, 5]), (
                "fully_preloaded_history_miss",
                name,
            )
        assert np.array_equal(
            reference_counts[0, :, :, 4], reference_counts[0, :, :, 5]
        )
    save("COMPLETE.json", d)
    print(json.dumps({k: v for k, v in d.items() if k != "promotions"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument(
        "--requests", type=int, choices=[1, 2, 4, 8, 16, 32, 64, 128], required=True
    )
    p.add_argument("--backend", choices=["hisparse_request"], required=True)
    p.add_argument("--gpu-slots", type=int, default=1048576)
    p.add_argument("--repeats", type=int, default=3)
    main(p.parse_args())
