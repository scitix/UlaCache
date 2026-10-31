"""Passive GPU replay probes with request/forward/management-group deduplication."""


class ReplayMetrics:
    columns = (
        "lookup_tokens",
        "pool_hit_tokens",
        "cpu_miss_tokens",
        "unallocated_tokens",
        "errors",
        "history_h2d_bytes",
        "victim_d2h_bytes",
        "new_cpu_backup_bytes",
    )

    def __init__(self, groups, queries, topk, *, device):
        import torch
        from ulacache.compiler import load_jit

        if min(groups, queries, topk) < 1:
            raise ValueError(
                "Replay probes need a positive active native query geometry"
            )
        self.counts = torch.zeros((groups, 8), dtype=torch.int64, device=device)
        size = 1 << (2 * queries * topk - 1).bit_length()
        self.keys = torch.full((size,), -1, dtype=torch.int64, device=device)
        self.module = load_jit(
            "udcache_sparse_replay_metrics_v1",
            cuda_files=["csrc/telemetry/sparse_replay.cuh"],
            cuda_wrappers=[
                ("hits", "udcache::telemetry::sparse_replay_count_hits"),
                ("plan", "udcache::telemetry::sparse_replay_count_plan"),
                ("current", "udcache::telemetry::sparse_replay_count_current"),
            ],
        )

    def before_plan(self, group, selected, current, requests, directory, width):
        self.keys.fill_(-1)
        self.module.hits(
            selected, current, requests, directory, self.keys, self.counts[group], width
        )

    def after_group(self, group, work, current, directory, *, layers, row_bytes, bound):
        self.module.plan(
            work.plan, work.counters, self.counts[group], layers, row_bytes, bound
        )
        self.module.current(current, directory, self.counts[group], layers, row_bytes)

    def snapshot(self):
        rows = self.counts.cpu().tolist()
        for row in rows:
            if row[3] or row[4] or row[0] != row[1] + row[2]:
                raise RuntimeError(
                    "Replay referenced unallocated history or failed lookup partition/bounds"
                )
        return dict(
            columns=self.columns,
            groups=rows,
            totals={
                key: sum(row[i] for row in rows) for i, key in enumerate(self.columns)
            },
            local_hit_tokens=0,
            scope="Unique valid historical native IDs per request/group/target forward before promotion. IO bytes cover every member layer.",
        )
