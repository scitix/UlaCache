"""Passive baseline probes bound to an explicitly supplied native spec header."""

import hashlib
from pathlib import Path

from .replay_metrics import ReplayMetrics


class HiSparseMetrics:
    columns = (*ReplayMetrics.columns[:6], "d2h_bytes", "new_cpu_backup_bytes")

    def __init__(self, groups, max_requests, width, topk, *, native_csrc, device):
        import torch
        from ulacache.compiler import load_jit

        if min(groups, max_requests, width, topk) < 1:
            raise ValueError("Native baseline probe geometry must be positive")
        native_csrc = Path(native_csrc).resolve()
        self.header = native_csrc / "kvcacheio/hisparse_spec.cuh"
        self.header_sha256 = hashlib.sha256(self.header.read_bytes()).hexdigest()
        self.counts = torch.zeros((groups, 8), dtype=torch.int64, device=device)
        size = 1 << (2 * max_requests * width * topk - 1).bit_length()
        self.keys = torch.full((size,), -1, dtype=torch.int64, device=device)
        self.flags = torch.zeros((size,), dtype=torch.int32, device=device)
        self.positions = torch.full(
            (max_requests * width,), -1, dtype=torch.int64, device=device
        )
        self.enabled = torch.zeros(1, dtype=torch.int32, device=device)
        self.module = load_jit(
            "udcache_hisparse_metrics_v1",
            self.header_sha256,
            cuda_files=["csrc/telemetry/hisparse_metrics.cuh"],
            extra_include_paths=[str(native_csrc)],
            cuda_wrappers=[
                ("observe", "udcache::telemetry::hisparse_metrics_observe"),
                ("miss_bytes", "udcache::telemetry::hisparse_metrics_miss_bytes"),
                ("backup", "udcache::telemetry::hisparse_metrics_backup"),
            ],
        )

    def before_swap(
        self,
        group,
        *,
        selected,
        requests,
        seq_lens,
        tokens,
        locs,
        state,
        host_locs,
        real_requests,
        hot,
        page_size,
    ):
        self.keys.fill_(-1)
        self.flags.zero_()
        self.module.observe(
            selected,
            requests,
            seq_lens,
            self.positions,
            tokens,
            locs,
            state.cache_index,
            state.scratch_state,
            host_locs,
            real_requests,
            self.enabled,
            self.keys,
            self.flags,
            self.counts[group],
            hot,
            page_size,
        )

    def after_swap(self, group, misses, real_requests, *, layers, row_bytes):
        self.module.miss_bytes(
            misses, real_requests, self.enabled, self.counts[group], layers, row_bytes
        )

    def backup(self, *, rows, layers, row_bytes):
        # Native fixed-width commit includes rejected reserved host padding.
        self.module.backup(self.enabled, self.counts[0], rows * layers * row_bytes)

    def snapshot(self):
        rows = self.counts.cpu().tolist()
        for row in rows:
            if row[3] or row[4] or row[0] != row[1] + row[2]:
                raise RuntimeError("Native baseline historical lookup partition failed")
        return dict(
            columns=self.columns,
            groups=rows,
            totals={
                key: sum(row[i] for row in rows) for i, key in enumerate(self.columns)
            },
            local_hit_tokens=0,
            native_header=str(self.header),
            native_header_sha256=self.header_sha256,
            scope="Unique valid historical addresses per request/group/target before promotion; MTP queries unioned. GPU hit if any valid occurrence can use the original native hot/extra-page lookup. IO uses real shared miss plans and fixed-width host backups, including rejected padding.",
        )
