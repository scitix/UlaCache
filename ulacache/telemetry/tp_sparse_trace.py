"""Passive TP local-miss capture alongside the original global sparse trace."""


class GpuTpMissTrace:
    """Use the address recorder's existing frame control and chunk geometry.

    Keep the actual captured source bucket row count separate from the valid
    query prefix. Neither count changes the native planner's launch geometry.
    """

    def __init__(self, recorder, *, max_gpu_bytes=3_000_000_000):
        import torch
        from ulacache.compiler import load_jit

        g = recorder.geometry
        size = g.chunk_frames * g.groups * g.query_rows * (g.topk + 1) * 4
        size += g.chunk_frames * g.groups * 4
        if size > max_gpu_bytes:
            raise ValueError("TP miss trace exceeds the explicit diagnostic allowance")
        self.recorder = recorder
        self.rows = torch.empty_like(recorder.selected)
        self.counts = torch.empty_like(recorder.current)
        self.source_rows = torch.empty_like(recorder.seen)
        self.module = load_jit(
            "udcache_tp_miss_trace_v1",
            cuda_files=["csrc/telemetry/tp_sparse_trace.cuh"],
            cuda_wrappers=[("record", "udcache::telemetry::record_tp_miss_trace")],
        )

    @property
    def gpu_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.rows, self.counts, self.source_rows)
        )

    def observe(self, group, rows, counts, active):
        g = self.recorder.geometry
        if type(group) is not int or not 0 <= group < g.groups:
            raise ValueError("Invalid TP management group")
        self.module.record(
            rows,
            counts,
            active,
            self.recorder.control,
            self.rows,
            self.counts,
            self.source_rows,
            group,
            g.frames,
            g.chunk_frames,
            g.groups,
            g.query_rows,
            g.topk,
        )

    def snapshot_chunk(self):
        """Call at the recorder's synchronizing chunk-drain boundary."""
        return dict(
            local_miss_rows=self.rows.cpu(),
            local_miss_counts=self.counts.cpu(),
            native_source_bucket_rows=self.source_rows.cpu(),
        )
