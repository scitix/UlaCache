"""Original asynchronous complete-record MLA/Index host exchange."""

from __future__ import annotations


class NativeExchangeBoundaryExecutor:
    """Original exchange on one boundary stream and fixed plan buffers.

    Plan row2 is H2D, row3 D2H; equal nonnegative row2/3 is the original swap.
    MLA expands page64 to token records; Index copies its original packed page.
    The same buffers are reused in stream order. This is not a decode task.
    """

    def __init__(
        self,
        *,
        mla_cpu,
        mla_gpu,
        index_cpu,
        device,
        capacity,
        readers_retired,
        plan=None,
        counters=None,
    ):
        import torch

        self.mla_cpu, self.mla_gpu, self.index_cpu = mla_cpu, mla_gpu, index_cpu
        self.index_gpu = None
        self.readers_retired = readers_retired
        self.capacity = capacity
        self.stream = torch.cuda.Stream(device=device)
        self.plan = (
            plan
            if plan is not None
            else torch.empty((5, capacity), dtype=torch.int32, device=device)
        )
        self.counters = (
            counters
            if counters is not None
            else torch.zeros(8, dtype=torch.int32, device=device)
        )
        # One reusable pinned metadata plane. Build complete token plans with
        # vector arithmetic from page IDs, rather than Python token objects.
        self.host_plan = torch.empty(
            (5, capacity), dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.page_offsets = torch.arange(64, dtype=torch.int32, device="cpu")

    def bind_index(self, storage):
        self.index_gpu = storage

    def submit(self, pool, tx):
        if not self.readers_retired():
            raise RuntimeError(
                "original all-peer reader retirement must precede boundary source reuse"
            )
        if self.index_gpu is None:
            raise NotImplementedError(
                "request transfer requires the real packed physical GPU Index plane"
            )
        import torch
        from ulacache.cache.kernels.cpu_residency import _host_module

        buffers = {
            "mla": (self.mla_cpu, self.mla_gpu),
            "index": (self.index_cpu, self.index_gpu),
        }
        shapes = pool.plane_shapes()
        for name, (host, gpu) in buffers.items():
            if (
                tuple(host.shape) != shapes[name + "_cpu"]
                or tuple(gpu.shape) != shapes[name + "_gpu"]
                or host.dtype != torch.uint8
                or gpu.dtype != torch.uint8
                or not host.is_pinned()
                or not gpu.is_cuda
                or not host.is_contiguous()
                or not gpu.is_contiguous()
            ):
                raise ValueError(
                    "request transfer requires exact contiguous registered byte planes"
                )
        if self.mla_gpu.device != self.index_gpu.device:
            raise ValueError("both request planes must be on the same owner device")
        module = _host_module()
        self.stream.wait_stream(torch.cuda.current_stream(self.mla_gpu.device))
        # Fixed metadata scratch is shared by the two planes. CPU staging is
        # copied synchronously before reuse; original exchange tasks stay on
        # the same IO stream and complete before later plan copies consume it.
        with torch.cuda.stream(self.stream):
            for wave in tx.waves:
                for name, (host, gpu) in buffers.items():
                    records = pool.page_size if name == "mla" else 1
                    pages_per_chunk = self.capacity // records
                    if not pages_per_chunk:
                        raise ValueError(
                            "fixed exchange scratch cannot fit one native page"
                        )
                    for first in range(0, len(wave), pages_per_chunk):
                        pages = wave[first : first + pages_per_chunk]
                        count = len(pages) * records
                        page_ids = torch.tensor(
                            [
                                (
                                    p.gpu_page,
                                    p.cpu_page,
                                    p.kind != "offload",
                                    p.kind != "onload",
                                )
                                for p in pages
                            ],
                            dtype=torch.int32,
                            device="cpu",
                        )
                        cpu_plan = self.host_plan.view(-1)[: 5 * count].view(5, count)
                        cpu_plan.fill_(-1)
                        offsets = self.page_offsets[:records]
                        gpu_rows = page_ids[:, 0:1] * records + offsets
                        cpu_rows = page_ids[:, 1:2] * records + offsets
                        cpu_plan[1].copy_(gpu_rows.flatten())
                        cpu_plan[2].copy_(
                            torch.where(page_ids[:, 2:3].bool(), cpu_rows, -1).flatten()
                        )
                        cpu_plan[3].copy_(
                            torch.where(page_ids[:, 3:4].bool(), cpu_rows, -1).flatten()
                        )
                        packed = self.plan.view(-1)[: 5 * count].view(5, count)
                        packed.copy_(cpu_plan, non_blocking=False)
                        self.counters[2:3].fill_(count)
                        module.exchange(
                            host,
                            gpu,
                            packed,
                            self.counters,
                            host.size(-1),
                            host.stride(-2),
                            gpu.stride(-2),
                            host.size(0),
                            host.stride(0),
                            gpu.stride(0),
                        )
            event = torch.cuda.Event()
            event.record(self.stream)
        return BoundaryCompletion(tx.transaction_id, event, self.stream)

    def close(self):
        self.stream.synchronize()
        del self.plan, self.counters, self.host_plan, self.page_offsets, self.index_gpu


class BoundaryCompletion:
    def __init__(self, transaction_id, event, stream):
        self.transaction_id, self.event, self.stream = transaction_id, event, stream

    def wait(self):
        self.event.synchronize()
        return self

    def wait_on(self, stream):
        stream.wait_event(self.event)

    def assert_complete(self):
        if not self.event.query():
            raise RuntimeError(
                "all MLA and Index boundary transfers have not completed"
            )
