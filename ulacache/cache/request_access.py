# SPDX-License-Identifier: Apache-2.0
"""Request-boundary glue for the original native reader retirement."""


class RequestAddressRuntime:
    def __init__(self, pool):
        self.pool = pool

    def validate(self, tier):
        import torch

        pool = self.pool
        if (
            tier.pool is not pool
            or pool.flowkv_host is not None
            or tier.page_map.dtype != torch.int32
            or tier.page_map.ndim != 1
            or not tier.page_map.is_contiguous()
            or tier.page_map.device != pool.flowkv_shards.device
            or tier.page_map.numel() * 64 != tier.logical_owner_rows
            or tier.physical_gpu_rows % 64
            or tuple(pool.flowkv_shards.shape)[1] != tier.physical_gpu_rows
            or not hasattr(pool, "flowkv_step_publication")
            or not hasattr(pool.flowkv_arena._module, "retire_request_pages")
        ):
            raise ValueError(
                "Request runtime requires the local mapped native physical pool"
            )

    def begin_prefill(self):
        # The original begin accepts zero queries. It advances the same epoch
        # and retires the preceding forward; a prefill chunk's much larger
        # output locations never enter decode's fixed current-query table.
        pool = self.pool
        pool.flowkv_step_publication.begin(
            pool._flowkv_empty_rows,
            pool.flowkv_arena.valid,
        )

    def finish_prefill(self):
        # Called by the existing forward context after the model has queued
        # all prefill reads. Publication runs on that stream, using the same
        # release flag as decode. No extra per-layer barrier is introduced.
        self.pool.flowkv_step_publication.publish()

    def readers_retired(self, tier):
        # This runs only at a native admission/release boundary. Both prefill
        # and decode now publish their epoch after their readers. Preserve
        # that original all-peer acquire rule without advancing the epoch.
        import torch

        self.validate(tier)
        pool = self.pool
        publication = pool.flowkv_step_publication
        pool.flowkv_arena._module.retire_request_pages(
            pool.flowkv_arena.step,
            publication.peers.ptrs,
        )
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(pool.flowkv_shards.device))
        done.synchronize()
        return True
