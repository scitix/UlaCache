"""Sample cache residency before the original planner changes it.

The serving adapter binds this observer once before Graph capture. Device-only
counters stay at fixed addresses. Enabling, resetting, and reading them require
an idle boundary; observation never updates cache residency, leases, or LRU.
"""

from ulacache.compiler import load_jit


class CacheHitSampler:
    def __init__(self, pool, *, samples_per_query=32, step_stride=16):
        import torch

        if (
            type(samples_per_query) is not int
            or type(step_stride) is not int
            or not 1 <= samples_per_query <= pool.flowkv_layout.topk
            or step_stride < 1
        ):
            raise ValueError("Positive sampling geometry must fit TopK")
        if pool._flowkv_namespaces != 1 or pool.flowkv_fused_topk:
            raise NotImplementedError(
                "Sampling requires ordinary shared-global unfused native TopK"
            )
        if pool.flowkv_host is None:
            raise NotImplementedError(
                "Global-GPU sampling requires the actual owner CPU-tier directory"
            )
        self.pool = pool
        self.samples_per_query, self.step_stride = samples_per_query, step_stride
        self.enabled = torch.zeros(1, dtype=torch.int32, device=pool.device)
        self.counters = torch.zeros(
            (pool.flowkv_layout.groups, 9), dtype=torch.int64, device=pool.device
        )
        self.active_on_host = False
        self.module = load_jit(
            "udcache_cache_hit_sampler",
            str(samples_per_query),
            str(step_stride),
            cuda_files=["csrc/telemetry/cache_hits.cuh"],
            cuda_wrappers=[("sample", "udcache::telemetry::sample_cache_hits")],
            extra_cuda_cflags=[
                f"-DUDCACHE_HIT_SAMPLES={samples_per_query}",
                f"-DUDCACHE_HIT_STEP_STRIDE={step_stride}",
            ],
        )

    def set_enabled(self, enabled, *, reset=False):
        """Call only outside inference, after native requests have drained."""
        if reset:
            self.counters.zero_()
        self.enabled.fill_(int(bool(enabled)))
        self.active_on_host = bool(enabled)

    def observe(self, layer, indices):
        """Read metadata on the original stream before plan_decode mutates it."""
        pool, layout = self.pool, self.pool.flowkv_layout
        group_id = layout.layer_groups.group(layer)
        if layout.layer_groups.anchors[group_id] != layer:
            return
        arena, group = pool.flowkv_arena, pool.flowkv_arena.groups[layer]
        self.module.sample(
            indices,
            arena.step,
            arena.active,
            arena.current_keys,
            group.hints,
            group.tags,
            pool.flowkv_host.directory_peers.device_ptr,
            self.enabled,
            self.counters[group_id],
            layout.capacity,
            pool.flowkv_host.logical_slots,
            pool._flowkv_storage_size,
            group_id,
            pool.flowkv_rank,
        )

    def snapshot(self):
        """Read one small counter tensor after the complete native cohort ends."""
        rows = self.counters.cpu().tolist()
        names = (
            "history_samples",
            "local_hits",
            "local_misses",
            "global_gpu_hits_after_local_miss",
            "cpu_after_local_miss",
            "unallocated_after_local_miss",
            "current_samples",
            "invalid_samples",
            "sampled_query_rows",
        )
        groups = [
            dict(group=group, **dict(zip(names, row))) for group, row in enumerate(rows)
        ]
        totals = [sum(row[column] for row in rows) for column in range(len(names))]
        weights = [
            1 + len(followers)
            for followers in self.pool.flowkv_layout.layer_groups.followers
        ]
        weighted = [
            sum(row[column] * weight for row, weight in zip(rows, weights))
            for column in range(len(names))
        ]

        def aggregate(values):
            history, local, misses, gpu, cpu, missing, *_ = values
            assert history == local + misses and misses == gpu + cpu + missing
            return dict(
                zip(names, values),
                local_hit_rate=local / history if history else None,
                global_gpu_hit_rate_after_local_miss=gpu / misses if misses else None,
                cpu_fraction_of_history_samples=cpu / history if history else None,
            )

        return dict(
            groups=groups,
            total=aggregate(totals),
            layer_weighted=aggregate(weighted),
            samples_per_query=self.samples_per_query,
            step_stride=self.step_stride,
            denominator="Historical TopK occurrences; current candidates and padding excluded",
            global_gpu_denominator="Historical local-miss occurrences, including repeated query references",
            scope="Deterministic rotating GPU sample over the full decode run, including batch drain; no LRU updates",
        )
