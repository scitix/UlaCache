"""CUDA shared-Index planning and prefetch stream lifetimes."""


class SharedIndexLifecycle:
    """Plan once per native Index; optionally overlap its group's history pull."""

    def __init__(self, groups, pool, *, prefetch=False):
        from ulacache.cache_config import PrefetchPolicy

        policy = getattr(pool, "flowkv_prefetch_policy", prefetch)
        if type(policy) is bool:
            policy = PrefetchPolicy(enabled=policy)
        if not isinstance(policy, PrefetchPolicy):
            raise TypeError("TP prefetch requires a boolean or PrefetchPolicy")
        if type(prefetch) is not bool and not isinstance(prefetch, PrefetchPolicy):
            raise TypeError("TP prefetch requires a boolean or PrefetchPolicy")
        if bool(prefetch) != policy.enabled:
            raise ValueError(
                "Prefetch must match the policy bound before pool construction"
            )
        completion = "layer" if policy.completion == "auto" else policy.completion
        if isinstance(prefetch, PrefetchPolicy):
            requested = (
                "layer" if prefetch.completion == "auto" else prefetch.completion
            )
            if requested != completion:
                raise ValueError(
                    "Prefetch completion differs from the bound native pool policy"
                )
        self.groups = groups
        self.pool = pool
        self.prefetch = policy.enabled
        self.completion = completion
        self._planned = None
        self._deferred_pending = False
        self._consumed = set()
        self._keepalive = {}
        self._followers_scheduled = set()
        self._followers_joined = set()
        if self.prefetch:
            import torch

            self._stream = torch.cuda.Stream(device=pool.device)
            self._anchor_ready = tuple(
                torch.cuda.Event() for _ in range(groups.group_count)
            )
            self._followers_ready = tuple(
                torch.cuda.Event() if layers and completion == "group" else None
                for layers in groups.followers
            )
            self._layer_ready = tuple(
                torch.cuda.Event()
                if completion == "layer" and layer not in groups.anchors
                else None
                for layer in range(groups.layer_count)
            )

    @property
    def in_forward(self):
        return self._planned is not None

    def begin_forward(self):
        if self.in_forward:
            raise RuntimeError("The previous shared Index forward has not finished")
        self._planned = set()
        self._deferred_pending = False
        self._consumed.clear()
        self._followers_scheduled.clear()
        self._followers_joined.clear()

    def join_idle_prefetch(self):
        """Order an out-of-forward mutation after all local history pulls."""
        if self.in_forward:
            raise RuntimeError(
                "History storage cannot change during a shared Index forward"
            )
        if self.prefetch:
            import torch

            torch.cuda.current_stream(self.pool.device).wait_stream(self._stream)

    def before_layer_prepare(self, layer):
        """Join only the selected completion boundary before Q/RoPE prepare."""
        group = self.groups.group(layer)
        followers = self.groups.followers[group]
        if not self.prefetch:
            return
        if layer in followers and (self.completion == "layer" or layer == followers[0]):
            key = layer if self.completion == "layer" else group
            if key not in self._followers_scheduled:
                raise RuntimeError("Follower history must be scheduled by its anchor")
            import torch

            torch.cuda.current_stream(self.pool.device).wait_event(
                self._layer_ready[layer]
                if self.completion == "layer"
                else self._followers_ready[group]
            )
            self._followers_joined.add(key)

    def on_index_ready(self, layer, native_indices):
        if self.prefetch:
            import torch

            self._stream.wait_stream(torch.cuda.current_stream(self.pool.device))
            with torch.cuda.stream(self._stream):
                self._plan_index(layer, native_indices)
                self._pull_history(layer)
                self._anchor_ready[self.groups.group(layer)].record(self._stream)
                # The payload event intentionally precedes deferred GPU-only
                # PB bookkeeping. Cold owners retain the original early path.
                host = getattr(self.pool, "flowkv_host", None)
                if host is not None:
                    self._deferred_pending |= host.finish_deferred_group(
                        layer, queries=native_indices.shape[0]
                    )
        else:
            self._plan_index(layer, native_indices)

    def _plan_index(self, layer, native_indices):
        from ulacache.cache.tp_cache import plan_decode

        group = self.groups.group(layer)
        if self.groups.anchors[group] != layer or group in self._planned:
            raise RuntimeError("Only one native Index result may plan each anchor")
        current = self.pool._flowkv_current_native
        rows = plan_decode(self.pool, layer, native_indices, current, begin=False)
        if self.pool.flowkv_host is not None:
            self.pool.flowkv_host.prepare_group(layer, queries=native_indices.shape[0])
        self._keepalive[group] = (
            native_indices,
            self.pool._flowkv_global_indices[:rows],
            current,
        )
        self._planned.add(group)

    def before_history_consume(self, layer):
        group = self.groups.group(layer)
        if group not in self._planned:
            raise RuntimeError("Layer history has no native Index metadata plan")
        if self.prefetch:
            if layer == self.groups.anchors[group]:
                import torch

                # The native writer below this hook reads the planned cursor
                # and retention metadata; join before the writer, not after it.
                torch.cuda.current_stream(self.pool.device).wait_event(
                    self._anchor_ready[group]
                )
            elif (
                layer if self.completion == "layer" else group
            ) not in self._followers_joined:
                raise RuntimeError("Follower history must join before Q preparation")
        else:
            self._pull_history(layer)

    def _pull_history(self, layer):
        from ulacache.cache.tp_cache import pull_grid_blocks, pull_records

        group = self.groups.group(layer)
        pool = self.pool
        rows = self._keepalive[group][0].shape[0]
        if rows:
            # The anchor alone resolves remap/retention; follower transport
            # reads only its immutable history-miss list, never current KV.
            kernel = pool.flowkv_arena.kernel()
            pull = (
                kernel.prefetch
                if layer == self.groups.anchors[group]
                else kernel.follower
            )
            pull_records(pool, layer, rows, pull, pull_grid_blocks(pool, rows))

    def after_attention(self, layer):
        self._consumed.add(layer)
        group = self.groups.group(layer)
        if (
            self.prefetch
            and layer == self.groups.anchors[group]
            and self.groups.followers[group]
        ):
            import torch

            self._stream.wait_stream(torch.cuda.current_stream(self.pool.device))
            with torch.cuda.stream(self._stream):
                for follower in self.groups.followers[group]:
                    self._pull_history(follower)
                    if self.completion == "layer":
                        self._layer_ready[follower].record(self._stream)
                        self._followers_scheduled.add(follower)
                if self.completion == "group":
                    self._followers_ready[group].record(self._stream)
                    self._followers_scheduled.add(group)

    def finish_forward(self):
        if self._consumed != set(range(self.groups.layer_count)):
            raise RuntimeError("Forward ended with unconsumed shared Index layers")
        # This ordinary dependency retires deferred PB work, including the
        # final anchor without followers, before any rank publishes read_done.
        # Existing StepPublication then waits every peer before next begin.
        if self.prefetch and self._deferred_pending:
            import torch

            torch.cuda.current_stream(self.pool.device).wait_stream(self._stream)
        self.pool.flowkv_step_publication.publish()
        self._planned = None

    def abort_forward(self):
        """Retire queued pulls before releasing partial-forward tensor owners."""
        try:
            if self.prefetch:
                self._stream.synchronize()
        finally:
            self._planned = None
            self._deferred_pending = False
            self._consumed.clear()
            self._keepalive.clear()
            self._followers_scheduled.clear()
            self._followers_joined.clear()
