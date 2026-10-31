"""Shared prefill-pipeline callbacks over position-sharded storage."""


class PositionPrefillMaterializer:
    def __init__(self, engine, assemble_history):
        if not callable(assemble_history):
            raise TypeError("Bind the existing CP-group history collective explicitly")
        self.engine = engine
        self.assemble_history = assemble_history
        self._current = {}

    def begin(self):
        self.engine._drain_transfers()
        self._current.clear()

    def record_current(self, layer, rows, nope, rope):
        """Borrow the complete native writer output after CP KV assembly."""
        self._current.clear()
        self._current[layer] = rows, nope, rope

    def gather_history(self, layer, plan, window):
        from ulacache.cache.kernels.read_only_prefill import gather_history

        group = self.engine.groups.group(layer)
        gather_history(
            plan.record_ids,
            plan.fresh_positions,
            self.engine.directory[group],
            self.engine.gpu[layer],
            self.engine.host.ptrs,
            window,
            layer=layer,
            host_slots=self.engine.host_slots,
            row_bytes=self.engine.row_bytes,
            launch=self.engine.prefill_exchange_launch,
            owner_count=self.engine.layout.degree,
            owner_rank=self.engine.layout.rank,
        )
        # Like the existing TP history assembly, owners contribute disjoint
        # byte rows, including explicit zeros for non-owner/fresh positions.
        # The bound collective must add a completion dependency to THIS stream.
        self.assemble_history(
            window.view(-1)[: plan.record_count * self.engine.row_bytes]
        )

    def install_current(self, layer, plan, window):
        from ulacache.cache.kernels.read_only_prefill import install_current

        if layer not in self._current:
            raise RuntimeError(
                "Native CP current KV must be assembled before its prefill read"
            )
        rows, nope, rope = self._current.pop(layer)
        install_current(
            plan.record_ids,
            plan.fresh_positions,
            rows,
            nope,
            rope,
            window,
            row_bytes=self.engine.row_bytes,
            launch=self.engine.prefill_exchange_launch,
        )

    def finish(self):
        self._current.clear()
