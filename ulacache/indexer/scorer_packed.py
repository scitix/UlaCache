"""Dense-unpack layout selection over the unchanged native score pipeline."""

from dataclasses import replace

from ulacache.indexer.scorer import CPIndexScorer


class PackedCPIndexScorer(CPIndexScorer):
    """Keep GPU-dynamic transport descriptors while exposing dense score rows."""

    def score(self, **kwargs):
        result = super().score(**kwargs)
        if getattr(self.group, "unpacks_scores", False):
            return replace(result, compact_query_offsets=None, compact_input_packs=None)
        return result
