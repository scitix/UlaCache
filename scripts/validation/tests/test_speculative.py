"""Verify native causal metadata and configured linear chains without devices."""
from types import SimpleNamespace

import pytest

from ulacache.speculative import LinearVerifyConfig, index_query_metadata, is_cache_step, draft_kv_cell_bytes


class Mode:
    def __init__(self, name):
        self.name = name

    def is_decode_or_idle(self):
        return self.name in ("decode", "idle")

    def is_target_verify(self):
        return self.name == "verify"


class Tensor:
    def __init__(self, shape, values=None):
        self.shape = shape
        self.values = values


def test_verify_preserves_native_causal_rows_without_allocation():
    # The first query must not see the second query's newly written slot.
    table = Tensor((4, 32))
    causal = Tensor((4,), [101, 102, 2001, 2002])
    metadata = SimpleNamespace(real_page_table=table,
                               dsa_seqlens_expanded=causal,
                               cache_seqlens_int32=Tensor((2,), [102, 2002]))
    rows, lengths = index_query_metadata(Mode("verify"), metadata)
    assert rows is table and lengths is causal
    assert lengths.values == [101, 102, 2001, 2002]
    metadata.dsa_seqlens_expanded = metadata.cache_seqlens_int32
    with pytest.raises(ValueError, match="one causal length"):
        index_query_metadata(Mode("verify"), metadata)


def test_ordinary_decode_uses_original_metadata_and_mode():
    table, lengths = Tensor((2, 32)), Tensor((2,))
    metadata = SimpleNamespace(real_page_table=table, cache_seqlens_int32=lengths)
    assert index_query_metadata(Mode("decode"), metadata) == (table, lengths)
    assert all(is_cache_step(Mode(name)) for name in ("decode", "idle", "verify"))
    assert not is_cache_step(Mode("prefill"))


def test_draft_retains_full_record_cost_despite_striped_target_storage():
    assert draft_kv_cell_bytes(row_bytes=656, layers=1) == 656
    assert draft_kv_cell_bytes(row_bytes=656, layers=2) == 1312
    with pytest.raises(ValueError):
        draft_kv_cell_bytes(row_bytes=656, layers=0)


@pytest.mark.parametrize("changes", [
    {"algorithm": "EAGLE3"}, {"steps": -1}, {"topk": 2},
    {"verify_tokens": 0}, {"verify_tokens": 3}, {"adaptive": True}, {"steps": True},
    {"algorithm": None}, {"topk": True}, {"verify_tokens": 2.0},
])
def test_unintegrated_speculation_is_rejected(changes):
    config = dict(algorithm="EAGLE", steps=1, topk=1, verify_tokens=2)
    LinearVerifyConfig(**config).validate()
    config.update(changes)
    with pytest.raises(NotImplementedError):
        LinearVerifyConfig(**config).validate()


@pytest.mark.parametrize('steps,verify_tokens', [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (7, 8), (16, 17),
    (3, 2), (7, 5), (63, 64),
])
def test_native_linear_lengths_have_no_artificial_step_ceiling(steps, verify_tokens):
    LinearVerifyConfig(algorithm='EAGLE', steps=steps, topk=1,
                       verify_tokens=verify_tokens).validate()


@pytest.mark.parametrize('width', [1, 3, 4, 5, 8, 17])
def test_longer_chains_reuse_every_native_causal_boundary(width):
    histories = [100, 65536, 131072]
    native_values = [history + position + 1 for history in histories for position in range(width)]
    table = Tensor((len(native_values), 2305))
    causal = Tensor((len(native_values),), native_values)
    metadata = SimpleNamespace(real_page_table=table, dsa_seqlens_expanded=causal,
        cache_seqlens_int32=Tensor((len(histories),), [history + width for history in histories]))
    rows, lengths = index_query_metadata(Mode('verify'), metadata)
    assert rows is table and lengths is causal
    assert lengths.values == native_values


def test_native_metadata_padding_does_not_authorize_extra_resident_queries():
    table = Tensor((5, 32))
    lengths = Tensor((5,), [101, 102, 103, 104, 105])
    metadata = SimpleNamespace(real_page_table=table, dsa_seqlens_expanded=lengths)
    assert index_query_metadata(Mode('verify'), metadata, max_queries=5) == (table, lengths)
    with pytest.raises(ValueError, match='resident KV capacity'):
        index_query_metadata(Mode('verify'), metadata, max_queries=4)
