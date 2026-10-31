"""Independent owner banks retire once after their actual final consumer."""

from types import SimpleNamespace

import pytest

from ulacache.cache.owner_cache import RecordOwnerPagedResidency


def engine(selection_by_layer=(0, 1, 0, 1), last_consumers=(2, 3), batched=True):
    events = []
    runtime = RecordOwnerPagedResidency.__new__(RecordOwnerPagedResidency)
    runtime.execution_plan = SimpleNamespace(selection_indices_by_layer=selection_by_layer,
        last_consumer_layers_by_selection=last_consumers,
        selection_producer_layers=(0, 1))
    live = {0, 1}
    def finish(domain):
        live.remove(domain)
        events.append(('local_finish', domain))
    runtime.owner_residency = SimpleNamespace(residency_states=(object(), object()),
        _live_residency_domains=live, finish_record_residency=finish)
    runtime.publication = SimpleNamespace(finish_all_readers=lambda: events.append('batch_publish'))
    runtime._batch_reader_retirement = batched
    runtime._step_reader_retirement_layer = 3
    runtime._completed_step_selections = set()
    return runtime, events


def test_interleaved_consumers_keep_domains_live_until_one_final_publication():
    runtime, events = engine()
    for layer in range(3):
        runtime.after_attention(layer)
        assert events == []
        assert runtime.owner_residency._live_residency_domains == {0, 1}
    runtime.after_attention(3)
    assert events == [('local_finish', 0), ('local_finish', 1), 'batch_publish']
    assert runtime.owner_residency._live_residency_domains == set()


def test_missing_consumer_cannot_publish_all_domains_as_retired():
    runtime, events = engine()
    with pytest.raises(RuntimeError, match='every owner selection'):
        runtime.after_attention(3)
    assert events == []
    assert runtime.owner_residency._live_residency_domains == {0, 1}


def test_duplicate_finish_is_rejected_before_any_publication():
    runtime, events = engine()
    runtime.after_attention(2)
    with pytest.raises(RuntimeError, match='exactly once'):
        runtime.after_attention(2)
    assert events == []


def test_unplanned_directory_cannot_be_published_by_the_batch():
    runtime, events = engine()
    runtime.owner_residency._live_residency_domains.remove(1)
    runtime.after_attention(2)
    with pytest.raises(RuntimeError, match='every owner directory'):
        runtime.after_attention(3)
    assert events == []


def test_per_selection_fallback_keeps_existing_reindex_boundary():
    runtime, events = engine(batched=False)
    runtime.finish_prefill_record_chunk = lambda layer: events.append(('selection_finish', layer))
    for layer in range(4):
        runtime.after_attention(layer)
    assert events == [('selection_finish', 2), ('selection_finish', 3)]
