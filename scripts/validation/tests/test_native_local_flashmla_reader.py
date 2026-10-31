"""The native FlashMLA leaf must consume the prepared local residency IDs."""
import ast
from types import SimpleNamespace

import pytest

from test_scheduler_receipt_composition import adapter_source


def bind_reader(events, *, position_sharded=False, fail=False):
    native_options = []
    class LocalPool:
        _decode = True
        _current = object()

        def flowkv_prepare_packed_attention(self, **kwargs):
            events.append(('map', kwargs))
            return [3, 7]

        def flowkv_after_attention(self, layer):
            events.append(('retire', layer))

    def native(_backend, _query, _cache, _width, _scale, _layer, _metadata, ids,
               attn_sink=None, *, dcp_partial=True):
        native_options.append((attn_sink, dcp_partial))
        events.append(('native', ids))
        if fail:
            raise RuntimeError('Native attention failed')
        return ids

    module = ast.parse(adapter_source('unified_local_adapter'))
    install = next(node for node in module.body
                   if isinstance(node, ast.FunctionDef) and node.name == 'install')
    reader = next(node for node in install.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'reader')
    namespace = dict(UnifiedLocalDSAPool=LocalPool, original_reader=native,
                     position_sharded=position_sharded, prefill_context_parallel=False)
    exec(compile(ast.Module(body=[reader], type_ignores=[]), 'native_local_reader', 'exec'), namespace)
    namespace['reader'].native_options = native_options
    return namespace['reader'], LocalPool


def call(reader, pool, ids, **options):
    return reader(SimpleNamespace(token_to_kv_pool=pool), 'query', 'encoded-kv',
                  512, 0.125, SimpleNamespace(layer_id=7), 'metadata', ids, **options)


def test_local_decode_maps_before_native_attention_and_retires_after_return():
    events = []
    reader, pool_type = bind_reader(events)
    pool = pool_type()
    assert call(reader, pool, [101, 203]) == [3, 7]
    assert [event[0] for event in events] == ['map', 'native', 'retire']
    assert events[0][1] == dict(layer_id=7, sparse_indices=[101, 203],
                               packed_out_loc=pool._current, decode_like=True)
    assert events[1:] == [('native', [3, 7]), ('retire', 7)]


@pytest.mark.parametrize('position_sharded', [False, True])
def test_native_control_and_existing_dcp_binding_are_not_remapped_twice(position_sharded):
    events = []
    reader, pool_type = bind_reader(events, position_sharded=position_sharded)
    pool = pool_type() if position_sharded else object()
    assert call(reader, pool, [3, 7]) == [3, 7]
    assert events == [('native', [3, 7])]


def test_failed_attention_does_not_publish_reader_retirement():
    events = []
    reader, pool_type = bind_reader(events, fail=True)
    with pytest.raises(RuntimeError, match='Native attention failed'):
        call(reader, pool_type(), [101, 203])
    assert [event[0] for event in events] == ['map', 'native']


@pytest.mark.parametrize('dcp_partial', [False, True])
def test_native_sink_and_dcp_options_reach_the_original_attention(dcp_partial):
    events = []
    reader, pool_type = bind_reader(events)
    sink = object()
    assert call(reader, pool_type(), [101, 203],
                attn_sink=sink, dcp_partial=dcp_partial) == [3, 7]
    assert reader.native_options == [(sink, dcp_partial)]
    assert [event[0] for event in events] == ['map', 'native', 'retire']
