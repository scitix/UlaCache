"""Replay must measure native MTP bursts without changing generated history."""
import asyncio
import json

import pytest

from importlib import import_module
_streaming = import_module("scripts.7_1_online.run")
SSEDecoder = _streaming.SSEDecoder
StreamAccumulator = _streaming.StreamAccumulator
generate = _streaming.generate
native_counters = _streaming.native_counters
summarize = _streaming.summarize


def test_sse_keeps_fragmented_multiline_events_and_final_line():
    parser = SSEDecoder()
    wire = b':comment\r\ndata: {"a":\r\ndata: 1}\r\n\r\ndata: [DONE]'
    events = []
    for byte in wire:
        events.extend(parser.feed(bytes([byte])))
    events.extend(parser.finish())
    assert events == [b'{"a":\n1}', b'[DONE]']


@pytest.mark.parametrize('mode', ['incremental', 'cumulative'])
def test_mtp_bursts_preserve_ids_and_use_token_denominator(mode):
    state = StreamAccumulator(output_mode=mode)
    state.add(dict(output_ids=[3, 4, 5], meta_info=dict(completion_tokens=3)), 2.)
    second = [6, 7] if mode == 'incremental' else [3, 4, 5, 6, 7]
    state.add(dict(output_ids=second, meta_info=dict(completion_tokens=5)), 4.)
    state.add(dict(meta_info=dict(completion_tokens=5)), 4.1)
    assert state.ids == [3, 4, 5, 6, 7]
    assert state.metrics()['tpot_seconds'] == .5
    assert state.metrics()['last_token_seconds'] == 4.
    assert state.metrics()['first_burst_tokens'] == 3


def test_cumulative_rewrite_and_wrong_incremental_mode_fail():
    state = StreamAccumulator(output_mode='cumulative')
    state.add(dict(output_ids=[1, 2], meta_info=dict(completion_tokens=2)), 1.)
    with pytest.raises(ValueError, match='committed prefix'):
        state.add(dict(output_ids=[1, 3, 4], meta_info=dict(completion_tokens=3)), 2.)
    state = StreamAccumulator()
    state.add(dict(output_ids=[1], meta_info=dict(completion_tokens=1)), 1.)
    with pytest.raises(ValueError, match='mode/count'):
        state.add(dict(output_ids=[1, 2], meta_info=dict(completion_tokens=2)), 2.)


def test_native_counter_identity_rejects_unrelated_acceptance_reports():
    with pytest.raises(ValueError, match='configured draft steps'):
        native_counters(dict(spec_verify_ct=3, spec_num_correct_drafts=2,
                             spec_num_proposed_drafts=3))
    assert native_counters({}, spec_on=False)['spec_verify_ct'] is None


class Session:
    def __init__(self, events, error=None):
        self.events, self.error, self.payload = events, error, None

    def post(self, url, *, json):
        self.payload = json
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    @property
    def content(self):
        return self

    status = 200

    async def iter_any(self):
        # Deliberately split inside JSON, rather than at SSE frame boundaries.
        wire = b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in self.events)
        for pos in range(0, len(wire), 7):
            yield wire[pos:pos+7]
        if self.error:
            raise self.error
        yield b'data: [DONE]\n\n'


def native_events(rank):
    meta = dict(id='request', prompt_tokens=2, completion_tokens=3,
                finish_reason=dict(type='length'), cached_tokens=1,
                spec_verify_ct=1, spec_num_correct_drafts=2,
                spec_num_proposed_drafts=2, dp_rank=rank)
    return [dict(output_ids=[7, 8, 9], meta_info=meta)]


def test_native_generate_routes_and_keeps_natural_outputs(tmp_path):
    session = Session(native_events(5))
    first = []
    result = asyncio.run(generate(session, dict(url='http://local', attention_dp=8,
        attention_tp=1), [1, 2], 3, 'request', 'private-instance', 5,
        tmp_path/'failure.json', on_first_token=lambda: first.append(True)))
    assert result['output_ids'] == [7, 8, 9]
    assert first == [True]
    assert session.payload['extra_key'] == 'private-instance'
    assert session.payload['routed_dp_rank'] == 5
    assert session.payload['sampling_params']['ignore_eos']
    assert 'forced_output_ids' not in session.payload
    assert not (tmp_path/'failure.json').exists()


@pytest.mark.parametrize('failure', ['wrong_rank', 'short_output', 'transport'])
def test_failure_preserves_partial_response_without_abort(tmp_path, failure):
    events = native_events(2 if failure == 'wrong_rank' else 0)
    if failure == 'short_output':
        events[0]['output_ids'] = [7]
        events[0]['meta_info']['completion_tokens'] = 1
    session = Session(events, OSError('connection lost') if failure == 'transport' else None)
    path = tmp_path/'failure.json'
    with pytest.raises((ValueError, OSError)):
        asyncio.run(generate(session, dict(url='http://local', attention_dp=8,
            attention_tp=1), [1, 2], 3, 'request', 'private', 0, path))
    saved = json.loads(path.read_text())
    assert saved['received_output_tokens'] > 0
    assert not saved['automatic_model_signals']
    assert not saved['automatic_native_abort']


def test_request_mean_differs_from_token_weighted_and_excludes_retraction_hits():
    rows = [dict(output_tokens=3, ttft_seconds=1., tpot_seconds=2., total_seconds=6.,
                 decode_arrival_span_seconds=4., input_tokens=10, cached_prefix_tokens=8),
            dict(output_tokens=5, ttft_seconds=2., tpot_seconds=1., total_seconds=7.,
                 decode_arrival_span_seconds=4., input_tokens=20, cached_prefix_tokens=23,
                 num_retractions=1)]
    stats = summarize(rows, 10.)
    assert stats['tpot_mean_seconds'] == 1.5
    assert stats['tpot_token_weighted_seconds'] == 8/6
    assert stats['total_mean_seconds'] == 6.5
    assert stats['output_tokens_per_second'] == .8
    assert stats['prefix_cache']['token_weighted_hit_rate'] == .8
    assert stats['prefix_cache']['retracted_requests'] == 1
