"""Whole-conversation scheduling and failure must not corrupt later cohorts."""
import asyncio
import json

from importlib import import_module

_module = import_module('scripts.7_1_online.run')
RecentTTFT = _module.RecentTTFT
native_idle = _module.native_idle
read_idle = _module.read_idle
replay = _module.replay


def trace():
    return dict(key='source', initial_input_ids=[1, 2], turns=[
        dict(append_input_ids=[], output_tokens=2),
        dict(append_input_ids=[20], output_tokens=3)])


def result(ids, budget, *, ttft=1.):
    return dict(output_ids=[71]*budget, meta_info=dict(cached_tokens=len(ids)-1),
                ttft_seconds=ttft, tpot_seconds=.01, total_seconds=ttft+.1,
                decode_arrival_span_seconds=(budget-1)*.01)


def test_threshold_is_recent32_completed_and_sticky():
    recent = RecentTTFT()
    for _ in range(31):
        recent.add(121.)
    assert not recent.failed
    recent.add(121.)
    assert recent.failed
    for _ in range(32):
        recent.add(1.)
    assert recent.mean == 1. and recent.failed
    boundary = RecentTTFT()
    for _ in range(32):
        boundary.add(120.)
    assert not boundary.failed


def test_native_idle_requires_all_ranks_and_strict_zero_counts():
    rows = [dict(dp_rank=r, num_running_reqs=0, num_waiting_reqs=0,
        num_waiting_uncached_tokens=0, num_used_tokens=0, num_total_tokens=0,
        num_active_tokens=0, num_prealloc_ready_tokens=0,
        disaggregation=dict(mode='null'),
        queues=dict(waiting=0, grammar=0, paused=0, retracted=0, prealloc_ready=0))
        for r in range(8)]
    assert native_idle(dict(loads=rows), 8)
    assert not native_idle(dict(loads=rows[:7]), 8)
    rows[3]['num_waiting_reqs'] = 1
    assert not native_idle(dict(loads=rows), 8)
    rows[3]['num_waiting_reqs'] = False
    assert not native_idle(dict(loads=rows), 8)


def test_actual_generated_history_and_private_instance_namespaces(tmp_path):
    calls = []

    async def generate(session, plan, ids, budget, rid, namespace, rank, path):
        calls.append(dict(ids=list(ids), budget=budget, namespace=namespace, rank=rank))
        await asyncio.sleep(0)
        return result(ids, budget)

    plan = dict(concurrency=2, instances=4, attention_dp=8, attention_tp=1)
    value = asyncio.run(replay(None, plan, [trace()], tmp_path, generate_request=generate))
    assert value['completed_turns'] == value['total_turns'] == 8
    assert value['completed_instances'] == 4 and not value['errors']
    assert value['full_original_conversations_replayed']
    namespaces = {c['namespace'] for c in calls}
    assert len(namespaces) == 4
    for key in namespaces:
        pair = [c for c in calls if c['namespace'] == key]
        assert [c['ids'] for c in pair] == [[1, 2], [1, 2, 71, 71, 20]]
        assert pair[0]['rank'] == pair[1]['rank']
    assert (tmp_path/'RESULTS.json').exists()


def test_fast_user_keeps_its_original_conversation_queue(tmp_path):
    """Unequal conversation durations must not change the paper's GPU assignment."""
    calls = {}

    async def run():
        fast_queue_finished = asyncio.Event()

        async def generate(session, plan, ids, budget, rid, namespace, rank, path):
            instance = int(namespace.rsplit('instance', 1)[1])
            calls[instance] = rank
            if instance == 0:
                await fast_queue_finished.wait()
            if instance == 3:
                fast_queue_finished.set()
            return result(ids, budget)

        row = trace()
        row['turns'] = row['turns'][:1]
        return await replay(None, dict(concurrency=2, instances=4,
            attention_dp=8, attention_tp=1), [row], tmp_path,
            generate_request=generate)

    value = asyncio.run(run())
    assert calls == {0: 0, 1: 1, 2: 0, 3: 1}
    assert value['completed_turns'] == 4


def test_one_request_error_stops_admission_but_drains_other_inflight(tmp_path):
    started, finished = [], []

    async def generate(session, plan, ids, budget, rid, namespace, rank, path):
        started.append(rank)
        if rank == 0:
            await asyncio.sleep(.001)
            raise OSError('stream lost')
        await asyncio.sleep(.01)
        finished.append(rank)
        return result(ids, budget)

    plan = dict(concurrency=2, instances=4, attention_dp=8, attention_tp=1)
    value = asyncio.run(replay(None, plan, [trace()], tmp_path, generate_request=generate))
    assert started == [0, 1] and finished == [1]
    assert value['active_requests'] == 0 and len(value['errors']) == 1
    assert value['completed_turns'] == 1 and value['completed_instances'] == 0
    assert not value['full_original_conversations_replayed']
    assert not value['protocol']['native_abort_sent']


def test_recent_ttft_failure_stops_future_turns_and_drains(tmp_path):
    async def generate(session, plan, ids, budget, rid, namespace, rank, path):
        await asyncio.sleep(0)
        return result(ids, budget, ttft=121.)

    row = trace()
    row['turns'] = [row['turns'][0]] + [row['turns'][1]]*40
    value = asyncio.run(replay(None, dict(concurrency=2, instances=4,
        attention_dp=8, attention_tp=1), [row], tmp_path, generate_request=generate))
    assert value['failed_ttft120']
    assert 32 <= value['completed_turns'] <= 33
    assert value['active_requests'] == 0 and value['completed_instances'] == 0
    saved = json.loads((tmp_path/'RESULTS.json').read_text())
    assert saved['failed_ttft120'] and not saved['errors']


def test_functional_endpoint_smoke_cannot_claim_whole_conversation_replay(tmp_path):
    calls = []

    async def generate(session, plan, ids, budget, rid, namespace, rank, path):
        calls.append(rid)
        await asyncio.sleep(0)
        return result(ids, budget)

    plan = dict(concurrency=2, instances=4, attention_dp=8, attention_tp=1,
                functional_smoke_turns_per_instance=1)
    value = asyncio.run(replay(None, plan, [trace()], tmp_path, generate_request=generate))
    assert value['completed_instances'] == 4 and value['completed_turns'] == 4
    assert value['purpose'] == 'functional_smoke'
    assert not value['full_original_conversations_replayed']
    assert len(calls) == 4
    source = json.loads((tmp_path/'SOURCE_WORKLOAD_STATISTICS.json').read_text())
    measured = json.loads((tmp_path/'WORKLOAD_STATISTICS.json').read_text())
    assert source['turns'] == 8 and measured['turns'] == 4


def test_final_stream_can_precede_native_drain_without_becoming_a_false_failure():
    row = dict(dp_rank=0, num_running_reqs=0, num_waiting_reqs=0,
        num_waiting_uncached_tokens=0, num_used_tokens=0, num_total_tokens=0,
        num_active_tokens=0, num_prealloc_ready_tokens=0,
        disaggregation=dict(mode='null'),
        queues=dict(waiting=0, grammar=0, paused=0, retracted=0, prealloc_ready=0))

    class Session:
        calls, status = 0, 200
        def get(self, url):
            self.calls += 1
            return self
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def json(self):
            return dict(loads=[dict(row, num_running_reqs=1 if self.calls == 1 else 0)])

    session = Session()
    assert native_idle(asyncio.run(read_idle(session, 'http://local', 1, poll_seconds=0)), 1)
    assert session.calls == 2
