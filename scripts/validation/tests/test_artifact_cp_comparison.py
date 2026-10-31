"""An incomplete or varying-batch run cannot satisfy the CP overhead target."""
import copy
import hashlib
import json

import pytest

from importlib import import_module

_module = import_module('scripts.7_3_low_latency.context_parallel')
BATCHES = _module.BATCHES
compare = _module.compare
read_arm = _module.read_arm
fixed_batch_window = _module.fixed_batch_window


def arm(kind, *, multiplier=1., density=2.4):
    cases = []
    for repeat in (1, 2, 3):
        for batch in BATCHES:
            cases.append(dict(repeat=repeat, batch=batch, request_tpot_ms=10*multiplier,
                request_ttft_seconds=.5, step_ms=24*multiplier,
                fixed_tpot_ms=24*multiplier/density,
                output_tokens=4096*batch, verify_steps=2048*batch,
                accepted_drafts=2047*batch, proposed_drafts=4096*batch))
    return dict(kind=kind, plan=dict(attention_dp=1, attention_tp=8,
        allocation_page_size=512, spec_on=True, speculative_steps=2,
        input_tokens=131072, output_tokens=4096, selected_input_token_sha256='same'),
        cases=cases, bindings=[])


def test_all_timing_metrics_and_all_rounds_are_required():
    baseline, candidate = arm('native'), arm('ulacache', multiplier=1.04)
    value = compare(baseline, candidate)
    assert value['all_batches_within5percent']
    assert value['batches'][0]['overhead_percent']['request_tpot_ms'] == pytest.approx(4.)
    candidate['cases'] = candidate['cases'][:-1]
    with pytest.raises(ValueError, match='Three actual repeats'):
        compare(baseline, candidate)


def test_fast_steps_cannot_hide_acceptance_density_regression():
    candidate = arm('ulacache', density=1.3)
    value = compare(arm('native'), candidate)
    assert not value['all_batches_within5percent']
    row = value['batches'][0]
    assert row['overhead_percent']['step_ms'] == 0
    assert row['overhead_percent']['fixed_tpot_ms'] > 80


def test_changed_prompt_or_draft_configuration_rejects_pair():
    for field in ['selected_input_token_sha256', 'speculative_steps']:
        candidate = arm('ulacache')
        candidate['plan'][field] = 'different'
        with pytest.raises(ValueError, match='paired workload'):
            compare(arm('native'), candidate)


def test_six_moments_detect_variable_batch_even_if_mean_batch_matches():
    # Equal counts at1 and3 have mean2, but second moment exposes the change.
    with pytest.raises(ValueError, match='batch changed'):
        fixed_batch_window([0]*6, [200, 400, 10000000, 1000, 20000000, 800], 2)
    actual = fixed_batch_window([0]*6, [200, 400, 10000000, 800, 20000000, 800], 2)
    assert actual['mean_step_ms'] == 50
    assert actual['effective_tpot_ms'] == 25
    changed = copy.copy([200, 400, 10000000, 800, 20000000, 800])
    changed[0] = -1
    with pytest.raises(ValueError, match='Counter reset'):
        fixed_batch_window([0]*6, changed, 2)


def test_completed_preemption_case_cannot_cross_a_serving_code_fix(tmp_path):
    root = tmp_path/'current'/'results'
    previous_root = tmp_path/'previous'

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    plan = arm('native')['plan'] | dict(variant='native', service_out=str(root.parent/'service'))
    prior = dict(plan, service_out=str(previous_root/'service'))
    for source in (root.parent, previous_root):
        for role in ('library', 'framework', 'runtime'):
            write(source/role/'SOURCE.json', dict(files={'module.py': 'same-sha'}))
    write(root/'ACTUAL_PLAN.json', plan)
    write(root/'STARTUP_ACCEPTANCE.json', dict(policies=[dict(rank=r, passed=True,
        kind='native', fp8_kv=True, dcp_size=8, MTP=[2, 1, 3],
        max_running_requests=64, cgbs=64, simulated_acceptance=False) for r in range(8)]))
    cases = []
    for repeat in (1, 2, 3):
        for batch in BATCHES:
            delta = [200, 200*batch, 10000000, 200*batch*batch, 10000000*batch, 400*batch]
            window = fixed_batch_window([0]*6, delta, batch)
            window.update(observed_wall_seconds=10.,
                before=dict(loads=dict(loads=[dict(decode_moments=[0]*6)])),
                after=dict(loads=dict(loads=[dict(decode_moments=delta)])))
            value = dict(requests_completed=True, fixed_batch_qualified=True,
                requested_batch=batch, input_tokens=131072, natural_output_tokens_per_request=4096,
                full_request_TPOT_mean_ms=10., full_request_TTFT_mean_seconds=.5,
                MTP_full_requests=dict(output_tokens=4096*batch, spec_verify_ct=2048*batch,
                    spec_num_correct_drafts=2047*batch, spec_num_proposed_drafts=4096*batch),
                fixed_window=window)
            path = root/f'round{repeat}'/f'B{batch:02d}'/'COMPLETE.json'
            item = dict(repeat=repeat, requested_batch=batch, fixed_batch_qualified=True)
            if (repeat, batch) == (1, 1):
                path = previous_root/'COMPLETE.json'
                prior_path = previous_root/'PLAN.json'
                prior_sha = write(prior_path, prior)
                item['retained_previous_epoch'] = dict(path=str(path),
                    sha256=write(path, value), plan_path=str(prior_path), plan_sha256=prior_sha)
            else:
                write(path, value)
            cases.append(item)
    write(root/'COMPLETE.json', dict(passed=True, cases=cases))
    assert len(read_arm(root, 'native')['cases']) == 15
    write(previous_root/'framework'/'SOURCE.json', dict(files={'module.py': 'different-code'}))
    with pytest.raises(ValueError, match='serving-code change'):
        read_arm(root, 'native')


def test_average_overhead_accepts_one_outlying_batch_without_hiding_it():
    candidate = arm('ulacache', multiplier=1.03)
    for row in candidate['cases']:
        if row['batch'] == 2:
            row['request_tpot_ms'] = 10.6
    result = compare(arm('native'), candidate)
    assert result['average_overhead_within5percent']
    assert result['mean_request_tpot_overhead_percent'] == pytest.approx(3.6)
    assert result['batches'][1]['overhead_percent']['request_tpot_ms'] == pytest.approx(6.)
    assert not result['all_batches_within5percent']
    for row in candidate['cases']:
        row['request_tpot_ms'] = 10.6
    assert not compare(arm('native'), candidate)['average_overhead_within5percent']
