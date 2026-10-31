"""Replay preserves real draft/sampler calls and validates workload identity."""

import json
import os
from types import SimpleNamespace as N

import pytest

from ulacache.tools.fixed_output import FixedMTPReplay, FixedTokenReplay, ReplayTrace


def trace_file(tmp_path):
    frame = dict(
        keys=["a", "b"],
        seq_lens=[64, 128],
        initial_topk_index=[[7], [8]],
        current_tokens=[2, 3],
        target_inputs=[[2, 7, 9], [3, 8, 10]],
        predict=[7, 9, 11, 8, 10, 12],
        accept_lens=[3, 2],
        accept_index=[[0, 1, 2], [3, 4, -1]],
    )
    last = dict(
        keys=["b"],
        seq_lens=[130],
        initial_topk_index=[[12]],
        current_tokens=[10],
        target_inputs=[[10, 12, 0]],
        predict=[12, 0, 0],
        accept_lens=[1],
        accept_index=[[0, -1, -1]],
    )
    data = dict(
        schema="ulacache-fixed-mtp-replay-v1",
        forced_acceptance=True,
        rejected_proposals="synthetic",
        output_tokens=4,
        episodes={
            "one": dict(
                batch=2,
                steps=2,
                topk=1,
                verify_width=3,
                expected_output_ids={"a": [2, 7, 9, 11], "b": [3, 8, 10, 12]},
                frames=[frame, last],
            )
        },
        cases={"1:1:2:0": "one"},
    )
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(data))
    return ReplayTrace(p)


def batch_for(trace, frame, device="cpu"):
    import torch

    reqs = [
        N(
            sampling_params=N(
                custom_params=dict(
                    udcache_snapshot_cohort="cohort",
                    ulacache_replay_round=1,
                    udcache_snapshot_batch=2,
                    ulacache_replay_sha256=trace.sha256,
                    udcache_snapshot_key=key,
                )
            )
        )
        for key in frame["keys"]
    ]
    return N(
        reqs=reqs,
        seq_lens_cpu=torch.tensor(frame["seq_lens"]),
        sampling_info=N(is_all_greedy=True),
        has_grammar=False,
        return_logprob=False,
        spec_info=N(
            topk_index=torch.zeros((len(reqs), 1), dtype=torch.int64, device=device),
            topk_p=torch.zeros((len(reqs), 1), device=device),
            bonus_tokens=torch.zeros(len(reqs), dtype=torch.int64, device=device),
        ),
    )


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if os.environ.get("ULACACHE_TEST_CUDA") == "1" else [])
)
def test_replay_keeps_native_execution_and_handles_completed_request_removal(
    tmp_path, device
):
    import torch

    trace = trace_file(tmp_path)
    engine = FixedMTPReplay(trace, attention_dp=1, rank=0)
    worker = N(speculative_num_steps=2, topk=1, speculative_num_draft_tokens=3)
    calls = []
    for frame in trace.episode(1, 1, 2, 0)["frames"]:
        batch = batch_for(trace, frame, device)
        count = len(batch.reqs)

        def draft(actual_worker, actual_batch):
            calls.append("draft")
            assert actual_worker is worker and actual_batch is batch
            assert (
                actual_batch.spec_info.topk_index.tolist()
                == frame["initial_topk_index"]
            )
            return N(
                topk=1,
                spec_steps=2,
                draft_token_num=3,
                draft_token=torch.zeros(count * 3, dtype=torch.int64, device=device),
            )

        verify = engine.draft(draft, worker, batch)
        assert verify.draft_token.tolist() == sum(frame["target_inputs"], [])

        def sample(actual_verify, actual_batch):
            calls.append("sample")
            assert actual_verify is verify and actual_batch is batch
            return (
                torch.zeros(count * 3, dtype=torch.int32, device=device),
                torch.zeros(count, dtype=torch.int32, device=device),
                torch.zeros((count, 3), dtype=torch.int32, device=device),
            )

        actual = engine.sample(sample, verify, batch)
        assert actual[0].tolist() == frame["predict"]
        assert actual[1].tolist() == frame["accept_lens"]
        assert actual[2].tolist() == frame["accept_index"]
    assert calls == ["draft", "sample", "draft", "sample"]
    assert engine.completed("cohort")
    receipt = engine.receipt("cohort")
    assert receipt["frames"] == receipt["draft_calls"] == receipt["sampler_calls"] == 2
    assert receipt["natural_accuracy_claimed"] is False


@pytest.mark.parametrize("change", ["hash", "position", "mixed"])
def test_replay_rejects_a_different_workload_before_native_execution(tmp_path, change):
    trace = trace_file(tmp_path)
    engine = FixedMTPReplay(trace, attention_dp=1, rank=0)
    batch = batch_for(trace, trace.episode(1, 1, 2, 0)["frames"][0])
    if change == "hash":
        for req in batch.reqs:
            req.sampling_params.custom_params["ulacache_replay_sha256"] = "wrong"
    elif change == "position":
        batch.seq_lens_cpu[0] += 1
    else:
        del batch.reqs[0].sampling_params.custom_params["ulacache_replay_round"]
    with pytest.raises(ValueError):
        engine.draft(
            lambda *_: pytest.fail("Native call before validation"),
            N(speculative_num_steps=2, topk=1, speculative_num_draft_tokens=3),
            batch,
        )


def test_packaged_paper_trace_covers_both_topologies_and_three_rounds():
    from pathlib import Path

    path = Path(__file__).parents[2] / "datasets/hy4_lowbs_replay.json.gz"
    if not path.is_file():
        pytest.skip("Experimental replay data is not included in the source release")
    trace = ReplayTrace(path)
    for dp in (1, 8):
        for repeat in (1, 2, 3):
            for batch in (1, 2, 4, 8):
                episodes = [
                    trace.episode(dp, repeat, batch, rank) for rank in range(dp)
                ]
                assert sum(e["batch"] for e in episodes) == batch
                assert all(
                    len(v) == 2048
                    for e in episodes
                    for v in e["expected_output_ids"].values()
                )


@pytest.mark.parametrize("batch", [N(), N(reqs=None), N(reqs=[])])
def test_graph_capture_and_natural_batches_do_not_activate_replay(tmp_path, batch):
    engine = FixedMTPReplay(trace_file(tmp_path), attention_dp=1, rank=0)
    token = object()
    assert engine.draft(lambda *_: token, object(), batch) is token
    assert engine.sample(lambda *_: token, object(), batch) is token
    assert not engine.cohorts and engine.pending is None


def plain_trace(tmp_path):
    data = dict(
        schema="ulacache-fixed-output-replay-v1",
        speculative=False,
        output_tokens=4,
        episodes={
            "one": dict(
                batch=1,
                keys=["a"],
                input_tokens=[64],
                saved_bootstrap_tokens=1,
                expected_output_ids={"a": [2, 7, 9, 11]},
            )
        },
        cases={"1:1:2:0": "one"},
    )
    p = tmp_path / "plain.json"
    p.write_text(json.dumps(data))
    return ReplayTrace(p)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if os.environ.get("ULACACHE_TEST_CUDA") == "1" else [])
)
def test_plain_replay_returns_token_views_after_native_execution(tmp_path, device):
    import torch

    trace = plain_trace(tmp_path)
    engine = FixedTokenReplay(trace, attention_dp=1, rank=0)
    info = N(
        is_all_greedy=True,
        grammars=None,
        has_custom_logit_processor=False,
        sampling_mask_batch_indices=None,
    )
    views = []
    for step, token in enumerate([7, 9, 11]):
        batch = batch_for(trace, dict(keys=["a"], seq_lens=[65 + step]), device)
        assert engine.begin(batch)
        original = torch.tensor([99], dtype=torch.int64, device=device)
        actual = engine.sample(original, info, False)
        assert original.item() == 99 and actual.item() == token
        views.append(actual)
        engine.end()
    assert len({v.untyped_storage().data_ptr() for v in views}) == 1
    receipt = engine.receipt("cohort")
    assert receipt["complete"] and receipt["forced_acceptance"] is False
    assert receipt["model_calls"] == receipt["sampler_calls"] == 3


def test_plain_replay_requires_a_sampler_for_each_native_forward(tmp_path):
    trace = plain_trace(tmp_path)
    engine = FixedTokenReplay(trace, attention_dp=1, rank=0)
    engine.begin(batch_for(trace, dict(keys=["a"], seq_lens=[65])))
    with pytest.raises(ValueError, match="omitted"):
        engine.end()


def test_plain_replay_rejects_a_step_before_the_restored_bootstrap(tmp_path):
    trace = plain_trace(tmp_path)
    engine = FixedTokenReplay(trace, attention_dp=1, rank=0)
    with pytest.raises(ValueError, match="position"):
        engine.begin(batch_for(trace, dict(keys=["a"], seq_lens=[64])))


def test_packaged_plain_trace_covers_both_topologies_and_three_rounds():
    from pathlib import Path

    path = Path(__file__).parents[2] / "datasets/hy4_lowbs_plain_replay.json.gz"
    if not path.is_file():
        pytest.skip("Experimental replay data is not included in the source release")
    trace = ReplayTrace(path)
    assert trace.speculative is False
    for dp in (1, 8):
        for repeat in (1, 2, 3):
            for batch in (1, 2, 4, 8):
                episodes = [
                    trace.episode(dp, repeat, batch, rank) for rank in range(dp)
                ]
                assert sum(e["batch"] for e in episodes) == batch
