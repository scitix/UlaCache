"""Fixed-history sampler for benchmarks; the native processor base is explicit.

Model forwards still execute. Only the native sampler selects the supplied
next output token. This is performance evidence, not natural correctness.
"""

import gzip
import hashlib
import json
from pathlib import Path


class ReplayTrace:
    """Portable recorded output IDs and linear-verification work for benchmarks."""

    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        schema = self.data.get("schema")
        self.speculative = schema == "ulacache-fixed-mtp-replay-v1"
        if self.speculative:
            if (
                self.data.get("forced_acceptance") is not True
                or self.data.get("rejected_proposals") != "synthetic"
            ):
                raise ValueError("Require an explicit controlled MTP replay trace")
        elif (
            schema != "ulacache-fixed-output-replay-v1"
            or self.data.get("speculative") is not False
        ):
            raise ValueError("Require a recorded output or MTP replay trace")
        for episode in self.data["episodes"].values():
            if self.speculative and (
                episode["steps"],
                episode["topk"],
                episode["verify_width"],
            ) != (
                2,
                1,
                3,
            ):
                raise ValueError("Replay requires linear EAGLE steps2/topk1/verify3")
            if any(
                len(ids) != self.data["output_tokens"]
                for ids in episode["expected_output_ids"].values()
            ):
                raise ValueError(
                    "Recorded output IDs do not cover the full output budget"
                )

    def episode(self, attention_dp, repeat, batch, rank):
        key = f"{attention_dp}:{repeat}:{batch}:{rank}"
        try:
            return self.data["episodes"][self.data["cases"][key]]
        except KeyError as error:
            raise ValueError(f"No recorded replay for {key}") from error


class FixedMTPReplay:
    """Retain real model and sampler calls while replaying one recorded workload.

    The serving adapter supplies native calls and batch objects. Tensor uploads
    occur once per cohort before token64; each measured step only selects views.
    This class has no serving-framework or hardware-specific imports.
    """

    def __init__(self, trace, *, attention_dp, rank):
        import torch

        if not trace.speculative:
            raise ValueError("MTP replay requires a speculative trace")
        self.trace, self.torch = trace, torch
        self.attention_dp, self.rank = attention_dp, rank
        self.cohorts, self.pending = {}, None

    @staticmethod
    def identity(batch):
        params = [
            r.sampling_params.custom_params or {}
            for r in (getattr(batch, "reqs", None) or [])
        ]
        if not params or not any("ulacache_replay_round" in p for p in params):
            return None
        if not all("ulacache_replay_round" in p for p in params):
            raise ValueError(
                "A replay batch cannot mix controlled and natural requests"
            )
        labels = {
            (
                p["udcache_snapshot_cohort"],
                p["ulacache_replay_round"],
                p["udcache_snapshot_batch"],
                p["ulacache_replay_sha256"],
            )
            for p in params
        }
        if len(labels) != 1:
            raise ValueError("A replay batch must belong to one recorded cohort")
        return (*labels.pop(), [p["udcache_snapshot_key"] for p in params])

    def _prepare(self, cohort, repeat, batch_size, draft):
        episode = self.trace.episode(self.attention_dp, repeat, batch_size, self.rank)
        size, frames = episode["batch"], episode["frames"]
        if not size or not frames:
            raise ValueError("An inactive replay rank cannot execute a draft forward")

        def tensor(name, width, dtype, fill=0):
            values = []
            for frame in frames:
                rows = frame[name]
                if name == "predict":
                    rows = [rows[i : i + 3] for i in range(0, len(rows), 3)]
                pad = [[fill] * width] if width else [fill]
                values.append(list(rows) + pad * (size - len(frame["keys"])))
            return self.torch.tensor(
                values, dtype=dtype, device=draft.topk_index.device
            )

        tensors = dict(
            initial=tensor("initial_topk_index", 1, draft.topk_index.dtype),
            current=tensor("current_tokens", 0, draft.bonus_tokens.dtype),
            inputs=tensor("target_inputs", 3, draft.topk_index.dtype),
            predict=tensor("predict", 3, self.torch.int32),
            lengths=tensor("accept_lens", 0, self.torch.int32),
            indices=tensor("accept_index", 3, self.torch.int32, -1),
            probabilities=self.torch.ones(
                (size, 1), dtype=draft.topk_p.dtype, device=draft.topk_index.device
            ),
        )
        state = dict(
            episode=episode,
            tensors=tensors,
            index=0,
            draft_calls=0,
            sampler_calls=0,
            repeat=repeat,
            batch=batch_size,
        )
        self.cohorts[cohort] = state
        return state

    def draft(self, original, worker, batch, *args, **kwargs):
        identity = self.identity(batch)
        if identity is None:
            return original(worker, batch, *args, **kwargs)
        cohort, repeat, size, sha, keys = identity
        if sha != self.trace.sha256:
            raise ValueError("Client and server replay traces differ")
        if (
            worker.speculative_num_steps,
            worker.topk,
            worker.speculative_num_draft_tokens,
        ) != (2, 1, 3):
            raise ValueError("Recorded MTP replay requires steps2/topk1/verify3")
        if (
            not batch.sampling_info.is_all_greedy
            or batch.has_grammar
            or batch.return_logprob
        ):
            raise ValueError(
                "Replay requires greedy sampling without grammar or logprobs"
            )
        draft = batch.spec_info
        state = self.cohorts.get(cohort)
        if state is None:
            state = self._prepare(cohort, repeat, size, draft)
        index = state["index"]
        if self.pending is not None or index >= len(state["episode"]["frames"]):
            raise ValueError(
                "Replay step is overlapping or exceeds its recorded history"
            )
        frame = state["episode"]["frames"][index]
        if keys != frame["keys"] or batch.seq_lens_cpu.tolist() != frame["seq_lens"]:
            raise ValueError("Replay request order or sequence positions differ")
        count, tensors = len(keys), state["tensors"]
        draft.topk_index = tensors["initial"][index, :count]
        draft.topk_p = tensors["probabilities"][:count]
        draft.bonus_tokens = tensors["current"][index, :count]
        result = original(worker, batch, *args, **kwargs)
        verify = result[0] if isinstance(result, tuple) else result
        if (verify.topk, verify.spec_steps, verify.draft_token_num) != (1, 2, 3):
            raise ValueError("Native draft returned incompatible verification geometry")
        scripted = tensors["inputs"][index, :count].reshape(-1)
        if (
            verify.draft_token.shape != scripted.shape
            or verify.draft_token.dtype != scripted.dtype
        ):
            raise ValueError("Native and recorded draft token layouts differ")
        verify.draft_token = scripted
        state["draft_calls"] += 1
        self.pending = batch, cohort, state, index, count
        return result

    def sample(self, original, verify, batch, *args, **kwargs):
        natural = original(verify, batch, *args, **kwargs)
        if self.pending is None:
            if self.identity(batch) is not None:
                raise ValueError("Controlled sampling requires its matching draft step")
            return natural
        pending, cohort, state, index, count = self.pending
        if pending is not batch or state["index"] != index:
            raise ValueError("Sampler and recorded draft step differ")
        tensors = state["tensors"]
        forced = (
            tensors["predict"][index, :count].reshape(-1),
            tensors["lengths"][index, :count],
            tensors["indices"][index, :count],
        )
        if len(natural) != len(forced) or any(
            (a.shape, a.dtype, a.device) != (b.shape, b.dtype, b.device)
            for a, b in zip(natural, forced)
        ):
            raise ValueError("Native sampler and recorded output layouts differ")
        state["sampler_calls"] += 1
        state["index"] += 1
        self.pending = None
        return forced

    def completed(self, cohort):
        state = self.cohorts[cohort]
        return state["index"] == len(state["episode"]["frames"])

    def receipt(self, cohort):
        state = self.cohorts[cohort]
        return dict(
            trace_sha256=self.trace.sha256,
            attention_dp=self.attention_dp,
            rank=self.rank,
            repeat=state["repeat"],
            batch=state["batch"],
            frames=state["index"],
            expected_frames=len(state["episode"]["frames"]),
            draft_calls=state["draft_calls"],
            model_calls=state["draft_calls"],
            sampler_calls=state["sampler_calls"],
            complete=self.completed(cohort),
            forced_acceptance=True,
            natural_accuracy_claimed=False,
        )


class FixedTokenReplay:
    """Return recorded token views after the real non-speculative sampler."""

    def __init__(self, trace, *, attention_dp, rank):
        import torch

        if trace.speculative:
            raise ValueError("Plain replay cannot use a speculative trace")
        self.trace, self.torch = trace, torch
        self.attention_dp, self.rank = attention_dp, rank
        self.cohorts, self.active = {}, None

    def begin(self, batch):
        identity = FixedMTPReplay.identity(batch)
        if identity is None:
            return False
        if self.active is not None:
            raise ValueError("Overlapping plain replay batches are unsupported")
        cohort, repeat, size, sha, keys = identity
        if sha != self.trace.sha256:
            raise ValueError("Client and server replay traces differ")
        state = self.cohorts.get(cohort)
        if state is None:
            episode = self.trace.episode(self.attention_dp, repeat, size, self.rank)
            state = dict(
                episode=episode,
                position=episode["saved_bootstrap_tokens"],
                repeat=repeat,
                batch=size,
                tensors=None,
                forward_calls=0,
                sampler_calls=0,
            )
            self.cohorts[cohort] = state
        position, episode = state["position"], state["episode"]
        # input_tokens counts the prompt. Native decode preparation has already
        # appended the previous output token to the KV sequence, including the
        # saved bootstrap token on the first restored step.
        expected = [n + position for n in episode["input_tokens"]]
        if (
            keys != episode["keys"]
            or batch.seq_lens_cpu.tolist() != expected
            or position >= self.trace.data["output_tokens"]
        ):
            raise ValueError("Plain replay request order or position differs")
        state["forward_calls"] += 1
        self.active = cohort, state, False
        return True

    def sample(self, result, sampling_info, return_logprob):
        if self.active is None:
            return result
        cohort, state, sampled = self.active
        episode = state["episode"]
        if (
            sampled
            or not sampling_info.is_all_greedy
            or return_logprob
            or sampling_info.grammars
            or sampling_info.has_custom_logit_processor
            or sampling_info.sampling_mask_batch_indices is not None
            or result.ndim != 1
            or result.numel() != len(episode["keys"])
            or result.dtype not in (self.torch.int32, self.torch.int64)
        ):
            raise ValueError("Plain replay requires one original greedy sampler call")
        if state["tensors"] is None:
            state["tensors"] = self.torch.tensor(
                [episode["expected_output_ids"][key] for key in episode["keys"]],
                dtype=result.dtype,
                device=result.device,
            ).T.contiguous()
        tokens = state["tensors"][state["position"]]
        state["position"] += 1
        state["sampler_calls"] += 1
        self.active = cohort, state, True
        return tokens

    def end(self):
        if self.active is None or not self.active[2]:
            raise ValueError("Plain replay batch omitted its original sampler")
        self.active = None

    def completed(self, cohort):
        return self.cohorts[cohort]["position"] == self.trace.data["output_tokens"]

    def receipt(self, cohort):
        state = self.cohorts[cohort]
        return dict(
            trace_sha256=self.trace.sha256,
            attention_dp=self.attention_dp,
            rank=self.rank,
            repeat=state["repeat"],
            batch=state["batch"],
            frames=state["sampler_calls"],
            expected_frames=self.trace.data["output_tokens"]
            - state["episode"]["saved_bootstrap_tokens"],
            forward_calls=state["forward_calls"],
            model_calls=state["forward_calls"],
            sampler_calls=state["sampler_calls"],
            complete=self.completed(cohort),
            forced_acceptance=False,
            natural_accuracy_claimed=False,
        )


def processor_string(native_processor_base):
    """Serialize through the caller's native framework processor implementation."""

    class FixedOutputProcessor(native_processor_base):
        def __call__(self, logits, custom_param_list):
            import torch

            if logits.ndim != 2 or len(custom_param_list) != logits.shape[0]:
                raise ValueError("Fixed-output sampler batch differs")
            targets = []
            for params in custom_param_list:
                ids = params["replay_ids"]
                request = params.get("__req__")
                position = params.get("replay_position")
                if position is None and request is not None:
                    position = len(request.output_ids)
                if type(position) is not int or not 0 <= position < len(ids):
                    raise ValueError(
                        "Fixed-output position exceeds the supplied history"
                    )
                token = ids[position]
                if type(token) is not int or not 0 <= token < logits.shape[1]:
                    raise ValueError("Fixed-output token exceeds the vocabulary")
                targets.append(token)
            logits.fill_(-float("inf"))
            selected = torch.tensor(targets, device=logits.device, dtype=torch.long)
            logits.scatter_(1, selected[:, None], 0.0)
            return logits

    return FixedOutputProcessor.to_str()
