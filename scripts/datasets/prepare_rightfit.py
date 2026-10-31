"""Freeze all42 selected complete published main conversations; retain cut metadata.

Download pinned public shards or verify extracted records. Outputs use public reasoning,
content and tool-call lengths, never aggregate usage or teacher output IDs.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import shutil
import uuid
from urllib.request import urlopen

os.environ["TOKENIZERS_PARALLELISM"] = "false"

REV = "92ffbd7daf5ce7bc4b1e996e962a46ca52876820"
TOKENIZER_SHA = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
CUT = re.compile(r"\[\.\.\.\s*(\d+)\s*characters truncated\s*\.\.\.\]", re.I)
BOUNDARY = re.compile(r"context_(?:prune|compaction(?!_failed))|session_restart|context_reset|partial_reconstruction")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def text(value):
    return value if isinstance(value, str) else "" if value is None else compact(value)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def render(messages):
    result, previous = [], None
    for message in messages:
        role, content = message["role"], message.get("content")
        if not isinstance(content, str):
            raise ValueError("Non-string input content")
        if role in ("system", "user"):
            result.append("<|" + role + "|>" + content)
        elif role == "tool":
            if previous != "tool":
                result.append("<|observation|>")
            result.append("<tool_response>" + content + "</tool_response>")
        else:
            raise ValueError(("Unexpected input role", role))
        previous = role
    return "".join(result)



def selected_lines(args, wanted, keys, receipt):
    """Fetch immutable public shards, or verify previously extracted records."""
    if args.records_root is not None:
        for key in sorted(keys):
            identifier = "rightfit-whole-" + sha(key.encode())[:24]
            raw = (args.records_root / (identifier + ".json")).read_bytes()
            if sha(raw) != wanted[key]["source_caveats"]["record_sha256"]:
                raise ValueError("Extracted source record checksum differs")
            yield raw
        return
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir / (receipt["sha256"] + ".jsonl.gz")
    if not cache.exists():
        url = receipt["url"]
        expected = "https://huggingface.co/datasets/yixuanli97/finding-the-right-fit/resolve/" + REV + "/"
        if not url.startswith(expected):
            raise ValueError("Source URL is not the pinned public dataset revision")
        partial = cache.with_name(cache.name + "." + uuid.uuid4().hex + ".partial")
        with urlopen(url, timeout=60) as response, partial.open("xb") as target:
            shutil.copyfileobj(response, target)
        if sha(partial.read_bytes()) != receipt["sha256"]:
            raise ValueError("Downloaded source shard checksum differs; partial file retained")
        partial.rename(cache)
    if sha(cache.read_bytes()) != receipt["sha256"]:
        raise ValueError("Cached source shard checksum differs")
    for line in gzip.open(cache, "rb"):
        record = json.loads(line)
        key = "/".join(record[k] for k in ("benchmark", "harness", "model", "task_id"))
        if key in keys:
            yield line

def main(args):
    from tokenizers import Tokenizer
    OUT = args.out
    OUT.mkdir(exist_ok=False, parents=True)
    tokenizer_path = args.tokenizer
    assert sha(tokenizer_path.read_bytes()) == TOKENIZER_SHA
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encode = lambda value: tokenizer.encode(value, add_special_tokens=False).ids
    count = lambda value: len(encode(value))
    review_path = args.selection
    selection = json.loads(review_path.read_bytes())
    assert selection["revision"] == REV and selection["tokenizer_sha256"] == TOKENIZER_SHA
    full42 = selection["conversations"]
    assert len(full42) == 42
    save(OUT / "WHOLE42_SOURCE_COMPARISON.json", full42)
    wanted = {row["key"]: row for row in full42}
    receipts, by_shard = {}, collections.defaultdict(set)
    for row in full42:
        source = row["source"]
        source_path = source["shard"]
        receipt = dict(sha256=source["shard_sha256"], url=source["url"])
        if source_path in receipts:
            assert receipts[source_path] == receipt
        receipts[source_path] = receipt
        by_shard[source_path].add(row["key"])
    traces, features, rejected, sources = [], [], [], []
    for source_path, keys in sorted(by_shard.items()):
        receipt = receipts[source_path]
        for line in selected_lines(args, wanted, keys, receipt):
            record = json.loads(line)
            key = "/".join(record[k] for k in ("benchmark", "harness", "model", "task_id"))
            if key not in keys:
                continue
            selected_row = wanted[key]
            assert sha(line) == selected_row["source_caveats"]["record_sha256"]
            identifier = "rightfit-whole-" + sha(key.encode())[:24]
            original = OUT / "source_records" / (identifier + ".json")
            original.parent.mkdir(exist_ok=True)
            original.write_bytes(line)
            events = record.get("harness_events") or []
            boundaries = [event for event in events if BOUNDARY.search(event.get("type", ""))]
            cuts = []
            for i, message in enumerate(record["messages"]):
                for field in ("content", "reasoning", "tool_calls"):
                    for match in CUT.finditer(text(message.get(field))):
                        cuts.append(dict(message_index=i, role=message["role"], field=field,
                                         omitted_characters=int(match[1])))
            # Cuts/context-event metadata never remove calls or reselect this frozen42.
            pending, turns, inputs, prompts = [], [], [], []
            occupied = 0
            for message_index, message in enumerate(record["messages"]):
                if message["role"] != "assistant":
                    pending.append(message)
                    continue
                components = {field: count(text(message.get(field))) for field in ("reasoning", "content")}
                components["tool_calls"] = count(compact(message["tool_calls"])) if message.get("tool_calls") else 0
                output = sum(components.values())
                if output <= 1:
                    raise ValueError(("Output0/1 must not be padded", key, message_index, output))
                first = not turns
                frame = ("[gMASK]<sop><|system|>Reasoning Effort: Max" if first else "") + render(pending) + "<|assistant|><think>"
                ids = encode(frame)
                occupied += len(ids)
                prompts.append(occupied)
                turns.append(dict(append_input_ids=[] if first else ids, output_tokens=output,
                    source_input_tokens=occupied, source_message_index=message_index,
                    source_call_index=len(turns), reference_length_annotation=dict(
                        visible_components=components, budget_basis="GLM tokenizer sum over all available published reasoning/content/compact tool_calls",
                        provider_native_API_usage=False, publication_truncation=bool(any(c["message_index"] == message_index for c in cuts)))))
                inputs.append(len(ids))
                occupied += output
                if first:
                    initial = ids
                pending = []
            outputs = [turn["output_tokens"] for turn in turns]
            assert len(turns) == selected_row["turns"]
            assert sum(outputs) == selected_row["output_budget_sum"]
            assert sum(inputs) == selected_row["new_input_sum"]
            assert occupied == selected_row["final_with_frames"]
            assert 262144 <= occupied <= 524288 and statistics.mean(outputs) > 2048
            assert 2 * sum(outputs) >= occupied
            source = dict(dataset="yixuanli97/finding-the-right-fit", revision=REV,
                shard=source_path, shard_sha256=receipt["sha256"], record_sha256=sha(line),
                url=f"https://huggingface.co/datasets/yixuanli97/finding-the-right-fit/resolve/{REV}/{source_path}",
                model=record["model"], model_id=record["model_id"], benchmark=record["benchmark"],
                task_id=record["task_id"], harness=record["harness"], original_source=record.get("source"))
            reasoning_calls = sum(turn["reference_length_annotation"]["visible_components"]["reasoning"] > 0 for turn in turns)
            reasoning_tokens = sum(turn["reference_length_annotation"]["visible_components"]["reasoning"] for turn in turns)
            has_system = any(message["role"] == "system" for message in record["messages"])
            tail_tokens = sum(count(text(message.get("content"))) for message in pending)
            trace = dict(key=identifier, episode_id=identifier, whole_original_episode=True,
                whole_observed_published_main_call_chain=True, start_call=0,
                trace_format="rightfit-whole42-public-output-length-v1",
                initial_input_ids=initial, turns=turns, source=source,
                source_assistant_calls_total=len(turns), source_message_count=len(record["messages"]),
                source_first_assistant_message=turns[0]["source_message_index"],
                source_last_assistant_message=turns[-1]["source_message_index"],
                source_harness_events=events, source_context_boundary_events=boundaries,
                context_reset_or_concatenation=False,
                subagents_merged=False, excluded_subagent_conversations=len(record.get("subagents") or []),
                teacher_output_ids_used=False, teacher_history_used=False,
                provider_per_call_lengths_found=0, publication_truncations=cuts,
                source_system_present=has_system,
                source_system_inferred=any("system_prompt_inferred" in event.get("type", "") for event in events),
                source_terminal_unanswered_messages=pending,
                source_terminal_unanswered_body_tokens=tail_tokens,
                terminal_input_not_fabricated_as_generation=True,
                public_output_fields_complete_without_cut_marker=not cuts,
                original_provider_hidden_reasoning_completeness_verified=False)
            traces.append(trace)
            feature = dict(key=identifier, model=record["model"], model_id=record["model_id"],
                task=record["task_id"], benchmark=record["benchmark"], harness=record["harness"],
                calls=len(turns), first_input_tokens=inputs[0], input_total=sum(inputs),
                new_input_including_first_mean=sum(inputs)/len(turns), later_input_mean=statistics.mean(inputs[1:]),
                output_total=sum(outputs), output_mean=statistics.mean(outputs), output_min=min(outputs),
                output_max=max(outputs), final_sequence=occupied, output_share_including_first=sum(outputs)/occupied,
                mean_request_prompt_tokens=statistics.mean(prompts), source_system_present=has_system,
                inferred_system=trace["source_system_inferred"], terminal_unanswered_messages=len(pending),
                terminal_unanswered_body_tokens=tail_tokens, visible_reasoning_tokens=reasoning_tokens,
                calls_with_reasoning=reasoning_calls, calls_without_separate_reasoning=len(turns)-reasoning_calls,
                source_record_sha256=sha(line), source_publication_cut_count=len(cuts), original_per_call_API_usage_available=False)
            features.append(feature)
            sources.append(dict(source, local_record=str(original.relative_to(OUT))))
    traces.sort(key=lambda row: row["key"])
    features.sort(key=lambda row: row["key"])
    raw = "".join(compact(row) + "\n" for row in traces).encode()
    assert len(traces) == 42
    if sha(raw) != selection["expected_trace_sha256"]:
        raise ValueError("Rebuilt tokenized trace differs from the pinned paper workload")
    (OUT / "traces.jsonl").write_bytes(raw)
    with (OUT / "SELECTED_WHOLE_CONVERSATIONS.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(features[0]))
        writer.writeheader(); writer.writerows(features)
    save(OUT / "PER_CONVERSATION_STATISTICS.json", features)
    save(OUT / "SOURCE_RECORD_MANIFEST.json", sources)
    total_calls = sum(row["calls"] for row in features)
    total_input = sum(row["input_total"] for row in features)
    total_output = sum(row["output_total"] for row in features)
    manifest = dict(independent_original_conversations=len(traces), input_candidate_count=42,
        rejected_after_full_field_recheck=rejected, trace_sha256=sha(raw), trace_bytes=len(raw),
        tokenizer_sha256=TOKENIZER_SHA, source_revision=REV,
        source_review_sha256=sha(review_path.read_bytes()), exporter_sha256=sha(Path(__file__).read_bytes()),
        calls=total_calls, original_per_call_API_usage_available=False,
        output_policy="Original public available reasoning+content+toolcall lengths re-tokenized with current GLM tokenizer; not provider API usage or text-faithful generated IDs.",
        input_policy="Original public nonassistant message deltas with GLM role/tool framing; each source first-to-last assistant call exactly once; actual natural GLM outputs enter later history.",
        no_source_output_ID_replay=True, no_source_assistant_history_replay=True,
        no_context_reset_or_concat=True, source_input_cut_rows=0,
        source_output_cut_rows=sum(row["source_publication_cut_count"] > 0 for row in features),
        cuts_are_metadata_not_an_exclusion=True,
        no_output0or1=True, output_share_includes_first_input=True,
        observed_public_reasoning_only=True, hidden_provider_reasoning_completeness_not_claimed=True,
        terminal_unanswered_input_policy="Preserved in source+trace metadata, not made into a fictional LLM call or counted as allocated resident KV.",
        mean_first_input_tokens=statistics.mean(row["first_input_tokens"] for row in features),
        turn_weighted_new_input_including_first=total_input/total_calls,
        turn_weighted_output=total_output/total_calls,
        mean_final_sequence=statistics.mean(row["final_sequence"] for row in features),
        min_final_sequence=min(row["final_sequence"] for row in features),
        max_final_sequence=max(row["final_sequence"] for row in features),
        total_output_share=total_output/(total_input+total_output),
        min_per_conversation_output_share=min(row["output_share_including_first"] for row in features),
        missing_system_conversations=sum(not row["source_system_present"] for row in features),
        terminal_unanswered_conversations=sum(row["terminal_unanswered_messages"] > 0 for row in features),
        separate_reasoning_absent_conversations=sum(row["visible_reasoning_tokens"] == 0 for row in features),
        models=dict(collections.Counter(row["model"] for row in features)),
        unique_tasks=len({(row["benchmark"], row["task"]) for row in features}),
        source_validation_scope=("Pinned individual record hashes" if args.records_root else "Pinned shard and record hashes"),
        static_recorded_input_observations=True, tool_execution_replayed=False,
        formal_benchmark_started=False, numerical_gate_not_relaxed=True,
        HTTP_requests=0, GPU_actions=0)
    save(OUT / "MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True,
                        help="External workload-selection JSON")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--records-root", type=Path,
        help="Optional exact extracted-record cache; every record checksum is verified")
    main(parser.parse_args())
