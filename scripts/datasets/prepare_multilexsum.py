"""Reconstruct a frozen Multi-LexSum paper workload."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import urllib.request
import uuid
from collections.abc import Mapping

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def native_chat_ids(tokenizer, text, template_options=None):
    encoded = tokenizer.apply_chat_template([dict(role="user", content=text)],
                                           tokenize=True, add_generation_prompt=True, **(template_options or {}))
    # GLM's native template requests a BatchEncoding in current Transformers.
    # Never confuse the number of dictionary fields with the token count.
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if len(ids) == 1 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Native chat template did not produce one nonempty token sequence")
    return list(ids)


def main(args):
    profile_path = args.selection
    profile = json.loads(profile_path.read_bytes())
    out, cache = args.out, args.cache
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    for name, binding in profile['source_files'].items():
        path = cache / name
        if not path.exists():
            partial = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
            print(json.dumps(dict(stage='download', name=name)), flush=True)
            with urllib.request.urlopen(binding['url'], timeout=60) as response, partial.open('xb') as target:
                for block in iter(lambda: response.read(8 << 20), b''):
                    target.write(block)
            if digest(partial) != binding['sha256']:
                raise ValueError('Source download checksum differs; partial retained: ' + name)
            partial.rename(path)
        if path.stat().st_size != binding['bytes'] or digest(path) != binding['sha256']:
            raise ValueError('Pinned source changed: ' + name)
    actual_tokenizer = {}
    for name, expected in profile['tokenizer_files'].items():
        actual_tokenizer[name] = digest(args.tokenizer / name)
        allowed = expected if isinstance(expected, list) else [expected]
        if actual_tokenizer[name] not in allowed:
            raise ValueError('Pinned tokenizer changed: ' + name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    requested = {row['key']: row for row in profile['requests']}
    cases = {}
    for split in ('train', 'dev', 'test'):
        with (cache / (split + '.json')).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['case_id'] in requested:
                    if row['case_id'] in cases:
                        raise ValueError('Duplicate selected case identity')
                    cases[row['case_id']] = (split, row)
    if cases.keys() != requested.keys():
        raise ValueError('Pinned selected cases are missing')
    documents = json.loads((cache / 'sources.json').read_text())
    request_path = out / 'requests.jsonl'
    with request_path.open('x') as stream:
        for index, metadata in enumerate(profile['requests']):
            split, case = cases[metadata['key']]
            doc_ids = case['case_documents']
            if split != metadata['split'] or doc_ids != metadata['document_ids']:
                raise ValueError('Complete source document identities or order changed')
            parts = [f'=== Document {i+1} ===\n' + documents[key]['doc_text']
                     for i, key in enumerate(doc_ids)]
            text = profile['instruction'] + '\n\n'.join(parts)
            if hashlib.sha256(text.encode()).hexdigest() != metadata['prompt_text_sha256']:
                raise ValueError('Complete source prompt changed')
            ids = native_chat_ids(tokenizer, text, profile.get('template_options'))
            raw = json.dumps(ids, separators=(',', ':')).encode()
            if len(ids) != metadata['input_tokens'] or hashlib.sha256(raw).hexdigest() != metadata['input_ids_sha256']:
                raise ValueError('Native chat template tokenization differs from the paper')
            stream.write(json.dumps(dict(**metadata, input_ids=ids), separators=(',', ':')) + '\n')
            print(json.dumps(dict(stage='tokenize_selected', completed=index+1, total=len(profile['requests']))), flush=True)
    if digest(request_path) != profile['expected_requests_sha256']:
        raise ValueError('Rebuilt workload differs from the frozen paper request bytes')
    lengths = [r['input_tokens'] for r in profile['requests']]
    result = dict(complete=True, dataset=profile['dataset'], revision=profile['revision'],
        release=profile['release'], source_files=profile['source_files'],
        tokenizer_files=actual_tokenizer, sampling_seed=profile['sampling_seed'],
        count=len(lengths), length=dict(min=min(lengths), max=max(lengths),
            mean=statistics.mean(lengths), median=statistics.median(lengths)),
        instruction=profile['instruction'], requests_file='requests.jsonl',
        requests_sha256=digest(request_path), requests=profile['requests'],
        selection_manifest_sha256=digest(profile_path), exporter_sha256=digest(__file__),
        source_transform='Complete original case documents; no truncation/padding/repetition',
        selection_policy='Frozen paper sample; only selected cases are tokenized',
        generation=dict(temperature=0, ignore_eos=True, max_new_tokens=2048,
            draft_steps=2, verify_tokens=3, eagle_topk=1, forced_output_ids=False),
        GPU_actions=0)
    (out / 'MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(complete=True, requests_sha256=result['requests_sha256'],
        count=result['count'], length=result['length'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True, help='External workload-selection JSON')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
