"""Inspect and verify portable request KV snapshots without loading a model."""

import argparse
import json
from .snapshot.format import SnapshotReader, finalize_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("inspect", "verify", "commit"))
    parser.add_argument("snapshot")
    args = parser.parse_args()
    if args.operation == "commit":
        value = finalize_snapshot(args.snapshot)
    else:
        reader = SnapshotReader(args.snapshot)
        if args.operation == "verify":
            value = reader.verify()
        else:
            value = {
                "layout": reader.metadata["layout"],
                "requests": [
                    {
                        "key": request.key,
                        "input_tokens": len(request.input_ids),
                        "computed_kv_tokens": request.next_position,
                        "next_token_id": request.next_token_id,
                        "generated_tokens": len(request.generated_ids),
                    }
                    for request in reader.requests
                ],
            }
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
