"""Report unique KV capacity from actual startup inventories; no model launch."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize(directory, mode, world_size):
    paths = sorted(directory.glob("target.rank*.json"))
    if len(paths) != world_size:
        raise ValueError(
            f"Expected {world_size} target-rank inventories in {directory}"
        )
    inventories = [json.loads(path.read_bytes()) for path in paths]
    logical = [row["pool_fields"]["logical_allocator_size"] for row in inventories]
    gpu, cpu = [], []
    for row in inventories:
        fields = row["pool_fields"]
        distribution = row.get("record_distribution") or {}
        if distribution.get("kind") == "owner_striped_records":
            owners = distribution["owner_count"]
            reserved = distribution["allocator_records_per_page"]
            if reserved % owners:
                raise ValueError("Reserved owner superpage must divide across owners")
            # The first owner superpage is the native zero/sentinel page.
            gpu.append(distribution["gpu_cold_record_capacity_per_owner"]
                       - reserved // owners)
            cpu.append(distribution["cpu_cold_record_capacity_per_owner"])
        elif distribution.get("kind") == "native_sparse_cache_partitions":
            partitions = distribution["required_partitions"]
            if len(partitions) != 1:
                raise ValueError("Report heterogeneous required partitions separately")
            part = next(iter(partitions.values()))
            gpu.append(part["device_record_capacity_per_owner"]
                       - part["reserved_logical_record_count"])
            cpu.append(part["host_record_capacity_per_owner"])
        elif "hisparse_device_buffer_size" in fields:
            # The device buffer caches the same CPU records; it adds no unique IDs.
            gpu.append(0)
            cpu.append(fields["logical_allocator_size"])
        else:
            gpu.append(fields["logical_allocator_size"])
            # Native HiCache CPU records are backup copies of GPU-addressable IDs.
            cpu.append(0)
    unique_owners = bool(inventories[0].get("record_distribution"))
    combine = sum if mode == "dpa" or unique_owners else min
    usable = sum(logical) if mode == "dpa" else min(logical)
    if min(gpu + cpu) < 0 or combine(gpu) + combine(cpu) != usable:
        raise ValueError("Unique GPU/CPU capacities do not match the native allocator")
    return dict(
        usable_history_slots=usable,
        unique_gpu_record_slots=combine(gpu),
        unique_cpu_record_slots=combine(cpu),
        native_request_slots_per_rank=[
            row["pool_fields"]["native_request_slots"] for row in inventories
        ],
        sources=[
            dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in paths
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", required=True, metavar="METHOD=AUDIT_DIRECTORY"
    )
    parser.add_argument("--mode", choices=("tp", "dpa"), required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for value in args.input:
        method, separator, path = value.partition("=")
        if not separator or not method or method in results:
            parser.error("Use a unique METHOD=AUDIT_DIRECTORY for each input")
        results[method] = summarize(Path(path), args.mode, args.world_size)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "capacity.json").write_text(json.dumps(results, indent=2) + "\n")
    with (args.out / "capacity.csv").open("w", newline="") as stream:
        columns = [
            "method",
            "usable_history_slots",
            "unique_gpu_record_slots",
            "unique_cpu_record_slots",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for method, row in results.items():
            writer.writerow(
                dict(method=method, **{key: row[key] for key in columns[1:]})
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
