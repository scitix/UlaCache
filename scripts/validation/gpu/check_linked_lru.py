"""Check empty, partial and full linked-LRU pools against a CPU reference."""

import json


def main():
    import torch
    from sglang.kernels.jit.utils import load_jit
    from ulacache.compiler import configure_compiler
    from ulacache.cache.kernels.cpu_residency import (
        create_host_tier_state,
        create_host_tier_workspace,
        commit_host_tier,
        release_host_tier_leases,
    )
    from ulacache.eviction.kernels.linked_list_lru import create_workspace, prepare

    configure_compiler(load_jit)
    gpu_slots, host_slots, records = 16, 64, 40
    transactions = [
        ([1, 1, 2, 3], 20),
        ([2, 4, 20, 21], 22),
        ([1, 5, 23, 24], 25),
        ([3, 6, 26, 27], 28),
        ([4, 7, 29, 30], 31),
        ([1, 2, 3, 4], 32),
    ]
    results = []
    for initial_hot in (0, 3, gpu_slots):
        source = create_host_tier_state(gpu_slots, host_slots, "cpu")
        source.epoch.fill_(1)
        source.access_time[1 : records + 1] = 1
        hot = torch.arange(1, initial_hot + 1)
        cold = torch.arange(initial_hot + 1, records + 1)
        source.gpu_tags[:initial_hot] = hot.int()
        source.directory[hot] = torch.arange(initial_hot)
        source.host_tags[: cold.numel()] = cold.int()
        source.directory[cold] = -torch.arange(cold.numel()) - 2
        source.gpu_free[: gpu_slots - initial_hot] = torch.arange(
            initial_hot, gpu_slots
        )
        source.host_free[: host_slots - cold.numel()] = torch.arange(
            cold.numel(), host_slots
        )
        source.free_counts[:] = torch.tensor(
            [gpu_slots - initial_hot, host_slots - cold.numel()]
        )
        state = type(source)(*(value.cuda() for value in source))
        work = create_host_tier_workspace(16, "cuda")
        scratch = create_workspace(state, 1, 4)
        expected_tags = source.gpu_tags.tolist()
        order = list(range(initial_hot, gpu_slots)) + list(range(initial_hot))
        free = list(range(initial_hot, gpu_slots))
        for selected, current in transactions:
            unique = sorted(set(selected + [current]))
            misses = []
            for row in unique:
                if row in expected_tags:
                    slot = expected_tags.index(row)
                    order.remove(slot)
                    order.append(slot)
                else:
                    misses.append(row)
            used = min(len(misses), len(free))
            destinations = free[len(free) - used :] if used else []
            if used:
                del free[-used:]
            for index, row in enumerate(misses):
                slot = destinations[index] if index < used else order[0]
                assert expected_tags[slot] not in unique
                expected_tags[slot] = row
                order.remove(slot)
                order.append(slot)
            prepare(
                torch.tensor([selected], dtype=torch.int32, device="cuda"),
                torch.tensor([current], dtype=torch.int64, device="cuda"),
                torch.ones(1, dtype=torch.int32, device="cuda"),
                0,
                state,
                work,
                scratch,
            )
            commit_host_tier(state, work)
            release_host_tier_leases(state, work)
            torch.cuda.synchronize()
            actual = state.gpu_tags.cpu().tolist()
            assert actual == expected_tags, (
                initial_hot,
                selected,
                current,
                actual,
                expected_tags,
            )
            locations = state.directory.cpu().tolist()
            gpu_tags, cpu_tags = actual, state.host_tags.cpu().tolist()
            assert int(state.free_counts[0]) == gpu_tags.count(-1)
            assert int(state.free_counts[1]) == cpu_tags.count(-1)
            for row in range(1, records + 1):
                location = locations[row]
                assert location != -1
                assert (
                    gpu_tags[location] if location >= 0 else cpu_tags[-location - 2]
                ) == row
            assert not bool(state.leases.any())
        results.append(dict(initial_hot=initial_hot, transactions=len(transactions)))
    print(json.dumps(dict(passed=True, cases=results)))


if __name__ == "__main__":
    main()
