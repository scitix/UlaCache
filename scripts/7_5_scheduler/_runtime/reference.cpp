// SPDX-License-Identifier: Apache-2.0
// CPU policy replay only. Its runtime is never reported as GPU kernel latency.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace {
struct Policy {
  int logical, slots, head, tail, mode;
  std::vector<int> tags, location, previous, next, order, buffer;
  Policy(int l, int g, const int* seed, const int* ages_order, int m)
      : logical(l), slots(g), mode(m), tags(seed, seed + g), location(l, -1),
        previous(g), next(g), order(ages_order, ages_order + g), buffer(g) {
    for (int i = 0; i < g; ++i) {
      assert(tags[i] < l);
      if(tags[i]>=0){assert(tags[i]>0&&location[tags[i]]==-1);location[tags[i]]=i;}
      const int slot = order[i];
      assert(slot >= 0 && slot < g);
      previous[slot] = i ? order[i - 1] : -1;
      next[slot] = i + 1 < g ? order[i + 1] : -1;
    }
    head = order.front(); tail = order.back();
  }
  void touch(int slot) {
    if (tail == slot) return;
    const int left = previous[slot], right = next[slot];
    if (left >= 0) next[left] = right; else head = right;
    if (right >= 0) previous[right] = left;
    next[tail] = slot; previous[slot] = tail; next[slot] = -1; tail = slot;
  }
  void replace(int slot, int uid) {
    assert(location[uid] < 0);
    if(tags[slot]>=0)location[tags[slot]] = -1;
    location[uid] = slot; tags[slot] = uid;
  }
  void update(const std::vector<int>& unique, const std::vector<int>& mark,
              int stamp) {
    std::vector<int> misses;
    for (int uid : unique) {
      const int slot = location[uid];
      if (slot < 0) misses.push_back(uid);
      else if (mode == 0) touch(slot);
    }
    if (mode == 0) {
      // Identical to linked_list_lru.cuh: UID-sorted hits, then UID-sorted misses.
      for (int uid : misses) {
        const int slot = head;
        assert(tags[slot]<0||mark[tags[slot]] != stamp);
        replace(slot, uid); touch(slot);
      }
    } else {
      // HiSparse stable partition: stale survivors, new misses, old-order hits.
      int nonhits = 0, hits = 0;
      for (int slot : order) {
        if (tags[slot]>=0&&mark[tags[slot]] == stamp) buffer[slots - 1 - hits++] = slot;
        else buffer[nonhits++] = slot;
      }
      const int count = int(misses.size());
      assert(count <= nonhits);
      for (int i = 0; i < count; ++i) replace(buffer[i], misses[i]);
      int pos = 0;
      for (int i = count; i < nonhits; ++i) order[pos++] = buffer[i];
      for (int i = 0; i < count; ++i) order[pos++] = buffer[i];
      for (int i = slots - 1; i >= nonhits; --i) order[pos++] = buffer[i];
      assert(pos == slots);
    }
    for (int uid : unique) assert(location[uid] >= 0);
  }
};
struct Replay {
  int logical, stamp = 0;
  std::vector<Policy> policies;
  std::vector<int> mark, current_mark, selected_mark, unique, current;
  Replay(int l, int g, const int* seed, const int* order)
      : logical(l), mark(l), current_mark(l), selected_mark(l) {
    policies.emplace_back(l, g, seed, order, 0);
    policies.emplace_back(l, g, seed, order, 1);
  }
  void step(const uintptr_t* pointers, const int* offsets, int ranks,
            int raw_per_rank, const int64_t* writers, int writer_count,
            int64_t* output) {
    ++stamp; unique.clear(); current.clear();
    std::fill(output, output + 2 * 13, 0);
    for (int i = 0; i < writer_count; ++i) {
      const int uid = int(writers[i]); assert(uid > 0 && uid < logical);
      if (current_mark[uid] != stamp) {
        current_mark[uid] = stamp; current.push_back(uid);
      }
    }
    for (int rank = 0; rank < ranks; ++rank) {
      const auto* raw = reinterpret_cast<const int32_t*>(pointers[rank]);
      for (int i = 0; i < raw_per_rank; ++i) {
        const int uid = raw[i] + offsets[rank];
        assert(uid > 0 && uid < logical);
        selected_mark[uid] = stamp;
        if (mark[uid] != stamp) { mark[uid] = stamp; unique.push_back(uid); }
        for (int method = 0; method < 2; ++method) {
          int64_t* out = output + 13 * method;
          const bool hot = policies[method].location[uid] >= 0;
          ++out[0]; out[1] += hot;
          if (current_mark[uid] == stamp) { ++out[2]; out[3] += hot; }
        }
      }
    }
    for (int uid : current)
      if (mark[uid] != stamp) { mark[uid] = stamp; unique.push_back(uid); }
    std::sort(unique.begin(), unique.end());
    for (int uid : unique) {
      for (int method = 0; method < 2; ++method) {
        int64_t* out = output + 13 * method;
        const bool hot = policies[method].location[uid] >= 0;
        ++out[8]; out[9] += hot;
        if (current_mark[uid] == stamp) {
          ++out[6]; out[7] += hot;
          if (selected_mark[uid] == stamp) { ++out[11]; out[12] += hot; }
        } else { ++out[4]; out[5] += hot; }
      }
    }
    for (Policy& policy : policies) policy.update(unique, mark, stamp);
  }
};
}
extern "C" {
void* make_replay(int logical, int slots, const int* seed, const int* order) {
  return new Replay(logical, slots, seed, order);
}
void replay_step(void* replay, const uintptr_t* raw, const int* offsets,
                 int ranks, int raw_per_rank, const int64_t* current,
                 int writer_count, int64_t* counts) {
  static_cast<Replay*>(replay)->step(raw, offsets, ranks, raw_per_rank,
                                     current, writer_count, counts);
}
void replay_order(void* replay, int method, int* ids) {
  const auto& policy = static_cast<Replay*>(replay)->policies[method];
  if (method == 0) {
    int pos = 0;
    for (int slot = policy.head; slot != -1; slot = policy.next[slot])
      ids[pos++] = policy.tags[slot];
    assert(pos == policy.slots);
  } else {
    for (int i = 0; i < policy.slots; ++i) ids[i] = policy.tags[policy.order[i]];
  }
}
void delete_replay(void* replay) { delete static_cast<Replay*>(replay); }
}
