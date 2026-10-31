// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

namespace sglang::flowkv_host {
enum Counter { NewCount, PromoteCount, PlanCount, VictimCount, Candidates, LeaseCount, FreeUsed, PendingReserve };

struct State {
  int64_t* directory;
  int* gpu_tags;
  int* host_tags;
  int64_t* access_time;
  int* leases;
  int64_t* epoch;
  int* gpu_free;
  int* host_free;
  int* free_counts;
  int logical_slots, gpu_slots, host_slots;
};

struct Workspace {
  int* new_rows;
  int* promote_rows;
  int* plan;
  int* counters;
  int* histogram;
  int64_t* prefix;
  int* candidates;
  int* leased;
  int capacity;
};


}  // namespace sglang::flowkv_host
