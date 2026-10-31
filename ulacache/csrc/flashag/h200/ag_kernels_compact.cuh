// flash ag single-node NVLink ALL-GATHER kernels for 8xH200 (sm_90a), NVSwitch.
//
// Separate from ar_kernels_h200.cuh on purpose: an all-gather is not an all-reduce
// with the arithmetic removed, and the one thing that makes the all-reduce family
// work does not transfer.
//
// ⚠ THE SENTINEL PROTOCOL IS UNAVAILABLE HERE, and this is a semantic wall rather
// than an engineering one. The all-reduce family proves arrival with "the value is
// no longer the sentinel", which is only legal because `sanitize()` may rewrite an
// input pack that collides with the sentinel: the output is a SUM, the sentinel is a
// NaN, and IEEE does not preserve NaN payloads, so the rewrite is numerically free.
// An all-gather's output is a bit-exact COPY. Rewriting a pack is a wrong answer.
// Every bit pattern of bf16/fp32 is legal data, so no value can be reserved.
//
// So arrival is proved OUT OF BAND, by a monotone call sequence number travelling
// with the data. Three layouts, differing only in what fraction of the wire the tag
// costs, because that fraction is the whole trade:
//
//     kModeLL    16 B unit = 8 B payload + 8 B seq          50.0% efficient
//     kModeLine  128 B line = 7 data packs + 1 seq pack     87.5% efficient
//     kModeD31   128 B line = 31 data words + 1 seq word    96.9% efficient
//
// LL is for the latency end, where the measured baseline is FLAT from 1 KB to 64 KB
// (4.71 -> 4.83 us, output growing 64x) and bytes therefore cost nothing. D31 is for
// the bandwidth end, where 16 MB already runs at 76-81% of link and a 12.5% tag
// would eat the entire margin. Line sits between them and is the cheapest to index.
//
// WHY A SEQUENCE NUMBER AND NOT A COUNT. mc_red counts arrivals by letting the
// switch SUM a 1.0 from each rank; an all-gather has no arithmetic, so there is
// nothing to count into. A sequence number needs no accumulation and -- unlike the
// all-reduce family's reset word -- needs no reset either: slot rotation means a
// stale slot holds the value from kNumSlots calls ago, which never equals the
// current one. That removes the whole host/device agreement that P::kResetWord
// requires.
//
// ⚠ WHY 4 SLOTS IS ENOUGH, AND WHY IT ONLY JUST IS. "A stale slot holds the tag from
// kNumSlots calls ago" says nothing about how far apart the ranks can drift, and the
// number that matters is the LIVE WINDOW:
//
//   * rank A cannot COMPLETE call j until every rank has SENT call j -- that is what
//     the poll waits for;
//   * so A completing j implies all ranks have sent j;
//   * PDL lets call j+1's prologue run while j is still in flight
//     (`pdl_trigger_secondary`), so A may already be sending j+1 while finishing j;
//   * the slowest rank B is then at worst still receiving j-1.
//
// ⟹ the live window is the three consecutive slots {j-1, j, j+1}. With kNumSlots = 4
// the margin is ONE slot, not three. Anyone shrinking it to 2 to save slab -- the
// natural reasoning being "a rank can only be one call ahead" -- lands exactly on the
// PDL term. Pinned by
// test_num_slots_matches_the_device_and_leaves_room_for_pdl, which is where the
// derivation has to live because the constant itself is kc::kNumSlots -- owned by the
// ALL-REDUCE header and shared by both families, each with its own window argument.
//
// ⚠ CORRECTNESS RESTS ON 128 B MULTIMEM DELIVERY BEING ATOMIC for the two line
// modes: eight lanes' `multimem.st` coalesce into one 128 B transaction, and the
// receiver may only conclude "the data is here" from the seq slot if the whole line
// lands together. Measured on this machine at 47 M checks with zero tears, against a
// positive control that does tear, and 256 B tears about 1 line in 8700 -- so 128 B
// is exactly the unit, which is also why NCCL's protocol is LL128 and not LL256.
// kModeLL does not rely on it: a single 16 B store is atomic on its own.
//
// Machine constants, all measured here (see ar_kernels_h200.cuh's header):
// launch floor 0.733-1.267 us, cross-GPU RTT 1.438 us, flat 8-way barrier 0.793 us,
// per-channel 361/381/366 GB/s link-side.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// The raw accessors, PDL bracket, multimem.st and slot rotation are protocol
// plumbing with nothing all-reduce-specific about them, so they are shared rather
// than copied. Nothing else is taken from that file.
#include "csrc/flashag/h200/common.cuh"

namespace sglang {
namespace flash_ag_kernels_h200 {

namespace kc = ::ulacache_flashag::flash_ag_common_h200;

using kc::ld_u4;
using kc::mm_st_raw;
using kc::pdl_trigger_secondary;
using kc::pdl_wait_primary;
using kc::slot_of;
using kc::st_out;
using kc::st_u4;
using kc::take_call_index;
using kc::take_call_index_uniform;

constexpr int kNumSlots = kc::kNumSlots;
constexpr int kMaxRanks = kc::kMaxRanks;
constexpr int kMaxSpinThreads = kc::kMaxSpinThreads;

// The three tag layouts. Named rather than numbered at the call sites so a policy
// row reads as an algorithm instead of an integer.
constexpr int kModeLL = 0;
constexpr int kModeLine = 1;
constexpr int kModeD31 = 2;
// D31's layout, but the sender writes seven UNICAST copies instead of one
// multimem.st. See the send block for why that can win at the bandwidth end.
constexpr int kModeUniD31 = 3;

// PHASE-DECOMPOSITION PROBES, k_ag_probe only, never shipped. They exist because
// round 1 concluded "6 MB and up is pinned against the link" from the TOTAL time of
// a three-phase kernel, without ever measuring a phase on its own -- so "pinned
// against the link" and "pinned against the local copy-out" were indistinguishable.
// Each of these deletes exactly one phase; the differences localise the cost.
//
// ⚠ THEY DO NOT PRODUCE A CORRECT `out` (except kProbeLocalOnly, which produces a
// correct one without touching the wire). A sweep running them must skip the check.
constexpr int kProbeNoop = 4;       // counter + PDL only: the launch/replay floor
constexpr int kProbeSendOnly = 5;   // D31 send + local copy, nobody polls
constexpr int kProbeLocalOnly = 6;  // all of `out` written locally, no wire at all
constexpr int kProbeNoStore = 7;    // send + local + poll, but never store to `out`
// ⚠ These two exist because the first two round-2 fixes both FAILED. Cutting the
// spin's byte count 4x (sparse poll) and its attempt count (backoff) each made 16 MB
// slightly SLOWER, which falsifies "the spin's L2 traffic is the wall". So the next
// question is whether the receive costs anything at all beyond waiting: no-poll does
// the receive's entire memory walk with the spin deleted, recv-local does it with
// the send deleted too.
constexpr int kProbeNoPoll = 8;     // send + local + slab read + out store, NO spin
constexpr int kProbeRecvLocal = 9;  // the receive walk alone: no send, no spin
// ⚠ THE PRICE TAG ON "EXCLUDE THE SENDER'S OWN COPY", measured before building it.
// `multimem.st` on an 8-way team delivers to the sender too, so a rank's ingress is
// D rather than D(N-1)/N -- 17.32 MB instead of 15.16 at a 16 MiB output. Round 1
// tried to reclaim that with seven unicast stores (section 10.3), lost 64%, and
// concluded the byte was not worth chasing. The untried third form is a SEVEN-WAY
// MULTICAST TEAM: one instruction, switch still does the fan-out, no self copy.
//
// That needs driver-level multicast objects, so this row prices it first: the LAST
// rank stays silent, which removes exactly one eighth of every other rank's ingress
// while leaving each remaining sender's egress untouched. If the time falls by ~7/8
// the wall is ingress and the seven-way team is worth building; if it does not move,
// egress or issue rate is the wall and the whole idea is dead for one probe's cost.
constexpr int kProbeSend7 = 10;

// kModeLine: 8 packs per 128 B line, one of them the tag.
constexpr int kLinePacks = 8;
constexpr int kLineData = 7;
// kModeD31: 32 words per 128 B line, one of them the tag.
constexpr int kD31Words = 32;
constexpr int kD31Data = 31;

// Wire packs (16 B units) one source's `cm`-pack shard occupies under each mode.
// The HOST sizes the slab from this, so it has to be the same arithmetic on both
// sides -- ops/policy_h200.py mirrors it.
__host__ __device__ __forceinline__ size_t wire_packs(int mode, size_t cm) {
  if (mode == kModeLL) return 2 * cm;                                 // 8B payload/unit
  if (mode == kModeLine) return ((cm + kLineData - 1) / kLineData) * kLinePacks;
  // kModeUniD31 shares D31's wire layout exactly -- only the instruction differs.
  // D31: cm packs = 4*cm words, laid out 31 data words per 32-word line.
  const size_t w = 4 * cm;
  return ((w + kD31Data - 1) / kD31Data) * (kD31Words / 4);
}

// ---------------------------------------------------------------------------
// The tag
// ---------------------------------------------------------------------------
// ⚠ The tag must never be zero: a freshly allocated slab is zero, so a zero tag
// would read as "already arrived" on the very first call and hand the receiver
// garbage. `take_call_index` is 1-based, and folding it through this keeps it so.
// The fold also spreads the low bits, which matters because a stale slot holds the
// tag from exactly kNumSlots calls ago and adjacent integers would differ in one
// bit -- a single torn word could then alias the live value.
__device__ __forceinline__ uint32_t tag_of(uint32_t call_index) {
  uint32_t t = call_index * 2654435761u;  // Knuth's multiplicative hash
  return t | 1u;                          // never 0, and never even
}

// ---------------------------------------------------------------------------
// kModeLL: 8 B of payload and an 8 B tag in every 16 B wire unit
// ---------------------------------------------------------------------------
// The most bytes and the least machinery: one aligned 16 B store is atomic by
// itself, so there is no line to keep together and no assumption to make about the
// switch. At the latency end the 2x is free -- the measured baseline does not move
// from 1 KB to 64 KB, so nothing there is paying for bytes.
//
// The tag is duplicated into both halves of the 8 B so the receiver can test one
// 32-bit word and still catch a half-written unit.
__device__ __forceinline__ uint4 ll_pack(uint2 payload, uint32_t tag) {
  return make_uint4(payload.x, payload.y, tag, tag);
}

__device__ __forceinline__ bool ll_ready(uint4 d, uint32_t tag) {
  return d.z == tag && d.w == tag;
}

// ---------------------------------------------------------------------------
// Poll strategies
// ---------------------------------------------------------------------------
// Polling consumes memory bandwidth shared with incoming peer writes.
// The strategies below trade polling traffic against detection latency.
//
//   kPollDense   every lane reads its own 16 B cell, tag broadcast by shuffle.
//                One 128 B transaction per octet per attempt. What round 1 shipped.
//   kPollSparse  ONLY the octet's lane 0 reads, and only the 4 B tag word. One 32 B
//                sector per octet per attempt -- 4x less. Costs one extra local
//                read latency after arrival, which pipelines across lines.
//   kPollBackoff sparse, plus `nanosleep` between attempts. Trades detection
//                latency for a further cut in attempt count.
constexpr int kPollDense = 0;
constexpr int kPollSparse = 1;
constexpr int kPollBackoff = 2;

// Relaxed 4 B load, the sparse poll's probe. Same reasoning as ld_u4: the outer
// `asm volatile` is what stops the compiler hoisting it out of the spin.
__device__ __forceinline__ uint32_t ld_u32_relaxed(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void spin_backoff() {
#if __CUDA_ARCH__ >= 700
  __nanosleep(32);
#endif
}

// One octet waits for its 128 B line and returns the calling lane's own pack.
//
// ⚠ THE TAG WORD IS `d.x` OF LANE 0 in both line layouts -- kModeLine writes an
// all-tag pack at pos 0 and kModeD31 puts the tag in word 0, which is lane 0's
// first word. So the sparse form can probe a single `uint32_t` at the octet's base
// cell and never touch the other seven.
template <int kPoll, bool kRelaxPoll>
__device__ __forceinline__ uint4 octet_wait(const uint4* cell, int pos, int src0,
                                           unsigned omask, uint32_t tag) {
  if constexpr (kPoll == kPollDense) {
    while (true) {
      const uint4 d = ld_u4<kRelaxPoll>(cell);
      if (__shfl_sync(omask, d.x, src0) == tag) return d;
    }
  } else {
    // `cell - pos` is the octet's base, i.e. lane 0's cell, whose first word is the
    // tag. Every lane computes the same address; only lane 0 loads it.
    const uint32_t* const tw = (const uint32_t*)(cell - pos);
    uint32_t seen;
    do {
      uint32_t v = 0;
      if (pos == 0) v = ld_u32_relaxed(tw);
      // The shuffle is also the octet's convergence point, so no __syncwarp.
      seen = __shfl_sync(omask, v, src0);
      if constexpr (kPoll == kPollBackoff) {
        if (seen != tag) spin_backoff();
      }
    } while (seen != tag);
    return ld_u4<kRelaxPoll>(cell);
  }
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------
// SHAPE. Three phases, and only the middle one touches the wire:
//
//   send   our shard `in[0 .. cm)` -> the multicast alias of slice `rank`, tagged
//   local  our shard -> `out[shard_begin(rank) ..)`, a pure local copy the wire
//          never sees, so it is issued between the send and the receive where it
//          fills the round trip rather than extending it
//   recv   poll each remote slice, copy payload -> `out`
//
// ⚠ NO BARRIER ANYWHERE, and no publish phase either. The all-reduce family has to
// copy the payload into the slab before the switch can reduce it; an all-gather
// broadcasts straight out of registers loaded from `in`, so the prologue that
// a700/a701 exist to hide simply does not exist here.
//
// ⚠ THE GRID IS FIXED PER BAND. Slot rotation counts one bump per warp per call, so
// a grid-stride loop that re-maps units to threads when the grid changes breaks the
// guarantee that two uses of one slot are separated by another call. The adapter
// rejects any grid but the policy table's.
template<bool Ragged>
__device__ __forceinline__ size_t compact_input_word(size_t word, const int32_t* offsets,
                                                    size_t columns, int queries, bool dense) {
  if constexpr (!Ragged) return word;
  if (dense) return word;
  int lo = 0, hi = queries;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (word >= static_cast<size_t>(__ldg(offsets + mid + 1))) lo = mid + 1;
    else hi = mid;
  }
  return size_t(lo) * columns + word - static_cast<size_t>(__ldg(offsets + lo));
}

template <class P, int NG, int kMode, int kIlp, int kPoll, bool kStreamOut,
          bool kRelaxPoll, bool kLocalFirst, bool kDynamicCount = false, bool kRaggedInput = false>
__global__ void k_ag_mcst(const uint4* __restrict__ in, uint4* __restrict__ out,
                          uint4* uc, uint4* mc, void* const* peer_buf,
                          int rank, size_t n4, uint32_t* counters, size_t region,
                          size_t slot_stride, size_t slice, const int64_t* input_packs = nullptr,
                          const int32_t* query_offsets = nullptr, size_t columns = 0, int queries = 0) {
#if __CUDA_ARCH__ >= 900
  // Ahead of the PDL wait: none of it depends on `in`, so it overlaps the
  // producer's tail.
  const uint32_t call_index = take_call_index_uniform(counters);
  const uint32_t tag = tag_of(call_index);
  const size_t base = region + slot_of(call_index) * slot_stride;

  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  // ⚠ `in` HOLDS ONLY OUR OWN SHARD -- it is not the whole payload with our slice
  // inside it, which is what the all-reduce family's `in` is. Indexing it by the
  // output offset reads past its end; `out` is the only side that carries the
  // rank offset. Every read of `in` below is [0, cm).
  //
  // The split is always exact: all_gather_into_tensor requires equal inputs and
  // the adapter checks `obytes % (16 * world) == 0`, so there is no remainder to
  // spread and no uneven-shard machinery.
  // Prefix metadata is produced on the same stream before this collective.
  // Acquire it before deriving the identical live payload on every rank.
  if constexpr (kDynamicCount) pdl_wait_primary();
  const size_t cm = kDynamicCount ? static_cast<size_t>(*input_packs) : n4 / (size_t)NG;
  const size_t bm = (size_t)rank * cm;
  const bool dense = cm == n4 / (size_t)NG;

  if constexpr (!kDynamicCount) pdl_wait_primary();

  // ---- send -------------------------------------------------------------
  // `mc + base + rank * slice` is OUR slice of the all-gather region, addressed
  // through the multicast alias so one store lands in all N copies. Egress is
  // therefore cm packs, not (N-1)*cm -- the whole reason this is the right
  // primitive for an all-gather.
  uint4* const mine = mc + base + (size_t)rank * slice;

  if constexpr (kMode == kModeLL) {
    // Work in 8 B units: each becomes one 16 B wire unit.
    const uint2* const in8 = (const uint2*)in;
    const size_t c8 = 2 * cm;
    size_t j = tid;
    if constexpr (kIlp > 1) {
      for (; j + (size_t)(kIlp - 1) * stride < c8; j += (size_t)kIlp * stride) {
        uint2 v[kIlp];
#pragma unroll
        for (int u = 0; u < kIlp; ++u) v[u] = in8[j + (size_t)u * stride];
#pragma unroll
        for (int u = 0; u < kIlp; ++u)
          mm_st_raw(mine + j + (size_t)u * stride, ll_pack(v[u], tag));
      }
    }
    for (; j < c8; j += stride) mm_st_raw(mine + j, ll_pack(in8[j], tag));
  } else if constexpr (kMode == kModeLine) {
    // One octet of 8 consecutive lanes owns one 128 B line, so the eight stores
    // coalesce into the single transaction whose atomicity the receiver relies on.
    // pos 0 is the tag, pos 1..7 carry data.
    const int pos = (int)(tid & 7);
    const size_t lines = (cm + kLineData - 1) / kLineData;
    const size_t lstride = stride >> 3;
    const uint4 tagpack = make_uint4(tag, tag, tag, tag);
    for (size_t l = tid >> 3; l < lines; l += lstride) {
      uint4 v = tagpack;
      if (pos != 0) {
        const size_t d = l * kLineData + (size_t)(pos - 1);
        // Past the end of the shard the slot still has to be WRITTEN -- the
        // receiver reads the whole line and only skips the copy, so leaving it
        // stale is harmless, but leaving it unwritten is not: the tag would arrive
        // for a line whose other sectors were never sent this round.
        v = d < cm ? in[d] : tagpack;
      }
      mm_st_raw(mine + l * kLinePacks + pos, v);
    }
  } else if constexpr (kMode == kModeUniD31) {
    // ⚠ WHY UNICAST CAN BEAT A BROADCAST HERE, which is the opposite of the
    // all-reduce family's conclusion. `multimem.st` delivers to EVERY member of the
    // multicast group INCLUDING THE SENDER, so a rank's inbound is the whole D, not
    // D(N-1)/N -- it pays to receive its own shard back. Writing N-1 unicast copies
    // instead moves that byte off the inbound direction and onto the outbound one:
    //
    //     multimem.st   in D      = 16 MB   out D/N       =  2 MB
    //     unicast       in D(N-1)/N = 14 MB out D(N-1)/N  = 14 MB
    //
    // NVLink is full duplex, so the critical path is max(in, out) rather than the
    // sum: 14 MB against 16 MB, i.e. 12.5% -- which is exactly the margin band D is
    // missing. It costs N-1 stores per pack instead of one, and at these sizes that
    // is ~3.5 stores per thread, nowhere near instruction-bound.
    const uint32_t* const inw = (const uint32_t*)in;
    const int pos = (int)(tid & 7);
    const size_t nw = 4 * cm;
    const size_t lines = (nw + kD31Data - 1) / kD31Data;
    const size_t lstride = stride >> 3;
    for (size_t l = tid >> 3; l < lines; l += lstride) {
      uint4 v;
      uint32_t* vw = (uint32_t*)&v;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int f = pos * 4 + k;
        if (f == 0) {
          vw[k] = tag;
        } else {
          const size_t d = l * kD31Data + (size_t)(f - 1);
          vw[k] = d < nw ? inw[compact_input_word<kRaggedInput>(d, query_offsets, columns, queries, dense)] : tag;
        }
      }
      const size_t at = (size_t)rank * slice + l * (kD31Words / 4) + pos;
#pragma unroll
      for (int q = 0; q < NG; ++q) {
        if (q == rank) continue;  // our own copy is the local walk's job
        st_u4<true>((uint4*)peer_buf[q] + base + at, v);
      }
    }
  } else {
    // D31: the tag costs ONE 32-bit word instead of a whole pack, so the displaced
    // elements shift into the next line. Indices run in words and are no longer
    // 16 B aligned, which is what costs four scalar loads per pack -- paid back in
    // bytes, and at this end bytes are the whole cost.
    const uint32_t* const inw = (const uint32_t*)in;
    const int pos = (int)(tid & 7);
    const size_t nw = 4 * cm;
    const size_t lines = (nw + kD31Data - 1) / kD31Data;
    const size_t lstride = stride >> 3;
    for (size_t l = tid >> 3; l < lines; l += lstride) {
      uint4 v;
      uint32_t* vw = (uint32_t*)&v;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int f = pos * 4 + k;  // word position inside the line, 0..31
        if (f == 0) {
          vw[k] = tag;
        } else {
          const size_t d = l * kD31Data + (size_t)(f - 1);
          vw[k] = d < nw ? inw[compact_input_word<kRaggedInput>(d, query_offsets, columns, queries, dense)] : tag;
        }
      }
      mm_st_raw(mine + l * (kD31Words / 4) + pos, v);
    }
  }

  // ---- local ------------------------------------------------------------
  // Our own shard never goes near the switch. Placed AFTER the send so the wire is
  // already busy while this runs, and before the receive so it is not competing
  // with the poll for issue slots. kLocalFirst flips it, which is a real axis: at
  // the small end the send is a handful of instructions and moving the copy ahead
  // of it delays the only thing anyone is waiting for.
  auto local_copy = [&] {
    for (size_t j = tid; j < cm; j += stride) st_out<kStreamOut>(out + bm + j, in[compact_input_word<kRaggedInput>(j * 4, query_offsets, columns, queries, dense) / 4]);
  };
  if constexpr (!kLocalFirst) local_copy();

  // ---- receive ----------------------------------------------------------
  // Every remote source in one flat walk, so a slow source does not stall the
  // sources behind it in a per-source loop -- the same argument as the all-reduce
  // family's flat all-gather receive.
  if constexpr (kMode == kModeLL) {
    const size_t c8max = 2 * cm;
    const size_t tot = (size_t)(NG - 1) * c8max;
    for (size_t t = tid; t < tot; t += stride) {
      const size_t k = t / c8max;
      const size_t u = t - k * c8max;
      const int s = (int)k + ((int)k >= rank ? 1 : 0);
      const uint4* const cell = uc + base + (size_t)s * slice + u;
      uint4 d = ld_u4<kRelaxPoll>(cell);
      while (!ll_ready(d, tag)) d = ld_u4<kRelaxPoll>(cell);
      uint2* const o8 = (uint2*)(out + (size_t)s * cm);
      o8[u] = make_uint2(d.x, d.y);
    }
  } else if constexpr (kMode == kModeLine) {
    const size_t lmax = (cm + kLineData - 1) / kLineData;
    const int pos = (int)(tid & 7);
    const int src0 = (int)(threadIdx.x & 31) & ~7;  // lane 0 of this octet
    const unsigned omask = 0xFFu << (unsigned)((threadIdx.x & 31) & 24);
    const size_t lstride = stride >> 3;
    const size_t tot = (size_t)(NG - 1) * lmax;
    for (size_t t = tid >> 3; t < tot; t += lstride) {
      const size_t k = t / lmax;
      const size_t l = t - k * lmax;
      const int s = (int)k + ((int)k >= rank ? 1 : 0);
      const uint4* const cell = uc + base + (size_t)s * slice + l * kLinePacks + pos;
      // kPollDense is the speculative read: every lane loads ITS OWN slot and one
      // shuffle broadcasts the octet's tag, so a match means 128 B atomicity has
      // already put this round's data in the other lanes' registers -- one hop
      // instead of two. kPollSparse gives that hop back to buy 4x less spin
      // traffic; which wins is a band decision, see octet_wait.
      const uint4 d = octet_wait<kPoll, kRelaxPoll>(cell, pos, src0, omask, tag);
      if (pos != 0) {
        const size_t dd = l * kLineData + (size_t)(pos - 1);
        if (dd < cm) st_out<kStreamOut>(out + (size_t)s * cm + dd, d);
      }
    }
  } else {  // kModeD31 and kModeUniD31 share the wire layout, so also the receive
    const size_t nw = 4 * cm;
    const size_t lmax = (nw + kD31Data - 1) / kD31Data;
    const int pos = (int)(tid & 7);
    const int src0 = (int)(threadIdx.x & 31) & ~7;
    const unsigned omask = 0xFFu << (unsigned)((threadIdx.x & 31) & 24);
    const size_t lstride = stride >> 3;
    const size_t tot = (size_t)(NG - 1) * lmax;
    for (size_t t = tid >> 3; t < tot; t += lstride) {
      const size_t k = t / lmax;
      const size_t l = t - k * lmax;
      const int s = (int)k + ((int)k >= rank ? 1 : 0);
      const uint4* const cell = uc + base + (size_t)s * slice + l * (kD31Words / 4) + pos;
      const uint4 d = octet_wait<kPoll, kRelaxPoll>(cell, pos, src0, omask, tag);
      uint32_t* const ow = (uint32_t*)(out + (size_t)s * cm);
      const uint32_t* dw = (const uint32_t*)&d;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int f = pos * 4 + q;
        if (f == 0) continue;
        const size_t dd = l * kD31Data + (size_t)(f - 1);
        if (dd < nw) ow[dd] = dw[q];
      }
    }
  }

  if constexpr (kLocalFirst) local_copy();

  pdl_trigger_secondary();
#endif
}

// ---------------------------------------------------------------------------
// Phase-decomposition probe
// ---------------------------------------------------------------------------
// Phase probes keep separate loops to avoid changing the production kernel.
//
//   noop        -> graph-replay launch overhead at this grid.
//   local-only  -> writing all of `out` from `in` with no wire. The floor any
//                  all-gather that materialises `out` in HBM cannot go below.
//   send-only   -> egress + switch fan-out + peer-side ingress, with nothing
//                  waiting on it. If this alone is already the total, the wall is
//                  the wire and only fewer bytes help.
//   no-store    -> send + poll, minus the store into `out`. total - no_store is the
//                  copy-out that is NOT hidden behind the wire.
template <class P, int NG, int kProbe>
__global__ void k_ag_probe(const uint4* __restrict__ in, uint4* __restrict__ out,
                           uint4* uc, uint4* mc, void* const* peer_buf,
                           int rank, size_t n4, uint32_t* counters, size_t region,
                           size_t slot_stride, size_t slice) {
#if __CUDA_ARCH__ >= 900
  const uint32_t call_index = take_call_index_uniform(counters);
  const uint32_t tag = tag_of(call_index);
  const size_t base = region + slot_of(call_index) * slot_stride;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t cm = n4 / (size_t)NG;
  const size_t bm = (size_t)rank * cm;

  pdl_wait_primary();

  if constexpr (kProbe == kProbeNoop) {
    // The counter store above is the only memory traffic. Keeping `tag` live so
    // ptxas cannot fold the hash away.
    if (tag == 0u && tid == n4) out[0] = make_uint4(0, 0, 0, 0);
    pdl_trigger_secondary();
    return;
  }

  if constexpr (kProbe == kProbeLocalOnly) {
    // Every pack of `out` written from `in`, wire untouched. `cm` is a power of two
    // for every size this harness feeds, so the source index is a mask rather than
    // a division -- a probe may assume what a shipping row may not.
    for (size_t j = tid; j < n4; j += stride) out[j] = in[j & (cm - 1)];
    pdl_trigger_secondary();
    return;
  }

  // ---- send (D31, verbatim from k_ag_mcst) ------------------------------
  uint4* const mine = mc + base + (size_t)rank * slice;
  if constexpr (kProbe == kProbeSend7) {
    // One eighth of the ingress deleted, every remaining sender's egress unchanged.
    // ⚠ THE SILENT RANK IS THE LAST ONE, NOT RANK 0, and that is not cosmetic: the
    // harness prints rank 0's own event timing, so silencing rank 0 measured the
    // rank that had nothing to do (1.68 us at 16 MB, i.e. the launch floor).
    if (rank == NG - 1) {
      pdl_trigger_secondary();
      return;
    }
  }
  if constexpr (kProbe != kProbeRecvLocal) {
    const uint32_t* const inw = (const uint32_t*)in;
    const int pos = (int)(tid & 7);
    const size_t nw = 4 * cm;
    const size_t lines = (nw + kD31Data - 1) / kD31Data;
    const size_t lstride = stride >> 3;
    for (size_t l = tid >> 3; l < lines; l += lstride) {
      uint4 v;
      uint32_t* vw = (uint32_t*)&v;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int f = pos * 4 + k;
        if (f == 0) {
          vw[k] = tag;
        } else {
          const size_t d = l * kD31Data + (size_t)(f - 1);
          vw[k] = d < nw ? inw[d] : tag;
        }
      }
      mm_st_raw(mine + l * (kD31Words / 4) + pos, v);
    }
  }

  for (size_t j = tid; j < cm; j += stride) out[bm + j] = in[j];

  // ⚠ kProbeSend7 MUST leave here too, not just kProbeSendOnly. Leaving it to fall
  // into the receive was an instant deadlock: rank 0 is the one that skipped its
  // send, so ranks 1-7 would spin on a slice that is never written.
  if constexpr (kProbe == kProbeSendOnly || kProbe == kProbeSend7) {
    pdl_trigger_secondary();
    return;
  }

  // ---- receive, with the store optionally removed -----------------------
  // ⚠ The loads survive kProbeNoStore because ld_u4 is `asm volatile`; only the
  // store into `out` is gone. Without that property this row would measure an
  // empty loop.
  {
    const size_t nw = 4 * cm;
    const size_t lmax = (nw + kD31Data - 1) / kD31Data;
    const int pos = (int)(tid & 7);
    const int src0 = (int)(threadIdx.x & 31) & ~7;
    const unsigned omask = 0xFFu << (unsigned)((threadIdx.x & 31) & 24);
    const size_t lstride = stride >> 3;
    const size_t tot = (size_t)(NG - 1) * lmax;
    for (size_t t = tid >> 3; t < tot; t += lstride) {
      const size_t k = t / lmax;
      const size_t l = t - k * lmax;
      const int s = (int)k + ((int)k >= rank ? 1 : 0);
      const uint4* const cell = uc + base + (size_t)s * slice + l * (kD31Words / 4) + pos;
      uint4 d;
      if constexpr (kProbe == kProbeNoPoll || kProbe == kProbeRecvLocal) {
        // One unconditional read of the same cell the real receive reads. Same
        // transaction, same address, no spin -- so the difference against `full`
        // is exactly what waiting costs.
        d = ld_u4<false>(cell);
      } else {
        while (true) {
          d = ld_u4<false>(cell);
          const uint32_t seen = __shfl_sync(omask, d.x, src0);
          if (seen == tag) break;
        }
      }
      if constexpr (kProbe != kProbeNoStore) {
        uint32_t* const ow = (uint32_t*)(out + (size_t)s * cm);
        const uint32_t* dw = (const uint32_t*)&d;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int f = pos * 4 + q;
          if (f == 0) continue;
          const size_t dd = l * kD31Data + (size_t)(f - 1);
          if (dd < nw) ow[dd] = dw[q];
        }
      }
    }
  }

  pdl_trigger_secondary();
#endif
}

}  // namespace flash_ag_kernels_h200
}  // namespace sglang
