// SHARED DEVICE LAYER for the flash all-gather kernels (B300 / sm_103a).
//
// ⚠ THIS FILE IS A COPY OF A COPY, AND THIS PARAGRAPH IS THE PRICE OF THAT.
// History: this layer was first lifted out of flash_ar/csrc/h200/ar_kernels.cuh
// into flash_ag/csrc/h200/common.cuh so that flash_ag would never have to open a
// flash_ar translation unit. THIS file is the second copy, for sm_103. So the
// three constants below now live in FOUR device copies in this repository
// (flash_ar/h200, flash_ar/b200, flash_ag/h200, flash_ag/b300) plus the host side
// (policy_{h200,b300}.py NUM_SLOTS). A drift between any two of them is an
// out-of-bounds write or a permanent spin rather than a compile error.
//
// ⚠ tests/test_flash_ag_dispatch_b300.py pins ALL of them against each other --
// if you change a constant here, expect that to go red. Both dispatch tests assert
// the FULL set rather than splitting it between them, so forgetting to add a
// fifth copy to either file is the only way to slip through. See
// flash-ag-round2.md section 8.2.
//
// ⚠ WHY THIS IS A COPY AND NOT THE SAME FILE AS csrc/h200/common.cuh: the b300
// kernel file gains an axis the h200 one does not have (the kSend axis in
// ag_kernels.cuh). Sharing one header would mean every b300 experiment
// re-codegens the h200 binary, and the h200 27-band baseline in
// flash-ag-round1.md section 15.2 was measured on that exact binary. The rule is
// the all-gather header's own: "A probe may duplicate; it may not perturb."
//
// Nothing all-reduce-specific is here: raw 16B accessors, the PDL bracket,
// multimem.st, the pack traits (the all-gather adapter's `PackOf` needs the two
// types; the kernels do not read their members), and the device-side slot
// rotation. The sentinel protocol, uneven-shard helpers and every kernel stay in
// the all-reduce family, where they belong.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace ulacache_flashag {
namespace flash_ag_common_b300 {

// kMaxRanks = 8 peers; host must not exceed.
//
// ⚠ THIS BOX HAS FOUR VISIBLE GPUS TODAY (torch.cuda.device_count() == 4) and the
// shipping table is world 4 only. The constant stays at 8 because it is a
// CEILING, not a configuration: kMaxRanks sizes per-source array indices
// (`uint4 d[NG]` in the multi-source polls), so raising it costs registers and
// lowering it would break a future world-8 row. The measured 8-GPU artifacts for
// this machine exist (runs/b300/three_legs_w8, 2026-09-18) -- the card count
// dropped later, it was not always four.
constexpr int kMaxRanks = 8;

// ⚠ kNumSlots power-of-two; must match NUM_SLOTS in the policy table of THIS
// family (flash_ag/ops/policy_b300.py). The all-reduce family has its own copy of
// this constant and its own derivation for the value -- see ag_kernels.cuh's
// header for the all-gather one (the live window is {j-1, j, j+1} and the margin
// is ONE slot). Do not merge the two arguments.
constexpr int kNumSlots = 4;

// ⚠ NOT occupancy tuning: spinning-prefix blocks must stay resident or spin never
// ends. Adapter refuses wider grids -- a wider grid is a DEADLOCK, not a slow
// configuration.
//
// ⚠⚠ PROVENANCE CORRECTED FOR sm_103, AND IT MATTERS HERE MORE THAN ON H200. The
// "~303104 threads hangs" datum this number was derived from is the ARCHITECTURAL
// thread ceiling of THIS part: 148 SMs x 64 warps x 32 lanes = 303104, measured on
// torch.cuda.get_device_properties(0).multi_processor_count == 148 for the B300
// this file targets. 262144 = 86.5% of that ceiling -- so on sm_103 the margin
// between "shipping grids" and "the wall" is much thinner than the same number
// gives on H200, where the ceiling is 132 x 64 x 32 = 270336 and 262144 is 97% of
// it. Both readings are safe, but the h200 comment ("NOT occupancy tuning") was
// copied here with a provenance that no longer describes this machine. Re-measure
// with --launch_bounds before raising this. See flash-ag-round2.md section 6.1.
constexpr int kMaxSpinThreads = 262144;

// ---------------------------------------------------------------------------
// Raw 16B accessors
// ---------------------------------------------------------------------------
// volatile vs relaxed.sys: ±0.5% here; outer asm volatile blocks load hoisting in spins.
template <bool kRelaxed>
__device__ __forceinline__ uint4 ld_u4(const void* p) {
  uint4 d;
  if constexpr (kRelaxed) {
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "l"(p)
                 : "memory");
  } else {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "l"(p)
                 : "memory");
  }
  return d;
}

template <bool kRelaxed>
__device__ __forceinline__ void st_u4(void* p, uint4 d) {
  if constexpr (kRelaxed) {
    asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w)
                 : "memory");
  } else {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w)
                 : "memory");
  }
}

// st_out: evict-first `.cs` for write-once output (optional kStream).
template <bool kStream>
__device__ __forceinline__ void st_out(uint4* p, uint4 d) {
  if constexpr (kStream) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w)
                 : "memory");
  } else {
    *p = d;
  }
}

// ---------------------------------------------------------------------------
// PDL (Programmatic Dependent Launch)
// ---------------------------------------------------------------------------
// ⚠ PLACEMENT: slot/offsets before pdl_wait; payload read after. All shipping rows use PDL.
__device__ __forceinline__ void pdl_wait_primary() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger_secondary() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

// multimem.st is bitwise broadcast; `.f32` is a width tag (ptxas rejects `.v4.u32`).
__device__ __forceinline__ void mm_st_raw(void* mc, uint4 v) {
#if __CUDA_ARCH__ >= 900
  const float4 f = *reinterpret_cast<const float4*>(&v);
  asm volatile(
      "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(mc), "f"(f.x), "f"(f.y), "f"(f.z), "f"(f.w)
      : "memory");
#endif
}
// ---------------------------------------------------------------------------
// dtype traits
// ---------------------------------------------------------------------------
// Pack interpretation, sentinel pattern, multimem suffix. bf16 accumulates in bf16x2 (NCCL semantics).
__device__ __forceinline__ uint32_t bf16x2_add(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// Widen/narrow for P::WAcc. Narrow uses hardware cvt — manual rounding breaks NaNs.
// ⚠ `cvt.rn.bf16x2.f32`: first operand → HIGH half; swap = wrong pack order.
__device__ __forceinline__ float bf16_lo_to_f32(uint32_t w) {
  const uint32_t u = (w & 0xFFFFu) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
__device__ __forceinline__ float bf16_hi_to_f32(uint32_t w) {
  const uint32_t u = w & 0xFFFF0000u;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
// ⚠ `cvt.rn.bf16x2.f32`: first operand → HIGH half; swap = wrong pack order.
__device__ __forceinline__ uint32_t f32x2_to_bf16x2(float hi, float lo) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
struct PkF32 {
  using Acc = float4;
  static constexpr uint32_t kSentWord = 0x7FBADBADu;
  static constexpr uint32_t kQNanWord = 0x7FC00000u;
  static constexpr uint32_t kResetWord = 0x80000000u;

  __device__ static Acc zero() {
    return make_float4(0.f, 0.f, 0.f, 0.f);
  }
  __device__ static Acc of(uint4 d) {
    return *reinterpret_cast<const float4*>(&d);
  }
  __device__ static uint4 raw(Acc a) {
    return *reinterpret_cast<const uint4*>(&a);
  }
  __device__ static void add(Acc& a, const Acc& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
  }
  using WAcc = Acc;
  __device__ static WAcc wof(uint4 d) {
    return of(d);
  }
  __device__ static uint4 wraw(const WAcc& a) {
    return raw(a);
  }
  __device__ static void wadd(WAcc& a, const WAcc& b) {
    add(a, b);
  }
  __device__ static Acc flag(float v) {
    return make_float4(v, 0.f, 0.f, 0.f);
  }
  __device__ static float flag_of(uint4 d) {
    return __int_as_float((int)d.x);
  }
  __device__ static bool any_nonfinite(Acc a) {
    const uint4 u = raw(a);
    const uint32_t e = 0x7F800000u;
    return ((u.x & e) == e) || ((u.y & e) == e) || ((u.z & e) == e) || ((u.w & e) == e);
  }
#if __CUDA_ARCH__ >= 900
  __device__ static void mmred(void* mc, Acc v) {
    asm volatile("multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(mc),
                 "f"(v.x),
                 "f"(v.y),
                 "f"(v.z),
                 "f"(v.w)
                 : "memory");
  }
  __device__ static Acc mmldreduce(const void* mc) {
    float4 r;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
                 : "l"(mc)
                 : "memory");
    return r;
  }
#endif
};
struct PkBF16 {
  using Acc = uint4;  // accumulated in place as bf16x2
  static constexpr uint32_t kSentWord = 0x7FC17FC1u;
  static constexpr uint32_t kQNanWord = 0x7FC07FC0u;  // two canonical bf16 qNaNs
// ⚠ kResetWord = -0 (not +0): slot is mm_red operand; must match host fill_reset_value.
  static constexpr uint32_t kResetWord = 0x80008000u;

  __device__ static Acc zero() {
    return make_uint4(0, 0, 0, 0);
  }
  __device__ static Acc of(uint4 d) {
    return d;
  }
  __device__ static uint4 raw(Acc a) {
    return a;
  }
  __device__ static void add(Acc& a, const Acc& b) {
    a.x = bf16x2_add(a.x, b.x);
    a.y = bf16x2_add(a.y, b.y);
    a.z = bf16x2_add(a.z, b.z);
    a.w = bf16x2_add(a.w, b.w);
  }

// WAcc: f32 accumulate, round once. ⚠ Rank-order fold (k_oneshot_push) still required for F1.
  struct WAcc {
    float4 lo, hi;  // elements 0,1,2,3 and 4,5,6,7 of the pack
  };
  __device__ static WAcc wof(uint4 d) {
    WAcc w;
    w.lo = make_float4(bf16_lo_to_f32(d.x), bf16_hi_to_f32(d.x), bf16_lo_to_f32(d.y), bf16_hi_to_f32(d.y));
    w.hi = make_float4(bf16_lo_to_f32(d.z), bf16_hi_to_f32(d.z), bf16_lo_to_f32(d.w), bf16_hi_to_f32(d.w));
    return w;
  }
  __device__ static void wadd(WAcc& a, const WAcc& b) {
    a.lo.x += b.lo.x;
    a.lo.y += b.lo.y;
    a.lo.z += b.lo.z;
    a.lo.w += b.lo.w;
    a.hi.x += b.hi.x;
    a.hi.y += b.hi.y;
    a.hi.z += b.hi.z;
    a.hi.w += b.hi.w;
  }
  __device__ static uint4 wraw(const WAcc& a) {
    return make_uint4(
        f32x2_to_bf16x2(a.lo.y, a.lo.x),
        f32x2_to_bf16x2(a.lo.w, a.lo.z),
        f32x2_to_bf16x2(a.hi.y, a.hi.x),
        f32x2_to_bf16x2(a.hi.w, a.hi.z));
  }
  __device__ static Acc flag(float v) {
    uint32_t u;
    const float f = v;
    memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;  // round-to-nearest-even on the way to bf16
    return make_uint4(u >> 16, 0, 0, 0);
  }
  __device__ static float flag_of(uint4 d) {
    const uint32_t u = (d.x & 0xFFFFu) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
  }
  __device__ static bool any_nonfinite(Acc a) {
    const uint32_t m = 0x7F807F80u;
    const uint32_t bump = 0x00800080u, top = 0x80008000u;
    return ((((a.x & m) + bump) | (((a.y & m) + bump)) | (((a.z & m) + bump)) | (((a.w & m) + bump))) & top) != 0u;
  }
#if __CUDA_ARCH__ >= 900
// ⚠ multimem suffix follows dtype — wrong suffix adds bf16 bits as fp32 (silent garbage).
  __device__ static void mmred(void* mc, Acc v) {
    asm volatile("multimem.red.relaxed.sys.global.add.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(mc),
                 "r"(v.x),
                 "r"(v.y),
                 "r"(v.z),
                 "r"(v.w)
                 : "memory");
  }
  __device__ static Acc mmldreduce(const void* mc) {
    uint4 r;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(mc)
                 : "memory");
    return r;
  }
#endif
};
// ---------------------------------------------------------------------------
// Device-side slot rotation (CUDA-graph safe)
// ---------------------------------------------------------------------------
// Host-frozen args break slot pick under graph replay → silent wrong answer.
// ⚠ Fixed grid per family; returns 1-based call index (zero buffer would match seq==0).
__device__ __forceinline__ uint32_t take_call_index(uint32_t* counters) {
  const unsigned lane = threadIdx.x & 31u;
  const size_t warp_id = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  uint32_t v = 0;
  if (lane == 0) {
    v = counters[warp_id] + 1u;
    counters[warp_id] = v;
  }
  return __shfl_sync(0xFFFFFFFFu, v, 0);
}

__device__ __forceinline__ size_t slot_of(uint32_t call_index) {
  return (size_t)((call_index - 1u) & (uint32_t)(kNumSlots - 1));
}

// Shfl-free: all lanes load same word; no sync before in[] gather.
__device__ __forceinline__ uint32_t take_call_index_uniform(uint32_t* counters) {
  const size_t warp_id = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const uint32_t v = counters[warp_id] + 1u;
  if ((threadIdx.x & 31u) == 0u) counters[warp_id] = v;
  return v;
}

}  // namespace flash_ag_common_b300
}  // namespace ulacache_flashag
