/*
 * kernel_moe_q80_shared_sg.cl — Shared-expert Q8_0 MoE GEMV kernels using the
 *                                transposed "SG" (sub-group block-read)
 *                                weight layout.
 *
 * Two kernels, used together to implement one "shared expert" MoE-FFN block
 * (a single dense expert applied to every token, gated by a per-token scalar
 * routing weight -- no expert-pool indirection):
 *   shared_gate_up_q8_0    — fused gate-proj + up-proj GEMV + SiLU(gate)*up
 *   shared_down_merge_q8_0 — down-proj GEMV, scaled by the per-token scalar
 *                            gate (optionally accumulated into outputs)
 *
 * Seed implementation for this task: the SLM-free register-shuffle "SG"
 * kernel carried over from q4km_moe_gemv_opt/q80/moe_gemv_q80_shared_sg.cl.
 *
 * Optimization over the original ocl_v2 extraction:
 *
 *   *** SLM-free activation distribution ***
 *   The original kernels cooperatively staged each 256-element activation
 *   window into an SLM scratch buffer (one intel_sub_group_block_read8 → 16
 *   lanes each write 2 floats to SLM), then issued a per-K-block
 *   sub_group_barrier(CLK_LOCAL_MEM_FENCE) so every lane could read the whole
 *   window back out of SLM.  Because a 16-wide sub-group maps to a single
 *   SIMD16 hardware thread on Intel GPUs, the activation window can instead be
 *   redistributed lane→lane entirely in registers with
 *   intel_sub_group_broadcast (a cheap EU shuffle), avoiding:
 *     - the SLM scatter/gather round-trip (extra load/store-unit latency), and
 *     - the per-iteration local-memory barrier.
 *   Each K-block now issues its wide block_read8 activation load, materialises
 *   the 256 activations in registers via 128 shuffles (reused by BOTH the gate
 *   and up projections for gate_up), and dots them straight against the
 *   sub-group-block-read weights.  The only barrier left is the single
 *   end-of-kernel KSPLIT cross-sub-group reduction.
 *
 * The dispatch grid, SG-packed Q8_0 weight byte layout and numerics are
 * otherwise unchanged from the original, so test_moe_gemv_sg_kernels.py drives
 * these kernels with no host-side change.
 *
 * == Weight layout (SG-packed Q8_0, per [N, K] weight matrix) ==
 *   pqs section: N*nbpr*256 bytes at offset 0 (row-group h, block bid at
 *     (h*nbpr+bid)*OPG*256 bytes; sub-block j at +j*OPG*32, chunk c at
 *     +c*OPG*4).
 *   pd  section: N*nbpr*16 bytes right after pqs (SoA fp16, entry (h,bid) at
 *     (h*nbpr+bid)*OPG*16 bytes, field j at j*OPG*2 + lid*2).
 *
 * REQUIRED (host dispatch in task.py depends on them):
 *   1. Entry-point names and argument order must remain EXACTLY:
 *        shared_gate_up_q8_0(gate_weights, up_weights, hidden_states,
 *                            gate_up_out, hidden_size, intermediate_size)
 *        shared_down_merge_q8_0(gate_up_in, down_weights, shared_gate,
 *                                final_out, intermediate_size, hidden_size,
 *                                is_acc)
 *   2. OPG must remain 16 (task.py hardcodes sub_group_size = 16 for the
 *      dispatch's local size).
 *   3. Dispatch (host side, fixed by task.py):
 *        shared_gate_up_q8_0:
 *          global = (intermediate_size, KSPLIT, token_len)
 *          local  = (OPG=16,            KSPLIT, 1)
 *        shared_down_merge_q8_0:
 *          global = (hidden_size, KSPLIT, token_len)
 *          local  = (OPG=16,      KSPLIT, 1)
 *      KSPLIT is supplied at build time as -DKSPLIT=<n> (n = 1,2,4,8,...);
 *      the kernel must keep working correctly for any such n (task.py tests
 *      several values).
 */

// [USER_INSTRUCTIONS_START]
//
// Optimization target: shared_gate_up_q8_0 and shared_down_merge_q8_0,
// seeded from the SLM-free register-shuffle "SG" kernel
// (q4km_moe_gemv_opt/q80/moe_gemv_q80_shared_sg.cl).
//
// Optimize MoE GEMV tips:
// 1. These kernels are memory-bandwidth bound (streaming Q8_0-quantized
//    weights); focus on maximizing effective DRAM bandwidth and minimizing
//    ALU work per weight, not on reducing FLOPs.
// 2. Use intel_sub_group_block_read / block_read8 / block_read_us8 for wide,
//    coalesced loads of the packed weight bytes and the fp16 scale record.
// 3. Use sub-group shuffles/broadcasts to redistribute the shared activation
//    window across lanes instead of a per-iteration SLM stage + barrier
//    round-trip (the seed kernel already does this; keep exploring variants
//    that reduce shuffle count or reuse the broadcast activations across
//    BOTH the gate and up projections in shared_gate_up_q8_0).
// 4. Minimize live registers per matrix (avoid spills) so IGC can keep more
//    weight block-reads in flight simultaneously across K-blocks.
// 5. Tune KSPLIT-dependent code paths: the kernel is built with varying
//    -DKSPLIT=<n> to trade reduction-depth for occupancy on small
//    dispatches; do not hardcode assumptions that break for KSPLIT > 1 or
//    KSPLIT == 1.
// 6. Consider the optional Q80_HALF_MUL half-precision multiply-accumulate
//    path already stubbed in SH_Q8_ACC_HALF4/SH_Q8_SUBBLOCK -- explore
//    whether it (or a similar reduced-precision inner product) is a net win
//    for this bandwidth-bound workload without regressing correctness.
// 7. Wide sub-group block reads for weight bytes and the packed fp16 scale
//    record (avoid many narrow global loads); recover per-lane scalar
//    fields from one wide read via shuffles rather than separate loads.
// 8. Don't use OpenCL keywords for variable names (e.g., avoid naming a variable "local" or "global" or "half").
//
// [USER_INSTRUCTIONS_END]

#pragma OPENCL EXTENSION cl_khr_fp16        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

/* [EVOLVE_START] */
#define OPG 16  /* = sub_group_size, fixed by the reference kernels */

#ifndef KSPLIT
#define KSPLIT 1
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

/* ---------------------------------------------------------------------------
 * Q8_0 sub-block dot product: 32 signed int8 weights (uint8 `wv`) dotted
 * against 32 fp32 activations carried in 2-half packs `aw`, scaled by `sc`,
 * accumulated into `acc`.
 *
 * Broadcast two lanes at a time to form one float4 activation chunk for each
 * 4-byte weight pack, then consume immediately. This preserves the 16 total
 * broadcasts per sub-block, while avoiding a 32-float activation scratch array.
 * ------------------------------------------------------------------------- */
#define SH_Q8_CHUNK4_V(d, u, a) \
    (d) = fma(convert_float4(as_char4((uint)(u))), (a), (d))

inline float4 sh_q8_0_act4(uint aw, uint lane) {
    float2 f0 = convert_float2(as_half2(sub_group_broadcast(aw, lane)));
    float2 f1 = convert_float2(as_half2(sub_group_broadcast(aw, lane + 1)));
    return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

#define SH_Q8_SUBBLOCK_AW(aw, wv, sc, acc)                                   \
    do {                                                                     \
        float4 _d = (float4)(0.0f);                                          \
        SH_Q8_CHUNK4_V(_d, (wv).s0, sh_q8_0_act4((aw),  0));                  \
        SH_Q8_CHUNK4_V(_d, (wv).s1, sh_q8_0_act4((aw),  2));                  \
        SH_Q8_CHUNK4_V(_d, (wv).s2, sh_q8_0_act4((aw),  4));                  \
        SH_Q8_CHUNK4_V(_d, (wv).s3, sh_q8_0_act4((aw),  6));                  \
        SH_Q8_CHUNK4_V(_d, (wv).s4, sh_q8_0_act4((aw),  8));                  \
        SH_Q8_CHUNK4_V(_d, (wv).s5, sh_q8_0_act4((aw), 10));                  \
        SH_Q8_CHUNK4_V(_d, (wv).s6, sh_q8_0_act4((aw), 12));                  \
        SH_Q8_CHUNK4_V(_d, (wv).s7, sh_q8_0_act4((aw), 14));                  \
        float _sum = (_d.s0 + _d.s1) + (_d.s2 + _d.s3);                      \
        (acc) = fma((sc), _sum, (acc));                                      \
    } while (0)

#define SH_Q8_SUBBLOCK_PAIR_AW(aw, gw, gsc, uw, usc, gacc, uacc)             \
    do {                                                                     \
        float4 _gd = (float4)(0.0f);                                         \
        float4 _ud = (float4)(0.0f);                                         \
        float4 _a;                                                           \
        _a = sh_q8_0_act4((aw),  0);                                         \
        SH_Q8_CHUNK4_V(_gd, (gw).s0, _a); SH_Q8_CHUNK4_V(_ud, (uw).s0, _a);   \
        _a = sh_q8_0_act4((aw),  2);                                         \
        SH_Q8_CHUNK4_V(_gd, (gw).s1, _a); SH_Q8_CHUNK4_V(_ud, (uw).s1, _a);   \
        _a = sh_q8_0_act4((aw),  4);                                         \
        SH_Q8_CHUNK4_V(_gd, (gw).s2, _a); SH_Q8_CHUNK4_V(_ud, (uw).s2, _a);   \
        _a = sh_q8_0_act4((aw),  6);                                         \
        SH_Q8_CHUNK4_V(_gd, (gw).s3, _a); SH_Q8_CHUNK4_V(_ud, (uw).s3, _a);   \
        _a = sh_q8_0_act4((aw),  8);                                         \
        SH_Q8_CHUNK4_V(_gd, (gw).s4, _a); SH_Q8_CHUNK4_V(_ud, (uw).s4, _a);   \
        _a = sh_q8_0_act4((aw), 10);                                         \
        SH_Q8_CHUNK4_V(_gd, (gw).s5, _a); SH_Q8_CHUNK4_V(_ud, (uw).s5, _a);   \
        _a = sh_q8_0_act4((aw), 12);                                         \
        SH_Q8_CHUNK4_V(_gd, (gw).s6, _a); SH_Q8_CHUNK4_V(_ud, (uw).s6, _a);   \
        _a = sh_q8_0_act4((aw), 14);                                         \
        SH_Q8_CHUNK4_V(_gd, (gw).s7, _a); SH_Q8_CHUNK4_V(_ud, (uw).s7, _a);   \
        float _gsum = (_gd.s0 + _gd.s1) + (_gd.s2 + _gd.s3);                 \
        float _usum = (_ud.s0 + _ud.s1) + (_ud.s2 + _ud.s3);                 \
        (gacc) = fma((gsc), _gsum, (gacc));                                  \
        (uacc) = fma((usc), _usum, (uacc));                                  \
    } while (0)

/* Decode the 8 SoA fp16 sub-block scales fetched by one block_read_us8. */
inline void sh_q8_0_decode_scales8(ushort8 dv, float *scale) {
    scale[0] = (float)as_half(dv.s0); scale[1] = (float)as_half(dv.s1);
    scale[2] = (float)as_half(dv.s2); scale[3] = (float)as_half(dv.s3);
    scale[4] = (float)as_half(dv.s4); scale[5] = (float)as_half(dv.s5);
    scale[6] = (float)as_half(dv.s6); scale[7] = (float)as_half(dv.s7);
}

/* Each sub-block chunk uses two lane broadcasts to form a float4 activation
 * vector in registers; no 32-float scratch array is materialized. */

/* =======================================================================
 * Kernel 1: shared_gate_up_q8_0
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void shared_gate_up_q8_0(
    const __global uchar * restrict gate_weights,
    const __global uchar * restrict up_weights,
    const __global half  * restrict hidden_states,
          __global half  * restrict gate_up_out,
    uint hidden_size,
    uint intermediate_size
) {
    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint t  = get_group_id(2);

    const uint nbpr   = hidden_size / 256u;
    const uint off_pd = (uint)intermediate_size * nbpr * 256u;

    const __global uint*   g_pqs_u = (const __global uint*)gate_weights;
    const __global ushort* g_pd_us = (const __global ushort*)(gate_weights + off_pd);
    const __global uint*   u_pqs_u = (const __global uint*)up_weights;
    const __global ushort* u_pd_us = (const __global ushort*)(up_weights + off_pd);

    const __global half* x = hidden_states + (size_t)t * hidden_size;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        /* Wide, coalesced loads that stay resident for the whole K-block:
         * the 256 activations (2 halves/lane) and the two fp16 scale vectors. */
        uint8 iv = intel_sub_group_block_read8((const __global uint *)(x + (size_t)bid * 256));

        const uint pd_off = (h * nbpr + bid) * (OPG * 8u);
        float gscale[8], uscale[8];
        sh_q8_0_decode_scales8(intel_sub_group_block_read_us8(g_pd_us + pd_off), gscale);
        sh_q8_0_decode_scales8(intel_sub_group_block_read_us8(u_pd_us + pd_off), uscale);

        const uint pe_u = (h * nbpr + bid) * (OPG * 256u / 4u);

        /* Interleave each sub-block's weight block-reads with its consumption:
         * peak live weight state is ~2 uint8 instead of 16 -> no spills, so IGC
         * keeps more outstanding reads in flight (attacks the SBID stalls),
         * while the 16 activation broadcasts overlap those loads.  The gathered
         * `act` feeds BOTH gate and up via the vectorized float4 dot product. */
        #define SB_GU(i, comp)                                                           \
            do {                                                                          \
                uint8 gw = intel_sub_group_block_read8(g_pqs_u + pe_u + (i) * OPG * 8);   \
                uint8 uw = intel_sub_group_block_read8(u_pqs_u + pe_u + (i) * OPG * 8);   \
                SH_Q8_SUBBLOCK_PAIR_AW(iv.comp, gw, gscale[i], uw, uscale[i],             \
                                       gate_acc, up_acc);                                \
            } while (0)

        SB_GU(0, s0); SB_GU(1, s1); SB_GU(2, s2); SB_GU(3, s3);
        SB_GU(4, s4); SB_GU(5, s5); SB_GU(6, s6); SB_GU(7, s7);
        #undef SB_GU
    }

    const uint row = h * OPG + hh;
#if KSPLIT == 1
    {
        float sig = 1.0f / (1.0f + native_exp(-gate_acc));
        gate_up_out[(size_t)t * intermediate_size + row] = (half)((sig * gate_acc) * up_acc);
    }
#else
    __local float slm_up[KSPLIT * OPG];
    __local float slm_gate[KSPLIT * OPG];
    slm_up[sg * OPG + hh]   = up_acc;
    slm_gate[sg * OPG + hh] = gate_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float u = 0.0f, gt = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) {
            u  += slm_up[r * OPG + hh];
            gt += slm_gate[r * OPG + hh];
        }
        float sig = 1.0f / (1.0f + native_exp(-gt));
        gate_up_out[(size_t)t * intermediate_size + row] = (half)((sig * gt) * u);
    }
#endif
}

/* =======================================================================
 * Kernel 2: shared_down_merge_q8_0
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void shared_down_merge_q8_0(
    const __global half  * restrict gate_up_in,
    const __global uchar * restrict down_weights,
    const __global half  * restrict shared_gate,
          __global half  * restrict final_out,
    uint intermediate_size,
    uint hidden_size,
    int  is_acc
) {
    const uint h   = get_group_id(0);
    const uint hh  = get_local_id(0);
    const uint sg  = get_local_id(1);
    const uint t   = get_group_id(2);

    const uint nbpr   = intermediate_size / 256u;
    const uint off_pd = (uint)hidden_size * nbpr * 256u;

    const __global uint*   d_pqs_u = (const __global uint*)down_weights;
    const __global ushort* d_pd_us = (const __global ushort*)(down_weights + off_pd);

    const __global half* x = gate_up_in + (size_t)t * intermediate_size;

    float acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        uint8 iv = intel_sub_group_block_read8((const __global uint *)(x + (size_t)bid * 256));

        const uint pd_off = (h * nbpr + bid) * (OPG * 8u);
        float scale[8];
        sh_q8_0_decode_scales8(intel_sub_group_block_read_us8(d_pd_us + pd_off), scale);

        const uint pe_u = (h * nbpr + bid) * (OPG * 256u / 4u);

        /* Same low-register interleaved schedule + vectorized float4 dot. */
        #define SB_D(i, comp)                                                            \
            do {                                                                          \
                uint8 w = intel_sub_group_block_read8(d_pqs_u + pe_u + (i) * OPG * 8);    \
                SH_Q8_SUBBLOCK_AW(iv.comp, w, scale[i], acc);                             \
            } while (0)

        SB_D(0, s0); SB_D(1, s1); SB_D(2, s2); SB_D(3, s3);
        SB_D(4, s4); SB_D(5, s5); SB_D(6, s6); SB_D(7, s7);
        #undef SB_D
    }

    const float gscale = (float)shared_gate[t];
    const uint row = h * OPG + hh;

#if KSPLIT == 1
    {
        float o = acc * gscale;
        if (is_acc)
            o += (float)final_out[(size_t)t * hidden_size + row];
        final_out[(size_t)t * hidden_size + row] = (half)o;
    }
#else
    __local float slm_red[KSPLIT * OPG];
    slm_red[sg * OPG + hh] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float o = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) o += slm_red[r * OPG + hh];
        o *= gscale;
        if (is_acc)
            o += (float)final_out[(size_t)t * hidden_size + row];
        final_out[(size_t)t * hidden_size + row] = (half)o;
    }
#endif
}

/* [EVOLVE_END] */
