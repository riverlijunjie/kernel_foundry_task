// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// [EVOLVE_START]
// GGUF MoE weight transcode kernel — Xe3-LPG tuned v5.
//
// Builds on v4 (6.975 ms). VTune showed SBID (67-70%) + Send (60-66%) stalls with
// ALU1 (INT) at 70-75% of instructions — the round/clamp/bit-pack chain is the
// dependency bottleneck. v5 changes:
//   * Vectorize the requant multiply + round + clamp over float8/int8 so the
//     compiler emits packed SIMD ALU ops (fewer instructions, shorter dep chains),
//     directly attacking the SBID/Dist-Acc stalls.
//   * Keep v4's fused amax + single (float)half materialization (kills scratch reload).
//   * Keep 16-wide sub-group for a clean 32B coalesced SC store and wide packed WQ stores.

#include "include/batch_headers/common.cl"

inline half FUNC(mtq_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

#if defined(GGUF_IS_Q4_0)
inline void FUNC(mtq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(mtq_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 16; ++j) {
        const uchar b = qs[j];
        out[j]      = (half)(((int)(b & 0x0F) - 8) * (float)d);
        out[j + 16] = (half)(((int)(b >> 4)   - 8) * (float)d);
    }
}
#endif

#if defined(GGUF_IS_Q8_0)
inline void FUNC(mtq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(mtq_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 32; ++j) out[j] = (half)((float)qs[j] * (float)d);
}
#endif

#if defined(GGUF_IS_Q4_K) || defined(GGUF_IS_Q5_K)
inline void FUNC(mtq_scale_min_k4)(int j, const __global uchar* q, uchar* d, uchar* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (uchar)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uchar)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}
#endif

#if defined(GGUF_IS_Q4_K)
inline void FUNC(mtq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(mtq_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(mtq_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qs     = blk + 16;
    int o = 0, is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(mtq_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(mtq_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 32; ++l) out[o++] = (half)(d1 * (float)(qs[l] & 0x0F) - m1);
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 32; ++l) out[o++] = (half)(d2 * (float)(qs[l] >> 4) - m2);
        qs += 32; is += 2;
    }
}
#endif

#if defined(GGUF_IS_Q5_K)
inline void FUNC(mtq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(mtq_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(mtq_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qh     = blk + 16;
    const __global uchar* ql     = blk + 48;
    int o = 0, is = 0; uchar u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(mtq_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(mtq_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 32; ++l) { const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0); out[o++] = (half)(d1 * (float)q - m1); }
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 32; ++l) { const int q = (int)(ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0); out[o++] = (half)(d2 * (float)q - m2); }
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}
#endif

#if defined(GGUF_IS_Q6_K)
inline void FUNC(mtq_decode_block)(const __global uchar* blk, __private half* out) {
    const __global uchar* ql = blk;
    const __global uchar* qh = blk + 128;
    const __global char*  sc = (const __global char*)(blk + 192);
    const float d = (float)FUNC_CALL(mtq_load_f16)(blk + 208);
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[o + l + 0]  = (half)(d * (float)sc[is + 0] * q1);
            out[o + l + 32] = (half)(d * (float)sc[is + 2] * q2);
            out[o + l + 64] = (half)(d * (float)sc[is + 4] * q3);
            out[o + l + 96] = (half)(d * (float)sc[is + 6] * q4);
        }
        o += 128; ql += 64; qh += 32; sc += 8;
    }
}
#endif

// ---- main transcode kernel ----
__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(moe_gguf_transcode)(
    const __global uchar* W,
          __global uchar* WQ,
          __global half*  SC
)
{
    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    const int n   = (int)get_global_id(0);
    const int blk = (int)get_global_id(1);
    const int e   = (int)get_global_id(2);
    if (n >= N_SIZE || blk >= blocks_per_row || e >= NUM_EXPERTS)
        return;

    // ---- Stage 1: decode ONE GGUF block into private registers. ----
    half blk_vals[GGUF_BLOCK_ELEM];
    const __global uchar* w_row =
        W + ((uint)e * (uint)N_SIZE + (uint)n) * (uint)blocks_per_row * GGUF_BLOCK_BYTES;
    FUNC_CALL(mtq_decode_block)(w_row + (uint)blk * GGUF_BLOCK_BYTES, blk_vals);

    const uint row_base       = ((uint)e * (uint)N_SIZE + (uint)n) * (uint)K_SIZE;
    const uint groups_per_row = (uint)K_SIZE / (uint)REQUANT_GROUP;
    const uint sc_expert_base = (uint)e * groups_per_row * (uint)N_SIZE;

    const int groups_per_block = GGUF_BLOCK_ELEM / REQUANT_GROUP;
#if !TRANSCODE_TO_I4
    __global char* wq_i8 = (__global char*)WQ;
#endif

    // ---- Stage 2: fused amax + vectorized requantize per group. ----
    __attribute__((opencl_unroll_hint))
    for (int gi = 0; gi < groups_per_block; ++gi) {
        const int off_in_blk = gi * REQUANT_GROUP;
        const int g          = blk * groups_per_block + gi;
        const int k0         = g * REQUANT_GROUP;

        // Materialize the group into a private float window ONCE (kills scratch reload
        // of blk_vals — the SBID stall source). 4-way fmax breaks the reduction chain.
        float gvf[REQUANT_GROUP];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; i += 4) {
            const float v0 = (float)blk_vals[off_in_blk + i + 0];
            const float v1 = (float)blk_vals[off_in_blk + i + 1];
            const float v2 = (float)blk_vals[off_in_blk + i + 2];
            const float v3 = (float)blk_vals[off_in_blk + i + 3];
            gvf[i + 0] = v0; gvf[i + 1] = v1; gvf[i + 2] = v2; gvf[i + 3] = v3;
            a0 = fmax(a0, fabs(v0));
            a1 = fmax(a1, fabs(v1));
            a2 = fmax(a2, fabs(v2));
            a3 = fmax(a3, fabs(v3));
        }
        const float amax = fmax(fmax(a0, a1), fmax(a2, a3));

        const float scale     = (amax > 0.0f) ? (amax * (1.0f / (float)QMAX)) : 1.0f;
        const float inv_scale = (amax > 0.0f) ? ((float)QMAX * native_recip(amax)) : 0.0f;

        // Coalesced scale store across the sub-group (adjacent n -> adjacent addr).
        SC[sc_expert_base + (uint)g * (uint)N_SIZE + (uint)n] = (half)scale;

        // Pre-materialize quantized ints. VECTORIZED over float8/int8 blocks so the
        // compiler emits packed SIMD multiply + round + clamp — fewer ALU1 instructions
        // and a shorter dependency chain, directly attacking the SBID/Dist-Acc stalls.
        int q[REQUANT_GROUP];
#if TRANSCODE_TO_I4
    #if (REQUANT_GROUP % 8) == 0
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; i += 8) {
            float8 fv = (float8)(gvf[i+0], gvf[i+1], gvf[i+2], gvf[i+3],
                                 gvf[i+4], gvf[i+5], gvf[i+6], gvf[i+7]);
            int8 qi = convert_int8(round(fv * inv_scale));
            qi = clamp(qi, (int8)(-8), (int8)(7)) & (int8)(0x0F);
            q[i+0]=qi.s0; q[i+1]=qi.s1; q[i+2]=qi.s2; q[i+3]=qi.s3;
            q[i+4]=qi.s4; q[i+5]=qi.s5; q[i+6]=qi.s6; q[i+7]=qi.s7;
        }
    #else
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; ++i)
            q[i] = clamp((int)round(gvf[i] * inv_scale), -8, 7) & 0x0F;
    #endif

        const uint byte_base = (row_base + (uint)k0) >> 1;
        int i = 0;
    #if (REQUANT_GROUP % 8) == 0
        __global uchar4* wq4 = (__global uchar4*)(WQ + byte_base);
        __attribute__((opencl_unroll_hint))
        for (; i <= REQUANT_GROUP - 8; i += 8) {
            wq4[i >> 3] = (uchar4)(
                (uchar)(q[i + 0] | (q[i + 1] << 4)),
                (uchar)(q[i + 2] | (q[i + 3] << 4)),
                (uchar)(q[i + 4] | (q[i + 5] << 4)),
                (uchar)(q[i + 6] | (q[i + 7] << 4)));
        }
    #endif
        __attribute__((opencl_unroll_hint))
        for (; i < REQUANT_GROUP; i += 2)
            WQ[byte_base + ((uint)i >> 1)] = (uchar)(q[i] | (q[i + 1] << 4));
#else
    #if (REQUANT_GROUP % 8) == 0
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; i += 8) {
            float8 fv = (float8)(gvf[i+0], gvf[i+1], gvf[i+2], gvf[i+3],
                                 gvf[i+4], gvf[i+5], gvf[i+6], gvf[i+7]);
            int8 qi = convert_int8(round(fv * inv_scale));
            qi = clamp(qi, (int8)(-128), (int8)(127));
            q[i+0]=qi.s0; q[i+1]=qi.s1; q[i+2]=qi.s2; q[i+3]=qi.s3;
            q[i+4]=qi.s4; q[i+5]=qi.s5; q[i+6]=qi.s6; q[i+7]=qi.s7;
        }
    #else
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; ++i)
            q[i] = clamp((int)round(gvf[i] * inv_scale), -128, 127);
    #endif

        const uint wq_base = row_base + (uint)k0;
        int i = 0;
    #if (REQUANT_GROUP % 4) == 0
        __global char4* wq4 = (__global char4*)(wq_i8 + wq_base);
        __attribute__((opencl_unroll_hint))
        for (; i <= REQUANT_GROUP - 4; i += 4)
            wq4[i >> 2] = (char4)((char)q[i], (char)q[i + 1], (char)q[i + 2], (char)q[i + 3]);
    #endif
        __attribute__((opencl_unroll_hint))
        for (; i < REQUANT_GROUP; ++i)
            wq_i8[wq_base + (uint)i] = (char)q[i];
#endif
    }
}
// [EVOLVE_END]
