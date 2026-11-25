
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"

// Fake group size for compatibility and computation performance balance.
// Each gk-iteration of the inner GEMV loop processes FAKE_GROUP_SIZE K-elements
// using a single (scale, zp) entry, so FAKE_GROUP_SIZE must divide both
// GATE_UP_GROUP_SIZE and DOWN_GROUP_SIZE. The traditional value is 128 which
// matches the sub_group_block_read tile widths below; for models whose weight
// quantization uses a smaller group (e.g. 64) we shrink FAKE_GROUP_SIZE so the
// per-iteration accumulation stays within one quant group.
#if defined(GATE_UP_GROUP_SIZE) && defined(DOWN_GROUP_SIZE)
#    if GATE_UP_GROUP_SIZE < DOWN_GROUP_SIZE
#        define MOE_MIN_GROUP_SIZE GATE_UP_GROUP_SIZE
#    else
#        define MOE_MIN_GROUP_SIZE DOWN_GROUP_SIZE
#    endif
#    if MOE_MIN_GROUP_SIZE < 128
#        define FAKE_GROUP_SIZE MOE_MIN_GROUP_SIZE
#    else
#        define FAKE_GROUP_SIZE 128
#    endif
#else
#    define FAKE_GROUP_SIZE 128
#endif

// Number of K-elements each work-item handles per gk-iteration via the
// intel_sub_group_block_read tile. Drives the inner-loop variant selection.
// Supported values: 2 (FAKE_GROUP_SIZE = 1 * SUBGROUP_SIZE * 2, e.g. SG=16 + 32),
// 4 (e.g. SG=32 + FAKE=128 or SG=16 + FAKE=64), and 8 (SG=16 + FAKE=128).
#define ELEMS_PER_LANE (FAKE_GROUP_SIZE / SUBGROUP_SIZE)

// Experts per token: MAX_TOPK for non-shared, MAX_TOPK+1 with shared expert.
// Used by batched GEMV to decompose flat workgroup ID into (token_idx, expert_slot).
#if SHARED_EXPERT_ENABLE
#define EXPERTS_PER_TOKEN (MAX_TOPK + 1)
#else
#define EXPERTS_PER_TOKEN MAX_TOPK
#endif

// Gate activation: SwiGLU (Swish, default), GeGLU-Tanh, or GeGLU-ERF.
// GATE_ACT_GELU_ERF takes precedence over GATE_ACT_GELU_TANH when both are set.
#ifdef GATE_ACT_GELU_ERF
// ERF Gelu: 0.5 * x * (1 + erf(x / sqrt(2))); A&S 7.1.26 fast erf approximation
inline float moe_mlp_fast_erf(float x) {
    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    float z = fabs(x);
    float t = 1.0f / (1.0f + p * z);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-(z * z));
    return (x >= 0.0f) ? y : -y;
}
#    define MOE_GATE_ACT(x) (0.5f * (x) * (1.0f + moe_mlp_fast_erf((x) * 0.7071067811865475f)))
#elif defined(GATE_ACT_GELU_TANH)
#    define MOE_GATE_ACT(x) (0.5f * (x) * (1.0f + (tanh(0.79788458347320556640625f * (x) * (1.0f + 0.044715f * (x) * (x))))))
#else
#    define MOE_GATE_ACT(x) ((x) / (1.0f + exp(-(x))))
#endif

// HAS_ZP: 1 = asymmetric quantization (subtract zero point), 0 = symmetric (no zero point)
#if HAS_ZP
#    define ZP_ADJUST_2(sum0, sum1, xg_sum_gk, z0, z1) ((sum0) - (xg_sum_gk) * (z0)), ((sum1) - (xg_sum_gk) * (z1))
#    define ZP_ADJUST_4(sum0123, xg_sum_gk, z) ((sum0123) - (xg_sum_gk) * (z))
#else
#    define ZP_ADJUST_2(sum0, sum1, xg_sum_gk, z0, z1) (sum0), (sum1)
#    define ZP_ADJUST_4(sum0123, xg_sum_gk, z) (sum0123)
#endif

// WEIGHT_IS_SIGNED: sign extension helpers for i4/i8 weights
#if WEIGHT_IS_SIGNED
// 4-bit sign extension: [0,15] -> [-8,7]. If bit3 is set, OR with 0xF0 to sign-extend to char.
#    define DEQUANT_4BIT_LO(v) convert_half((char)(((v) & 0x08) ? ((v) | 0xF0) : ((v) & 0x0F)))
#    define DEQUANT_4BIT_HI(v) convert_half((char)((((v) >> 4) & 0x08) ? (((v) >> 4) | 0xF0) : (((v) >> 4) & 0x0F)))
// 8-bit sign extension: reinterpret uchar as signed char
#    define DEQUANT_8BIT(v)    convert_half(as_char(v))
#else
#    define DEQUANT_4BIT_LO(v) convert_half((v) & 0x0F)
#    define DEQUANT_4BIT_HI(v) convert_half((v) >> 4)
#    define DEQUANT_8BIT(v)    convert_half(v)
#endif

// ============================================================================
// Native GGUF block decode (WEIGHT_COMPRESSEION_DT == 3). The expert weights stay as raw GGUF
// super-blocks (MOE_SPEC.md §4.3.2); each output row is a contiguous run of K/BLOCK_ELEM blocks.
// The four per-format `moe_gguf_dot_*` helpers stream-decode one block against a HIDDEN-slice of
// the SLM-cached activation `xx` (no materialised dequant), returning the block's float dot.
// Type codes (JIT GATEUP_DECODE / DOWN_DECODE / SHARED_DECODE): 1=Q4_K 2=Q5_K 3=Q6_K 4=Q8_0.
// Forked from gguf/fc_gguf_opt.cl gguf_block_dot with activation read from __local instead of global.
// ============================================================================
#if WEIGHT_COMPRESSEION_DT == 3

#define MOE_GGUF_Q4_K 1
#define MOE_GGUF_Q5_K 2
#define MOE_GGUF_Q6_K 3
#define MOE_GGUF_Q8_0 4

// All helpers are wrapped in FUNC()/FUNC_CALL() so that, when the gate_up / down / reduce stage
// sources are combined into one OpenCL batch, each stage gets a uniquely-suffixed copy (avoids
// duplicate-definition build failures — the existing stage GEMVs sidestep this via GATE_UP_ENABLE /
// DOWN_ENABLE guards, but the GGUF decoders are shared across stages).
inline int FUNC(moe_gguf_blk_elem)(int code) {
    return (code == MOE_GGUF_Q8_0) ? 32 : 256;
}
inline int FUNC(moe_gguf_blk_bytes)(int code) {
    // Q8_0: 2(d)+32(qs)=34; Q4_K: 2(d)+2(dmin)+12(scales)+128(qs)=144;
    // Q5_K: 2+2+12+32(qh)+128(ql)=176; Q6_K: 128(ql)+64(qh)+16(sc)+2(d)=210.
    return (code == MOE_GGUF_Q8_0) ? 34 : (code == MOE_GGUF_Q4_K) ? 144 : (code == MOE_GGUF_Q5_K) ? 176 : 210;
}

// Unaligned little-endian ggml_half load.
inline half FUNC(moe_gguf_load_f16)(const __global uchar* p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(u);
}

// 6-bit packed sub-block scale/min extraction shared by Q4_K / Q5_K (ggml get_scale_min_k4).
inline void FUNC(moe_gguf_get_scale_min_k4)(int j, const __global uchar* q, uchar* d, uchar* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uchar)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uchar)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

inline float FUNC(moe_gguf_dot_q8_0)(const __global uchar* blk, const __local half* xx) {
    const float d = (float)FUNC_CALL(moe_gguf_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    float acc = 0.0f;
    for (int j = 0; j < 32; ++j)
        acc += (float)xx[j] * ((float)qs[j] * d);
    return acc;
}

inline float FUNC(moe_gguf_dot_q4_k)(const __global uchar* blk, const __local half* xx) {
    const float d    = (float)FUNC_CALL(moe_gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(moe_gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;   // 12 bytes
    const __global uchar* qs     = blk + 16;  // 128 bytes
    float acc = 0.0f;
    int ai = 0, is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(moe_gguf_get_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(moe_gguf_get_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        float sq1 = 0.0f, sa1 = 0.0f, sq2 = 0.0f, sa2 = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float av = (float)xx[ai + l];
            sq1 += av * (float)(qs[l] & 0x0F);
            sa1 += av;
        }
        for (int l = 0; l < 32; ++l) {
            const float av = (float)xx[ai + 32 + l];
            sq2 += av * (float)(qs[l] >> 4);
            sa2 += av;
        }
        acc += d1 * sq1 - m1 * sa1 + d2 * sq2 - m2 * sa2;
        qs += 32;
        is += 2;
        ai += 64;
    }
    return acc;
}

inline float FUNC(moe_gguf_dot_q5_k)(const __global uchar* blk, const __local half* xx) {
    const float d    = (float)FUNC_CALL(moe_gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(moe_gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;   // 12 bytes
    const __global uchar* qh     = blk + 16;  // 32 bytes (high bit-plane)
    const __global uchar* ql     = blk + 48;  // 128 bytes (low 4 bits)
    float acc = 0.0f;
    int ai = 0, is = 0;
    uchar u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(moe_gguf_get_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(moe_gguf_get_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        float sq1 = 0.0f, sa1 = 0.0f, sq2 = 0.0f, sa2 = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float av = (float)xx[ai + l];
            const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
            sq1 += av * (float)q;
            sa1 += av;
        }
        for (int l = 0; l < 32; ++l) {
            const float av = (float)xx[ai + 32 + l];
            const int q = (int)(ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            sq2 += av * (float)q;
            sa2 += av;
        }
        acc += d1 * sq1 - m1 * sa1 + d2 * sq2 - m2 * sa2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
        ai += 64;
    }
    return acc;
}

inline float FUNC(moe_gguf_dot_q6_k)(const __global uchar* blk, const __local half* xx) {
    const __global uchar* ql = blk;                                 // 128 bytes (low 4 bits)
    const __global uchar* qh = blk + 128;                           // 64 bytes (high 2 bits)
    const __global char*  sc = (const __global char*)(blk + 192);   // 16 signed scales
    const float d = (float)FUNC_CALL(moe_gguf_load_f16)(blk + 208);
    float acc = 0.0f;
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc += (float)xx[o + l + 0]  * (d * (float)sc[is + 0] * q1);
            acc += (float)xx[o + l + 32] * (d * (float)sc[is + 2] * q2);
            acc += (float)xx[o + l + 64] * (d * (float)sc[is + 4] * q3);
            acc += (float)xx[o + l + 96] * (d * (float)sc[is + 6] * q4);
        }
        o  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return acc;
}

// Per-expert weight byte size and row byte stride for a GGUF-coded projection [N, K]. Defined here
// (before the dp4a/LISA blocks) so those GEMVs can call it.
inline long FUNC(moe_gguf_expert_bytes)(int code, int N, int K) {
    return (long)N * (K / FUNC_CALL(moe_gguf_blk_elem)(code)) * FUNC_CALL(moe_gguf_blk_bytes)(code);
}
inline long FUNC(moe_gguf_row_bytes)(int code, int K) {
    return (long)(K / FUNC_CALL(moe_gguf_blk_elem)(code)) * FUNC_CALL(moe_gguf_blk_bytes)(code);
}

#if MOE_GGUF_DP4A
// =============================================================================================
// int8-activation dp4a decode (opt-in OV_GPU_GGUF_MOE_DP4A=1). The activation is pre-quantized to
// signed int8 (xq) + per-32 f32 scale (xsc) in SLM by the kernel entry; both the unpacked weight
// code and the activation enter the hardware integer dot product (dot_acc_sat_4x8packed_*), cutting
// the per-element unpack ALU that bottlenecks the scalar float decode (mirrors gguf/fc_gguf_dp4a.cl,
// which measured ~1.6-1.8x on Q4_K / ~2x on Q5_K). Q6_K also runs int8-activation dp4a (SWAR unpack
// of the 4-bit low + 2-bit high plane, -32 offset folded via all-ones dot). Weights read from the
// RAW GGUF block.
// =============================================================================================
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

// Q8_0: signed int8 weights, one 32-element group. signed x signed dp4a.
inline float FUNC(moe_dp_dot_q8_0)(const __global uchar* blk, const __local char* aq, const __local float* asc) {
    const float d = (float)FUNC_CALL(moe_gguf_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    int S = 0;
    for (int l = 0; l < 32; l += 4) {
        const uint w = as_uint(vload4(0, qs + l));
        const uint a = as_uint(vload4(0, aq + l));
        S = dot_acc_sat_4x8packed_ss_int(w, a, S);
    }
    return d * asc[0] * (float)S;
}

// Q4_K: unsigned 4-bit weights (0..15) + per-32 sub-scale/min; unsigned x signed dp4a. The min term is
// folded via a dot against an all-ones weight (Sqa), reusing the already-loaded activation lanes.
inline float FUNC(moe_dp_dot_q4_k)(const __global uchar* blk, const __local char* aq, const __local float* asc) {
    const float d    = (float)FUNC_CALL(moe_gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(moe_gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qs     = blk + 16;
    float acc = 0.0f;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(moe_gguf_get_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(moe_gguf_get_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        const __local char* aq_lo = aq + 0;
        const __local char* aq_hi = aq + 32;
        // v6: gate_up (Q4_K) is NOT compute-bound at 32-lane saturation -> keep the in-loop all-ones
        // dp4a (Sqa) here; the SLM-read variant (v5) regressed gate_up. Q5_K/down uses xsum instead.
        int Sqq1 = 0, Sqa1 = 0, Sqq2 = 0, Sqa2 = 0;
        for (int l = 0; l < 32; l += 4) {
            const uint qlu = as_uint(vload4(0, qs + l));
            const uint lo4 = qlu & 0x0F0F0F0Fu;
            const uint hi4 = (qlu >> 4) & 0x0F0F0F0Fu;
            const uint au_lo = as_uint(vload4(0, aq_lo + l));
            const uint au_hi = as_uint(vload4(0, aq_hi + l));
            Sqq1 = dot_acc_sat_4x8packed_us_int(lo4, au_lo, Sqq1);
            Sqq2 = dot_acc_sat_4x8packed_us_int(hi4, au_hi, Sqq2);
            Sqa1 = dot_acc_sat_4x8packed_us_int(0x01010101u, au_lo, Sqa1);
            Sqa2 = dot_acc_sat_4x8packed_us_int(0x01010101u, au_hi, Sqa2);
        }
        acc += asc[is + 0] * (d1 * Sqq1 - m1 * Sqa1);
        acc += asc[is + 1] * (d2 * Sqq2 - m2 * Sqa2);
        qs += 32; is += 2; aq += 64;
    }
    return acc;
}

// Q5_K: 4-bit low + 1-bit high plane (0..31) + per-32 sub-scale/min; unsigned x signed dp4a.
inline float FUNC(moe_dp_dot_q5_k)(const __global uchar* blk, const __local char* aq, const __local float* asc, const __local int* axsum) {
    const float d    = (float)FUNC_CALL(moe_gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(moe_gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qh     = blk + 16;
    const __global uchar* ql     = blk + 48;
    float acc = 0.0f;
    int is = 0, bit = 0;
    for (int iter = 0; iter < 4; ++iter) {
        uchar sc, m;
        FUNC_CALL(moe_gguf_get_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(moe_gguf_get_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        const __local char* aq_lo = aq + (2 * iter) * 32;
        const __local char* aq_hi = aq + (2 * iter + 1) * 32;
        // v5: min-fold sums precomputed in axsum (see quant_act_i8) -> drop the two all-ones dp4a.
        int Sqq1 = 0, Sqq2 = 0;
        for (int l = 0; l < 32; l += 4) {
            const uint qlu = as_uint(vload4(0, ql + l));
            const uint qhu = as_uint(vload4(0, qh + l));
            const uint lo4 = qlu & 0x0F0F0F0Fu;
            const uint hi4 = (qlu >> 4) & 0x0F0F0F0Fu;
            const uint wlo = lo4 | (((qhu >> bit)       & 0x01010101u) << 4);
            const uint whi = hi4 | (((qhu >> (bit + 1)) & 0x01010101u) << 4);
            const uint au_lo = as_uint(vload4(0, aq_lo + l));
            const uint au_hi = as_uint(vload4(0, aq_hi + l));
            Sqq1 = dot_acc_sat_4x8packed_us_int(wlo, au_lo, Sqq1);
            Sqq2 = dot_acc_sat_4x8packed_us_int(whi, au_hi, Sqq2);
        }
        acc += asc[2 * iter]     * (d1 * Sqq1 - m1 * axsum[2 * iter]);
        acc += asc[2 * iter + 1] * (d2 * Sqq2 - m2 * axsum[2 * iter + 1]);
        ql += 32; is += 2; bit += 2;
    }
    return acc;
}

// One SWAR-unpack + dp4a step for a single packed uint (4 weights) of a low-nibble stream (A) and
// its paired high-nibble stream (C). qlc/qhc are the matching ql/qh uint lanes; the Q6_K 2 high
// bits come from qh shifted by sh_lo/sh_hi. Sqq accumulates w*a, Sqa accumulates a (to fold the -32
// dequant offset). Mirrors the proven standalone gguf/fc_gguf_dp4a.cl dp_block_dot_q6k decoder.
#define MOE_DP_Q6K_STEP(qlc, qhc, aAc, aCc, Sa, Aa, Sc, Ac, sh_lo, sh_hi) do {                       \
    const uint _wA = ((qlc) & 0x0F0F0F0Fu)        | ((((qhc) >> (sh_lo)) & 0x03030303u) << 4);       \
    const uint _wC = (((qlc) >> 4) & 0x0F0F0F0Fu) | ((((qhc) >> (sh_hi)) & 0x03030303u) << 4);       \
    (Sa) = dot_acc_sat_4x8packed_us_int(_wA, (aAc), (Sa));                                           \
    (Aa) = dot_acc_sat_4x8packed_us_int(0x01010101u, (aAc), (Aa));                                   \
    (Sc) = dot_acc_sat_4x8packed_us_int(_wC, (aCc), (Sc));                                           \
    (Ac) = dot_acc_sat_4x8packed_us_int(0x01010101u, (aCc), (Ac));                                   \
} while (0)

// Q6_K: 6-bit weights (4-bit low + 2-bit high plane, dequant offset -32) + per-16 signed scales;
// unsigned x signed dp4a. Streaming dual-dp4a over one 256-element block against the int8 activation
// slice `aq` (natural block order) with its eight per-32 scales `asc`. The -32 offset is folded via
// Sqa (dot against all-ones). Ported from the standalone dp_block_dot_q6k (int8-activation path).
inline float FUNC(moe_dp_dot_q6_k)(const __global uchar* blk, const __local char* aq, const __local float* asc) {
    const __global uchar* ql = blk;                                 // 128 bytes (low 4 bits)
    const __global uchar* qh = blk + 128;                           // 64 bytes (high 2 bits)
    const __global char*  sc = (const __global char*)(blk + 192);   // 16 signed scales
    const float d = (float)FUNC_CALL(moe_gguf_load_f16)(blk + 208);
    float acc = 0.0f;

    for (int hf = 0; hf < 2; ++hf) {
        const int o  = hf * 128;     // weight/activation base offset within block
        const int g  = hf * 8;       // weight-scale group base (sc index)
        const int ag = hf * 4;       // activation-scale group base (asc index)

        unroll_for (int pr = 0; pr < 2; ++pr) {
            const int ql_off = pr * 32;            // pr=0 -> ql[l] (streams 0,2); pr=1 -> ql[l+32] (streams 1,3)
            const int sh_lo  = pr * 2;             // qh shift for the low-nibble stream  (0 or 2)
            const int sh_hi  = pr * 2 + 4;         // qh shift for the high-nibble stream (4 or 6)
            const __local char* act_lo = aq + o + pr * 32;        // low-nibble stream  (stream pr)
            const __local char* act_hi = aq + o + pr * 32 + 64;   // high-nibble stream (stream pr+2)

            int SloA = 0, AloA = 0, ShiA = 0, AhiA = 0;   // low-nibble stream: lo16 + hi16 sc groups
            int SloC = 0, AloC = 0, ShiC = 0, AhiC = 0;   // high-nibble stream
            unroll_for (int l = 0; l < 32; l += 16) {
                const uint4 QL = as_uint4(vload16(0, ql + ql_off + l));
                const uint4 QH = as_uint4(vload16(0, qh + l));
                const uint4 AA = as_uint4(vload16(0, (const __local uchar*)act_lo + l));
                const uint4 AC = as_uint4(vload16(0, (const __local uchar*)act_hi + l));
                if (l < 16) {
                    MOE_DP_Q6K_STEP(QL.x, QH.x, AA.x, AC.x, SloA, AloA, SloC, AloC, sh_lo, sh_hi);
                    MOE_DP_Q6K_STEP(QL.y, QH.y, AA.y, AC.y, SloA, AloA, SloC, AloC, sh_lo, sh_hi);
                    MOE_DP_Q6K_STEP(QL.z, QH.z, AA.z, AC.z, SloA, AloA, SloC, AloC, sh_lo, sh_hi);
                    MOE_DP_Q6K_STEP(QL.w, QH.w, AA.w, AC.w, SloA, AloA, SloC, AloC, sh_lo, sh_hi);
                } else {
                    MOE_DP_Q6K_STEP(QL.x, QH.x, AA.x, AC.x, ShiA, AhiA, ShiC, AhiC, sh_lo, sh_hi);
                    MOE_DP_Q6K_STEP(QL.y, QH.y, AA.y, AC.y, ShiA, AhiA, ShiC, AhiC, sh_lo, sh_hi);
                    MOE_DP_Q6K_STEP(QL.z, QH.z, AA.z, AC.z, ShiA, AhiA, ShiC, AhiC, sh_lo, sh_hi);
                    MOE_DP_Q6K_STEP(QL.w, QH.w, AA.w, AC.w, ShiA, AhiA, ShiC, AhiC, sh_lo, sh_hi);
                }
            }
            // low-nibble stream = stream pr; high-nibble stream = stream pr+2.
            acc += d * asc[ag + pr]     * ((float)sc[g + 2 * pr]     * (float)(SloA - 32 * AloA)
                                         + (float)sc[g + 2 * pr + 1] * (float)(ShiA - 32 * AhiA));
            acc += d * asc[ag + pr + 2] * ((float)sc[g + 2 * pr + 4] * (float)(SloC - 32 * AloC)
                                         + (float)sc[g + 2 * pr + 5] * (float)(ShiC - 32 * AhiC));
        }

        ql += 64; qh += 32;
    }
    return acc;
}

// (v5) moe_gguf_lane_rowdot_dp4a removed: the (row,block) gemv drivers use moe_dp_one_block directly.

// Cooperatively quantize the SLM half activation x2[K] into signed int8 xq[K] + per-32 f32 scale
// xsc[K/32] (symmetric absmax/127). Call after the x2 barrier; caller must barrier before use.
inline void FUNC(moe_gguf_quant_act_i8)(const __local half* x2, __local char* xq, __local float* xsc, __local int* xsum, int K) {
    const int tid = get_sub_group_id() * SUBGROUP_SIZE + get_sub_group_local_id();
    const int tot = get_num_sub_groups() * SUBGROUP_SIZE;
    for (int g = tid; g < K / 32; g += tot) {
        const int base = g * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i)
            amax = fmax(amax, fabs((float)x2[base + i]));
        const float scale = amax * (1.0f / 127.0f);
        const float inv   = amax > 0.0f ? (127.0f / amax) : 0.0f;
        xsc[g] = scale;
        int s = 0;
        for (int i = 0; i < 32; ++i) {
            const int q = clamp(convert_int_rte((float)x2[base + i] * inv), -127, 127);
            xq[base + i] = (char)q;
            s += q;
        }
        xsum[g] = s;   // v5: per-group activation sum for the Q4_K/Q5_K min-fold (kills all-ones dp4a)
    }
}

// v2: decode one full 256-elem GGUF block (dispatch by code). Shared by the row-block gemv drivers.
// v5: axsum = per-group activation sums for this block (Q4_K/Q5_K min-fold); ignored by Q8_0/Q6_K.
inline float FUNC(moe_dp_one_block)(int code, const __global uchar* blk, const __local char* aq, const __local float* asc, const __local int* axsum) {
    if (code == MOE_GGUF_Q8_0) return FUNC_CALL(moe_dp_dot_q8_0)(blk, aq, asc);
    else if (code == MOE_GGUF_Q4_K) return FUNC_CALL(moe_dp_dot_q4_k)(blk, aq, asc);
    else if (code == MOE_GGUF_Q5_K) return FUNC_CALL(moe_dp_dot_q5_k)(blk, aq, asc, axsum);
    else return FUNC_CALL(moe_dp_dot_q6_k)(blk, aq, asc);
}

// gate/up GEMV using int8-activation dp4a.
// [MAPELITES gate_up variant D2: memory_opt=2, low register pressure] per-row fused gate+up. Reads the
// up AND gate blocks of one row interleaved (2 concurrent DRAM streams = the memory-level-parallelism
// win) but processes rows SEQUENTIALLY with just 2 scalar accumulators -> avoids the 2*N_BLOCK acc[]
// register blowup of the unit-striped fused variant that regressed the shared-expert gate_up kernel.
inline void FUNC(moe_gguf_gate_up_gemv_fused)(int code, const __global uchar* up_base, const __global uchar* gate_base,
                                              __global half* y, int N, int K,
                                              const __local char* xq, const __local float* xsc, const __local int* xsum, const __local half* x2) {
    const int lane = get_sub_group_local_id();
    const int n_start = get_global_id(2) * N_BLOCK;
    const long rb = FUNC_CALL(moe_gguf_row_bytes)(code, K);
    const int be = FUNC_CALL(moe_gguf_blk_elem)(code);
    const int bb = FUNC_CALL(moe_gguf_blk_bytes)(code);
    const int nblk = K / be;
    unroll_for (int r = 0; r < N_BLOCK; ++r) {
        const int n = n_start + r;
        float up_p = 0.0f, gate_p = 0.0f;
        for (int b = lane; b < nblk; b += SUBGROUP_SIZE) {
            const long off = (long)n * rb + (long)b * bb;
            const __local char* aqb = xq + b * be;
            const __local float* ascb = xsc + b * (be / 32);
            const __local int* axsb = xsum + b * (be / 32);
            up_p   += FUNC_CALL(moe_dp_one_block)(code, up_base + off, aqb, ascb, axsb);
            gate_p += FUNC_CALL(moe_dp_one_block)(code, gate_base + off, aqb, ascb, axsb);
        }
        float su = sub_group_reduce_add(up_p);
        float sg = sub_group_reduce_add(gate_p);
        if (lane == 0)
            y[n] = MOE_GATE_ACT(sg) * su;
    }
}

// down GEMV using int8-activation dp4a.
// v2: same (row, block) unit-striping across all 32 lanes as gate_up.
inline void FUNC(moe_gguf_down_gemv_dp4a)(int code, const __global uchar* row_base, MOE_DTYPE routing_weight_val,
                                          __global half* y, int N, int K, const __local char* xq, const __local float* xsc, const __local int* xsum, const __local half* x2) {
    const int lane = get_sub_group_local_id();
    const int n_start = get_global_id(2) * N_BLOCK;
    const long rb = FUNC_CALL(moe_gguf_row_bytes)(code, K);
    const int be = FUNC_CALL(moe_gguf_blk_elem)(code);
    const int bb = FUNC_CALL(moe_gguf_blk_bytes)(code);
    const int nblk = K / be;
    const int units = N_BLOCK * nblk;
    float acc[N_BLOCK];
    unroll_for (int r = 0; r < N_BLOCK; ++r) acc[r] = 0.0f;
    for (int u = lane; u < units; u += SUBGROUP_SIZE) {
        const int b = u % nblk;
        const int r = u / nblk;
        const __global uchar* blk = row_base + (long)(n_start + r) * rb + (long)b * bb;
        acc[r] += FUNC_CALL(moe_dp_one_block)(code, blk, xq + b * be, xsc + b * (be / 32), xsum + b * (be / 32));
    }
    unroll_for (int r = 0; r < N_BLOCK; ++r) {
        float sum = sub_group_reduce_add(acc[r]);
        if (lane == 0)
            y[n_start + r] = sum * routing_weight_val;
    }
}
#endif  // MOE_GGUF_DP4A

#if MOE_GGUF_LISA
// =============================================================================================
// LISA-packed coalesced decode (opt-in OV_GPU_GGUF_MOE_LISA=1). The expert weights were byte-
// transposed at the transformation stage (RepackGGUFMoEWeights) into an N-grouped, lane-interleaved
// layout: per expert, rows are grouped by SUBGROUP_SIZE(=32); within a group, byte i of block b of
// the 32 rows are contiguous. The decode runs ONE OUTPUT ROW PER LANE (32 rows / subgroup, no cross-
// lane reduce), so (a) all 32 lanes are busy even though routed rows have only 2-8 blocks each, and
// (b) each lane stages its block with a coalesced load (adjacent lanes read adjacent bytes). Weights
// decode bit-exactly (pure byte permutation). Requires SUBGROUP_SIZE == the pack group size (32).
// =============================================================================================
#define MOE_GGUF_MAX_BLK_BYTES 210  // Q6_K block byte size (largest of Q4_K/Q5_K/Q6_K/Q8_0)

inline half FUNC(moe_gguf_load_f16p)(const uchar* p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(u);
}
inline void FUNC(moe_gguf_scale_min_k4p)(int j, const uchar* q, uchar* d, uchar* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uchar)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uchar)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

inline float FUNC(moe_gguf_dot_q8_0_p)(const uchar* blk, const __local half* xx) {
    const float d = (float)FUNC_CALL(moe_gguf_load_f16p)(blk);
    const char* qs = (const char*)(blk + 2);
    float acc = 0.0f;
    for (int j = 0; j < 32; ++j)
        acc += (float)xx[j] * ((float)qs[j] * d);
    return acc;
}
inline float FUNC(moe_gguf_dot_q4_k_p)(const uchar* blk, const __local half* xx) {
    const float d    = (float)FUNC_CALL(moe_gguf_load_f16p)(blk);
    const float dmin = (float)FUNC_CALL(moe_gguf_load_f16p)(blk + 2);
    const uchar* scales = blk + 4;
    const uchar* qs     = blk + 16;
    float acc = 0.0f;
    int ai = 0, is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(moe_gguf_scale_min_k4p)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(moe_gguf_scale_min_k4p)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        float sq1 = 0.0f, sa1 = 0.0f, sq2 = 0.0f, sa2 = 0.0f;
        for (int l = 0; l < 32; ++l) { const float av = (float)xx[ai + l];      sq1 += av * (float)(qs[l] & 0x0F); sa1 += av; }
        for (int l = 0; l < 32; ++l) { const float av = (float)xx[ai + 32 + l]; sq2 += av * (float)(qs[l] >> 4);   sa2 += av; }
        acc += d1 * sq1 - m1 * sa1 + d2 * sq2 - m2 * sa2;
        qs += 32; is += 2; ai += 64;
    }
    return acc;
}
inline float FUNC(moe_gguf_dot_q5_k_p)(const uchar* blk, const __local half* xx) {
    const float d    = (float)FUNC_CALL(moe_gguf_load_f16p)(blk);
    const float dmin = (float)FUNC_CALL(moe_gguf_load_f16p)(blk + 2);
    const uchar* scales = blk + 4;
    const uchar* qh     = blk + 16;
    const uchar* ql     = blk + 48;
    float acc = 0.0f;
    int ai = 0, is = 0; uchar u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(moe_gguf_scale_min_k4p)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(moe_gguf_scale_min_k4p)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        float sq1 = 0.0f, sa1 = 0.0f, sq2 = 0.0f, sa2 = 0.0f;
        for (int l = 0; l < 32; ++l) { const float av = (float)xx[ai + l];      const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0); sq1 += av * (float)q; sa1 += av; }
        for (int l = 0; l < 32; ++l) { const float av = (float)xx[ai + 32 + l]; const int q = (int)(ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0); sq2 += av * (float)q; sa2 += av; }
        acc += d1 * sq1 - m1 * sa1 + d2 * sq2 - m2 * sa2;
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2; ai += 64;
    }
    return acc;
}
inline float FUNC(moe_gguf_dot_q6_k_p)(const uchar* blk, const __local half* xx) {
    const uchar* ql = blk;
    const uchar* qh = blk + 128;
    const char*  sc = (const char*)(blk + 192);
    const float d = (float)FUNC_CALL(moe_gguf_load_f16p)(blk + 208);
    float acc = 0.0f;
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc += (float)xx[o + l + 0]  * (d * (float)sc[is + 0] * q1);
            acc += (float)xx[o + l + 32] * (d * (float)sc[is + 2] * q2);
            acc += (float)xx[o + l + 64] * (d * (float)sc[is + 4] * q3);
            acc += (float)xx[o + l + 96] * (d * (float)sc[is + 6] * q4);
        }
        o += 128; ql += 64; qh += 32; sc += 8;
    }
    return acc;
}

// Full-row dot for one lane's output row against the SLM activation x2, reading the LISA-packed
// expert weight with a coalesced per-block staging load. `gbase` is this row-group's byte base.
inline float FUNC(moe_gguf_lisa_rowdot)(int code, const __global uchar* gbase, int lane, const __local half* x2, int K) {
    const int be = FUNC_CALL(moe_gguf_blk_elem)(code);
    const int bb = FUNC_CALL(moe_gguf_blk_bytes)(code);
    const int bpr = K / be;
    float acc = 0.0f;
    uchar blk[MOE_GGUF_MAX_BLK_BYTES];
    for (int b = 0; b < bpr; b++) {
        const __global uchar* blane = gbase + (long)b * bb * SUBGROUP_SIZE + lane;
        for (int i = 0; i < bb; i++)
            blk[i] = blane[(long)i * SUBGROUP_SIZE];  // 32 lanes read 32 consecutive bytes -> coalesced
        const __local half* xx = x2 + b * be;
        if (code == MOE_GGUF_Q8_0)
            acc += FUNC_CALL(moe_gguf_dot_q8_0_p)(blk, xx);
        else if (code == MOE_GGUF_Q4_K)
            acc += FUNC_CALL(moe_gguf_dot_q4_k_p)(blk, xx);
        else if (code == MOE_GGUF_Q5_K)
            acc += FUNC_CALL(moe_gguf_dot_q5_k_p)(blk, xx);
        else
            acc += FUNC_CALL(moe_gguf_dot_q6_k_p)(blk, xx);
    }
    return acc;
}

// gate/up GEMV, LISA-packed, one output row per lane (no cross-lane reduce). `expert_base` is the
// packed expert weight base; group index = get_global_id(2), lane = row within the 32-row group.
inline void FUNC(moe_gguf_gate_up_gemv_lisa)(int code, const __global uchar* expert_base, __global half* y, int N, int K,
                                             const __local half* x2, const bool silu) {
    const int lane = get_sub_group_local_id();
    const int g = get_global_id(2);
    const int n = g * SUBGROUP_SIZE + lane;
    if (n >= N)
        return;
    const int be = FUNC_CALL(moe_gguf_blk_elem)(code);
    const int bb = FUNC_CALL(moe_gguf_blk_bytes)(code);
    const int bpr = K / be;
    const __global uchar* gbase = expert_base + (long)g * SUBGROUP_SIZE * bpr * bb;
    const float sum = FUNC_CALL(moe_gguf_lisa_rowdot)(code, gbase, lane, x2, K);
    if (silu)
        y[n] *= MOE_GATE_ACT(sum);
    else
        y[n] = sum;
}

// down GEMV, LISA-packed, one output row per lane.
inline void FUNC(moe_gguf_down_gemv_lisa)(int code, const __global uchar* expert_base, MOE_DTYPE routing_weight_val,
                                          __global half* y, int N, int K, const __local half* x2) {
    const int lane = get_sub_group_local_id();
    const int g = get_global_id(2);
    const int n = g * SUBGROUP_SIZE + lane;
    if (n >= N)
        return;
    const int be = FUNC_CALL(moe_gguf_blk_elem)(code);
    const int bb = FUNC_CALL(moe_gguf_blk_bytes)(code);
    const int bpr = K / be;
    const __global uchar* gbase = expert_base + (long)g * SUBGROUP_SIZE * bpr * bb;
    const float sum = FUNC_CALL(moe_gguf_lisa_rowdot)(code, gbase, lane, x2, K);
    y[n] = sum * routing_weight_val;
}
#endif  // MOE_GGUF_LISA

// Per-lane partial dot of one output row (of length K) against the SLM activation x2. Lane `id`
// handles the strided subset of the K/BLOCK_ELEM blocks; the caller reduces across the subgroup.
inline float FUNC(moe_gguf_lane_rowdot)(int code, const __global uchar* row, const __local half* x2, int K) {
    const int id = get_sub_group_local_id();
    const int be = FUNC_CALL(moe_gguf_blk_elem)(code);
    const int bb = FUNC_CALL(moe_gguf_blk_bytes)(code);
    const int nblk = K / be;
    float acc = 0.0f;
    for (int b = id; b < nblk; b += SUBGROUP_SIZE) {
        const __global uchar* blk = row + (long)b * bb;
        const __local half* xx = x2 + b * be;
        if (code == MOE_GGUF_Q8_0)
            acc += FUNC_CALL(moe_gguf_dot_q8_0)(blk, xx);
        else if (code == MOE_GGUF_Q4_K)
            acc += FUNC_CALL(moe_gguf_dot_q4_k)(blk, xx);
        else if (code == MOE_GGUF_Q5_K)
            acc += FUNC_CALL(moe_gguf_dot_q5_k)(blk, xx);
        else
            acc += FUNC_CALL(moe_gguf_dot_q6_k)(blk, xx);
    }
    return acc;
}

// gate/up GEMV over GGUF blocks: N output rows distributed by get_global_id(2)*N_BLOCK, each row
// reduced across the subgroup. `silu` combines gate into the up result already in y (SwiGLU).
inline void FUNC(moe_gguf_gate_up_gemv)(int code, const __global uchar* row_base, __global half* y, int N, int K,
                                        const __local half* x2, const bool silu) {
    const int id_local = get_sub_group_local_id();
    const int n_start = get_global_id(2) * N_BLOCK;
    const int n_end = n_start + N_BLOCK;
    const long rb = FUNC_CALL(moe_gguf_row_bytes)(code, K);
    for (int n = n_start; n < n_end; n++) {
        float partial = FUNC_CALL(moe_gguf_lane_rowdot)(code, row_base + (long)n * rb, x2, K);
        float sum = sub_group_reduce_add(partial);
        if (id_local == 0) {
            if (silu)
                y[n] *= MOE_GATE_ACT(sum);
            else
                y[n] = sum;
        }
    }
}

// down GEMV over GGUF blocks: N output rows, each scaled by routing_weight_val.
inline void FUNC(moe_gguf_down_gemv)(int code, const __global uchar* row_base, MOE_DTYPE routing_weight_val,
                                     __global half* y, int N, int K, const __local half* x2) {
    const int id_local = get_sub_group_local_id();
    const int n_start = get_global_id(2) * N_BLOCK;
    const int n_end = n_start + N_BLOCK;
    const long rb = FUNC_CALL(moe_gguf_row_bytes)(code, K);
    for (int n = n_start; n < n_end; n++) {
        float partial = FUNC_CALL(moe_gguf_lane_rowdot)(code, row_base + (long)n * rb, x2, K);
        float sum = sub_group_reduce_add(partial);
        if (id_local == 0)
            y[n] = sum * routing_weight_val;
    }
}

#endif  // WEIGHT_COMPRESSEION_DT == 3

#if GATE_UP_ENABLE
inline void gate_up_gemv_n2x_u4(const __global uchar* weight,
                                __global half* scales,
                                __global uchar* zps,
                                __global half* y,
                                int N,
                                int K,
                                half* x2,
                                float* xg_sum,
                                const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        __global half* S = scales + n;
#if HAS_ZP
        __global uchar* Z = zps + n / 2;
#endif
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
#if HAS_ZP
            int zp_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N / 2;
            uchar z = Z[zp_offset];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);
#endif

#    if ELEMS_PER_LANE == 4
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(_sub_group_block_read_slm_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b.s1)), 0);
            sum0.s0 = fma(a.s2, (DEQUANT_4BIT_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s3, (DEQUANT_4BIT_HI(b.s1)), sum0.s1);

            sum1.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b2.s1)), 0);
            sum1.s0 = fma(a.s2, (DEQUANT_4BIT_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s3, (DEQUANT_4BIT_HI(b2.s1)), sum1.s1);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1]) * s0;
            sum_all1 += (sum1[0] + sum1[1]) * s1;
#endif
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 K-elements (1 byte = 2 nibbles).
            half sum0;
            half sum1;
            half2 a = as_half2(_sub_group_block_read_slm_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar b = intel_sub_group_block_read_uc((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar b2 = intel_sub_group_block_read_uc((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0 = fma(a.s0, (DEQUANT_4BIT_LO(b)), (half)0);
            sum0 = fma(a.s1, (DEQUANT_4BIT_HI(b)), sum0);

            sum1 = fma(a.s0, (DEQUANT_4BIT_LO(b2)), (half)0);
            sum1 = fma(a.s1, (DEQUANT_4BIT_HI(b2)), sum1);

#if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#endif
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(_sub_group_block_read_slm_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b.s1)), 0);
            sum0.s2 = fma(a.s2, (DEQUANT_4BIT_LO(b.s2)), 0);
            sum0.s3 = fma(a.s3, (DEQUANT_4BIT_LO(b.s3)), 0);

            sum0.s0 = fma(a.s4, (DEQUANT_4BIT_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s5, (DEQUANT_4BIT_HI(b.s1)), sum0.s1);
            sum0.s2 = fma(a.s6, (DEQUANT_4BIT_HI(b.s2)), sum0.s2);
            sum0.s3 = fma(a.s7, (DEQUANT_4BIT_HI(b.s3)), sum0.s3);

            sum1.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b2.s1)), 0);
            sum1.s2 = fma(a.s2, (DEQUANT_4BIT_LO(b2.s2)), 0);
            sum1.s3 = fma(a.s3, (DEQUANT_4BIT_LO(b2.s3)), 0);

            sum1.s0 = fma(a.s4, (DEQUANT_4BIT_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s5, (DEQUANT_4BIT_HI(b2.s1)), sum1.s1);
            sum1.s2 = fma(a.s6, (DEQUANT_4BIT_HI(b2.s2)), sum1.s2);
            sum1.s3 = fma(a.s7, (DEQUANT_4BIT_HI(b2.s3)), sum1.s3);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#endif
#    endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= MOE_GATE_ACT(sum_all0);
                y[n + 1] *= MOE_GATE_ACT(sum_all1);
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

inline void gate_up_gemv_n2x_u8(const __global uchar* weight,
                                __global half* scales,
                                __global uchar* zps,
                                __global half* y,
                                int N,
                                int K,
                                half* x2,
                                float* xg_sum,
                                const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K;
        float sum_all0 = 0;
        float sum_all1 = 0;
        __global half* S = scales + n;
#if HAS_ZP
        __global uchar* Z = zps + n;
#endif
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
#if HAS_ZP
            int zp_offset = (gk * FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            half z0 = convert_half(Z[zp_offset]);
            half z1 = convert_half(Z[zp_offset + 1]);
#endif

#    if ELEMS_PER_LANE == 4
            float2 sum0;
            float2 sum1;
            half4 a = as_half4(_sub_group_block_read_slm_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), 0.0f);
            sum0.s0 = fma((float)a.s2, (float)(DEQUANT_8BIT(b.s2)), sum0.s0);
            sum0.s1 = fma((float)a.s3, (float)(DEQUANT_8BIT(b.s3)), sum0.s1);

            sum1.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), 0.0f);
            sum1.s0 = fma((float)a.s2, (float)(DEQUANT_8BIT(b2.s2)), sum1.s0);
            sum1.s1 = fma((float)a.s3, (float)(DEQUANT_8BIT(b2.s3)), sum1.s1);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1]) * s0;
            sum_all1 += (sum1[0] + sum1[1]) * s1;
#endif
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 K-elements (8-bit weights, 1 byte each).
            float sum0;
            float sum1;
            half2 a = as_half2(_sub_group_block_read_slm_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), sum0);

            sum1 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), sum1);

#if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#endif
#    else
            float4 sum0;
            float4 sum1;
            half8 a = as_half8(_sub_group_block_read_slm_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar8 b = intel_sub_group_block_read_uc8((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar8 b2 = intel_sub_group_block_read_uc8((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), 0.0f);
            sum0.s2 = fma((float)a.s2, (float)(DEQUANT_8BIT(b.s2)), 0.0f);
            sum0.s3 = fma((float)a.s3, (float)(DEQUANT_8BIT(b.s3)), 0.0f);

            sum0.s0 = fma((float)a.s4, (float)(DEQUANT_8BIT(b.s4)), sum0.s0);
            sum0.s1 = fma((float)a.s5, (float)(DEQUANT_8BIT(b.s5)), sum0.s1);
            sum0.s2 = fma((float)a.s6, (float)(DEQUANT_8BIT(b.s6)), sum0.s2);
            sum0.s3 = fma((float)a.s7, (float)(DEQUANT_8BIT(b.s7)), sum0.s3);

            sum1.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), 0.0f);
            sum1.s2 = fma((float)a.s2, (float)(DEQUANT_8BIT(b2.s2)), 0.0f);
            sum1.s3 = fma((float)a.s3, (float)(DEQUANT_8BIT(b2.s3)), 0.0f);

            sum1.s0 = fma((float)a.s4, (float)(DEQUANT_8BIT(b2.s4)), sum1.s0);
            sum1.s1 = fma((float)a.s5, (float)(DEQUANT_8BIT(b2.s5)), sum1.s1);
            sum1.s2 = fma((float)a.s6, (float)(DEQUANT_8BIT(b2.s6)), sum1.s2);
            sum1.s3 = fma((float)a.s7, (float)(DEQUANT_8BIT(b2.s7)), sum1.s3);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#endif
#    endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= MOE_GATE_ACT(sum_all0);
                y[n + 1] *= MOE_GATE_ACT(sum_all1);
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

inline void gate_up_gemv_n2x_f16(const __global half* weight, __global half* y, int N, int K, half* x2, const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global half* B = weight + n * K;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
#    if ELEMS_PER_LANE == 4
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half4 b = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half4 b2 = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s0 = fma(a.s2, b.s2, sum0.s0);
            sum0.s1 = fma(a.s3, b.s3, sum0.s1);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s0 = fma(a.s2, b2.s2, sum1.s0);
            sum1.s1 = fma(a.s3, b2.s3, sum1.s1);

            sum_all0 += sum0[0] + sum0[1];
            sum_all1 += sum1[0] + sum1[1];
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 fp16 elements.
            half sum0;
            half sum1;
            half2 a = as_half2(intel_sub_group_block_read_us2((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half2 b = as_half2(intel_sub_group_block_read_us2((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half2 b2 = as_half2(intel_sub_group_block_read_us2((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0 = fma(a.s0, b.s0, (half)0);
            sum0 = fma(a.s1, b.s1, sum0);

            sum1 = fma(a.s0, b2.s0, (half)0);
            sum1 = fma(a.s1, b2.s1, sum1);

            sum_all0 += sum0;
            sum_all1 += sum1;
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half8 b = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half8 b2 = as_half8(intel_sub_group_block_read_us8((const __global ushort*)(B + K + gk * FAKE_GROUP_SIZE)));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s2 = fma(a.s2, b.s2, 0);
            sum0.s3 = fma(a.s3, b.s3, 0);

            sum0.s0 = fma(a.s4, b.s4, sum0.s0);
            sum0.s1 = fma(a.s5, b.s5, sum0.s1);
            sum0.s2 = fma(a.s6, b.s6, sum0.s2);
            sum0.s3 = fma(a.s7, b.s7, sum0.s3);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s2 = fma(a.s2, b2.s2, 0);
            sum1.s3 = fma(a.s3, b2.s3, 0);

            sum1.s0 = fma(a.s4, b2.s4, sum1.s0);
            sum1.s1 = fma(a.s5, b2.s5, sum1.s1);
            sum1.s2 = fma(a.s6, b2.s6, sum1.s2);
            sum1.s3 = fma(a.s7, b2.s7, sum1.s3);

            sum_all0 += sum0[0] + sum0[1] + sum0[2] + sum0[3];
            sum_all1 += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#    endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= MOE_GATE_ACT(sum_all0);
                y[n + 1] *= MOE_GATE_ACT(sum_all1);
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(mlp_gate_up)(
    const __global int* expert_list,
    const __global MOE_WEI_DT* gate_weight_addr,
    const __global MOE_SCALE_DT* gate_scale_addr,
    const __global MOE_ZP_DT* gate_zp_addr,
    const __global MOE_WEI_DT* up_weight_addr,
    const __global MOE_SCALE_DT* up_scale_addr,
    const __global MOE_ZP_DT* up_zp_addr,
#    if SHARED_EXPERT_ENABLE
    const __global MOE_WEI_DT* shared_gate_weight,
    const __global MOE_SCALE_DT* shared_gate_scale,
    const __global MOE_ZP_DT* shared_gate_zp,
    const __global MOE_WEI_DT* shared_up_weight,
    const __global MOE_SCALE_DT* shared_up_scale,
    const __global MOE_ZP_DT* shared_up_zp,
    const __global half* shared_gate_gate_weight,  // [HIDDEN_SIZE] (assuming no scale/zp for now, or pre-dequantized)
    __global MOE_DTYPE* shared_gate_out,           // [token_num] — output: sigmoid(dot(x, gate_weight)) for shared expert
#    endif
    __global MOE_DTYPE* x,    // [token_num, HIDDEN_SIZE]
    __global MOE_DTYPE* y) {  // [token_num * EXPERTS_PER_TOKEN, INTERMEDIATE_SIZE]
    // global: [token_num*EXPERTS_PER_TOKEN, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    // Batched GEMV: each workgroup handles one (token, expert) pair.
    // For single token (token_num=1), flat_id == expert_slot and token_idx == 0.
    int flat_id = get_global_id(0);
    int token_idx = flat_id / EXPERTS_PER_TOKEN;
    int expert_slot = flat_id % EXPERTS_PER_TOKEN;
    y += flat_id * INTERMEDIATE_SIZE;

    // Check if we are processing the Shared Expert
#    if SHARED_EXPERT_ENABLE
    bool is_shared = (expert_slot == MAX_TOPK);
#    else
    bool is_shared = false;
#    endif

#    if WEIGHT_COMPRESSEION_DT == 0
    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / GATE_UP_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2 / GATE_UP_GROUP_SIZE;
#    elif WEIGHT_COMPRESSEION_DT == 3
    // GGUF: per-expert byte size of the routed gate/up projection (Q4_K); scale/zp are unused.
    const int expert_wei_size = (int)FUNC_CALL(moe_gguf_expert_bytes)(GATEUP_DECODE, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    const int expert_scale_size = 0;
    const int expert_zp_size = 0;
#    else
    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / GATE_UP_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / GATE_UP_GROUP_SIZE;
#    endif

    int expert_id = 0;
    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global MOE_WEI_DT* gate_weight;
    __global MOE_SCALE_DT* gate_scale;
    __global MOE_ZP_DT* gate_zp;
    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global MOE_WEI_DT* up_weight;
    __global MOE_SCALE_DT* up_scale;
    __global MOE_ZP_DT* up_zp;

    if (!is_shared) {
        expert_id = expert_list[token_idx * MAX_TOPK + expert_slot];
        // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
        gate_weight = (__global MOE_WEI_DT*)(gate_weight_addr + expert_id * expert_wei_size);
        gate_scale = (__global MOE_SCALE_DT*)(gate_scale_addr + expert_id * expert_scale_size);
        gate_zp = (__global MOE_ZP_DT*)(gate_zp_addr + expert_id * expert_zp_size);

        // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
        up_weight = (__global MOE_WEI_DT*)(up_weight_addr + expert_id * expert_wei_size);
        up_scale = (__global MOE_SCALE_DT*)(up_scale_addr + expert_id * expert_scale_size);
        up_zp = (__global MOE_ZP_DT*)(up_zp_addr + expert_id * expert_zp_size);
    }
#    if SHARED_EXPERT_ENABLE
    else {
        // Use shared expert pointers directly
        // [HIDDEN_SIZE, SHARED_INTERMEDIATE_SIZE] - assume layout/size match sparse experts for now
        gate_weight = (__global MOE_WEI_DT*)shared_gate_weight;
        gate_scale = (__global MOE_SCALE_DT*)shared_gate_scale;
        gate_zp = (__global MOE_ZP_DT*)shared_gate_zp;
        up_weight = (__global MOE_WEI_DT*)shared_up_weight;
        up_scale = (__global MOE_SCALE_DT*)shared_up_scale;
        up_zp = (__global MOE_ZP_DT*)shared_up_zp;
    }
#    endif

#    if GATE_UP_GROUP_SIZE % FAKE_GROUP_SIZE != 0
    if (get_sub_group_id() == 0 && get_sub_group_local_id() == 0) {
        printf("GATE_UP_GROUP_SIZE(%d) must be a multiple of FAKE_GROUP_SIZE(%d)", GATE_UP_GROUP_SIZE, FAKE_GROUP_SIZE);
    }
    return;
#    endif

    __local half x2[HIDDEN_SIZE];
#if HAS_ZP
    __local float xg_sum[HIDDEN_SIZE / FAKE_GROUP_SIZE];
#else
    __local float xg_sum[1];  // unused placeholder for function signature
#endif
#if MOE_GGUF_DP4A && WEIGHT_COMPRESSEION_DT == 3
    __local char  xq2[HIDDEN_SIZE];        // int8 activation for dp4a
    __local float xsc2[HIDDEN_SIZE / 32];  // per-32 activation scale
    __local int   xsum2[HIDDEN_SIZE / 32]; // v5: per-32 activation int sum (Q4_K/Q5_K min-fold)
#endif
#    if SHARED_EXPERT_ENABLE
    __local float shared_gate_partial[SUBGROUP_NUM];  // one slot per subgroup for scalar gate reduction
#    endif

#    if WEIGHT_COMPRESSEION_DT == 0
    //# interleaving x into x2
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + token_idx * HIDDEN_SIZE + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < HIDDEN_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
#if HAS_ZP
        float x_group_sum = 0;
#endif
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE / 2; j += SUBGROUP_SIZE) {
            half even = px[2 * j + 0];
            half odd = px[2 * j + 1];
            px2[j] = even;
            px2[j + FAKE_GROUP_SIZE / 2] = odd;
#if HAS_ZP
            x_group_sum += even + odd;
#endif
        }
#if HAS_ZP
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
#endif
    }
#    else
    //# load x into slm
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + token_idx * HIDDEN_SIZE + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < HIDDEN_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
#if HAS_ZP
        float x_group_sum = 0;
#endif
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE; j += SUBGROUP_SIZE) {
            half value = px[j];
            px2[j] = value;
#if HAS_ZP
            x_group_sum += value;
#endif
        }
#if HAS_ZP
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
#endif
    }
#    endif

    barrier(CLK_LOCAL_MEM_FENCE);

#if MOE_GGUF_DP4A && WEIGHT_COMPRESSEION_DT == 3
    FUNC_CALL(moe_gguf_quant_act_i8)(x2, xq2, xsc2, xsum2, HIDDEN_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#    if SHARED_EXPERT_ENABLE
    // Compute scalar gate for shared expert using all threads in the workgroup.
    // Only the N-block-0 workgroup writes shared_gate_out[token_idx]; other N-block workgroups
    // skip this entirely to avoid redundant work and races.
    if (is_shared && get_group_id(2) == 0) {
        // Step 1: every thread accumulates a partial dot product over its slice of HIDDEN_SIZE.
        // With num_sg subgroups * SUBGROUP_SIZE threads, each thread handles ~14 elements
        // (HIDDEN_SIZE=3584 / 256 threads) instead of one thread doing all 3584.
        float gate_val = 0.0f;
        int thread_id = id_sg * SUBGROUP_SIZE + id_local;
        int total_threads = num_sg * SUBGROUP_SIZE;
        for (int i = thread_id; i < HIDDEN_SIZE; i += total_threads) {
            gate_val += (float)x[token_idx * HIDDEN_SIZE + i] * (float)shared_gate_gate_weight[i];
        }

        // Step 2: intra-subgroup reduction.
        gate_val = sub_group_reduce_add(gate_val);

        // Step 3: store per-subgroup partial to SLM.
        if (id_local == 0) {
            shared_gate_partial[id_sg] = gate_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Step 4: subgroup 0 does final cross-subgroup reduction and writes sigmoid result.
        if (id_sg == 0) {
            float final_val = (id_local < num_sg) ? shared_gate_partial[id_local] : 0.0f;
            final_val = sub_group_reduce_add(final_val);
            if (id_local == 0) {
                shared_gate_out[token_idx] = (MOE_DTYPE)(1.0f / (1.0f + exp(-final_val)));
            }
        }
    }
    // shared_gate_out[token_idx] is consumed by the next kernel (mlp_down) — no barrier needed here.
#    endif

#    if WEIGHT_COMPRESSEION_DT == 0
    gate_up_gemv_n2x_u4(up_weight, up_scale, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
    gate_up_gemv_n2x_u4(gate_weight, gate_scale, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
#    elif WEIGHT_COMPRESSEION_DT == 1
    gate_up_gemv_n2x_u8(up_weight, up_scale, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
    gate_up_gemv_n2x_u8(gate_weight, gate_scale, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
#    elif WEIGHT_COMPRESSEION_DT == 2
    gate_up_gemv_n2x_f16(up_weight, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, false);
    gate_up_gemv_n2x_f16(gate_weight, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, true);
#    elif WEIGHT_COMPRESSEION_DT == 3
    // Routed experts decode Q4_K (GATEUP_DECODE); the shared expert decodes Q8_0 (SHARED_DECODE).
    const int gu_code = is_shared ? SHARED_DECODE : GATEUP_DECODE;
#if MOE_GGUF_LISA
    FUNC_CALL(moe_gguf_gate_up_gemv_lisa)(gu_code, (const __global uchar*)up_weight, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, false);
    FUNC_CALL(moe_gguf_gate_up_gemv_lisa)(gu_code, (const __global uchar*)gate_weight, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, true);
#elif MOE_GGUF_DP4A
    FUNC_CALL(moe_gguf_gate_up_gemv_fused)(gu_code, (const __global uchar*)up_weight, (const __global uchar*)gate_weight, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, xq2, xsc2, xsum2, x2);
#else
    FUNC_CALL(moe_gguf_gate_up_gemv)(gu_code, (const __global uchar*)up_weight, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, false);
    FUNC_CALL(moe_gguf_gate_up_gemv)(gu_code, (const __global uchar*)gate_weight, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, true);
#endif
#    endif
}

#elif DOWN_ENABLE

inline void down_gemv_n2x_u4(const __global uchar* weight,
                             __global half* scales,
                             __global uchar* zps,
                             MOE_DTYPE routing_weight_val,
                             __global half* y,
                             int N,
                             int K,
                             half* x2,
                             float* xg_sum) {
    int id_local = get_sub_group_local_id();
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
#if HAS_ZP
        __global uchar* Z = zps + n / 2;
#endif
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
#if HAS_ZP
            int zp_offset = (gk * FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N / 2;
            ushort z = Z[zp_offset];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);
#endif

#    if ELEMS_PER_LANE == 4
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(_sub_group_block_read_slm_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b.s1)), 0);
            sum0.s0 = fma(a.s2, (DEQUANT_4BIT_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s3, (DEQUANT_4BIT_HI(b.s1)), sum0.s1);

            sum1.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b2.s1)), 0);
            sum1.s0 = fma(a.s2, (DEQUANT_4BIT_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s3, (DEQUANT_4BIT_HI(b2.s1)), sum1.s1);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1]) * s0;
            sum_all1 += (sum1[0] + sum1[1]) * s1;
#endif
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 K-elements (1 byte = 2 nibbles).
            half sum0;
            half sum1;
            half2 a = as_half2(_sub_group_block_read_slm_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar b = intel_sub_group_block_read_uc((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar b2 = intel_sub_group_block_read_uc((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0 = fma(a.s0, (DEQUANT_4BIT_LO(b)), (half)0);
            sum0 = fma(a.s1, (DEQUANT_4BIT_HI(b)), sum0);

            sum1 = fma(a.s0, (DEQUANT_4BIT_LO(b2)), (half)0);
            sum1 = fma(a.s1, (DEQUANT_4BIT_HI(b2)), sum1);

#if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#endif
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(_sub_group_block_read_slm_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b.s0)), 0);
            sum0.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b.s1)), 0);
            sum0.s2 = fma(a.s2, (DEQUANT_4BIT_LO(b.s2)), 0);
            sum0.s3 = fma(a.s3, (DEQUANT_4BIT_LO(b.s3)), 0);

            sum0.s0 = fma(a.s4, (DEQUANT_4BIT_HI(b.s0)), sum0.s0);
            sum0.s1 = fma(a.s5, (DEQUANT_4BIT_HI(b.s1)), sum0.s1);
            sum0.s2 = fma(a.s6, (DEQUANT_4BIT_HI(b.s2)), sum0.s2);
            sum0.s3 = fma(a.s7, (DEQUANT_4BIT_HI(b.s3)), sum0.s3);

            sum1.s0 = fma(a.s0, (DEQUANT_4BIT_LO(b2.s0)), 0);
            sum1.s1 = fma(a.s1, (DEQUANT_4BIT_LO(b2.s1)), 0);
            sum1.s2 = fma(a.s2, (DEQUANT_4BIT_LO(b2.s2)), 0);
            sum1.s3 = fma(a.s3, (DEQUANT_4BIT_LO(b2.s3)), 0);

            sum1.s0 = fma(a.s4, (DEQUANT_4BIT_HI(b2.s0)), sum1.s0);
            sum1.s1 = fma(a.s5, (DEQUANT_4BIT_HI(b2.s1)), sum1.s1);
            sum1.s2 = fma(a.s6, (DEQUANT_4BIT_HI(b2.s2)), sum1.s2);
            sum1.s3 = fma(a.s7, (DEQUANT_4BIT_HI(b2.s3)), sum1.s3);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#endif
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weight_val;
            y[n + 1] = sum_all1 * routing_weight_val;
        }
    }
}

inline void down_gemv_n2x_u8(const __global uchar* weight,
                             __global half* scales,
                             __global uchar* zps,
                             MOE_DTYPE routing_weight_val,
                             __global half* y,
                             int N,
                             int K,
                             half* x2,
                             float* xg_sum) {
    int id_local = get_sub_group_local_id();
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K;
        __global half* S = scales + n;
#if HAS_ZP
        __global uchar* Z = zps + n;
#endif
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = (gk * FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
#if HAS_ZP
            int zp_offset = (gk * FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            half z0 = convert_half(Z[zp_offset]);
            half z1 = convert_half(Z[zp_offset + 1]);
#endif

#    if ELEMS_PER_LANE == 4
            float2 sum0;
            float2 sum1;
            half4 a = as_half4(_sub_group_block_read_slm_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)B + K + gk * FAKE_GROUP_SIZE);

            sum0.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), 0.0f);
            sum0.s0 = fma((float)a.s2, (float)(DEQUANT_8BIT(b.s2)), sum0.s0);
            sum0.s1 = fma((float)a.s3, (float)(DEQUANT_8BIT(b.s3)), sum0.s1);

            sum1.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), 0.0f);
            sum1.s0 = fma((float)a.s2, (float)(DEQUANT_8BIT(b2.s2)), sum1.s0);
            sum1.s1 = fma((float)a.s3, (float)(DEQUANT_8BIT(b2.s3)), sum1.s1);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1]) * s0;
            sum_all1 += (sum1[0] + sum1[1]) * s1;
#endif
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 K-elements (8-bit weights, 1 byte each).
            float sum0;
            float sum1;
            half2 a = as_half2(_sub_group_block_read_slm_us2((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)B + K + gk * FAKE_GROUP_SIZE);

            sum0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), sum0);

            sum1 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), sum1);

#if HAS_ZP
            sum_all0 += (sum0 - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1 - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += sum0 * s0;
            sum_all1 += sum1 * s1;
#endif
#    else
            float4 sum0;
            float4 sum1;
            half8 a = as_half8(_sub_group_block_read_slm_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar8 b = intel_sub_group_block_read_uc8((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar8 b2 = intel_sub_group_block_read_uc8((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b.s1)), 0.0f);
            sum0.s2 = fma((float)a.s2, (float)(DEQUANT_8BIT(b.s2)), 0.0f);
            sum0.s3 = fma((float)a.s3, (float)(DEQUANT_8BIT(b.s3)), 0.0f);

            sum0.s0 = fma((float)a.s4, (float)(DEQUANT_8BIT(b.s4)), sum0.s0);
            sum0.s1 = fma((float)a.s5, (float)(DEQUANT_8BIT(b.s5)), sum0.s1);
            sum0.s2 = fma((float)a.s6, (float)(DEQUANT_8BIT(b.s6)), sum0.s2);
            sum0.s3 = fma((float)a.s7, (float)(DEQUANT_8BIT(b.s7)), sum0.s3);

            sum1.s0 = fma((float)a.s0, (float)(DEQUANT_8BIT(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(DEQUANT_8BIT(b2.s1)), 0.0f);
            sum1.s2 = fma((float)a.s2, (float)(DEQUANT_8BIT(b2.s2)), 0.0f);
            sum1.s3 = fma((float)a.s3, (float)(DEQUANT_8BIT(b2.s3)), 0.0f);

            sum1.s0 = fma((float)a.s4, (float)(DEQUANT_8BIT(b2.s4)), sum1.s0);
            sum1.s1 = fma((float)a.s5, (float)(DEQUANT_8BIT(b2.s5)), sum1.s1);
            sum1.s2 = fma((float)a.s6, (float)(DEQUANT_8BIT(b2.s6)), sum1.s2);
            sum1.s3 = fma((float)a.s7, (float)(DEQUANT_8BIT(b2.s7)), sum1.s3);

#if HAS_ZP
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z1) * s1;
#else
            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#endif
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weight_val;
            y[n + 1] = sum_all1 * routing_weight_val;
        }
    }
}

inline void down_gemv_n2x_f16(const __global half* weight, MOE_DTYPE routing_weight_val, __global half* y, int N, int K, half* x2) {
    int id_local = get_sub_group_local_id();
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global half* B = weight + n * K;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {

#    if ELEMS_PER_LANE == 4
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half4 b = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half4 b2 = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s0 = fma(a.s2, b.s2, sum0.s0);
            sum0.s1 = fma(a.s3, b.s3, sum0.s1);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s0 = fma(a.s2, b2.s2, sum1.s0);
            sum1.s1 = fma(a.s3, b2.s3, sum1.s1);

            sum_all0 += sum0[0] + sum0[1];
            sum_all1 += sum1[0] + sum1[1];
#    elif ELEMS_PER_LANE == 2
            // Each lane reads 2 fp16 elements.
            half sum0;
            half sum1;
            half2 a = as_half2(intel_sub_group_block_read_us2((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half2 b = as_half2(intel_sub_group_block_read_us2((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half2 b2 = as_half2(intel_sub_group_block_read_us2((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0 = fma(a.s0, b.s0, (half)0);
            sum0 = fma(a.s1, b.s1, sum0);

            sum1 = fma(a.s0, b2.s0, (half)0);
            sum1 = fma(a.s1, b2.s1, sum1);

            sum_all0 += sum0;
            sum_all1 += sum1;
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(_sub_group_block_read_slm_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half8 b = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half8 b2 = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s2 = fma(a.s2, b.s2, 0);
            sum0.s3 = fma(a.s3, b.s3, 0);

            sum0.s0 = fma(a.s4, b.s4, sum0.s0);
            sum0.s1 = fma(a.s5, b.s5, sum0.s1);
            sum0.s2 = fma(a.s6, b.s6, sum0.s2);
            sum0.s3 = fma(a.s7, b.s7, sum0.s3);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s2 = fma(a.s2, b2.s2, 0);
            sum1.s3 = fma(a.s3, b2.s3, 0);

            sum1.s0 = fma(a.s4, b2.s4, sum1.s0);
            sum1.s1 = fma(a.s5, b2.s5, sum1.s1);
            sum1.s2 = fma(a.s6, b2.s6, sum1.s2);
            sum1.s3 = fma(a.s7, b2.s7, sum1.s3);

            sum_all0 += sum0[0] + sum0[1] + sum0[2] + sum0[3];
            sum_all1 += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weight_val;
            y[n + 1] = sum_all1 * routing_weight_val;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(mlp_down)(const __global int* expert_list,
                                                                           const __global MOE_WEI_DT* down_weight_addr,
                                                                           const __global MOE_SCALE_DT* down_scale_addr,
                                                                           const __global MOE_ZP_DT* down_zp_addr,
#    if SHARED_EXPERT_ENABLE
                                                                           const __global MOE_WEI_DT* shared_down_weight,
                                                                           const __global MOE_SCALE_DT* shared_down_scale,
                                                                           const __global MOE_ZP_DT* shared_down_zp,
#    endif
                                                                           const __global MOE_DTYPE* x,          // [token_num * EXPERTS_PER_TOKEN, INTERMEDIATE_SIZE]
                                                                           const __global MOE_DTYPE* routing_weights,  // [token_num * MAX_TOPK] compact buffer from MoERouterFused
#    if SHARED_EXPERT_ENABLE
                                                                           const __global MOE_DTYPE* shared_gate_in,   // [token_num] separate shared expert gate values
#    endif
                                                                           __global MOE_DTYPE* y) {              // [token_num * EXPERTS_PER_TOKEN, HIDDEN_SIZE]
    // global: [token_num*EXPERTS_PER_TOKEN, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int flat_id = get_global_id(0);
    int token_idx = flat_id / EXPERTS_PER_TOKEN;
    int expert_slot = flat_id % EXPERTS_PER_TOKEN;
    x += flat_id * INTERMEDIATE_SIZE;
    y += flat_id * HIDDEN_SIZE;

#    if SHARED_EXPERT_ENABLE
    bool is_shared = (expert_slot == MAX_TOPK);
#    else
    bool is_shared = false;
#    endif

#    if WEIGHT_COMPRESSEION_DT == 0
    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / DOWN_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2 / DOWN_GROUP_SIZE;
#    elif WEIGHT_COMPRESSEION_DT == 3
    // GGUF: per-expert byte size of the routed down projection [HIDDEN_SIZE, INTERMEDIATE_SIZE].
    const int expert_wei_size = (int)FUNC_CALL(moe_gguf_expert_bytes)(DOWN_DECODE, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    const int expert_scale_size = 0;
    const int expert_zp_size = 0;
#    else
    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / DOWN_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / DOWN_GROUP_SIZE;
#    endif
    int expert_id = 0;

    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global MOE_WEI_DT* weight;
    __global MOE_SCALE_DT* scales;
    __global MOE_ZP_DT* zps;

    if (!is_shared) {
        expert_id = expert_list[token_idx * MAX_TOPK + expert_slot];
        // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
        weight = (__global MOE_WEI_DT*)(down_weight_addr + expert_id * expert_wei_size);
        scales = (__global MOE_SCALE_DT*)(down_scale_addr + expert_id * expert_scale_size);
        zps = (__global MOE_ZP_DT*)(down_zp_addr + expert_id * expert_zp_size);
    }
#    if SHARED_EXPERT_ENABLE
    else {
        weight = (__global MOE_WEI_DT*)shared_down_weight;
        scales = (__global MOE_SCALE_DT*)shared_down_scale;
        zps = (__global MOE_ZP_DT*)shared_down_zp;
    }
#    endif

#    if DOWN_GROUP_SIZE % FAKE_GROUP_SIZE != 0
    if (get_sub_group_id() == 0 && get_sub_group_local_id() == 0) {
        printf("DOWN_GROUP_SIZE(%d) must be a multiple of FAKE_GROUP_SIZE(%d)", DOWN_GROUP_SIZE, FAKE_GROUP_SIZE);
    }
    return;
#    endif

    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;

    __local half x2[INTERMEDIATE_SIZE];
#if HAS_ZP
    __local float xg_sum[INTERMEDIATE_SIZE / FAKE_GROUP_SIZE];
#else
    __local float xg_sum[1];  // unused placeholder for function signature
#endif
#if MOE_GGUF_DP4A && WEIGHT_COMPRESSEION_DT == 3
    __local char  xq2[INTERMEDIATE_SIZE];        // int8 activation for dp4a
    __local float xsc2[INTERMEDIATE_SIZE / 32];  // per-32 activation scale
    __local int   xsum2[INTERMEDIATE_SIZE / 32]; // v5: per-32 activation int sum (Q5_K min-fold)
#endif

#    if WEIGHT_COMPRESSEION_DT == 0
    //# interleaving x into x2
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < INTERMEDIATE_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
#if HAS_ZP
        float x_group_sum = 0;
#endif
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE / 2; j += SUBGROUP_SIZE) {
            half even = px[2 * j + 0];
            half odd = px[2 * j + 1];
            px2[j] = even;
            px2[j + FAKE_GROUP_SIZE / 2] = odd;
#if HAS_ZP
            x_group_sum += even + odd;
#endif
        }
#if HAS_ZP
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
#endif
    }
#    else
    //# load x into slm
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < INTERMEDIATE_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
#if HAS_ZP
        float x_group_sum = 0;
#endif
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE; j += SUBGROUP_SIZE) {
            half value = px[j];
            px2[j] = value;
#if HAS_ZP
            x_group_sum += value;
#endif
        }
#if HAS_ZP
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
#endif
    }
#    endif

    barrier(CLK_LOCAL_MEM_FENCE);

#if MOE_GGUF_DP4A && WEIGHT_COMPRESSEION_DT == 3
    FUNC_CALL(moe_gguf_quant_act_i8)(x2, xq2, xsc2, xsum2, INTERMEDIATE_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    // Compute the routing weight for this (token, expert) work item.
    // routing_weights is compact [token_num * MAX_TOPK]; shared expert gate is separate.
#    if SHARED_EXPERT_ENABLE
    MOE_DTYPE routing_weight_val = is_shared ? shared_gate_in[token_idx] : routing_weights[token_idx * MAX_TOPK + expert_slot];
#    else
    MOE_DTYPE routing_weight_val = routing_weights[token_idx * MAX_TOPK + expert_slot];
#    endif

#    if WEIGHT_COMPRESSEION_DT == 0
    down_gemv_n2x_u4(weight, scales, zps, routing_weight_val, y, N, K, x2, xg_sum);
#    elif WEIGHT_COMPRESSEION_DT == 1
    down_gemv_n2x_u8(weight, scales, zps, routing_weight_val, y, N, K, x2, xg_sum);
#    elif WEIGHT_COMPRESSEION_DT == 2
    down_gemv_n2x_f16(weight, routing_weight_val, y, N, K, x2);
#    elif WEIGHT_COMPRESSEION_DT == 3
    // Routed down decodes Q5_K/Q6_K (DOWN_DECODE); the shared expert down decodes Q8_0 (SHARED_DECODE).
    const int down_code = is_shared ? SHARED_DECODE : DOWN_DECODE;
#if MOE_GGUF_LISA
    FUNC_CALL(moe_gguf_down_gemv_lisa)(down_code, (const __global uchar*)weight, routing_weight_val, y, N, K, x2);
#elif MOE_GGUF_DP4A
    FUNC_CALL(moe_gguf_down_gemv_dp4a)(down_code, (const __global uchar*)weight, routing_weight_val, y, N, K, xq2, xsc2, xsum2, x2);
#else
    FUNC_CALL(moe_gguf_down_gemv)(down_code, (const __global uchar*)weight, routing_weight_val, y, N, K, x2);
#endif
#    endif
}

#else
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(mlp_reduce)(const __global MOE_DTYPE* x,  // [token_num * REDUCE_COUNT, HIDDEN_SIZE]
                                                                             __global MOE_DTYPE* y) {      // [token_num, HIDDEN_SIZE]
    // gws={token_num, HIDDEN_SIZE}, lws={1, min(max_wgs, 1024)}
    int token_idx = get_global_id(0);
    int n = get_global_id(1);
#    if SHARED_EXPERT_ENABLE
#        define REDUCE_COUNT (MAX_TOPK + 1)
#    else
#        define REDUCE_COUNT MAX_TOPK
#    endif
    half sum[REDUCE_COUNT] = {0};
    __attribute__((opencl_unroll_hint(REDUCE_COUNT))) for (int i = 0; i < REDUCE_COUNT; i++) {
        sum[i] = as_half(intel_sub_group_block_read_us((const __global ushort*)(x + (token_idx * REDUCE_COUNT + i) * HIDDEN_SIZE + n)));
    }
    for (int i = 1; i < REDUCE_COUNT; i++) {
        sum[0] += sum[i];
    }
    intel_sub_group_block_write_us((__global ushort*)(y + token_idx * HIDDEN_SIZE + n), as_ushort(sum[0]));
}
#endif
