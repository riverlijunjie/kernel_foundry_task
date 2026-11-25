// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// [EVOLVE_START]
// GGUF prefill GEMM (OCL dpas) for Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and Q8_0 (ported from the CM gemm_q*_full.cm kernels).
//
// One source for all six formats: the host defines exactly one GG_FMT_* macro per kernel and gives each format its
// own entry point (fc_gguf_gemm_<fmt>). Each section below is the per-format tile codec (gg_fetch / gg_decode, see
// fc_gguf_gemm_body.cl) that decodes the SG-shuffled weight layout; the shared pipeline lives in the body.

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// ===== inlined from fc_gguf_gemm_common.cl =====
// Shared helpers for the GGUF prefill GEMM kernels (fc_gguf_gemm_opt.cl), ported from the CM
// kernels cm/gguf/gemm_q*_full.cm. Pure helpers only; the pipeline itself lives in
// fc_gguf_gemm_body.cl.
//
// Operand layout used with intel_sub_group_f16_f16_matrix_mad_k16 (SIMD16, M=8, N=16, K=16):
//   A  (short8): lane = K column, element i = token row i      (2D block read gives this directly)
//   B  (int8)  : lane = output column n, element sd = half2 {K=2*sd, K=2*sd+1} of that column
//   acc(float8): lane = output column n, element i = token row i

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#if defined(cl_intel_subgroup_local_block_io)
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#endif

// Pack two floats into a half2 (round-to-nearest-even, same as the CM float->half assignment).
inline uint FUNC(gg_pk)(float a, float b) {
    return as_uint(convert_half2_rte((float2)(a, b)));
}

// Load the activation tile [8 tokens x 32 K] starting at column x, row y as {lo: K 0..15, hi: K 16..31}.
// Rows >= M are returned as zero by the 2D block hardware.
#if GG_A_F32
inline void FUNC(gg_load_a)(const __global INPUT0_TYPE* A, uint K, uint M, int x, int y, short8* lo, short8* hi) {
    uint8 u0;
    uint8 u1;
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)A, K * 4, M, K * 4, (int2)(x, y), (private uint*)&u0);
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)A, K * 4, M, K * 4, (int2)(x + 16, y), (private uint*)&u1);
    *lo = as_short8(convert_half8_rte(as_float8(u0)));
    *hi = as_short8(convert_half8_rte(as_float8(u1)));
}
#else
inline void FUNC(gg_load_a)(const __global INPUT0_TYPE* A, uint K, uint M, int x, int y, short8* lo, short8* hi) {
    ushort16 u;
    intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)A, K * 2, M, K * 2, (int2)(x, y), (private ushort*)&u);
    *lo = as_short8(u.lo);
    *hi = as_short8(u.hi);
}
#endif

// Load nt (2 or 4) consecutive 8-token tiles x 32 K with one 2D block read (f16 activations only).
#define GG_LOAD_A_MULTI(A, K, M, x, y, nt, alo, ahi)                                                                    \
    do {                                                                                                               \
        ushort gg_u[16 * (nt)];                                                                                        \
        if ((nt) == 4)                                                                                                 \
            intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(A), (K) * 2, (M), (K) * 2, (int2)((x), (y)), gg_u); \
        else                                                                                                           \
            intel_sub_group_2d_block_read_16b_16r16x2c((__global void*)(A), (K) * 2, (M), (K) * 2, (int2)((x), (y)), gg_u); \
        __attribute__((opencl_unroll_hint)) for (uint gg_t = 0; gg_t < (nt); ++gg_t) {                                 \
            alo[gg_t] = as_short8(vload8(0, gg_u + 8 * gg_t));                                                         \
            ahi[gg_t] = as_short8(vload8(0, gg_u + 8 * (nt) + 8 * gg_t));                                              \
        }                                                                                                              \
    } while (0)

// SLM tile access: a tile is 8 x 16 uints laid out [sd][lane] so that every lane touches a distinct bank.
#if defined(cl_intel_subgroup_local_block_io) && !defined(GG_SLM_NO_BLOCK)
#define GG_SLM_BLOCK_IO 1
inline int8 FUNC(gg_slm_rd8)(const __local uint* p, uint lane) {
    return as_int8(intel_sub_group_block_read8(p));
}

inline void FUNC(gg_slm_wr8)(__local uint* p, uint lane, uint8 v) {
    intel_sub_group_block_write8(p, v);
}
#else
inline int8 FUNC(gg_slm_rd8)(const __local uint* p, uint lane) {
    return as_int8((uint8)(p[lane], p[16 + lane], p[32 + lane], p[48 + lane], p[64 + lane], p[80 + lane], p[96 + lane], p[112 + lane]));
}

inline void FUNC(gg_slm_wr8)(__local uint* p, uint lane, uint8 v) {
    p[lane] = v.s0;
    p[16 + lane] = v.s1;
    p[32 + lane] = v.s2;
    p[48 + lane] = v.s3;
    p[64 + lane] = v.s4;
    p[80 + lane] = v.s5;
    p[96 + lane] = v.s6;
    p[112 + lane] = v.s7;
}
#endif

inline float FUNC(gg_f8_el)(float8 v, uint i) {
    switch (i) {
    case 0: return v.s0;
    case 1: return v.s1;
    case 2: return v.s2;
    case 3: return v.s3;
    case 4: return v.s4;
    case 5: return v.s5;
    case 6: return v.s6;
    default: return v.s7;
    }
}

// Four unsigned bytes (consecutive K) -> two half2 words of byte * s + o.
inline uint2 FUNC(gg_cvt4u)(uint bytes, float s, float o) {
    const float4 f = mad(convert_float4(as_uchar4(bytes)), (float4)(s), (float4)(o));
    return (uint2)(FUNC_CALL(gg_pk)(f.s0, f.s1), FUNC_CALL(gg_pk)(f.s2, f.s3));
}

// Four signed bytes (consecutive K) -> two half2 words of byte * s.
inline uint2 FUNC(gg_cvt4s)(uint bytes, float s) {
    const float4 f = convert_float4(as_char4(bytes)) * s;
    return (uint2)(FUNC_CALL(gg_pk)(f.s0, f.s1), FUNC_CALL(gg_pk)(f.s2, f.s3));
}

// Q4-style decode of one sub-block (32 K) for this lane: element = nibble * scale - minv.
// q holds 4 chunks; chunk c byte b low nibble is K = 4c+b, high nibble is K = 16+4c+b.
inline void FUNC(gg_decode_nib4)(uint4 q, float scale, float minv, uint8* blo, uint8* bhi) {
    const uint2 l0 = FUNC_CALL(gg_cvt4u)(q.s0 & 0x0F0F0F0Fu, scale, -minv);
    const uint2 l1 = FUNC_CALL(gg_cvt4u)(q.s1 & 0x0F0F0F0Fu, scale, -minv);
    const uint2 l2 = FUNC_CALL(gg_cvt4u)(q.s2 & 0x0F0F0F0Fu, scale, -minv);
    const uint2 l3 = FUNC_CALL(gg_cvt4u)(q.s3 & 0x0F0F0F0Fu, scale, -minv);
    const uint2 h0 = FUNC_CALL(gg_cvt4u)((q.s0 >> 4) & 0x0F0F0F0Fu, scale, -minv);
    const uint2 h1 = FUNC_CALL(gg_cvt4u)((q.s1 >> 4) & 0x0F0F0F0Fu, scale, -minv);
    const uint2 h2 = FUNC_CALL(gg_cvt4u)((q.s2 >> 4) & 0x0F0F0F0Fu, scale, -minv);
    const uint2 h3 = FUNC_CALL(gg_cvt4u)((q.s3 >> 4) & 0x0F0F0F0Fu, scale, -minv);
    *blo = (uint8)(l0, l1, l2, l3);
    *bhi = (uint8)(h0, h1, h2, h3);
}
// ===== end fc_gguf_gemm_common.cl =====

#if GG_FMT_Q4_0
// GGUF Q4_0 prefill GEMM (port of cm/gguf/gemm_q40_full.cm). w = (q - 8) * d.
// Weights (shuffle layout): pqs_T(128B/block) || pd_T(16B/block).

#define GG_RAW_T uint8

inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = intel_sub_group_block_read4((const __global uint*)(W + (ulong)blk * 2048 + j * 256));
    const uint d = intel_sub_group_block_read_us((const __global ushort*)(W + (ulong)num_blocks * 128 + (ulong)blk * 256 + j * 32));
    return (uint8)(q, d, 0, 0, 0);
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const float d = convert_float(as_half((ushort)r.s4));
    FUNC_CALL(gg_decode_nib4)(r.s0123, d, d * 8.0f, blo, bhi);
}

#elif GG_FMT_Q4_1
// GGUF Q4_1 prefill GEMM (port of cm/gguf/gemm_q41_full.cm). w = q * d + m.
// Weights (shuffle layout): pqs_T(128B/block) || pd_T(16B/block) || pm_T(16B/block).

#define GG_RAW_T uint8

inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = intel_sub_group_block_read4((const __global uint*)(W + (ulong)blk * 2048 + j * 256));
    const ulong meta = (ulong)blk * 256 + j * 32;
    const uint d = intel_sub_group_block_read_us((const __global ushort*)(W + (ulong)num_blocks * 128 + meta));
    const uint m = intel_sub_group_block_read_us((const __global ushort*)(W + (ulong)num_blocks * 144 + meta));
    return (uint8)(q, d, m, 0, 0);
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const float d = convert_float(as_half((ushort)r.s4));
    const float m = convert_float(as_half((ushort)r.s5));
    FUNC_CALL(gg_decode_nib4)(r.s0123, d, -m, blo, bhi);
}

#elif GG_FMT_Q4_K
// GGUF Q4_K prefill GEMM (port of cm/gguf/gemm_q4k_full.cm). w = q * (sq * d) - (mq * dmin).
// Weights (shuffle layout): pqs_T(128B/block) || psl_T(16B/block, SoA fields sl,ml,sh,mh,d,dmin).

#define GG_RAW_T uint8

// Raw tile: q(4) | sl | ml | sh,mh | d,dmin.
inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = intel_sub_group_block_read4((const __global uint*)(W + (ulong)blk * 2048 + j * 256));
    const __global uchar* psl = W + (ulong)num_blocks * 128 + (ulong)blk * 256;
    const uint2 slml = intel_sub_group_block_read2((const __global uint*)psl);
    const uint2 meta = as_uint2(intel_sub_group_block_read_us4((const __global ushort*)(psl + 128)));
    return (uint8)(q, slml, meta);
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const uint sh = r.s6 & 0xFFFFu;
    const uint mh = r.s6 >> 16;
    const float d = convert_float(as_half((ushort)(r.s7 & 0xFFFFu)));
    const float dmin = convert_float(as_half((ushort)(r.s7 >> 16)));
    const uint sq = ((r.s4 >> (j * 4)) & 0xFu) | (((sh >> (j * 2)) & 0x3u) << 4);
    const uint mq = ((r.s5 >> (j * 4)) & 0xFu) | (((mh >> (j * 2)) & 0x3u) << 4);
    FUNC_CALL(gg_decode_nib4)(r.s0123, (float)sq * d, (float)mq * dmin, blo, bhi);
}

#elif GG_FMT_Q5_K
// GGUF Q5_K prefill GEMM (port of cm/gguf/gemm_q5k_full.cm). w = (nib | bit5 << 4) * (sq * d) - (mq * dmin).
// Weights (shuffle layout): pqs_T(128B/block) || pqh_T(32B/block) || psl_T(16B/block).

#define GG_RAW_T uint16

// Raw tile: q(4) | H | sl | ml | sh,mh | d,dmin.
inline uint16 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = intel_sub_group_block_read4((const __global uint*)(W + (ulong)blk * 2048 + j * 256));
    const uint H = intel_sub_group_block_read((const __global uint*)(W + (ulong)num_blocks * 128 + (ulong)blk * 512 + j * 64));
    const __global uchar* psl = W + (ulong)num_blocks * 160 + (ulong)blk * 256;
    const uint2 slml = intel_sub_group_block_read2((const __global uint*)psl);
    const uint2 meta = as_uint2(intel_sub_group_block_read_us4((const __global ushort*)(psl + 128)));
    return (uint16)(q, H, slml, meta, 0, 0, 0, 0, 0, 0, 0);
}

inline void FUNC(gg_decode)(uint16 r, uint j, uint8* blo, uint8* bhi) {
    const uint H = r.s4;
    const uint sh = r.s7 & 0xFFFFu;
    const uint mh = r.s7 >> 16;
    const float d = convert_float(as_half((ushort)(r.s8 & 0xFFFFu)));
    const float dmin = convert_float(as_half((ushort)(r.s8 >> 16)));
    const uint sq = ((r.s5 >> (j * 4)) & 0xFu) | (((sh >> (j * 2)) & 0x3u) << 4);
    const uint mq = ((r.s6 >> (j * 4)) & 0xFu) | (((mh >> (j * 2)) & 0x3u) << 4);
    const float scale = (float)sq * d;
    const float minv = (float)mq * dmin;
    // Byte b of chunk c: low nibble K=4c+b with 5th bit at H bit 8b+c, high nibble K=16+4c+b with H bit 8b+4+c.
    uint2 lo[4];
    uint2 hi[4];
    __attribute__((opencl_unroll_hint))
    for (uint c = 0; c < 4; ++c) {
        const uint u = c == 0 ? r.s0 : c == 1 ? r.s1 : c == 2 ? r.s2 : r.s3;
        lo[c] = FUNC_CALL(gg_cvt4u)((u & 0x0F0F0F0Fu) | (((H >> c) & 0x01010101u) << 4), scale, -minv);
        hi[c] = FUNC_CALL(gg_cvt4u)(((u >> 4) & 0x0F0F0F0Fu) | (((H >> (c + 4)) & 0x01010101u) << 4), scale, -minv);
    }
    *blo = (uint8)(lo[0], lo[1], lo[2], lo[3]);
    *bhi = (uint8)(hi[0], hi[1], hi[2], hi[3]);
}

#elif GG_FMT_Q6_K
// GGUF Q6_K prefill GEMM (port of cm/gguf/gemm_q6k_full.cm). w = ((nib | h2 << 4) - 32) * (d * sc).
// Weights (shuffle layout): pql_T(128B/block) || pqh_T(64B/block) || ps_T(16B/block) || pd_T(2B/block).

#define GG_RAW_T uint8

// Raw tile: q(4) | HL | HH | d | scales(lo, hi bytes).
inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = intel_sub_group_block_read4((const __global uint*)(W + (ulong)blk * 2048 + j * 256));
    const uint2 h = intel_sub_group_block_read2((const __global uint*)(W + (ulong)num_blocks * 128 + (ulong)blk * 1024 + (2 * j) * 64));
    const uint d = intel_sub_group_block_read_us((const __global ushort*)(W + (ulong)num_blocks * 208 + (ulong)blk * 32));
    const uchar2 sc = intel_sub_group_block_read_uc2(W + (ulong)num_blocks * 192 + (ulong)blk * 256 + (2 * j) * 16);
    return (uint8)(q, h, d, (uint)as_ushort(sc));
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const uint HL = r.s4;
    const uint HH = r.s5;
    const float d = convert_float(as_half((ushort)r.s6));
    const float clo = d * (float)(char)(r.s7 & 0xFFu);
    const float chi = d * (float)(char)(r.s7 >> 8);
    // Byte b of chunk c: low nibble K=4c+b with upper bits at HL bits 8b+2c, high nibble K=16+4c+b with HH bits 8b+2c.
    uint2 lo[4];
    uint2 hi[4];
    __attribute__((opencl_unroll_hint))
    for (uint c = 0; c < 4; ++c) {
        const uint u = c == 0 ? r.s0 : c == 1 ? r.s1 : c == 2 ? r.s2 : r.s3;
        lo[c] = FUNC_CALL(gg_cvt4u)((u & 0x0F0F0F0Fu) | (((HL >> (2 * c)) & 0x03030303u) << 4), clo, -32.0f * clo);
        hi[c] = FUNC_CALL(gg_cvt4u)(((u >> 4) & 0x0F0F0F0Fu) | (((HH >> (2 * c)) & 0x03030303u) << 4), chi, -32.0f * chi);
    }
    *blo = (uint8)(lo[0], lo[1], lo[2], lo[3]);
    *bhi = (uint8)(hi[0], hi[1], hi[2], hi[3]);
}

#elif GG_FMT_Q8_0
// GGUF Q8_0 prefill GEMM (port of cm/gguf/gemm_q80_full.cm). w = q * d.
// Weights (shuffle layout): pqs_T(256B/block) || pd_T(16B/block). Sub-block payload: 512B = lo(256B) || hi(256B).

#define GG_RAW_T uint16

// Raw tile: q lo(4) | q hi(4) | d.
inline uint16 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint8 q = intel_sub_group_block_read8((const __global uint*)(W + (ulong)blk * 4096 + j * 512));
    const uint d = intel_sub_group_block_read_us((const __global ushort*)(W + (ulong)num_blocks * 256 + (ulong)blk * 256 + j * 32));
    return (uint16)(q, d, 0, 0, 0, 0, 0, 0, 0);
}

inline void FUNC(gg_decode)(uint16 r, uint j, uint8* blo, uint8* bhi) {
    const float d = convert_float(as_half((ushort)r.s8));
    *blo = (uint8)(FUNC_CALL(gg_cvt4s)(r.s0, d), FUNC_CALL(gg_cvt4s)(r.s1, d), FUNC_CALL(gg_cvt4s)(r.s2, d), FUNC_CALL(gg_cvt4s)(r.s3, d));
    *bhi = (uint8)(FUNC_CALL(gg_cvt4s)(r.s4, d), FUNC_CALL(gg_cvt4s)(r.s5, d), FUNC_CALL(gg_cvt4s)(r.s6, d), FUNC_CALL(gg_cvt4s)(r.s7, d));
}

#else
#error "fc_gguf_gemm_opt.cl: define exactly one GG_FMT_* format macro"
#endif

// ===== inlined from fc_gguf_gemm_body.cl =====
// GGUF prefill GEMM pipeline template (ported from the CM gemm_q*_full.cm kernels). Included by
// fc_gguf_gemm_opt.cl after it has defined the per-format tile codec:
//
//   GG_RAW_T gg_fetch(const __global uchar* W, uint num_blocks, uint blk, uint j)   // raw encoded tile
//   void gg_decode(GG_RAW_T raw, uint j, uint8* blo, uint8* bhi)                  // -> B operands
//
// for sub-block `j` of row-group/K-block `blk` and this lane's output column; blo/bhi are the two
// B operands (K 0..15, K 16..31) of intel_sub_group_f16_f16_matrix_mad_k16.
//
// Work decomposition (identical to the CM kernel, with the 16 lanes of a CM thread becoming one
// sub-group): work-group = TOKEN_LOCAL sub-groups, each owning TOKENS_PER_THREAD tokens x
// (ROW_GROUPS * 16) outputs. The sub-groups of a work-group share the weight tile: every phase the
// SLM_NPROD (row group, sub-block) tiles of that phase are decoded cooperatively (one per
// sub-group), published through an SLM ring, then consumed by all sub-groups.
//
// JIT constants: ROW_GROUPS, TOKEN_GROUPS, TOKEN_LOCAL, ROW_BLOCKS, SLM_JBLK, PF_A, GG_A_F32, GG_OUT_F32.

#define GG_TOKENS_PER_TILE 8
#define GG_SLM_SLOTS 2
#define GG_SLM_NPROD (SLM_JBLK * ROW_GROUPS * ROW_BLOCKS)
#define GG_TOKEN_SLOTS (TOKEN_LOCAL / ROW_BLOCKS)
#define GG_TOKENS_PER_THREAD (GG_TOKENS_PER_TILE * TOKEN_GROUPS)
#define GG_SLM_JPS (8 / SLM_JBLK)
#define GG_TILE_U 128
#define GG_PHASE_U (GG_SLM_NPROD * 2 * GG_TILE_U)
#if GG_TOKENS_PER_THREAD > 32
#define GG_PFA_ROWS 32
#else
#define GG_PFA_ROWS GG_TOKENS_PER_THREAD
#endif
#define GG_PFA_SETS (GG_TOKENS_PER_THREAD / GG_PFA_ROWS)

// Prefetch GG_PFA_ROWS rows x 32 K columns of the next K-block's activations (cache-line sized window).
#if GG_PFA_ROWS == 32
#define GG_PREFETCH_A_F16(A, K, M, pc) intel_sub_group_2d_block_prefetch_16b_32r16x2c((__global void*)(A), (K) * 2, (M), (K) * 2, (pc))
#define GG_PREFETCH_A_F32(A, K, M, pc)                                                                  \
    intel_sub_group_2d_block_prefetch_32b_32r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc));   \
    intel_sub_group_2d_block_prefetch_32b_32r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc) + (int2)(16, 0))
#elif GG_PFA_ROWS == 16
#define GG_PREFETCH_A_F16(A, K, M, pc) intel_sub_group_2d_block_prefetch_16b_16r16x2c((__global void*)(A), (K) * 2, (M), (K) * 2, (pc))
#define GG_PREFETCH_A_F32(A, K, M, pc)                                                                  \
    intel_sub_group_2d_block_prefetch_32b_16r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc));   \
    intel_sub_group_2d_block_prefetch_32b_16r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc) + (int2)(16, 0))
#else
#define GG_PREFETCH_A_F16(A, K, M, pc) intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)(A), (K) * 2, (M), (K) * 2, (pc))
#define GG_PREFETCH_A_F32(A, K, M, pc)                                                                  \
    intel_sub_group_2d_block_prefetch_32b_8r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc));    \
    intel_sub_group_2d_block_prefetch_32b_8r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc) + (int2)(16, 0))
#endif

#if GG_SLM_NPROD >= TOKEN_LOCAL
#define GG_PER_SG (GG_SLM_NPROD / TOKEN_LOCAL)
#else
#define GG_PER_SG 1
#endif

// (blk, j) of producer slot w (w = lid + u * TOKEN_LOCAL) in phase s.
#define GG_TILE_COORD(s, w, blk, j)                                              \
    const uint kb_##blk = (s) / GG_SLM_JPS;                                      \
    const uint jb_##blk = (s) - kb_##blk * GG_SLM_JPS;                           \
    const uint rbp_##blk = (w) / (ROW_GROUPS * SLM_JBLK);                        \
    const uint rem_##blk = (w) - rbp_##blk * (ROW_GROUPS * SLM_JBLK);            \
    const uint rg_##blk = rem_##blk / SLM_JBLK;                                  \
    const uint j = jb_##blk * SLM_JBLK + (rem_##blk - rg_##blk * SLM_JBLK);      \
    const uint blk = ((gh * ROW_BLOCKS + rbp_##blk) * ROW_GROUPS + rg_##blk) * nbpr + kb_##blk

// Load the raw (still encoded) weight tiles this sub-group produces in phase s.
inline void FUNC(gg_fetch_stage)(const __global uchar* W, uint num_blocks, uint nbpr, uint gh, uint lid, uint s, uint nstages,
                                 GG_RAW_T* raw) {
    if (s >= nstages) {
        return;
    }
    __attribute__((opencl_unroll_hint))
    for (uint u = 0; u < GG_PER_SG; ++u) {
        const uint w = lid + u * TOKEN_LOCAL;
        if (GG_SLM_NPROD >= TOKEN_LOCAL || w < GG_SLM_NPROD) {
            GG_TILE_COORD(s, w, blk, j);
            raw[u] = FUNC_CALL(gg_fetch)(W, num_blocks, blk, j);
        }
    }
}

// Decode the raw tiles of phase s and publish them into SLM slot (s & 1).
inline void FUNC(gg_store_stage)(__local uint* slm, uint lid, uint lane, uint s, uint nstages, const GG_RAW_T* raw) {
    if (s >= nstages) {
        return;
    }
    const uint sbase = (s & (GG_SLM_SLOTS - 1)) * GG_PHASE_U;
    const uint jb = s % GG_SLM_JPS;
    __attribute__((opencl_unroll_hint))
    for (uint u = 0; u < GG_PER_SG; ++u) {
        const uint w = lid + u * TOKEN_LOCAL;
        if (GG_SLM_NPROD >= TOKEN_LOCAL || w < GG_SLM_NPROD) {
            const uint j = jb * SLM_JBLK + w % SLM_JBLK;
            uint8 blo;
            uint8 bhi;
            FUNC_CALL(gg_decode)(raw[u], j, &blo, &bhi);
            __local uint* dst = slm + sbase + (w * 2) * GG_TILE_U;
            FUNC_CALL(gg_slm_wr8)(dst, lane, blo);
            FUNC_CALL(gg_slm_wr8)(dst + GG_TILE_U, lane, bhi);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, TOKEN_LOCAL, 1)))
KERNEL(fc_gguf_gemm)(OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,
    const __global uchar* W,
    __global OUTPUT_TYPE* C,
    uint M,
    uint K,
    uint N
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint lane = get_sub_group_local_id();
    const uint nbpr = K / 256;
    const uint num_blocks = (K / 256) * N;

#if GG_N_BAND
    // Walk the token blocks of GG_N_BAND row-group blocks before moving on, which shortens the reuse
    // distance of a weight slice when the whole weight matrix does not fit the GPU caches.
    const uint ngn = get_num_groups(0);
    const uint lin = get_group_id(1) * ngn + get_group_id(0);
    const uint band_groups = GG_N_BAND * get_num_groups(1);
    const uint bi = lin / band_groups;
    const uint brem = lin - bi * band_groups;
    const uint bw = min((uint)GG_N_BAND, ngn - bi * GG_N_BAND);
    const uint gt = brem / bw;
    const uint gh = bi * GG_N_BAND + (brem - gt * bw);
#else
    const uint gh = get_group_id(0);
    const uint gt = get_group_id(1);
#endif
    const uint lid = get_local_id(1);
    const uint rb = lid % ROW_BLOCKS;
    const uint ts = lid / ROW_BLOCKS;
    const uint h0 = (gh * ROW_BLOCKS + rb) * ROW_GROUPS;
    const uint t = gt * GG_TOKEN_SLOTS + ts;
    const uint tile_base = t * GG_TOKENS_PER_THREAD;

    float8 acc[ROW_GROUPS * TOKEN_GROUPS];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0; i < ROW_GROUPS * TOKEN_GROUPS; ++i) {
        acc[i] = (float8)(0.0f);
    }

    __local uint slm[GG_SLM_SLOTS * GG_PHASE_U];

    const uint nstages = nbpr * GG_SLM_JPS;

    // Weight tiles of phase s+1 are decoded into the other SLM slot while phase s is consumed.
    GG_RAW_T raw[GG_PER_SG];
    FUNC_CALL(gg_fetch_stage)(W, num_blocks, nbpr, gh, lid, 0, nstages, raw);
    FUNC_CALL(gg_store_stage)(slm, lid, lane, 0, nstages, raw);

    for (uint s = 0; s < nstages; ++s) {
        barrier(CLK_LOCAL_MEM_FENCE);

        FUNC_CALL(gg_fetch_stage)(W, num_blocks, nbpr, gh, lid, s + 1, nstages, raw);
        FUNC_CALL(gg_store_stage)(slm, lid, lane, s + 1, nstages, raw);

        const uint kb = s / GG_SLM_JPS;
        const uint jb = s - kb * GG_SLM_JPS;
#if PF_A
        if (jb == 0 && kb + 1 < nbpr) {
            __attribute__((opencl_unroll_hint))
            for (uint r = 0; r < GG_PFA_SETS; ++r) {
                const int2 pc = (int2)((kb + 1) * 256, tile_base + r * GG_PFA_ROWS);
#if GG_A_F32
                GG_PREFETCH_A_F32(A, K, M, pc);
#else
                GG_PREFETCH_A_F16(A, K, M, pc);
#endif
            }
        }
#endif
        const uint sbase = (s & (GG_SLM_SLOTS - 1)) * GG_PHASE_U;

        __attribute__((opencl_unroll_hint))
        for (uint jl = 0; jl < SLM_JBLK; ++jl) {
            const uint j = jb * SLM_JBLK + jl;
            short8 alo[TOKEN_GROUPS];
            short8 ahi[TOKEN_GROUPS];
#if !GG_A_F32 && (TOKEN_GROUPS == 2 || TOKEN_GROUPS == 4)
            GG_LOAD_A_MULTI(A, K, M, (int)(kb * 256 + j * 32), (int)tile_base, TOKEN_GROUPS, alo, ahi);
#else
            __attribute__((opencl_unroll_hint))
            for (uint tg = 0; tg < TOKEN_GROUPS; ++tg) {
                FUNC_CALL(gg_load_a)(A, K, M, (int)(kb * 256 + j * 32), (int)(tile_base + tg * GG_TOKENS_PER_TILE), &alo[tg], &ahi[tg]);
            }
#endif
            __attribute__((opencl_unroll_hint))
            for (uint rg = 0; rg < ROW_GROUPS; ++rg) {
                const uint ti = (rb * ROW_GROUPS + rg) * SLM_JBLK + jl;
                const __local uint* tp = slm + sbase + (ti * 2) * GG_TILE_U;
                const int8 blo = FUNC_CALL(gg_slm_rd8)(tp, lane);
                const int8 bhi = FUNC_CALL(gg_slm_rd8)(tp + GG_TILE_U, lane);
                __attribute__((opencl_unroll_hint))
                for (uint tg = 0; tg < TOKEN_GROUPS; ++tg) {
                    acc[rg * TOKEN_GROUPS + tg] = intel_sub_group_f16_f16_matrix_mad_k16(alo[tg], blo, acc[rg * TOKEN_GROUPS + tg]);
                    acc[rg * TOKEN_GROUPS + tg] = intel_sub_group_f16_f16_matrix_mad_k16(ahi[tg], bhi, acc[rg * TOKEN_GROUPS + tg]);
                }
            }
        }
    }

    // Epilogue: lane = output column, element i of acc = token row i.
    __attribute__((opencl_unroll_hint))
    for (uint tg = 0; tg < TOKEN_GROUPS; ++tg) {
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < GG_TOKENS_PER_TILE; ++i) {
            const uint bm = tile_base + tg * GG_TOKENS_PER_TILE + i;
            if (bm < M) {
                __attribute__((opencl_unroll_hint))
                for (uint rg = 0; rg < ROW_GROUPS; ++rg) {
                    const uint n = (h0 + rg) * 16 + lane;
                    const float v = FUNC_CALL(gg_f8_el)(acc[rg * TOKEN_GROUPS + tg], i);
#if HAS_FUSED_OPS
                    {
                        const uint out_b = bm / OUTPUT_FEATURE_NUM;
                        const uint out_f = bm - out_b * OUTPUT_FEATURE_NUM;
                        OUTPUT_TYPE dequantized = TO_OUTPUT_TYPE(v);
                        FUSED_OPS;
                        C[(ulong)bm * N + n] = FUSED_OPS_RESULT;
                    }
#elif GG_OUT_F32
                    C[(ulong)bm * N + n] = v;
#else
                    C[(ulong)bm * N + n] = convert_half_rte(v);
#endif
                }
            }
        }
    }
}
// ===== end fc_gguf_gemm_body.cl =====

// [EVOLVE_END]
