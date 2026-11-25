// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q8_0 dp4a reading a SIZE-EXACT byte-transposed LISA layout (keeps 34 B/block, ZERO expansion).
//
// Q8_0 is used by the attention / GDN projections. On the native block layout the plain GEMV reads
// each 34-byte block strided (adjacent subgroup lanes 34 B apart) -> uncoalesced (~33% of peak BW,
// the top decode kernels). This kernel reads the byte-transposed layout produced by
// RepackGGUFWeightsQ6K (generalised to Q8_0): group = 16 blocks; byte i of all 16 blocks is stored
// contiguously -> dst(group g, block-lane L, byte i) = g*(16*34) + i*16 + L. Lane L gathers its 34
// bytes strided by 16 into a private buffer (the 16 lanes hit 16 CONSECUTIVE bytes per byte-index ->
// coalesced), then runs the trivial Q8_0 dp4a (weight is already signed int8; no bit-unpack). The
// activation is pre-quantised to int8 + per-32 f32 scale (fc_gguf_prequant.cl). Bit-close to the
// float GEMV (only the int8 activation quant adds error, same as every other dp4a path).

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

inline half FUNC(q8e_load_f16)(const __private uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

// One Q8_0 block (34 B: 2-byte f16 scale + 32 int8) from private, dp4a against int8 activation aq with
// its single per-32 scale asc[0]. Signed x signed hardware dot.
inline float FUNC(q8e_block_dot)(const __private uchar* blk, const __global char* aq, const __global float* asc) {
    const float d = (float)FUNC_CALL(q8e_load_f16)(blk);
    const __private char* qs = (const __private char*)(blk + 2);
    int S = 0;
    __attribute__((opencl_unroll_hint))
    for (int l = 0; l < 32; l += 4) {
        const uint w = as_uint(vload4(0, qs + l));
        const uint a = as_uint(vload4(0, aq + l));
        S = dot_acc_sat_4x8packed_ss_int(w, a, S);
    }
    return d * asc[0] * (float)S;
}

#ifndef NROW
#define NROW 1
#endif
#define Q8E_BB 34

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
KERNEL(fc_gguf_dp4a_q8_0_packed)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global char*        Aq,
    const __global float*       Asc,
    const __global uchar*       W,    // size-exact byte-transposed Q8_0 weight
          __global OUTPUT_TYPE* C
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const int lane = (int)get_sub_group_local_id();
    const int bm   = (int)get_global_id(1);
    const int n0   = (int)(get_global_id(0) / SG_SIZE) * NROW;
    if (n0 >= N_SIZE)
        return;

    const int bpr = K_SIZE / GGUF_BLOCK_ELEM;
    const __global char*  aq_row  = Aq  + (uint)bm * (uint)K_SIZE;
    const __global float* asc_row = Asc + (uint)bm * (uint)(K_SIZE / 32);
    const uint group_stride = (uint)SG_SIZE * Q8E_BB;   // bytes per 16-block group, per row

    float partial[NROW];
    unroll_for (int r = 0; r < NROW; ++r)
        partial[r] = 0.0f;

    uchar blk[Q8E_BB];

    for (int g = 0; g < GGUF_G; ++g) {
        const int kb = g * SG_SIZE + lane;
        if (kb >= bpr)
            continue;
        const __global char*  aq_blk  = aq_row  + (uint)kb * GGUF_BLOCK_ELEM;
        const __global float* asc_blk = asc_row + (uint)kb * (GGUF_BLOCK_ELEM / 32);
        unroll_for (int r = 0; r < NROW; ++r) {
            if ((n0 + r) >= N_SIZE)
                continue;
            const __global uchar* gbase = W + ((uint)(n0 + r) * (uint)GGUF_G + (uint)g) * group_stride + (uint)lane;
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < Q8E_BB; ++i)
                blk[i] = gbase[(uint)i * 16u];     // 16 lanes -> 16 consecutive bytes (coalesced)
            partial[r] += FUNC_CALL(q8e_block_dot)((const __private uchar*)blk, aq_blk, asc_blk);
        }
    }

    unroll_for (int r = 0; r < NROW; ++r) {
        const float total = sub_group_reduce_add(partial[r]);
        if (lane == 0 && (n0 + r) < N_SIZE) {
            const uint n = (uint)(n0 + r);
#if HAS_FUSED_OPS
            const uint out_b = (uint)bm / OUTPUT_FEATURE_NUM;
            const uint out_f = (uint)bm - out_b * OUTPUT_FEATURE_NUM;
            OUTPUT_TYPE dequantized = TO_OUTPUT_TYPE(total);
            FUSED_OPS;
            C[(uint)bm * (uint)N_SIZE + n] = FUSED_OPS_RESULT;
#else
            C[(uint)bm * (uint)N_SIZE + n] = TO_OUTPUT_TYPE(total);
#endif
        }
    }
}
