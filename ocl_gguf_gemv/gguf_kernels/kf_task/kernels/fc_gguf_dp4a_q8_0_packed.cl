// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q8_0 dp4a on SIZE-EXACT byte-transposed LISA layout (34 B/block, ZERO expansion).
// STRICTLY BANDWIDTH-BOUND GEMV (~1 dp4a/weight byte, 133 GB/s LPDDR5X). Runtime fixed
// by single-pass weight read; optimize transaction coalescing & sub-group issue efficiency.
// Perf recipe summary:
// 1. intel_sub_group_block_read_uc8 128B coalesced bursts (4×512B load, lower tx overhead)
// 2. NROW=4 register blocking: activation loaded once per lane, reused 4× (only real data reuse)
//    No SLM tiling: weight blocks lane-private, cross-lane reuse none, barriers hurt BW bound path
// 3. Pre-fetch all NROW uc2 block scales upfront to overlap heavy weight bursts & dp4a chains
// 4. Dual independent dp4a accumulators hide dot latency with minimal GRF footprint
// 5. __global const restrict for all read-only buffers: enable compiler load reordering

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#ifndef NROW
#define NROW 4
#endif
#define Q8E_BB        34
#define GGUF_BLOCK_EL 32

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
KERNEL(fc_gguf_dp4a_q8_0_packed)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global char*        restrict Aq,
    const __global float*       restrict Asc,
    const __global uchar*       restrict W, // size-exact byte-transposed Q8_0 weight
          __global OUTPUT_TYPE* restrict C
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    // Subgroup & output tile indices
    const int lane  = get_sub_group_local_id();
    const int bm    = get_global_id(1);
    const int tile0  = (get_global_id(0) / SG_SIZE) * NROW;
    if (tile0 >= N_SIZE)
        return;

    // Precompute compile-time foldable strides
    const uint blk_per_group    = SG_SIZE;
    const uint bytes_per_sg_blk = blk_per_group * Q8E_BB;
    const uint group_row_stride = GGUF_G * bytes_per_sg_blk;
    const uint k_blocks         = K_SIZE / GGUF_BLOCK_EL;

    // Batch row base pointers (all read-only, restrict qualified)
    const __global char*  restrict aq_batch  = Aq  + (uint)bm * K_SIZE;
    const __global float* restrict asc_batch = Asc + (uint)bm * (K_SIZE / GGUF_BLOCK_EL);
    const __global uchar* restrict w_base    = W;

    // Initialize partial accumulators for NROW output tiles
    float partial[NROW] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Iterate over all GGUF quantization groups
    unroll_for(int g = 0; g < GGUF_G; g++)
    {
        const int kb_idx = g * SG_SIZE + lane;
        const bool kb_valid = (kb_idx < k_blocks);

        // Activation block load: single load reused across all NROW rows
        uint a_pack[8] = {0u};
        float asc_scale = 0.0f;
        if (kb_valid)
        {
            const uint aq_off  = (uint)kb_idx * GGUF_BLOCK_EL;
            const uint asc_off = (uint)kb_idx;
            const char16 a_lo = vload16(0, aq_batch + aq_off);
            const char16 a_hi = vload16(1, aq_batch + aq_off);

            // Pack 32 int8 activation into 8 uint dot operands
            a_pack[0] = as_uint(a_lo.s0123); a_pack[1] = as_uint(a_lo.s4567);
            a_pack[2] = as_uint(a_lo.s89ab); a_pack[3] = as_uint(a_lo.scdef);
            a_pack[4] = as_uint(a_hi.s0123); a_pack[5] = as_uint(a_hi.s4567);
            a_pack[6] = as_uint(a_hi.s89ab); a_pack[7] = as_uint(a_hi.scdef);
            asc_scale = asc_batch[asc_off];
        }

        // Weight group base for current output tile start
        const uint gbase_byte_off = ((uint)tile0 * GGUF_G + (uint)g) * bytes_per_sg_blk;
        const __global uchar* restrict sg_w_group = w_base + gbase_byte_off;

        // Step1: Pre-fetch all NROW scale factors UPFRONT to overlap weight burst loads
        float row_scale[NROW] = {0.0f, 0.0f, 0.0f, 0.0f};
        unroll_for(int r = 0; r < NROW; r++)
        {
            const int out_n = tile0 + r;
            if (out_n >= N_SIZE)
                continue;

            const __global uchar* restrict scale_ptr = sg_w_group + r * group_row_stride;
            const uchar2 sc_pair = intel_sub_group_block_read_uc2(scale_ptr);
            const ushort half_bits = (ushort)sc_pair.s0 | ((ushort)sc_pair.s1 << 8);
            const float w_scale = as_half(half_bits);
            row_scale[r] = asc_scale * w_scale;
        }

        // Step2: Weight matmul & dp4a dot accumulation for each output row tile
        unroll_for(int r = 0; r < NROW; r++)
        {
            const int out_n = tile0 + r;
            if (out_n >= N_SIZE)
                continue;

            // Weight block base: skip first 2 uc2 scale bytes (offset=32)
            const uint row_w_off = r * group_row_stride + 32u;
            const __global uchar* restrict w_q_ptr = sg_w_group + row_w_off;

            // Four coalesced 128B uc8 block reads (aligned 16-byte offsets, 512B total per subgroup)
            const uchar8 w0 = intel_sub_group_block_read_uc8(w_q_ptr +  0u * 16u);
            const uchar8 w1 = intel_sub_group_block_read_uc8(w_q_ptr +  8u * 16u);
            const uchar8 w2 = intel_sub_group_block_read_uc8(w_q_ptr + 16u * 16u);
            const uchar8 w3 = intel_sub_group_block_read_uc8(w_q_ptr + 24u * 16u);

            // Dual independent dp4a accumulators to hide dot instruction latency
            int S0 = 0, S1 = 0;
            S0 = dot_acc_sat_4x8packed_ss_int(as_uint(w0.lo), a_pack[0], S0);
            S1 = dot_acc_sat_4x8packed_ss_int(as_uint(w0.hi), a_pack[1], S1);
            S0 = dot_acc_sat_4x8packed_ss_int(as_uint(w1.lo), a_pack[2], S0);
            S1 = dot_acc_sat_4x8packed_ss_int(as_uint(w1.hi), a_pack[3], S1);
            S0 = dot_acc_sat_4x8packed_ss_int(as_uint(w2.lo), a_pack[4], S0);
            S1 = dot_acc_sat_4x8packed_ss_int(as_uint(w2.hi), a_pack[5], S1);
            S0 = dot_acc_sat_4x8packed_ss_int(as_uint(w3.lo), a_pack[6], S0);
            S1 = dot_acc_sat_4x8packed_ss_int(as_uint(w3.hi), a_pack[7], S1);

            // Dequant & accumulate partial sum
            const float total_dot = (float)(S0 + S1);
            partial[r] = mad(row_scale[r], total_dot, partial[r]);
        }
    }

    // Subgroup sum reduction + output write (lane 0 only per output row tile)
    unroll_for(int r = 0; r < NROW; r++)
    {
        const int out_n = tile0 + r;
        if (out_n >= N_SIZE)
            continue;

        const float sum = sub_group_reduce_add(partial[r]);
        if (lane != 0)
            continue;

        const uint output_idx = (uint)bm * N_SIZE + (uint)out_n;
#if HAS_FUSED_OPS
        const uint out_batch = (uint)bm / OUTPUT_FEATURE_NUM;
        const uint out_feat  = (uint)bm - out_batch * OUTPUT_FEATURE_NUM;
        OUTPUT_TYPE val = TO_OUTPUT_TYPE(sum);
        FUSED_OPS;
        C[output_idx] = FUSED_OPS_RESULT;
#else
        C[output_idx] = TO_OUTPUT_TYPE(sum);
#endif
    }
}
