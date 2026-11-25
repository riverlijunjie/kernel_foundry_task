// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// [EVOLVE_START]
// Q8_0 dp4a on SIZE-EXACT byte-transposed LISA layout (34 B/block, ZERO expansion).
// STRICTLY BANDWIDTH-BOUND GEMV (~1 dp4a/weight byte, 133 GB/s LPDDR5X). Runtime is
// fixed by the one-pass weight read; the only lever is transaction/issue efficiency.
//
// Recipe (converges to the 0.137ms BW wall):
//   (1) intel_sub_group_block_read_uc8 -> 128-byte coalesced bursts (4 reads/512 B,
//       half the transaction count vs uc4 -> lower issue overhead, deeper burst).
//   (2) NROW=4 register blocking: int8 activation prefetched ONCE per lane, reused 4x.
//       This is the ONLY genuine reuse (weight blocks are lane-private -> no cross-lane
//       reuse -> SLM tiling would only ADD a barrier on this BW-bound path -> omitted).
//   (3) All NROW small scale reads (uc2) issued UP FRONT so they fully overlap the big
//       weight bursts + dp4a chains.
//   (4) Two independent dp4a accumulators per row: hide dp4a latency at minimal GRF.
//   (5) __global const restrict on every read-only buffer -> free load reordering.

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#ifndef NROW
#define NROW 4
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
    const __global char*  const restrict aq_row  = Aq  + (uint)bm * (uint)K_SIZE;
    const __global float* const restrict asc_row = Asc + (uint)bm * (uint)(K_SIZE / 32);
    const __global uchar* const restrict Wr      = W;

    const uint group_stride = (uint)SG_SIZE * Q8E_BB;   // bytes per 16-block group, per row
    const uint row_stride   = (uint)GGUF_G * group_stride;

    float partial[NROW];
    unroll_for (int r = 0; r < NROW; ++r)
        partial[r] = 0.0f;

    for (int g = 0; g < GGUF_G; ++g) {
        const int kb = g * SG_SIZE + lane;

        // ---- Activation block: loaded ONCE, reused across all NROW output rows. ----
        const bool valid = (kb < bpr);
        const __global char*  aq_blk  = aq_row  + (uint)(valid ? kb : 0) * GGUF_BLOCK_ELEM;
        const __global float* asc_blk = asc_row + (uint)(valid ? kb : 0) * (GGUF_BLOCK_ELEM / 32);
        const float asc0 = valid ? asc_blk[0] : 0.0f;

        // Prefetch the 8 packed int8x4 activation words once (32 int8 -> 8 uints).
        // Two vload16 folds cut activation address arithmetic vs eight vload4.
        uint a_pack[8];
        {
            const char16 a_lo = vload16(0, aq_blk);
            const char16 a_hi = vload16(1, aq_blk);
            a_pack[0] = as_uint(a_lo.s0123); a_pack[1] = as_uint(a_lo.s4567);
            a_pack[2] = as_uint(a_lo.s89ab); a_pack[3] = as_uint(a_lo.scdef);
            a_pack[4] = as_uint(a_hi.s0123); a_pack[5] = as_uint(a_hi.s4567);
            a_pack[6] = as_uint(a_hi.s89ab); a_pack[7] = as_uint(a_hi.scdef);
        }

        // Base of group g for output row n0.
        const __global uchar* gbase =
            Wr + ((uint)n0 * (uint)GGUF_G + (uint)g) * group_stride;

        // ---- Issue all NROW small scale reads UP FRONT (overlap big weight bursts). ----
        float dscale[NROW];
        {
            const __global uchar* sb = gbase;
            unroll_for (int r = 0; r < NROW; ++r) {
                if ((n0 + r) < N_SIZE) {
                    const uchar2 sc = intel_sub_group_block_read_uc2(sb);
                    const ushort sbits = (ushort)sc.s0 | ((ushort)sc.s1 << 8);
                    dscale[r] = (valid ? (float)as_half(sbits) : 0.0f) * asc0;
                } else {
                    dscale[r] = 0.0f;
                }
                sb += row_stride;
            }
        }

        unroll_for (int r = 0; r < NROW; ++r) {
            if ((n0 + r) >= N_SIZE) {
                gbase += row_stride;
                continue;
            }

            // Quant bytes begin at byte-index 2 -> offset 2*16 = 32.
            const __global uchar* qbase = gbase + 32u;

            // FOUR coalesced 128-byte weight bursts (uc8): same 512 bytes/lane-group,
            // half the transaction count vs uc4 -> lower issue overhead, deeper burst.
            // Mutually independent -> issued up-front, kept in flight -> saturate LPDDR5X.
            const uchar8 wb0 = intel_sub_group_block_read_uc8(qbase + 0u  * 16u);
            const uchar8 wb1 = intel_sub_group_block_read_uc8(qbase + 8u  * 16u);
            const uchar8 wb2 = intel_sub_group_block_read_uc8(qbase + 16u * 16u);
            const uchar8 wb3 = intel_sub_group_block_read_uc8(qbase + 24u * 16u);

            // Two independent accumulators: hide dp4a latency at minimal GRF cost.
            // Each uchar8 holds two int8x4 dp4a words (little-endian reinterpret).
            int S0 = 0, S1 = 0;
            S0 = dot_acc_sat_4x8packed_ss_int(as_uint(wb0.lo), a_pack[0], S0);
            S1 = dot_acc_sat_4x8packed_ss_int(as_uint(wb0.hi), a_pack[1], S1);
            S0 = dot_acc_sat_4x8packed_ss_int(as_uint(wb1.lo), a_pack[2], S0);
            S1 = dot_acc_sat_4x8packed_ss_int(as_uint(wb1.hi), a_pack[3], S1);
            S0 = dot_acc_sat_4x8packed_ss_int(as_uint(wb2.lo), a_pack[4], S0);
            S1 = dot_acc_sat_4x8packed_ss_int(as_uint(wb2.hi), a_pack[5], S1);
            S0 = dot_acc_sat_4x8packed_ss_int(as_uint(wb3.lo), a_pack[6], S0);
            S1 = dot_acc_sat_4x8packed_ss_int(as_uint(wb3.hi), a_pack[7], S1);

            partial[r] = mad(dscale[r], (float)(S0 + S1), partial[r]);
            gbase += row_stride;   // next output row, same group
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
// [EVOLVE_END]