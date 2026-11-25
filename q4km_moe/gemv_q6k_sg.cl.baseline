/*
 * gemv_q6k_sg.cl  — High-performance Q6_K GEMV with Intel sub-group block reads.
 *
 * == Transposed ("SG") weight layout — SAME total size as original ==
 *
 * Original layout (4 separate arrays):
 *   pql[total_blk * 128]  -- lower 4-bit quants, 8 sub-blocks × 16 B
 *   pqh[total_blk * 64]   -- upper 2-bit quants, 8 sub-blocks × 8 B
 *   ps [total_blk * 16]   -- int8 scales (16 per block)
 *   pd [total_blk * 2]    -- fp16 super-scale d
 *
 * Transposed layout (same total sizes):
 *
 *   pql_T:  total_blk * 128 bytes
 *     Entry (h, bid): OPG * 128 = 2048 bytes at (h*nbpr+bid) * OPG * 128
 *     Sub-block j at j * OPG * 16 within entry
 *     Chunk c (0..3) at j*OPG*16 + c*OPG*4 within entry
 *     Lane lid gets: pql[blk*128 + j*16 + c*4 .. +3]  where blk=(h*OPG+lid)*nbpr+bid
 *
 *   pqh_T:  total_blk * 64 bytes
 *     Entry (h, bid): OPG * 64 = 1024 bytes at (h*nbpr+bid) * OPG * 64
 *     Sub-block j at j * OPG * 8 within entry
 *     Chunk c (0..1) at j*OPG*8 + c*OPG*4 within entry
 *     Lane lid gets: pqh[blk*64 + j*8 + c*4 .. +3]
 *
 *   ps_T:   total_blk * 16 bytes
 *     Entry (h, bid): OPG * 16 = 256 bytes at (h*nbpr+bid) * OPG * 16
 *     Scale index si (0..15) at si * OPG bytes within entry
 *     Lane lid: 1 int8 byte → read with intel_sub_group_block_read_uc
 *
 *   pd_T:   total_blk * 2 bytes
 *     Entry (h, bid): OPG * 2 = 32 bytes at (h*nbpr+bid) * OPG * 2
 *     Lane lid: 1 ushort (fp16) → read with intel_sub_group_block_read_us
 *
 * == Dispatch ==
 *   global = output_len,  local = OPG = 16
 *   h  = get_group_id(0),  hh = get_local_id(0) = lane = row within group
 *
 * == Verification ==
 *   sizeof(pql_T) = (N/OPG)*nbpr * OPG*128 = N*nbpr*128 = N*K/256*128  ✓ (= pql)
 *   sizeof(pqh_T) = N*K/256*64   ✓
 *   sizeof(ps_T)  = N*K/256*16   ✓
 *   sizeof(pd_T)  = N*K/256*2    ✓
 */

#pragma OPENCL EXTENSION cl_khr_fp16        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define OPG_Q6K 16

inline float _q6k_u16_to_float(ushort raw) { return (float)as_half(raw); }

/*
 * Helper macro: reconstruct one weight pair (lo half + hi half of sub-block j, position k)
 * and accumulate into acc.
 *   ql_byte = pql nibbles for position k (lo nibble = lower weight, hi nibble = upper weight)
 *   qh_lo   = 32-bit qh word for sub-block j lower half (qh_T bytes 0..3)
 *   qh_hi   = 32-bit qh word for sub-block j upper half (qh_T bytes 4..7)
 *   sh       = bit shift from shift table for position k
 *   clo, chi = d * scale[2j], d * scale[2j+1]
 *
 * Shift table: {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30}
 * Position k in the sub-block maps to shift table entry k.
 */
#define Q6K_ACCUM(inp_lo, inp_hi, ql_byte, qh_lo_w, qh_hi_w, sh, clo, chi, acc) \
    do { \
        uint upper_lo = ((qh_lo_w) >> (sh)) & 0x03u; \
        uint upper_hi = ((qh_hi_w) >> (sh)) & 0x03u; \
        float wlo = (float)(((ql_byte) & 0x0Fu) | (upper_lo << 4)) - 32.0f; \
        float whi = (float)((((ql_byte) >> 4) & 0x0Fu) | (upper_hi << 4)) - 32.0f; \
        (acc) = fma((inp_lo), wlo * (clo), (acc)); \
        (acc) = fma((inp_hi), whi * (chi), (acc)); \
    } while (0)

__attribute__((intel_reqd_sub_group_size(OPG_Q6K)))
__kernel void gemv_q6k_sg(
    __global const half   *inputs,   /* [input_len]               */
    __global const uchar  *pql_T,    /* [total_blk * OPG * 128]   same size as pql */
    __global const uchar  *pqh_T,    /* [total_blk * OPG * 64]    same size as pqh */
    __global const char   *ps_T,     /* [total_blk * OPG * 16]    same size as ps  */
    __global const uchar  *pd_T,     /* [total_blk * OPG * 2]     same size as pd  */
    __global       half   *outputs,  /* [output_len]               */
    uint input_len,
    uint output_len
) {
    const uint nbpr = input_len / 256;

    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);

    __local float slm_inp[256];
    __local float slm_acc[OPG_Q6K];
    slm_acc[hh] = 0.0f;

    for (uint bid = 0; bid < nbpr; bid++) {

        /* ---- Load 256 inputs into SLM ---- */
        {
            const __global ushort *inp_us = (const __global ushort *)(inputs + bid * 256);
            for (uint k = 0; k < 256 / OPG_Q6K; k++) {
                ushort v = intel_sub_group_block_read_us(inp_us + k * OPG_Q6K);
                slm_inp[k * OPG_Q6K + hh] = (float)as_half(v);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* ---- Read d and 16 int8 scales ---- */
        uint pd_entry = (h * nbpr + bid) * OPG_Q6K * 2;
        ushort d_raw = intel_sub_group_block_read_us(
            (const __global ushort *)(pd_T + pd_entry));
        float d = _q6k_u16_to_float(d_raw);

        uint ps_entry = (h * nbpr + bid) * OPG_Q6K * 16;
        char sc[16];
        for (int si = 0; si < 16; si++) {
            sc[si] = (char)intel_sub_group_block_read_uc(
                (const __global uchar *)(ps_T + ps_entry + (uint)si * OPG_Q6K));
        }

        /* ---- Weight loads + FMA over 8 sub-blocks ---- */
        uint pql_entry = (h * nbpr + bid) * OPG_Q6K * 128;  /* 2048 B */
        uint pqh_entry = (h * nbpr + bid) * OPG_Q6K * 64;   /* 1024 B */
        float acc = 0.0f;

        for (int j = 0; j < 8; j++) {
            float clo = d * (float)sc[2 * j];
            float chi = d * (float)sc[2 * j + 1];

            /* pql_T: sub-block j, 4 chunks × OPG*4 bytes */
            uint pql_j = pql_entry + (uint)j * OPG_Q6K * 16;
            uint ql0w = intel_sub_group_block_read((const __global uint *)(pql_T + pql_j + 0 * OPG_Q6K * 4));
            uint ql1w = intel_sub_group_block_read((const __global uint *)(pql_T + pql_j + 1 * OPG_Q6K * 4));
            uint ql2w = intel_sub_group_block_read((const __global uint *)(pql_T + pql_j + 2 * OPG_Q6K * 4));
            uint ql3w = intel_sub_group_block_read((const __global uint *)(pql_T + pql_j + 3 * OPG_Q6K * 4));

            /* pqh_T: sub-block j, 2 chunks × OPG*4 bytes */
            uint pqh_j = pqh_entry + (uint)j * OPG_Q6K * 8;
            uint qh0w = intel_sub_group_block_read((const __global uint *)(pqh_T + pqh_j + 0 * OPG_Q6K * 4));
            uint qh1w = intel_sub_group_block_read((const __global uint *)(pqh_T + pqh_j + 1 * OPG_Q6K * 4));

            /* Reconstruct 32-bit qh words for the shift-table extraction */
            uint qh_lo = qh0w;
            uint qh_hi = qh1w;

            uchar ql00 = (uchar)( ql0w        & 0xFFu);
            uchar ql01 = (uchar)((ql0w >>  8) & 0xFFu);
            uchar ql02 = (uchar)((ql0w >> 16) & 0xFFu);
            uchar ql03 = (uchar)((ql0w >> 24) & 0xFFu);
            uchar ql10 = (uchar)( ql1w        & 0xFFu);
            uchar ql11 = (uchar)((ql1w >>  8) & 0xFFu);
            uchar ql12 = (uchar)((ql1w >> 16) & 0xFFu);
            uchar ql13 = (uchar)((ql1w >> 24) & 0xFFu);
            uchar ql20 = (uchar)( ql2w        & 0xFFu);
            uchar ql21 = (uchar)((ql2w >>  8) & 0xFFu);
            uchar ql22 = (uchar)((ql2w >> 16) & 0xFFu);
            uchar ql23 = (uchar)((ql2w >> 24) & 0xFFu);
            uchar ql30 = (uchar)( ql3w        & 0xFFu);
            uchar ql31 = (uchar)((ql3w >>  8) & 0xFFu);
            uchar ql32 = (uchar)((ql3w >> 16) & 0xFFu);
            uchar ql33 = (uchar)((ql3w >> 24) & 0xFFu);

            /* 16 positions; shift table {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30} */
            /* chunk 0: k=0..3  (ql0),  shift = {0, 8, 16, 24} */
            Q6K_ACCUM(slm_inp[j*32+ 0], slm_inp[j*32+16], ql00, qh_lo, qh_hi,  0, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+ 1], slm_inp[j*32+17], ql01, qh_lo, qh_hi,  8, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+ 2], slm_inp[j*32+18], ql02, qh_lo, qh_hi, 16, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+ 3], slm_inp[j*32+19], ql03, qh_lo, qh_hi, 24, clo, chi, acc);
            /* chunk 1: k=4..7  (ql1),  shift = {2, 10, 18, 26} */
            Q6K_ACCUM(slm_inp[j*32+ 4], slm_inp[j*32+20], ql10, qh_lo, qh_hi,  2, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+ 5], slm_inp[j*32+21], ql11, qh_lo, qh_hi, 10, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+ 6], slm_inp[j*32+22], ql12, qh_lo, qh_hi, 18, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+ 7], slm_inp[j*32+23], ql13, qh_lo, qh_hi, 26, clo, chi, acc);
            /* chunk 2: k=8..11 (ql2),  shift = {4, 12, 20, 28} */
            Q6K_ACCUM(slm_inp[j*32+ 8], slm_inp[j*32+24], ql20, qh_lo, qh_hi,  4, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+ 9], slm_inp[j*32+25], ql21, qh_lo, qh_hi, 12, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+10], slm_inp[j*32+26], ql22, qh_lo, qh_hi, 20, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+11], slm_inp[j*32+27], ql23, qh_lo, qh_hi, 28, clo, chi, acc);
            /* chunk 3: k=12..15 (ql3), shift = {6, 14, 22, 30} */
            Q6K_ACCUM(slm_inp[j*32+12], slm_inp[j*32+28], ql30, qh_lo, qh_hi,  6, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+13], slm_inp[j*32+29], ql31, qh_lo, qh_hi, 14, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+14], slm_inp[j*32+30], ql32, qh_lo, qh_hi, 22, clo, chi, acc);
            Q6K_ACCUM(slm_inp[j*32+15], slm_inp[j*32+31], ql33, qh_lo, qh_hi, 30, clo, chi, acc);
        }

        slm_acc[hh] += acc;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint row = h * OPG_Q6K + hh;
    if (row < output_len) {
        outputs[row] = convert_half_rte(slm_acc[hh]);
    }
}
