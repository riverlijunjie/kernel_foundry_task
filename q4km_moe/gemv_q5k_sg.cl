/*
 * gemv_q5k_sg.cl  — High-performance Q5_K GEMV with Intel sub-group block reads.
 *
 * This is the Q5_K counterpart of gemv_q4k_sg.cl / gemv_q6k_sg.cl, built on
 * the same "transposed SG layout + K-split occupancy" optimization strategy
 * used in gemv_q6k_sg.cl.
 *
 * == Q5_K format recap (see gemv_q5k_shuffle.cl) ==
 *   Row-major shuffled buffers (produced by ShuffleQ5KL1):
 *     pqs [num_blocks * 128]  -- packed 4-bit low quants (same packing as Q4_K)
 *     pqh [num_blocks *  32]  -- packed high bit (1 bit/weight, 4 bytes/sub-block)
 *     psl [num_blocks *  16]  -- scales/mins/d/dmin (identical 16-byte layout to Q4_K)
 *
 *   Per block (256 weights), sub-block j (j=0..7) covers 32 weights:
 *     qs[j*16+k] = low4(w[j*32+k]) | (low4(w[j*32+16+k]) << 4)     k=0..15
 *     high bit of weight index wi (0..31) within sub-block j is bit
 *       bit_idx = wi/4  of byte  byte_idx = wi%4  of the 4-byte word
 *       qh_word = qh[j*4 .. j*4+3] (little-endian uint32)
 *     weight = low_nibble | (high_bit << 4)      (range 0..31)
 *     dequant: weight * scale[j] - min[j]
 *
 * == Transposed ("SG") weight layout — SAME total size as original ==
 *
 * Group rows by OPG=16. For row-group h and K-block bid the OPG blocks are
 * interleaved in 4-byte chunks so a single sub-group block read delivers
 * 4 contiguous bytes to each of the 16 lanes (identical scheme to pqs_T in
 * gemv_q4k_sg.cl).
 *
 *   pqs_T:  total_blk * 128 bytes
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 128 bytes (2048 B)
 *     Sub-block j at j*OPG*16, chunk c (0..3) at j*OPG*16 + c*OPG*4.
 *
 *   pqh_T:  total_blk * 32 bytes
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 32 bytes (512 B)
 *     Sub-block j's 4-byte qh word for all OPG lanes at j*OPG*4
 *     (lane lid's word at j*OPG*4 + lid*4) — one intel_sub_group_block_read8
 *     per K-block fetches all 8 sub-block qh words at once.
 *
 *   psl_T:  total_blk * 16 bytes  (identical SoA layout to gemv_q4k_sg.cl)
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes (256 B), SoA fields:
 *       [ sl_u32×OPG=64B | ml_u32×OPG=64B | sh_u16×OPG=32B |
 *         mh_u16×OPG=32B | d_u16×OPG=32B  | dmin_u16×OPG=32B ]
 *
 * == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
 *   Same technique as gemv_q4k_sg.cl / gemv_q6k_sg.cl: KSPLIT sub-groups per
 *   work-group cover the SAME OPG output rows but each owns a strided subset
 *   of K-blocks; a single work-group barrier at the end reduces the partials.
 *   Total live threads = (N/OPG) * KSPLIT.
 *
 * == Other memory optimizations (unchanged strategy) ==
 *   * Wide intel_sub_group_block_read8 weight loads; all long-latency global
 *     loads issued BEFORE the SLM sync so they overlap decode work.
 *   * Per-block quantized scale/min factored OUT of the inner loop:
 *         acc += scale_j*dot_j - min_j*isum_j
 *   * Per-sub-group activation SLM is DOUBLE-BUFFERED: exactly ONE
 *     sub_group_barrier per K-block; sub-groups never wait on each other in
 *     the K-loop.
 *
 * == Dispatch ==
 *   global = (output_len, KSPLIT)   local = (OPG=16, KSPLIT)
 *   h  = get_group_id(0)   hh = get_local_id(0)   sg = get_local_id(1)
 */

#pragma OPENCL EXTENSION cl_khr_fp16        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define OPG 16  /* = sub_group_size */

#ifndef KSPLIT
#define KSPLIT 1
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

/* Accumulate one 4-byte qs chunk (8 quantized 4-bit weights: 4 low + 4 high
 * nibbles, i.e. weight positions k=c*4..c*4+3 for the low half and
 * k+16=c*4+16..c*4+19 for the high half) plus the corresponding high bits
 * extracted from the 32-bit qh word for this sub-block.
 *
 * Derivation of the high-bit shift (see gemv_q5k_shuffle.cl):
 *   low-half weight index  wi = c*4+i   -> byte_idx = i, bit_idx = c
 *   high-half weight index wi = c*4+i+16 -> byte_idx = i, bit_idx = c+4
 *   high_bit = (qh_word >> (byte_idx*8 + bit_idx)) & 1
 */
#define ACC_CHUNK_Q5(dot, u, qhw, c, in)                                     \
    do {                                                                     \
        uint _w = (u);                                                       \
        (dot) = fma((in)[(c)*4 + 0],                                         \
                    (float)(( _w        & 0xFu) |                            \
                            ((((qhw) >> (0*8 + (c)    )) & 1u) << 4)),        \
                    (dot));                                                  \
        (dot) = fma((in)[(c)*4 + 1],                                         \
                    (float)(((_w >>  8) & 0xFu) |                            \
                            ((((qhw) >> (1*8 + (c)    )) & 1u) << 4)),        \
                    (dot));                                                  \
        (dot) = fma((in)[(c)*4 + 2],                                         \
                    (float)(((_w >> 16) & 0xFu) |                            \
                            ((((qhw) >> (2*8 + (c)    )) & 1u) << 4)),        \
                    (dot));                                                  \
        (dot) = fma((in)[(c)*4 + 3],                                         \
                    (float)(((_w >> 24) & 0xFu) |                            \
                            ((((qhw) >> (3*8 + (c)    )) & 1u) << 4)),        \
                    (dot));                                                  \
        (dot) = fma((in)[16 + (c)*4 + 0],                                    \
                    (float)(((_w >>  4) & 0xFu) |                            \
                            ((((qhw) >> (0*8 + (c) + 4)) & 1u) << 4)),        \
                    (dot));                                                  \
        (dot) = fma((in)[16 + (c)*4 + 1],                                    \
                    (float)(((_w >> 12) & 0xFu) |                            \
                            ((((qhw) >> (1*8 + (c) + 4)) & 1u) << 4)),        \
                    (dot));                                                  \
        (dot) = fma((in)[16 + (c)*4 + 2],                                    \
                    (float)(((_w >> 20) & 0xFu) |                            \
                            ((((qhw) >> (2*8 + (c) + 4)) & 1u) << 4)),        \
                    (dot));                                                  \
        (dot) = fma((in)[16 + (c)*4 + 3],                                    \
                    (float)(((_w >> 28) & 0xFu) |                            \
                            ((((qhw) >> (3*8 + (c) + 4)) & 1u) << 4)),        \
                    (dot));                                                  \
    } while (0)

/* -----------------------------------------------------------------------
 * gemv_q5k_sg
 * ----------------------------------------------------------------------- */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void gemv_q5k_sg(
    __global const half   * restrict inputs,   /* [input_len]              */
    __global const uchar  * restrict pqs_T,    /* [total_blk * OPG * 128]  */
    __global const uchar  * restrict pqh_T,    /* [total_blk * OPG * 32]   */
    __global const uchar  * restrict psl_T,    /* [total_blk * OPG * 16]   */
    __global       half   * restrict outputs,  /* [output_len]             */
    uint input_len,
    uint output_len
) {
    const uint nbpr = input_len / 256;

    const uint h  = get_group_id(0);    /* row-group  */
    const uint hh = get_local_id(0);    /* lane = output row within group */
    const uint sg = get_local_id(1);    /* sub-group / K-split index       */

    /* Per-sub-group activation SLM (double-buffered) + reduction scratch. */
    __local float slm_inp[KSPLIT * 2 * 256];
    __local float slm_red[KSPLIT * OPG];
    __local float *sbase = slm_inp + sg * (2 * 256);

    const __global uint    *pqs_u  = (const __global uint    *)pqs_T;
    const __global uint    *pqh_u  = (const __global uint    *)pqh_T;
    const __global uint    *psl_pu = (const __global uint    *)psl_T;
    const __global ushort  *psl_us = (const __global ushort  *)psl_T;

    float acc = 0.0f;

    /* Each sub-group processes a strided subset of the K-blocks. */
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {

        __local float *sb = sbase + (bid & 1u) * 256;   /* current buffer */

        /* =============================================================== *
         * Issue all long-latency global loads first (weights + scales).   *
         * =============================================================== */
        uint pqs_e = (h * nbpr + bid) * (OPG * 128 / 4);   /* uints */
        uint8 w0 = intel_sub_group_block_read8(pqs_u + pqs_e +   0);  /* sub 0,1 */
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pqs_e + 128);  /* sub 2,3 */
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pqs_e + 256);  /* sub 4,5 */
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pqs_e + 384);  /* sub 6,7 */

        uint pqh_e = (h * nbpr + bid) * (OPG * 32 / 4);    /* uints */
        uint8 hq = intel_sub_group_block_read8(pqh_u + pqh_e);  /* 8 sub-block qh words */

        uint   psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);   /* uints  */
        uint   psl_us_off = (h * nbpr + bid) * (OPG * 16 / 2);   /* ushort */
        uint   sl     = intel_sub_group_block_read   (psl_pu + psl_pu_off +  0);
        uint   ml     = intel_sub_group_block_read   (psl_pu + psl_pu_off + 16);
        ushort sh     = intel_sub_group_block_read_us(psl_us + psl_us_off + 64);
        ushort mh     = intel_sub_group_block_read_us(psl_us + psl_us_off + 80);
        ushort d_raw  = intel_sub_group_block_read_us(psl_us + psl_us_off + 96);
        ushort dn_raw = intel_sub_group_block_read_us(psl_us + psl_us_off + 112);

        /* ---- Load 256 activations into SLM with ONE wide block read ---- */
        {
            const __global uint *inp_u = (const __global uint *)(inputs + bid * 256);
            uint8 iv = intel_sub_group_block_read8(inp_u);
            half2 t;
            t = as_half2(iv.s0); sb[(0*OPG + hh)*2 + 0] = t.s0; sb[(0*OPG + hh)*2 + 1] = t.s1;
            t = as_half2(iv.s1); sb[(1*OPG + hh)*2 + 0] = t.s0; sb[(1*OPG + hh)*2 + 1] = t.s1;
            t = as_half2(iv.s2); sb[(2*OPG + hh)*2 + 0] = t.s0; sb[(2*OPG + hh)*2 + 1] = t.s1;
            t = as_half2(iv.s3); sb[(3*OPG + hh)*2 + 0] = t.s0; sb[(3*OPG + hh)*2 + 1] = t.s1;
            t = as_half2(iv.s4); sb[(4*OPG + hh)*2 + 0] = t.s0; sb[(4*OPG + hh)*2 + 1] = t.s1;
            t = as_half2(iv.s5); sb[(5*OPG + hh)*2 + 0] = t.s0; sb[(5*OPG + hh)*2 + 1] = t.s1;
            t = as_half2(iv.s6); sb[(6*OPG + hh)*2 + 0] = t.s0; sb[(6*OPG + hh)*2 + 1] = t.s1;
            t = as_half2(iv.s7); sb[(7*OPG + hh)*2 + 0] = t.s0; sb[(7*OPG + hh)*2 + 1] = t.s1;
        }

        /* ---- Decode 8 (scale, min) pairs (overlaps outstanding loads) ---- */
        float scale[8], minv[8];
        {
            float d    = (float)as_half(d_raw);
            float dmin = (float)as_half(dn_raw);
            unroll_for (int i = 0; i < 8; i++) {
                uint sq = ((sl >> (i * 4)) & 0xFu) | ((uint)((sh >> (i * 2)) & 0x3u) << 4);
                uint mq = ((ml >> (i * 4)) & 0xFu) | ((uint)((mh >> (i * 2)) & 0x3u) << 4);
                scale[i] = (float)sq * d;
                minv[i]  = (float)mq * dmin;
            }
        }

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        /* ---- Per-sub-block activation sums (weight independent) ---- */
        float isum[8];
        unroll_for (int j = 0; j < 8; j++) {
            float s = 0.0f;
            unroll_for (int i = 0; i < 32; i++)
                s += sb[j * 32 + i];
            isum[j] = s;
        }

        /* ---- Dot products + factored scale/min combine ---- */
        float dot;
        __local const float *in;

        in = sb + 0*32; dot = 0.0f;
        ACC_CHUNK_Q5(dot, w0.s0, hq.s0, 0, in); ACC_CHUNK_Q5(dot, w0.s1, hq.s0, 1, in);
        ACC_CHUNK_Q5(dot, w0.s2, hq.s0, 2, in); ACC_CHUNK_Q5(dot, w0.s3, hq.s0, 3, in);
        acc = fma(scale[0], dot, acc); acc = fma(-minv[0], isum[0], acc);

        in = sb + 1*32; dot = 0.0f;
        ACC_CHUNK_Q5(dot, w0.s4, hq.s1, 0, in); ACC_CHUNK_Q5(dot, w0.s5, hq.s1, 1, in);
        ACC_CHUNK_Q5(dot, w0.s6, hq.s1, 2, in); ACC_CHUNK_Q5(dot, w0.s7, hq.s1, 3, in);
        acc = fma(scale[1], dot, acc); acc = fma(-minv[1], isum[1], acc);

        in = sb + 2*32; dot = 0.0f;
        ACC_CHUNK_Q5(dot, w1.s0, hq.s2, 0, in); ACC_CHUNK_Q5(dot, w1.s1, hq.s2, 1, in);
        ACC_CHUNK_Q5(dot, w1.s2, hq.s2, 2, in); ACC_CHUNK_Q5(dot, w1.s3, hq.s2, 3, in);
        acc = fma(scale[2], dot, acc); acc = fma(-minv[2], isum[2], acc);

        in = sb + 3*32; dot = 0.0f;
        ACC_CHUNK_Q5(dot, w1.s4, hq.s3, 0, in); ACC_CHUNK_Q5(dot, w1.s5, hq.s3, 1, in);
        ACC_CHUNK_Q5(dot, w1.s6, hq.s3, 2, in); ACC_CHUNK_Q5(dot, w1.s7, hq.s3, 3, in);
        acc = fma(scale[3], dot, acc); acc = fma(-minv[3], isum[3], acc);

        in = sb + 4*32; dot = 0.0f;
        ACC_CHUNK_Q5(dot, w2.s0, hq.s4, 0, in); ACC_CHUNK_Q5(dot, w2.s1, hq.s4, 1, in);
        ACC_CHUNK_Q5(dot, w2.s2, hq.s4, 2, in); ACC_CHUNK_Q5(dot, w2.s3, hq.s4, 3, in);
        acc = fma(scale[4], dot, acc); acc = fma(-minv[4], isum[4], acc);

        in = sb + 5*32; dot = 0.0f;
        ACC_CHUNK_Q5(dot, w2.s4, hq.s5, 0, in); ACC_CHUNK_Q5(dot, w2.s5, hq.s5, 1, in);
        ACC_CHUNK_Q5(dot, w2.s6, hq.s5, 2, in); ACC_CHUNK_Q5(dot, w2.s7, hq.s5, 3, in);
        acc = fma(scale[5], dot, acc); acc = fma(-minv[5], isum[5], acc);

        in = sb + 6*32; dot = 0.0f;
        ACC_CHUNK_Q5(dot, w3.s0, hq.s6, 0, in); ACC_CHUNK_Q5(dot, w3.s1, hq.s6, 1, in);
        ACC_CHUNK_Q5(dot, w3.s2, hq.s6, 2, in); ACC_CHUNK_Q5(dot, w3.s3, hq.s6, 3, in);
        acc = fma(scale[6], dot, acc); acc = fma(-minv[6], isum[6], acc);

        in = sb + 7*32; dot = 0.0f;
        ACC_CHUNK_Q5(dot, w3.s4, hq.s7, 0, in); ACC_CHUNK_Q5(dot, w3.s5, hq.s7, 1, in);
        ACC_CHUNK_Q5(dot, w3.s6, hq.s7, 2, in); ACC_CHUNK_Q5(dot, w3.s7, hq.s7, 3, in);
        acc = fma(scale[7], dot, acc); acc = fma(-minv[7], isum[7], acc);
    }

    /* ---- Reduce KSPLIT partial sums, sub-group 0 writes the fp16 output. */
#if KSPLIT == 1
    uint row = h * OPG + hh;
    if (row < output_len)
        outputs[row] = convert_half_rte(acc);
#else
    slm_red[sg * OPG + hh] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float sum = 0.0f;
        unroll_for (int g = 0; g < KSPLIT; g++)
            sum += slm_red[g * OPG + hh];
        uint row = h * OPG + hh;
        if (row < output_len)
            outputs[row] = convert_half_rte(sum);
    }
#endif
}
