/*
 * gemv_q4_1_sg.cl  — High-performance Q4_1 GEMV with Intel sub-group block reads.
 *
 * == Q4_1 format recap ==
 *   GGUF block_q4_1 = { half d; half m; uint8_t qs[16]; }  (20 bytes, QK=32 weights)
 *     qs[k] low  nibble = weight[k]      (k = 0..15)
 *     qs[k] high nibble = weight[k+16]
 *     dequant:  w = nibble * d + m       (nibble in 0..15, used DIRECTLY)
 *
 *   To reuse the 256-weight ("super-block") machinery of gemv_q4_0_sg.cl we
 *   group EIGHT consecutive Q4_1 blocks into one super-block of 256 weights.
 *   Each super-block therefore carries:
 *     * 8 sub-blocks x 16 qs bytes            = 128 bytes  (weights,  "pqs")
 *     * 8 fp16 scales d                       =  16 bytes  (scales,   "pd")
 *     * 8 fp16 minima m                       =  16 bytes  (minima,   "pm")
 *   which equals the original 8 x 20 = 160 bytes — a pure reorder.
 *
 *   The qs nibble layout is IDENTICAL to a Q4_0 / Q4_K sub-block, so the
 *   ACC_CHUNK dot-product macro is reused verbatim.  Because w = nibble*d + m
 *   we factor the per-sub-block scale/min out of the inner loop:
 *       acc += d_j * dot_j  +  m_j * isum_j
 *   This mirrors gemv_q4_0_sg.cl exactly, EXCEPT the min contribution is a
 *   per-block fp16 value (loaded from pm_T) added with a "+" sign, whereas
 *   Q4_0's min was the constant bias (8 * d_j) subtracted.
 *
 * == Transposed ("SG") weight layout — SAME total size as original ==
 *
 *   pqs_T:  total_blk * 128 bytes
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 128 bytes (2048 B)
 *     Sub-block j at j*OPG*16, chunk c (0..3) at j*OPG*16 + c*OPG*4.
 *
 *   pd_T:   total_blk * 16 bytes    (per-block scales d)
 *   pm_T:   total_blk * 16 bytes    (per-block minima m)
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes (256 B), SoA fields:
 *       field j (0..7) occupies OPG fp16, lane lid's value at j*OPG*2 + lid*2.
 *     One intel_sub_group_block_read_us8 fetches all 8 per lane.
 *
 * == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
 *   Same technique as gemv_q4_0_sg.cl: KSPLIT sub-groups per work-group cover
 *   the SAME OPG output rows but each owns a strided subset of the K-blocks;
 *   a single work-group barrier at the end reduces the partials.
 *   Total live threads = (N/OPG) * KSPLIT.
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

/* Accumulate one 4-byte chunk (8 quantized weights: 4 low + 4 high nibbles)
 * into the sub-block dot product.  `in` points at slm base of the sub-block,
 * `c` is the chunk index (0..3).  Identical to gemv_q4_0_sg.cl. */
#define ACC_CHUNK(dot, u, c, in)                                             \
    do {                                                                     \
        uint _w = (u);                                                       \
        (dot) = fma((in)[(c)*4 + 0],      (float)( _w        & 0xFu), (dot)); \
        (dot) = fma((in)[(c)*4 + 1],      (float)((_w >>  8) & 0xFu), (dot)); \
        (dot) = fma((in)[(c)*4 + 2],      (float)((_w >> 16) & 0xFu), (dot)); \
        (dot) = fma((in)[(c)*4 + 3],      (float)((_w >> 24) & 0xFu), (dot)); \
        (dot) = fma((in)[16 + (c)*4 + 0], (float)((_w >>  4) & 0xFu), (dot)); \
        (dot) = fma((in)[16 + (c)*4 + 1], (float)((_w >> 12) & 0xFu), (dot)); \
        (dot) = fma((in)[16 + (c)*4 + 2], (float)((_w >> 20) & 0xFu), (dot)); \
        (dot) = fma((in)[16 + (c)*4 + 3], (float)((_w >> 28) & 0xFu), (dot)); \
    } while (0)

/* -----------------------------------------------------------------------
 * gemv_q4_1_sg
 * ----------------------------------------------------------------------- */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void gemv_q4_1_sg(
    __global const half   * restrict inputs,   /* [input_len]              */
    __global const uchar  * restrict pqs_T,    /* [total_blk * OPG * 128]  */
    __global const uchar  * restrict pd_T,     /* [total_blk * OPG * 16]   */
    __global const uchar  * restrict pm_T,     /* [total_blk * OPG * 16]   */
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

    const __global uint   *pqs_u = (const __global uint   *)pqs_T;
    const __global ushort *pd_us = (const __global ushort *)pd_T;
    const __global ushort *pm_us = (const __global ushort *)pm_T;

    float acc = 0.0f;

    /* Each sub-group processes a strided subset of the K-blocks. */
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {

        __local float *sb = sbase + (bid & 1u) * 256;   /* current buffer */

        /* =============================================================== *
         * Issue all long-latency global loads first (weights + scales).   *
         * =============================================================== */
        uint pe_u = (h * nbpr + bid) * (OPG * 128 / 4);   /* uints */
        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);  /* sub 0,1 */
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);  /* sub 2,3 */
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);  /* sub 4,5 */
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);  /* sub 6,7 */

        /* 8 fp16 scales + 8 fp16 minima per lane (SoA), one wide read each. */
        uint    pd_off = (h * nbpr + bid) * (OPG * 8);    /* ushorts */
        ushort8 dv = intel_sub_group_block_read_us8(pd_us + pd_off);
        ushort8 mv = intel_sub_group_block_read_us8(pm_us + pd_off);

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

        /* ---- Decode 8 (scale, min) pairs (overlaps outstanding loads) ----
         * Q4_1:  w = nibble * d + m  =>  scale_j = d_j , minv_j = m_j        */
        float scale[8], minv[8];
        {
            ushort dh[8], mh[8];
            dh[0]=dv.s0; dh[1]=dv.s1; dh[2]=dv.s2; dh[3]=dv.s3;
            dh[4]=dv.s4; dh[5]=dv.s5; dh[6]=dv.s6; dh[7]=dv.s7;
            mh[0]=mv.s0; mh[1]=mv.s1; mh[2]=mv.s2; mh[3]=mv.s3;
            mh[4]=mv.s4; mh[5]=mv.s5; mh[6]=mv.s6; mh[7]=mv.s7;
            unroll_for (int i = 0; i < 8; i++) {
                scale[i] = (float)as_half(dh[i]);
                minv[i]  = (float)as_half(mh[i]);
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
        ACC_CHUNK(dot, w0.s0, 0, in); ACC_CHUNK(dot, w0.s1, 1, in);
        ACC_CHUNK(dot, w0.s2, 2, in); ACC_CHUNK(dot, w0.s3, 3, in);
        acc = fma(scale[0], dot, acc); acc = fma(minv[0], isum[0], acc);

        in = sb + 1*32; dot = 0.0f;
        ACC_CHUNK(dot, w0.s4, 0, in); ACC_CHUNK(dot, w0.s5, 1, in);
        ACC_CHUNK(dot, w0.s6, 2, in); ACC_CHUNK(dot, w0.s7, 3, in);
        acc = fma(scale[1], dot, acc); acc = fma(minv[1], isum[1], acc);

        in = sb + 2*32; dot = 0.0f;
        ACC_CHUNK(dot, w1.s0, 0, in); ACC_CHUNK(dot, w1.s1, 1, in);
        ACC_CHUNK(dot, w1.s2, 2, in); ACC_CHUNK(dot, w1.s3, 3, in);
        acc = fma(scale[2], dot, acc); acc = fma(minv[2], isum[2], acc);

        in = sb + 3*32; dot = 0.0f;
        ACC_CHUNK(dot, w1.s4, 0, in); ACC_CHUNK(dot, w1.s5, 1, in);
        ACC_CHUNK(dot, w1.s6, 2, in); ACC_CHUNK(dot, w1.s7, 3, in);
        acc = fma(scale[3], dot, acc); acc = fma(minv[3], isum[3], acc);

        in = sb + 4*32; dot = 0.0f;
        ACC_CHUNK(dot, w2.s0, 0, in); ACC_CHUNK(dot, w2.s1, 1, in);
        ACC_CHUNK(dot, w2.s2, 2, in); ACC_CHUNK(dot, w2.s3, 3, in);
        acc = fma(scale[4], dot, acc); acc = fma(minv[4], isum[4], acc);

        in = sb + 5*32; dot = 0.0f;
        ACC_CHUNK(dot, w2.s4, 0, in); ACC_CHUNK(dot, w2.s5, 1, in);
        ACC_CHUNK(dot, w2.s6, 2, in); ACC_CHUNK(dot, w2.s7, 3, in);
        acc = fma(scale[5], dot, acc); acc = fma(minv[5], isum[5], acc);

        in = sb + 6*32; dot = 0.0f;
        ACC_CHUNK(dot, w3.s0, 0, in); ACC_CHUNK(dot, w3.s1, 1, in);
        ACC_CHUNK(dot, w3.s2, 2, in); ACC_CHUNK(dot, w3.s3, 3, in);
        acc = fma(scale[6], dot, acc); acc = fma(minv[6], isum[6], acc);

        in = sb + 7*32; dot = 0.0f;
        ACC_CHUNK(dot, w3.s4, 0, in); ACC_CHUNK(dot, w3.s5, 1, in);
        ACC_CHUNK(dot, w3.s6, 2, in); ACC_CHUNK(dot, w3.s7, 3, in);
        acc = fma(scale[7], dot, acc); acc = fma(minv[7], isum[7], acc);
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
