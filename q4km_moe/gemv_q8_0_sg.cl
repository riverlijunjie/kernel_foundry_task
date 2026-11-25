/*
 * gemv_q8_0_sg.cl  — High-performance Q8_0 GEMV with Intel sub-group block reads.
 *
 * == Q8_0 format recap ==
 *   GGUF block_q8_0 = { half d; int8_t qs[32]; }  (34 bytes, QK=32 weights)
 *     dequant:  w = q * d          (q is signed int8, no zero-point / min)
 *
 *   To reuse the 256-weight ("super-block") machinery of gemv_q4k_sg.cl we
 *   group EIGHT consecutive Q8_0 blocks into one super-block of 256 weights.
 *   Each super-block therefore carries:
 *     * 8 sub-blocks × 32 qs bytes            = 256 bytes  (weights,  "pqs")
 *     * 8 fp16 scales d                       =  16 bytes  (scales,   "pd")
 *   which equals the original 8 × 34 = 272 bytes — a pure reorder.
 *
 *   Because w = q * d and q is signed, there is NO min/bias term:
 *       acc += d_j * dot_j          dot_j = sum_i x_i * q_i   (32 weights)
 *
 * == Transposed ("SG") weight layout — SAME total size as original ==
 *
 *   pqs_T:  total_blk * 256 bytes
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 256 bytes (4096 B)
 *     Sub-block j at j*OPG*32, chunk c (0..7) at j*OPG*32 + c*OPG*4.
 *     One intel_sub_group_block_read8 per sub-block delivers 8 int8x4 chunks
 *     (32 signed weights) to each lane.
 *
 *   pd_T:   total_blk * 16 bytes
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes (256 B), SoA fields:
 *       field j (0..7) occupies OPG fp16, lane lid's d at j*OPG*2 + lid*2.
 *     One intel_sub_group_block_read_us8 fetches all 8 scales per lane.
 *
 * == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
 *   Same technique as gemv_q4k_sg.cl: KSPLIT sub-groups per work-group cover
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

/* Accumulate one 4-byte chunk (4 signed int8 weights) into the sub-block dot
 * product.  `in` points at slm base of the sub-block, `c` is the chunk index
 * (0..7); weights c*4..c*4+3 map to activations in[c*4..c*4+3]. */
#define ACC_CHUNK_Q8(dot, u, c, in)                                          \
    do {                                                                     \
        char4 _q = as_char4((uint)(u));                                      \
        (dot) = fma((in)[(c)*4 + 0], (float)_q.s0, (dot));                   \
        (dot) = fma((in)[(c)*4 + 1], (float)_q.s1, (dot));                   \
        (dot) = fma((in)[(c)*4 + 2], (float)_q.s2, (dot));                   \
        (dot) = fma((in)[(c)*4 + 3], (float)_q.s3, (dot));                   \
    } while (0)

/* Process one full sub-block j (32 signed weights = 8 chunks). */
#define Q8_SUBBLOCK(in, wv, sc, acc)                                         \
    do {                                                                     \
        float _dot = 0.0f;                                                   \
        ACC_CHUNK_Q8(_dot, (wv).s0, 0, in); ACC_CHUNK_Q8(_dot, (wv).s1, 1, in); \
        ACC_CHUNK_Q8(_dot, (wv).s2, 2, in); ACC_CHUNK_Q8(_dot, (wv).s3, 3, in); \
        ACC_CHUNK_Q8(_dot, (wv).s4, 4, in); ACC_CHUNK_Q8(_dot, (wv).s5, 5, in); \
        ACC_CHUNK_Q8(_dot, (wv).s6, 6, in); ACC_CHUNK_Q8(_dot, (wv).s7, 7, in); \
        (acc) = fma((sc), _dot, (acc));                                      \
    } while (0)

/* -----------------------------------------------------------------------
 * gemv_q8_0_sg
 * ----------------------------------------------------------------------- */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void gemv_q8_0_sg(
    __global const half   * restrict inputs,   /* [input_len]              */
    __global const uchar  * restrict pqs_T,    /* [total_blk * OPG * 256]  */
    __global const uchar  * restrict pd_T,     /* [total_blk * OPG * 16]   */
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

    float acc = 0.0f;

    /* Each sub-group processes a strided subset of the K-blocks. */
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {

        __local float *sb = sbase + (bid & 1u) * 256;   /* current buffer */

        /* =============================================================== *
         * Issue all long-latency global loads first (weights + scales).   *
         * Each sub-block is OPG*32 bytes = OPG*8 uints apart.              *
         * =============================================================== */
        uint pe_u = (h * nbpr + bid) * (OPG * 256 / 4);   /* uints */
        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u + 0*OPG*8);  /* sub 0 */
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 1*OPG*8);  /* sub 1 */
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 2*OPG*8);  /* sub 2 */
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 3*OPG*8);  /* sub 3 */
        uint8 w4 = intel_sub_group_block_read8(pqs_u + pe_u + 4*OPG*8);  /* sub 4 */
        uint8 w5 = intel_sub_group_block_read8(pqs_u + pe_u + 5*OPG*8);  /* sub 5 */
        uint8 w6 = intel_sub_group_block_read8(pqs_u + pe_u + 6*OPG*8);  /* sub 6 */
        uint8 w7 = intel_sub_group_block_read8(pqs_u + pe_u + 7*OPG*8);  /* sub 7 */

        /* 8 fp16 scales per lane (SoA), fetched with ONE wide block read. */
        uint    pd_off = (h * nbpr + bid) * (OPG * 8);    /* ushorts */
        ushort8 dv = intel_sub_group_block_read_us8(pd_us + pd_off);

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

        /* ---- Decode 8 fp16 scales (overlaps outstanding loads) ---- */
        float scale[8];
        {
            ushort dh[8];
            dh[0]=dv.s0; dh[1]=dv.s1; dh[2]=dv.s2; dh[3]=dv.s3;
            dh[4]=dv.s4; dh[5]=dv.s5; dh[6]=dv.s6; dh[7]=dv.s7;
            unroll_for (int i = 0; i < 8; i++)
                scale[i] = (float)as_half(dh[i]);
        }

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        /* ---- 8 sub-blocks: dot products + per-sub-block scale ---- */
        __local const float *in;
        in = sb + 0*32; Q8_SUBBLOCK(in, w0, scale[0], acc);
        in = sb + 1*32; Q8_SUBBLOCK(in, w1, scale[1], acc);
        in = sb + 2*32; Q8_SUBBLOCK(in, w2, scale[2], acc);
        in = sb + 3*32; Q8_SUBBLOCK(in, w3, scale[3], acc);
        in = sb + 4*32; Q8_SUBBLOCK(in, w4, scale[4], acc);
        in = sb + 5*32; Q8_SUBBLOCK(in, w5, scale[5], acc);
        in = sb + 6*32; Q8_SUBBLOCK(in, w6, scale[6], acc);
        in = sb + 7*32; Q8_SUBBLOCK(in, w7, scale[7], acc);
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
