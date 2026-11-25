/*
 * gemv_q4k_sg.cl  — High-performance Q4_K GEMV with Intel sub-group block reads.
 *
 * == Transposed ("SG") weight layout — SAME total size as original ==
 *
 * Original (row-major) layout:
 *   pqs[total_blk * 128]   psl[total_blk * 16]
 *   where total_blk = N * K / 256
 *
 *   For block blk = row * nbpr + bid:
 *     pqs[blk*128 .. blk*128+127]  -- 8 sub-blocks × 16 bytes
 *     psl[blk*16  .. blk*16+15]    -- sl(4B) ml(4B) sh(2B) mh(2B) d(2B) dmin(2B)
 *
 * Transposed layout (produced by pack_q4k_sg in the test):
 *   Group rows by OPG=16. For row-group h and K-block bid the OPG blocks are
 *   interleaved in 4-byte chunks so a single sub-group block read delivers
 *   4 contiguous weight bytes to each of the 16 lanes.
 *
 * pqs_T:  total_blk * 128 bytes
 *   Entry (h, bid) at byte offset (h*nbpr + bid) * OPG * 128   (2048 B)
 *   Sub-block j at j*OPG*16, chunk c (0..3) at j*OPG*16 + c*OPG*4.
 *
 * psl_T:  total_blk * 16 bytes
 *   Entry (h, bid) at (h*nbpr + bid) * OPG * 16  (256 B), SoA fields:
 *     [ sl_u32×OPG=64B | ml_u32×OPG=64B | sh_u16×OPG=32B |
 *       mh_u16×OPG=32B | d_u16×OPG=32B  | dmin_u16×OPG=32B ]
 *
 * == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
 *   The original dispatch used ONE sub-group per work-group, so the number of
 *   concurrent hardware threads == N/OPG.  For small N this leaves the machine
 *   badly under-occupied (e.g. N=1024 -> only 64 threads on a 160-XVE GPU),
 *   which is the dominant bottleneck for this memory-bound kernel: too few
 *   in-flight loads to keep the DRAM pipe full.
 *
 *   Fix: put KSPLIT sub-groups in each work-group (local = OPG x KSPLIT).  All
 *   KSPLIT sub-groups of a group cover the SAME OPG output rows but each owns a
 *   strided subset of the K-blocks (sub-group sg handles bid = sg, sg+KSPLIT,
 *   ...).  Every sub-group keeps a private partial-sum register `acc`; at the
 *   very end the KSPLIT partials are reduced through a tiny SLM buffer with a
 *   SINGLE work-group barrier and sub-group 0 writes the fp16 output.
 *
 *   This multiplies the number of live hardware threads by KSPLIT
 *   (total = (N/OPG) * KSPLIT) — dramatically raising occupancy and memory
 *   level parallelism for small N — while staying a single kernel / single
 *   event (so the memory-bound work is fully overlapped and the benchmark
 *   timing remains a single dispatch).
 *
 * == Other memory optimizations (unchanged) ==
 *   * Weights read with wide intel_sub_group_block_read8 (512 B/call), all
 *     long-latency global loads issued BEFORE the SLM sync so they overlap the
 *     barrier and the scalar decode/reduction work.
 *   * Per-block quantized scale/min factored OUT of the inner loop:
 *         acc += scale_j*dot_j - min_j*isum_j
 *     halving the FMA count.
 *   * Per-sub-group activation SLM is DOUBLE-BUFFERED so exactly ONE
 *     sub_group_barrier per K-block is needed (write->read visibility), and
 *     sub-groups never wait on each other inside the K-loop.
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
 * `c` is the chunk index (0..3). */
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
 * gemv_q4k_sg
 * ----------------------------------------------------------------------- */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void gemv_q4k_sg(
    __global const half   * restrict inputs,   /* [input_len]              */
    __global const uchar  * restrict pqs_T,    /* [total_blk * OPG * 128]  */
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
    const __global uint    *psl_pu = (const __global uint    *)psl_T;
    const __global ushort  *psl_us = (const __global ushort  *)psl_T;

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
        ACC_CHUNK(dot, w0.s0, 0, in); ACC_CHUNK(dot, w0.s1, 1, in);
        ACC_CHUNK(dot, w0.s2, 2, in); ACC_CHUNK(dot, w0.s3, 3, in);
        acc = fma(scale[0], dot, acc); acc = fma(-minv[0], isum[0], acc);

        in = sb + 1*32; dot = 0.0f;
        ACC_CHUNK(dot, w0.s4, 0, in); ACC_CHUNK(dot, w0.s5, 1, in);
        ACC_CHUNK(dot, w0.s6, 2, in); ACC_CHUNK(dot, w0.s7, 3, in);
        acc = fma(scale[1], dot, acc); acc = fma(-minv[1], isum[1], acc);

        in = sb + 2*32; dot = 0.0f;
        ACC_CHUNK(dot, w1.s0, 0, in); ACC_CHUNK(dot, w1.s1, 1, in);
        ACC_CHUNK(dot, w1.s2, 2, in); ACC_CHUNK(dot, w1.s3, 3, in);
        acc = fma(scale[2], dot, acc); acc = fma(-minv[2], isum[2], acc);

        in = sb + 3*32; dot = 0.0f;
        ACC_CHUNK(dot, w1.s4, 0, in); ACC_CHUNK(dot, w1.s5, 1, in);
        ACC_CHUNK(dot, w1.s6, 2, in); ACC_CHUNK(dot, w1.s7, 3, in);
        acc = fma(scale[3], dot, acc); acc = fma(-minv[3], isum[3], acc);

        in = sb + 4*32; dot = 0.0f;
        ACC_CHUNK(dot, w2.s0, 0, in); ACC_CHUNK(dot, w2.s1, 1, in);
        ACC_CHUNK(dot, w2.s2, 2, in); ACC_CHUNK(dot, w2.s3, 3, in);
        acc = fma(scale[4], dot, acc); acc = fma(-minv[4], isum[4], acc);

        in = sb + 5*32; dot = 0.0f;
        ACC_CHUNK(dot, w2.s4, 0, in); ACC_CHUNK(dot, w2.s5, 1, in);
        ACC_CHUNK(dot, w2.s6, 2, in); ACC_CHUNK(dot, w2.s7, 3, in);
        acc = fma(scale[5], dot, acc); acc = fma(-minv[5], isum[5], acc);

        in = sb + 6*32; dot = 0.0f;
        ACC_CHUNK(dot, w3.s0, 0, in); ACC_CHUNK(dot, w3.s1, 1, in);
        ACC_CHUNK(dot, w3.s2, 2, in); ACC_CHUNK(dot, w3.s3, 3, in);
        acc = fma(scale[6], dot, acc); acc = fma(-minv[6], isum[6], acc);

        in = sb + 7*32; dot = 0.0f;
        ACC_CHUNK(dot, w3.s4, 0, in); ACC_CHUNK(dot, w3.s5, 1, in);
        ACC_CHUNK(dot, w3.s6, 2, in); ACC_CHUNK(dot, w3.s7, 3, in);
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
