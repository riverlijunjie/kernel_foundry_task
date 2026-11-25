/*
 * gemv_q6k_sg.cl  — High-performance Q6_K GEMV with Intel sub-group block reads.
 *
 * == Transposed ("SG") weight layout — SAME total size as original ==
 *
 *   pql_T:  total_blk * 128 bytes
 *     Entry (h, bid) at (h*nbpr+bid) * OPG * 128 bytes (2048 B)
 *     Sub-block j at j*OPG*16, chunk c (0..3) at j*OPG*16 + c*OPG*4.
 *
 *   pqh_T:  total_blk * 64 bytes
 *     Entry (h, bid) at (h*nbpr+bid) * OPG * 64 bytes (1024 B)
 *     Sub-block j at j*OPG*8, chunk c (0..1) at j*OPG*8 + c*OPG*4.
 *
 *   ps_T:   total_blk * 16 bytes
 *     Entry (h, bid) at (h*nbpr+bid) * OPG * 16 bytes (256 B)
 *     Scale si (0..15) at si*OPG, 1 int8 byte per lane.
 *
 *   pd_T:   total_blk * 2 bytes
 *     Entry (h, bid) at (h*nbpr+bid) * OPG * 2 bytes (32 B)
 *     1 ushort (fp16) per lane.
 *
 * == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
 *   The original dispatch used ONE sub-group per work-group, so the number of
 *   concurrent hardware threads == N/OPG.  For small N this leaves the machine
 *   badly under-occupied (e.g. N=1024 -> 64 threads on a 160-XVE GPU), which
 *   dominates this memory-bound kernel — too few in-flight loads to fill DRAM.
 *
 *   Fix: put KSPLIT sub-groups in each work-group (local = OPG x KSPLIT).  All
 *   KSPLIT sub-groups cover the SAME OPG output rows but each owns a strided
 *   subset of the K-blocks (sub-group sg handles bid = sg, sg+KSPLIT, ...).
 *   Each keeps a private partial `acc`; a single work-group barrier at the end
 *   reduces the KSPLIT partials through a small SLM buffer and sub-group 0
 *   writes the fp16 output.  Total live threads = (N/OPG) * KSPLIT.
 *
 * == Other memory optimizations (unchanged) ==
 *   * Wide intel_sub_group_block_read8 weight loads; all long-latency global
 *     loads issued BEFORE the SLM sync so they overlap decode work.
 *   * Q6_K reconstruction w = qval - 32 and per-sub-block scale factored out:
 *       acc += scale_j * (dot_j - 32*isum_j).
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

/* Accumulate one ql uint (4 positions, low+high nibble weights) into the two
 * sub-block dot products and activation sums. */
#define Q6_CHUNK(qlw, in, k0, s0, s1, s2, s3, qh_lo, qh_hi,                     \
                 dot_lo, dot_hi, isum_lo, isum_hi)                             \
    do {                                                                      \
        uint _q = (qlw);                                                      \
        uint _b0 =  _q        & 0xFFu;                                        \
        uint _b1 = (_q >>  8) & 0xFFu;                                        \
        uint _b2 = (_q >> 16) & 0xFFu;                                        \
        uint _b3 = (_q >> 24) & 0xFFu;                                        \
        float _l0 = (in)[(k0) + 0];  float _h0 = (in)[16 + (k0) + 0];         \
        float _l1 = (in)[(k0) + 1];  float _h1 = (in)[16 + (k0) + 1];         \
        float _l2 = (in)[(k0) + 2];  float _h2 = (in)[16 + (k0) + 2];         \
        float _l3 = (in)[(k0) + 3];  float _h3 = (in)[16 + (k0) + 3];         \
        (dot_lo) = fma(_l0, (float)((_b0 & 0xFu) | ((((qh_lo) >> (s0)) & 3u) << 4)), (dot_lo)); \
        (dot_lo) = fma(_l1, (float)((_b1 & 0xFu) | ((((qh_lo) >> (s1)) & 3u) << 4)), (dot_lo)); \
        (dot_lo) = fma(_l2, (float)((_b2 & 0xFu) | ((((qh_lo) >> (s2)) & 3u) << 4)), (dot_lo)); \
        (dot_lo) = fma(_l3, (float)((_b3 & 0xFu) | ((((qh_lo) >> (s3)) & 3u) << 4)), (dot_lo)); \
        (dot_hi) = fma(_h0, (float)(((_b0 >> 4) & 0xFu) | ((((qh_hi) >> (s0)) & 3u) << 4)), (dot_hi)); \
        (dot_hi) = fma(_h1, (float)(((_b1 >> 4) & 0xFu) | ((((qh_hi) >> (s1)) & 3u) << 4)), (dot_hi)); \
        (dot_hi) = fma(_h2, (float)(((_b2 >> 4) & 0xFu) | ((((qh_hi) >> (s2)) & 3u) << 4)), (dot_hi)); \
        (dot_hi) = fma(_h3, (float)(((_b3 >> 4) & 0xFu) | ((((qh_hi) >> (s3)) & 3u) << 4)), (dot_hi)); \
        (isum_lo) += _l0 + _l1 + _l2 + _l3;                                   \
        (isum_hi) += _h0 + _h1 + _h2 + _h3;                                   \
    } while (0)

/* Process one full sub-block j (32 weights) given its 4 ql words + 2 qh words. */
#define Q6_SUBBLOCK(in, ql0, ql1, ql2, ql3, qh_lo, qh_hi, clo, chi, acc)      \
    do {                                                                      \
        float _dl = 0.0f, _dh = 0.0f, _sl = 0.0f, _sh_ = 0.0f;                \
        Q6_CHUNK(ql0, in,  0,  0,  8, 16, 24, qh_lo, qh_hi, _dl, _dh, _sl, _sh_); \
        Q6_CHUNK(ql1, in,  4,  2, 10, 18, 26, qh_lo, qh_hi, _dl, _dh, _sl, _sh_); \
        Q6_CHUNK(ql2, in,  8,  4, 12, 20, 28, qh_lo, qh_hi, _dl, _dh, _sl, _sh_); \
        Q6_CHUNK(ql3, in, 12,  6, 14, 22, 30, qh_lo, qh_hi, _dl, _dh, _sl, _sh_); \
        (acc) = fma((clo), _dl - 32.0f * _sl, (acc));                         \
        (acc) = fma((chi), _dh - 32.0f * _sh_, (acc));                        \
    } while (0)

__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void gemv_q6k_sg(
    __global const half   * restrict inputs,   /* [input_len]              */
    __global const uchar  * restrict pql_T,    /* [total_blk * OPG * 128]  */
    __global const uchar  * restrict pqh_T,    /* [total_blk * OPG * 64]   */
    __global const char   * restrict ps_T,     /* [total_blk * OPG * 16]   */
    __global const uchar  * restrict pd_T,     /* [total_blk * OPG * 2]    */
    __global       half   * restrict outputs,  /* [output_len]             */
    uint input_len,
    uint output_len
) {
    const uint nbpr = input_len / 256;

    const uint h  = get_group_id(0);    /* row-group */
    const uint hh = get_local_id(0);    /* lane = output row within group */
    const uint sg = get_local_id(1);    /* sub-group / K-split index       */

    /* Per-sub-group activation SLM (double-buffered) + reduction scratch. */
    __local float slm_inp[KSPLIT * 2 * 256];
    __local float slm_red[KSPLIT * OPG];
    __local float *sbase = slm_inp + sg * (2 * 256);

    const __global uint *pql_u = (const __global uint *)pql_T;
    const __global uint *pqh_u = (const __global uint *)pqh_T;

    float acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {

        __local float *sb = sbase + (bid & 1u) * 256;   /* current buffer */

        /* =============================================================== *
         * Issue all long-latency global loads first (weights + scales).   *
         * =============================================================== */
        uint pql_e = (h * nbpr + bid) * (OPG * 128 / 4);   /* uints */
        uint8 w0 = intel_sub_group_block_read8(pql_u + pql_e +   0);  /* sub 0,1 */
        uint8 w1 = intel_sub_group_block_read8(pql_u + pql_e + 128);  /* sub 2,3 */
        uint8 w2 = intel_sub_group_block_read8(pql_u + pql_e + 256);  /* sub 4,5 */
        uint8 w3 = intel_sub_group_block_read8(pql_u + pql_e + 384);  /* sub 6,7 */

        uint pqh_e = (h * nbpr + bid) * (OPG * 64 / 4);    /* uints */
        uint8 hq0 = intel_sub_group_block_read8(pqh_u + pqh_e +   0);  /* sub 0..3 */
        uint8 hq1 = intel_sub_group_block_read8(pqh_u + pqh_e + 128);  /* sub 4..7 */

        uint ps_e = (h * nbpr + bid) * (OPG * 16);         /* bytes */
        uchar16 scq = intel_sub_group_block_read_uc16(
            (const __global uchar *)(ps_T + ps_e));

        uint pd_e = (h * nbpr + bid) * (OPG * 2);          /* bytes */
        ushort d_raw = intel_sub_group_block_read_us(
            (const __global ushort *)(pd_T + pd_e));

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

        /* ---- Decode d and the 8 (clo, chi) scale pairs (overlaps loads) ---- */
        float d = (float)as_half(d_raw);
        char sc[16];
        sc[ 0]=as_char(scq.s0); sc[ 1]=as_char(scq.s1); sc[ 2]=as_char(scq.s2); sc[ 3]=as_char(scq.s3);
        sc[ 4]=as_char(scq.s4); sc[ 5]=as_char(scq.s5); sc[ 6]=as_char(scq.s6); sc[ 7]=as_char(scq.s7);
        sc[ 8]=as_char(scq.s8); sc[ 9]=as_char(scq.s9); sc[10]=as_char(scq.sa); sc[11]=as_char(scq.sb);
        sc[12]=as_char(scq.sc); sc[13]=as_char(scq.sd); sc[14]=as_char(scq.se); sc[15]=as_char(scq.sf);

        float clo[8], chi[8];
        unroll_for (int j = 0; j < 8; j++) {
            clo[j] = d * (float)sc[2*j];
            chi[j] = d * (float)sc[2*j + 1];
        }

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        /* ---- 8 sub-blocks: dot products + factored scale/bias combine ---- */
        __local const float *in;

        in = sb + 0*32; Q6_SUBBLOCK(in, w0.s0, w0.s1, w0.s2, w0.s3, hq0.s0, hq0.s1, clo[0], chi[0], acc);
        in = sb + 1*32; Q6_SUBBLOCK(in, w0.s4, w0.s5, w0.s6, w0.s7, hq0.s2, hq0.s3, clo[1], chi[1], acc);
        in = sb + 2*32; Q6_SUBBLOCK(in, w1.s0, w1.s1, w1.s2, w1.s3, hq0.s4, hq0.s5, clo[2], chi[2], acc);
        in = sb + 3*32; Q6_SUBBLOCK(in, w1.s4, w1.s5, w1.s6, w1.s7, hq0.s6, hq0.s7, clo[3], chi[3], acc);
        in = sb + 4*32; Q6_SUBBLOCK(in, w2.s0, w2.s1, w2.s2, w2.s3, hq1.s0, hq1.s1, clo[4], chi[4], acc);
        in = sb + 5*32; Q6_SUBBLOCK(in, w2.s4, w2.s5, w2.s6, w2.s7, hq1.s2, hq1.s3, clo[5], chi[5], acc);
        in = sb + 6*32; Q6_SUBBLOCK(in, w3.s0, w3.s1, w3.s2, w3.s3, hq1.s4, hq1.s5, clo[6], chi[6], acc);
        in = sb + 7*32; Q6_SUBBLOCK(in, w3.s4, w3.s5, w3.s6, w3.s7, hq1.s6, hq1.s7, clo[7], chi[7], acc);
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
