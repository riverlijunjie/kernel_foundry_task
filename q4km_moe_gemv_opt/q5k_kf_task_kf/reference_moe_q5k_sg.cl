/*
 * reference_moe_q5k_sg.cl
 *
 * Baseline (reference) implementation of the Q5_K MoE "SG" (sub-group
 * block-read, transposed weight layout) GEMV kernels:
 *   group_up_gate_q5k_sg — fused up-proj + gate-proj GEMV + SiLU(gate)*up
 *   down_merge_q5k_sg    — down-proj GEMV, weighted-sum merged over experts
 *
 * These kernels are used as the correctness oracle and performance baseline.
 * Do NOT modify this file.
 *
 * == Transposed ("SG") per-expert layout — SAME total size as the plain
 *    shuffled Q5_K layout ==
 * For an expert weight matrix of [output_len(N), input_len(K)] with
 * nbpr = K/256 K-blocks per row and nrg = N/OPG row-groups:
 *
 *   pqs_T:  num_blocks * 128 bytes  (low nibbles, identical layout to Q4_K)
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 128 bytes (2048 B)
 *     Sub-block j at j*OPG*16, chunk c (0..3) at j*OPG*16 + c*OPG*4.
 *   pqh_T:  num_blocks * 32 bytes   (1 high bit / weight)
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 32 bytes (512 B):
 *       sub-block j's 4-byte qh word for all OPG lanes at j*OPG*4.
 *   psl_T:  num_blocks * 16 bytes  (SoA, identical to Q4_K psl_T)
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes (256 B):
 *       [ sl_u32×OPG=64B | ml_u32×OPG=64B | sh_u16×OPG=32B |
 *         mh_u16×OPG=32B | d_u16×OPG=32B  | dmin_u16×OPG=32B ]
 *   expert_size = num_blocks * 176  (128 qs + 32 qh + 16 sl) — unchanged.
 *
 * == Dispatch ==
 *   group_up_gate_q5k_sg:
 *     global = (output_len, KSPLIT, output_num*token_len)
 *     local  = (OPG=16,    KSPLIT, 1)
 *   down_merge_q5k_sg:
 *     global = (output_len, KSPLIT, token_len)
 *     local  = (OPG=16,    KSPLIT, 1)
 */

/* [REFERENCE_START] */

#pragma OPENCL EXTENSION cl_khr_fp16        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define OPG 16  /* = sub_group_size */

#ifndef KSPLIT
#define KSPLIT 1
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

/* Accumulate one 4-byte qs chunk word `u` + its matching 4-byte qh word `H`
 * (8 quantized 5-bit weights: 4 low + 4 high nibbles, each with 1 extra bit
 * from H) into the sub-block dot product.  `in` points at the sub-block SLM
 * base.
 *
 *     weight (   4c+r) = (u >> (8r    )) & 0xF | ((H >> (r*8 + c    )) & 1) << 4
 *     weight (16+4c+r) = (u >> (8r + 4)) & 0xF | ((H >> (r*8 + c + 4)) & 1) << 4
 */
#define ACC_CHUNK5(dot, u, H, c, in)                                                    \
    do {                                                                                \
        uint _w = (u); uint _h = (H);                                                   \
        (dot) = fma((in)[(c)*4 + 0],                                                    \
                    (float)(( _w        & 0xFu) | (((_h >> (0*8 + (c)    )) & 1u) << 4)), (dot)); \
        (dot) = fma((in)[(c)*4 + 1],                                                    \
                    (float)(((_w >>  8) & 0xFu) | (((_h >> (1*8 + (c)    )) & 1u) << 4)), (dot)); \
        (dot) = fma((in)[(c)*4 + 2],                                                    \
                    (float)(((_w >> 16) & 0xFu) | (((_h >> (2*8 + (c)    )) & 1u) << 4)), (dot)); \
        (dot) = fma((in)[(c)*4 + 3],                                                    \
                    (float)(((_w >> 24) & 0xFu) | (((_h >> (3*8 + (c)    )) & 1u) << 4)), (dot)); \
        (dot) = fma((in)[16 + (c)*4 + 0],                                               \
                    (float)(((_w >>  4) & 0xFu) | (((_h >> (0*8 + (c) + 4)) & 1u) << 4)), (dot)); \
        (dot) = fma((in)[16 + (c)*4 + 1],                                               \
                    (float)(((_w >> 12) & 0xFu) | (((_h >> (1*8 + (c) + 4)) & 1u) << 4)), (dot)); \
        (dot) = fma((in)[16 + (c)*4 + 2],                                               \
                    (float)(((_w >> 20) & 0xFu) | (((_h >> (2*8 + (c) + 4)) & 1u) << 4)), (dot)); \
        (dot) = fma((in)[16 + (c)*4 + 3],                                               \
                    (float)(((_w >> 28) & 0xFu) | (((_h >> (3*8 + (c) + 4)) & 1u) << 4)), (dot)); \
    } while (0)

/* Decode the 8 (scale, min) pairs from the SoA psl record fields (identical
 * to the Q4_K psl record — Q5_K reuses exactly the same 16-byte layout). */
inline void q5k_decode(uint sl, uint ml, ushort sh, ushort mh,
                       ushort d_raw, ushort dn_raw,
                       float *scale, float *minv)
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

/* Whole-block dot product from block-read weight+qh words + SLM activations. */
inline float q5k_acc(uint8 w0, uint8 w1, uint8 w2, uint8 w3, uint8 hq,
                     const float *scale, const float *minv, const float *isum,
                     __local const float *sb)
{
    float acc = 0.0f, dot;
    __local const float *in;

    in = sb + 0*32; dot = 0.0f;
    ACC_CHUNK5(dot, w0.s0, hq.s0, 0, in); ACC_CHUNK5(dot, w0.s1, hq.s0, 1, in);
    ACC_CHUNK5(dot, w0.s2, hq.s0, 2, in); ACC_CHUNK5(dot, w0.s3, hq.s0, 3, in);
    acc = fma(scale[0], dot, acc); acc = fma(-minv[0], isum[0], acc);

    in = sb + 1*32; dot = 0.0f;
    ACC_CHUNK5(dot, w0.s4, hq.s1, 0, in); ACC_CHUNK5(dot, w0.s5, hq.s1, 1, in);
    ACC_CHUNK5(dot, w0.s6, hq.s1, 2, in); ACC_CHUNK5(dot, w0.s7, hq.s1, 3, in);
    acc = fma(scale[1], dot, acc); acc = fma(-minv[1], isum[1], acc);

    in = sb + 2*32; dot = 0.0f;
    ACC_CHUNK5(dot, w1.s0, hq.s2, 0, in); ACC_CHUNK5(dot, w1.s1, hq.s2, 1, in);
    ACC_CHUNK5(dot, w1.s2, hq.s2, 2, in); ACC_CHUNK5(dot, w1.s3, hq.s2, 3, in);
    acc = fma(scale[2], dot, acc); acc = fma(-minv[2], isum[2], acc);

    in = sb + 3*32; dot = 0.0f;
    ACC_CHUNK5(dot, w1.s4, hq.s3, 0, in); ACC_CHUNK5(dot, w1.s5, hq.s3, 1, in);
    ACC_CHUNK5(dot, w1.s6, hq.s3, 2, in); ACC_CHUNK5(dot, w1.s7, hq.s3, 3, in);
    acc = fma(scale[3], dot, acc); acc = fma(-minv[3], isum[3], acc);

    in = sb + 4*32; dot = 0.0f;
    ACC_CHUNK5(dot, w2.s0, hq.s4, 0, in); ACC_CHUNK5(dot, w2.s1, hq.s4, 1, in);
    ACC_CHUNK5(dot, w2.s2, hq.s4, 2, in); ACC_CHUNK5(dot, w2.s3, hq.s4, 3, in);
    acc = fma(scale[4], dot, acc); acc = fma(-minv[4], isum[4], acc);

    in = sb + 5*32; dot = 0.0f;
    ACC_CHUNK5(dot, w2.s4, hq.s5, 0, in); ACC_CHUNK5(dot, w2.s5, hq.s5, 1, in);
    ACC_CHUNK5(dot, w2.s6, hq.s5, 2, in); ACC_CHUNK5(dot, w2.s7, hq.s5, 3, in);
    acc = fma(scale[5], dot, acc); acc = fma(-minv[5], isum[5], acc);

    in = sb + 6*32; dot = 0.0f;
    ACC_CHUNK5(dot, w3.s0, hq.s6, 0, in); ACC_CHUNK5(dot, w3.s1, hq.s6, 1, in);
    ACC_CHUNK5(dot, w3.s2, hq.s6, 2, in); ACC_CHUNK5(dot, w3.s3, hq.s6, 3, in);
    acc = fma(scale[6], dot, acc); acc = fma(-minv[6], isum[6], acc);

    in = sb + 7*32; dot = 0.0f;
    ACC_CHUNK5(dot, w3.s4, hq.s7, 0, in); ACC_CHUNK5(dot, w3.s5, hq.s7, 1, in);
    ACC_CHUNK5(dot, w3.s6, hq.s7, 2, in); ACC_CHUNK5(dot, w3.s7, hq.s7, 3, in);
    acc = fma(scale[7], dot, acc); acc = fma(-minv[7], isum[7], acc);
    return acc;
}

/* Load 256 half activations into SLM (sb[n] = (float)activation_half[n]) via
 * 2 sub-group block reads of 8 ushorts each (intel_sub_group_block_read_us8).
 * Each us8 read pulls 128 contiguous fp16 values (256 B) with all 16 lanes
 * participating, matching the wide-transaction strategy used for the weights. */
inline void load_inp_f16(__local float *sb, const __global half *src, uint hh)
{
    const __global ushort *u = (const __global ushort *)src;
    ushort8 i0 = intel_sub_group_block_read_us8(u +   0);
    ushort8 i1 = intel_sub_group_block_read_us8(u + 128);
    sb[ 0*OPG + hh] = (float)as_half(i0.s0); sb[ 1*OPG + hh] = (float)as_half(i0.s1);
    sb[ 2*OPG + hh] = (float)as_half(i0.s2); sb[ 3*OPG + hh] = (float)as_half(i0.s3);
    sb[ 4*OPG + hh] = (float)as_half(i0.s4); sb[ 5*OPG + hh] = (float)as_half(i0.s5);
    sb[ 6*OPG + hh] = (float)as_half(i0.s6); sb[ 7*OPG + hh] = (float)as_half(i0.s7);
    sb[ 8*OPG + hh] = (float)as_half(i1.s0); sb[ 9*OPG + hh] = (float)as_half(i1.s1);
    sb[10*OPG + hh] = (float)as_half(i1.s2); sb[11*OPG + hh] = (float)as_half(i1.s3);
    sb[12*OPG + hh] = (float)as_half(i1.s4); sb[13*OPG + hh] = (float)as_half(i1.s5);
    sb[14*OPG + hh] = (float)as_half(i1.s6); sb[15*OPG + hh] = (float)as_half(i1.s7);
}

/* =======================================================================
 * Kernel 1: group_up_gate_q5k_sg
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void group_up_gate_q5k_sg(
    __global const half   * restrict inputs,   /* [token_len * input_len]      */
    __global const uchar  * restrict ups,      /* [num_experts * expert_size]  */
    __global const uchar  * restrict gates,    /* [num_experts * expert_size]  */
    __global const uint   * restrict indexes,  /* [token_len*aligned_output_num] or NULL */
    __global       float  * restrict outputs,  /* [token_len*output_num*output_len] */
    uint output_num,
    uint aligned_output_num,
    uint token_len,
    uint input_len,
    uint output_len
) {
    const uint nbpr       = input_len / 256;
    const uint num_blocks = input_len * output_len / 256;
    const uint expert_size = num_blocks * 176;  /* 128 qs + 32 qh + 16 sl */

    const uint h  = get_group_id(0);   /* row-group over output_len/OPG */
    const uint hh = get_local_id(0);   /* lane = output row within group */
    const uint sg = get_local_id(1);   /* K-split sub-group index         */
    const uint vt = get_group_id(2);
    const uint v  = vt % output_num;
    const uint t  = vt / output_num;

    const uint index = indexes ? indexes[t * aligned_output_num + v] : v;

    __local float slm_inp [KSPLIT * 2 * 256];
    __local float slm_up  [KSPLIT * OPG];
    __local float slm_gate[KSPLIT * OPG];
    __local float *sbase = slm_inp + sg * (2 * 256);

    /* Expert weight section bases. */
    const __global uchar *up_pqs  = ups   + (size_t)index * expert_size;
    const __global uchar *up_pqh  = up_pqs + (size_t)num_blocks * 128;
    const __global uchar *up_psl  = up_pqs + (size_t)num_blocks * 160;
    const __global uchar *gt_pqs  = gates + (size_t)index * expert_size;
    const __global uchar *gt_pqh  = gt_pqs + (size_t)num_blocks * 128;
    const __global uchar *gt_psl  = gt_pqs + (size_t)num_blocks * 160;

    const __global uint   *up_pqs_u = (const __global uint   *)up_pqs;
    const __global uint   *up_pqh_u = (const __global uint   *)up_pqh;
    const __global uint   *up_psl_pu= (const __global uint   *)up_psl;
    const __global ushort *up_psl_us= (const __global ushort *)up_psl;
    const __global uint   *gt_pqs_u = (const __global uint   *)gt_pqs;
    const __global uint   *gt_pqh_u = (const __global uint   *)gt_pqh;
    const __global uint   *gt_psl_pu= (const __global uint   *)gt_psl;
    const __global ushort *gt_psl_us= (const __global ushort *)gt_psl;

    float up_acc   = 0.0f;
    float gate_acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float *sb = sbase + (bid & 1u) * 256;

        const uint pe_u        = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e       = (h * nbpr + bid) * (OPG *  32 / 4);
        const uint psl_pu_off  = (h * nbpr + bid) * (OPG * 16 / 4);
        const uint psl_us_off  = (h * nbpr + bid) * (OPG * 16 / 2);

        /* ---- issue all long-latency global loads first ---- */
        uint8 uw0 = intel_sub_group_block_read8(up_pqs_u + pe_u +   0);
        uint8 uw1 = intel_sub_group_block_read8(up_pqs_u + pe_u + 128);
        uint8 uw2 = intel_sub_group_block_read8(up_pqs_u + pe_u + 256);
        uint8 uw3 = intel_sub_group_block_read8(up_pqs_u + pe_u + 384);
        uint8 uhq = intel_sub_group_block_read8(up_pqh_u + pqh_e);
        uint   usl  = intel_sub_group_block_read   (up_psl_pu + psl_pu_off +  0);
        uint   uml  = intel_sub_group_block_read   (up_psl_pu + psl_pu_off + 16);
        ushort ush  = intel_sub_group_block_read_us(up_psl_us + psl_us_off + 64);
        ushort umh  = intel_sub_group_block_read_us(up_psl_us + psl_us_off + 80);
        ushort ud   = intel_sub_group_block_read_us(up_psl_us + psl_us_off + 96);
        ushort udn  = intel_sub_group_block_read_us(up_psl_us + psl_us_off + 112);

        uint8 gw0 = intel_sub_group_block_read8(gt_pqs_u + pe_u +   0);
        uint8 gw1 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 128);
        uint8 gw2 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 256);
        uint8 gw3 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 384);
        uint8 ghq = intel_sub_group_block_read8(gt_pqh_u + pqh_e);
        uint   gsl  = intel_sub_group_block_read   (gt_psl_pu + psl_pu_off +  0);
        uint   gml  = intel_sub_group_block_read   (gt_psl_pu + psl_pu_off + 16);
        ushort gsh  = intel_sub_group_block_read_us(gt_psl_us + psl_us_off + 64);
        ushort gmh  = intel_sub_group_block_read_us(gt_psl_us + psl_us_off + 80);
        ushort gd   = intel_sub_group_block_read_us(gt_psl_us + psl_us_off + 96);
        ushort gdn  = intel_sub_group_block_read_us(gt_psl_us + psl_us_off + 112);

        load_inp_f16(sb, inputs + (size_t)t * input_len + (size_t)bid * 256, hh);

        float uscale[8], uminv[8], gscale[8], gminv[8];
        q5k_decode(usl, uml, ush, umh, ud, udn, uscale, uminv);
        q5k_decode(gsl, gml, gsh, gmh, gd, gdn, gscale, gminv);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        float isum[8];
        unroll_for (int j = 0; j < 8; j++) {
            float s = 0.0f;
            unroll_for (int i = 0; i < 32; i++) s += sb[j * 32 + i];
            isum[j] = s;
        }

        up_acc   += q5k_acc(uw0, uw1, uw2, uw3, uhq, uscale, uminv, isum, sb);
        gate_acc += q5k_acc(gw0, gw1, gw2, gw3, ghq, gscale, gminv, isum, sb);
    }

#if KSPLIT == 1
    {
        float g   = gate_acc;
        float sig = 1.0f / (1.0f + exp(-g));
        outputs[(size_t)t * output_num * output_len + (size_t)v * output_len
                + h * OPG + hh] = (sig * g) * up_acc;
    }
#else
    slm_up  [sg * OPG + hh] = up_acc;
    slm_gate[sg * OPG + hh] = gate_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float u = 0.0f, gt = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) {
            u  += slm_up  [r * OPG + hh];
            gt += slm_gate[r * OPG + hh];
        }
        float sig = 1.0f / (1.0f + exp(-gt));
        outputs[(size_t)t * output_num * output_len + (size_t)v * output_len
                + h * OPG + hh] = (sig * gt) * u;
    }
#endif
}

/* =======================================================================
 * Kernel 2: down_merge_q5k_sg
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void down_merge_q5k_sg(
    __global const half   * restrict inputs,     /* [token_len*input_num*input_len] */
    __global const uchar  * restrict downs,      /* [num_experts * expert_size]     */
    __global const uint   * restrict indexes,    /* [token_len*aligned_input_num] or NULL */
    __global const float  * restrict efficients, /* [token_len*aligned_input_num]   */
    __global       float  * restrict outputs,    /* [token_len*output_len]          */
    uint input_num,
    uint aligned_input_num,
    uint token_len,
    uint input_len,
    uint output_len,
    int  is_acc
) {
    const uint nbpr       = input_len / 256;
    const uint num_blocks = input_len * output_len / 256;
    const uint expert_size = num_blocks * 176;

    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint t  = get_group_id(2);

    __local float slm_inp[KSPLIT * 2 * 256];
    __local float slm_red[KSPLIT * OPG];
    __local float *sbase = slm_inp + sg * (2 * 256);

    const uint total = input_num * nbpr;
    float acc = 0.0f;

    for (uint idx = sg; idx < total; idx += KSPLIT) {
        const uint e   = idx / nbpr;
        const uint bid = idx % nbpr;
        __local float *sb = sbase + (idx & 1u) * 256;

        const uint index     = indexes ? indexes[t * aligned_input_num + e] : e;
        const float efficient = efficients[t * aligned_input_num + e];

        const __global uchar *d_pqs = downs + (size_t)index * expert_size;
        const __global uchar *d_pqh = d_pqs + (size_t)num_blocks * 128;
        const __global uchar *d_psl = d_pqs + (size_t)num_blocks * 160;
        const __global uint   *d_pqs_u = (const __global uint   *)d_pqs;
        const __global uint   *d_pqh_u = (const __global uint   *)d_pqh;
        const __global uint   *d_psl_pu= (const __global uint   *)d_psl;
        const __global ushort *d_psl_us= (const __global ushort *)d_psl;

        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e      = (h * nbpr + bid) * (OPG *  32 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);
        const uint psl_us_off = (h * nbpr + bid) * (OPG * 16 / 2);

        uint8 w0 = intel_sub_group_block_read8(d_pqs_u + pe_u +   0);
        uint8 w1 = intel_sub_group_block_read8(d_pqs_u + pe_u + 128);
        uint8 w2 = intel_sub_group_block_read8(d_pqs_u + pe_u + 256);
        uint8 w3 = intel_sub_group_block_read8(d_pqs_u + pe_u + 384);
        uint8 hq = intel_sub_group_block_read8(d_pqh_u + pqh_e);
        uint   sl  = intel_sub_group_block_read   (d_psl_pu + psl_pu_off +  0);
        uint   ml  = intel_sub_group_block_read   (d_psl_pu + psl_pu_off + 16);
        ushort sh  = intel_sub_group_block_read_us(d_psl_us + psl_us_off + 64);
        ushort mh  = intel_sub_group_block_read_us(d_psl_us + psl_us_off + 80);
        ushort dd  = intel_sub_group_block_read_us(d_psl_us + psl_us_off + 96);
        ushort ddn = intel_sub_group_block_read_us(d_psl_us + psl_us_off + 112);

        load_inp_f16(sb, inputs + (size_t)t * input_num * input_len
                     + (size_t)e * input_len + (size_t)bid * 256, hh);

        float scale[8], minv[8];
        q5k_decode(sl, ml, sh, mh, dd, ddn, scale, minv);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        float isum[8];
        unroll_for (int j = 0; j < 8; j++) {
            float s = 0.0f;
            unroll_for (int i = 0; i < 32; i++) s += sb[j * 32 + i];
            isum[j] = s;
        }

        acc += q5k_acc(w0, w1, w2, w3, hq, scale, minv, isum, sb) * efficient;
    }

#if KSPLIT == 1
    {
        uint row = h * OPG + hh;
        float o = acc;
        if (is_acc) o += outputs[(size_t)t * output_len + row];
        outputs[(size_t)t * output_len + row] = o;
    }
#else
    slm_red[sg * OPG + hh] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float o = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) o += slm_red[r * OPG + hh];
        uint row = h * OPG + hh;
        if (is_acc) o += outputs[(size_t)t * output_len + row];
        outputs[(size_t)t * output_len + row] = o;
    }
#endif
}

/* [REFERENCE_END] */
