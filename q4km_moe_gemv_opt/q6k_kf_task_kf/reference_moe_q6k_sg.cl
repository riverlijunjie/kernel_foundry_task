/*
 * reference_moe_q6k_sg.cl
 *
 * Baseline (reference) implementation of the Q6_K MoE "SG" (sub-group
 * block-read, transposed weight layout) GEMV kernels:
 *   group_up_gate_q6k_sg — fused up-proj + gate-proj GEMV + SiLU(gate)*up
 *   down_merge_q6k_sg    — down-proj GEMV, weighted-sum merged over experts
 *
 * These kernels are used as the correctness oracle and performance baseline.
 * Do NOT modify this file.
 *
 * == Transposed ("SG") per-expert layout — SAME total size as the plain
 *    shuffled Q6_K layout ==
 * For an expert weight matrix of [output_len(N), input_len(K)] with
 * nbpr = K/256 K-blocks per row and nrg = N/OPG row-groups:
 *
 *   pql_T:  num_blocks * 128 bytes (low nibbles)
 *   pqh_T:  num_blocks * 64 bytes  (upper 2 bits; sub-block j at j*OPG*8,
 *           half c(0..1) at j*OPG*8 + c*OPG*4)
 *   ps_T:   num_blocks * 16 bytes  (int8 scales, si at si*OPG)
 *   pd_T:   num_blocks * 2  bytes  (fp16 super-scale, 1 ushort/lane)
 *   expert_size = num_blocks * 210  (128+64+16+2) — unchanged.
 *
 * == Dispatch ==
 *   group_up_gate_q6k_sg:
 *     global = (output_len, KSPLIT, output_num*token_len)
 *     local  = (OPG=16,    KSPLIT, 1)
 *   down_merge_q6k_sg:
 *     global = (output_len, KSPLIT, token_len)
 *     local  = (OPG=16,    KSPLIT, 1)
 *
 * This baseline mirrors the seed "SG" sub-group block-read kernel with
 * whole-window SLM activation pre-staging and the up/gate-split-across-
 * sub-groups (SPLIT_UG) reduction scheme (identical to the initial contents
 * of kernel_moe_q6k_sg.cl, carried over from
 * q4km_moe_gemv_opt/q6k/moe_gemv_q6k_sg.cl), and is kept fixed here so that
 * every evolved candidate in kernel_moe_q6k_sg.cl is checked (correctness)
 * and timed (speedup) against a stable, known-good implementation.
 */

/* [REFERENCE_START] */

#pragma OPENCL EXTENSION cl_khr_fp16        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define OPG 16

#ifndef KSPLIT
#define KSPLIT 1
#endif

#if KSPLIT >= 2 && (KSPLIT % 2) == 0
#define SPLIT_UG 1
#else
#define SPLIT_UG 0
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

/* Activations are staged in SLM as float; the dot product accumulates in fp32. */
typedef float slm_t;
#define SB_STORE(dst, f2) (*(__local float2 *)(dst) = (f2))
#define XLOAD(p)          (*(__local const float4 *)(p))

/* Activation SLM budget, in 256-element blocks (1 block = 1 KiB as float). */
#ifndef SLM_ACT_BLOCKS
#define SLM_ACT_BLOCKS 16
#endif

#define SLM_STREAM_FLOATS (KSPLIT * 256)
#if (SLM_ACT_BLOCKS * 256) > SLM_STREAM_FLOATS
#define SLM_POOL_FLOATS (SLM_ACT_BLOCKS * 256)
#else
#define SLM_POOL_FLOATS SLM_STREAM_FLOATS
#endif

#define W2_MASK 0x000F000Fu   /* two 4-bit low parts, 16 bit apart        */
#define Q6_MASK 0x00300030u   /* their two 2-bit high parts, 16 bit apart */

/* One sub-block (32 weights) = 4 ql chunk words + 2 qh words (one per half) +
 * 32 SLM activations.  For chunk c and byte r:
 *     weight (   4c+r) = (q >> (8r  )) & 0xF | ((HL >> (r*8 + 2c)) & 3) << 4
 *     weight (16+4c+r) = (q >> (8r+4)) & 0xF | ((HH >> (r*8 + 2c)) & 3) << 4 */
#define Q6K_CHUNK(q, HL, HH, c)                                               \
    do {                                                                      \
        uint _q = (q);                                                        \
        uint _a = ( _q        & W2_MASK) | (((HL) << 4) >> (2 * (c)) & Q6_MASK);\
        uint _b = ((_q >>  8) & W2_MASK) | ((HL) >> (4 + 2 * (c))    & Q6_MASK);\
        uint _c = ((_q >>  4) & W2_MASK) | (((HH) << 4) >> (2 * (c)) & Q6_MASK);\
        uint _d = ((_q >> 12) & W2_MASK) | ((HH) >> (4 + 2 * (c))    & Q6_MASK);\
        float4 _wl = (float4)((float)as_ushort2(_a).x, (float)as_ushort2(_b).x,\
                              (float)(_a >> 16),       (float)(_b >> 16));     \
        float4 _wh = (float4)((float)as_ushort2(_c).x, (float)as_ushort2(_d).x,\
                              (float)(_c >> 16),       (float)(_d >> 16));     \
        vlo = fma(XLOAD(in + (c) * 4),      _wl, vlo);                        \
        vhi = fma(XLOAD(in + 16 + (c) * 4), _wh, vhi);                        \
    } while (0)

inline float2 q6k_subblock(uint q0, uint q1, uint q2, uint q3,
                           uint HL, uint HH, __local const slm_t *in)
{
    float4 vlo = (float4)(0.0f);
    float4 vhi = (float4)(0.0f);
    Q6K_CHUNK(q0, HL, HH, 0);
    Q6K_CHUNK(q1, HL, HH, 1);
    Q6K_CHUNK(q2, HL, HH, 2);
    Q6K_CHUNK(q3, HL, HH, 3);
    return (float2)((vlo.x + vlo.y) + (vlo.z + vlo.w),
                    (vhi.x + vhi.y) + (vhi.z + vhi.w));
}

#define Q6K_SUB(j, a, b, c, d, hl, hh)                                        \
    do {                                                                      \
        float2 _dv = q6k_subblock(a, b, c, d, hl, hh, sb + (j) * 32);         \
        acc = fma(clo[j], _dv.x, acc);                                        \
        acc = fma(chi[j], _dv.y, acc);                                        \
    } while (0)

inline float q6k_acc(uint8 w0, uint8 w1, uint8 w2, uint8 w3,
                     uint8 hq0, uint8 hq1,
                     const float *clo, const float *chi,
                     __local const slm_t *sb)
{
    float acc = 0.0f;
    Q6K_SUB(0, w0.s0, w0.s1, w0.s2, w0.s3, hq0.s0, hq0.s1);
    Q6K_SUB(1, w0.s4, w0.s5, w0.s6, w0.s7, hq0.s2, hq0.s3);
    Q6K_SUB(2, w1.s0, w1.s1, w1.s2, w1.s3, hq0.s4, hq0.s5);
    Q6K_SUB(3, w1.s4, w1.s5, w1.s6, w1.s7, hq0.s6, hq0.s7);
    Q6K_SUB(4, w2.s0, w2.s1, w2.s2, w2.s3, hq1.s0, hq1.s1);
    Q6K_SUB(5, w2.s4, w2.s5, w2.s6, w2.s7, hq1.s2, hq1.s3);
    Q6K_SUB(6, w3.s0, w3.s1, w3.s2, w3.s3, hq1.s4, hq1.s5);
    Q6K_SUB(7, w3.s4, w3.s5, w3.s6, w3.s7, hq1.s6, hq1.s7);
    return acc;
}

inline void q6k_decode(uchar16 scq, ushort d_raw, float *clo, float *chi)
{
    float d = (float)as_half(d_raw);
    char sc[16];
    sc[ 0]=as_char(scq.s0); sc[ 1]=as_char(scq.s1); sc[ 2]=as_char(scq.s2); sc[ 3]=as_char(scq.s3);
    sc[ 4]=as_char(scq.s4); sc[ 5]=as_char(scq.s5); sc[ 6]=as_char(scq.s6); sc[ 7]=as_char(scq.s7);
    sc[ 8]=as_char(scq.s8); sc[ 9]=as_char(scq.s9); sc[10]=as_char(scq.sa); sc[11]=as_char(scq.sb);
    sc[12]=as_char(scq.sc); sc[13]=as_char(scq.sd); sc[14]=as_char(scq.se); sc[15]=as_char(scq.sf);
    unroll_for (int j = 0; j < 8; j++) {
        clo[j] = d * (float)sc[2 * j];
        chi[j] = d * (float)sc[2 * j + 1];
    }
}

inline uint8 stage_inp_f16(__local slm_t *sb, const __global half *src, uint hh)
{
    uint8 iv = intel_sub_group_block_read8((const __global uint *)src);
#define Q6K_ST(j, comp) \
    SB_STORE(sb + (j) * 32 + 2 * hh, convert_float2(as_half2(comp)));
    Q6K_ST(0, iv.s0) Q6K_ST(1, iv.s1) Q6K_ST(2, iv.s2) Q6K_ST(3, iv.s3)
    Q6K_ST(4, iv.s4) Q6K_ST(5, iv.s5) Q6K_ST(6, iv.s6) Q6K_ST(7, iv.s7)
#undef Q6K_ST
    return iv;
}

inline float q6k_corr(uint8 iv, const float *clo, const float *chi)
{
    float corr = 0.0f;
#define Q6K_STAGE(j, comp)                                                    \
    {                                                                         \
        float2 _f = convert_float2(as_half2(comp));                           \
        float _v = _f.x + _f.y;                                               \
        _v += intel_sub_group_shuffle_xor(_v, 1);                             \
        _v += intel_sub_group_shuffle_xor(_v, 2);                             \
        _v += intel_sub_group_shuffle_xor(_v, 4);                             \
        corr = fma(clo[j], intel_sub_group_shuffle(_v, 0), corr);             \
        corr = fma(chi[j], intel_sub_group_shuffle(_v, 8), corr);             \
    }
    Q6K_STAGE(0, iv.s0) Q6K_STAGE(1, iv.s1) Q6K_STAGE(2, iv.s2) Q6K_STAGE(3, iv.s3)
    Q6K_STAGE(4, iv.s4) Q6K_STAGE(5, iv.s5) Q6K_STAGE(6, iv.s6) Q6K_STAGE(7, iv.s7)
#undef Q6K_STAGE
    return corr;
}

inline void prestage_block_q6k(__local slm_t *sb, __local float *hs_slm,
                               const __global half *src, uint hh)
{
    uint8 iv = stage_inp_f16(sb, src, hh);
#define Q6K_PS(j, comp)                                                       \
    {                                                                         \
        float2 _f = convert_float2(as_half2(comp));                          \
        float _v = _f.x + _f.y;                                              \
        _v += intel_sub_group_shuffle_xor(_v, 1);                            \
        _v += intel_sub_group_shuffle_xor(_v, 2);                            \
        _v += intel_sub_group_shuffle_xor(_v, 4);                            \
        float _lo = intel_sub_group_shuffle(_v, 0);                          \
        float _hi = intel_sub_group_shuffle(_v, 8);                          \
        if (hh == 0) {                                                       \
            hs_slm[j]     = _lo;                                             \
            hs_slm[8 + j] = _hi;                                             \
        }                                                                    \
    }
    Q6K_PS(0, iv.s0) Q6K_PS(1, iv.s1) Q6K_PS(2, iv.s2) Q6K_PS(3, iv.s3)
    Q6K_PS(4, iv.s4) Q6K_PS(5, iv.s5) Q6K_PS(6, iv.s6) Q6K_PS(7, iv.s7)
#undef Q6K_PS
}

inline float q6k_corr_slm(__local const float *hs,
                          const float *clo, const float *chi)
{
    float corr = 0.0f;
    unroll_for (int j = 0; j < 8; j++) {
        corr = fma(clo[j], hs[j],     corr);
        corr = fma(chi[j], hs[8 + j], corr);
    }
    return corr;
}

/* =======================================================================
 * Kernel 1: group_up_gate_q6k_sg
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void group_up_gate_q6k_sg(
    __global const half   * restrict inputs,
    __global const uchar  * restrict ups,
    __global const uchar  * restrict gates,
    __global const uint   * restrict indexes,
    __global       float  * restrict outputs,
    uint output_num,
    uint aligned_output_num,
    uint token_len,
    uint input_len,
    uint output_len
) {
    const uint nbpr       = input_len / 256;
    const uint num_blocks = input_len * output_len / 256;
    const uint expert_size = num_blocks * 210;  /* 128 ql + 64 qh + 16 ps + 2 pd */

    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint vt = get_group_id(2);
    const uint v  = vt % output_num;
    const uint t  = vt / output_num;

    const uint index = indexes ? indexes[t * aligned_output_num + v] : v;

    const __global uchar *up_pql = ups   + (size_t)index * expert_size;
    const __global uchar *gt_pql = gates + (size_t)index * expert_size;

#if SPLIT_UG
    #define KHALF (KSPLIT / 2)
    const uint is_gate = sg >= KHALF;
    const uint sg_k    = sg - is_gate * KHALF;

    __local slm_t slm_inp[SLM_POOL_FLOATS];
    __local float slm_hs[SLM_ACT_BLOCKS * 16];
    __local float slm_red[2 * KHALF * OPG];
    __local slm_t *sbase = slm_inp + sg * 256;

    const __global uchar *base = is_gate ? gt_pql : up_pql;
    const __global uint  *pql_u = (const __global uint  *)base;
    const __global uint  *pqh_u = (const __global uint  *)(base + (size_t)num_blocks * 128);
    const __global uchar *ps    = base + (size_t)num_blocks * 192;
    const __global uchar *pd    = base + (size_t)num_blocks * 208;

    const bool prestage = (nbpr <= SLM_ACT_BLOCKS);
    if (prestage) {
        for (uint b = sg; b < nbpr; b += KSPLIT)
            prestage_block_q6k(slm_inp + (size_t)b * 256, slm_hs + b * 16,
                               inputs + (size_t)t * input_len + (size_t)b * 256, hh);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float acc = 0.0f;
    for (uint bid = sg_k; bid < nbpr; bid += KHALF) {
        const uint pql_e = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e = (h * nbpr + bid) * (OPG *  64 / 4);
        const uint ps_e  = (h * nbpr + bid) * (OPG * 16);
        const uint pd_e  = (h * nbpr + bid) * (OPG * 2);

        uint8 w0  = intel_sub_group_block_read8(pql_u + pql_e +   0);
        uint8 w1  = intel_sub_group_block_read8(pql_u + pql_e + 128);
        uint8 w2  = intel_sub_group_block_read8(pql_u + pql_e + 256);
        uint8 w3  = intel_sub_group_block_read8(pql_u + pql_e + 384);
        uint8 hq0 = intel_sub_group_block_read8(pqh_u + pqh_e +   0);
        uint8 hq1 = intel_sub_group_block_read8(pqh_u + pqh_e + 128);
        uchar16 scq = intel_sub_group_block_read_uc16(ps + ps_e);
        ushort  dr  = intel_sub_group_block_read_us((const __global ushort *)(pd + pd_e));

        float clo[8], chi[8];
        q6k_decode(scq, dr, clo, chi);

        __local const slm_t *sb;
        float corr;
        if (prestage) {
            sb = slm_inp + (size_t)bid * 256;          /* already resident */
            corr = q6k_corr_slm(slm_hs + bid * 16, clo, chi);
        } else {
            sb = sbase;
            uint8 iv = stage_inp_f16(sbase,
                                     inputs + (size_t)t * input_len
                                     + (size_t)bid * 256, hh);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            corr = q6k_corr(iv, clo, chi);
        }
        acc += q6k_acc(w0, w1, w2, w3, hq0, hq1, clo, chi, sb) - 32.0f * corr;
    }

    slm_red[sg * OPG + hh] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float u = 0.0f, gt = 0.0f;
        unroll_for (int r = 0; r < KHALF; r++) {
            u  += slm_red[r * OPG + hh];
            gt += slm_red[(KHALF + r) * OPG + hh];
        }
        float sig = 1.0f / (1.0f + native_exp(-gt));
        outputs[(size_t)t * output_num * output_len + (size_t)v * output_len
                + h * OPG + hh] = (sig * gt) * u;
    }
    #undef KHALF
#else
    __local slm_t slm_inp [SLM_POOL_FLOATS];
    __local float slm_hs  [SLM_ACT_BLOCKS * 16];
    __local float slm_up  [KSPLIT * OPG];
    __local float slm_gate[KSPLIT * OPG];
    __local slm_t *sbase = slm_inp + sg * 256;

    const __global uint  *up_pql_u = (const __global uint  *)up_pql;
    const __global uint  *up_pqh_u = (const __global uint  *)(up_pql + (size_t)num_blocks * 128);
    const __global uchar *up_ps    = up_pql + (size_t)num_blocks * 192;
    const __global uchar *up_pd    = up_pql + (size_t)num_blocks * 208;
    const __global uint  *gt_pql_u = (const __global uint  *)gt_pql;
    const __global uint  *gt_pqh_u = (const __global uint  *)(gt_pql + (size_t)num_blocks * 128);
    const __global uchar *gt_ps    = gt_pql + (size_t)num_blocks * 192;
    const __global uchar *gt_pd    = gt_pql + (size_t)num_blocks * 208;

    const bool prestage = (nbpr <= SLM_ACT_BLOCKS);
    if (prestage) {
        for (uint b = sg; b < nbpr; b += KSPLIT)
            prestage_block_q6k(slm_inp + (size_t)b * 256, slm_hs + b * 16,
                               inputs + (size_t)t * input_len + (size_t)b * 256, hh);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float up_acc = 0.0f, gate_acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        const uint pql_e = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e = (h * nbpr + bid) * (OPG *  64 / 4);
        const uint ps_e  = (h * nbpr + bid) * (OPG * 16);
        const uint pd_e  = (h * nbpr + bid) * (OPG * 2);

        uint8 uw0 = intel_sub_group_block_read8(up_pql_u + pql_e +   0);
        uint8 uw1 = intel_sub_group_block_read8(up_pql_u + pql_e + 128);
        uint8 uw2 = intel_sub_group_block_read8(up_pql_u + pql_e + 256);
        uint8 uw3 = intel_sub_group_block_read8(up_pql_u + pql_e + 384);
        uint8 uh0 = intel_sub_group_block_read8(up_pqh_u + pqh_e +   0);
        uint8 uh1 = intel_sub_group_block_read8(up_pqh_u + pqh_e + 128);
        uchar16 uscq = intel_sub_group_block_read_uc16(up_ps + ps_e);
        ushort  udr  = intel_sub_group_block_read_us((const __global ushort *)(up_pd + pd_e));
        uint8 gw0 = intel_sub_group_block_read8(gt_pql_u + pql_e +   0);
        uint8 gw1 = intel_sub_group_block_read8(gt_pql_u + pql_e + 128);
        uint8 gw2 = intel_sub_group_block_read8(gt_pql_u + pql_e + 256);
        uint8 gw3 = intel_sub_group_block_read8(gt_pql_u + pql_e + 384);
        uint8 gh0 = intel_sub_group_block_read8(gt_pqh_u + pqh_e +   0);
        uint8 gh1 = intel_sub_group_block_read8(gt_pqh_u + pqh_e + 128);
        uchar16 gscq = intel_sub_group_block_read_uc16(gt_ps + ps_e);
        ushort  gdr  = intel_sub_group_block_read_us((const __global ushort *)(gt_pd + pd_e));

        float uclo[8], uchi[8], gclo[8], gchi[8];
        q6k_decode(uscq, udr, uclo, uchi);
        q6k_decode(gscq, gdr, gclo, gchi);

        __local const slm_t *sb;
        float ucorr, gcorr;
        if (prestage) {
            sb = slm_inp + (size_t)bid * 256;
            ucorr = q6k_corr_slm(slm_hs + bid * 16, uclo, uchi);
            gcorr = q6k_corr_slm(slm_hs + bid * 16, gclo, gchi);
        } else {
            sb = sbase;
            uint8 iv = stage_inp_f16(sbase,
                                     inputs + (size_t)t * input_len
                                     + (size_t)bid * 256, hh);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            ucorr = q6k_corr(iv, uclo, uchi);
            gcorr = q6k_corr(iv, gclo, gchi);
        }
        up_acc   += q6k_acc(uw0, uw1, uw2, uw3, uh0, uh1, uclo, uchi, sb)
                    - 32.0f * ucorr;
        gate_acc += q6k_acc(gw0, gw1, gw2, gw3, gh0, gh1, gclo, gchi, sb)
                    - 32.0f * gcorr;
    }

#if KSPLIT == 1
    {
        float g   = gate_acc;
        float sig = 1.0f / (1.0f + native_exp(-g));
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
        float sig = 1.0f / (1.0f + native_exp(-gt));
        outputs[(size_t)t * output_num * output_len + (size_t)v * output_len
                + h * OPG + hh] = (sig * gt) * u;
    }
#endif
#endif /* SPLIT_UG */
}

/* =======================================================================
 * Kernel 2: down_merge_q6k_sg
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void down_merge_q6k_sg(
    __global const half   * restrict inputs,
    __global const uchar  * restrict downs,
    __global const uint   * restrict indexes,
    __global const float  * restrict efficients,
    __global       float  * restrict outputs,
    uint input_num,
    uint aligned_input_num,
    uint token_len,
    uint input_len,
    uint output_len,
    int  is_acc
) {
    const uint nbpr       = input_len / 256;
    const uint num_blocks = input_len * output_len / 256;
    const uint expert_size = num_blocks * 210;

    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint t  = get_group_id(2);

    __local slm_t slm_inp[SLM_POOL_FLOATS];
    __local float slm_hs[SLM_ACT_BLOCKS * 16];
    __local float slm_red[KSPLIT * OPG];
    __local slm_t *sbase = slm_inp + sg * 256;

    const bool fast_route = (input_num <= OPG) && (aligned_input_num >= OPG);
    uint  idx_reg = 0u;
    uint  eff_reg = 0u;
    if (fast_route) {
        if (indexes)
            idx_reg = intel_sub_group_block_read(indexes + (size_t)t * aligned_input_num);
        eff_reg = intel_sub_group_block_read(
                      (const __global uint *)(efficients + (size_t)t * aligned_input_num));
    }
    const uint total = input_num * nbpr;

    const bool prestage = (total <= SLM_ACT_BLOCKS);
    if (prestage) {
        for (uint b = sg; b < total; b += KSPLIT) {
            const uint be = b / nbpr;
            const uint bb = b - be * nbpr;
            prestage_block_q6k(slm_inp + (size_t)b * 256, slm_hs + b * 16,
                               inputs + (size_t)t * input_num * input_len
                               + (size_t)be * input_len + (size_t)bb * 256, hh);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint per  = (total + KSPLIT - 1) / KSPLIT;
    const uint from = sg * per;
    const uint to   = min(from + per, total);
    float acc = 0.0f;

    for (uint idx = from; idx < to; idx++) {
        const uint e   = idx / nbpr;
        const uint bid = idx - e * nbpr;

        uint  index;
        float efficient;
        if (fast_route) {
            index     = indexes ? intel_sub_group_shuffle(idx_reg, e) : e;
            efficient = as_float(intel_sub_group_shuffle(eff_reg, e));
        } else {
            index     = indexes ? indexes[t * aligned_input_num + e] : e;
            efficient = efficients[t * aligned_input_num + e];
        }

        const __global uchar *d_pql = downs + (size_t)index * expert_size;
        const __global uint  *pql_u = (const __global uint  *)d_pql;
        const __global uint  *pqh_u = (const __global uint  *)(d_pql + (size_t)num_blocks * 128);
        const __global uchar *ps    = d_pql + (size_t)num_blocks * 192;
        const __global uchar *pd    = d_pql + (size_t)num_blocks * 208;

        const uint pql_e = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e = (h * nbpr + bid) * (OPG *  64 / 4);
        const uint ps_e  = (h * nbpr + bid) * (OPG * 16);
        const uint pd_e  = (h * nbpr + bid) * (OPG * 2);

        uint8 w0  = intel_sub_group_block_read8(pql_u + pql_e +   0);
        uint8 w1  = intel_sub_group_block_read8(pql_u + pql_e + 128);
        uint8 w2  = intel_sub_group_block_read8(pql_u + pql_e + 256);
        uint8 w3  = intel_sub_group_block_read8(pql_u + pql_e + 384);
        uint8 hq0 = intel_sub_group_block_read8(pqh_u + pqh_e +   0);
        uint8 hq1 = intel_sub_group_block_read8(pqh_u + pqh_e + 128);
        uchar16 scq = intel_sub_group_block_read_uc16(ps + ps_e);
        ushort  dr  = intel_sub_group_block_read_us((const __global ushort *)(pd + pd_e));

        float clo[8], chi[8];
        q6k_decode(scq, dr, clo, chi);

        __local const slm_t *sb;
        float corr;
        if (prestage) {
            sb = slm_inp + (size_t)idx * 256;
            corr = q6k_corr_slm(slm_hs + idx * 16, clo, chi);
        } else {
            sb = sbase;
            uint8 iv = stage_inp_f16(sbase,
                                     inputs + (size_t)t * input_num * input_len
                                     + (size_t)e * input_len + (size_t)bid * 256, hh);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            corr = q6k_corr(iv, clo, chi);
        }
        acc = fma(q6k_acc(w0, w1, w2, w3, hq0, hq1, clo, chi, sb) - 32.0f * corr,
                  efficient, acc);
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
