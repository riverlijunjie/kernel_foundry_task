/*
 * moe_gemv_q5k_sg.cl — High-performance Q5_K MoE GEMV with Intel sub-group
 *                      block reads (transposed "SG" weight layout).
 *
 * ============================ OPTIMIZATION NOTES =========================
 * A pure-read kernel with this dispatch and these block reads sustains
 * ~410 GB/s on Arc B580, so the inner loop is structured to stay
 * bandwidth/latency bound rather than ALU bound:
 *
 *  1) PACKED PAIR WORDS.  Two weights that live 16 bits apart in the qs word
 *     are extracted with a single AND (0x000F000F); their two 5th bits are
 *     ALSO 16 bits apart in the 32-bit qh word, so ONE shift + one masked OR
 *     completes BOTH 5-bit weights.  The pair then converts to float with a
 *     single mov for the low half and shr + mov for the high half.
 *
 *  2) VECTOR ACCUMULATORS.  The 32 mads per sub-block are spread over 8
 *     independent float4 lanes, so the FP pipe always has work in flight;
 *     activations come from SLM as float4 (one d32x4 SLM load per 4 weights).
 *
 *  3) ACTIVATION SUMS FOR FREE.  The 256 activations arrive in ONE
 *     intel_sub_group_block_read8; component j covers exactly sub-block j, so
 *     the 8 isum[] values are 8 sub-group reductions.
 *
 *  4) UP/GATE SPLIT ACROSS SUB-GROUPS (SPLIT_UG, KSPLIT>=2): the first half of
 *     the sub-groups reduce the UP matrix, the second half the GATE matrix,
 *     halving live weight registers and doubling the in-flight loads per
 *     matrix.  The halves meet in SLM at the existing work-group barrier.
 *
 *  5) The psl scale record is fetched with ONE block_read4 (the whole 256-byte
 *     SoA record); the per-lane 16-bit fields are recovered with shuffles.
 *
 * == Transposed ("SG") per-expert layout — SAME total size as original ==
 *   pqs_T:  num_blocks * 128 bytes  (low nibbles)
 *   pqh_T:  num_blocks * 32 bytes   (1 high bit / weight; sub-block j's 4-byte
 *           qh word for all OPG lanes at j*OPG*4)
 *   psl_T:  num_blocks * 16 bytes   (SoA, identical to Q4_K psl_T)
 *   expert_size = num_blocks * 176  (128 qs + 32 qh + 16 sl).
 *
 * == Dispatch ==
 *   group_up_gate_q5k_sg: global=(output_len,KSPLIT,output_num*token_len)
 *   down_merge_q5k_sg   : global=(output_len,KSPLIT,token_len)
 *   local=(OPG=16,KSPLIT,1)
 */

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

/* ===========================================================================
 * ACTIVATION PRE-STAGING (ported from ocl_v2/moe_3gemm_swiglu_mlp.cl):
 * stage the WHOLE activation window into SLM once, together with the per
 * sub-block activation sums (isum[]), behind a SINGLE work-group barrier; the
 * GEMV inner loop then only reads SLM (no re-stage, no re-reduce, no
 * per-iteration barrier), so IGC can keep every iteration's weight block-reads
 * in flight.  Falls back to a streaming single-slot path when the window
 * exceeds the SLM budget.  The SLM pool is shared by the two modes.
 * ========================================================================= */

/* Activation SLM budget, in 256-element blocks (1 block = 1 KiB as float). */
#ifndef SLM_ACT_BLOCKS
#define SLM_ACT_BLOCKS 16
#endif

/* Streaming fallback: a SINGLE 256-float slot per sub-group (a
 * sub_group_barrier orders each stage->use).  Keeping the pool flat is what
 * lets down_merge run at KSPLIT=16 (2048 live sub-groups) without growing SLM
 * past the pre-stage budget. */
#define SLM_STREAM_FLOATS (KSPLIT * 256)
#if (SLM_ACT_BLOCKS * 256) > SLM_STREAM_FLOATS
#define SLM_POOL_FLOATS (SLM_ACT_BLOCKS * 256)
#else
#define SLM_POOL_FLOATS SLM_STREAM_FLOATS
#endif

#define W2_MASK 0x000F000Fu   /* two 4-bit weights, 16 bit apart          */
#define QH_MASK 0x00100010u   /* their two 5th bits, also 16 bit apart    */

/* Activations are staged in SLM as float; the dot product accumulates in fp32. */
typedef float slm_t;
#define XL4(p)                (*(__local const float4 *)(p))
#define SB_STORE(dst, f2)     (*(__local float2 *)(dst) = (f2))

/* ---------------------------------------------------------------------------
 * One sub-block (32 weights): 4 qs chunk words + the sub-block's 32-bit qh
 * word + 32 SLM activations.  For chunk c and byte r:
 *     weight (   4c+r) = (q >> (8r    )) & 0xF | ((H >> (r*8 + c    )) & 1) << 4
 *     weight (16+4c+r) = (q >> (8r + 4)) & 0xF | ((H >> (r*8 + c + 4)) & 1) << 4
 * The pairs (r=0,r=2) and (r=1,r=3) are 16 bits apart in BOTH q and H.
 * ------------------------------------------------------------------------- */
#define Q5K_CHUNK(q, H, c)                                                    \
    do {                                                                      \
        uint _q = (q), _H = (H);                                              \
        uint _a = ( _q        & W2_MASK) | ((_H << (4 - (c))) & QH_MASK);     \
        uint _b = ((_q >>  8) & W2_MASK) | ((_H >> (4 + (c))) & QH_MASK);     \
        uint _c = ((_q >>  4) & W2_MASK) | ((_H >>      (c) ) & QH_MASK);     \
        uint _d = ((_q >> 12) & W2_MASK) | ((_H >> (8 + (c))) & QH_MASK);     \
        float4 _wl = (float4)((float)as_ushort2(_a).x, (float)as_ushort2(_b).x,\
                              (float)(_a >> 16),       (float)(_b >> 16));    \
        float4 _wh = (float4)((float)as_ushort2(_c).x, (float)as_ushort2(_d).x,\
                              (float)(_c >> 16),       (float)(_d >> 16));    \
        vlo = fma(XL4(in + (c) * 4),      _wl, vlo);                          \
        vhi = fma(XL4(in + 16 + (c) * 4), _wh, vhi);                          \
    } while (0)

inline float q5k_subblock(uint q0, uint q1, uint q2, uint q3, uint H,
                          __local const slm_t *in)
{
    float4 vlo = (float4)(0.0f);
    float4 vhi = (float4)(0.0f);
    Q5K_CHUNK(q0, H, 0);
    Q5K_CHUNK(q1, H, 1);
    Q5K_CHUNK(q2, H, 2);
    Q5K_CHUNK(q3, H, 3);
    float4 s = vlo + vhi;
    return (s.x + s.y) + (s.z + s.w);
}

inline float q5k_acc(uint8 w0, uint8 w1, uint8 w2, uint8 w3, uint8 hq,
                     const float *scale, const float *minv, const float *isum,
                     __local const slm_t *sb)
{
    /* Dual accumulators reduce dependency depth; separate min sum. */
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    acc0 = fma(scale[0], q5k_subblock(w0.s0, w0.s1, w0.s2, w0.s3, hq.s0, sb + 0 * 32), acc0);
    acc1 = fma(scale[1], q5k_subblock(w0.s4, w0.s5, w0.s6, w0.s7, hq.s1, sb + 1 * 32), acc1);
    acc0 = fma(scale[2], q5k_subblock(w1.s0, w1.s1, w1.s2, w1.s3, hq.s2, sb + 2 * 32), acc0);
    acc1 = fma(scale[3], q5k_subblock(w1.s4, w1.s5, w1.s6, w1.s7, hq.s3, sb + 3 * 32), acc1);
    acc0 = fma(scale[4], q5k_subblock(w2.s0, w2.s1, w2.s2, w2.s3, hq.s4, sb + 4 * 32), acc0);
    acc1 = fma(scale[5], q5k_subblock(w2.s4, w2.s5, w2.s6, w2.s7, hq.s5, sb + 5 * 32), acc1);
    acc0 = fma(scale[6], q5k_subblock(w3.s0, w3.s1, w3.s2, w3.s3, hq.s6, sb + 6 * 32), acc0);
    acc1 = fma(scale[7], q5k_subblock(w3.s4, w3.s5, w3.s6, w3.s7, hq.s7, sb + 7 * 32), acc1);

    float min0 = 0.0f;
    float min1 = 0.0f;
    min0 = fma(minv[0], isum[0], min0);
    min1 = fma(minv[1], isum[1], min1);
    min0 = fma(minv[2], isum[2], min0);
    min1 = fma(minv[3], isum[3], min1);
    min0 = fma(minv[4], isum[4], min0);
    min1 = fma(minv[5], isum[5], min1);
    min0 = fma(minv[6], isum[6], min0);
    min1 = fma(minv[7], isum[7], min1);
    return (acc0 + acc1) - (min0 + min1);
}

/* ---------------------------------------------------------------------------
 * psl record: ONE block_read4 pulls the complete 256-byte SoA record.
 *   component 0 -> sl[lane]   component 1 -> ml[lane]
 *   component 2 -> sh[2l],sh[2l+1] (lane<8) / mh[..] (lane>=8)
 *   component 3 -> d [2l],d [2l+1] (lane<8) / dmin[..] (lane>=8)
 * ------------------------------------------------------------------------- */
inline void q5k_decode(const __global uint *psl_pu, uint off, uint hh,
                       float *scale, float *minv)
{
    uint4 r = intel_sub_group_block_read4(psl_pu + off);

    const uint half_sel = (hh & 1u) * 16u;
    const uint src_lane = hh >> 1;

    uint shmh_lo = intel_sub_group_shuffle(r.s2, src_lane);
    uint shmh_hi = intel_sub_group_shuffle(r.s2, src_lane + 8u);
    uint ddmn_lo = intel_sub_group_shuffle(r.s3, src_lane);
    uint ddmn_hi = intel_sub_group_shuffle(r.s3, src_lane + 8u);

    uint sl = r.s0;
    uint ml = r.s1;
    uint sh = (shmh_lo >> half_sel) & 0xFFFFu;
    uint mh = (shmh_hi >> half_sel) & 0xFFFFu;
    float d    = (float)as_half((ushort)((ddmn_lo >> half_sel) & 0xFFFFu));
    float dmin = (float)as_half((ushort)((ddmn_hi >> half_sel) & 0xFFFFu));

    unroll_for (int i = 0; i < 8; i++) {
        uint sq = ((sl >> (i * 4)) & 0xFu) | (((sh >> (i * 2)) & 0x3u) << 4);
        uint mq = ((ml >> (i * 4)) & 0xFu) | (((mh >> (i * 2)) & 0x3u) << 4);
        scale[i] = (float)sq * d;
        minv[i]  = (float)mq * dmin;
    }
}

/* Stage 256 fp16 activations into SLM as float + the 8 per-sub-block sums. */
inline void load_inp_f16(__local slm_t *sb, const __global half *src, uint hh,
                         float *isum)
{
    uint8 iv = intel_sub_group_block_read8((const __global uint *)src);
#define Q5K_STAGE(j, comp)                                                   \
    {                                                                        \
        float2 _f = convert_float2(as_half2(comp));                          \
        SB_STORE(sb + (j) * 32 + 2 * hh, _f);                                \
        isum[j] = sub_group_reduce_add(_f.x + _f.y);                         \
    }
    Q5K_STAGE(0, iv.s0) Q5K_STAGE(1, iv.s1) Q5K_STAGE(2, iv.s2) Q5K_STAGE(3, iv.s3)
    Q5K_STAGE(4, iv.s4) Q5K_STAGE(5, iv.s5) Q5K_STAGE(6, iv.s6) Q5K_STAGE(7, iv.s7)
#undef Q5K_STAGE
}

/* Pre-stage ONE activation block (256 halves) + publish its 8 sub-block sums. */
inline void prestage_block(__local slm_t *sb, __local float *isum_slm,
                           const __global half *src, uint hh)
{
    float isum[8];
    load_inp_f16(sb, src, hh, isum);
    if (hh == 0) {
        unroll_for (int j = 0; j < 8; j++) isum_slm[j] = isum[j];
    }
}

/* Recover the 8 pre-computed sub-block sums with two uniform float4 SLM reads. */
#define LOAD_ISUM_SLM(isum, isum_slm) \
    do { \
        float4 _i0 = *(__local const float4 *)(isum_slm); \
        float4 _i1 = *(__local const float4 *)((isum_slm) + 4); \
        (isum)[0] = _i0.x; (isum)[1] = _i0.y; (isum)[2] = _i0.z; (isum)[3] = _i0.w; \
        (isum)[4] = _i1.x; (isum)[5] = _i1.y; (isum)[6] = _i1.z; (isum)[7] = _i1.w; \
    } while (0)

/* =======================================================================
 * Kernel 1: group_up_gate_q5k_sg
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__attribute__((reqd_work_group_size(OPG, KSPLIT, 1)))
__attribute__((vec_type_hint(float4)))
__kernel void group_up_gate_q5k_sg(
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
    const uint expert_size = num_blocks * 176;  /* 128 qs + 32 qh + 16 sl */

    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint vt = get_group_id(2);
    const uint v  = vt % output_num;
    const uint t  = vt / output_num;

    const __global half *inputs_t = inputs + (size_t)t * input_len;
    const __global uint *indexes_t = indexes ? indexes + (size_t)t * aligned_output_num : 0;
    const size_t out_base = (size_t)t * output_num * output_len
                            + (size_t)v * output_len;
    const uint pe_stride  = (OPG * 128 / 4);
    const uint pqh_stride = (OPG *  32 / 4);
    const uint psl_stride = (OPG *  16 / 4);
    const uint pe_base    = h * nbpr * pe_stride;
    const uint pqh_base   = h * nbpr * pqh_stride;
    const uint psl_base   = h * nbpr * psl_stride;
    uint index;
    if (indexes) {
        uint idx_lane = 0u;
        if (hh == 0) idx_lane = indexes_t[v];
        index = intel_sub_group_shuffle(idx_lane, 0);
    } else {
        index = v;
    }

    const __global uchar *up_pqs = ups   + (size_t)index * expert_size;
    const __global uchar *gt_pqs = gates + (size_t)index * expert_size;

#if SPLIT_UG
    #define KHALF (KSPLIT / 2)
    const uint is_gate = sg >= KHALF;
    const uint sg_k    = sg - is_gate * KHALF;

    __local slm_t __attribute__((aligned(64))) slm_inp[SLM_POOL_FLOATS];
    __local float __attribute__((aligned(64))) slm_isum[SLM_ACT_BLOCKS * 8];
    __local float __attribute__((aligned(64))) slm_red[2 * KHALF * OPG];
    __local slm_t *sbase = slm_inp + sg * 256;

    const __global uchar *base  = is_gate ? gt_pqs : up_pqs;
    const __global uint *pqs_u  = (const __global uint *)base;
    const __global uint *pqh_u  = (const __global uint *)(base + (size_t)num_blocks * 128);
    const __global uint *psl_pu = (const __global uint *)(base + (size_t)num_blocks * 160);

    const bool prestage = (nbpr <= SLM_ACT_BLOCKS);
    if (prestage) {
        for (uint b = sg; b < nbpr; b += KSPLIT)
            prestage_block(slm_inp + (size_t)b * 256, slm_isum + b * 8,
                           inputs_t + (size_t)b * 256, hh);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float acc_rg = 0.0f;
    for (uint bid = sg_k; bid < nbpr; bid += KHALF) {
        const uint pe_u       = pe_base  + bid * pe_stride;
        const uint pqh_e      = pqh_base + bid * pqh_stride;
        const uint psl_pu_off = psl_base + bid * psl_stride;

        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);
        uint8 hq = intel_sub_group_block_read8(pqh_u + pqh_e);

        float isum[8], scale[8], minv[8];
        q5k_decode(psl_pu, psl_pu_off, hh, scale, minv);

        __local const slm_t *sb;
        if (prestage) {
            sb = slm_inp + (size_t)bid * 256;          /* already resident */
            LOAD_ISUM_SLM(isum, slm_isum + bid * 8);
        } else {
            sb = sbase;
            load_inp_f16(sbase,
                         inputs_t + (size_t)bid * 256, hh, isum);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        }

        acc_rg += q5k_acc(w0, w1, w2, w3, hq, scale, minv, isum, sb);
    }

    slm_red[sg * OPG + hh] = acc_rg;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float u = 0.0f, gt = 0.0f;
        unroll_for (int r = 0; r < KHALF; r++) {
            u  += slm_red[r * OPG + hh];
            gt += slm_red[(KHALF + r) * OPG + hh];
        }
        float sig = 1.0f / (1.0f + native_exp(-gt));
        outputs[out_base + h * OPG + hh] = (sig * gt) * u;
    }
    #undef KHALF
#else
    __local slm_t __attribute__((aligned(64))) slm_inp [SLM_POOL_FLOATS];
    __local float __attribute__((aligned(64))) slm_isum[SLM_ACT_BLOCKS * 8];
    __local float __attribute__((aligned(64))) slm_up  [KSPLIT * OPG];
    __local float __attribute__((aligned(64))) slm_gate[KSPLIT * OPG];
    __local slm_t *sbase = slm_inp + sg * 256;

    const __global uint *up_pqs_u  = (const __global uint *)up_pqs;
    const __global uint *up_pqh_u  = (const __global uint *)(up_pqs + (size_t)num_blocks * 128);
    const __global uint *up_psl_pu = (const __global uint *)(up_pqs + (size_t)num_blocks * 160);
    const __global uint *gt_pqs_u  = (const __global uint *)gt_pqs;
    const __global uint *gt_pqh_u  = (const __global uint *)(gt_pqs + (size_t)num_blocks * 128);
    const __global uint *gt_psl_pu = (const __global uint *)(gt_pqs + (size_t)num_blocks * 160);

    const bool prestage = (nbpr <= SLM_ACT_BLOCKS);
    if (prestage) {
        for (uint b = sg; b < nbpr; b += KSPLIT)
            prestage_block(slm_inp + (size_t)b * 256, slm_isum + b * 8,
                           inputs_t + (size_t)b * 256, hh);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Register-blocked up/gate accumulators. */
    float2 acc_rg = (float2)(0.0f);

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        const uint pe_u       = pe_base  + bid * pe_stride;
        const uint pqh_e      = pqh_base + bid * pqh_stride;
        const uint psl_pu_off = psl_base + bid * psl_stride;

        uint8 uw0 = intel_sub_group_block_read8(up_pqs_u + pe_u +   0);
        uint8 uw1 = intel_sub_group_block_read8(up_pqs_u + pe_u + 128);
        uint8 uw2 = intel_sub_group_block_read8(up_pqs_u + pe_u + 256);
        uint8 uw3 = intel_sub_group_block_read8(up_pqs_u + pe_u + 384);
        uint8 uhq = intel_sub_group_block_read8(up_pqh_u + pqh_e);
        uint8 gw0 = intel_sub_group_block_read8(gt_pqs_u + pe_u +   0);
        uint8 gw1 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 128);
        uint8 gw2 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 256);
        uint8 gw3 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 384);
        uint8 ghq = intel_sub_group_block_read8(gt_pqh_u + pqh_e);

        float isum[8], uscale[8], uminv[8], gscale[8], gminv[8];
        q5k_decode(up_psl_pu, psl_pu_off, hh, uscale, uminv);
        q5k_decode(gt_psl_pu, psl_pu_off, hh, gscale, gminv);

        __local const slm_t *sb;
        if (prestage) {
            sb = slm_inp + (size_t)bid * 256;
            LOAD_ISUM_SLM(isum, slm_isum + bid * 8);
        } else {
            sb = sbase;
            load_inp_f16(sbase,
                         inputs_t + (size_t)bid * 256, hh, isum);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        }

        acc_rg.s0 += q5k_acc(uw0, uw1, uw2, uw3, uhq, uscale, uminv, isum, sb);
        acc_rg.s1 += q5k_acc(gw0, gw1, gw2, gw3, ghq, gscale, gminv, isum, sb);
    }

#if KSPLIT == 1
    {
        float g   = acc_rg.s1;
        float sig = 1.0f / (1.0f + native_exp(-g));
        outputs[out_base + h * OPG + hh] = (sig * g) * acc_rg.s0;
    }
#else
    slm_up  [sg * OPG + hh] = acc_rg.s0;
    slm_gate[sg * OPG + hh] = acc_rg.s1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float u = 0.0f, gt = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) {
            u  += slm_up  [r * OPG + hh];
            gt += slm_gate[r * OPG + hh];
        }
        float sig = 1.0f / (1.0f + native_exp(-gt));
        outputs[out_base + h * OPG + hh] = (sig * gt) * u;
    }
#endif
#endif /* SPLIT_UG */
}

/* =======================================================================
 * Kernel 2: down_merge_q5k_sg
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__attribute__((reqd_work_group_size(OPG, KSPLIT, 1)))
__attribute__((vec_type_hint(float4)))
__kernel void down_merge_q5k_sg(
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
    const uint expert_size = num_blocks * 176;

    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint t  = get_group_id(2);

    const __global half *inputs_t = inputs + (size_t)t * input_num * input_len;
    const __global uint *indexes_t = indexes ? indexes + (size_t)t * aligned_input_num : 0;
    const __global float *eff_t = efficients + (size_t)t * aligned_input_num;
    const size_t out_base = (size_t)t * output_len;
    const uint pe_stride  = (OPG * 128 / 4);
    const uint pqh_stride = (OPG *  32 / 4);
    const uint psl_stride = (OPG *  16 / 4);
    const uint pe_base    = h * nbpr * pe_stride;
    const uint pqh_base   = h * nbpr * pqh_stride;
    const uint psl_base   = h * nbpr * psl_stride;

    __local slm_t __attribute__((aligned(64))) slm_inp[SLM_POOL_FLOATS];
    __local float __attribute__((aligned(64))) slm_isum[SLM_ACT_BLOCKS * 8];
    __local float __attribute__((aligned(64))) slm_red[KSPLIT * OPG];
    __local slm_t *sbase = slm_inp + sg * 256;

    /* ---- Hoisted routing: ONE 64-byte sub-group block read gives lane l the
     * routing info of expert l; the loop recovers its own with a shuffle
     * instead of a serial dependent scalar load per iteration.  Falls back to
     * scalar loads when input_num > OPG. ---- */
    const bool fast_route = (input_num <= OPG) && (aligned_input_num >= OPG);
    uint  idx_reg = 0u;
    uint  eff_reg = 0u;
    if (fast_route) {
        if (indexes)
            idx_reg = intel_sub_group_block_read(indexes_t);
        eff_reg = intel_sub_group_block_read((const __global uint *)eff_t);
    }
    const uint total = input_num * nbpr;

    /* ---- one-shot activation pre-staging over the whole [input_num,
     * input_len] block set (`total` blocks). ---- */
    const bool prestage = (total <= SLM_ACT_BLOCKS);
    if (prestage) {
        for (uint b = sg; b < total; b += KSPLIT) {
            const uint be = b / nbpr;
            const uint bb = b - be * nbpr;
            prestage_block(slm_inp + (size_t)b * 256, slm_isum + b * 8,
                           inputs_t + (size_t)be * input_len + (size_t)bb * 256, hh);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Contiguous chunk distribution over the KSPLIT sub-groups (better
     * per-thread DRAM locality: each sub-group walks one expert's K-blocks). */
    const uint per  = (total + KSPLIT - 1) / KSPLIT;
    const uint from = sg * per;
    const uint to   = min(from + per, total);
    float acc_rg[1];
    acc_rg[0] = 0.0f;

    for (uint idx = from; idx < to; idx++) {
        const uint e   = idx / nbpr;
        const uint bid = idx - e * nbpr;

        uint  index;
        float efficient;
        if (fast_route) {
            index     = indexes ? intel_sub_group_shuffle(idx_reg, e) : e;
            efficient = as_float(intel_sub_group_shuffle(eff_reg, e));
        } else {
            uint idx_lane = 0u;
            uint eff_lane = 0u;
            if (hh == 0) {
                idx_lane = indexes ? indexes_t[e] : e;
                eff_lane = as_uint(eff_t[e]);
            }
            index     = intel_sub_group_shuffle(idx_lane, 0);
            efficient = as_float(intel_sub_group_shuffle(eff_lane, 0));
        }

        const __global uchar *d_pqs = downs + (size_t)index * expert_size;
        const __global uint *pqs_u  = (const __global uint *)d_pqs;
        const __global uint *pqh_u  = (const __global uint *)(d_pqs + (size_t)num_blocks * 128);
        const __global uint *psl_pu = (const __global uint *)(d_pqs + (size_t)num_blocks * 160);

        const uint pe_u       = pe_base  + bid * pe_stride;
        const uint pqh_e      = pqh_base + bid * pqh_stride;
        const uint psl_pu_off = psl_base + bid * psl_stride;

        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);
        uint8 hq = intel_sub_group_block_read8(pqh_u + pqh_e);

        float isum[8], scale[8], minv[8];
        q5k_decode(psl_pu, psl_pu_off, hh, scale, minv);

        __local const slm_t *sb;
        if (prestage) {
            sb = slm_inp + (size_t)idx * 256;
            LOAD_ISUM_SLM(isum, slm_isum + idx * 8);
        } else {
            sb = sbase;
            load_inp_f16(sbase,
                         inputs_t + (size_t)e * input_len + (size_t)bid * 256, hh, isum);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        }

        acc_rg[0] = fma(q5k_acc(w0, w1, w2, w3, hq, scale, minv, isum, sb),
                        efficient, acc_rg[0]);
    }

#if KSPLIT == 1
    {
        uint row = h * OPG + hh;
        float o = acc_rg[0];
        if (is_acc) o += outputs[out_base + row];
        outputs[out_base + row] = o;
    }
#else
    slm_red[sg * OPG + hh] = acc_rg[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float o = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) o += slm_red[r * OPG + hh];
        uint row = h * OPG + hh;
        if (is_acc) o += outputs[out_base + row];
        outputs[out_base + row] = o;
    }
#endif
}