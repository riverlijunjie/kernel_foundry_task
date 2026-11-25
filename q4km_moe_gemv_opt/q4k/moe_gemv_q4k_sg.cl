/*
 * moe_gemv_q4k_sg.cl — High-performance Q4_K MoE GEMV with Intel sub-group
 *                      block reads (transposed "SG" weight layout).
 *
 * ============================ OPTIMIZATION NOTES =========================
 * Roofline reality check (Arc B580, 456 GB/s peak): a *pure read* kernel with
 * the identical dispatch + identical block-read pattern reaches 410 GB/s
 * (0.023 ms for the 9.4 MB gate_up working set).  This kernel therefore aims to
 * remove all ALU overhead from the de-quantisation inner loop so the GEMV stays
 * bandwidth/latency bound rather than ALU bound.  The final inner loop costs
 * ~3 SIMD16 instructions per weight:
 *
 *  1) BYTE-PARALLEL NIBBLE EXTRACTION.  A 4-byte weight word holds 8 nibbles;
 *     they are turned into ready-to-use floats without any int->float convert
 *     by exploiting the fp32 bit pattern (see the dequant core below).
 *
 *  2) VECTOR (float4) ACCUMULATORS.  8 independent float4 lanes keep 8 mads in
 *     flight instead of one serial dependency chain; activations are pulled
 *     from SLM as float4 (one d32x4 SLM load per 4 weights, uniform address).
 *
 *  3) ACTIVATION SUMS FOR FREE.  isum[j] (the -min*sum term of Q4_K) is
 *     produced by 8 sub_group_reduce_add on the staged activations instead of
 *     256 SLM re-reads + adds per block per matrix.
 *
 *  4) UP/GATE SPLIT ACROSS SUB-GROUPS (SPLIT_UG).  For KSPLIT>=2 the first half
 *     of the sub-groups reduce the UP matrix and the second half the GATE
 *     matrix, halving live weight registers (no spills) and doubling the
 *     in-flight loads per matrix.  The two partials meet in SLM at the single
 *     existing work-group barrier.
 *
 *  5) One intel_sub_group_block_read8 (512 B) for the 256 activations and one
 *     block_read4 (256 B) for the whole psl SoA scale record; the per-lane
 *     fields are recovered with sub-group shuffles.
 *
 * == Dequant core (fp32 bit pattern, no int->float convert) ==
 * For a 4-bit weight v,  as_float(0x3F800000 | (v << 19)) == 1.0f + v/16
 * exactly, so ONE shift + ONE bfn turns a nibble into a float.  The implied
 * +16 weight bias (after the x16 rescale) is removed for free using the
 * activation sum Q4_K already needs:
 *       acc += (16*scale)*dot' - (min + 16*scale)*isum
 * Using the mantissa (not a 2^23 bias) keeps the summands O(1), so the fp32
 * accumulation stays fully accurate.
 *
 * == Transposed ("SG") per-expert layout — SAME total size as original ==
 *   pqs_T:  num_blocks * 128 bytes   (2048 B / (h,bid) entry)
 *     Sub-block j at j*OPG*16, chunk c(0..3) at j*OPG*16 + c*OPG*4.
 *   psl_T:  num_blocks * 16 bytes (SoA):
 *     [ sl_u32xOPG | ml_u32xOPG | sh_u16xOPG | mh_u16xOPG | d_u16xOPG | dmin_u16xOPG ]
 *   expert_size = num_blocks * 144  (128 qs + 16 sl).
 *
 * == Dispatch ==
 *   group_up_gate_q4k_sg: global=(output_len,KSPLIT,output_num*token_len)
 *   down_merge_q4k_sg   : global=(output_len,KSPLIT,token_len)
 *   local=(OPG=16,KSPLIT,1)
 */

#pragma OPENCL EXTENSION cl_khr_fp16        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define OPG 16  /* = sub_group_size */

#ifndef KSPLIT
#define KSPLIT 1
#endif

/* Split the up/gate matrices across the two halves of the sub-groups
 * (requires an even KSPLIT; single-matrix fallback otherwise). */
#if KSPLIT >= 2 && (KSPLIT % 2) == 0
#define SPLIT_UG 1
#else
#define SPLIT_UG 0
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

/* ===========================================================================
 * ACTIVATION PRE-STAGING (ported from ocl_v2/moe_3gemm_swiglu_mlp.cl)
 * Stage the WHOLE activation window into SLM once, together with the per
 * sub-block activation sums (isum[]), behind a SINGLE work-group barrier; the
 * GEMV inner loop then only reads SLM (no re-stage, no re-reduce, no
 * per-iteration barrier), so IGC can keep every iteration's weight block-reads
 * in flight simultaneously.  If the window does not fit the SLM budget the loop
 * falls back to a streaming single-slot path.  The SLM pool is shared by the
 * two modes (max of both sizes), so pre-staging costs no extra SLM.
 * ========================================================================= */

/* Activation SLM budget, in 256-element blocks (1 block = 1 KiB as float). */
#ifndef SLM_ACT_BLOCKS
#define SLM_ACT_BLOCKS 16
#endif

/* Streaming fallback: a SINGLE 256-float slot per sub-group (a
 * sub_group_barrier orders each stage->use).  Keeping the pool flat is what
 * lets down_merge run at KSPLIT=16 (2048 live sub-groups) without growing SLM
 * past the pre-stage budget — doubling the in-flight sub-groups is the main
 * lever for this latency-bound (hot==cold) per-token GEMV. */
#define SLM_STREAM_FLOATS (KSPLIT * 256)
#if (SLM_ACT_BLOCKS * 256) > SLM_STREAM_FLOATS
#define SLM_POOL_FLOATS (SLM_ACT_BLOCKS * 256)
#else
#define SLM_POOL_FLOATS SLM_STREAM_FLOATS
#endif

/* Activations are staged in SLM as float; the dot product accumulates in fp32. */
typedef float slm_t;
#define XL4(p)                (*(__local const float4 *)(p))
#define SB_STORE(dst, f2)     (*(__local float2 *)(dst) = (f2))

#define FMASK   0x00780000u     /* fp32 mantissa slot for a 4-bit weight  */
#define FBIAS   0x3F800000u     /* fp32 1.0                               */

/* nibble of `q` at the shifted position -> float 1.0f + v/16 */
#define QW(shifted) as_float(((shifted) & FMASK) | FBIAS)

/* weights arrive biased by +16 and pre-divided by DOT_RESCALE:
 *   acc += (RESCALE*scale)*dot' - (min + 16*scale)*isum */
#define DOT_RESCALE 16.0f

#define Q4K_CHUNK(q, c)                                                       \
    do {                                                                      \
        uint _q = (q);                                                        \
        float4 _wl = (float4)(QW(_q << 19), QW(_q << 11),                     \
                              QW(_q <<  3), QW(_q >>  5));                     \
        float4 _wh = (float4)(QW(_q << 15), QW(_q <<  7),                     \
                              QW(_q >>  1), QW(_q >>  9));                     \
        vlo = fma(XL4(in + (c) * 4),      _wl, vlo);                          \
        vhi = fma(XL4(in + 16 + (c) * 4), _wh, vhi);                          \
    } while (0)

inline float q4k_subblock(uint q0, uint q1, uint q2, uint q3,
                          __local const slm_t *in)
{
    float4 vlo = (float4)(0.0f), vhi = (float4)(0.0f);
    Q4K_CHUNK(q0, 0); Q4K_CHUNK(q1, 1); Q4K_CHUNK(q2, 2); Q4K_CHUNK(q3, 3);
    float4 s = vlo + vhi;
    return (s.x + s.y) + (s.z + s.w);
}

/* Whole-block dot product: 8 sub-blocks, per-sub-block scale/min folded in. */
inline float q4k_acc(uint8 w0, uint8 w1, uint8 w2, uint8 w3,
                     const float *scale, const float *minv, const float *isum,
                     __local const slm_t *sb)
{
    float acc = 0.0f;
    acc = fma(scale[0], q4k_subblock(w0.s0, w0.s1, w0.s2, w0.s3, sb + 0 * 32), acc);
    acc = fma(scale[1], q4k_subblock(w0.s4, w0.s5, w0.s6, w0.s7, sb + 1 * 32), acc);
    acc = fma(scale[2], q4k_subblock(w1.s0, w1.s1, w1.s2, w1.s3, sb + 2 * 32), acc);
    acc = fma(scale[3], q4k_subblock(w1.s4, w1.s5, w1.s6, w1.s7, sb + 3 * 32), acc);
    acc = fma(scale[4], q4k_subblock(w2.s0, w2.s1, w2.s2, w2.s3, sb + 4 * 32), acc);
    acc = fma(scale[5], q4k_subblock(w2.s4, w2.s5, w2.s6, w2.s7, sb + 5 * 32), acc);
    acc = fma(scale[6], q4k_subblock(w3.s0, w3.s1, w3.s2, w3.s3, sb + 6 * 32), acc);
    acc = fma(scale[7], q4k_subblock(w3.s4, w3.s5, w3.s6, w3.s7, sb + 7 * 32), acc);
    unroll_for (int j = 0; j < 8; j++) acc = fma(-minv[j], isum[j], acc);
    return acc;
}

/* ---------------------------------------------------------------------------
 * psl record: ONE block_read4 pulls the complete 256-byte SoA record.
 *   component 0 -> sl[lane]        (bytes   0.. 63)
 *   component 1 -> ml[lane]        (bytes  64..127)
 *   component 2 -> sh[2l],sh[2l+1] for lane<8, mh[..] for lane>=8
 *   component 3 -> d [2l],d [2l+1] for lane<8, dmin[..] for lane>=8
 * The 16-bit fields are recovered with 4 sub-group shuffles.
 * ------------------------------------------------------------------------- */
inline void q4k_decode(const __global uint *psl_pu, uint off, uint hh,
                       float *scale, float *minv)
{
    uint4 r = intel_sub_group_block_read4(psl_pu + off);

    const uint half_sel = (hh & 1u) * 16u;      /* which 16-bit half */
    const uint src_lane = hh >> 1;              /* packing lane      */

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
        float sc = (float)sq * d;
        scale[i] = sc * DOT_RESCALE;
        minv[i]  = fma(16.0f, sc, (float)mq * dmin);
    }
}

/* ---------------------------------------------------------------------------
 * Stage 256 fp16 activations into SLM and produce the 8 per-sub-block sums.
 * ONE 512-byte sub-group block read: component j / lane l holds the fp16 pair
 * (32j+2l, 32j+2l+1), i.e. component j covers exactly sub-block j, so
 *       isum[j] = sub_group_reduce_add(lo + hi).
 * ------------------------------------------------------------------------- */
inline void load_inp_f16(__local slm_t *sb, const __global half *src, uint hh,
                         float *isum)
{
    uint8 iv = intel_sub_group_block_read8((const __global uint *)src);
#define Q4K_STAGE(j, comp)                                                   \
    {                                                                        \
        float2 _f = convert_float2(as_half2(comp));                          \
        SB_STORE(sb + (j) * 32 + 2 * hh, _f);                                \
        isum[j] = sub_group_reduce_add(_f.x + _f.y);                         \
    }
    Q4K_STAGE(0, iv.s0) Q4K_STAGE(1, iv.s1) Q4K_STAGE(2, iv.s2) Q4K_STAGE(3, iv.s3)
    Q4K_STAGE(4, iv.s4) Q4K_STAGE(5, iv.s5) Q4K_STAGE(6, iv.s6) Q4K_STAGE(7, iv.s7)
#undef Q4K_STAGE
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
#define LOAD_ISUM_SLM(isum, isum_slm)                                         \
    do {                                                                      \
        float4 _i0 = *(__local const float4 *)(isum_slm);                     \
        float4 _i1 = *(__local const float4 *)((isum_slm) + 4);               \
        (isum)[0] = _i0.x; (isum)[1] = _i0.y; (isum)[2] = _i0.z; (isum)[3] = _i0.w; \
        (isum)[4] = _i1.x; (isum)[5] = _i1.y; (isum)[6] = _i1.z; (isum)[7] = _i1.w; \
    } while (0)

/* =======================================================================
 * Kernel 1: group_up_gate_q4k_sg
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void group_up_gate_q4k_sg(
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
    const uint expert_size = num_blocks * 144;  /* 128 qs + 16 sl */

    const uint h  = get_group_id(0);   /* row-group over output_len/OPG */
    const uint hh = get_local_id(0);   /* lane = output row within group */
    const uint sg = get_local_id(1);   /* K-split sub-group index         */
    const uint vt = get_group_id(2);
    const uint v  = vt % output_num;
    const uint t  = vt / output_num;

    const uint index = indexes ? indexes[t * aligned_output_num + v] : v;

    const __global uchar *up_pqs  = ups   + (size_t)index * expert_size;
    const __global uchar *gt_pqs  = gates + (size_t)index * expert_size;

#if SPLIT_UG
    /* --- sub-groups [0, KSPLIT/2) reduce UP, [KSPLIT/2, KSPLIT) reduce GATE --- */
    #define KHALF (KSPLIT / 2)
    const uint is_gate = sg >= KHALF;
    const uint sg_k    = sg - is_gate * KHALF;

    __local slm_t slm_inp[SLM_POOL_FLOATS];
    __local float slm_isum[SLM_ACT_BLOCKS * 8];
    __local float slm_red[2 * KHALF * OPG];
    __local slm_t *sbase = slm_inp + sg * 256;

    const __global uchar *base   = is_gate ? gt_pqs : up_pqs;
    const __global uint  *pqs_u  = (const __global uint *)base;
    const __global uint  *psl_pu = (const __global uint *)(base + (size_t)num_blocks * 128);

    const bool prestage = (nbpr <= SLM_ACT_BLOCKS);
    if (prestage) {
        for (uint b = sg; b < nbpr; b += KSPLIT)
            prestage_block(slm_inp + (size_t)b * 256, slm_isum + b * 8,
                           inputs + (size_t)t * input_len + (size_t)b * 256, hh);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float acc = 0.0f;
    for (uint bid = sg_k; bid < nbpr; bid += KHALF) {
        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);

        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);

        float isum[8], scale[8], minv[8];
        q4k_decode(psl_pu, psl_pu_off, hh, scale, minv);

        __local const slm_t *sb;
        if (prestage) {
            sb = slm_inp + (size_t)bid * 256;          /* already resident */
            LOAD_ISUM_SLM(isum, slm_isum + bid * 8);
        } else {
            sb = sbase;
            load_inp_f16(sbase,
                         inputs + (size_t)t * input_len + (size_t)bid * 256, hh, isum);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        }

        acc += q4k_acc(w0, w1, w2, w3, scale, minv, isum, sb);
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
    /* --- every sub-group reduces both matrices (KSPLIT == 1 fallback) --- */
    __local slm_t slm_inp [SLM_POOL_FLOATS];
    __local float slm_isum[SLM_ACT_BLOCKS * 8];
    __local float slm_up  [KSPLIT * OPG];
    __local float slm_gate[KSPLIT * OPG];
    __local slm_t *sbase = slm_inp + sg * 256;

    const __global uint *up_pqs_u  = (const __global uint *)up_pqs;
    const __global uint *up_psl_pu = (const __global uint *)(up_pqs + (size_t)num_blocks * 128);
    const __global uint *gt_pqs_u  = (const __global uint *)gt_pqs;
    const __global uint *gt_psl_pu = (const __global uint *)(gt_pqs + (size_t)num_blocks * 128);

    const bool prestage = (nbpr <= SLM_ACT_BLOCKS);
    if (prestage) {
        for (uint b = sg; b < nbpr; b += KSPLIT)
            prestage_block(slm_inp + (size_t)b * 256, slm_isum + b * 8,
                           inputs + (size_t)t * input_len + (size_t)b * 256, hh);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float up_acc = 0.0f, gate_acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);

        uint8 uw0 = intel_sub_group_block_read8(up_pqs_u + pe_u +   0);
        uint8 uw1 = intel_sub_group_block_read8(up_pqs_u + pe_u + 128);
        uint8 uw2 = intel_sub_group_block_read8(up_pqs_u + pe_u + 256);
        uint8 uw3 = intel_sub_group_block_read8(up_pqs_u + pe_u + 384);
        uint8 gw0 = intel_sub_group_block_read8(gt_pqs_u + pe_u +   0);
        uint8 gw1 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 128);
        uint8 gw2 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 256);
        uint8 gw3 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 384);

        float isum[8], uscale[8], uminv[8], gscale[8], gminv[8];
        q4k_decode(up_psl_pu, psl_pu_off, hh, uscale, uminv);
        q4k_decode(gt_psl_pu, psl_pu_off, hh, gscale, gminv);

        __local const slm_t *sb;
        if (prestage) {
            sb = slm_inp + (size_t)bid * 256;
            LOAD_ISUM_SLM(isum, slm_isum + bid * 8);
        } else {
            sb = sbase;
            load_inp_f16(sbase,
                         inputs + (size_t)t * input_len + (size_t)bid * 256, hh, isum);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        }

        up_acc   += q4k_acc(uw0, uw1, uw2, uw3, uscale, uminv, isum, sb);
        gate_acc += q4k_acc(gw0, gw1, gw2, gw3, gscale, gminv, isum, sb);
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
 * Kernel 2: down_merge_q4k_sg
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void down_merge_q4k_sg(
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
    const uint expert_size = num_blocks * 144;

    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint t  = get_group_id(2);

    __local slm_t slm_inp[SLM_POOL_FLOATS];
    __local float slm_isum[SLM_ACT_BLOCKS * 8];
    __local float slm_red[KSPLIT * OPG];
    __local slm_t *sbase = slm_inp + sg * 256;

    /* ---- Hoisted routing: ONE 64-byte sub-group block read gives lane l the
     * routing info of expert l; the loop recovers its own with a shuffle
     * (register op) instead of a serial dependent scalar load per iteration.
     * Falls back to scalar loads when input_num > OPG. ---- */
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

    /* ---- one-shot activation pre-staging over the whole [input_num,
     * input_len] block set (`total` blocks). ---- */
    const bool prestage = (total <= SLM_ACT_BLOCKS);
    if (prestage) {
        for (uint b = sg; b < total; b += KSPLIT) {
            const uint be = b / nbpr;
            const uint bb = b - be * nbpr;
            prestage_block(slm_inp + (size_t)b * 256, slm_isum + b * 8,
                           inputs + (size_t)t * input_num * input_len
                           + (size_t)be * input_len + (size_t)bb * 256, hh);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Contiguous chunk distribution over the KSPLIT sub-groups: each sub-group
     * walks the K-blocks of ONE expert (weight slices are expert_size apart),
     * for better per-thread DRAM locality. */
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

        const __global uchar *d_pqs = downs + (size_t)index * expert_size;
        const __global uint *d_pqs_u  = (const __global uint *)d_pqs;
        const __global uint *d_psl_pu = (const __global uint *)(d_pqs + (size_t)num_blocks * 128);

        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);

        uint8 w0 = intel_sub_group_block_read8(d_pqs_u + pe_u +   0);
        uint8 w1 = intel_sub_group_block_read8(d_pqs_u + pe_u + 128);
        uint8 w2 = intel_sub_group_block_read8(d_pqs_u + pe_u + 256);
        uint8 w3 = intel_sub_group_block_read8(d_pqs_u + pe_u + 384);

        float isum[8], scale[8], minv[8];
        q4k_decode(d_psl_pu, psl_pu_off, hh, scale, minv);

        __local const slm_t *sb;
        if (prestage) {
            sb = slm_inp + (size_t)idx * 256;
            LOAD_ISUM_SLM(isum, slm_isum + idx * 8);
        } else {
            sb = sbase;
            load_inp_f16(sbase,
                         inputs + (size_t)t * input_num * input_len
                         + (size_t)e * input_len + (size_t)bid * 256, hh, isum);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        }

        acc = fma(q4k_acc(w0, w1, w2, w3, scale, minv, isum, sb), efficient, acc);
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
