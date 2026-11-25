/*
 * moe_gemv_q5k_sg.cl — High-performance Q5_K MoE GEMV with Intel sub-group
 *                      block reads (transposed "SG" weight layout).
 *
 * ============================ OPTIMIZATION NOTES =========================
 * A pure-read kernel with this dispatch and these block reads sustains
 * ~410 GB/s on Arc B580 (0.028 ms for the 11.5 MB gate_up working set), while
 * the previous version of this kernel needed 0.076-0.084 ms: it was ALU bound
 * in the de-quantisation loop, not bandwidth bound.  The old inner loop spent
 * ~8 SIMD16 instructions per 5-bit weight (shr+and for the nibble,
 * shr+and+shl+or for the 5th bit, int->float convert, mad) plus a redundant
 * 256-add activation sum per block per matrix.
 *
 * The rewrite (same numerics, same memory layout) costs ~2.4:
 *
 *  1) PACKED PAIR WORDS.  Two weights that live 16 bits apart in the qs word
 *     are extracted with a single AND (0x000F000F) -- and, crucially, their
 *     two 5th bits are ALSO 16 bits apart in the 32-bit qh word (bit r*8+c for
 *     weight r of chunk c), so ONE shift + one masked OR completes BOTH 5-bit
 *     weights.  The pair then converts to float with a single
 *     "mov (16) dst:f src<2;1,0>:uw" for the low half and shr + mov for the
 *     high half: 5.5 ops per 2 weights instead of 16.
 *
 *  2) VECTOR ACCUMULATORS.  32 mads per sub-block used to form ONE serial
 *     dependency chain; they are now spread over 8 independent float4 lanes,
 *     so the FP pipe always has work in flight.  Activations come from SLM as
 *     float4 (one d32x4 SLM load per 4 weights, uniform address).
 *
 *  3) ACTIVATION SUMS FOR FREE.  The 256 activations arrive in ONE
 *     intel_sub_group_block_read8; component j / lane l holds the fp16 pair
 *     (32j+2l, 32j+2l+1), i.e. component j covers exactly sub-block j, so the
 *     8 isum[] values are 8 sub-group reductions instead of 2048 SLM
 *     loads + adds per block.
 *
 *  4) UP/GATE SPLIT ACROSS SUB-GROUPS (SPLIT_UG): the first half of the
 *     sub-groups reduce the UP matrix, the second half the GATE matrix.  This
 *     halves the live weight registers (the old kernel spilled), doubles the
 *     in-flight loads per matrix, and the two halves meet in SLM at the single
 *     work-group barrier that was already there.
 *
 *  5) The psl scale record is fetched with ONE block_read4 (the whole 256-byte
 *     SoA record) instead of 6 narrow block reads; the per-lane 16-bit fields
 *     are recovered with 4 sub-group shuffles.
 *
 * == Transposed ("SG") per-expert layout — SAME total size as original ==
 * Identical to pack_q5k_sg().  nbpr = K/256, nrg = N/OPG:
 *   pqs_T:  num_blocks * 128 bytes  (low nibbles, same layout as Q4_K pqs_T)
 *     Entry (h,bid) at (h*nbpr+bid)*OPG*128, sub-block j at j*OPG*16,
 *     chunk c(0..3) at j*OPG*16 + c*OPG*4.
 *   pqh_T:  num_blocks * 32 bytes   (1 high bit / weight)
 *     Entry (h,bid) at (h*nbpr+bid)*OPG*32, sub-block j's 4-byte qh word for
 *     all OPG lanes at j*OPG*4 (lane lid at j*OPG*4 + lid*4).
 *   psl_T:  num_blocks * 16 bytes   (SoA, identical to Q4_K psl_T)
 *   expert_size = num_blocks * 176  (128 qs + 32 qh + 16 sl) — unchanged.
 *
 * == Dispatch (unchanged) ==
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

#ifndef SPLIT_UG
#if KSPLIT >= 2 && (KSPLIT % 2) == 0
#define SPLIT_UG 1
#else
#define SPLIT_UG 0
#endif
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

/* down_merge work distribution over the KSPLIT sub-groups: 0 = round robin,
 * 1 = contiguous chunks (better per-thread DRAM locality: a sub-group then
 * walks the K-blocks of ONE expert instead of hopping between experts, whose
 * weight slices are expert_size apart). */
#ifndef DM_BLOCKED
#define DM_BLOCKED 1
#endif

/* down_merge: fetch the whole routing table with one sub-group block read and
 * recover each expert's entry with a shuffle, instead of a dependent scalar
 * load per iteration. */
#ifndef DM_HOIST_ROUTE
#define DM_HOIST_ROUTE 1
#endif

#define W2_MASK 0x000F000Fu   /* two 4-bit weights, 16 bit apart          */
#define QH_MASK 0x00100010u   /* their two 5th bits, also 16 bit apart    */

/* ---------------------------------------------------------------------------
 * One sub-block (32 weights): 4 qs chunk words + the sub-block's 32-bit qh
 * word + 32 SLM activations.  For chunk c and byte r:
 *     weight (   4c+r) = (q >> (8r    )) & 0xF | ((H >> (r*8 + c    )) & 1) << 4
 *     weight (16+4c+r) = (q >> (8r + 4)) & 0xF | ((H >> (r*8 + c + 4)) & 1) << 4
 * The pairs (r=0,r=2) and (r=1,r=3) are 16 bits apart in BOTH q and H, so each
 * pair costs 1 and + 1 shift + 1 masked or.
 * ------------------------------------------------------------------------- */
#define Q5K_CHUNK(q, H, c)                                                    \
    do {                                                                      \
        uint _q = (q), _H = (H);                                              \
        uint _a = ( _q        & W2_MASK) | ((_H << (4 - (c))) & QH_MASK);      \
        uint _b = ((_q >>  8) & W2_MASK) | ((_H >> (4 + (c))) & QH_MASK);      \
        uint _c = ((_q >>  4) & W2_MASK) | ((_H >>      (c) ) & QH_MASK);      \
        uint _d = ((_q >> 12) & W2_MASK) | ((_H >> (8 + (c))) & QH_MASK);      \
        float4 _wl = (float4)((float)as_ushort2(_a).x, (float)as_ushort2(_b).x,\
                              (float)(_a >> 16),       (float)(_b >> 16));     \
        float4 _wh = (float4)((float)as_ushort2(_c).x, (float)as_ushort2(_d).x,\
                              (float)(_c >> 16),       (float)(_d >> 16));     \
        vlo = fma(*(__local const float4 *)(in + (c) * 4),      _wl, vlo);      \
        vhi = fma(*(__local const float4 *)(in + 16 + (c) * 4), _wh, vhi);      \
    } while (0)

inline float q5k_subblock(uint q0, uint q1, uint q2, uint q3, uint H,
                          __local const float *in)
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
                     __local const float *sb)
{
    float acc = 0.0f;
    acc = fma(scale[0], q5k_subblock(w0.s0, w0.s1, w0.s2, w0.s3, hq.s0, sb + 0 * 32), acc);
    acc = fma(scale[1], q5k_subblock(w0.s4, w0.s5, w0.s6, w0.s7, hq.s1, sb + 1 * 32), acc);
    acc = fma(scale[2], q5k_subblock(w1.s0, w1.s1, w1.s2, w1.s3, hq.s2, sb + 2 * 32), acc);
    acc = fma(scale[3], q5k_subblock(w1.s4, w1.s5, w1.s6, w1.s7, hq.s3, sb + 3 * 32), acc);
    acc = fma(scale[4], q5k_subblock(w2.s0, w2.s1, w2.s2, w2.s3, hq.s4, sb + 4 * 32), acc);
    acc = fma(scale[5], q5k_subblock(w2.s4, w2.s5, w2.s6, w2.s7, hq.s5, sb + 5 * 32), acc);
    acc = fma(scale[6], q5k_subblock(w3.s0, w3.s1, w3.s2, w3.s3, hq.s6, sb + 6 * 32), acc);
    acc = fma(scale[7], q5k_subblock(w3.s4, w3.s5, w3.s6, w3.s7, hq.s7, sb + 7 * 32), acc);
    unroll_for (int j = 0; j < 8; j++) acc = fma(-minv[j], isum[j], acc);
    return acc;
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
inline void load_inp_f16(__local float *sb, const __global half *src, uint hh,
                         float *isum)
{
    uint8 iv = intel_sub_group_block_read8((const __global uint *)src);
#define Q5K_STAGE(j, comp)                                                   \
    {                                                                        \
        float2 _f = convert_float2(as_half2(comp));                           \
        *(__local float2 *)(sb + (j) * 32 + 2 * hh) = _f;                     \
        isum[j] = sub_group_reduce_add(_f.x + _f.y);                          \
    }
    Q5K_STAGE(0, iv.s0) Q5K_STAGE(1, iv.s1) Q5K_STAGE(2, iv.s2) Q5K_STAGE(3, iv.s3)
    Q5K_STAGE(4, iv.s4) Q5K_STAGE(5, iv.s5) Q5K_STAGE(6, iv.s6) Q5K_STAGE(7, iv.s7)
#undef Q5K_STAGE
}

/* =======================================================================
 * Kernel 1: group_up_gate_q5k_sg
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
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

    const uint index = indexes ? indexes[t * aligned_output_num + v] : v;

    const __global uchar *up_pqs = ups   + (size_t)index * expert_size;
    const __global uchar *gt_pqs = gates + (size_t)index * expert_size;

#if SPLIT_UG
    #define KHALF (KSPLIT / 2)
    const uint is_gate = sg >= KHALF;
    const uint sg_k    = sg - is_gate * KHALF;

    __local float slm_inp[KSPLIT * 2 * 256];
    __local float slm_red[2 * KHALF * OPG];
    __local float *sbase = slm_inp + sg * (2 * 256);

    const __global uchar *base  = is_gate ? gt_pqs : up_pqs;
    const __global uint *pqs_u  = (const __global uint *)base;
    const __global uint *pqh_u  = (const __global uint *)(base + (size_t)num_blocks * 128);
    const __global uint *psl_pu = (const __global uint *)(base + (size_t)num_blocks * 160);

    float acc = 0.0f;
    for (uint bid = sg_k; bid < nbpr; bid += KHALF) {
        __local float *sb = sbase + (bid & 1u) * 256;

        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e      = (h * nbpr + bid) * (OPG *  32 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG *  16 / 4);

        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);
        uint8 hq = intel_sub_group_block_read8(pqh_u + pqh_e);

        float isum[8], scale[8], minv[8];
        q5k_decode(psl_pu, psl_pu_off, hh, scale, minv);
        load_inp_f16(sb, inputs + (size_t)t * input_len + (size_t)bid * 256, hh, isum);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        acc += q5k_acc(w0, w1, w2, w3, hq, scale, minv, isum, sb);
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
    __local float slm_inp [KSPLIT * 2 * 256];
    __local float slm_up  [KSPLIT * OPG];
    __local float slm_gate[KSPLIT * OPG];
    __local float *sbase = slm_inp + sg * (2 * 256);

    const __global uint *up_pqs_u  = (const __global uint *)up_pqs;
    const __global uint *up_pqh_u  = (const __global uint *)(up_pqs + (size_t)num_blocks * 128);
    const __global uint *up_psl_pu = (const __global uint *)(up_pqs + (size_t)num_blocks * 160);
    const __global uint *gt_pqs_u  = (const __global uint *)gt_pqs;
    const __global uint *gt_pqh_u  = (const __global uint *)(gt_pqs + (size_t)num_blocks * 128);
    const __global uint *gt_psl_pu = (const __global uint *)(gt_pqs + (size_t)num_blocks * 160);

    float up_acc = 0.0f, gate_acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float *sb = sbase + (bid & 1u) * 256;

        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e      = (h * nbpr + bid) * (OPG *  32 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG *  16 / 4);

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
        load_inp_f16(sb, inputs + (size_t)t * input_len + (size_t)bid * 256, hh, isum);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        up_acc   += q5k_acc(uw0, uw1, uw2, uw3, uhq, uscale, uminv, isum, sb);
        gate_acc += q5k_acc(gw0, gw1, gw2, gw3, ghq, gscale, gminv, isum, sb);
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
 * Kernel 2: down_merge_q5k_sg
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
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

    __local float slm_inp[KSPLIT * 2 * 256];
    __local float slm_red[KSPLIT * OPG];
    __local float *sbase = slm_inp + sg * (2 * 256);


    /* ---- Hoisted routing --------------------------------------------------
     * The per-iteration  indexes[]/efficients[]  scalar loads were a *serial*
     * dependency: load index -> compute the expert base address -> load the
     * weights, i.e. two round trips before any weight byte could be requested.
     * ONE 64-byte sub-group block read instead gives lane l the routing info of
     * expert l, and the loop recovers its own with a shuffle (register op). */
    const bool fast_route = DM_HOIST_ROUTE && (input_num <= OPG)
                            && (aligned_input_num >= OPG);
    uint  idx_reg = 0u;
    uint  eff_reg = 0u;
    if (fast_route) {
        if (indexes)
            idx_reg = intel_sub_group_block_read(indexes + (size_t)t * aligned_input_num);
        eff_reg = intel_sub_group_block_read(
                      (const __global uint *)(efficients + (size_t)t * aligned_input_num));
    }
    const uint total = input_num * nbpr;
#if DM_BLOCKED
    const uint per  = (total + KSPLIT - 1) / KSPLIT;
    const uint from = sg * per;
    const uint to   = min(from + per, total);
#endif
    float acc = 0.0f;

#if DM_BLOCKED
    for (uint idx = from; idx < to; idx++) {
#else
    for (uint idx = sg; idx < total; idx += KSPLIT) {
#endif
        const uint e   = idx / nbpr;
        const uint bid = idx - e * nbpr;
        __local float *sb = sbase + (idx & 1u) * 256;

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
        const __global uint *pqs_u  = (const __global uint *)d_pqs;
        const __global uint *pqh_u  = (const __global uint *)(d_pqs + (size_t)num_blocks * 128);
        const __global uint *psl_pu = (const __global uint *)(d_pqs + (size_t)num_blocks * 160);

        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e      = (h * nbpr + bid) * (OPG *  32 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG *  16 / 4);

        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);
        uint8 hq = intel_sub_group_block_read8(pqh_u + pqh_e);

        float isum[8], scale[8], minv[8];
        q5k_decode(psl_pu, psl_pu_off, hh, scale, minv);
        load_inp_f16(sb, inputs + (size_t)t * input_num * input_len
                     + (size_t)e * input_len + (size_t)bid * 256, hh, isum);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        acc = fma(q5k_acc(w0, w1, w2, w3, hq, scale, minv, isum, sb), efficient, acc);
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
