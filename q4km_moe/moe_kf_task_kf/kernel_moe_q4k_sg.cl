/*
 * kernel_moe_q4k_sg.cl — High-performance Q4_K MoE GEMV with Intel sub-group
 *                        block reads (transposed "SG" weight layout).
 *
 * Two kernels, used together to implement one MoE-FFN block (top-K routed
 * experts, K/8/256-active routing):
 *   group_up_gate_q4k_sg — fused up-proj + gate-proj GEMV + SiLU(gate)*up
 *   down_merge_q4k_sg    — down-proj GEMV, weighted-sum merged over the
 *                          active experts (accumulate into outputs)
 *
 * == Transposed ("SG") per-expert layout — SAME total size as the plain
 *    shuffled Q4_K layout ==
 * For an expert weight matrix of [output_len(N), input_len(K)] with
 * nbpr = K/256 K-blocks per row and nrg = N/OPG row-groups:
 *
 *   pqs_T:  num_blocks * 128 bytes
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 128 bytes (2048 B)
 *     Sub-block j at j*OPG*16, chunk c (0..3) at j*OPG*16 + c*OPG*4.
 *   psl_T:  num_blocks * 16 bytes  (SoA)
 *     Entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes (256 B):
 *       [ sl_u32×OPG=64B | ml_u32×OPG=64B | sh_u16×OPG=32B |
 *         mh_u16×OPG=32B | d_u16×OPG=32B  | dmin_u16×OPG=32B ]
 *   expert_size = num_blocks * 144  (128 qs + 16 sl) — unchanged.
 *
 * REQUIRED (host dispatch in task.py depends on them):
 *   1. Entry-point names and argument order must remain EXACTLY:
 *        group_up_gate_q4k_sg(inputs, ups, gates, indexes, outputs,
 *                              output_num, aligned_output_num, token_len,
 *                              input_len, output_len)
 *        down_merge_q4k_sg(inputs, downs, indexes, efficients, outputs,
 *                           input_num, aligned_input_num, token_len,
 *                           input_len, output_len, is_acc)
 *   2. OPG must remain 16 (task.py hardcodes sub_group_size = 16 for the
 *      routing-index alignment and the local dispatch size).
 *   3. Dispatch (host side, fixed by task.py):
 *        group_up_gate_q4k_sg:
 *          global = (output_len, KSPLIT, output_num*token_len)
 *          local  = (OPG=16,    KSPLIT, 1)
 *        down_merge_q4k_sg:
 *          global = (output_len, KSPLIT, token_len)
 *          local  = (OPG=16,    KSPLIT, 1)
 *      KSPLIT is supplied at build time as -DKSPLIT=<n> (n = 1,2,4,8,...);
 *      the kernel must keep working correctly for any such n (task.py tests
 *      several values).
 */

// [USER_INSTRUCTIONS_START]
//
// Optimize MoE GEMV tips:
// 1. These kernels are memory-bandwidth bound (de-quantizing 4-bit weights);
//    focus on maximizing effective DRAM bandwidth and minimizing ALU work
//    per weight, not on reducing FLOPs.
// 2. Use intel_sub_group_block_read / block_read8 for wide, coalesced loads
//    of the packed weight bytes and the psl scale/min record.
// 3. Use sub-group shuffles to recover per-lane scalar fields from a single
//    wide block read instead of many narrow reads.
// 4. Minimize live registers per matrix (avoid spills) — consider splitting
//    the up/gate reduction work across sub-groups (SPLIT_UG-style) so both
//    weight vectors are never simultaneously live in the same thread.
// 5. Compute the Q4_K activation-sum term (needed for the -min*sum part)
//    from the same wide activation load used for the dot product, instead
//    of a second pass over the staged activations.
// 6. Keep KSPLIT working generically: the kernel is built with varying
//    -DKSPLIT=<n> to trade reduction-depth for occupancy on small dispatches
//    (few experts/output rows); do not hardcode assumptions that break for
//    KSPLIT > 1 or KSPLIT == 1.
//
// [USER_INSTRUCTIONS_END]

#pragma OPENCL EXTENSION cl_khr_fp16        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

/* [EVOLVE_START] */
#define OPG 16  /* = sub_group_size */

#ifndef KSPLIT
#define KSPLIT 1
#endif

/* Split the up/gate matrices across the two halves of the sub-groups.
 * Requires an even KSPLIT; falls back automatically otherwise. */
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

/* ===========================================================================
 * De-quantisation core.
 * ========================================================================= */
#ifndef ACC_MODE
#define ACC_MODE 0
#endif

#define W2_MASK 0x000F000Fu     /* two 4-bit weights, 16 bit apart        */
#define FMASK   0x00780000u     /* fp32 mantissa slot for a 4-bit weight  */
#define FBIAS   0x3F800000u     /* fp32 1.0                               */

#if ACC_MODE == 2
/* --------------- fp32 bit pattern (no convert) -- default --------------- */
typedef float slm_t;
#define WEIGHT_BIAS16 1
#define DOT_RESCALE   16.0f     /* weights arrive as (16 + v)/16 */

/* nibble of `q` at bit b -> float 1.0f + v/16 (mantissa bits 19..22) */
#define QW(shifted) as_float(((shifted) & FMASK) | FBIAS)

#define Q4K_CHUNK(q, c)                                                       \
    do {                                                                      \
        uint _q = (q);                                                         \
        float4 _wl = (float4)(QW(_q << 19), QW(_q << 11),                      \
                              QW(_q <<  3), QW(_q >>  5));                     \
        float4 _wh = (float4)(QW(_q << 15), QW(_q <<  7),                      \
                              QW(_q >>  1), QW(_q >>  9));                     \
        vlo = fma(*(__local const float4 *)(in + (c) * 4),      _wl, vlo);      \
        vhi = fma(*(__local const float4 *)(in + 16 + (c) * 4), _wh, vhi);      \
    } while (0)

inline float q4k_subblock(uint q0, uint q1, uint q2, uint q3,
                          __local const float *in)
{
    float4 vlo = (float4)(0.0f), vhi = (float4)(0.0f);
    Q4K_CHUNK(q0, 0); Q4K_CHUNK(q1, 1); Q4K_CHUNK(q2, 2); Q4K_CHUNK(q3, 3);
    float4 s = vlo + vhi;
    return (s.x + s.y) + (s.z + s.w);
}

#else
/* ------------------------ exact fp32 conversion ------------------------ */
typedef float slm_t;
#define WEIGHT_BIAS16 0
#define DOT_RESCALE   1.0f

#define Q4K_CHUNK(q, c)                                                       \
    do {                                                                      \
        uint _q = (q);                                                         \
        uint _a =  _q        & W2_MASK;   /* lo0 | lo2 << 16 */                \
        uint _b = (_q >>  8) & W2_MASK;   /* lo1 | lo3 << 16 */                \
        uint _c = (_q >>  4) & W2_MASK;   /* hi0 | hi2 << 16 */                \
        uint _d = (_q >> 12) & W2_MASK;   /* hi1 | hi3 << 16 */                \
        float4 _wl = (float4)((float)as_ushort2(_a).x, (float)as_ushort2(_b).x, \
                              (float)(_a >> 16),       (float)(_b >> 16));      \
        float4 _wh = (float4)((float)as_ushort2(_c).x, (float)as_ushort2(_d).x, \
                              (float)(_c >> 16),       (float)(_d >> 16));      \
        vlo = fma(*(__local const float4 *)(in + (c) * 4),      _wl, vlo);      \
        vhi = fma(*(__local const float4 *)(in + 16 + (c) * 4), _wh, vhi);      \
    } while (0)

inline float q4k_subblock(uint q0, uint q1, uint q2, uint q3,
                          __local const float *in)
{
    float4 vlo = (float4)(0.0f), vhi = (float4)(0.0f);
    Q4K_CHUNK(q0, 0); Q4K_CHUNK(q1, 1); Q4K_CHUNK(q2, 2); Q4K_CHUNK(q3, 3);
    float4 s = vlo + vhi;
    return (s.x + s.y) + (s.z + s.w);
}
#endif

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
#if WEIGHT_BIAS16
        scale[i] = sc * DOT_RESCALE;
        minv[i]  = fma(16.0f, sc, (float)mq * dmin);
#else
        scale[i] = sc;
        minv[i]  = (float)mq * dmin;
#endif
    }
}

/* ---------------------------------------------------------------------------
 * Stage 256 fp16 activations into SLM and produce the 8 per-sub-block
 * activation sums with ONE 512-byte sub-group block read.
 * ------------------------------------------------------------------------- */
inline void load_inp_f16(__local slm_t *sb, const __global half *src, uint hh,
                         float *isum)
{
    uint8 iv = intel_sub_group_block_read8((const __global uint *)src);
#define Q4K_STAGE(j, comp)                                                   \
    {                                                                        \
        float2 _f = convert_float2(as_half2(comp));                          \
        *(__local float2 *)(sb + (j) * 32 + 2 * hh) = _f;                     \
        isum[j] = sub_group_reduce_add(_f.x + _f.y);                          \
    }
    Q4K_STAGE(0, iv.s0) Q4K_STAGE(1, iv.s1) Q4K_STAGE(2, iv.s2) Q4K_STAGE(3, iv.s3)
    Q4K_STAGE(4, iv.s4) Q4K_STAGE(5, iv.s5) Q4K_STAGE(6, iv.s6) Q4K_STAGE(7, iv.s7)
#undef Q4K_STAGE
}

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

    __local slm_t slm_inp[KSPLIT * 2 * 256];
    __local float slm_red[2 * KHALF * OPG];
    __local slm_t *sbase = slm_inp + sg * (2 * 256);

    const __global uchar *base   = is_gate ? gt_pqs : up_pqs;
    const __global uint  *pqs_u  = (const __global uint *)base;
    const __global uint  *psl_pu = (const __global uint *)(base + (size_t)num_blocks * 128);

    float acc = 0.0f;
    for (uint bid = sg_k; bid < nbpr; bid += KHALF) {
        __local slm_t *sb = sbase + (bid & 1u) * 256;

        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);

        /* ---- issue all long-latency global loads first ---- */
        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);

        float isum[8], scale[8], minv[8];
        q4k_decode(psl_pu, psl_pu_off, hh, scale, minv);
        load_inp_f16(sb, inputs + (size_t)t * input_len + (size_t)bid * 256, hh, isum);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

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
    __local slm_t slm_inp [KSPLIT * 2 * 256];
    __local float slm_up  [KSPLIT * OPG];
    __local float slm_gate[KSPLIT * OPG];
    __local slm_t *sbase = slm_inp + sg * (2 * 256);

    const __global uint *up_pqs_u  = (const __global uint *)up_pqs;
    const __global uint *up_psl_pu = (const __global uint *)(up_pqs + (size_t)num_blocks * 128);
    const __global uint *gt_pqs_u  = (const __global uint *)gt_pqs;
    const __global uint *gt_psl_pu = (const __global uint *)(gt_pqs + (size_t)num_blocks * 128);

    float up_acc = 0.0f, gate_acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local slm_t *sb = sbase + (bid & 1u) * 256;

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
        load_inp_f16(sb, inputs + (size_t)t * input_len + (size_t)bid * 256, hh, isum);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

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

    __local slm_t slm_inp[KSPLIT * 2 * 256];
    __local float slm_red[KSPLIT * OPG];
    __local slm_t *sbase = slm_inp + sg * (2 * 256);


    /* ---- Hoisted routing --------------------------------------------------
     * ONE 64-byte sub-group block read gives lane l the routing info of
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
        __local slm_t *sb = sbase + (idx & 1u) * 256;

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
        load_inp_f16(sb, inputs + (size_t)t * input_num * input_len
                     + (size_t)e * input_len + (size_t)bid * 256, hh, isum);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

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
/* [EVOLVE_END] */
