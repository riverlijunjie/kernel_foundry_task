/*
 * moe_gemv_q6k_sg.cl — High-performance Q6_K MoE GEMV with Intel sub-group
 *                      block reads (transposed "SG" weight layout).
 *
 * ============================ OPTIMIZATION NOTES =========================
 * A pure-read kernel with this dispatch and these block reads sustains
 * ~410 GB/s on Arc B580 (0.034 ms for the 13.8 MB gate_up working set), while
 * the previous version needed 0.092-0.115 ms: it was ALU bound in the
 * de-quantisation loop, not bandwidth bound.  The old inner loop spent ~8
 * SIMD16 instructions per 6-bit weight (shr+and for the nibble, shr+and+shl+or
 * for the upper 2 bits, int->float convert, mad) and re-summed the activations
 * with 2 adds per weight.
 *
 * The rewrite (same numerics, same memory layout) costs ~2.5:
 *
 *  1) PACKED PAIR WORDS.  Two weights that live 16 bits apart in the ql word
 *     are isolated with a single AND (0x000F000F) -- and their two 2-bit high
 *     fields are ALSO 16 bits apart in the qh word (bits r*8+2c for weight r of
 *     chunk c), so ONE shift + one masked OR completes BOTH 6-bit weights.  The
 *     pair converts with a single "mov (16) dst:f src<2;1,0>:uw" for the low
 *     half and shr + mov for the high half.
 *
 *  2) VECTOR ACCUMULATORS.  The 32 mads of a sub-block used to form one serial
 *     dependency chain; they now feed 8 independent float4 lanes, and the
 *     activations come from SLM as float4 (one d32x4 SLM load per 4 weights).
 *
 *  3) ACTIVATION SUMS FOR FREE.  Q6_K needs sum(x) separately for the low and
 *     the high half of every sub-block (they have different int8 scales) to
 *     apply the -32 offset.  The 256 activations arrive in ONE
 *     intel_sub_group_block_read8 where component j / lane l holds the fp16
 *     pair (32j+2l, 32j+2l+1): lanes 0..7 are exactly the low half and lanes
 *     8..15 exactly the high half, so THREE shuffle-xor butterfly steps plus
 *     two broadcasts produce both sums (~8 instructions instead of 512 adds).
 *
 *  4) UP/GATE SPLIT ACROSS SUB-GROUPS (SPLIT_UG): the first half of the
 *     sub-groups reduce the UP matrix, the second half the GATE matrix, which
 *     halves the live weight registers (Q6_K keeps 6 x uint8 per matrix) and
 *     doubles the in-flight loads per matrix.  The halves meet in SLM at the
 *     work-group barrier that was already there.
 *
 * == Transposed ("SG") per-expert layout — SAME total size as original ==
 * Identical to pack_q6k_sg().  nbpr = K/256, nrg = N/OPG:
 *   pql_T:  num_blocks * 128 bytes (low nibbles)
 *     Entry (h,bid) at (h*nbpr+bid)*OPG*128, sub-block j at j*OPG*16,
 *     chunk c(0..3) at j*OPG*16 + c*OPG*4.
 *   pqh_T:  num_blocks * 64 bytes  (upper 2 bits)
 *     Entry (h,bid) at (h*nbpr+bid)*OPG*64, sub-block j at j*OPG*8,
 *     half c(0..1) at j*OPG*8 + c*OPG*4.
 *   ps_T:   num_blocks * 16 bytes  (int8 scales, si at si*OPG)
 *   pd_T:   num_blocks * 2  bytes  (fp16 super-scale, 1 ushort/lane)
 *   expert_size = num_blocks * 210  (128+64+16+2) — unchanged.
 *
 * == Dispatch (unchanged) ==
 *   group_up_gate_q6k_sg: global=(output_len,KSPLIT,output_num*token_len)
 *   down_merge_q6k_sg   : global=(output_len,KSPLIT,token_len)
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
#ifndef SLM_SCALAR
#define SLM_SCALAR 0
#endif
#if SLM_SCALAR
#define XLOAD(p) ((float4)((p)[0], (p)[1], (p)[2], (p)[3]))
#else
#define XLOAD(p) (*(__local const float4 *)(p))
#endif

#ifndef DM_BLOCKED
#define DM_BLOCKED 1
#endif

/* down_merge: fetch the whole routing table with one sub-group block read and
 * recover each expert's entry with a shuffle, instead of a dependent scalar
 * load per iteration. */
#ifndef DM_HOIST_ROUTE
#define DM_HOIST_ROUTE 1
#endif

#define W2_MASK 0x000F000Fu   /* two 4-bit low parts, 16 bit apart        */
#define Q6_MASK 0x00300030u   /* their two 2-bit high parts, 16 bit apart */

/* No-convert mantissa trick (same idea as ACC_MODE==2 in moe_gemv_q4k_sg.cl
 * and the analogous QW5 in moe_gemv_q5k_sg.cl): a 6-bit weight v in [0,63]
 * satisfies
 *     as_float(0x3F800000 | (v << 17)) == 1.0f + v/64
 * exactly (6 mantissa bits, bits 17..22).  _a/_b/_c/_d already pack the
 * nibble (bits 0..3) and its 2-bit high field (bits 4..5, thanks to Q6_MASK
 * landing the shifted qh bits exactly there) into a ready 6-bit field in
 * BOTH the low 16 bits and the high 16 bits of the word, so each needs only
 * one shift (into the bit 17..22 window) + one masked-or instead of an
 * int->float convert.  The 1.0f offset (and the usual -32 bias) are folded
 * into the per-sub-block scale in q6k_decode / the accumulation call sites
 * (weights arrive as (64 + v)/64). */
#define FMASK6  0x007E0000u    /* fp32 mantissa slot for a 6-bit weight  */
#define FBIAS   0x3F800000u    /* fp32 1.0                               */
#define QW6(shifted) as_float(((shifted) & FMASK6) | FBIAS)

/* ---------------------------------------------------------------------------
 * One sub-block (32 weights) = 4 ql chunk words + 2 qh words (one per half) +
 * 32 SLM activations.  For chunk c and byte r:
 *     weight (   4c+r) = (q >> (8r  )) & 0xF | ((HL >> (r*8 + 2c)) & 3) << 4
 *     weight (16+4c+r) = (q >> (8r+4)) & 0xF | ((HH >> (r*8 + 2c)) & 3) << 4
 * Pairs (r=0,r=2) and (r=1,r=3) are 16 bits apart in q AND in HL/HH.
 * Results are kept separately for the two halves because Q6_K scales them with
 * two different int8 scales.
 * ------------------------------------------------------------------------- */
#define Q6K_CHUNK(q, HL, HH, c)                                               \
    do {                                                                      \
        uint _q = (q);                                                         \
        uint _a = ( _q        & W2_MASK) | (((HL) << 4) >> (2 * (c)) & Q6_MASK);\
        uint _b = ((_q >>  8) & W2_MASK) | ((HL) >> (4 + 2 * (c))    & Q6_MASK);\
        uint _c = ((_q >>  4) & W2_MASK) | (((HH) << 4) >> (2 * (c)) & Q6_MASK);\
        uint _d = ((_q >> 12) & W2_MASK) | ((HH) >> (4 + 2 * (c))    & Q6_MASK);\
        float4 _wl = (float4)(QW6(_a << 17), QW6(_b << 17),                    \
                              QW6(_a <<  1), QW6(_b <<  1));                   \
        float4 _wh = (float4)(QW6(_c << 17), QW6(_d << 17),                    \
                              QW6(_c <<  1), QW6(_d <<  1));                   \
        vlo = fma(XLOAD(in + (c) * 4),      _wl, vlo);                         \
        vhi = fma(XLOAD(in + 16 + (c) * 4), _wh, vhi);                          \
    } while (0)

inline float2 q6k_subblock(uint q0, uint q1, uint q2, uint q3,
                           uint HL, uint HH, __local const float *in)
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

/* acc += clo*(dot_lo - 32*isum_lo) + chi*(dot_hi - 32*isum_hi).  Nesting the
 * -32 correction inside the fma keeps 16 registers free (no pre-scaled scale
 * arrays), which matters: Q6_K already holds 6 uint8 of weights per block. */
#define Q6K_SUB(j, a, b, c, d, hl, hh)                                        \
    do {                                                                      \
        float2 _dv = q6k_subblock(a, b, c, d, hl, hh, sb + (j) * 32);          \
        acc = fma(clo[j], _dv.x, acc);                                        \
        acc = fma(chi[j], _dv.y, acc);                                        \
    } while (0)

inline float q6k_acc(uint8 w0, uint8 w1, uint8 w2, uint8 w3,
                     uint8 hq0, uint8 hq1,
                     const float *clo, const float *chi,
                     __local const float *sb)
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
    /* weights arrive as (64 + v)/64 (see Q6K_CHUNK/QW6).  With
     *   dot(v,x) - 32*isum = 64*dot'(x) - 96*isum
     * pre-scaling clo/chi by 64 here lets the call sites simply use
     *   acc += q6k_acc(...) - 1.5f * corr(...)
     * instead of "- 32.0f * corr" (see group_up_gate/down_merge below). */
    unroll_for (int j = 0; j < 8; j++) {
        clo[j] = 64.0f * d * (float)sc[2 * j];
        chi[j] = 64.0f * d * (float)sc[2 * j + 1];
    }
}

/* ---------------------------------------------------------------------------
 * Stage 256 fp16 activations into SLM as float and return the ready-made
 * -32-offset correction of the whole block:
 *       corr = sum_j ( clo[j]*sum(x of low half j) + chi[j]*sum(x of high j) )
 * so the caller only needs  acc += dot_terms - 32*corr .
 *
 * Folding the 16 half-sums into ONE scalar right where they are produced (a)
 * keeps 16 registers free -- with 6 uint8 of Q6_K weights live, holding the
 * sums until the end of the block made IGC split the 6 wide block reads into
 * two batches, which cost ~20% of the achievable bandwidth -- and (b) costs
 * exactly the same number of fmas as applying them per sub-block.
 *
 * Component j / lane l of the block read holds the fp16 pair
 * (32j+2l, 32j+2l+1): lanes 0..7 are exactly the low half of sub-block j and
 * lanes 8..15 the high half, so three shuffle-xor butterfly steps leave every
 * lane of a group holding that half's total and two broadcasts publish them.
 * ------------------------------------------------------------------------- */
inline uint8 stage_inp_f16(__local float *sb, const __global half *src, uint hh)
{
    uint8 iv = intel_sub_group_block_read8((const __global uint *)src);
#define Q6K_ST(j, comp) \
    *(__local float2 *)(sb + (j) * 32 + 2 * hh) = convert_float2(as_half2(comp));
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
        float2 _f = convert_float2(as_half2(comp));                            \
        float _v = _f.x + _f.y;                                               \
        _v += intel_sub_group_shuffle_xor(_v, 1);                              \
        _v += intel_sub_group_shuffle_xor(_v, 2);                              \
        _v += intel_sub_group_shuffle_xor(_v, 4);                              \
        corr = fma(clo[j], intel_sub_group_shuffle(_v, 0), corr);              \
        corr = fma(chi[j], intel_sub_group_shuffle(_v, 8), corr);              \
    }
    Q6K_STAGE(0, iv.s0) Q6K_STAGE(1, iv.s1) Q6K_STAGE(2, iv.s2) Q6K_STAGE(3, iv.s3)
    Q6K_STAGE(4, iv.s4) Q6K_STAGE(5, iv.s5) Q6K_STAGE(6, iv.s6) Q6K_STAGE(7, iv.s7)
#undef Q6K_STAGE
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

    __local float slm_inp[KSPLIT * 2 * 256];
    __local float slm_red[2 * KHALF * OPG];
    __local float *sbase = slm_inp + sg * (2 * 256);

    const __global uchar *base = is_gate ? gt_pql : up_pql;
    const __global uint  *pql_u = (const __global uint  *)base;
    const __global uint  *pqh_u = (const __global uint  *)(base + (size_t)num_blocks * 128);
    const __global uchar *ps    = base + (size_t)num_blocks * 192;
    const __global uchar *pd    = base + (size_t)num_blocks * 208;

    float acc = 0.0f;
    for (uint bid = sg_k; bid < nbpr; bid += KHALF) {
        __local float *sb = sbase + (bid & 1u) * 256;

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

        /* stage first: the SLM stores (and thus the barrier) then depend only
         * on the activation read, not on the scale reads. */
        uint8 iv = stage_inp_f16(sb, inputs + (size_t)t * input_len
                                 + (size_t)bid * 256, hh);
        float clo[8], chi[8];
        q6k_decode(scq, dr, clo, chi);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        float corr = q6k_corr(iv, clo, chi);
        acc += q6k_acc(w0, w1, w2, w3, hq0, hq1, clo, chi, sb) - 1.5f * corr;
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

    const __global uint  *up_pql_u = (const __global uint  *)up_pql;
    const __global uint  *up_pqh_u = (const __global uint  *)(up_pql + (size_t)num_blocks * 128);
    const __global uchar *up_ps    = up_pql + (size_t)num_blocks * 192;
    const __global uchar *up_pd    = up_pql + (size_t)num_blocks * 208;
    const __global uint  *gt_pql_u = (const __global uint  *)gt_pql;
    const __global uint  *gt_pqh_u = (const __global uint  *)(gt_pql + (size_t)num_blocks * 128);
    const __global uchar *gt_ps    = gt_pql + (size_t)num_blocks * 192;
    const __global uchar *gt_pd    = gt_pql + (size_t)num_blocks * 208;

    float up_acc = 0.0f, gate_acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float *sb = sbase + (bid & 1u) * 256;

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

        uint8 iv = stage_inp_f16(sb, inputs + (size_t)t * input_len
                                 + (size_t)bid * 256, hh);
        float uclo[8], uchi[8], gclo[8], gchi[8];
        q6k_decode(uscq, udr, uclo, uchi);
        q6k_decode(gscq, gdr, gclo, gchi);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);


        float ucorr = q6k_corr(iv, uclo, uchi);
        float gcorr = q6k_corr(iv, gclo, gchi);
        up_acc   += q6k_acc(uw0, uw1, uw2, uw3, uh0, uh1, uclo, uchi, sb)
                    - 1.5f * ucorr;
        gate_acc += q6k_acc(gw0, gw1, gw2, gw3, gh0, gh1, gclo, gchi, sb)
                    - 1.5f * gcorr;
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

        uint8 iv = stage_inp_f16(sb, inputs + (size_t)t * input_num * input_len
                                 + (size_t)e * input_len + (size_t)bid * 256, hh);
        float clo[8], chi[8];
        q6k_decode(scq, dr, clo, chi);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        float corr = q6k_corr(iv, clo, chi);
        acc = fma(q6k_acc(w0, w1, w2, w3, hq0, hq1, clo, chi, sb) - 1.5f * corr,
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
