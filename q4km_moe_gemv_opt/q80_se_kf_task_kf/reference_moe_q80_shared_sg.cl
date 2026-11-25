/*
 * reference_moe_q80_shared_sg.cl
 *
 * Baseline (reference) implementation of the shared-expert Q8_0 "SG"
 * (sub-group block-read, transposed weight layout) GEMV kernels:
 *   shared_gate_up_q8_0      — fused gate-proj + up-proj GEMV + SiLU(gate)*up
 *   shared_down_merge_q8_0   — down-proj GEMV, scaled by a per-token scalar
 *                              gate and optionally accumulated into outputs
 *
 * These kernels are used as the correctness oracle and performance baseline.
 * Do NOT modify this file.
 *
 * This baseline mirrors the ORIGINAL (pre-optimization) "ocl_v2" extraction
 * described in the header of moe_gemv_q80_shared_sg.cl / kernel_moe_q80_shared_sg.cl:
 * each 256-element activation window is cooperatively staged into an SLM
 * scratch buffer (one intel_sub_group_block_read8 -> 16 lanes each write 2
 * floats to SLM), followed by a per-K-block sub_group_barrier(CLK_LOCAL_MEM_FENCE)
 * so every lane can read the whole window back out of SLM for the dot
 * product.  This is numerically identical to (but slower than) the
 * SLM-free, register-shuffle-based kernel that evolve targets are compared
 * against for correctness and speedup.
 *
 * == Weight layout (SG-packed Q8_0, per [N, K] weight matrix) ==
 *   pqs section: N*nbpr*256 bytes at offset 0 (row-group h, block bid at
 *     (h*nbpr+bid)*OPG*256 bytes; sub-block j at +j*OPG*32, chunk c at
 *     +c*OPG*4).
 *   pd  section: N*nbpr*16 bytes right after pqs (SoA fp16, entry (h,bid) at
 *     (h*nbpr+bid)*OPG*16 bytes, field j at j*OPG*2 + lid*2).
 *
 * == Dispatch ==
 *   shared_gate_up_q8_0:
 *     global = (intermediate_size, KSPLIT, token_len)
 *     local  = (OPG=16,            KSPLIT, 1)
 *   shared_down_merge_q8_0:
 *     global = (hidden_size, KSPLIT, token_len)
 *     local  = (OPG=16,      KSPLIT, 1)
 */

/* [REFERENCE_START] */

#pragma OPENCL EXTENSION cl_khr_fp16        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define OPG 16  /* = sub_group_size, fixed by the reference kernels */

#ifndef KSPLIT
#define KSPLIT 1
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

/* Accumulate one 4-byte chunk (4 signed int8 weights) into a sub-block dot
 * product against the SLM-staged 32-activation window `sb`. */
#define Q8_ACC_CHUNK(dot, u, c, sb)                                          \
    do {                                                                     \
        char4 _q = as_char4((uint)(u));                                      \
        (dot) = fma((sb)[(c) * 4 + 0], (float)_q.s0, (dot));                 \
        (dot) = fma((sb)[(c) * 4 + 1], (float)_q.s1, (dot));                 \
        (dot) = fma((sb)[(c) * 4 + 2], (float)_q.s2, (dot));                 \
        (dot) = fma((sb)[(c) * 4 + 3], (float)_q.s3, (dot));                 \
    } while (0)

/* One SG-packed Q8_0 sub-block (32 signed weights as a uint8) dotted against
 * 32 SLM-resident activations `sb`, scaled and accumulated into `acc`. */
#define Q8_SUBBLOCK(sb, wv, sc, acc)                                                        \
    do {                                                                                    \
        float _dot = 0.0f;                                                                  \
        Q8_ACC_CHUNK(_dot, (wv).s0, 0, sb); Q8_ACC_CHUNK(_dot, (wv).s1, 1, sb);              \
        Q8_ACC_CHUNK(_dot, (wv).s2, 2, sb); Q8_ACC_CHUNK(_dot, (wv).s3, 3, sb);              \
        Q8_ACC_CHUNK(_dot, (wv).s4, 4, sb); Q8_ACC_CHUNK(_dot, (wv).s5, 5, sb);              \
        Q8_ACC_CHUNK(_dot, (wv).s6, 6, sb); Q8_ACC_CHUNK(_dot, (wv).s7, 7, sb);              \
        (acc) = fma((sc), _dot, (acc));                                                     \
    } while (0)

/* Decode the 8 SoA fp16 sub-block scales fetched by one
 * intel_sub_group_block_read_us8 per lane. */
inline void q8_0_decode_scales8(ushort8 dv, float *scale) {
    ushort dh[8];
    dh[0] = dv.s0; dh[1] = dv.s1; dh[2] = dv.s2; dh[3] = dv.s3;
    dh[4] = dv.s4; dh[5] = dv.s5; dh[6] = dv.s6; dh[7] = dv.s7;
    unroll_for (int i = 0; i < 8; i++)
        scale[i] = (float)as_half(dh[i]);
}

/* Stage 256 fp16 activations into SLM (sb[n] = (float)activation_half[n]) via
 * 2 sub-group block reads of 8 ushorts each (intel_sub_group_block_read_us8).
 * Each us8 read pulls 128 contiguous fp16 values (256 B) with all 16 lanes
 * participating -- identical wide-transaction strategy to the weight reads. */
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
 * Kernel 1: shared_gate_up_q8_0
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void shared_gate_up_q8_0(
    const __global uchar * restrict gate_weights,  /* SG-packed Q8_0 [intermediate_size, hidden_size] */
    const __global uchar * restrict up_weights,    /* SG-packed Q8_0 [intermediate_size, hidden_size] */
    const __global half  * restrict hidden_states, /* [token_len, hidden_size] */
          __global half  * restrict gate_up_out,   /* [token_len, intermediate_size] */
    uint hidden_size,
    uint intermediate_size
) {
    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint t  = get_group_id(2);

    const uint nbpr   = hidden_size / 256u;
    const uint off_pd = (uint)intermediate_size * nbpr * 256u;

    const __global uint*   g_pqs_u = (const __global uint*)gate_weights;
    const __global ushort* g_pd_us = (const __global ushort*)(gate_weights + off_pd);
    const __global uint*   u_pqs_u = (const __global uint*)up_weights;
    const __global ushort* u_pd_us = (const __global ushort*)(up_weights + off_pd);

    const __global half* x = hidden_states + (size_t)t * hidden_size;

    /* Double-buffered SLM staging window (identical shape to the reference
     * Q4_K/Q5_K/Q6_K "shuffle" kernels): each sub-group gets a 2*256-float
     * scratch pair so stage(bid+1) can start while use(bid) is still live. */
    __local float slm_inp[KSPLIT * 2 * 256];
    __local float *sbase = slm_inp + sg * (2 * 256);

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float *sb = sbase + (bid & 1u) * 256;

        const uint pe_u = (h * nbpr + bid) * (OPG * 256u / 4u);
        uint8 gw0 = intel_sub_group_block_read8(g_pqs_u + pe_u + 0 * OPG * 8);
        uint8 gw1 = intel_sub_group_block_read8(g_pqs_u + pe_u + 1 * OPG * 8);
        uint8 gw2 = intel_sub_group_block_read8(g_pqs_u + pe_u + 2 * OPG * 8);
        uint8 gw3 = intel_sub_group_block_read8(g_pqs_u + pe_u + 3 * OPG * 8);
        uint8 gw4 = intel_sub_group_block_read8(g_pqs_u + pe_u + 4 * OPG * 8);
        uint8 gw5 = intel_sub_group_block_read8(g_pqs_u + pe_u + 5 * OPG * 8);
        uint8 gw6 = intel_sub_group_block_read8(g_pqs_u + pe_u + 6 * OPG * 8);
        uint8 gw7 = intel_sub_group_block_read8(g_pqs_u + pe_u + 7 * OPG * 8);
        uint8 uw0 = intel_sub_group_block_read8(u_pqs_u + pe_u + 0 * OPG * 8);
        uint8 uw1 = intel_sub_group_block_read8(u_pqs_u + pe_u + 1 * OPG * 8);
        uint8 uw2 = intel_sub_group_block_read8(u_pqs_u + pe_u + 2 * OPG * 8);
        uint8 uw3 = intel_sub_group_block_read8(u_pqs_u + pe_u + 3 * OPG * 8);
        uint8 uw4 = intel_sub_group_block_read8(u_pqs_u + pe_u + 4 * OPG * 8);
        uint8 uw5 = intel_sub_group_block_read8(u_pqs_u + pe_u + 5 * OPG * 8);
        uint8 uw6 = intel_sub_group_block_read8(u_pqs_u + pe_u + 6 * OPG * 8);
        uint8 uw7 = intel_sub_group_block_read8(u_pqs_u + pe_u + 7 * OPG * 8);

        const uint pd_off = (h * nbpr + bid) * (OPG * 8u);
        ushort8 gdv = intel_sub_group_block_read_us8(g_pd_us + pd_off);
        ushort8 udv = intel_sub_group_block_read_us8(u_pd_us + pd_off);

        load_inp_f16(sb, x + (size_t)bid * 256, hh);

        float gscale[8], uscale[8];
        q8_0_decode_scales8(gdv, gscale);
        q8_0_decode_scales8(udv, uscale);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        Q8_SUBBLOCK(sb + 0 * 32, gw0, gscale[0], gate_acc); Q8_SUBBLOCK(sb + 0 * 32, uw0, uscale[0], up_acc);
        Q8_SUBBLOCK(sb + 1 * 32, gw1, gscale[1], gate_acc); Q8_SUBBLOCK(sb + 1 * 32, uw1, uscale[1], up_acc);
        Q8_SUBBLOCK(sb + 2 * 32, gw2, gscale[2], gate_acc); Q8_SUBBLOCK(sb + 2 * 32, uw2, uscale[2], up_acc);
        Q8_SUBBLOCK(sb + 3 * 32, gw3, gscale[3], gate_acc); Q8_SUBBLOCK(sb + 3 * 32, uw3, uscale[3], up_acc);
        Q8_SUBBLOCK(sb + 4 * 32, gw4, gscale[4], gate_acc); Q8_SUBBLOCK(sb + 4 * 32, uw4, uscale[4], up_acc);
        Q8_SUBBLOCK(sb + 5 * 32, gw5, gscale[5], gate_acc); Q8_SUBBLOCK(sb + 5 * 32, uw5, uscale[5], up_acc);
        Q8_SUBBLOCK(sb + 6 * 32, gw6, gscale[6], gate_acc); Q8_SUBBLOCK(sb + 6 * 32, uw6, uscale[6], up_acc);
        Q8_SUBBLOCK(sb + 7 * 32, gw7, gscale[7], gate_acc); Q8_SUBBLOCK(sb + 7 * 32, uw7, uscale[7], up_acc);
    }

    const uint row = h * OPG + hh;
#if KSPLIT == 1
    {
        float sig = 1.0f / (1.0f + exp(-gate_acc));
        gate_up_out[(size_t)t * intermediate_size + row] = (half)((sig * gate_acc) * up_acc);
    }
#else
    __local float slm_up[KSPLIT * OPG];
    __local float slm_gate[KSPLIT * OPG];
    slm_up[sg * OPG + hh]   = up_acc;
    slm_gate[sg * OPG + hh] = gate_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float u = 0.0f, gt = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) {
            u  += slm_up[r * OPG + hh];
            gt += slm_gate[r * OPG + hh];
        }
        float sig = 1.0f / (1.0f + exp(-gt));
        gate_up_out[(size_t)t * intermediate_size + row] = (half)((sig * gt) * u);
    }
#endif
}

/* =======================================================================
 * Kernel 2: shared_down_merge_q8_0
 * ===================================================================== */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void shared_down_merge_q8_0(
    const __global half  * restrict gate_up_in,   /* [token_len, intermediate_size] */
    const __global uchar * restrict down_weights, /* SG-packed Q8_0 [hidden_size, intermediate_size] */
    const __global half  * restrict shared_gate,  /* [token_len] scalar gate */
          __global half  * restrict final_out,    /* [token_len, hidden_size] */
    uint intermediate_size,
    uint hidden_size,
    int  is_acc
) {
    const uint h   = get_group_id(0);
    const uint hh  = get_local_id(0);
    const uint sg  = get_local_id(1);
    const uint t   = get_group_id(2);

    const uint nbpr   = intermediate_size / 256u;
    const uint off_pd = (uint)hidden_size * nbpr * 256u;

    const __global uint*   d_pqs_u = (const __global uint*)down_weights;
    const __global ushort* d_pd_us = (const __global ushort*)(down_weights + off_pd);

    const __global half* x = gate_up_in + (size_t)t * intermediate_size;

    __local float slm_inp[KSPLIT * 2 * 256];
    __local float *sbase = slm_inp + sg * (2 * 256);

    float acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float *sb = sbase + (bid & 1u) * 256;

        const uint pe_u = (h * nbpr + bid) * (OPG * 256u / 4u);
        uint8 w0 = intel_sub_group_block_read8(d_pqs_u + pe_u + 0 * OPG * 8);
        uint8 w1 = intel_sub_group_block_read8(d_pqs_u + pe_u + 1 * OPG * 8);
        uint8 w2 = intel_sub_group_block_read8(d_pqs_u + pe_u + 2 * OPG * 8);
        uint8 w3 = intel_sub_group_block_read8(d_pqs_u + pe_u + 3 * OPG * 8);
        uint8 w4 = intel_sub_group_block_read8(d_pqs_u + pe_u + 4 * OPG * 8);
        uint8 w5 = intel_sub_group_block_read8(d_pqs_u + pe_u + 5 * OPG * 8);
        uint8 w6 = intel_sub_group_block_read8(d_pqs_u + pe_u + 6 * OPG * 8);
        uint8 w7 = intel_sub_group_block_read8(d_pqs_u + pe_u + 7 * OPG * 8);

        const uint pd_off = (h * nbpr + bid) * (OPG * 8u);
        ushort8 dv = intel_sub_group_block_read_us8(d_pd_us + pd_off);

        load_inp_f16(sb, x + (size_t)bid * 256, hh);

        float scale[8];
        q8_0_decode_scales8(dv, scale);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        Q8_SUBBLOCK(sb + 0 * 32, w0, scale[0], acc);
        Q8_SUBBLOCK(sb + 1 * 32, w1, scale[1], acc);
        Q8_SUBBLOCK(sb + 2 * 32, w2, scale[2], acc);
        Q8_SUBBLOCK(sb + 3 * 32, w3, scale[3], acc);
        Q8_SUBBLOCK(sb + 4 * 32, w4, scale[4], acc);
        Q8_SUBBLOCK(sb + 5 * 32, w5, scale[5], acc);
        Q8_SUBBLOCK(sb + 6 * 32, w6, scale[6], acc);
        Q8_SUBBLOCK(sb + 7 * 32, w7, scale[7], acc);
    }

    const float gscale = (float)shared_gate[t];
    const uint row = h * OPG + hh;

#if KSPLIT == 1
    {
        float o = acc * gscale;
        if (is_acc)
            o += (float)final_out[(size_t)t * hidden_size + row];
        final_out[(size_t)t * hidden_size + row] = (half)o;
    }
#else
    __local float slm_red[KSPLIT * OPG];
    slm_red[sg * OPG + hh] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float o = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) o += slm_red[r * OPG + hh];
        o *= gscale;
        if (is_acc)
            o += (float)final_out[(size_t)t * hidden_size + row];
        final_out[(size_t)t * hidden_size + row] = (half)o;
    }
#endif
}

/* [REFERENCE_END] */
