/*
 * moe_gemv_q4k_shuffle.cl
 *
 * OpenCL MoE kernels for Q4_K on shuffled weight layout, translated from the
 * SYCL/ESIMD sources:
 *   runGroupUpAndGateVecQ4KL1_xve_impl  (gemv_q4k_L1.xve.cpp)
 *   runDownMergeQ4KL1_xve_impl          (gemv_q4k_L1.xve.cpp)
 *
 * ============================ OPTIMIZATION NOTES ==========================
 * These kernels are strongly memory-bound (each expert's weight matrix is
 * re-fetched from DRAM every dispatch).  The original translation read the
 * quantized weights one *byte* at a time (pqs_blk[j*16+k]) and the 256 input
 * activations one *float* at a time, generating a huge number of narrow
 * global-load instructions -> the kernels were latency/issue bound and left
 * the DRAM pipe under-utilised.
 *
 * Following the same memory-bandwidth philosophy as gemv_q4k_sg.cl (which
 * uses intel_sub_group_block_read8 to move 512 B per call), this version:
 *
 *   1. Reads the 128-byte packed-quant block with WIDE vector loads:
 *        uint4 per 32-weight sub-block  (16 B / load instead of 16x1 B).
 *      A raw intel_sub_group_block_read cannot be used directly here because
 *      the fixed shuffle layout gives each lane one *whole* contiguous
 *      128-byte K-block (128-byte stride between lanes, not the interleaved
 *      layout block-reads require); the uint4 loads achieve the same goal of
 *      maximal-width memory transactions / minimal instruction count while
 *      keeping the on-device layout, kernel signatures and results identical.
 *   2. Loads the 256 activations with float8 vector reads.
 *   3. Decodes the 16-byte scale/min record with four uint loads instead of
 *      sixteen byte loads.
 *   4. Factors the per-sub-block quantized scale/min OUT of the inner loop:
 *        acc += scale_j*dot_j - min_j*isum_j
 *      (isum_j = sum of the 32 activations) which halves the FMA count.
 *
 * All of the above are bit-for-bit equivalent to the original math; the
 * on-device shuffle layout, kernel names, argument order and OPG/BS #defines
 * are unchanged so the existing dispatch / test harness is untouched.
 * ==========================================================================
 *
 * Shuffled layout (produced by ShuffleQ4KL1), identical for `ups`, `gates`
 * and `downs` expert weight tables:
 *   pqs  [num_blocks * 128]  -- packed 4-bit quants, 16 rows interleaved per 128-byte block
 *   psl  [num_blocks * 16]   -- scales/mins/d/dmin packed as:
 *       bytes [0..3]   : sl   (scale low  nibbles, 8 values in 4 bytes)
 *       bytes [4..7]   : ml   (min   low  nibbles, 8 values in 4 bytes)
 *       bytes [8..9]   : sh   (scale high 2-bits, 8 values in 2 bytes)
 *       bytes [10..11] : mh   (min   high 2-bits, 8 values in 2 bytes)
 *       bytes [12..13] : d    (fp16 super-scale)
 *       bytes [14..15] : dmin (fp16 super-min)
 *   expert_size = num_blocks * 144   (128 qs + 16 sl)
 *
 * Kernels provided:
 *   group_up_gate_q4k -- MoE up+gate fused GEMV with SiLU(gate) * up activation
 *   down_merge_q4k     -- MoE down GEMV with per-expert weighted accumulation
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define unroll_hint __attribute__((opencl_unroll_hint))

/* -----------------------------------------------------------------------
 * Helper: reconstruct scale/min arrays for one Q4_K block.
 * Reads the 16-byte record as four little-endian uints (vs 16 byte loads).
 *   pu[0] = sl, pu[1] = ml, pu[2] = sh|mh<<16, pu[3] = d|dmin<<16
 * --------------------------------------------------------------------- */
inline void q4k_decode_scales(const __global uchar *psl,
                               float *scale_out, float *min_out)
{
    const __global uint *pu = (const __global uint *)psl;
    uint sl = pu[0];
    uint ml = pu[1];
    uint p2 = pu[2];
    uint p3 = pu[3];
    uint  sh   = p2 & 0xFFFFu;
    uint  mh   = p2 >> 16;
    float d    = (float)as_half((ushort)(p3 & 0xFFFFu));
    float dmin = (float)as_half((ushort)(p3 >> 16));

    unroll_hint
    for (int i = 0; i < 8; i++) {
        uint sq = ((sl >> (i*4)) & 0xF) | (((sh >> (i*2)) & 0x3) << 4);
        uint mq = ((ml >> (i*4)) & 0xF) | (((mh >> (i*2)) & 0x3) << 4);
        scale_out[i] = (float)sq * d;
        min_out[i]   = (float)mq * dmin;
    }
}

/* -----------------------------------------------------------------------
 * Whole-block dot product for one Q4_K block (256 weights, 8 sub-blocks).
 *   pqs_blk : pointer to start of 128-byte qs section for this block
 *   inp     : 256 activations for this block row (private)
 *   scale, minv : decoded per-sub-block factors
 * Uses uint4 wide loads + factored scale/min accumulation.
 * --------------------------------------------------------------------- */
inline float q4k_block_dot(const __global uchar *pqs_blk,
                           const float *inp,
                           const float *scale, const float *minv)
{
    const __global uint4 *pw = (const __global uint4 *)pqs_blk;
    float acc = 0.0f;

    unroll_hint
    for (int j = 0; j < 8; j++) {
        uint4 w = pw[j];                 /* 16 B = one 32-weight sub-block */
        uint  ws[4] = { w.x, w.y, w.z, w.w };
        const float *in0 = inp + j * 32;       /* low-nibble activations  */
        const float *in1 = inp + j * 32 + 16;  /* high-nibble activations */

        float dot  = 0.0f;
        float isum = 0.0f;
        unroll_hint
        for (int c = 0; c < 4; c++) {
            uint u = ws[c];
            float a0 = in0[c*4+0], a1 = in0[c*4+1], a2 = in0[c*4+2], a3 = in0[c*4+3];
            float b0 = in1[c*4+0], b1 = in1[c*4+1], b2 = in1[c*4+2], b3 = in1[c*4+3];
            dot = fma(a0, (float)( u        & 0xFu), dot);
            dot = fma(a1, (float)((u >>  8) & 0xFu), dot);
            dot = fma(a2, (float)((u >> 16) & 0xFu), dot);
            dot = fma(a3, (float)((u >> 24) & 0xFu), dot);
            dot = fma(b0, (float)((u >>  4) & 0xFu), dot);
            dot = fma(b1, (float)((u >> 12) & 0xFu), dot);
            dot = fma(b2, (float)((u >> 20) & 0xFu), dot);
            dot = fma(b3, (float)((u >> 28) & 0xFu), dot);
            isum += a0 + a1 + a2 + a3 + b0 + b1 + b2 + b3;
        }
        acc = fma(scale[j],  dot,  acc);
        acc = fma(-minv[j],  isum, acc);
    }
    return acc;
}

/* Load 256 activations into a private buffer using float8 vector reads. */
inline void q4k_load_inp(float *inp, const __global float *src)
{
    const __global float8 *pin = (const __global float8 *)src;
    float8 *pv = (float8 *)inp;
    unroll_hint
    for (int i = 0; i < 32; i++)
        pv[i] = pin[i];
}

/* Global-input variant of q4k_block_dot (no private staging) -- used by the
 * down kernel whose tiny workgroups (local=nprow) are sensitive to the 1 KB
 * private activation array; the small activation vector stays L1-resident. */
inline float q4k_block_dot_g(const __global uchar *pqs_blk,
                             const __global float *inp,
                             const float *scale, const float *minv)
{
    const __global uint4  *pw  = (const __global uint4  *)pqs_blk;
    const __global float4 *pin = (const __global float4 *)inp;
    float acc = 0.0f;

    unroll_hint
    for (int j = 0; j < 8; j++) {
        uint4 w = pw[j];
        uint  ws[4] = { w.x, w.y, w.z, w.w };
        float4 lo0 = pin[j*8 + 0], lo1 = pin[j*8 + 1], lo2 = pin[j*8 + 2], lo3 = pin[j*8 + 3];
        float4 hi0 = pin[j*8 + 4], hi1 = pin[j*8 + 5], hi2 = pin[j*8 + 6], hi3 = pin[j*8 + 7];
        float4 in0[4] = { lo0, lo1, lo2, lo3 };
        float4 in1[4] = { hi0, hi1, hi2, hi3 };

        float dot  = 0.0f;
        float isum = 0.0f;
        unroll_hint
        for (int c = 0; c < 4; c++) {
            uint u = ws[c];
            float4 a = in0[c];
            float4 b = in1[c];
            dot = fma(a.x, (float)( u        & 0xFu), dot);
            dot = fma(a.y, (float)((u >>  8) & 0xFu), dot);
            dot = fma(a.z, (float)((u >> 16) & 0xFu), dot);
            dot = fma(a.w, (float)((u >> 24) & 0xFu), dot);
            dot = fma(b.x, (float)((u >>  4) & 0xFu), dot);
            dot = fma(b.y, (float)((u >> 12) & 0xFu), dot);
            dot = fma(b.z, (float)((u >> 20) & 0xFu), dot);
            dot = fma(b.w, (float)((u >> 28) & 0xFu), dot);
            isum += a.x+a.y+a.z+a.w + b.x+b.y+b.z+b.w;
        }
        acc = fma(scale[j],  dot,  acc);
        acc = fma(-minv[j],  isum, acc);
    }
    return acc;
}

/* =======================================================================
 * Kernel 1: group_up_gate_q4k
 * ===================================================================== */
#define OPG_UG 8
#define BS_UG  1

__kernel
void group_up_gate_q4k(
    __global const float  *inputs,          /* [token_len * input_len] */
    __global const uchar  *ups,             /* [num_experts * expert_size] */
    __global const uchar  *gates,           /* [num_experts * expert_size] */
    __global const uint   *indexes,         /* [token_len * aligned_output_num] or NULL */
    __global       float  *outputs,         /* [token_len * output_num * output_len] */
    uint output_num,
    uint aligned_output_num,
    uint token_len,
    uint input_len,
    uint output_len
) {
    uint num_blocks_per_row = input_len / 256;
    uint num_blocks         = input_len * output_len / 256;
    uint expert_size        = num_blocks * 144;  /* Q4K: 128 qs + 16 sl */

    uint local_thread = num_blocks_per_row / BS_UG;

    uint h  = get_group_id(0);
    uint hh = get_local_id(0);
    uint v  = get_group_id(1);
    uint t  = get_group_id(2);

    uint index = indexes ? indexes[t * aligned_output_num + v] : v;

    __local float slm_up  [64 * OPG_UG];
    __local float slm_gate[64 * OPG_UG];

    float outUp  [OPG_UG]; for (int l = 0; l < OPG_UG; l++) outUp[l]   = 0.0f;
    float outGate[OPG_UG]; for (int l = 0; l < OPG_UG; l++) outGate[l] = 0.0f;

    for (uint c = 0; c < BS_UG; c++) {
        uint blockIdx = hh * BS_UG + c;

        float inp[256];
        q4k_load_inp(inp, inputs + (size_t)t * input_len + 256 * blockIdx);

        /* --- up projection --- */
        {
            const __global uchar *pqs_w = ups + (size_t)index * expert_size;
            const __global uchar *psl_w = pqs_w + (size_t)num_blocks * 128;

            unroll_hint
            for (int l = 0; l < OPG_UG; l++) {
                uint blk_idx = h * OPG_UG * num_blocks_per_row + (uint)l * num_blocks_per_row + blockIdx;
                float scale[8], minv[8];
                q4k_decode_scales(psl_w + (size_t)blk_idx * 16, scale, minv);
                outUp[l] += q4k_block_dot(pqs_w + (size_t)blk_idx * 128, inp, scale, minv);
            }
        }

        /* --- gate projection --- */
        {
            const __global uchar *pqs_w = gates + (size_t)index * expert_size;
            const __global uchar *psl_w = pqs_w + (size_t)num_blocks * 128;

            unroll_hint
            for (int l = 0; l < OPG_UG; l++) {
                uint blk_idx = h * OPG_UG * num_blocks_per_row + (uint)l * num_blocks_per_row + blockIdx;
                float scale[8], minv[8];
                q4k_decode_scales(psl_w + (size_t)blk_idx * 16, scale, minv);
                outGate[l] += q4k_block_dot(pqs_w + (size_t)blk_idx * 128, inp, scale, minv);
            }
        }
    }

    for (int l = 0; l < OPG_UG; l++) {
        slm_up  [hh * OPG_UG + l] = outUp[l];
        slm_gate[hh * OPG_UG + l] = outGate[l];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (hh == 0) {
        float accUp  [OPG_UG]; for (int l = 0; l < OPG_UG; l++) accUp[l]   = 0.0f;
        float accGate[OPG_UG]; for (int l = 0; l < OPG_UG; l++) accGate[l] = 0.0f;
        for (uint i = 0; i < local_thread; i++) {
            for (int l = 0; l < OPG_UG; l++) {
                accUp[l]   += slm_up  [i * OPG_UG + l];
                accGate[l] += slm_gate[i * OPG_UG + l];
            }
        }
        /* SiLU(gate) * up  ==  sigmoid(gate) * gate * up */
        for (int l = 0; l < OPG_UG; l++) {
            float g       = accGate[l];
            float sig     = 1.0f / (1.0f + exp(-g));
            float silu_g  = sig * g;
            outputs[(size_t)t * output_num * output_len + (size_t)v * output_len + h * OPG_UG + l] =
                silu_g * accUp[l];
        }
    }
}

/* =======================================================================
 * Kernel 2: down_merge_q4k
 * ===================================================================== */
#define OPG_DM 4

__kernel
void down_merge_q4k(
    __global const float  *inputs,       /* [token_len * input_num * input_len] */
    __global const uchar  *downs,        /* [num_experts * expert_size] */
    __global const uint   *indexes,      /* [token_len * aligned_input_num] or NULL */
    __global const float  *efficients,   /* [token_len * aligned_input_num] */
    __global       float  *outputs,      /* [token_len * output_len] */
    uint input_num,
    uint aligned_input_num,
    uint token_len,
    uint input_len,
    uint output_len,
    int  is_acc
) {
    uint num_blocks_per_row = input_len / 256;
    uint num_blocks         = input_len * output_len / 256;
    uint expert_size        = num_blocks * 144;

    uint h  = get_group_id(0);
    uint hh = get_local_id(0);
    uint t  = get_group_id(1);

    __local float slm[64 * OPG_DM];

    float outData[OPG_DM]; for (int l = 0; l < OPG_DM; l++) outData[l] = 0.0f;

    for (int e = 0; e < (int)input_num; e++) {
        uint  index     = indexes ? indexes[t * aligned_input_num + e] : (uint)e;
        float efficient = efficients[t * aligned_input_num + e];

        const __global uchar *pqs_w = downs + (size_t)index * expert_size;
        const __global uchar *psl_w = pqs_w + (size_t)num_blocks * 128;

        const __global float *inp = inputs + (size_t)t * input_num * input_len
                                  + (size_t)e * input_len + 256 * hh;

        unroll_hint
        for (int l = 0; l < OPG_DM; l++) {
            uint blk_idx = h * OPG_DM * num_blocks_per_row + (uint)l * num_blocks_per_row + hh;
            float scale[8], minv[8];
            q4k_decode_scales(psl_w + (size_t)blk_idx * 16, scale, minv);
            outData[l] += q4k_block_dot_g(pqs_w + (size_t)blk_idx * 128, inp, scale, minv) * efficient;
        }
    }

    for (int l = 0; l < OPG_DM; l++)
        slm[hh * OPG_DM + l] = outData[l];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (hh == 0) {
        float accOut[OPG_DM]; for (int l = 0; l < OPG_DM; l++) accOut[l] = 0.0f;
        for (uint i = 0; i < num_blocks_per_row; i++)
            for (int l = 0; l < OPG_DM; l++)
                accOut[l] += slm[i * OPG_DM + l];
        if (is_acc) {
            for (int l = 0; l < OPG_DM; l++)
                accOut[l] += outputs[(size_t)t * output_len + h * OPG_DM + l];
        }
        for (int l = 0; l < OPG_DM; l++)
            outputs[(size_t)t * output_len + h * OPG_DM + l] = accOut[l];
    }
}
