/*
 * moe_gemv_q5k_shuffle.cl
 *
 * OpenCL MoE kernel for Q5_K on shuffled weight layout, translated from the
 * SYCL/ESIMD source:
 *   runDownMergeQ5KL1_xve_xe2_impl<OPG>  (gemv_q5k_L1.xve-xe2.cpp)
 *
 * ============================ OPTIMIZATION NOTES ==========================
 * Memory-bound MoE GEMV.  Same bandwidth-oriented rewrite as
 * moe_gemv_q4k_shuffle.cl (see notes there):
 *   1. 128-byte low-nibble quant block read with uint4 wide loads.
 *   2. 32-byte high-bit (qh) block read with uint loads (one uint / sub-block).
 *   3. 256 activations loaded with float8 vector reads.
 *   4. Scale/min record decoded with four uint loads.
 *   5. Per-sub-block scale/min factored out of the inner loop
 *        acc += scale_j*dot_j - min_j*isum_j
 * A raw intel_sub_group_block_read is not applicable to this fixed
 * shuffle layout (each lane owns a whole contiguous 128-byte K-block, not
 * the interleaved layout block-reads require); the wide vector loads deliver
 * the same maximal-width DRAM transactions while keeping the layout, kernel
 * signatures and results bit-identical.
 * ==========================================================================
 *
 * Shuffled layout (produced by ShuffleQ5KL1):
 *   pqs  [num_blocks * 128]  -- packed 4-bit lower quants (same packing as Q4_K)
 *   pqh  [num_blocks *  32]  -- packed high bits (1 bit per weight)
 *   psl  [num_blocks *  16]  -- scales/mins/d/dmin (same 16-byte structure as Q4_K)
 *   expert_size = num_blocks * 176   (128 qs + 32 qh + 16 sl)
 *
 * Per shuffled block (256 weights):
 *   Sub-block j (j=0..7) covers 32 weights at positions [j*32 .. j*32+31].
 *   qs layout: qs[j*16+k] = low4(w[j*32+k]) | (low4(w[j*32+16+k]) << 4)
 *   qh layout: qh[j*4+k]  packs the high bit of 8 weights:
 *       bit s of qh[j*4+k] = high_bit(w[j*32 + k + s*4])   for s=0..7
 *   Full weight = low_nibble(qs) | (high_bit(qh) << 4)      (range 0..31)
 *
 * Kernels provided:
 *   group_up_gate_q5k -- MoE up+gate fused GEMV with SiLU(gate) * up activation
 *   down_merge_q5k     -- MoE down GEMV with per-expert weighted accumulation
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define unroll_hint __attribute__((opencl_unroll_hint))

/* -----------------------------------------------------------------------
 * Decode scale/min for one Q5_K block (same 16-byte record as Q4_K).
 * --------------------------------------------------------------------- */
inline void q5k_decode_scales(const __global uchar *psl,
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
 * Whole-block dot product for one Q5_K block (256 weights, 8 sub-blocks).
 *   pqs_blk : base of 128-byte qs block (low nibbles)
 *   pqh_blk : base of 32-byte qh block  (1 high bit per weight)
 *
 * For weight index k (0..15) in the lower half:  shift = (k%4)*8 + (k/4)
 *   -> for k = c*4+kk:  shift_lo = kk*8 + c,  shift_hi(k+16) = kk*8 + c + 4
 * Uses uint4 / uint wide loads + factored scale/min accumulation.
 * --------------------------------------------------------------------- */
inline float q5k_block_dot(const __global uchar *pqs_blk,
                           const __global uchar *pqh_blk,
                           const float *inp,
                           const float *scale, const float *minv)
{
    const __global uint4 *pw = (const __global uint4 *)pqs_blk;
    const __global uint  *ph = (const __global uint  *)pqh_blk;  /* 8 words */
    float acc = 0.0f;

    unroll_hint
    for (int j = 0; j < 8; j++) {
        uint4 w  = pw[j];
        uint  qh = ph[j];
        uint  ws[4] = { w.x, w.y, w.z, w.w };
        const float *in0 = inp + j * 32;
        const float *in1 = inp + j * 32 + 16;

        float dot  = 0.0f;
        float isum = 0.0f;
        unroll_hint
        for (int c = 0; c < 4; c++) {
            uint u = ws[c];
            unroll_hint
            for (int kk = 0; kk < 4; kk++) {
                int   sh   = kk * 8 + c;
                float w0   = (float)((u >> (kk*8))     & 0xFu)
                           + (float)((qh >> sh)        & 1u) * 16.0f;
                float w1   = (float)((u >> (kk*8 + 4)) & 0xFu)
                           + (float)((qh >> (sh + 4))  & 1u) * 16.0f;
                float a = in0[c*4 + kk];
                float b = in1[c*4 + kk];
                dot  = fma(a, w0, dot);
                dot  = fma(b, w1, dot);
                isum += a + b;
            }
        }
        acc = fma(scale[j],  dot,  acc);
        acc = fma(-minv[j],  isum, acc);
    }
    return acc;
}

/* Load 256 activations into a private buffer using float8 vector reads. */
inline void q5k_load_inp(float *inp, const __global float *src)
{
    const __global float8 *pin = (const __global float8 *)src;
    float8 *pv = (float8 *)inp;
    unroll_hint
    for (int i = 0; i < 32; i++)
        pv[i] = pin[i];
}

/* Global-input variant of q5k_block_dot (no private staging), for the
 * tiny-workgroup down kernel. */
inline float q5k_block_dot_g(const __global uchar *pqs_blk,
                             const __global uchar *pqh_blk,
                             const __global float *inp,
                             const float *scale, const float *minv)
{
    const __global uint4  *pw  = (const __global uint4  *)pqs_blk;
    const __global uint   *ph  = (const __global uint   *)pqh_blk;
    const __global float4 *pin = (const __global float4 *)inp;
    float acc = 0.0f;

    unroll_hint
    for (int j = 0; j < 8; j++) {
        uint4 w  = pw[j];
        uint  qh = ph[j];
        uint  ws[4] = { w.x, w.y, w.z, w.w };
        float4 in0[4] = { pin[j*8+0], pin[j*8+1], pin[j*8+2], pin[j*8+3] };
        float4 in1[4] = { pin[j*8+4], pin[j*8+5], pin[j*8+6], pin[j*8+7] };

        float dot  = 0.0f;
        float isum = 0.0f;
        unroll_hint
        for (int c = 0; c < 4; c++) {
            uint u = ws[c];
            float ain[4] = { in0[c].x, in0[c].y, in0[c].z, in0[c].w };
            float bin[4] = { in1[c].x, in1[c].y, in1[c].z, in1[c].w };
            unroll_hint
            for (int kk = 0; kk < 4; kk++) {
                int   sh = kk * 8 + c;
                float w0 = (float)((u >> (kk*8))     & 0xFu) + (float)((qh >> sh)       & 1u) * 16.0f;
                float w1 = (float)((u >> (kk*8 + 4)) & 0xFu) + (float)((qh >> (sh + 4)) & 1u) * 16.0f;
                float a = ain[kk];
                float b = bin[kk];
                dot  = fma(a, w0, dot);
                dot  = fma(b, w1, dot);
                isum += a + b;
            }
        }
        acc = fma(scale[j],  dot,  acc);
        acc = fma(-minv[j],  isum, acc);
    }
    return acc;
}

/* =======================================================================
 * Kernel: group_up_gate_q5k
 * ===================================================================== */
#define OPG_UG5 8
#define BS_UG5  1

__kernel
void group_up_gate_q5k(
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
    uint expert_size        = num_blocks * 176;  /* Q5K: 128 qs + 32 qh + 16 sl */

    uint local_thread = num_blocks_per_row / BS_UG5;

    uint h  = get_group_id(0);
    uint hh = get_local_id(0);
    uint v  = get_group_id(1);
    uint t  = get_group_id(2);

    uint index = indexes ? indexes[t * aligned_output_num + v] : v;

    __local float slm_up  [64 * OPG_UG5];
    __local float slm_gate[64 * OPG_UG5];

    float outUp  [OPG_UG5]; for (int l = 0; l < OPG_UG5; l++) outUp[l]   = 0.0f;
    float outGate[OPG_UG5]; for (int l = 0; l < OPG_UG5; l++) outGate[l] = 0.0f;

    for (uint c = 0; c < BS_UG5; c++) {
        uint blockIdx = hh * BS_UG5 + c;

        float inp[256];
        q5k_load_inp(inp, inputs + (size_t)t * input_len + 256 * blockIdx);

        /* --- up projection --- */
        {
            const __global uchar *pqs_w = ups + (size_t)index * expert_size;
            const __global uchar *pqh_w = pqs_w + (size_t)num_blocks * 128;
            const __global uchar *psl_w = pqh_w + (size_t)num_blocks * 32;

            unroll_hint
            for (int l = 0; l < OPG_UG5; l++) {
                uint blk_idx = h * OPG_UG5 * num_blocks_per_row + (uint)l * num_blocks_per_row + blockIdx;
                float scale[8], minv[8];
                q5k_decode_scales(psl_w + (size_t)blk_idx * 16, scale, minv);
                outUp[l] += q5k_block_dot(pqs_w + (size_t)blk_idx * 128,
                                          pqh_w + (size_t)blk_idx * 32, inp, scale, minv);
            }
        }

        /* --- gate projection --- */
        {
            const __global uchar *pqs_w = gates + (size_t)index * expert_size;
            const __global uchar *pqh_w = pqs_w + (size_t)num_blocks * 128;
            const __global uchar *psl_w = pqh_w + (size_t)num_blocks * 32;

            unroll_hint
            for (int l = 0; l < OPG_UG5; l++) {
                uint blk_idx = h * OPG_UG5 * num_blocks_per_row + (uint)l * num_blocks_per_row + blockIdx;
                float scale[8], minv[8];
                q5k_decode_scales(psl_w + (size_t)blk_idx * 16, scale, minv);
                outGate[l] += q5k_block_dot(pqs_w + (size_t)blk_idx * 128,
                                            pqh_w + (size_t)blk_idx * 32, inp, scale, minv);
            }
        }
    }

    for (int l = 0; l < OPG_UG5; l++) {
        slm_up  [hh * OPG_UG5 + l] = outUp[l];
        slm_gate[hh * OPG_UG5 + l] = outGate[l];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (hh == 0) {
        float accUp  [OPG_UG5]; for (int l = 0; l < OPG_UG5; l++) accUp[l]   = 0.0f;
        float accGate[OPG_UG5]; for (int l = 0; l < OPG_UG5; l++) accGate[l] = 0.0f;
        for (uint i = 0; i < local_thread; i++) {
            for (int l = 0; l < OPG_UG5; l++) {
                accUp[l]   += slm_up  [i * OPG_UG5 + l];
                accGate[l] += slm_gate[i * OPG_UG5 + l];
            }
        }
        /* SiLU(gate) * up  ==  sigmoid(gate) * gate * up */
        for (int l = 0; l < OPG_UG5; l++) {
            float g       = accGate[l];
            float sig     = 1.0f / (1.0f + exp(-g));
            float silu_g  = sig * g;
            outputs[(size_t)t * output_num * output_len + (size_t)v * output_len + h * OPG_UG5 + l] =
                silu_g * accUp[l];
        }
    }
}

/* =======================================================================
 * Kernel: down_merge_q5k
 * ===================================================================== */
#define OPG_DM5 4

__kernel
void down_merge_q5k(
    __global const float  *inputs,       /* [token_len * input_num * input_len] */
    __global const uchar  *downs,        /* [num_experts * expert_size], expert_size = num_blocks*176 */
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
    uint expert_size        = num_blocks * 176;  /* Q5K: 128 qs + 32 qh + 16 sl */

    uint h  = get_group_id(0);
    uint hh = get_local_id(0);
    uint t  = get_group_id(1);

    __local float slm[64 * OPG_DM5];

    float outData[OPG_DM5]; for (int l = 0; l < OPG_DM5; l++) outData[l] = 0.0f;

    for (int e = 0; e < (int)input_num; e++) {
        uint  index     = indexes ? indexes[t * aligned_input_num + e] : (uint)e;
        float efficient = efficients[t * aligned_input_num + e];

        const __global uchar *pqs_w = downs + (size_t)index * expert_size;
        const __global uchar *pqh_w = pqs_w + (size_t)num_blocks * 128;
        const __global uchar *psl_w = pqh_w + (size_t)num_blocks * 32;

        const __global float *inp = inputs + (size_t)t * input_num * input_len
                                  + (size_t)e * input_len + 256 * hh;

        unroll_hint
        for (int l = 0; l < OPG_DM5; l++) {
            uint blk_idx = h * OPG_DM5 * num_blocks_per_row + (uint)l * num_blocks_per_row + hh;
            float scale[8], minv[8];
            q5k_decode_scales(psl_w + (size_t)blk_idx * 16, scale, minv);
            outData[l] += q5k_block_dot_g(pqs_w + (size_t)blk_idx * 128,
                                          pqh_w + (size_t)blk_idx * 32, inp, scale, minv) * efficient;
        }
    }

    for (int l = 0; l < OPG_DM5; l++)
        slm[hh * OPG_DM5 + l] = outData[l];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (hh == 0) {
        float accOut[OPG_DM5]; for (int l = 0; l < OPG_DM5; l++) accOut[l] = 0.0f;
        for (uint i = 0; i < num_blocks_per_row; i++)
            for (int l = 0; l < OPG_DM5; l++)
                accOut[l] += slm[i * OPG_DM5 + l];
        if (is_acc) {
            for (int l = 0; l < OPG_DM5; l++)
                accOut[l] += outputs[(size_t)t * output_len + h * OPG_DM5 + l];
        }
        for (int l = 0; l < OPG_DM5; l++)
            outputs[(size_t)t * output_len + h * OPG_DM5 + l] = accOut[l];
    }
}
