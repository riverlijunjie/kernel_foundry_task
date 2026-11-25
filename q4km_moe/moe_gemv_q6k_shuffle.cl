/*
 * moe_gemv_q6k_shuffle.cl
 *
 * OpenCL MoE kernels for Q6_K on shuffled weight layout, translated from the
 * SYCL/ESIMD sources:
 *   runSparseGemvQ6KL1_xve_impl<OPG, BS>  (gemv_q6k_L1.xve.cpp)  -- adapted
 *                                            into a fused up+gate MoE kernel
 *   runDownMergeQ6KL1_xve_impl<OPG>       (gemv_q6k_L1.xve.cpp)
 *
 * ============================ OPTIMIZATION NOTES ==========================
 * Memory-bound MoE GEMV.  Same bandwidth-oriented rewrite as
 * moe_gemv_q4k_shuffle.cl (see notes there):
 *   1. 128-byte low-nibble block (ql) read with uint4 wide loads.
 *   2. 64-byte high-2-bit block (qh) read with uint loads (2 uints/sub-block).
 *   3. 256 activations loaded with float8 vector reads.
 *   4. Per-half (lower/upper) scale factored out of the inner loop:
 *        acc += clo*(dot_lo - 32*isum_lo) + chi*(dot_hi - 32*isum_hi)
 *      which removes the per-weight "-32" subtract and per-weight scale mul.
 * A raw intel_sub_group_block_read is not applicable to this fixed shuffle
 * layout (each lane owns a whole contiguous 128-byte K-block, not the
 * interleaved layout block-reads require); the wide vector loads give the
 * same maximal-width DRAM transactions while keeping the layout, kernel
 * signatures and results bit-identical.
 * ==========================================================================
 *
 * Shuffled layout (produced by ShuffleQ6KL1):
 *   pql  [num_blocks * 128]  -- packed lower nibbles (4 bits per weight)
 *   pqh  [num_blocks *  64]  -- packed upper 2 bits (4 weights per byte)
 *   ps   [num_blocks *  16]  -- int8 scales (16 per block)
 *   pd   [num_blocks *   2]  -- fp16 super-scale d
 *   expert_size = num_blocks * 210  (128+64+16+2)
 *
 * shift table (same as SYCL): {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30}
 *   upper_bits[k] = (qh_word >> shift[k]) & 0x03  (for k=0..15)
 * Full weight: lower_half[k] = ql_lo[k] | (upper_bits_lo[k] << 4)  (range 0..63)
 *              upper_half[k] = ql_hi[k] | (upper_bits_hi[k] << 4)
 * Dequantized:  (weight - 32) * d * scale[2*j]    lower half
 *               (weight - 32) * d * scale[2*j+1]  upper half
 *
 * Kernels provided:
 *   group_up_gate_q6k -- MoE up+gate fused GEMV with SiLU(gate) * up activation
 *   down_merge_q6k    -- MoE down GEMV with per-expert weighted accumulation
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define unroll_hint __attribute__((opencl_unroll_hint))

__constant uint Q6K_SHIFT[16] = {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30};

inline float q6k_load_fp16(const __global uchar *p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_half(u);
}

/* -----------------------------------------------------------------------
 * Whole-block dot product for one Q6_K block (256 weights, 8 sub-blocks).
 *   pql_blk : base of 128-byte ql section (low nibbles)
 *   pqh_blk : base of 64-byte qh section  (upper 2 bits)
 *   ps_blk  : base of 16-byte int8 scale section
 *   d       : fp16 super-scale (as float)
 * Uses uint4 / uint wide loads + per-half factored scale accumulation.
 * --------------------------------------------------------------------- */
inline float q6k_block_dot(const __global uchar *pql_blk,
                           const __global uchar *pqh_blk,
                           const __global char  *ps_blk,
                           float d,
                           const float *inp)
{
    const __global uint4 *pw = (const __global uint4 *)pql_blk;
    const __global uint  *ph = (const __global uint  *)pqh_blk;  /* 16 words */
    float acc = 0.0f;

    unroll_hint
    for (int j = 0; j < 8; j++) {
        uint4 w     = pw[j];
        uint  qh_lo = ph[j*2 + 0];
        uint  qh_hi = ph[j*2 + 1];
        uint  ws[4] = { w.x, w.y, w.z, w.w };
        float clo = d * (float)ps_blk[2*j + 0];
        float chi = d * (float)ps_blk[2*j + 1];
        const float *in0 = inp + j * 32;
        const float *in1 = inp + j * 32 + 16;

        float dot_lo = 0.0f, dot_hi = 0.0f;
        float sum_lo = 0.0f, sum_hi = 0.0f;
        unroll_hint
        for (int c = 0; c < 4; c++) {
            uint u = ws[c];
            unroll_hint
            for (int kk = 0; kk < 4; kk++) {
                int  k  = c*4 + kk;
                uint sh = Q6K_SHIFT[k];
                float w0 = (float)(((u >> (kk*8))     & 0xFu) | (((qh_lo >> sh) & 0x3u) << 4));
                float w1 = (float)(((u >> (kk*8 + 4)) & 0xFu) | (((qh_hi >> sh) & 0x3u) << 4));
                float a = in0[k];
                float b = in1[k];
                dot_lo = fma(a, w0, dot_lo); sum_lo += a;
                dot_hi = fma(b, w1, dot_hi); sum_hi += b;
            }
        }
        acc = fma(clo, dot_lo - 32.0f * sum_lo, acc);
        acc = fma(chi, dot_hi - 32.0f * sum_hi, acc);
    }
    return acc;
}

/* Load 256 activations into a private buffer using float8 vector reads. */
inline void q6k_load_inp(float *inp, const __global float *src)
{
    const __global float8 *pin = (const __global float8 *)src;
    float8 *pv = (float8 *)inp;
    unroll_hint
    for (int i = 0; i < 32; i++)
        pv[i] = pin[i];
}

/* Global-input variant of q6k_block_dot (no private staging), for the
 * tiny-workgroup down kernel. */
inline float q6k_block_dot_g(const __global uchar *pql_blk,
                             const __global uchar *pqh_blk,
                             const __global char  *ps_blk,
                             float d,
                             const __global float *inp)
{
    const __global uint4  *pw  = (const __global uint4  *)pql_blk;
    const __global uint   *ph  = (const __global uint   *)pqh_blk;
    const __global float4 *pin = (const __global float4 *)inp;
    float acc = 0.0f;

    unroll_hint
    for (int j = 0; j < 8; j++) {
        uint4 w     = pw[j];
        uint  qh_lo = ph[j*2 + 0];
        uint  qh_hi = ph[j*2 + 1];
        uint  ws[4] = { w.x, w.y, w.z, w.w };
        float clo = d * (float)ps_blk[2*j + 0];
        float chi = d * (float)ps_blk[2*j + 1];
        float4 in0[4] = { pin[j*8+0], pin[j*8+1], pin[j*8+2], pin[j*8+3] };
        float4 in1[4] = { pin[j*8+4], pin[j*8+5], pin[j*8+6], pin[j*8+7] };

        float dot_lo = 0.0f, dot_hi = 0.0f;
        float sum_lo = 0.0f, sum_hi = 0.0f;
        unroll_hint
        for (int c = 0; c < 4; c++) {
            uint u = ws[c];
            float ain[4] = { in0[c].x, in0[c].y, in0[c].z, in0[c].w };
            float bin[4] = { in1[c].x, in1[c].y, in1[c].z, in1[c].w };
            unroll_hint
            for (int kk = 0; kk < 4; kk++) {
                int  k  = c*4 + kk;
                uint sh = Q6K_SHIFT[k];
                float w0 = (float)(((u >> (kk*8))     & 0xFu) | (((qh_lo >> sh) & 0x3u) << 4));
                float w1 = (float)(((u >> (kk*8 + 4)) & 0xFu) | (((qh_hi >> sh) & 0x3u) << 4));
                float a = ain[kk];
                float b = bin[kk];
                dot_lo = fma(a, w0, dot_lo); sum_lo += a;
                dot_hi = fma(b, w1, dot_hi); sum_hi += b;
            }
        }
        acc = fma(clo, dot_lo - 32.0f * sum_lo, acc);
        acc = fma(chi, dot_hi - 32.0f * sum_hi, acc);
    }
    return acc;
}

/* =======================================================================
 * Kernel 1: group_up_gate_q6k
 * ===================================================================== */
#define OPG_UG_Q6 8
#define BS_UG_Q6  1

__kernel
void group_up_gate_q6k(
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
    uint expert_size        = num_blocks * 210;  /* Q6K: 128 ql + 64 qh + 16 ps + 2 pd */

    uint local_thread = num_blocks_per_row / BS_UG_Q6;

    uint h  = get_group_id(0);
    uint hh = get_local_id(0);
    uint v  = get_group_id(1);
    uint t  = get_group_id(2);

    uint index = indexes ? indexes[t * aligned_output_num + v] : v;

    __local float slm_up  [64 * OPG_UG_Q6];
    __local float slm_gate[64 * OPG_UG_Q6];

    float outUp  [OPG_UG_Q6]; for (int l = 0; l < OPG_UG_Q6; l++) outUp[l]   = 0.0f;
    float outGate[OPG_UG_Q6]; for (int l = 0; l < OPG_UG_Q6; l++) outGate[l] = 0.0f;

    for (uint c = 0; c < BS_UG_Q6; c++) {
        uint blockIdx = hh * BS_UG_Q6 + c;

        float inp[256];
        q6k_load_inp(inp, inputs + (size_t)t * input_len + 256 * blockIdx);

        /* --- up projection --- */
        {
            const __global uchar *pql_w = ups + (size_t)index * expert_size;
            const __global uchar *pqh_w = pql_w + (size_t)num_blocks * 128;
            const __global char  *ps_w  = (const __global char *)(pqh_w + (size_t)num_blocks * 64);
            const __global uchar *pd_w  = (const __global uchar *)(ps_w  + (size_t)num_blocks * 16);

            unroll_hint
            for (int l = 0; l < OPG_UG_Q6; l++) {
                uint blk_idx = h * OPG_UG_Q6 * num_blocks_per_row + (uint)l * num_blocks_per_row + blockIdx;
                float d = q6k_load_fp16(pd_w + (size_t)blk_idx * 2);
                outUp[l] += q6k_block_dot(pql_w + (size_t)blk_idx * 128,
                                          pqh_w + (size_t)blk_idx * 64,
                                          ps_w  + (size_t)blk_idx * 16, d, inp);
            }
        }

        /* --- gate projection --- */
        {
            const __global uchar *pql_w = gates + (size_t)index * expert_size;
            const __global uchar *pqh_w = pql_w + (size_t)num_blocks * 128;
            const __global char  *ps_w  = (const __global char *)(pqh_w + (size_t)num_blocks * 64);
            const __global uchar *pd_w  = (const __global uchar *)(ps_w  + (size_t)num_blocks * 16);

            unroll_hint
            for (int l = 0; l < OPG_UG_Q6; l++) {
                uint blk_idx = h * OPG_UG_Q6 * num_blocks_per_row + (uint)l * num_blocks_per_row + blockIdx;
                float d = q6k_load_fp16(pd_w + (size_t)blk_idx * 2);
                outGate[l] += q6k_block_dot(pql_w + (size_t)blk_idx * 128,
                                            pqh_w + (size_t)blk_idx * 64,
                                            ps_w  + (size_t)blk_idx * 16, d, inp);
            }
        }
    }

    for (int l = 0; l < OPG_UG_Q6; l++) {
        slm_up  [hh * OPG_UG_Q6 + l] = outUp[l];
        slm_gate[hh * OPG_UG_Q6 + l] = outGate[l];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (hh == 0) {
        float accUp  [OPG_UG_Q6]; for (int l = 0; l < OPG_UG_Q6; l++) accUp[l]   = 0.0f;
        float accGate[OPG_UG_Q6]; for (int l = 0; l < OPG_UG_Q6; l++) accGate[l] = 0.0f;
        for (uint i = 0; i < local_thread; i++) {
            for (int l = 0; l < OPG_UG_Q6; l++) {
                accUp[l]   += slm_up  [i * OPG_UG_Q6 + l];
                accGate[l] += slm_gate[i * OPG_UG_Q6 + l];
            }
        }
        /* SiLU(gate) * up  ==  sigmoid(gate) * gate * up */
        for (int l = 0; l < OPG_UG_Q6; l++) {
            float g       = accGate[l];
            float sig     = 1.0f / (1.0f + exp(-g));
            float silu_g  = sig * g;
            outputs[(size_t)t * output_num * output_len + (size_t)v * output_len + h * OPG_UG_Q6 + l] =
                silu_g * accUp[l];
        }
    }
}

/* =======================================================================
 * Kernel 2: down_merge_q6k
 * ===================================================================== */
#define OPG_DM6 4

__kernel
void down_merge_q6k(
    __global const float  *inputs,      /* [token_len * input_num * input_len] */
    __global const uchar  *downs,       /* [num_experts * expert_size], expert_size=num_blocks*210 */
    __global const uint   *indexes,     /* [token_len * aligned_input_num] or NULL */
    __global const float  *efficients,  /* [token_len * aligned_input_num] */
    __global       float  *outputs,     /* [token_len * output_len] */
    uint input_num,
    uint aligned_input_num,
    uint token_len,
    uint input_len,
    uint output_len,
    int  is_acc
) {
    uint num_blocks_per_row = input_len / 256;
    uint num_blocks         = input_len * output_len / 256;
    uint expert_size        = num_blocks * 210;  /* Q6K: 128+64+16+2 */

    uint h  = get_group_id(0);
    uint hh = get_local_id(0);
    uint t  = get_group_id(1);

    __local float slm[64 * OPG_DM6];

    float outData[OPG_DM6]; for (int l = 0; l < OPG_DM6; l++) outData[l] = 0.0f;

    for (int e = 0; e < (int)input_num; e++) {
        uint  index     = indexes ? indexes[t * aligned_input_num + e] : (uint)e;
        float efficient = efficients[t * aligned_input_num + e];

        const __global uchar *pql_w = downs + (size_t)index * expert_size;
        const __global uchar *pqh_w = pql_w + (size_t)num_blocks * 128;
        const __global char  *ps_w  = (const __global char *)(pqh_w + (size_t)num_blocks * 64);
        const __global uchar *pd_w  = (const __global uchar *)(ps_w  + (size_t)num_blocks * 16);

        const __global float *inp = inputs + (size_t)t * input_num * input_len
                                  + (size_t)e * input_len + 256 * hh;

        unroll_hint
        for (int l = 0; l < OPG_DM6; l++) {
            uint blk_idx = h * OPG_DM6 * num_blocks_per_row + (uint)l * num_blocks_per_row + hh;
            float d = q6k_load_fp16(pd_w + (size_t)blk_idx * 2);
            outData[l] += q6k_block_dot_g(pql_w + (size_t)blk_idx * 128,
                                          pqh_w + (size_t)blk_idx * 64,
                                          ps_w  + (size_t)blk_idx * 16, d, inp) * efficient;
        }
    }

    for (int l = 0; l < OPG_DM6; l++)
        slm[hh * OPG_DM6 + l] = outData[l];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (hh == 0) {
        float accOut[OPG_DM6]; for (int l = 0; l < OPG_DM6; l++) accOut[l] = 0.0f;
        for (uint i = 0; i < num_blocks_per_row; i++)
            for (int l = 0; l < OPG_DM6; l++)
                accOut[l] += slm[i * OPG_DM6 + l];
        if (is_acc) {
            for (int l = 0; l < OPG_DM6; l++)
                accOut[l] += outputs[(size_t)t * output_len + h * OPG_DM6 + l];
        }
        for (int l = 0; l < OPG_DM6; l++)
            outputs[(size_t)t * output_len + h * OPG_DM6 + l] = accOut[l];
    }
}
