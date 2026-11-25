/*
 * gemv_q4k_shuffle.cl
 *
 * OpenCL kernels for Q4_K GEMV on shuffled weight layout.
 *
 * Shuffled layout (produced by ShuffleQ4KL1):
 *   pqs  [num_blocks * 128]  -- packed 4-bit quants, 16 rows interleaved per 128-byte block
 *   psl  [num_blocks * 16]   -- scales/mins/d/dmin packed as:
 *       bytes [0..3]   : sl   (scale low  nibbles, 8 values in 4 bytes)
 *       bytes [4..7]   : ml   (min   low  nibbles, 8 values in 4 bytes)
 *       bytes [8..9]   : sh   (scale high 2-bits, 8 values in 2 bytes)
 *       bytes [10..11] : mh   (min   high 2-bits, 8 values in 2 bytes)
 *       bytes [12..13] : d    (fp16 super-scale)
 *       bytes [14..15] : dmin (fp16 super-min)
 *
 * Per shuffled block the 256-element row is split into 8 sub-blocks of 32:
 *   sub-block j stores weights [j*32 .. j*32+31] with low nibble in qs[j*16..j*16+15]
 *   and high nibble packed alongside them.
 *
 * Scale/min reconstruction (identical to SYCL kernel):
 *   scalequant[i] = (sl >> (i*4)) & 0xF  |  ((sh >> (i*2)) & 0x3) << 4
 *   scale[i]      = scalequant[i] * d
 *   minquant[i]   = (ml >> (i*4)) & 0xF  |  ((mh >> (i*2)) & 0x3) << 4
 *   min[i]        = minquant[i]   * dmin
 *
 * Kernels provided:
 *   gemv_q4k          -- plain GEMV  (token_len == 1)
 *   group_up_gate_q4k -- MoE up+gate fused GEMV with SiLU activation
 *   down_merge_q4k    -- MoE down GEMV with expert accumulation
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* -----------------------------------------------------------------------
 * Helper: load fp16 from two bytes (little-endian)
 * --------------------------------------------------------------------- */
inline float load_fp16(const __global uchar *p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_half(u);
}

/* -----------------------------------------------------------------------
 * Helper: reconstruct scale/min arrays for one Q4_K block
 *   sl  = uint (4-byte little-endian) at psl+0
 *   ml  = uint (4-byte little-endian) at psl+4
 *   sh  = ushort at psl+8
 *   mh  = ushort at psl+10
 *   d   = fp16  at psl+12
 *   dmin= fp16  at psl+14
 * --------------------------------------------------------------------- */
inline void decode_scales_q4k(const __global uchar *psl,
                                float *scale_out, float *min_out)
{
    uint  sl   = (uint)psl[0] | ((uint)psl[1]<<8) | ((uint)psl[2]<<16) | ((uint)psl[3]<<24);
    uint  ml   = (uint)psl[4] | ((uint)psl[5]<<8) | ((uint)psl[6]<<16) | ((uint)psl[7]<<24);
    uint  sh   = (uint)psl[8] | ((uint)psl[9]<<8);
    uint  mh   = (uint)psl[10]| ((uint)psl[11]<<8);
    float d    = load_fp16(psl + 12);
    float dmin = load_fp16(psl + 14);

    for (int i = 0; i < 8; i++) {
        uint sq = ((sl >> (i*4)) & 0xF) | (((sh >> (i*2)) & 0x3) << 4);
        uint mq = ((ml >> (i*4)) & 0xF) | (((mh >> (i*2)) & 0x3) << 4);
        scale_out[i] = (float)sq * d;
        min_out[i]   = (float)mq * dmin;
    }
}

/* -----------------------------------------------------------------------
 * dot product for one Q4_K sub-block j (32 weights) against 32 inputs
 *   pqs_blk : pointer to start of 128-byte qs section for this block
 *   input   : 256 floats for this block row
 * --------------------------------------------------------------------- */
inline float q4k_subblock_dot(const __global uchar *pqs_blk,
                               const float *input,
                               float scale, float min_val,
                               int j)
{
    float acc = 0.0f;
    for (int k = 0; k < 16; k++) {
        uchar byte = pqs_blk[j * 16 + k];
        float w0 = (float)(byte & 0x0F);
        float w1 = (float)((byte >> 4) & 0x0F);
        w0 = w0 * scale - min_val;
        w1 = w1 * scale - min_val;
        acc += input[j * 32 + k]      * w0;
        acc += input[j * 32 + 16 + k] * w1;
    }
    return acc;
}

/* =======================================================================
 * Kernel 1: gemv_q4k
 *
 * Plain GEMV: y[output_len] = W[output_len, input_len] * x[input_len]
 *
 * Layout:
 *   pqs[num_blocks * 128]     -- quants
 *   psl[num_blocks * 16]      -- scales/d
 *   num_blocks = input_len * output_len / 256
 *
 * Launch:
 *   global_size = (output_len / OPG) * num_blocks_per_row
 *   local_size  = num_blocks_per_row        (= input_len / 256)
 *   OPG         = outputs_per_group (passed as #define or constant, here 16)
 *
 * Each work-group processes OPG output rows. Threads within the group split
 * the K dimension (num_blocks_per_row blocks). Partial sums are reduced via
 * local memory.
 * ===================================================================== */
#define OPG_Q4K 16

__kernel
void gemv_q4k(
    __global const float  *inputs,      /* [input_len] */
    __global const uchar  *pqs,         /* [num_blocks * 128] */
    __global const uchar  *psl,         /* [num_blocks * 16]  */
    __global       float  *outputs,     /* [output_len] */
    uint input_len,
    uint output_len
) {
    uint num_blocks_per_row = input_len / 256;
    uint h  = get_group_id(0);   /* output group index (group covers OPG_Q4K outputs) */
    uint hh = get_local_id(0);   /* block index within row */

    __local float slm[64 * OPG_Q4K];  /* partial sums: [num_blocks_per_row * OPG_Q4K] */

    /* Load 256 inputs for this thread's block */
    float inp[256];
    for (int i = 0; i < 256; i++)
        inp[i] = inputs[256 * hh + i];

    /* Compute OPG_Q4K output elements */
    float outData[OPG_Q4K];
    for (int l = 0; l < OPG_Q4K; l++) outData[l] = 0.0f;

    for (int l = 0; l < OPG_Q4K; l++) {
        uint blk_idx = h * OPG_Q4K * num_blocks_per_row
                       + (uint)l * num_blocks_per_row + hh;
        const __global uchar *qs_blk = pqs + blk_idx * 128;
        const __global uchar *sl_blk = psl + blk_idx * 16;

        float scale[8], minv[8];
        decode_scales_q4k(sl_blk, scale, minv);

        float acc = 0.0f;
        for (int j = 0; j < 8; j++)
            acc += q4k_subblock_dot(qs_blk, inp, scale[j], minv[j], j);
        outData[l] = acc;
    }

    for (int l = 0; l < OPG_Q4K; l++)
        slm[hh * OPG_Q4K + l] = outData[l];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (hh == 0) {
        float accOut[OPG_Q4K];
        for (int l = 0; l < OPG_Q4K; l++) accOut[l] = 0.0f;
        for (uint i = 0; i < num_blocks_per_row; i++)
            for (int l = 0; l < OPG_Q4K; l++)
                accOut[l] += slm[i * OPG_Q4K + l];
        for (int l = 0; l < OPG_Q4K; l++)
            outputs[h * OPG_Q4K + l] = accOut[l];
    }
}

/* =======================================================================
 * Kernel 2: group_up_gate_q4k
 *
 * MoE fused up+gate GEMV with SiLU activation.
 * For each expert v (from indexes[t * aligned_output_num + v]) and token t:
 *   up_out   = W_up  [v] * x[t]
 *   gate_out = W_gate[v] * x[t]
 *   out[t,v] = silu(gate_out) * up_out
 *
 * Template params OPG=8, BS=1 (optimal per benchmark).
 *
 * Buffer layout:
 *   ups   : [num_experts * expert_size]  (expert_size = num_blocks * 144)
 *   gates : [num_experts * expert_size]
 *   indexes : [token_len * aligned_output_num]  (uint32, nullable)
 *
 * Launch:
 *   global = (output_len/OPG * local_per_group, output_num, token_len)
 *   local  = (local_per_group, 1, 1)
 *   local_per_group = num_blocks_per_row / BS
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
        for (int i = 0; i < 256; i++)
            inp[i] = inputs[t * input_len + 256 * blockIdx + i];

        /* --- up projection --- */
        {
            const __global uchar *weights = ups + index * expert_size;
            const __global uchar *pqs_w   = weights;
            const __global uchar *psl_w   = pqs_w + num_blocks * 128;

            for (int l = 0; l < OPG_UG; l++) {
                uint blk_idx = h * OPG_UG * num_blocks_per_row + (uint)l * num_blocks_per_row + blockIdx;
                float scale[8], minv[8];
                decode_scales_q4k(psl_w + blk_idx * 16, scale, minv);
                float acc = 0.0f;
                for (int j = 0; j < 8; j++)
                    acc += q4k_subblock_dot(pqs_w + blk_idx * 128, inp, scale[j], minv[j], j);
                outUp[l] += acc;
            }
        }

        /* --- gate projection --- */
        {
            const __global uchar *weights = gates + index * expert_size;
            const __global uchar *pqs_w   = weights;
            const __global uchar *psl_w   = pqs_w + num_blocks * 128;

            for (int l = 0; l < OPG_UG; l++) {
                uint blk_idx = h * OPG_UG * num_blocks_per_row + (uint)l * num_blocks_per_row + blockIdx;
                float scale[8], minv[8];
                decode_scales_q4k(psl_w + blk_idx * 16, scale, minv);
                float acc = 0.0f;
                for (int j = 0; j < 8; j++)
                    acc += q4k_subblock_dot(pqs_w + blk_idx * 128, inp, scale[j], minv[j], j);
                outGate[l] += acc;
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
        /* SiLU(gate) * up */
        for (int l = 0; l < OPG_UG; l++) {
            float g = accGate[l];
            float silu_g = g / (1.0f + exp(-g));  /* silu(x) = x * sigmoid(x) */
            outputs[t * output_num * output_len + v * output_len + h * OPG_UG + l] =
                silu_g * accUp[l];
        }
    }
}

/* =======================================================================
 * Kernel 3: down_merge_q4k
 *
 * MoE down projection: weighted sum over input_num experts.
 *   out[t] += efficient[e] * W_down[index[e]] * in[t, e]
 *
 * Launch:
 *   global = (output_len/OPG * num_blocks_per_row, token_len)
 *   local  = (num_blocks_per_row, 1)
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

        const __global uchar *weights = downs + index * expert_size;
        const __global uchar *pqs_w   = weights;
        const __global uchar *psl_w   = pqs_w + num_blocks * 128;

        float inp[256];
        for (int i = 0; i < 256; i++)
            inp[i] = inputs[t * input_num * input_len + e * input_len + 256 * hh + i];

        float expertOut[OPG_DM];
        for (int l = 0; l < OPG_DM; l++) {
            uint blk_idx = h * OPG_DM * num_blocks_per_row + (uint)l * num_blocks_per_row + hh;
            float scale[8], minv[8];
            decode_scales_q4k(psl_w + blk_idx * 16, scale, minv);
            float acc = 0.0f;
            for (int j = 0; j < 8; j++)
                acc += q4k_subblock_dot(pqs_w + blk_idx * 128, inp, scale[j], minv[j], j);
            expertOut[l] = acc;
        }

        for (int l = 0; l < OPG_DM; l++)
            outData[l] += expertOut[l] * efficient;
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
                accOut[l] += outputs[t * output_len + h * OPG_DM + l];
        }
        for (int l = 0; l < OPG_DM; l++)
            outputs[t * output_len + h * OPG_DM + l] = accOut[l];
    }
}
