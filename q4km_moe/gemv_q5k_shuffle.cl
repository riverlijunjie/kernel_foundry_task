/*
 * gemv_q5k_shuffle.cl
 *
 * OpenCL kernels for Q5_K GEMV on shuffled weight layout.
 *
 * Shuffled layout (produced by ShuffleQ5KL1):
 *   pqs  [num_blocks * 128]  -- packed 4-bit lower quants (same packing as Q4_K)
 *   pqh  [num_blocks *  32]  -- packed high bits (1 bit per weight → 8 bits pack 8 weights)
 *   psl  [num_blocks *  16]  -- scales/mins/d/dmin (same 16-byte structure as Q4_K)
 *
 * Per shuffled block (256 weights):
 *   Sub-block j (j=0..7) covers 32 weights at positions [j*32 .. j*32+31].
 *   qs layout: qs[j*16+k] = low4(w[j*32+k]) | (low4(w[j*32+16+k]) << 4)
 *   qh layout: qh[j*4+k]  packs the high bit of 8 weights:
 *       bit s of qh[j*4+k] = high_bit(w[j*32 + k + s*4])   for s=0..7
 *   So weight value = low_nibble(qs) | (high_bit(qh) << 4)  (range 0..31)
 *
 * Scale/min layout identical to Q4_K (see gemv_q4k_shuffle.cl).
 *
 * Kernels provided:
 *   gemv_q5k          -- plain GEMV
 *   down_merge_q5k    -- MoE down GEMV with expert accumulation
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* -----------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------- */
inline float load_fp16_q5k(const __global uchar *p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_half(u);
}

inline void decode_scales_q5k(const __global uchar *psl,
                                float *scale_out, float *min_out)
{
    uint  sl   = (uint)psl[0] | ((uint)psl[1]<<8) | ((uint)psl[2]<<16) | ((uint)psl[3]<<24);
    uint  ml   = (uint)psl[4] | ((uint)psl[5]<<8) | ((uint)psl[6]<<16) | ((uint)psl[7]<<24);
    uint  sh   = (uint)psl[8] | ((uint)psl[9]<<8);
    uint  mh   = (uint)psl[10]| ((uint)psl[11]<<8);
    float d    = load_fp16_q5k(psl + 12);
    float dmin = load_fp16_q5k(psl + 14);

    for (int i = 0; i < 8; i++) {
        uint sq = ((sl >> (i*4)) & 0xF) | (((sh >> (i*2)) & 0x3) << 4);
        uint mq = ((ml >> (i*4)) & 0xF) | (((mh >> (i*2)) & 0x3) << 4);
        scale_out[i] = (float)sq * d;
        min_out[i]   = (float)mq * dmin;
    }
}

/* Sub-block dot for Q5_K: sub-block j, 32 weights.
 * pqs_blk : base of 128-byte qs block
 * pqh_blk : base of 32-byte qh block
 * The high bit of weight[j*32 + k + s*4] is stored in bit s of qh[j*4 + k%4].
 *   for k in 0..15: weight index within sub-block = k  (lower half) or k+16 (upper half)
 * Reconstruction per SYCL source:
 *   low nibble  from qs: (qs[j*16+k] & 0x0F) -> weight lower half
 *                        (qs[j*16+k] >> 4)    -> weight upper half
 *   high bit from qh:
 *       high_word = qh_blk[j] as uint32 (4 bytes, little-endian)
 *       The SYCL kernel uses high_value_mask:
 *         bit s of high_word maps to weight position interleaved as:
 *         mask layout [{byte0bit0->w0}, {byte1bit0->w1}, {byte2bit0->w2}, {byte3bit0->w3},
 *                       byte0bit1->w4, ...] i.e. column-major within the 4-byte word.
 *       high_value_mask[i] selects bit: byte (i%4), bit-position (i/4)
 *       So for weight index wi within sub-block:
 *         byte_idx = wi % 4, bit_idx = wi / 4
 *         high_bit = (qh_word >> (byte_idx*8 + bit_idx)) & 1
 */
inline float q5k_subblock_dot(const __global uchar *pqs_blk,
                               const __global uchar *pqh_blk,
                               const float *input,
                               float scale, float min_val,
                               int j)
{
    /* Load the 4-byte qh word for sub-block j */
    uint qh_word = (uint)pqh_blk[j*4]
                 | ((uint)pqh_blk[j*4+1] << 8)
                 | ((uint)pqh_blk[j*4+2] << 16)
                 | ((uint)pqh_blk[j*4+3] << 24);

    float acc = 0.0f;
    for (int k = 0; k < 16; k++) {
        uchar byte = pqs_blk[j * 16 + k];

        /* Lower half: weight index within sub-block = k */
        float w0 = (float)(byte & 0x0F);
        int byte_idx0 = k % 4;
        int bit_idx0  = k / 4;
        if ((qh_word >> (byte_idx0 * 8 + bit_idx0)) & 1u)
            w0 += 16.0f;
        w0 = w0 * scale - min_val;

        /* Upper half: weight index within sub-block = k + 16 */
        float w1 = (float)((byte >> 4) & 0x0F);
        int byte_idx1 = (k + 16) % 4;
        int bit_idx1  = (k + 16) / 4;
        if ((qh_word >> (byte_idx1 * 8 + bit_idx1)) & 1u)
            w1 += 16.0f;
        w1 = w1 * scale - min_val;

        acc += input[j * 32 + k]      * w0;
        acc += input[j * 32 + 16 + k] * w1;
    }
    return acc;
}

/* =======================================================================
 * Kernel 1: gemv_q5k
 *
 * Plain GEMV.
 * Launch:
 *   global_size = (output_len / OPG_Q5K) * num_blocks_per_row
 *   local_size  = num_blocks_per_row
 * ===================================================================== */
#define OPG_Q5K 16

__kernel
void gemv_q5k(
    __global const float  *inputs,
    __global const uchar  *pqs,
    __global const uchar  *pqh,
    __global const uchar  *psl,
    __global       float  *outputs,
    uint input_len,
    uint output_len
) {
    uint num_blocks_per_row = input_len / 256;
    uint h  = get_group_id(0);
    uint hh = get_local_id(0);

    __local float slm[64 * OPG_Q5K];

    float inp[256];
    for (int i = 0; i < 256; i++)
        inp[i] = inputs[256 * hh + i];

    float outData[OPG_Q5K]; for (int l = 0; l < OPG_Q5K; l++) outData[l] = 0.0f;

    for (int l = 0; l < OPG_Q5K; l++) {
        uint blk_idx = h * OPG_Q5K * num_blocks_per_row
                       + (uint)l * num_blocks_per_row + hh;
        float scale[8], minv[8];
        decode_scales_q5k(psl + blk_idx * 16, scale, minv);
        float acc = 0.0f;
        for (int j = 0; j < 8; j++)
            acc += q5k_subblock_dot(pqs + blk_idx * 128, pqh + blk_idx * 32,
                                    inp, scale[j], minv[j], j);
        outData[l] = acc;
    }

    for (int l = 0; l < OPG_Q5K; l++)
        slm[hh * OPG_Q5K + l] = outData[l];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (hh == 0) {
        float accOut[OPG_Q5K]; for (int l = 0; l < OPG_Q5K; l++) accOut[l] = 0.0f;
        for (uint i = 0; i < num_blocks_per_row; i++)
            for (int l = 0; l < OPG_Q5K; l++)
                accOut[l] += slm[i * OPG_Q5K + l];
        for (int l = 0; l < OPG_Q5K; l++)
            outputs[h * OPG_Q5K + l] = accOut[l];
    }
}

/* =======================================================================
 * Kernel 2: down_merge_q5k
 *
 * MoE down projection.
 * Launch:
 *   global = (output_len/OPG_DM5 * num_blocks_per_row, token_len)
 *   local  = (num_blocks_per_row, 1)
 * ===================================================================== */
#define OPG_DM5 4

__kernel
void down_merge_q5k(
    __global const float  *inputs,
    __global const uchar  *downs,       /* [num_experts * expert_size], expert_size = num_blocks*176 */
    __global const uint   *indexes,
    __global const float  *efficients,
    __global       float  *outputs,
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

        const __global uchar *weights = downs + index * expert_size;
        const __global uchar *pqs_w   = weights;
        const __global uchar *pqh_w   = pqs_w + num_blocks * 128;
        const __global uchar *psl_w   = pqh_w + num_blocks * 32;

        float inp[256];
        for (int i = 0; i < 256; i++)
            inp[i] = inputs[t * input_num * input_len + e * input_len + 256 * hh + i];

        float expertOut[OPG_DM5];
        for (int l = 0; l < OPG_DM5; l++) {
            uint blk_idx = h * OPG_DM5 * num_blocks_per_row + (uint)l * num_blocks_per_row + hh;
            float scale[8], minv[8];
            decode_scales_q5k(psl_w + blk_idx * 16, scale, minv);
            float acc = 0.0f;
            for (int j = 0; j < 8; j++)
                acc += q5k_subblock_dot(pqs_w + blk_idx * 128, pqh_w + blk_idx * 32,
                                        inp, scale[j], minv[j], j);
            expertOut[l] = acc;
        }

        for (int l = 0; l < OPG_DM5; l++)
            outData[l] += expertOut[l] * efficient;
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
                accOut[l] += outputs[t * output_len + h * OPG_DM5 + l];
        }
        for (int l = 0; l < OPG_DM5; l++)
            outputs[t * output_len + h * OPG_DM5 + l] = accOut[l];
    }
}
