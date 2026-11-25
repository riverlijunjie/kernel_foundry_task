/*
 * reference_q4k_q6k.cl
 *
 * Baseline (reference) implementations of plain GEMV for two quantization formats:
 *   gemv_q4k  -- Q4_K shuffled layout (ShuffleQ4KL1)
 *   gemv_q6k  -- Q6_K shuffled layout (ShuffleQ6KL1)
 *
 * These kernels are used as the correctness oracle and performance baseline.
 * Do NOT modify this file.
 *
 * ---- Q4_K shuffled layout (per block, 256 weights) ----
 *   pqs [num_blocks * 128]  packed 4-bit quants
 *   psl [num_blocks * 16]   bytes [0..3] sl (scale low nibbles)
 *                            bytes [4..7] ml (min   low nibbles)
 *                            bytes [8..9] sh (scale high 2-bits)
 *                            bytes [10..11] mh (min high 2-bits)
 *                            bytes [12..13] d  (fp16 super-scale)
 *                            bytes [14..15] dmin (fp16 super-min)
 *
 * ---- Q6_K shuffled layout (per block, 256 weights) ----
 *   pql [num_blocks * 128]  packed lower nibbles (4 bits/weight)
 *   pqh [num_blocks *  64]  packed upper 2 bits  (4 weights/byte)
 *   ps  [num_blocks *  16]  int8 scales (16 per block)
 *   pd  [num_blocks *   2]  fp16 super-scale d
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* [REFERENCE_START] */

/* ===================================================================
 * Q4_K helpers
 * =================================================================== */

inline float load_fp16(const __global uchar *p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_half(u);
}

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

/* -----------------------------------------------------------------------
 * gemv_q4k: plain GEMV  y[output_len] = W[output_len, input_len] * x[input_len]
 *
 * Launch:
 *   global_size = (output_len / OPG_Q4K) * num_blocks_per_row
 *   local_size  = num_blocks_per_row           (= input_len / 256)
 * --------------------------------------------------------------------- */
#define OPG_Q4K 16

__kernel
void gemv_q4k(
    __global const float  *inputs,      /* [input_len]        */
    __global const uchar  *pqs,         /* [num_blocks * 128] */
    __global const uchar  *psl,         /* [num_blocks * 16]  */
    __global       float  *outputs,     /* [output_len]       */
    uint input_len,
    uint output_len
) {
    uint num_blocks_per_row = input_len / 256;
    uint h  = get_group_id(0);
    uint hh = get_local_id(0);

    __local float slm[64 * OPG_Q4K];

    float inp[256];
    for (int i = 0; i < 256; i++)
        inp[i] = inputs[256 * hh + i];

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


/* ===================================================================
 * Q6_K helpers
 * =================================================================== */

inline float load_fp16_q6k(const __global uchar *p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_half(u);
}

/* Bit-shift table for upper-2-bit extraction from the packed qh word.
 * shift[k] is the bit offset for position k (0..15).
 * Pattern: {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30} */
inline uint qh_shift(int k) {
    const uint tbl[16] = {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30};
    return tbl[k];
}

inline float q6k_subblock_dot(const __global uchar *pql_blk,
                               const __global uchar *pqh_blk,
                               const __global char  *ps_blk,
                               float d,
                               const float *input,
                               int j)
{
    float combined_lo = d * (float)ps_blk[2 * j];
    float combined_hi = d * (float)ps_blk[2 * j + 1];

    uint qh_lo = (uint)pqh_blk[j*8+0] | ((uint)pqh_blk[j*8+1]<<8)
               | ((uint)pqh_blk[j*8+2]<<16) | ((uint)pqh_blk[j*8+3]<<24);
    uint qh_hi = (uint)pqh_blk[j*8+4] | ((uint)pqh_blk[j*8+5]<<8)
               | ((uint)pqh_blk[j*8+6]<<16) | ((uint)pqh_blk[j*8+7]<<24);

    float acc = 0.0f;
    for (int k = 0; k < 16; k++) {
        uchar ql_byte = pql_blk[j * 16 + k];
        uint  sh = qh_shift(k);

        float w0 = (float)((ql_byte & 0x0F) | (((qh_lo >> sh) & 0x03u) << 4));
        w0 = (w0 - 32.0f) * combined_lo;

        float w1 = (float)(((ql_byte >> 4) & 0x0F) | (((qh_hi >> sh) & 0x03u) << 4));
        w1 = (w1 - 32.0f) * combined_hi;

        acc += input[j * 32 + k]      * w0;
        acc += input[j * 32 + 16 + k] * w1;
    }
    return acc;
}

/* -----------------------------------------------------------------------
 * gemv_q6k: plain GEMV  y[output_len] = W * x[input_len]
 *
 * Launch:
 *   local_range = num_blocks_per_row / BS_Q6K
 *   global_size = (output_len / OPG_Q6K) * local_range
 *   local_size  = local_range
 * --------------------------------------------------------------------- */
#define OPG_Q6K 16
#define BS_Q6K  2

__kernel
void gemv_q6k(
    __global const float  *inputs,      /* [input_len]        */
    __global const uchar  *pql,         /* [num_blocks * 128] */
    __global const uchar  *pqh,         /* [num_blocks *  64] */
    __global const char   *ps,          /* [num_blocks *  16] int8 */
    __global const uchar  *pd_buf,      /* [num_blocks *   2] fp16 */
    __global       float  *outputs,     /* [output_len]       */
    uint input_len,
    uint output_len
) {
    uint num_blocks_per_row = input_len / 256;
    uint local_range = num_blocks_per_row / BS_Q6K;
    uint h  = get_group_id(0);
    uint hh = get_local_id(0);

    __local float slm[64 * OPG_Q6K];

    float outData[OPG_Q6K]; for (int l = 0; l < OPG_Q6K; l++) outData[l] = 0.0f;

    for (uint c = 0; c < BS_Q6K; c++) {
        uint blockIdx = hh * BS_Q6K + c;
        float inp[256];
        for (int i = 0; i < 256; i++)
            inp[i] = inputs[256 * blockIdx + i];

        for (int l = 0; l < OPG_Q6K; l++) {
            uint blk_idx = h * OPG_Q6K * num_blocks_per_row
                           + (uint)l * num_blocks_per_row + blockIdx;
            float d = load_fp16_q6k(pd_buf + blk_idx * 2);
            float acc = 0.0f;
            for (int j = 0; j < 8; j++)
                acc += q6k_subblock_dot(pql + blk_idx * 128, pqh + blk_idx * 64,
                                        ps  + blk_idx * 16,  d, inp, j);
            outData[l] += acc;
        }
    }

    for (int l = 0; l < OPG_Q6K; l++)
        slm[hh * OPG_Q6K + l] = outData[l];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (hh == 0) {
        float accOut[OPG_Q6K]; for (int l = 0; l < OPG_Q6K; l++) accOut[l] = 0.0f;
        for (uint i = 0; i < local_range; i++)
            for (int l = 0; l < OPG_Q6K; l++)
                accOut[l] += slm[i * OPG_Q6K + l];
        for (int l = 0; l < OPG_Q6K; l++)
            outputs[h * OPG_Q6K + l] = accOut[l];
    }
}

/* [REFERENCE_END] */
