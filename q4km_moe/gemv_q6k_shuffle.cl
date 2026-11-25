/*
 * gemv_q6k_shuffle.cl
 *
 * OpenCL kernels for Q6_K GEMV on shuffled weight layout.
 *
 * Shuffled layout (produced by ShuffleQ6KL1):
 *   pql  [num_blocks * 128]  -- packed lower nibbles (4 bits per weight)
 *   pqh  [num_blocks *  64]  -- packed upper 2 bits (4 weights per byte)
 *   ps   [num_blocks *  16]  -- int8 scales (16 per block)
 *   pd   [num_blocks *   2]  -- fp16 super-scale d
 *   total per block: 210 bytes (128+64+16+2)
 *
 * Per shuffled block (256 weights), sub-block j (j=0..7) covers 32 weights:
 *   ql[j*16+k] packs two weights:
 *       lower_weight_lo[k]  = ql[j*16+k] & 0x0F
 *       upper_weight_lo[k]  = (ql[j*16+k] >> 4) & 0x0F
 *   qh stores the upper 2 bits in interleaved column-major form within a 4-byte word:
 *       qh_word_lo = qh[j*8+0..3] as uint32   (for lower 16 weights)
 *       qh_word_hi = qh[j*8+4..7] as uint32   (for upper 16 weights)
 *       shift table (same as SYCL): {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30}
 *       upper_bits[k] = (qh_word >> shift[k]) & 0x03  (for k=0..15)
 *   Full weight: lower_half[k]  = lower_lo[k] | (upper_bits_lo[k] << 4)  -- range 0..63
 *                upper_half[k]  = upper_lo[k] | (upper_bits_hi[k] << 4)
 *   Dequantized: (weight - 32) * d * scale[2*j]    for lower half
 *                (weight - 32) * d * scale[2*j+1]  for upper half
 *
 * Scales: ps[b*16 .. b*16+15] are int8 (signed) values, one per half-sub-block.
 * Super-scale: pd[b] is fp16.
 *
 * Kernels provided:
 *   gemv_q6k          -- plain GEMV
 *   sparse_gemv_q6k   -- sparse GEMV (only compute indexed output rows)
 *   down_merge_q6k    -- MoE down GEMV with expert accumulation
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* -----------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------- */
inline float load_fp16_q6k(const __global uchar *p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_half(u);
}

/* The shift table used in the SYCL kernel for qh upper-2-bit extraction.
 * shift[k] gives the bit offset within a 32-bit word for position k (0..15). */
inline uint qh_shift(int k) {
    /* Pattern: {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30} */
    const uint tbl[16] = {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30};
    return tbl[k];
}

/* Sub-block dot for Q6_K.
 *   pql_blk : base of 128-byte ql section for this block
 *   pqh_blk : base of 64-byte qh section for this block
 *   ps_blk  : base of 16-byte scale section (int8[16])
 *   d       : fp16 super-scale (as float)
 *   j       : sub-block index (0..7)
 */
inline float q6k_subblock_dot(const __global uchar *pql_blk,
                               const __global uchar *pqh_blk,
                               const __global char  *ps_blk,
                               float d,
                               const float *input,
                               int j)
{
    float combined_lo = d * (float)ps_blk[2 * j];
    float combined_hi = d * (float)ps_blk[2 * j + 1];

    /* Load the two 32-bit qh words for this sub-block */
    uint qh_lo = (uint)pqh_blk[j*8+0] | ((uint)pqh_blk[j*8+1]<<8)
               | ((uint)pqh_blk[j*8+2]<<16) | ((uint)pqh_blk[j*8+3]<<24);
    uint qh_hi = (uint)pqh_blk[j*8+4] | ((uint)pqh_blk[j*8+5]<<8)
               | ((uint)pqh_blk[j*8+6]<<16) | ((uint)pqh_blk[j*8+7]<<24);

    float acc = 0.0f;
    for (int k = 0; k < 16; k++) {
        uchar ql_byte = pql_blk[j * 16 + k];
        uint  sh = qh_shift(k);

        /* Lower half: ql low nibble + 2 upper bits from qh_lo */
        float w0 = (float)((ql_byte & 0x0F) | (((qh_lo >> sh) & 0x03u) << 4));
        w0 = (w0 - 32.0f) * combined_lo;

        /* Upper half: ql high nibble + 2 upper bits from qh_hi */
        float w1 = (float)(((ql_byte >> 4) & 0x0F) | (((qh_hi >> sh) & 0x03u) << 4));
        w1 = (w1 - 32.0f) * combined_hi;

        acc += input[j * 32 + k]      * w0;
        acc += input[j * 32 + 16 + k] * w1;
    }
    return acc;
}

/* =======================================================================
 * Kernel 1: gemv_q6k
 *
 * Plain GEMV: y[output_len] = W * x[input_len]
 *
 * Launch:
 *   global_size = (output_len / OPG_Q6K) * num_blocks_per_row
 *   local_size  = num_blocks_per_row / BS_Q6K
 *   (BS_Q6K chunks per thread, OPG_Q6K=16, BS_Q6K=2 per benchmark)
 * ===================================================================== */
#define OPG_Q6K 16
#define BS_Q6K  2

__kernel
void gemv_q6k(
    __global const float  *inputs,      /* [input_len] */
    __global const uchar  *pql,         /* [num_blocks * 128] */
    __global const uchar  *pqh,         /* [num_blocks * 64]  */
    __global const char   *ps,          /* [num_blocks * 16]  int8 */
    __global const uchar  *pd_buf,      /* [num_blocks * 2]   fp16 */
    __global       float  *outputs,     /* [output_len] */
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
                                        ps + blk_idx * 16, d, inp, j);
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

/* =======================================================================
 * Kernel 2: sparse_gemv_q6k
 *
 * Sparse GEMV: compute only the output rows listed in `indexes[0..index_len)`.
 * Non-indexed positions must be filled with -inf by the caller before launch.
 *
 * Launch:
 *   global = (index_len / OPG_Q6K) * local_range
 *   local  = local_range = num_blocks_per_row / BS_Q6K
 * ===================================================================== */
__kernel
void sparse_gemv_q6k(
    __global const float  *inputs,
    __global const uchar  *pql,
    __global const uchar  *pqh,
    __global const char   *ps,
    __global const uchar  *pd_buf,
    __global       float  *outputs,
    __global const uint   *indexes,     /* [index_len] */
    uint index_len,
    uint input_len,
    uint output_len
) {
    uint num_blocks_per_row = input_len / 256;
    uint local_range = num_blocks_per_row / BS_Q6K;
    uint h  = get_group_id(0);
    uint hh = get_local_id(0);

    __local float slm[64 * OPG_Q6K];

    /* Load the OPG_Q6K row indices for this group */
    uint row_idx[OPG_Q6K];
    for (int l = 0; l < OPG_Q6K; l++)
        row_idx[l] = indexes[h * OPG_Q6K + l];

    float outData[OPG_Q6K]; for (int l = 0; l < OPG_Q6K; l++) outData[l] = 0.0f;

    for (uint c = 0; c < BS_Q6K; c++) {
        uint blockIdx = hh * BS_Q6K + c;
        float inp[256];
        for (int i = 0; i < 256; i++)
            inp[i] = inputs[256 * blockIdx + i];

        for (int l = 0; l < OPG_Q6K; l++) {
            uint blk_idx = row_idx[l] * num_blocks_per_row + blockIdx;
            float d = load_fp16_q6k(pd_buf + blk_idx * 2);
            float acc = 0.0f;
            for (int j = 0; j < 8; j++)
                acc += q6k_subblock_dot(pql + blk_idx * 128, pqh + blk_idx * 64,
                                        ps + blk_idx * 16, d, inp, j);
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
        /* Scatter to indexed output positions */
        for (int l = 0; l < OPG_Q6K; l++)
            outputs[row_idx[l]] = accOut[l];
    }
}

/* =======================================================================
 * Kernel 3: down_merge_q6k
 *
 * MoE down projection.
 * Launch:
 *   global = (output_len/OPG_DM6 * num_blocks_per_row, token_len)
 *   local  = (num_blocks_per_row, 1)
 * ===================================================================== */
#define OPG_DM6 4

__kernel
void down_merge_q6k(
    __global const float  *inputs,
    __global const uchar  *downs,       /* [num_experts * expert_size], expert_size=num_blocks*210 */
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
    uint expert_size        = num_blocks * 210;  /* Q6K: 128+64+16+2 */

    uint h  = get_group_id(0);
    uint hh = get_local_id(0);
    uint t  = get_group_id(1);

    __local float slm[64 * OPG_DM6];

    float outData[OPG_DM6]; for (int l = 0; l < OPG_DM6; l++) outData[l] = 0.0f;

    for (int e = 0; e < (int)input_num; e++) {
        uint  index     = indexes ? indexes[t * aligned_input_num + e] : (uint)e;
        float efficient = efficients[t * aligned_input_num + e];

        const __global uchar *weights = downs + index * expert_size;
        const __global uchar *pql_w   = weights;
        const __global uchar *pqh_w   = pql_w + num_blocks * 128;
        const __global char  *ps_w    = (const __global char *)(pqh_w + num_blocks * 64);
        const __global uchar *pd_w    = (const __global uchar *)(ps_w  + num_blocks * 16);

        float inp[256];
        for (int i = 0; i < 256; i++)
            inp[i] = inputs[t * input_num * input_len + e * input_len + 256 * hh + i];

        float expertOut[OPG_DM6];
        for (int l = 0; l < OPG_DM6; l++) {
            uint blk_idx = h * OPG_DM6 * num_blocks_per_row + (uint)l * num_blocks_per_row + hh;
            float d = load_fp16_q6k(pd_w + blk_idx * 2);
            float acc = 0.0f;
            for (int j = 0; j < 8; j++)
                acc += q6k_subblock_dot(pql_w + blk_idx * 128, pqh_w + blk_idx * 64,
                                        ps_w + blk_idx * 16, d, inp, j);
            expertOut[l] = acc;
        }

        for (int l = 0; l < OPG_DM6; l++)
            outData[l] += expertOut[l] * efficient;
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
                accOut[l] += outputs[t * output_len + h * OPG_DM6 + l];
        }
        for (int l = 0; l < OPG_DM6; l++)
            outputs[t * output_len + h * OPG_DM6 + l] = accOut[l];
    }
}
