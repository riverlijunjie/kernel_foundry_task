/*
 * kernel_q4k_q6k.cl
 *
 * Optimization target: plain GEMV for Q4_K and Q6_K shuffled weight layouts.
 *   gemv_q4k  -- Q4_K ShuffleQ4KL1 layout
 *   gemv_q6k  -- Q6_K ShuffleQ6KL1 layout
 *
 * ---- Q4_K shuffled layout (per block, 256 weights) ----
 *   pqs [num_blocks * 128]  packed 4-bit quants
 *   psl [num_blocks * 16]   bytes [0..3]  sl  (scale low nibbles, 8 values in 4 bytes)
 *                            bytes [4..7]  ml  (min   low nibbles, 8 values in 4 bytes)
 *                            bytes [8..9]  sh  (scale high 2-bits, 8 values in 2 bytes)
 *                            bytes [10..11] mh  (min   high 2-bits, 8 values in 2 bytes)
 *                            bytes [12..13] d   (fp16 super-scale)
 *                            bytes [14..15] dmin (fp16 super-min)
 *
 * ---- Q6_K shuffled layout (per block, 256 weights) ----
 *   pql [num_blocks * 128]  packed lower nibbles (4 bits/weight)
 *   pqh [num_blocks *  64]  packed upper 2 bits  (4 weights/byte), column-major
 *   ps  [num_blocks *  16]  int8 scales (16 per block)
 *   pd  [num_blocks *   2]  fp16 super-scale d
 *
 * REQUIRED (host dispatch depends on these):
 *   1. Entry-point names and signatures must remain EXACTLY:
 *        gemv_q4k(inputs, pqs, psl, outputs, input_len, output_len)
 *        gemv_q6k(inputs, pql, pqh, ps, pd_buf, outputs, input_len, output_len)
 *   2. OPG_Q4K, OPG_Q6K, BS_Q6K defines must stay present (task.py parses them
 *      to compute dispatch dimensions). Their values may change but must satisfy:
 *        output_len % OPG_Q4K == 0
 *        output_len % OPG_Q6K == 0
 *        (input_len/256) % BS_Q6K == 0
 */

// [USER_INSTRUCTIONS_START]
//
// Optimize gemv tips:
// 1. GEMV is memory bound, focus on reducing memory access latency.
// 2. Use local memory to reduce global memory access latency if possible.
// 3. Optimize work-group sizes based on the GPU architecture.
// 4. Minimize divergent branches within the kernel.
// 5. Unroll loops where applicable to increase instruction-level parallelism.
// 6. Use vectorized data types and block read to improve memory bandwidth utilization, such as intel_sub_group_block_read_uc8/intel_sub_group_block_read_uc16, etc
// 7. One subgroup can process one block of 16 or 32 elements, which can be used to optimize memory access patterns.
// 8. Consider using fused multiply-add (FMA) operations if supported by the hardware.
// 9. Implement double buffering to overlap computation with memory transfers.
// 10. For int4 and int8 data types, consider using efficient bitwise unpacking techniques to optimize data loading.
//
// [USER_INSTRUCTIONS_END]

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

/* [EVOLVE_START] */
#define OPG 16  /* = sub_group_size */

inline float _u16_to_float(ushort raw) { return (float)as_half(raw); }

/* -----------------------------------------------------------------------
 * gemv_q4k_sg
 * ----------------------------------------------------------------------- */
__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void gemv_q4k(
    __global const half   *inputs,   /* [input_len]              */
    __global const uchar  *pqs_T,    /* [total_blk * OPG * 128]  same total size as pqs */
    __global const uchar  *psl_T,    /* [total_blk * OPG * 16]   same total size as psl */
    __global       half   *outputs,  /* [output_len]             */
    uint input_len,
    uint output_len
) {
    const uint nbpr = input_len / 256;

    const uint h  = get_group_id(0);    /* row-group  */
    const uint hh = get_local_id(0);    /* lane = output row within group */

    /* Mixed precision: fp16 IO + fp32 accumulation for numerical stability */
    __local float slm_inp[256];
    __local float slm_acc[OPG];
    slm_acc[hh] = 0.0f;

    for (uint bid = 0; bid < nbpr; bid++) {

        /* ---- Cooperatively load 256 input half values into SLM ---- */
        /* All OPG lanes load disjoint slices: lane hh loads elements [k*OPG+hh] for k=0..15 */
        {
            const __global ushort *inp_us = (const __global ushort *)(inputs + bid * 256);
            for (uint k = 0; k < 256 / OPG; k++) {
                /* intel_sub_group_block_read_us: each lane reads inp_us[k*OPG + hh] */
                ushort v = intel_sub_group_block_read_us(inp_us + k * OPG);
                slm_inp[k * OPG + hh] = (float)as_half(v);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* ---- Decode scales for block (h, bid) ---- */
        /* psl_T entry: OPG * 16 = 256 bytes at (h*nbpr+bid)*OPG*16            */
        /* Fields SoA: sl(64B) ml(64B) sh(32B) mh(32B) d(32B) dmin(32B)        */
        {
            uint psl_base = (h * nbpr + bid) * OPG * 16;
            const __global uint   *psl_u  = (const __global uint   *)(psl_T + psl_base);
            const __global ushort *psl_us = (const __global ushort *)(psl_T + psl_base);

            uint   sl     = intel_sub_group_block_read   (psl_u  +  0);   /* bytes [  0.. 63] */
            uint   ml     = intel_sub_group_block_read   (psl_u  + 16);   /* bytes [ 64..127] */
            ushort sh     = intel_sub_group_block_read_us(psl_us + 64);   /* bytes [128..159] */
            ushort mh     = intel_sub_group_block_read_us(psl_us + 80);   /* bytes [160..191] */
            ushort d_raw  = intel_sub_group_block_read_us(psl_us + 96);   /* bytes [192..223] */
            ushort dn_raw = intel_sub_group_block_read_us(psl_us + 112);  /* bytes [224..255] */

            float d    = _u16_to_float(d_raw);
            float dmin = _u16_to_float(dn_raw);

            /* Decode 8 scale/min values (same bit logic as SYCL kernel) */
            float scale[8], minv[8];
            for (int i = 0; i < 8; i++) {
                uint sq  = ((sl >> (i * 4)) & 0xFu)
                         | ((uint)((sh >> (i * 2)) & 0x3u) << 4);
                uint mq  = ((ml >> (i * 4)) & 0xFu)
                         | ((uint)((mh >> (i * 2)) & 0x3u) << 4);
                scale[i] = (float)sq * d;
                minv[i]  = (float)mq * dmin;
            }

            /* ---- Weight loads + FMA over 8 sub-blocks × 4 chunks ---- */
            /* pqs_T entry: OPG * 128 = 2048 bytes at (h*nbpr+bid)*OPG*128      */
            /* Sub-block j at offset j*OPG*16 within entry                      */
            /* Chunk c    at offset j*OPG*16 + c*OPG*4 within entry             */
            uint pqs_entry = (h * nbpr + bid) * OPG * 128;
            float acc = 0.0f;

            for (int j = 0; j < 8; j++) {
                float s = scale[j];
                float m = minv[j];
                uint  j_off = pqs_entry + (uint)j * OPG * 16;

                for (int chunk = 0; chunk < 4; chunk++) {
                    uint  off = j_off + (uint)chunk * OPG * 4;
                    /* uint block read: lane hh gets 4 contiguous bytes at off + hh*4 */
                    uint qcw = intel_sub_group_block_read((const __global uint *)(pqs_T + off));

                    uchar qb0 = (uchar)( qcw        & 0xFFu);
                    uchar qb1 = (uchar)((qcw >>  8) & 0xFFu);
                    uchar qb2 = (uchar)((qcw >> 16) & 0xFFu);
                    uchar qb3 = (uchar)((qcw >> 24) & 0xFFu);

                    uint base = (uint)j * 32u + (uint)chunk * 4u;

                    acc = fma(slm_inp[base +  0], fma((float)((qb0     ) & 0xFu), s, -m), acc);
                    acc = fma(slm_inp[base + 16], fma((float)((qb0 >> 4) & 0xFu), s, -m), acc);
                    acc = fma(slm_inp[base +  1], fma((float)((qb1     ) & 0xFu), s, -m), acc);
                    acc = fma(slm_inp[base + 17], fma((float)((qb1 >> 4) & 0xFu), s, -m), acc);
                    acc = fma(slm_inp[base +  2], fma((float)((qb2     ) & 0xFu), s, -m), acc);
                    acc = fma(slm_inp[base + 18], fma((float)((qb2 >> 4) & 0xFu), s, -m), acc);
                    acc = fma(slm_inp[base +  3], fma((float)((qb3     ) & 0xFu), s, -m), acc);
                    acc = fma(slm_inp[base + 19], fma((float)((qb3 >> 4) & 0xFu), s, -m), acc);
                }
            }

            slm_acc[hh] += acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint row = h * OPG + hh;
    if (row < output_len) {
        outputs[row] = convert_half_rte(slm_acc[hh]);
    }
}

/* ===================================================================
 * Q6_K
 * =================================================================== */
inline float load_fp16_q6k(const __global uchar *p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_half(u);
}

/* Vectorized sub-block dot for Q6_K.
 * High 2-bit layout: k = 4*p + q  ->  bits (2*p) of packed byte q. */
inline float q6k_subblock_dot(const __global uchar *pql_blk,
                              const __global uchar *pqh_blk,
                              const __global char  *ps_blk,
                              float d,
                              const float *input,
                              int j)
{
    float combined_lo = d * (float)ps_blk[2 * j];
    float combined_hi = d * (float)ps_blk[2 * j + 1];

    uchar b0 = pqh_blk[j*8+0], b1 = pqh_blk[j*8+1];
    uchar b2 = pqh_blk[j*8+2], b3 = pqh_blk[j*8+3];
    uchar b4 = pqh_blk[j*8+4], b5 = pqh_blk[j*8+5];
    uchar b6 = pqh_blk[j*8+6], b7 = pqh_blk[j*8+7];

    uchar16 hl = (uchar16)(
        b0 & 3, b1 & 3, b2 & 3, b3 & 3,
        (b0>>2)&3, (b1>>2)&3, (b2>>2)&3, (b3>>2)&3,
        (b0>>4)&3, (b1>>4)&3, (b2>>4)&3, (b3>>4)&3,
        (b0>>6)&3, (b1>>6)&3, (b2>>6)&3, (b3>>6)&3);
    uchar16 hu = (uchar16)(
        b4 & 3, b5 & 3, b6 & 3, b7 & 3,
        (b4>>2)&3, (b5>>2)&3, (b6>>2)&3, (b7>>2)&3,
        (b4>>4)&3, (b5>>4)&3, (b6>>4)&3, (b7>>4)&3,
        (b4>>6)&3, (b5>>6)&3, (b6>>6)&3, (b7>>6)&3);

    uchar16 ql = vload16(0, pql_blk + j * 16);
    float16 w0 = convert_float16((ql & (uchar16)0x0F) | (hl << (uchar16)4)) - 32.0f;
    float16 w1 = convert_float16((ql >> (uchar16)4)   | (hu << (uchar16)4)) - 32.0f;

    float16 in0 = vload16(0, input + j * 32);
    float16 in1 = vload16(0, input + j * 32 + 16);

    return hsum16(in0 * (w0 * combined_lo) + in1 * (w1 * combined_hi));
}

#define OPG_Q6K 16
#define BS_Q6K  2

__kernel
void gemv_q6k(
    __global const float  *inputs,
    __global const uchar  *pql,
    __global const uchar  *pqh,
    __global const char   *ps,
    __global const uchar  *pd_buf,
    __global       float  *outputs,
    uint input_len,
    uint output_len
) {
    uint num_blocks_per_row = input_len / 256;
    uint local_range = num_blocks_per_row / BS_Q6K;
    uint h  = get_group_id(0);
    uint hh = get_local_id(0);
    uint ls = get_local_size(0);

    __local float slm[64 * OPG_Q6K];

    float outData[OPG_Q6K];
    #pragma unroll
    for (int l = 0; l < OPG_Q6K; l++) outData[l] = 0.0f;

    for (uint c = 0; c < BS_Q6K; c++) {
        uint blockIdx = hh * BS_Q6K + c;

        float inp[256];
        const __global float *in_base = inputs + 256 * blockIdx;
        #pragma unroll
        for (int i = 0; i < 16; i++)
            vstore16(vload16(0, in_base + i * 16), 0, inp + i * 16);

        for (int l = 0; l < OPG_Q6K; l++) {
            uint blk_idx = h * OPG_Q6K * num_blocks_per_row
                           + (uint)l * num_blocks_per_row + blockIdx;
            float d = load_fp16_q6k(pd_buf + blk_idx * 2);
            const __global uchar *pql_blk = pql + blk_idx * 128;
            const __global uchar *pqh_blk = pqh + blk_idx * 64;
            const __global char  *ps_blk  = ps  + blk_idx * 16;
            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                acc += q6k_subblock_dot(pql_blk, pqh_blk, ps_blk, d, inp, j);
            outData[l] += acc;
        }
    }

    #pragma unroll
    for (int l = 0; l < OPG_Q6K; l++)
        slm[hh * OPG_Q6K + l] = outData[l];

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Parallel reduction across local_range columns of partials. */
    for (uint l = hh; l < OPG_Q6K; l += ls) {
        float acc = 0.0f;
        for (uint i = 0; i < local_range; i++)
            acc += slm[i * OPG_Q6K + l];
        outputs[h * OPG_Q6K + l] = acc;
    }
}
/* [EVOLVE_END] */
