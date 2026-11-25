// kernel_q80_sg.cl — Q8_0 GEMV with Intel sub-group block reads (SG layout).
//
// Computes  C[bm, n] = sum_k A[bm, k] * W[n, k]  for a Q8_0 weight matrix
// that has been reordered into the SG-transposed ("shuffle") layout.
//
// == Q8_0 format recap ==
//   GGUF block_q8_0 = { half d; int8_t qs[32]; }  (34 bytes, QK=32 weights)
//     dequant:  w = q * d          (q is signed int8, no zero-point / min)
//
//   EIGHT consecutive Q8_0 blocks are grouped into one super-block of 256
//   weights.  Each super-block carries:
//     * 8 sub-blocks x 32 qs bytes            = 256 bytes  (weights, "pqs")
//     * 8 fp16 scales d                       =  16 bytes  (scales,  "pd")
//   which equals the original 8 x 34 = 272 bytes — a pure reorder.
//
// == Combined weight buffer layout (W = [pqs_T | pd_T]) ==
//   pqs_T:  total_blk * 256 bytes at offset 0.
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 256 bytes (4096 B for OPG=16).
//     Sub-block j at j*OPG*32, chunk c (0..7) at j*OPG*32 + c*OPG*4.
//     One intel_sub_group_block_read8 per sub-block delivers 8 int8x4 chunks.
//   pd_T:   total_blk * 16 bytes at offset total_blk*256.
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes (256 B for OPG=16).
//     SoA fp16 field j at j*OPG*2 + lid*2.
//     One intel_sub_group_block_read_us8 fetches all 8 scales per lane.
//
// == Build options (set by task.py for each shape) ==
//   -DKSPLIT=<n>   K-split factor (sub-groups per work-group); default 1
//   -DK_SIZE=<K>   number of input features (must be multiple of 256)
//   -DN_SIZE=<N>   number of output features (must be multiple of OPG=16)
//
// == Dispatch ==
//   global_size = (N_SIZE, BM, KSPLIT)
//   local_size  = (OPG,    1,  KSPLIT)
//
// == Constraints for optimization (task.py depends on them) ==
//   1. Entry-point name must remain EXACTLY: fc_gguf_q8_0_sg
//   2. Argument order must remain EXACTLY: (A, W, C)
//   3. OPG (sub_group_size) must remain 16.
//   4. The kernel must produce correct results for any KSPLIT value that
//      task.py selects (1, 2, 4, 8, ...).
//   5. Dispatch is fixed by task.py:
//        global=(N_SIZE, BM, KSPLIT)  local=(OPG, 1, KSPLIT)

#pragma OPENCL EXTENSION cl_intel_subgroups       : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef OPG
#define OPG 16  // = sub_group_size = output rows per sub-group
#endif

#ifndef KSPLIT
#define KSPLIT 1
#endif

// Accumulate one 4-byte chunk (4 signed int8 weights) into the sub-block dot
// product.  `in` points at the SLM base of the sub-block; `c` is the chunk
// index (0..7); weights c*4..c*4+3 map to activations in[c*4..c*4+3].
#define ACC_CHUNK_Q8(dot, u, c, in)                                              \
    do {                                                                         \
        char4 _q = as_char4((uint)(u));                                          \
        (dot) = fma((in)[(c)*4 + 0], (float)_q.s0, (dot));                      \
        (dot) = fma((in)[(c)*4 + 1], (float)_q.s1, (dot));                      \
        (dot) = fma((in)[(c)*4 + 2], (float)_q.s2, (dot));                      \
        (dot) = fma((in)[(c)*4 + 3], (float)_q.s3, (dot));                      \
    } while (0)

// Process one full sub-block j (32 signed weights = 8 chunks of 4 bytes).
#define Q8_SUBBLOCK(in, wv, sc, acc)                                             \
    do {                                                                         \
        float _dot = 0.0f;                                                       \
        ACC_CHUNK_Q8(_dot, (wv).s0, 0, in); ACC_CHUNK_Q8(_dot, (wv).s1, 1, in); \
        ACC_CHUNK_Q8(_dot, (wv).s2, 2, in); ACC_CHUNK_Q8(_dot, (wv).s3, 3, in); \
        ACC_CHUNK_Q8(_dot, (wv).s4, 4, in); ACC_CHUNK_Q8(_dot, (wv).s5, 5, in); \
        ACC_CHUNK_Q8(_dot, (wv).s6, 6, in); ACC_CHUNK_Q8(_dot, (wv).s7, 7, in); \
        (acc) = fma((sc), _dot, (acc));                                          \
    } while (0)

/* [EVOLVE_START] */

__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void fc_gguf_q8_0_sg(
    const __global half*  A,   // activation [BM, K_SIZE]  (fp16)
    const __global uchar* W,   // SG-packed weight buffer  [pqs_T | pd_T]
          __global float* C    // output      [BM, N_SIZE] (fp32)
)
{
    const uint nbpr = (uint)K_SIZE / 256u;             // K-blocks per row
    const uint h    = (uint)get_group_id(0);           // row-group index
    const uint hh   = (uint)get_sub_group_local_id();  // lane (output row within group)
    const uint bm   = (uint)get_global_id(1);          // activation row (BM dimension)
    const uint sg   = (uint)get_local_id(2);           // K-split sub-group index

    // Per-sub-group activation SLM (double-buffered) + KSPLIT reduction scratch.
    __local float slm_inp[KSPLIT * 2 * 256];
#if KSPLIT > 1
    __local float slm_red[KSPLIT * OPG];
#endif
    __local float* sbase = slm_inp + sg * (2 * 256);

    const uint total_blk = (uint)N_SIZE * nbpr;
    const uint off_pqs = 0u;
    const uint off_pd  = total_blk * 256u;

    const __global uint*   pqs_u = (const __global uint*)(W + off_pqs);
    const __global ushort* pd_us = (const __global ushort*)(W + off_pd);

    const __global half* A_row = A + (uint)bm * (uint)K_SIZE;

    float acc = 0.0f;

    // Each sub-group processes a strided subset of the K-blocks.
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float* sb = sbase + (bid & 1u) * 256;   // double-buffer slot

        // Issue all long-latency global loads first (weights + scales).
        uint pe_u = (h * nbpr + bid) * (OPG * 256 / 4);   // uint offset
        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u + 0*OPG*8);  // sub-block 0
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 1*OPG*8);  // sub-block 1
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 2*OPG*8);  // sub-block 2
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 3*OPG*8);  // sub-block 3
        uint8 w4 = intel_sub_group_block_read8(pqs_u + pe_u + 4*OPG*8);  // sub-block 4
        uint8 w5 = intel_sub_group_block_read8(pqs_u + pe_u + 5*OPG*8);  // sub-block 5
        uint8 w6 = intel_sub_group_block_read8(pqs_u + pe_u + 6*OPG*8);  // sub-block 6
        uint8 w7 = intel_sub_group_block_read8(pqs_u + pe_u + 7*OPG*8);  // sub-block 7

        // 8 fp16 scales per lane (SoA), fetched with ONE wide block read.
        uint    pd_off = (h * nbpr + bid) * (OPG * 8);    // ushort offset
        ushort8 dv = intel_sub_group_block_read_us8(pd_us + pd_off);

        // Load 256 activations into SLM (each lane writes 16 floats at stride OPG).
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < 16u; ++i) {
            sb[i * OPG + hh] = (float)A_row[bid * 256u + i * OPG + hh];
        }

        // Decode 8 fp16 scales (overlaps outstanding loads).
        float scale[8];
        {
            ushort dh[8];
            dh[0]=dv.s0; dh[1]=dv.s1; dh[2]=dv.s2; dh[3]=dv.s3;
            dh[4]=dv.s4; dh[5]=dv.s5; dh[6]=dv.s6; dh[7]=dv.s7;
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < 8; ++i)
                scale[i] = (float)as_half(dh[i]);
        }

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        // 8 sub-blocks: dot products + per-sub-block scale.
        __local const float* in;
        in = sb + 0*32; Q8_SUBBLOCK(in, w0, scale[0], acc);
        in = sb + 1*32; Q8_SUBBLOCK(in, w1, scale[1], acc);
        in = sb + 2*32; Q8_SUBBLOCK(in, w2, scale[2], acc);
        in = sb + 3*32; Q8_SUBBLOCK(in, w3, scale[3], acc);
        in = sb + 4*32; Q8_SUBBLOCK(in, w4, scale[4], acc);
        in = sb + 5*32; Q8_SUBBLOCK(in, w5, scale[5], acc);
        in = sb + 6*32; Q8_SUBBLOCK(in, w6, scale[6], acc);
        in = sb + 7*32; Q8_SUBBLOCK(in, w7, scale[7], acc);
    }

    // Reduce KSPLIT partial sums; sub-group 0 writes the output.
#if KSPLIT > 1
    slm_red[sg * OPG + hh] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0)
        return;
    float sum = 0.0f;
    __attribute__((opencl_unroll_hint))
    for (int g = 0; g < KSPLIT; ++g)
        sum += slm_red[g * OPG + hh];
    acc = sum;
#endif

    const uint n = h * OPG + hh;
    if (n < (uint)N_SIZE) {
        C[bm * (uint)N_SIZE + n] = acc;
    }
}

/* [EVOLVE_END] */
