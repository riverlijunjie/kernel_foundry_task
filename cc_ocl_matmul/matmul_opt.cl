/*
 * cc_ocl_matmul — final kernel (v7)
 *   Per-SG tile  : 32 (M) x 32 (N), 8 float8 accumulators
 *   WG shape     : WG_M x WG_N = 4 x 8 sub-groups (32 SGs, 512 WIs)
 *   WG-tile      : 128 (M) x 256 (N)
 *   K-step       : 32 (two k=16 DPAS rounds; B uses VNNI-transform read)
 *   Dispatch     : SWIZZLE=4 M-major block walk for L2 A-panel reuse
 *   GWS = (ceil(M/128)*WG_SIZE, ceil(N/256))   LWS = (WG_SIZE, 1)
 *
 * Measured on Intel Arc B580 @ M=2048, K=2560, N=2048, fp16:
 *   reference (naive scalar) : 33.84 ms
 *   this kernel (min of 100) :  0.261 ms     ~70.9% of 116 TFLOPS roofline
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define TILE_M   32
#define TILE_N   32
#define WG_M     4
#define WG_N     8
#define SG_SIZE  16

#define WG_TILE_M (WG_M * TILE_M)         /* 64  */
#define WG_TILE_N (WG_N * TILE_N)         /* 512 */
#define NUM_SG    (WG_M * WG_N)           /* 32  */
#define WG_SIZE   (NUM_SG * SG_SIZE)      /* 512 */


__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global       half* restrict C,
    int M, int K, int N)
{
    const int sg_id  = get_sub_group_id();
    const int sg_row = sg_id / WG_N;
    const int sg_col = sg_id % WG_N;

    /* Swizzle dispatch: walk WGs in M-major blocks of SWIZZLE so consecutive
     * WGs share the A-panel (which only changes when block_id changes). */
    #define SWIZZLE 4
    const int grid_m = get_num_groups(0);
    const int grid_n = get_num_groups(1);
    const int gid_m = get_group_id(0);
    const int gid_n = get_group_id(1);
    const int linear = gid_n * grid_m + gid_m;
    const int block_size = SWIZZLE * grid_n;
    const int block_id = linear / block_size;
    const int block_off = linear % block_size;
    const int rem_m = grid_m - block_id * SWIZZLE;
    const int cur_swizzle = (rem_m < SWIZZLE) ? rem_m : SWIZZLE;
    const int gm = block_id * SWIZZLE + (block_off % cur_swizzle);
    const int gn = block_off / cur_swizzle;
    const int wg_tile_m_off = gm * WG_TILE_M;
    const int wg_tile_n_off = gn * WG_TILE_N;
    const int tile_m = wg_tile_m_off + sg_row * TILE_M;
    const int tile_n = wg_tile_n_off + sg_col * TILE_N;

    float8 acc00=0, acc01=0;
    float8 acc10=0, acc11=0;
    float8 acc20=0, acc21=0;
    float8 acc30=0, acc31=0;

    const int A_pitch = K * (int)sizeof(half);
    const int B_pitch = N * (int)sizeof(half);

    for (int k0 = 0; k0 < K; k0 += 32) {
        ushort a_data[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_pitch, M, A_pitch,
            (int2)(k0, tile_m), a_data);

        uint b0_data[16], b1_data[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_pitch, K, B_pitch,
            (int2)(tile_n, k0),      b0_data);
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_pitch, K, B_pitch,
            (int2)(tile_n, k0 + 16), b1_data);


        #define DPAS_ROUND(b_buf, a_off) do {                              \
            int8 b0 = as_int8(*(__private uint8*)&(b_buf)[0]);             \
            int8 b1 = as_int8(*(__private uint8*)&(b_buf)[8]);             \
            short8 a0 = as_short8(*(__private ushort8*)&a_data[(a_off)+0]); \
            short8 a1 = as_short8(*(__private ushort8*)&a_data[(a_off)+8]); \
            short8 a2 = as_short8(*(__private ushort8*)&a_data[(a_off)+16]);\
            short8 a3 = as_short8(*(__private ushort8*)&a_data[(a_off)+24]);\
            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00); \
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01); \
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10); \
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11); \
            acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc20); \
            acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc21); \
            acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc30); \
            acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc31); \
        } while (0)

        DPAS_ROUND(b0_data,  0);  /* k=k0+0..15  */
        DPAS_ROUND(b1_data, 32);  /* k=k0+16..31 */
        #undef DPAS_ROUND
    }

    #define STORE_BLOCK(acc, row_off, col_off) do { \
        ushort cd[8]; \
        cd[0]=as_ushort((half)acc.s0); cd[1]=as_ushort((half)acc.s1); \
        cd[2]=as_ushort((half)acc.s2); cd[3]=as_ushort((half)acc.s3); \
        cd[4]=as_ushort((half)acc.s4); cd[5]=as_ushort((half)acc.s5); \
        cd[6]=as_ushort((half)acc.s6); cd[7]=as_ushort((half)acc.s7); \
        intel_sub_group_2d_block_write_16b_8r16x1c( \
            (__global void*)C, B_pitch, M, B_pitch, \
            (int2)(tile_n + (col_off), tile_m + (row_off)), cd); \
    } while (0)

    STORE_BLOCK(acc00,  0,  0); STORE_BLOCK(acc01,  0, 16);
    STORE_BLOCK(acc10,  8,  0); STORE_BLOCK(acc11,  8, 16);
    STORE_BLOCK(acc20, 16,  0); STORE_BLOCK(acc21, 16, 16);
    STORE_BLOCK(acc30, 24,  0); STORE_BLOCK(acc31, 24, 16);
    #undef STORE_BLOCK
}
