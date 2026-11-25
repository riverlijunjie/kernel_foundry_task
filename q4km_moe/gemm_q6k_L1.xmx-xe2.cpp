#include <stdint.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "scratch_buffer.h"
#include "ggml_types.h"
#include "gemm_manager.h"
#include <cassert>

typedef sycl::half fp16;

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template<typename IT>
ESIMD_INLINE void shuffle_input(IT *inputs, fp16 *outputs, uint32_t token_len, uint32_t input_len, nd_item<2>& ndi)
{
    const uint32_t blockWidth = 4096;
    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    uint32_t localRange = ndi.get_local_range(0);
    int localLinearId = ndi.get_local_linear_id();

    uint32_t alignedTokenLen = (token_len + 255) / 256 * 256;

    uint32_t curBlockWidth = input_len - h * blockWidth;
    curBlockWidth = curBlockWidth > blockWidth ? blockWidth : curBlockWidth;

    simd<IT, 1024> inputData = 0;
    simd<fp16, 1024> outputData = 0;
    
    uint32_t v_blk_idx = v >> 4;
    uint32_t v_inn_idx = v & 0x0F;
    uint32_t inputOffset = v * 16 * input_len + h * blockWidth + localLinearId * 64;
    uint32_t outputOffset = v_blk_idx * 256 * input_len + h * 256 * blockWidth +  localLinearId * 256 * 64 + v_inn_idx * 16 * 16;
    uint32_t readOffset = inputOffset;
    uint32_t writeOffset = outputOffset;

    if (h * blockWidth + localLinearId * 64 < input_len)
    {
#pragma unroll
        for (int j = 0; j < 16; j++)
        {
            if (v * 16 + j < token_len)
            {
                inputData.template select<64, 1>(j * 64) = block_load<IT, 64>(inputs + readOffset);
                readOffset += input_len;
            }
        }

        for (int j = 0; j < 16 && v*16+j < token_len; j++)
        {
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                outputData.select<16, 1>(k * 256 + j * 16) = inputData.template select<16, 1>(j * 64 + k * 16);
            }
        }

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            block_store<fp16, 256>(outputs + writeOffset, outputData.select<256, 1>(j * 256));
            writeOffset += 16 * 256;
        }
    }

    inputData = 0.0;
    outputData = 0.0;
    inputOffset = v * 16 * input_len + h * blockWidth + (localLinearId + localRange)* 64;
    outputOffset = v_blk_idx * 256 * input_len + h * 256 * blockWidth +  (localRange + localLinearId) * 256 * 64 + v_inn_idx * 16 * 16;
    readOffset = inputOffset;
    writeOffset = outputOffset;
    if (h * blockWidth + (localRange + localLinearId) * 64 < input_len)
    {
#pragma unroll
        for (int j = 0; j < 16; j++)
        {
            if (v * 16 + j < token_len)
            {
                inputData.template select<64, 1>(j * 64) = block_load<IT, 64>(inputs + readOffset);
                readOffset += input_len;
            }
        }

        for (int j = 0; j < 16 && v*16+j < token_len; j++)
        {
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                outputData.select<16, 1>(k * 256 + j * 16) = inputData.template select<16, 1>(j * 64 + k * 16);
            }
        }

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            block_store<fp16, 256>(outputs + writeOffset, outputData.select<256, 1>(j * 256));
            writeOffset += 16 * 256;
        }
    }

}

template<typename OT>
ESIMD_INLINE void gemm_q6kweights_xmx_xe2(fp16 *inputs, uint8_t *weights, OT *outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, OT* scratch, nd_item<2>& ndi, uint8_t *debug)
{
    slm_init(32 * 16 * 32);
    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;

    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    int localLinearId = ndi.get_local_linear_id();
    int hh = localLinearId & 0x07;
    int vv = localLinearId >> 3;

    if (v * 256 + vv * 64 >= token_len)
    {
        return;
    }

    uint32_t slm_offset = localLinearId * 32 * 16;

    uint32_t dump_idx = 0;

    uint8_t *pql = weights;
    uint8_t *pqh = pql + num_blocks * 128;
    int8_t *ps = (int8_t *)(pqh + num_blocks * 64);
    fp16 *pd = (fp16 *)(ps + num_blocks * 16);

    uint32_t localOffsetInput = v * 256 * input_len + vv * 64 * 16;
    uint32_t localOffsetWeightY = h *256 + hh * 32;
    uint32_t localOffsetWeightX_high = 0;
    uint32_t localOffsetWeightX_low = 0;
    uint32_t localOffsetScaleX = 0;
    uint8_t inner_idx = 0;
    simd<uint32_t, 32> d_offsets({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    d_offsets += (h * 256 + hh * 32);
    d_offsets *= (num_blocks_per_row * sizeof(fp16));

    simd<fp16, 128> AData0_tik;
    simd<fp16, 128> AData1_tik;
    simd<fp16, 128> AData2_tik;
    simd<fp16, 128> AData3_tik;
    simd<fp16, 128> AData4_tik;
    simd<fp16, 128> AData5_tik;
    simd<fp16, 128> AData6_tik;
    simd<fp16, 128> AData7_tik;

    simd<uint8_t, 512> BData;
    simd<uint8_t, 256> BData0;
    simd<uint8_t, 256> BData1;
    simd<uint8_t, 128> BData0_hi;
    simd<uint8_t, 128> BData1_hi;

    simd<fp16, 256> BData0_tik;
    simd<fp16, 256> BData1_tik;

    simd<fp16, 128> AData0_tok;
    simd<fp16, 128> AData1_tok;
    simd<fp16, 128> AData2_tok;
    simd<fp16, 128> AData3_tok;
    simd<fp16, 128> AData4_tok;
    simd<fp16, 128> AData5_tok;
    simd<fp16, 128> AData6_tok;
    simd<fp16, 128> AData7_tok;

    simd<fp16, 256> BData0_tok;
    simd<fp16, 256> BData1_tok;

    simd<fp16, 32> scales_buf;
    simd<fp16, 32> global_d;

    simd<float, 128> CData00 = 0.0;
    simd<float, 128> CData10 = 0.0;
    simd<float, 128> CData20 = 0.0;
    simd<float, 128> CData30 = 0.0;
    simd<float, 128> CData40 = 0.0;
    simd<float, 128> CData50 = 0.0;
    simd<float, 128> CData60 = 0.0;
    simd<float, 128> CData70 = 0.0;

    simd<float, 128> CData01 = 0.0;
    simd<float, 128> CData11 = 0.0;
    simd<float, 128> CData21 = 0.0;
    simd<float, 128> CData31 = 0.0;
    simd<float, 128> CData41 = 0.0;
    simd<float, 128> CData51 = 0.0;
    simd<float, 128> CData61 = 0.0;
    simd<float, 128> CData71 = 0.0;

    uint32_t loopNum = input_len / 32;

    // per 256
    BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)ps, num_blocks_per_row * 16 - 1, output_len - 1, num_blocks_per_row * 16 - 1, localOffsetScaleX, localOffsetWeightY);
    localOffsetScaleX += 4;
    slm_block_store<uint8_t, 512>(slm_offset, BData);
    simd_mask<32> gather_mask = 1;
    global_d = gather<fp16, 32, 1>(pd, d_offsets, gather_mask);
    d_offsets += sizeof(fp16);

    // per 32
    BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)pql, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX_low, localOffsetWeightY);
    localOffsetWeightX_low += 4;
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
        BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
        BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
        BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
        BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
    }

    BData.bit_cast_view<uint32_t>().select<64, 1>(0) = load_2d<uint32_t, 2, 32, 1, true, false>((uint32_t *)pqh, input_len/4 - 1, output_len -1, input_len/4 - 1, localOffsetWeightX_high, localOffsetWeightY);
    localOffsetWeightX_high += 2;
#pragma unroll
    for (int j = 0; j < 2; j++)
    {
        BData0_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
        BData0_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
        BData1_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
        BData1_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
    }

    // per 16
    AData0_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
    AData1_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
    AData2_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
    AData3_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
    AData4_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
    AData5_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
    AData6_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
    AData7_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);
    
    localOffsetInput += 16 * 256;

    uint8_t blk_idx = inner_idx >> 2;
    uint8_t step_idx = inner_idx & 0x03;
    simd<int8_t, 128> Bytes = slm_block_load<int8_t, 128>(slm_offset + blk_idx * 128);
    scales_buf = Bytes.select<32, 4>(step_idx);
    scales_buf = scales_buf * global_d;
    ++ inner_idx;

    BData0_tik.select<64, 1>(0) = (BData0.select<64, 1>(0) & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0x03) << 4);
    BData0_tik.select<64, 1>(64) = (BData0.select<64, 1>(64) & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0x0c) << 2);
    BData0_tik.select<64, 1>(128) = (BData0.select<64, 1>(128) & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0x30));
    BData0_tik.select<64, 1>(192) = (BData0.select<64, 1>(192) & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0xc0) >> 2);
    BData1_tik.select<64, 1>(0) = (BData1.select<64, 1>(0) & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0x03) << 4);
    BData1_tik.select<64, 1>(64) = (BData1.select<64, 1>(64) & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0x0c) << 2);
    BData1_tik.select<64, 1>(128) = (BData1.select<64, 1>(128) & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0x30));
    BData1_tik.select<64, 1>(192) = (BData1.select<64, 1>(192) & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0xc0) >> 2);

    simd<fp16, 32> s;

    s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
    s.select<16, 2>(1) = s.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData0_tik.select<32, 1>(j * 32) = (BData0_tik.select<32, 1>(j * 32) - 32) * s;
    }

    s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
    s.select<16, 2>(1) = s.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData1_tik.select<32, 1>(j * 32) = (BData1_tik.select<32, 1>(j * 32) - 32) * s;
    }
    // if (h == 0 && v == 0 && hh == 0 && vv == 0)
    // {
    //     simd<fp16, 16> temp;
    //     temp.select<2, 1>(0) = BData0_tik.select<2, 1>(0);
    //     temp.select<2, 1>(2) = BData0_tik.select<2, 1>(32);
    //     temp.select<2, 1>(4) = BData0_tik.select<2, 1>(32 * 2);
    //     temp.select<2, 1>(6) = BData0_tik.select<2, 1>(32 * 3);
    //     temp.select<2, 1>(8) = BData0_tik.select<2, 1>(32 * 4);
    //     temp.select<2, 1>(10) = BData0_tik.select<2, 1>(32 * 5);
    //     temp.select<2, 1>(12) = BData0_tik.select<2, 1>(32 * 6);
    //     temp.select<2, 1>(14) = BData0_tik.select<2, 1>(32 * 7);
    //     block_store<fp16, 16>((fp16 *)debug + dump_idx * 16, temp);
    //     ++ dump_idx;
    // }

    for (int i = 0; i < loopNum - 1; i++)
    {
        AData0_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
        AData1_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
        AData2_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
        AData3_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
        AData4_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
        AData5_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
        AData6_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
        AData7_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

        localOffsetInput += 16 * 256;

        blk_idx = inner_idx >> 2;
        step_idx = inner_idx & 0x03;
        Bytes = slm_block_load<int8_t, 128>(slm_offset + blk_idx * 128);
        scales_buf = Bytes.select<32, 4>(step_idx);
        scales_buf = scales_buf * global_d;
        ++ inner_idx;

        BData0_tok.select<64, 1>(0) = (BData0.select<64, 1>(0) >> 4) + ((BData0_hi.select<64, 1>(64) & 0x03) << 4);
        BData0_tok.select<64, 1>(64) = (BData0.select<64, 1>(64) >> 4) + ((BData0_hi.select<64, 1>(64) & 0x0c) << 2);
        BData0_tok.select<64, 1>(128) = (BData0.select<64, 1>(128) >> 4) + ((BData0_hi.select<64, 1>(64) & 0x30));
        BData0_tok.select<64, 1>(192) = (BData0.select<64, 1>(192) >> 4) + ((BData0_hi.select<64, 1>(64) & 0xc0) >> 2);
        BData1_tok.select<64, 1>(0) = (BData1.select<64, 1>(0) >> 4) + ((BData1_hi.select<64, 1>(64) & 0x03) << 4);
        BData1_tok.select<64, 1>(64) = (BData1.select<64, 1>(64) >> 4) + ((BData1_hi.select<64, 1>(64) & 0x0c) << 2);
        BData1_tok.select<64, 1>(128) = (BData1.select<64, 1>(128) >> 4) + ((BData1_hi.select<64, 1>(64) & 0x30));
        BData1_tok.select<64, 1>(192) = (BData1.select<64, 1>(192) >> 4) + ((BData1_hi.select<64, 1>(64) & 0xc0) >> 2);

        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tok.select<32, 1>(j * 32) = (BData0_tok.select<32, 1>(j * 32) - 32) * s;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tok.select<32, 1>(j * 32) = (BData1_tok.select<32, 1>(j * 32) - 32) * s;
        }

        // if (h == 0 && v == 0 && hh == 0 && vv == 0)
        // {
        //     simd<fp16, 16> temp;
        //     temp.select<2, 1>(0) = BData0_tok.select<2, 1>(0);
        //     temp.select<2, 1>(2) = BData0_tok.select<2, 1>(32);
        //     temp.select<2, 1>(4) = BData0_tok.select<2, 1>(32 * 2);
        //     temp.select<2, 1>(6) = BData0_tok.select<2, 1>(32 * 3);
        //     temp.select<2, 1>(8) = BData0_tok.select<2, 1>(32 * 4);
        //     temp.select<2, 1>(10) = BData0_tok.select<2, 1>(32 * 5);
        //     temp.select<2, 1>(12) = BData0_tok.select<2, 1>(32 * 6);
        //     temp.select<2, 1>(14) = BData0_tok.select<2, 1>(32 * 7);
        //     block_store<fp16, 16>((fp16 *)debug + dump_idx * 16, temp);
        //     ++ dump_idx;
        // }

        CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tik, AData0_tik);
        CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tik, AData1_tik);
        CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tik, AData2_tik);
        CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tik, AData3_tik);
        CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tik, AData4_tik);
        CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tik, AData5_tik);
        CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tik, AData6_tik);
        CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tik, AData7_tik);

        CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tik, AData0_tik);
        CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tik, AData1_tik);
        CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tik, AData2_tik);
        CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tik, AData3_tik);
        CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tik, AData4_tik);
        CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tik, AData5_tik);
        CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tik, AData6_tik);
        CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tik, AData7_tik);


        AData0_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
        AData1_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
        AData2_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
        AData3_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
        AData4_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
        AData5_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
        AData6_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
        AData7_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

        localOffsetInput += 16 * 256;

        if (inner_idx == 16)
        {
            inner_idx = 0;
            BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)ps, num_blocks_per_row * 16 - 1, output_len - 1, num_blocks_per_row * 16 - 1, localOffsetScaleX, localOffsetWeightY);
            localOffsetScaleX += 4;
            slm_block_store<uint8_t, 512>(slm_offset, BData);
            global_d = gather<fp16, 32, 1>(pd, d_offsets, gather_mask);
            d_offsets += sizeof(fp16);
        }

        BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)pql, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX_low, localOffsetWeightY);
        localOffsetWeightX_low += 4;

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
        }

        BData.bit_cast_view<uint32_t>().select<64, 1>(0) = load_2d<uint32_t, 2, 32, 1, true, false>((uint32_t *)pqh, input_len/4 - 1, output_len -1, input_len/4 - 1, localOffsetWeightX_high, localOffsetWeightY);
        localOffsetWeightX_high += 2;
#pragma unroll
        for (int j = 0; j < 2; j++)
        {
            BData0_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
            BData0_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
            BData1_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
            BData1_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
        }

        uint8_t blk_idx = inner_idx >> 2;
        uint8_t step_idx = inner_idx & 0x03;
        Bytes = slm_block_load<int8_t, 128>(slm_offset + blk_idx * 128);
        scales_buf = Bytes.select<32, 4>(step_idx);
        scales_buf = scales_buf * global_d;
        ++ inner_idx;

        BData0_tik.select<64, 1>(0) = (BData0.select<64, 1>(0) & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0x03) << 4);
        BData0_tik.select<64, 1>(64) = (BData0.select<64, 1>(64) & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0x0c) << 2);
        BData0_tik.select<64, 1>(128) = (BData0.select<64, 1>(128) & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0x30));
        BData0_tik.select<64, 1>(192) = (BData0.select<64, 1>(192) & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0xc0) >> 2);
        BData1_tik.select<64, 1>(0) = (BData1.select<64, 1>(0) & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0x03) << 4);
        BData1_tik.select<64, 1>(64) = (BData1.select<64, 1>(64) & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0x0c) << 2);
        BData1_tik.select<64, 1>(128) = (BData1.select<64, 1>(128) & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0x30));
        BData1_tik.select<64, 1>(192) = (BData1.select<64, 1>(192) & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0xc0) >> 2);

        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tik.select<32, 1>(j * 32) = (BData0_tik.select<32, 1>(j * 32) - 32) * s;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tik.select<32, 1>(j * 32) = (BData1_tik.select<32, 1>(j * 32) - 32) * s;
        }

        // if (h == 0 && v == 0 && hh == 0 && vv == 0)
        // {
        //     simd<fp16, 16> temp;
        //     temp.select<2, 1>(0) = BData0_tik.select<2, 1>(0);
        //     temp.select<2, 1>(2) = BData0_tik.select<2, 1>(32);
        //     temp.select<2, 1>(4) = BData0_tik.select<2, 1>(32 * 2);
        //     temp.select<2, 1>(6) = BData0_tik.select<2, 1>(32 * 3);
        //     temp.select<2, 1>(8) = BData0_tik.select<2, 1>(32 * 4);
        //     temp.select<2, 1>(10) = BData0_tik.select<2, 1>(32 * 5);
        //     temp.select<2, 1>(12) = BData0_tik.select<2, 1>(32 * 6);
        //     temp.select<2, 1>(14) = BData0_tik.select<2, 1>(32 * 7);
        //     block_store<fp16, 16>((fp16 *)debug + dump_idx * 16, temp);
        //     ++ dump_idx;
        // }

        CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tok, AData0_tok);
        CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tok, AData1_tok);
        CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tok, AData2_tok);
        CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tok, AData3_tok);
        CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tok, AData4_tok);
        CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tok, AData5_tok);
        CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tok, AData6_tok);
        CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tok, AData7_tok);

        CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tok, AData0_tok);
        CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tok, AData1_tok);
        CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tok, AData2_tok);
        CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tok, AData3_tok);
        CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tok, AData4_tok);
        CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tok, AData5_tok);
        CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tok, AData6_tok);
        CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tok, AData7_tok);
    }

    AData0_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
    AData1_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
    AData2_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
    AData3_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
    AData4_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
    AData5_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
    AData6_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
    AData7_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

    localOffsetInput += 16 * 256;

    blk_idx = inner_idx >> 2;
    step_idx = inner_idx & 0x03;
    Bytes = slm_block_load<int8_t, 128>(slm_offset + blk_idx * 128);
    scales_buf = Bytes.select<32, 4>(step_idx);
    scales_buf = scales_buf * global_d;
    ++ inner_idx;

    BData0_tok.select<64, 1>(0) = (BData0.select<64, 1>(0) >> 4) + ((BData0_hi.select<64, 1>(64) & 0x03) << 4);
    BData0_tok.select<64, 1>(64) = (BData0.select<64, 1>(64) >> 4) + ((BData0_hi.select<64, 1>(64) & 0x0c) << 2);
    BData0_tok.select<64, 1>(128) = (BData0.select<64, 1>(128) >> 4) + ((BData0_hi.select<64, 1>(64) & 0x30));
    BData0_tok.select<64, 1>(192) = (BData0.select<64, 1>(192) >> 4) + ((BData0_hi.select<64, 1>(64) & 0xc0) >> 2);
    BData1_tok.select<64, 1>(0) = (BData1.select<64, 1>(0) >> 4) + ((BData1_hi.select<64, 1>(64) & 0x03) << 4);
    BData1_tok.select<64, 1>(64) = (BData1.select<64, 1>(64) >> 4) + ((BData1_hi.select<64, 1>(64) & 0x0c) << 2);
    BData1_tok.select<64, 1>(128) = (BData1.select<64, 1>(128) >> 4) + ((BData1_hi.select<64, 1>(64) & 0x30));
    BData1_tok.select<64, 1>(192) = (BData1.select<64, 1>(192) >> 4) + ((BData1_hi.select<64, 1>(64) & 0xc0) >> 2);

    s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
    s.select<16, 2>(1) = s.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData0_tok.select<32, 1>(j * 32) = (BData0_tok.select<32, 1>(j * 32) - 32) * s;
    }

    s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
    s.select<16, 2>(1) = s.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData1_tok.select<32, 1>(j * 32) = (BData1_tok.select<32, 1>(j * 32) - 32) * s;
    }

    // if (h == 0 && v == 0 && hh == 0 && vv == 0)
    // {
    //     simd<fp16, 16> temp;
    //     temp.select<2, 1>(0) = BData0_tok.select<2, 1>(0);
    //     temp.select<2, 1>(2) = BData0_tok.select<2, 1>(32);
    //     temp.select<2, 1>(4) = BData0_tok.select<2, 1>(32 * 2);
    //     temp.select<2, 1>(6) = BData0_tok.select<2, 1>(32 * 3);
    //     temp.select<2, 1>(8) = BData0_tok.select<2, 1>(32 * 4);
    //     temp.select<2, 1>(10) = BData0_tok.select<2, 1>(32 * 5);
    //     temp.select<2, 1>(12) = BData0_tok.select<2, 1>(32 * 6);
    //     temp.select<2, 1>(14) = BData0_tok.select<2, 1>(32 * 7);
    //     block_store<fp16, 16>((fp16 *)debug + dump_idx * 16, temp);
    //     ++ dump_idx;
    // }

    CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tik, AData0_tik);
    CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tik, AData1_tik);
    CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tik, AData2_tik);
    CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tik, AData3_tik);
    CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tik, AData4_tik);
    CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tik, AData5_tik);
    CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tik, AData6_tik);
    CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tik, AData7_tik);

    CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tik, AData0_tik);
    CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tik, AData1_tik);
    CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tik, AData2_tik);
    CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tik, AData3_tik);
    CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tik, AData4_tik);
    CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tik, AData5_tik);
    CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tik, AData6_tik);
    CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tik, AData7_tik);

    CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tok, AData0_tok);
    CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tok, AData1_tok);
    CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tok, AData2_tok);
    CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tok, AData3_tok);
    CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tok, AData4_tok);
    CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tok, AData5_tok);
    CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tok, AData6_tok);
    CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tok, AData7_tok);

    CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tok, AData0_tok);
    CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tok, AData1_tok);
    CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tok, AData2_tok);
    CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tok, AData3_tok);
    CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tok, AData4_tok);
    CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tok, AData5_tok);
    CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tok, AData6_tok);
    CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tok, AData7_tok);


    uint32_t localOutputOffsetX = h * 256 + hh * 32;
    uint32_t localOutputOffsetY = v * 256 + vv * 64;

    OT *write_buffer = outputs;
    uint32_t surfaceHeight = token_len;
    
    if (v * 256 + vv * 64 + 64 > token_len)
    {
        write_buffer = scratch;
        localOutputOffsetY = 0;
        surfaceHeight = 64;
    }

    simd<OT, 128> outData0;
    simd<OT, 128> outData1;

    outData0 = CData00;
    outData1 = CData01;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY, outData0);
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX + 16, localOutputOffsetY, outData1);
    localOutputOffsetY += 8;
    outData0 = CData10;
    outData1 = CData11;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY, outData0);
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX + 16, localOutputOffsetY, outData1);
    localOutputOffsetY += 8;
    outData0 = CData20;
    outData1 = CData21;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY, outData0);
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX + 16, localOutputOffsetY, outData1);
    localOutputOffsetY += 8;
    outData0 = CData30;
    outData1 = CData31;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY, outData0);
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX + 16, localOutputOffsetY, outData1);
    localOutputOffsetY += 8;
    outData0 = CData40;
    outData1 = CData41;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY, outData0);
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX + 16, localOutputOffsetY, outData1);
    localOutputOffsetY += 8;
    outData0 = CData50;
    outData1 = CData51;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY, outData0);
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX + 16, localOutputOffsetY, outData1);
    localOutputOffsetY += 8;
    outData0 = CData60;
    outData1 = CData61;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY, outData0);
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX + 16, localOutputOffsetY, outData1);
    localOutputOffsetY += 8;
    outData0 = CData70;
    outData1 = CData71;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY, outData0);
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX + 16, localOutputOffsetY, outData1);

}



template <typename IT, typename OT>
void runGemmQ6KL1_xmx_xe2_impl(void *stream, IT *inputs, uint8_t *weights, OT* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, ScratchBufferMgr *smgr)
{
    sycl::queue *q = (sycl::queue *)stream;
    smgr->Lock();

    const uint32_t blockWidth = 4096;
    uint32_t threadShuffleH = (input_len + blockWidth - 1)/blockWidth;
    uint32_t alignedTokenLen = (token_len + 255)/256 * 256;
    uint32_t threadShuffleV = alignedTokenLen / 16;
    uint32_t localThreadShuffle = 32;

    fp16 *shuffled_inputs = (fp16 *)smgr->Allocate(q, alignedTokenLen * input_len * sizeof(fp16));

    sycl::range<2> GlobalRangeShuffle(localThreadShuffle * threadShuffleH, threadShuffleV);
    sycl::range<2> LocalRangeShuffle(localThreadShuffle, 1);
    sycl::nd_range<2> RangeShuffle(GlobalRangeShuffle, LocalRangeShuffle);

    q->submit([&](handler& cgh) {
        cgh.parallel_for(RangeShuffle, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            shuffle_input(inputs, shuffled_inputs, token_len, input_len, ndi);
        });
    });

    uint32_t threadA = (token_len + 255)/256;
    uint32_t threadB = (output_len + 255)/256;
    uint32_t localThreadNum = 32;

    sycl::range<2> GlobalRange(localThreadNum * threadB, threadA);
    sycl::range<2> LocalRange(localThreadNum, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    OT *outputScratch = (OT *)smgr->Allocate(q, 64 * output_len * sizeof(OT));

    // uint8_t* debug = (uint8_t *)malloc_shared(2560 * 1024, *q);

    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemm_q6kweights_xmx_xe2(shuffled_inputs, weights, outputs, token_len, input_len, output_len, outputScratch, ndi, nullptr);
        });
    });

    // q->wait();
    // FILE *fp = nullptr;
    // fopen_s(&fp, "debug.bin", "wb");
    // fwrite(debug, 1, 2560*1024, fp);
    // fclose(fp);

    uint32_t tailStart = token_len/64*64;
    uint32_t tailLen = token_len - tailStart;
    if (tailLen > 0)
    {
        q->memcpy(outputs + tailStart * output_len, outputScratch, tailLen * output_len * sizeof(OT));
    }
    smgr->Unlock();
}


void runGemmQ6KL1_xmx_xe2(void *stream, const void *inputs, const void *weights, const void *bias, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int bias_precision, int output_precision, ScratchBufferMgr *smgr)
{
    assert(bias == nullptr);
    if (input_precision == 0 && output_precision == 0)
    {
        runGemmQ6KL1_xmx_xe2_impl(stream, (float *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len, smgr);
    }
    else if (input_precision == 0 && output_precision == 1)
    {
        runGemmQ6KL1_xmx_xe2_impl(stream, (float *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len, smgr);
    }
    else if (input_precision == 1 && output_precision == 0)
    {
        runGemmQ6KL1_xmx_xe2_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len, smgr);
    }
    else if (input_precision == 1 && output_precision == 1)
    {
        runGemmQ6KL1_xmx_xe2_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len, smgr);
    }
}

static RegisterGpuGemm g(GGML_TYPE_Q6_K, "GEMM.Q6_K.ESIMD.L1", PFLAG(PLATFORM_LNL) | PFLAG(PLATFORM_PTL) | PFLAG(PLATFORM_WCL), runGemmQ6KL1_xmx_xe2);
