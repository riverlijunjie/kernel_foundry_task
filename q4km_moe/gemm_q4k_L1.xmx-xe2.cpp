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
ESIMD_INLINE void gemm_q4kweights_xmx_xe2(fp16 *inputs, uint8_t *weights, OT *outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, OT* scratch, nd_item<2>& ndi, uint8_t *debug)
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

    uint8_t *pqs = weights;
    uint8_t *psl = pqs + num_blocks * 128;

    uint32_t localOffsetInput = v * 256 * input_len + vv * 64 * 16;
    uint32_t localOffsetWeightY = h *256 + hh * 32;
    uint32_t localOffsetWeightX = 0;
    uint32_t localOffsetQuantX = 0;

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
    simd<fp16, 32> zps_buf;

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
    uint8_t slshift = 0;
    uint8_t shshift = 0;
    BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)psl, num_blocks_per_row * 16 - 1, output_len - 1, num_blocks_per_row * 16 - 1, localOffsetQuantX, localOffsetWeightY);
    localOffsetQuantX += 4;
    slm_block_store<uint8_t, 512>(slm_offset, BData);

    // per 32
    BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)pqs, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);
    localOffsetWeightX += 4;

    simd<uint32_t, 32> Bytes = slm_block_load<uint32_t, 32>(slm_offset);
    simd<uint8_t, 32> lowbits0 = (Bytes >> slshift) & 0x0f;
    Bytes = slm_block_load<uint32_t, 32>(slm_offset + 32 * 4);
    simd<uint8_t, 32> lowbits1 = (Bytes >> slshift) & 0x0f;
    Bytes = slm_block_load<uint32_t, 32>(slm_offset + 32 * 8);
    simd<uint8_t, 32> highbits = (Bytes >> shshift) & 0x03;
    scales_buf = lowbits0 + (highbits << 4);
    highbits = (Bytes >> (shshift + 16)) & 0x03;
    zps_buf = lowbits1 + (highbits << 4);
    Bytes = slm_block_load<uint32_t, 32>(slm_offset + 32 * 12);
    scales_buf = scales_buf * Bytes.bit_cast_view<fp16>().select<32, 2>(0);
    zps_buf = zps_buf * Bytes.bit_cast_view<fp16>().select<32, 2>(1);

    slshift += 4;
    shshift += 2;

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

#pragma unroll
    for (int j = 0; j < 4; j++)
    {
        BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
        BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
        BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
        BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
    }

    BData0_tik = BData0 & 0x0f;
    BData1_tik = BData1 & 0x0f;

    simd<fp16, 32> s;
    simd<fp16, 32> z;

    s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData0_tik.select<32, 1>(j * 32) = BData0_tik.select<32, 1>(j * 32) * s - z;
    }

    s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData1_tik.select<32, 1>(j * 32) = BData1_tik.select<32, 1>(j * 32) * s - z;
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

        BData0_tok = BData0 >> 4;
        BData1_tok = BData1 >> 4;

        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
        z.select<16, 2>(1) = z.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tok.select<32, 1>(j * 32) = BData0_tok.select<32, 1>(j * 32) * s - z;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
        z.select<16, 2>(1) = z.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tok.select<32, 1>(j * 32) = BData1_tok.select<32, 1>(j * 32) * s - z;
        }

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

        if (slshift == 32)
        {
            slshift = 0;
            shshift = 0;
            BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)psl, num_blocks_per_row * 16 - 1, output_len - 1, num_blocks_per_row * 16 - 1, localOffsetQuantX, localOffsetWeightY);
            localOffsetQuantX += 4;
            slm_block_store<uint8_t, 512>(slm_offset, BData);
        }

        BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)pqs, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);
        localOffsetWeightX += 4;

        Bytes = slm_block_load<uint32_t, 32>(slm_offset);
        lowbits0 = (Bytes >> slshift) & 0x0f;
        Bytes = slm_block_load<uint32_t, 32>(slm_offset + 32 * 4);
        lowbits1 = (Bytes >> slshift) & 0x0f;
        Bytes = slm_block_load<uint32_t, 32>(slm_offset + 32 * 8);
        highbits = (Bytes >> shshift) & 0x03;
        scales_buf = lowbits0 + (highbits << 4);
        highbits = (Bytes >> (shshift + 16)) & 0x03;
        zps_buf = lowbits1 + (highbits << 4);
        Bytes = slm_block_load<uint32_t, 32>(slm_offset + 32 * 12);
        scales_buf = scales_buf * Bytes.bit_cast_view<fp16>().select<32, 2>(0);
        zps_buf = zps_buf * Bytes.bit_cast_view<fp16>().select<32, 2>(1);
        slshift += 4;
        shshift += 2;
        
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
        }
        BData0_tik = BData0 & 0x0f;
        BData1_tik = BData1 & 0x0f;

        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
        z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tik.select<32, 1>(j * 32) = BData0_tik.select<32, 1>(j * 32) * s - z;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
        z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tik.select<32, 1>(j * 32) = BData1_tik.select<32, 1>(j * 32) * s - z;
        }

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

    BData0_tok = BData0 >> 4;
    BData1_tok = BData1 >> 4;

    s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData0_tok.select<32, 1>(j * 32) = BData0_tok.select<32, 1>(j * 32) * s - z;
    }

    s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData1_tok.select<32, 1>(j * 32) = BData1_tok.select<32, 1>(j * 32) * s - z;
    }

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


template<typename OT>
ESIMD_INLINE void upandgate_q4kweights_xmx_xe2(fp16 *inputs, uint8_t *up, uint8_t *gate, OT *outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, OT* scratch, nd_item<2>& ndi, uint8_t *debug)
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

    uint8_t *pqs_up = up;
    uint8_t *psl_up = pqs_up + num_blocks * 128;
    uint8_t *pqs_gate = gate;
    uint8_t *psl_gate = pqs_gate + num_blocks * 128;

    uint32_t localOffsetInput = v * 256 * input_len + vv * 64 * 16;
    uint32_t localOffsetWeightY = h *128 + hh * 16;
    uint32_t localOffsetWeightX = 0;
    uint32_t localOffsetQuantX = 0;

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

    simd<fp16, 256> BData0_tik;
    simd<fp16, 256> BData1_tik;

    simd<fp16, 32> scales_buf;
    simd<fp16, 32> zps_buf;

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

    uint8_t slshift = 0;
    uint8_t shshift = 0;
    uint32_t loopNum = input_len / 16;
    for (uint32_t l = 0; l < loopNum; l++)
    {
        if (l % 16 == 0) // per 256
        {
            slshift = 0;
            shshift = 0;
            BData.bit_cast_view<uint32_t>().select<64, 1>(0) = load_2d<uint32_t, 4, 16, 1, true, false>((uint32_t *)psl_up, num_blocks_per_row * 16 - 1, output_len - 1, num_blocks_per_row * 16 - 1, localOffsetQuantX, localOffsetWeightY);
            BData.bit_cast_view<uint32_t>().select<64, 1>(64) = load_2d<uint32_t, 4, 16, 1, true, false>((uint32_t *)psl_gate, num_blocks_per_row * 16 - 1, output_len - 1, num_blocks_per_row * 16 - 1, localOffsetQuantX, localOffsetWeightY);
            localOffsetQuantX += 4;
            slm_block_store<uint8_t, 512>(slm_offset, BData);
        }

        if (l % 2 == 0) // per 32
        {
            BData.bit_cast_view<uint32_t>().select<64, 1>(0) = load_2d<uint32_t, 4, 16, 1, true, false>((uint32_t *)pqs_up, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);
            BData.bit_cast_view<uint32_t>().select<64, 1>(64) = load_2d<uint32_t, 4, 16, 1, true, false>((uint32_t *)pqs_gate, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);
            localOffsetWeightX += 4;

            simd<uint32_t, 32> Bytes;
            Bytes.select<16, 1>(0) = slm_block_load<uint32_t, 16>(slm_offset);
            Bytes.select<16, 1>(16) = slm_block_load<uint32_t, 16>(slm_offset + 256);
            simd<uint8_t, 32> lowbits0 = (Bytes >> slshift) & 0x0f;
            Bytes.select<16, 1>(0) = slm_block_load<uint32_t, 16>(slm_offset + 16 * 4);
            Bytes.select<16, 1>(16) = slm_block_load<uint32_t, 16>(slm_offset + 256 + 16 * 4);
            simd<uint8_t, 32> lowbits1 = (Bytes >> slshift) & 0x0f;
            Bytes.select<16, 1>(0) = slm_block_load<uint32_t, 16>(slm_offset + 16 * 8);
            Bytes.select<16, 1>(16) = slm_block_load<uint32_t, 16>(slm_offset + 256 + 16 * 8);
            simd<uint8_t, 32> highbits = (Bytes >> shshift) & 0x03;
            scales_buf = lowbits0 + (highbits << 4);
            highbits = (Bytes >> (shshift + 16)) & 0x03;
            zps_buf = lowbits1 + (highbits << 4);
            Bytes.select<16, 1>(0) = slm_block_load<uint32_t, 16>(slm_offset + 16 * 12);
            Bytes.select<16, 1>(16) = slm_block_load<uint32_t, 16>(slm_offset + 256 + 16 * 12);
            scales_buf = scales_buf * Bytes.bit_cast_view<fp16>().select<32, 2>(0);
            zps_buf = zps_buf * Bytes.bit_cast_view<fp16>().select<32, 2>(1);

            slshift += 4;
            shshift += 2;

    #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 16);
                BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 16 + 1);
                BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 16 + 128);
                BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 16 + 128 + 1);
            }

        }

        AData0_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
        AData1_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
        AData2_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
        AData3_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
        AData4_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
        AData5_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
        AData6_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
        AData7_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);
        
        localOffsetInput += 16 * 256;

        if (l % 2 == 0) {
            BData0_tik = BData0 & 0x0f;
            BData1_tik = BData1 & 0x0f;
        } else {
            BData0_tik = BData0 >> 4;
            BData1_tik = BData1 >> 4;
        }

        simd<fp16, 32> s;
        simd<fp16, 32> z;

        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
        z.select<16, 2>(1) = z.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tik.select<32, 1>(j * 32) = BData0_tik.select<32, 1>(j * 32) * s - z;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
        z.select<16, 2>(1) = z.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tik.select<32, 1>(j * 32) = BData1_tik.select<32, 1>(j * 32) * s - z;
        }


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

    }

    uint32_t localOutputOffsetX = h * 128 + hh * 16;
    uint32_t localOutputOffsetY = v * 256 + vv * 64;



    OT *write_buffer = outputs;
    uint32_t surfaceHeight = token_len;
    simd<OT, 128> outData0;
    simd<OT, 128> outData1;

    simd<float, 128> temp = exp(-CData01);
    temp = temp + 1.0f;
    temp = 1.0f / temp;
    temp = temp * CData01;
    outData0 = temp * CData00;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY, outData0);

    temp = exp(-CData11);
    temp = temp + 1.0f;
    temp = 1.0f / temp;
    temp = temp * CData11;
    outData0 = temp * CData10;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY + 8, outData0);

    temp = exp(-CData21);
    temp = temp + 1.0f;
    temp = 1.0f / temp;
    temp = temp * CData21;
    outData0 = temp * CData20;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY + 16, outData0);

    temp = exp(-CData31);
    temp = temp + 1.0f;
    temp = 1.0f / temp;
    temp = temp * CData31;
    outData0 = temp * CData30;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY + 24, outData0);

    temp = exp(-CData41);
    temp = temp + 1.0f;
    temp = 1.0f / temp;
    temp = temp * CData41;
    outData0 = temp * CData40;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY + 32, outData0);

    temp = exp(-CData51);
    temp = temp + 1.0f;
    temp = 1.0f / temp;
    temp = temp * CData51;
    outData0 = temp * CData50;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY + 40, outData0);

    temp = exp(-CData61);
    temp = temp + 1.0f;
    temp = 1.0f / temp;
    temp = temp * CData61;
    outData0 = temp * CData60;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY + 48, outData0);

    temp = exp(-CData71);
    temp = temp + 1.0f;
    temp = 1.0f / temp;
    temp = temp * CData71;
    outData0 = temp * CData70;
    store_2d<OT, 16, 8>(write_buffer, output_len * sizeof(OT) - 1, surfaceHeight, output_len * sizeof(OT) - 1, localOutputOffsetX, localOutputOffsetY + 56, outData0);
}


template <typename IT, typename OT>
void runGemmQ4KL1_xmx_xe2_impl(void *stream, IT *inputs, uint8_t *weights, OT* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, ScratchBufferMgr *smgr)
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
            gemm_q4kweights_xmx_xe2(shuffled_inputs, weights, outputs, token_len, input_len, output_len, outputScratch, ndi, nullptr);
        });
    });

    uint32_t tailStart = token_len/64*64;
    uint32_t tailLen = token_len - tailStart;
    if (tailLen > 0)
    {
        q->memcpy(outputs + tailStart * output_len, outputScratch, tailLen * output_len * sizeof(OT));
    }
    smgr->Unlock();
}

template <typename IT, typename OT>
void runUpAndGateMatQ4KL1_xmx_xe2_impl(void *stream, const IT *inputs, const uint8_t *up, const uint8_t *gate, OT* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, ScratchBufferMgr *smgr)
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

    IT *nc_inputs = const_cast<IT *>(inputs);
    uint8_t *nc_up = const_cast<uint8_t *>(up);
    uint8_t *nc_gate = const_cast<uint8_t *>(gate);

    q->submit([&](handler& cgh) {
        cgh.parallel_for(RangeShuffle, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            shuffle_input(nc_inputs, shuffled_inputs, token_len, input_len, ndi);
        });
    });

    uint32_t threadA = (token_len + 255)/256;
    uint32_t threadB = (output_len + 127)/128;
    uint32_t localThreadNum = 32;

    sycl::range<2> GlobalRange(localThreadNum * threadB, threadA);
    sycl::range<2> LocalRange(localThreadNum, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    OT *outputScratch = (OT *)smgr->Allocate(q, 64 * output_len * sizeof(OT));

    // uint8_t* debug = (uint8_t *)malloc_shared(2560 * 1024, *q);

    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            upandgate_q4kweights_xmx_xe2(shuffled_inputs, nc_up, nc_gate, outputs, token_len, input_len, output_len, outputScratch, ndi, nullptr);
        });
    });

    // the up and gate are called by internal, so the output buffer are pre-allocated and we can make sure they are 64 token aligned
    // uint32_t tailStart = token_len/64*64;
    // uint32_t tailLen = token_len - tailStart;
    // if (tailLen > 0)
    // {
    //     q->memcpy(outputs + tailStart * output_len, outputScratch, tailLen * output_len * sizeof(OT));
    // }
    smgr->Unlock();
}

void runGemmQ4KL1_xmx_xe2(void *stream, const void *inputs, const void *weights, const void *bias, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int bias_precision, int output_precision, ScratchBufferMgr *smgr)
{
    assert(bias == nullptr);
    if (input_precision == 0 && output_precision == 0)
    {
        runGemmQ4KL1_xmx_xe2_impl(stream, (float *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len, smgr);
    }
    else if (input_precision == 0 && output_precision == 1)
    {
        runGemmQ4KL1_xmx_xe2_impl(stream, (float *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len, smgr);
    }
    else if (input_precision == 1 && output_precision == 0)
    {
        runGemmQ4KL1_xmx_xe2_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len, smgr);
    }
    else if (input_precision == 1 && output_precision == 1)
    {
        runGemmQ4KL1_xmx_xe2_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len, smgr);
    }
}

void runUpAndGateMatQ4KL1_xmx_xe2(void *stream, const void *inputs, const void *up, const void *gate, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int output_precision, ScratchBufferMgr *smgr)
{
    if (input_precision == 0 && output_precision == 0)
    {
        runUpAndGateMatQ4KL1_xmx_xe2_impl(stream, (float *)inputs, (uint8_t *)up, (uint8_t *)gate, (float *)outputs, token_len, input_len, output_len, smgr);
    }
    else if (input_precision == 0 && output_precision == 1)
    {
        runUpAndGateMatQ4KL1_xmx_xe2_impl(stream, (float *)inputs, (uint8_t *)up, (uint8_t *)gate, (fp16 *)outputs, token_len, input_len, output_len, smgr);
    }
    else if (input_precision == 1 && output_precision == 0)
    {
        runUpAndGateMatQ4KL1_xmx_xe2_impl(stream, (fp16 *)inputs, (uint8_t *)up, (uint8_t *)gate, (float *)outputs, token_len, input_len, output_len, smgr);
    }
    else if (input_precision == 1 && output_precision == 1)
    {
        runUpAndGateMatQ4KL1_xmx_xe2_impl(stream, (fp16 *)inputs, (uint8_t *)up, (uint8_t *)gate, (fp16 *)outputs, token_len, input_len, output_len, smgr);
    }
}


// VariousGemm: fused dispatch over multiple weight matrices sharing the same shuffled input.
// shuffle_input runs once; one GEMM kernel + optional tail-memcpy is submitted per expert.
// outputScratch is sized to 64 * max(output_lens) and reused across experts — safe because
// the SYCL in-order queue guarantees each kernel+memcpy pair completes before the next expert.
template <typename IT, typename OT>
void runVariousGemmQ4KL1_xmx_xe2_impl(void *stream, IT *inputs, const uint8_t **weights_vec, OT **outputs_vec,
                                       uint32_t token_len, uint32_t input_len,
                                       uint32_t *output_lens, uint32_t output_count,
                                       ScratchBufferMgr *smgr)
{
    sycl::queue *q = (sycl::queue *)stream;
    smgr->Lock();

    const uint32_t blockWidth = 4096;
    uint32_t threadShuffleH  = (input_len + blockWidth - 1) / blockWidth;
    uint32_t alignedTokenLen = (token_len + 255) / 256 * 256;
    uint32_t threadShuffleV  = alignedTokenLen / 16;
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

    uint32_t max_output_len = 0;
    for (uint32_t i = 0; i < output_count; i++)
        if (output_lens[i] > max_output_len) max_output_len = output_lens[i];
    OT *outputScratch = (OT *)smgr->Allocate(q, 64 * max_output_len * sizeof(OT));

    uint32_t tailStart = token_len / 64 * 64;
    uint32_t tailLen   = token_len - tailStart;

    for (uint32_t i = 0; i < output_count; i++)
    {
        uint8_t *weights    = const_cast<uint8_t *>(weights_vec[i]);
        OT      *outputs    = outputs_vec[i];
        uint32_t output_len = output_lens[i];

        uint32_t threadA        = (token_len  + 255) / 256;
        uint32_t threadB        = (output_len + 255) / 256;
        uint32_t localThreadNum = 32;

        sycl::range<2> GlobalRange(localThreadNum * threadB, threadA);
        sycl::range<2> LocalRange(localThreadNum, 1);
        sycl::nd_range<2> Range(GlobalRange, LocalRange);

        q->submit([&](handler& cgh) {
            cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                gemm_q4kweights_xmx_xe2(shuffled_inputs, weights, outputs, token_len, input_len, output_len, outputScratch, ndi, nullptr);
            });
        });

        if (tailLen > 0)
        {
            q->memcpy(outputs + tailStart * output_len, outputScratch, tailLen * output_len * sizeof(OT));
        }
    }

    smgr->Unlock();
}

void runVariousGemmQ4KL1_xmx_xe2(void *stream, const void *inputs, const void **weights, void **outputs,
                                   uint32_t token_len, uint32_t input_len, uint32_t *output_lens,
                                   uint32_t output_count, int input_precision, int output_precision,
                                   ScratchBufferMgr *smgr)
{
    if (input_precision == 0 && output_precision == 0)
        runVariousGemmQ4KL1_xmx_xe2_impl(stream, (float *)inputs, (const uint8_t **)weights, (float **)outputs, token_len, input_len, output_lens, output_count, smgr);
    else if (input_precision == 0 && output_precision == 1)
        runVariousGemmQ4KL1_xmx_xe2_impl(stream, (float *)inputs, (const uint8_t **)weights, (fp16 **)outputs, token_len, input_len, output_lens, output_count, smgr);
    else if (input_precision == 1 && output_precision == 0)
        runVariousGemmQ4KL1_xmx_xe2_impl(stream, (fp16 *)inputs, (const uint8_t **)weights, (float **)outputs, token_len, input_len, output_lens, output_count, smgr);
    else if (input_precision == 1 && output_precision == 1)
        runVariousGemmQ4KL1_xmx_xe2_impl(stream, (fp16 *)inputs, (const uint8_t **)weights, (fp16 **)outputs, token_len, input_len, output_lens, output_count, smgr);
}


static RegisterGpuGemm g(GGML_TYPE_Q4_K, "GEMM.Q4_K.ESIMD.L1", PFLAG(PLATFORM_LNL) | PFLAG(PLATFORM_PTL) | PFLAG(PLATFORM_WCL), runGemmQ4KL1_xmx_xe2);
static RegisterGpuVariousGemm vv(GGML_TYPE_Q4_K, "GEMM.Q4_K.ESIMD.L1", PFLAG(PLATFORM_LNL) | PFLAG(PLATFORM_PTL) | PFLAG(PLATFORM_WCL), runVariousGemmQ4KL1_xmx_xe2);
static RegisterGpuMatUpAndGate gm(GGML_TYPE_Q4_K, "GEMM.Q4_K.ESIMD.L1", PFLAG(PLATFORM_LNL) | PFLAG(PLATFORM_PTL) | PFLAG(PLATFORM_WCL), runUpAndGateMatQ4KL1_xmx_xe2);
