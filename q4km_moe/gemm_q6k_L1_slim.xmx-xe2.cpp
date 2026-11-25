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

// Input shuffle: converts [token_len, input_len] IT inputs to fp16 in DPAS-friendly layout.
// After shuffle, [8 tokens x 16 K-elements] = 128 fp16 tiles are contiguous.
// Copied from gemm_q6k_L1.xmx-xe2.cpp with unique name to avoid ODR violations.
template<typename IT>
ESIMD_INLINE void shuffle_input_slim(IT *inputs, fp16 *outputs, uint32_t token_len, uint32_t input_len, nd_item<2>& ndi)
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

// Slim GEMM compute kernel for Q6K weights using DPAS.
// 1D dispatch: threads split K dimension, SLM reduction to produce output.
// Each thread handles one Q6K block (256 K-elements) for OPG=32 output columns.
// Tokens batched in groups of 8 to fill DPAS A-matrix.
constexpr uint32_t OPG_Q6K_SLIM = 32;

template<typename OT>
ESIMD_INLINE void gemm_q6k_slim_xmx_xe2(fp16 *inputs, uint8_t *weights, OT *outputs,
    uint32_t token_len, uint32_t input_len, uint32_t output_len, nd_item<1>& ndi)
{
    constexpr uint32_t OPG = OPG_Q6K_SLIM;
    uint32_t localRange = ndi.get_local_range(0);

    // SLM: scale caching (512 bytes/thread) during K-loop, then reused for reduction
    slm_init(64 * 512);

    uint32_t h = ndi.get_group(0);
    uint32_t hh = ndi.get_local_id(0);

    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;

    uint8_t *pql = weights;
    uint8_t *pqh = pql + num_blocks * 128;
    int8_t *ps = (int8_t *)(pqh + num_blocks * 64);
    fp16 *pd = (fp16 *)(ps + num_blocks * 16);

    // Input offset in shuffled buffer:
    // K-step stride = 16 * 256 fp16. Thread hh starts at K-step hh*16.
    uint32_t localOffsetInput = hh * 16 * 16 * 256;

    // Weight addressing: Y = output column, X = K position
    uint32_t localOffsetWeightY = h * OPG;
    uint32_t localOffsetWeightX_low = hh * 32;   // ql: hh * 128 bytes / 4
    uint32_t localOffsetWeightX_high = hh * 16;   // qh: hh * 64 bytes / 4
    uint32_t localOffsetScaleX = hh * 4;           // ps: hh * 16 bytes / 4

    // Load super-block scale (d) for 32 output cols at Q6K block hh
    simd<uint32_t, 32> d_offsets({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    d_offsets += (h * OPG);
    d_offsets *= (num_blocks_per_row * sizeof(fp16));
    d_offsets += hh * sizeof(fp16);
    simd_mask<32> gather_mask = 1;
    simd<fp16, 32> global_d = gather<fp16, 32, 1>(pd, d_offsets, gather_mask);

    // Load all 16 scales for this Q6K block across 32 outputs into SLM
    uint32_t slm_offset = hh * 512;
    simd<uint8_t, 512> scaleData;
    scaleData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>(
        (uint32_t *)ps, num_blocks_per_row * 16 - 1, output_len - 1,
        num_blocks_per_row * 16 - 1, localOffsetScaleX, localOffsetWeightY);
    slm_block_store<uint8_t, 512>(slm_offset, scaleData);

    // Accumulators: 2 token batches x 2 output tiles
    simd<float, 128> CData00 = 0.0f;  // batch0 (tokens 0-7),  cols 0-15
    simd<float, 128> CData01 = 0.0f;  // batch0 (tokens 0-7),  cols 16-31
    simd<float, 128> CData10 = 0.0f;  // batch1 (tokens 8-15), cols 0-15
    simd<float, 128> CData11 = 0.0f;  // batch1 (tokens 8-15), cols 16-31

    // Weight registers
    simd<uint8_t, 512> BData;
    simd<uint8_t, 256> BData0;
    simd<uint8_t, 256> BData1;
    simd<uint8_t, 128> BData0_hi;
    simd<uint8_t, 128> BData1_hi;
    simd<fp16, 256> BWeights0;
    simd<fp16, 256> BWeights1;

    // Input registers (token batches)
    simd<fp16, 128> AData0;
    simd<fp16, 128> AData1;

    simd<fp16, 32> scales_buf;
    simd<fp16, 32> s;
    uint8_t inner_idx = 0;

    // Main loop: 8 iterations, each processes 2 K-steps (lo + hi nibble from same ql)
    for (int i = 0; i < 8; i++)
    {
        // --- Load ql (4x32 uint32 tile) ---
        BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>(
            (uint32_t *)pql, input_len / 2 - 1, output_len - 1,
            input_len / 2 - 1, localOffsetWeightX_low, localOffsetWeightY);
        localOffsetWeightX_low += 4;

        // Deinterleave into BData0 (cols 0-15) and BData1 (cols 16-31)
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
        }

        // --- Load qh (2x32 uint32 tile) ---
        BData.bit_cast_view<uint32_t>().select<64, 1>(0) = load_2d<uint32_t, 2, 32, 1, true, false>(
            (uint32_t *)pqh, input_len / 4 - 1, output_len - 1,
            input_len / 4 - 1, localOffsetWeightX_high, localOffsetWeightY);
        localOffsetWeightX_high += 2;

        // Deinterleave qh
#pragma unroll
        for (int j = 0; j < 2; j++)
        {
            BData0_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
            BData0_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
            BData1_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
            BData1_hi.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
        }

        // ===== Sub-step A: low nibble (16 K-elements) =====
        {
            uint8_t blk_idx = inner_idx >> 2;
            uint8_t step_idx = inner_idx & 0x03;
            simd<int8_t, 128> Bytes = slm_block_load<int8_t, 128>(slm_offset + blk_idx * 128);
            scales_buf = Bytes.select<32, 4>(step_idx);
            scales_buf = scales_buf * global_d;
            ++inner_idx;

            // Dequant low nibble: reconstruct 6-bit values from ql low nibble + qh bits
            BWeights0.select<64, 1>(0)   = (BData0.select<64, 1>(0)   & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0x03) << 4);
            BWeights0.select<64, 1>(64)  = (BData0.select<64, 1>(64)  & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0x0c) << 2);
            BWeights0.select<64, 1>(128) = (BData0.select<64, 1>(128) & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0x30));
            BWeights0.select<64, 1>(192) = (BData0.select<64, 1>(192) & 0x0f) + ((BData0_hi.select<64, 1>(0) & 0xc0) >> 2);

            BWeights1.select<64, 1>(0)   = (BData1.select<64, 1>(0)   & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0x03) << 4);
            BWeights1.select<64, 1>(64)  = (BData1.select<64, 1>(64)  & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0x0c) << 2);
            BWeights1.select<64, 1>(128) = (BData1.select<64, 1>(128) & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0x30));
            BWeights1.select<64, 1>(192) = (BData1.select<64, 1>(192) & 0x0f) + ((BData1_hi.select<64, 1>(0) & 0xc0) >> 2);

            // Apply scales: (val - 32) * combined_scale
            s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
            s.select<16, 2>(1) = s.select<16, 2>(0);
#pragma unroll
            for (int j = 0; j < 8; j++)
                BWeights0.select<32, 1>(j * 32) = (BWeights0.select<32, 1>(j * 32) - 32) * s;

            s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
            s.select<16, 2>(1) = s.select<16, 2>(0);
#pragma unroll
            for (int j = 0; j < 8; j++)
                BWeights1.select<32, 1>(j * 32) = (BWeights1.select<32, 1>(j * 32) - 32) * s;

            // Load A-data from shuffled buffer
            AData0 = block_load<fp16, 128>(inputs + localOffsetInput);       // tokens 0-7
            if (token_len > 8)
                AData1 = block_load<fp16, 128>(inputs + localOffsetInput + 128); // tokens 8-15
            localOffsetInput += 16 * 256;

            // DPAS: accumulate partial sums
            CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BWeights0, AData0);
            CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BWeights1, AData0);
            if (token_len > 8)
            {
                CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BWeights0, AData1);
                CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BWeights1, AData1);
            }
        }

        // ===== Sub-step B: high nibble (next 16 K-elements) =====
        {
            uint8_t blk_idx = inner_idx >> 2;
            uint8_t step_idx = inner_idx & 0x03;
            simd<int8_t, 128> Bytes = slm_block_load<int8_t, 128>(slm_offset + blk_idx * 128);
            scales_buf = Bytes.select<32, 4>(step_idx);
            scales_buf = scales_buf * global_d;
            ++inner_idx;

            // Dequant high nibble: ql >> 4, different qh bits
            BWeights0.select<64, 1>(0)   = (BData0.select<64, 1>(0)   >> 4) + ((BData0_hi.select<64, 1>(64) & 0x03) << 4);
            BWeights0.select<64, 1>(64)  = (BData0.select<64, 1>(64)  >> 4) + ((BData0_hi.select<64, 1>(64) & 0x0c) << 2);
            BWeights0.select<64, 1>(128) = (BData0.select<64, 1>(128) >> 4) + ((BData0_hi.select<64, 1>(64) & 0x30));
            BWeights0.select<64, 1>(192) = (BData0.select<64, 1>(192) >> 4) + ((BData0_hi.select<64, 1>(64) & 0xc0) >> 2);

            BWeights1.select<64, 1>(0)   = (BData1.select<64, 1>(0)   >> 4) + ((BData1_hi.select<64, 1>(64) & 0x03) << 4);
            BWeights1.select<64, 1>(64)  = (BData1.select<64, 1>(64)  >> 4) + ((BData1_hi.select<64, 1>(64) & 0x0c) << 2);
            BWeights1.select<64, 1>(128) = (BData1.select<64, 1>(128) >> 4) + ((BData1_hi.select<64, 1>(64) & 0x30));
            BWeights1.select<64, 1>(192) = (BData1.select<64, 1>(192) >> 4) + ((BData1_hi.select<64, 1>(64) & 0xc0) >> 2);

            s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
            s.select<16, 2>(1) = s.select<16, 2>(0);
#pragma unroll
            for (int j = 0; j < 8; j++)
                BWeights0.select<32, 1>(j * 32) = (BWeights0.select<32, 1>(j * 32) - 32) * s;

            s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
            s.select<16, 2>(1) = s.select<16, 2>(0);
#pragma unroll
            for (int j = 0; j < 8; j++)
                BWeights1.select<32, 1>(j * 32) = (BWeights1.select<32, 1>(j * 32) - 32) * s;

            // Load A-data for next K position
            AData0 = block_load<fp16, 128>(inputs + localOffsetInput);
            if (token_len > 8)
                AData1 = block_load<fp16, 128>(inputs + localOffsetInput + 128);
            localOffsetInput += 16 * 256;

            CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BWeights0, AData0);
            CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BWeights1, AData0);
            if (token_len > 8)
            {
                CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BWeights0, AData1);
                CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BWeights1, AData1);
            }
        }
    }

    // === SLM Reduction (batched: 8 tokens per round, distributed across threads) ===
    // Instead of 1 barrier per token (33 barriers for 16 tokens), batch all tokens
    // in a batch into SLM at once, then reduce with 1 barrier per batch (4 barriers total).
    barrier();

    // SLM layout for reduction: thread k stores at k * 8 * OPG * sizeof(float)
    // Each thread stores 8 partial results of OPG floats = 1024 bytes
    constexpr uint32_t SLM_RED_STRIDE = 8 * OPG * sizeof(float);  // 1024 bytes per thread

    // Round 1: batch 0 (tokens 0..min(8, token_len)-1)
    {
        uint32_t numTokens = token_len > 8 ? 8 : token_len;
        for (uint32_t v = 0; v < numTokens; v++)
        {
            simd<float, OPG> partial;
            partial.select<16, 1>(0) = CData00.select<16, 1>(v * 16);
            partial.select<16, 1>(16) = CData01.select<16, 1>(v * 16);
            slm_block_store<float, OPG>(hh * SLM_RED_STRIDE + v * OPG * sizeof(float), partial);
        }
        barrier();
        // Distribute reduction: each thread handles tokens at stride localRange
        for (uint32_t v = hh; v < numTokens; v += localRange)
        {
            simd<float, OPG> accOut = 0.0f;
            for (uint32_t k = 0; k < localRange; k++)
                accOut += slm_block_load<float, OPG>(k * SLM_RED_STRIDE + v * OPG * sizeof(float));
            block_store<OT, OPG>(outputs + v * output_len + h * OPG, accOut);
        }
    }

    // Round 2: batch 1 (tokens 8..token_len-1)
    if (token_len > 8)
    {
        barrier();
        uint32_t numTokens = token_len - 8;
        for (uint32_t v = 0; v < numTokens; v++)
        {
            simd<float, OPG> partial;
            partial.select<16, 1>(0) = CData10.select<16, 1>(v * 16);
            partial.select<16, 1>(16) = CData11.select<16, 1>(v * 16);
            slm_block_store<float, OPG>(hh * SLM_RED_STRIDE + v * OPG * sizeof(float), partial);
        }
        barrier();
        for (uint32_t v = hh; v < numTokens; v += localRange)
        {
            simd<float, OPG> accOut = 0.0f;
            for (uint32_t k = 0; k < localRange; k++)
                accOut += slm_block_load<float, OPG>(k * SLM_RED_STRIDE + v * OPG * sizeof(float));
            block_store<OT, OPG>(outputs + (v + 8) * output_len + h * OPG, accOut);
        }
    }
}

template <typename IT, typename OT>
void runGemmQ6KL1_slim_xmx_xe2_impl(void *stream, IT *inputs, uint8_t *weights, OT *outputs,
    uint32_t token_len, uint32_t input_len, uint32_t output_len, ScratchBufferMgr *smgr)
{
    sycl::queue *q = (sycl::queue *)stream;
    smgr->Lock();

    // Phase 1: Shuffle input to DPAS-friendly layout
    const uint32_t blockWidth = 4096;
    uint32_t alignedTokenLen = (token_len + 255) / 256 * 256;
    uint32_t threadShuffleH = (input_len + blockWidth - 1) / blockWidth;
    uint32_t threadShuffleV = alignedTokenLen / 16;
    uint32_t localThreadShuffle = 32;

    fp16 *shuffled_inputs = (fp16 *)smgr->Allocate(q, alignedTokenLen * input_len * sizeof(fp16));

    sycl::range<2> GlobalRangeShuffle(localThreadShuffle * threadShuffleH, threadShuffleV);
    sycl::range<2> LocalRangeShuffle(localThreadShuffle, 1);
    sycl::nd_range<2> RangeShuffle(GlobalRangeShuffle, LocalRangeShuffle);

    q->submit([&](handler& cgh) {
        cgh.parallel_for(RangeShuffle, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            shuffle_input_slim(inputs, shuffled_inputs, token_len, input_len, ndi);
        });
    });

    // Phase 2: DPAS compute with K-split and SLM reduction
    constexpr uint32_t OPG = OPG_Q6K_SLIM;
    uint32_t groups = (output_len + OPG - 1) / OPG;
    uint32_t localRange = input_len / 256;  // one thread per Q6K block

    sycl::nd_range<1> Range(groups * localRange, localRange);

    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            gemm_q6k_slim_xmx_xe2(shuffled_inputs, weights, outputs,
                token_len, input_len, output_len, ndi);
        });
    });

    smgr->Unlock();
}

void runGemmQ6KL1_slim_xmx_xe2(void *stream, const void *inputs, const void *weights, const void *bias, void *outputs,
    uint32_t token_len, uint32_t input_len, uint32_t output_len,
    int input_precision, int bias_precision, int output_precision, ScratchBufferMgr *smgr)
{
    assert(bias == nullptr);
    if (input_precision == 0 && output_precision == 0)
        runGemmQ6KL1_slim_xmx_xe2_impl(stream, (float *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len, smgr);
    else if (input_precision == 0 && output_precision == 1)
        runGemmQ6KL1_slim_xmx_xe2_impl(stream, (float *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len, smgr);
    else if (input_precision == 1 && output_precision == 0)
        runGemmQ6KL1_slim_xmx_xe2_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len, smgr);
    else if (input_precision == 1 && output_precision == 1)
        runGemmQ6KL1_slim_xmx_xe2_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len, smgr);
}

static RegisterGpuSlimGemm g(GGML_TYPE_Q6_K, "GEMM.Q6_K.ESIMD.L1", PFLAG(PLATFORM_LNL) | PFLAG(PLATFORM_PTL) | PFLAG(PLATFORM_WCL), runGemmQ6KL1_slim_xmx_xe2);
