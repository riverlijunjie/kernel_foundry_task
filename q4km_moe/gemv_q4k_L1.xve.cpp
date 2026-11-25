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

template <typename IT, typename OT>
void runGemvQ4KL1_xve_impl(void *stream, IT *inputs, uint8_t *weights, OT* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;
    uint8_t *pqs = weights;
    uint8_t *psl = pqs + num_blocks * 128;

    const uint32_t outputPerGroup = 16;
    uint32_t groups = (output_len + outputPerGroup - 1) / outputPerGroup;
    uint32_t localRange = num_blocks_per_row;
    sycl::nd_range<1> Range(groups * localRange, localRange);

    sycl::queue *q = (sycl::queue *)stream;
    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            slm_init(sizeof(float) * 64 * outputPerGroup);

            simd<uint32_t, 8> low_mask({0x0000000f, 0x000000f0, 0x00000f00, 0x0000f000, 0x000f0000, 0x00f00000, 0x0f000000, 0xf0000000});
            simd<uint32_t, 8> low_shift({0, 4, 8, 12, 16, 20, 24, 28});
            simd<uint32_t, 8> high_mask({0x0003, 0x000c, 0x0030, 0x00c0, 0x0300, 0x0c00, 0x3000, 0xc000});
            simd<uint32_t, 8> high_shift({0, 2, 4, 6, 8, 10, 12, 14});

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);

            simd<float, 256> inputData = block_load<IT, 256>(inputs + 256 * hh);
            uint32_t blk_index = h * outputPerGroup * num_blocks_per_row + hh;
            uint32_t qs_offset = blk_index * 128;
            uint32_t sl_offset = blk_index * 16;

            simd<float, outputPerGroup> outputData = 0.0;

    #pragma unroll
            for (int l = 0; l < outputPerGroup; l++)
            {
                simd<uint8_t, 128> qsData = block_load<uint8_t, 128>(pqs + qs_offset);
                simd<uint8_t, 16> temp = block_load<uint8_t, 16>(psl + sl_offset);
                uint32_t slData = temp.bit_cast_view<uint32_t>()[0];
                uint32_t mlData = temp.bit_cast_view<uint32_t>()[1];
                uint16_t shData = temp.bit_cast_view<uint16_t>()[4];
                uint16_t mhData = temp.bit_cast_view<uint16_t>()[5];
                fp16 d_temp = temp.bit_cast_view<fp16>()[6];
                fp16 dmin_temp = temp.bit_cast_view<fp16>()[7];
                float d = (float)d_temp;
                float dmin = (float)dmin_temp;

                qs_offset += num_blocks_per_row * 128;
                sl_offset += num_blocks_per_row * 16;

                simd<uint32_t, 8> scalequant = slData & low_mask;
                scalequant = scalequant >> low_shift;
                simd<uint16_t, 8> scalequanthi = shData & high_mask;
                scalequanthi = scalequanthi >> high_shift;
                scalequanthi = scalequanthi << 4;
                scalequant = scalequant + scalequanthi;
                simd<float, 8> scaleData = scalequant * d;

                simd<uint32_t, 8> minquant = mlData & low_mask;
                minquant = minquant >> low_shift;
                simd<uint16_t, 8> minquanthi = mhData & high_mask;
                minquanthi = minquanthi >> high_shift;
                minquanthi = minquanthi << 4;
                minquant = minquant + minquanthi;
                simd<float, 8> minData = minquant * dmin;

                simd<float, 32> acc = 0.0;

    #pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    simd<float, 32> weightData;
                    weightData.select<16, 1>(0) = qsData.select<16, 1>(16 * j) & 0x0f;
                    weightData.select<16, 1>(16) = (qsData.select<16, 1>(16 * j) >> 4) & 0x0f;
                    float s = scaleData[j];
                    float m = minData[j];
                    weightData = weightData * s - m;

                    acc += inputData.select<32, 1>(j * 32) * weightData;
                }

                outputData[l] = sycl::ext::intel::esimd::detail::sum<float, float, 32>(acc);
            }

            slm_block_store<float, outputPerGroup>(sizeof(float) * hh * outputPerGroup, outputData);

            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> accOut = 0.0;
                for (int i = 0; i < num_blocks_per_row; i++)
                {
                    accOut += slm_block_load<float, outputPerGroup>(sizeof(float) * i * outputPerGroup);
                }

                block_store<OT, outputPerGroup>(outputs + h * outputPerGroup, accOut);
            }

        });
    });
}


void runGemvQ4KL1_xve(void *stream, const void *inputs, const void *weights, const void *bias, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int bias_precision, int output_precision, ScratchBufferMgr *smgr)
{
    assert(bias == nullptr);
    if (input_precision == 0 && output_precision == 0)
    {
        runGemvQ4KL1_xve_impl(stream, (float *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 0 && output_precision == 1)
    {
        runGemvQ4KL1_xve_impl(stream, (float *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 1 && output_precision == 0)
    {
        runGemvQ4KL1_xve_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 1 && output_precision == 1)
    {
        runGemvQ4KL1_xve_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len);
    }
}


// OPG: outputs per work-group. BS: Q4K blocks per thread (each block = 256 inputs).
template <uint32_t OPG, uint32_t BS, typename IT, typename OT>
void runGroupUpAndGateVecQ4KL1_xve_impl(void *stream, const IT *inputs, const uint8_t *ups, const uint8_t *gates, const uint32_t *indexes, OT* outputs, uint32_t output_num, uint32_t token_len, uint32_t input_len, uint32_t output_len)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;
    uint32_t expert_size = num_blocks * 144;

    uint32_t aligned_output_num = (output_num + 15) / 16 * 16;

    constexpr uint32_t outputPerGroup = OPG;
    constexpr uint32_t numChunks = BS;  // blocks per thread
    uint32_t groups = (output_len + outputPerGroup - 1) / outputPerGroup;
    uint32_t localThread = num_blocks_per_row / numChunks;
    sycl::range<3> GlobalRange(groups * localThread, output_num, token_len);
    sycl::range<3> LocalRange(localThread, 1, 1);
    sycl::nd_range<3> Range(GlobalRange, LocalRange);

    sycl::queue *q = (sycl::queue *)stream;
    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            slm_init(sizeof(float) * 64 * outputPerGroup * 2);

            constexpr uint32_t slmOffsetUpResult = 0;
            constexpr uint32_t slmOffsetGateResult = sizeof(float) * 64 * outputPerGroup;

            simd<uint32_t, 8> low_mask({0x0000000f, 0x000000f0, 0x00000f00, 0x0000f000, 0x000f0000, 0x00f00000, 0x0f000000, 0xf0000000});
            simd<uint32_t, 8> low_shift({0, 4, 8, 12, 16, 20, 24, 28});
            simd<uint32_t, 8> high_mask({0x0003, 0x000c, 0x0030, 0x00c0, 0x0300, 0x0c00, 0x3000, 0xc000});
            simd<uint32_t, 8> high_shift({0, 2, 4, 6, 8, 10, 12, 14});

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);
            int v = ndi.get_group(1);
            int t = ndi.get_group(2);

            uint32_t index = indexes ? indexes[t * aligned_output_num + v] : (uint32_t)v;

            simd<float, outputPerGroup> outputDataUp = 0.0;
            simd<float, outputPerGroup> outputDataGate = 0.0;

            for (uint32_t c = 0; c < numChunks; c++)
            {
                uint32_t blockIdx = hh * numChunks + c;  // which block within the row
                simd<float, 256> inputData = block_load<IT, 256>(inputs + t * input_len + 256 * blockIdx);

                // ======================================= up ===========================================
                {
                    const uint8_t *weights = ups + index * expert_size;
                    const uint8_t *pqs = weights;
                    const uint8_t *psl = pqs + num_blocks * 128;
                    uint32_t blk_index = h * outputPerGroup * num_blocks_per_row + blockIdx;
                    uint32_t qs_offset = blk_index * 128;
                    uint32_t sl_offset = blk_index * 16;

            #pragma unroll
                    for (int l = 0; l < (int)outputPerGroup; l++)
                    {
                        simd<uint8_t, 128> qsData = block_load<uint8_t, 128>(pqs + qs_offset);
                        simd<uint8_t, 16> temp = block_load<uint8_t, 16>(psl + sl_offset);

                        uint32_t slData = temp.bit_cast_view<uint32_t>()[0];
                        uint32_t mlData = temp.bit_cast_view<uint32_t>()[1];
                        uint16_t shData = temp.bit_cast_view<uint16_t>()[4];
                        uint16_t mhData = temp.bit_cast_view<uint16_t>()[5];
                        fp16 d_temp = temp.bit_cast_view<fp16>()[6];
                        fp16 dmin_temp = temp.bit_cast_view<fp16>()[7];
                        float d = (float)d_temp;
                        float dmin = (float)dmin_temp;

                        qs_offset += num_blocks_per_row * 128;
                        sl_offset += num_blocks_per_row * 16;

                        simd<uint32_t, 8> scalequant = slData & low_mask;
                        scalequant = scalequant >> low_shift;
                        simd<uint16_t, 8> scalequanthi = shData & high_mask;
                        scalequanthi = scalequanthi >> high_shift;
                        scalequanthi = scalequanthi << 4;
                        scalequant = scalequant + scalequanthi;
                        simd<float, 8> scaleData = scalequant * d;

                        simd<uint32_t, 8> minquant = mlData & low_mask;
                        minquant = minquant >> low_shift;
                        simd<uint16_t, 8> minquanthi = mhData & high_mask;
                        minquanthi = minquanthi >> high_shift;
                        minquanthi = minquanthi << 4;
                        minquant = minquant + minquanthi;
                        simd<float, 8> minData = minquant * dmin;

                        simd<float, 32> acc = 0.0;
            #pragma unroll
                        for (int j = 0; j < 8; j++)
                        {
                            simd<float, 32> weightData;
                            weightData.select<16, 1>(0) = qsData.select<16, 1>(16 * j) & 0x0f;
                            weightData.select<16, 1>(16) = (qsData.select<16, 1>(16 * j) >> 4) & 0x0f;
                            float s = scaleData[j];
                            float m = minData[j];
                            weightData = weightData * s - m;
                            acc += inputData.select<32, 1>(j * 32) * weightData;
                        }
                        outputDataUp[l] += sycl::ext::intel::esimd::detail::sum<float, float, 32>(acc);
                    }
                }

                // ======================================= gate ===========================================
                {
                    const uint8_t *weights = gates + index * expert_size;
                    const uint8_t *pqs = weights;
                    const uint8_t *psl = pqs + num_blocks * 128;
                    uint32_t blk_index = h * outputPerGroup * num_blocks_per_row + blockIdx;
                    uint32_t qs_offset = blk_index * 128;
                    uint32_t sl_offset = blk_index * 16;

            #pragma unroll
                    for (int l = 0; l < (int)outputPerGroup; l++)
                    {
                        simd<uint8_t, 128> qsData = block_load<uint8_t, 128>(pqs + qs_offset);
                        simd<uint8_t, 16> temp = block_load<uint8_t, 16>(psl + sl_offset);

                        uint32_t slData = temp.bit_cast_view<uint32_t>()[0];
                        uint32_t mlData = temp.bit_cast_view<uint32_t>()[1];
                        uint16_t shData = temp.bit_cast_view<uint16_t>()[4];
                        uint16_t mhData = temp.bit_cast_view<uint16_t>()[5];
                        fp16 d_temp = temp.bit_cast_view<fp16>()[6];
                        fp16 dmin_temp = temp.bit_cast_view<fp16>()[7];
                        float d = (float)d_temp;
                        float dmin = (float)dmin_temp;

                        qs_offset += num_blocks_per_row * 128;
                        sl_offset += num_blocks_per_row * 16;

                        simd<uint32_t, 8> scalequant = slData & low_mask;
                        scalequant = scalequant >> low_shift;
                        simd<uint16_t, 8> scalequanthi = shData & high_mask;
                        scalequanthi = scalequanthi >> high_shift;
                        scalequanthi = scalequanthi << 4;
                        scalequant = scalequant + scalequanthi;
                        simd<float, 8> scaleData = scalequant * d;

                        simd<uint32_t, 8> minquant = mlData & low_mask;
                        minquant = minquant >> low_shift;
                        simd<uint16_t, 8> minquanthi = mhData & high_mask;
                        minquanthi = minquanthi >> high_shift;
                        minquanthi = minquanthi << 4;
                        minquant = minquant + minquanthi;
                        simd<float, 8> minData = minquant * dmin;

                        simd<float, 32> acc = 0.0;
            #pragma unroll
                        for (int j = 0; j < 8; j++)
                        {
                            simd<float, 32> weightData;
                            weightData.select<16, 1>(0) = qsData.select<16, 1>(16 * j) & 0x0f;
                            weightData.select<16, 1>(16) = (qsData.select<16, 1>(16 * j) >> 4) & 0x0f;
                            float s = scaleData[j];
                            float m = minData[j];
                            weightData = weightData * s - m;
                            acc += inputData.select<32, 1>(j * 32) * weightData;
                        }
                        outputDataGate[l] += sycl::ext::intel::esimd::detail::sum<float, float, 32>(acc);
                    }
                }
            } // end chunk loop

            slm_block_store<float, outputPerGroup>(sizeof(float) * hh * outputPerGroup + slmOffsetUpResult, outputDataUp);
            slm_block_store<float, outputPerGroup>(sizeof(float) * hh * outputPerGroup + slmOffsetGateResult, outputDataGate);

            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> accOut = 0.0;
                for (int i = 0; i < (int)localThread; i++)
                    accOut += slm_block_load<float, outputPerGroup>(sizeof(float) * i * outputPerGroup + slmOffsetUpResult);
                slm_block_store<float, outputPerGroup>(slmOffsetUpResult, accOut);
            }
            if (hh == 1)
            {
                simd<float, outputPerGroup> accOut = 0.0;
                for (int i = 0; i < (int)localThread; i++)
                    accOut += slm_block_load<float, outputPerGroup>(sizeof(float) * i * outputPerGroup + slmOffsetGateResult);
                slm_block_store<float, outputPerGroup>(slmOffsetGateResult, accOut);
            }

            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> acc_up = slm_block_load<float, outputPerGroup>(slmOffsetUpResult);
                simd<float, outputPerGroup> acc_gate = slm_block_load<float, outputPerGroup>(slmOffsetGateResult);

                simd<float, outputPerGroup> temp = exp(-acc_gate);
                temp = temp + 1.0f;
                temp = 1.0f / temp;
                temp = temp * acc_gate;
                temp = temp * acc_up;
                block_store<OT, outputPerGroup>(outputs + t * output_num * output_len + v * output_len + h * outputPerGroup, temp);
            }

        });
    });
}


template <uint32_t OPG, uint32_t BS>
void runGroupUpAndGateVecQ4KL1_xve_typed(void *stream, const void *inputs, const uint8_t *ups, const uint8_t *gates, const uint32_t *indexes, void *outputs, uint32_t output_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int output_precision)
{
    if (input_precision == 0 && output_precision == 0)
        runGroupUpAndGateVecQ4KL1_xve_impl<OPG, BS>(stream, (const float *)inputs, ups, gates, indexes, (float *)outputs, output_num, token_len, input_len, output_len);
    else if (input_precision == 0 && output_precision == 1)
        runGroupUpAndGateVecQ4KL1_xve_impl<OPG, BS>(stream, (const float *)inputs, ups, gates, indexes, (fp16 *)outputs, output_num, token_len, input_len, output_len);
    else if (input_precision == 1 && output_precision == 0)
        runGroupUpAndGateVecQ4KL1_xve_impl<OPG, BS>(stream, (const fp16 *)inputs, ups, gates, indexes, (float *)outputs, output_num, token_len, input_len, output_len);
    else if (input_precision == 1 && output_precision == 1)
        runGroupUpAndGateVecQ4KL1_xve_impl<OPG, BS>(stream, (const fp16 *)inputs, ups, gates, indexes, (fp16 *)outputs, output_num, token_len, input_len, output_len);
}

void runGroupUpAndGateVecQ4KL1_xve(void *stream, const void *inputs, const void *ups, const void *gates, const void *indexes, void* outputs, uint32_t output_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int output_precision, ScratchBufferMgr *smgr)
{
    // Benchmark (output_num=8, input_len=2048, output_len=512):
    //   OPG=4/BS=1: 102.8 us, OPG=8/BS=1: 93.8 us, OPG=16/BS=1: 102.1 us, OPG=32/BS=1: 123.0 us
    //   BS=2,4 all slower than BS=1. OPG=8 BS=1 is optimal.
    runGroupUpAndGateVecQ4KL1_xve_typed<8, 1>(stream, inputs, (const uint8_t *)ups, (const uint8_t *)gates, (const uint32_t *)indexes, outputs, output_num, token_len, input_len, output_len, input_precision, output_precision);
}

static uint8_t *shared_buffer_q4k = nullptr;
static uint32_t shared_idx_q4k = 0;

template <typename IT, typename OT>
void runVariousGemvQ4KL1_xve_impl(void *stream, const IT *inputs, const uint8_t **weights_vec,
                                   OT **outputs_vec, uint32_t token_len, uint32_t input_len,
                                   uint32_t *output_lens, uint32_t output_count)
{
    if (shared_buffer_q4k == nullptr)
    {
        shared_buffer_q4k = (uint8_t *)malloc_shared(256 * 64, *(sycl::queue *)stream);
    }
    const void **weightsPtr  = (const void **)(shared_buffer_q4k + shared_idx_q4k * 256);
    void **outputsPtr        = (void **)(shared_buffer_q4k + shared_idx_q4k * 256 + 8 * sizeof(void *));
    uint32_t *outputLensPtr  = (uint32_t *)(shared_buffer_q4k + shared_idx_q4k * 256 + 8 * 2 * sizeof(void *));
    ++shared_idx_q4k;
    shared_idx_q4k = shared_idx_q4k % 64;

    uint32_t output_len_all = 0;
    for (uint32_t i = 0; i < output_count; i++)
    {
        weightsPtr[i]    = weights_vec[i];
        outputsPtr[i]    = outputs_vec[i];
        outputLensPtr[i] = output_lens[i];
        output_len_all  += output_lens[i];
    }

    const uint32_t outputPerGroup = 16;
    uint32_t localRange = input_len / 256;  // num_blocks_per_row, shared across all experts
    int groups = (output_len_all + outputPerGroup - 1) / outputPerGroup;

    sycl::nd_range<1> Range(groups * localRange, localRange);

    sycl::queue *q = (sycl::queue *)stream;
    q->submit([&](handler &cgh) {
        cgh.parallel_for(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            slm_init(sizeof(float) * 64 * outputPerGroup);

            simd<uint32_t, 8> low_mask({0x0000000f, 0x000000f0, 0x00000f00, 0x0000f000, 0x000f0000, 0x00f00000, 0x0f000000, 0xf0000000});
            simd<uint32_t, 8> low_shift({0, 4, 8, 12, 16, 20, 24, 28});
            simd<uint32_t, 8> high_mask({0x0003, 0x000c, 0x0030, 0x00c0, 0x0300, 0x0c00, 0x3000, 0xc000});
            simd<uint32_t, 8> high_shift({0, 2, 4, 6, 8, 10, 12, 14});

            uint32_t h  = ndi.get_group(0);
            uint32_t hh = ndi.get_local_id(0);

            // Route work-group h to the correct expert
            uint32_t outputOffset = h * outputPerGroup;
            uint32_t accOffset = 0;
            uint32_t idx = 0;
            for (; idx < 8; idx++)
            {
                accOffset += outputLensPtr[idx];
                if (outputOffset < accOffset)
                    break;
            }

            uint8_t  *weights    = (uint8_t *)weightsPtr[idx];
            void     *outputs    = outputsPtr[idx];
            uint32_t  output_len = outputLensPtr[idx];
            h = (outputOffset - (accOffset - output_len)) / outputPerGroup;

            uint32_t num_blocks = input_len * output_len / 256;
            const uint8_t *pqs = weights;
            const uint8_t *psl = pqs + num_blocks * 128;

            simd<float, 256> inputData = block_load<IT, 256>(inputs + 256 * hh);

            uint32_t blk_index = h * outputPerGroup * localRange + hh;
            uint32_t qs_offset = blk_index * 128;
            uint32_t sl_offset = blk_index * 16;

            simd<float, outputPerGroup> outputData = 0.0f;

#pragma unroll
            for (int l = 0; l < outputPerGroup; l++)
            {
                simd<uint8_t, 128> qsData = block_load<uint8_t, 128>(pqs + qs_offset);
                simd<uint8_t, 16> temp = block_load<uint8_t, 16>(psl + sl_offset);
                uint32_t slData = temp.bit_cast_view<uint32_t>()[0];
                uint32_t mlData = temp.bit_cast_view<uint32_t>()[1];
                uint16_t shData = temp.bit_cast_view<uint16_t>()[4];
                uint16_t mhData = temp.bit_cast_view<uint16_t>()[5];
                fp16 d_temp = temp.bit_cast_view<fp16>()[6];
                fp16 dmin_temp = temp.bit_cast_view<fp16>()[7];
                float d = (float)d_temp;
                float dmin = (float)dmin_temp;

                qs_offset += localRange * 128;
                sl_offset += localRange * 16;

                simd<uint32_t, 8> scalequant = slData & low_mask;
                scalequant = scalequant >> low_shift;
                simd<uint16_t, 8> scalequanthi = shData & high_mask;
                scalequanthi = scalequanthi >> high_shift;
                scalequanthi = scalequanthi << 4;
                scalequant = scalequant + scalequanthi;
                simd<float, 8> scaleData = scalequant * d;

                simd<uint32_t, 8> minquant = mlData & low_mask;
                minquant = minquant >> low_shift;
                simd<uint16_t, 8> minquanthi = mhData & high_mask;
                minquanthi = minquanthi >> high_shift;
                minquanthi = minquanthi << 4;
                minquant = minquant + minquanthi;
                simd<float, 8> minData = minquant * dmin;

                simd<float, 32> acc = 0.0f;

#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    simd<float, 32> weightData;
                    weightData.select<16, 1>(0) = qsData.select<16, 1>(16 * j) & 0x0f;
                    weightData.select<16, 1>(16) = (qsData.select<16, 1>(16 * j) >> 4) & 0x0f;
                    float s = scaleData[j];
                    float m = minData[j];
                    weightData = weightData * s - m;
                    acc += inputData.select<32, 1>(j * 32) * weightData;
                }

                outputData[l] = sycl::ext::intel::esimd::detail::sum<float, float, 32>(acc);
            }

            slm_block_store<float, outputPerGroup>(sizeof(float) * hh * outputPerGroup, outputData);

            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> accOut = 0.0f;
                for (int i = 0; i < (int)localRange; i++)
                {
                    accOut += slm_block_load<float, outputPerGroup>(sizeof(float) * i * outputPerGroup);
                }
                block_store<OT, outputPerGroup>((OT *)outputs + h * outputPerGroup, accOut);
            }
        });
    });
}

void runVariousGemmQ4KL1_xve(void *stream, const void *inputs, const void **weights, void **outputs,
                              uint32_t token_len, uint32_t input_len, uint32_t *output_lens,
                              uint32_t output_count, int input_precision, int output_precision,
                              ScratchBufferMgr *smgr)
{
    if (input_precision == 0 && output_precision == 0)
        runVariousGemvQ4KL1_xve_impl<float, float>(stream, (const float *)inputs, (const uint8_t **)weights, (float **)outputs, token_len, input_len, output_lens, output_count);
    else if (input_precision == 0 && output_precision == 1)
        runVariousGemvQ4KL1_xve_impl<float, fp16>(stream, (const float *)inputs, (const uint8_t **)weights, (fp16 **)outputs, token_len, input_len, output_lens, output_count);
    else if (input_precision == 1 && output_precision == 0)
        runVariousGemvQ4KL1_xve_impl<fp16, float>(stream, (const fp16 *)inputs, (const uint8_t **)weights, (float **)outputs, token_len, input_len, output_lens, output_count);
    else if (input_precision == 1 && output_precision == 1)
        runVariousGemvQ4KL1_xve_impl<fp16, fp16>(stream, (const fp16 *)inputs, (const uint8_t **)weights, (fp16 **)outputs, token_len, input_len, output_lens, output_count);
}

template <uint32_t OPG, typename IT, typename OT>
void runDownMergeQ4KL1_xve_impl(void *stream, const IT *inputs, const uint8_t *downs, const uint32_t *indexes, const float* efficients, OT* outputs, uint32_t input_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int is_acc)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;
    uint32_t expert_size = num_blocks * 144;  // Q4K: 128 (qs) + 16 (scales)

    constexpr uint32_t outputPerGroup = OPG;
    uint32_t groups = (output_len + outputPerGroup - 1) / outputPerGroup;
    uint32_t localThread = num_blocks_per_row;

    uint32_t aligned_input_num = (input_num + 15) / 16 * 16;

    sycl::range<2> GlobalRange(groups * localThread, token_len);
    sycl::range<2> LocalRange(localThread, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    sycl::queue *q = (sycl::queue *)stream;
    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            slm_init(sizeof(float) * 64 * outputPerGroup);

            simd<uint32_t, 8> low_mask({0x0000000f, 0x000000f0, 0x00000f00, 0x0000f000, 0x000f0000, 0x00f00000, 0x0f000000, 0xf0000000});
            simd<uint32_t, 8> low_shift({0, 4, 8, 12, 16, 20, 24, 28});
            simd<uint32_t, 8> high_mask({0x0003, 0x000c, 0x0030, 0x00c0, 0x0300, 0x0c00, 0x3000, 0xc000});
            simd<uint32_t, 8> high_shift({0, 2, 4, 6, 8, 10, 12, 14});

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);
            int t = ndi.get_group(1);

            simd<float, outputPerGroup> outputData = 0.0;

            for (int e = 0; e < (int)input_num; e++)
            {
                uint32_t index = indexes ? indexes[t * aligned_input_num + e] : (uint32_t)e;
                float efficient = efficients[t * aligned_input_num + e];

                const uint8_t *weights = downs + index * expert_size;
                const uint8_t *pqs = weights;
                const uint8_t *psl = pqs + num_blocks * 128;

                simd<float, 256> inputData = block_load<IT, 256>(inputs + t * input_num * input_len + e * input_len + 256 * hh);

                uint32_t blk_index = h * outputPerGroup * num_blocks_per_row + hh;
                uint32_t qs_offset = blk_index * 128;
                uint32_t sl_offset = blk_index * 16;

                simd<float, outputPerGroup> expertOutputData = 0.0;

    #pragma unroll
                for (int l = 0; l < (int)outputPerGroup; l++)
                {
                    simd<uint8_t, 128> qsData = block_load<uint8_t, 128>(pqs + qs_offset);
                    simd<uint8_t, 16> temp = block_load<uint8_t, 16>(psl + sl_offset);

                    uint32_t slData = temp.bit_cast_view<uint32_t>()[0];
                    uint32_t mlData = temp.bit_cast_view<uint32_t>()[1];
                    uint16_t shData = temp.bit_cast_view<uint16_t>()[4];
                    uint16_t mhData = temp.bit_cast_view<uint16_t>()[5];
                    fp16 d_temp = temp.bit_cast_view<fp16>()[6];
                    fp16 dmin_temp = temp.bit_cast_view<fp16>()[7];
                    float d = (float)d_temp;
                    float dmin = (float)dmin_temp;

                    qs_offset += num_blocks_per_row * 128;
                    sl_offset += num_blocks_per_row * 16;

                    simd<uint32_t, 8> scalequant = slData & low_mask;
                    scalequant = scalequant >> low_shift;
                    simd<uint16_t, 8> scalequanthi = shData & high_mask;
                    scalequanthi = scalequanthi >> high_shift;
                    scalequanthi = scalequanthi << 4;
                    scalequant = scalequant + scalequanthi;
                    simd<float, 8> scaleData = scalequant * d;

                    simd<uint32_t, 8> minquant = mlData & low_mask;
                    minquant = minquant >> low_shift;
                    simd<uint16_t, 8> minquanthi = mhData & high_mask;
                    minquanthi = minquanthi >> high_shift;
                    minquanthi = minquanthi << 4;
                    minquant = minquant + minquanthi;
                    simd<float, 8> minData = minquant * dmin;

                    simd<float, 32> acc = 0.0;

    #pragma unroll
                    for (int j = 0; j < 8; j++)
                    {
                        simd<float, 32> weightData;
                        weightData.select<16, 1>(0) = qsData.select<16, 1>(16 * j) & 0x0f;
                        weightData.select<16, 1>(16) = (qsData.select<16, 1>(16 * j) >> 4) & 0x0f;
                        float s = scaleData[j];
                        float m = minData[j];
                        weightData = weightData * s - m;

                        acc += inputData.select<32, 1>(j * 32) * weightData;
                    }

                    expertOutputData[l] = sycl::ext::intel::esimd::detail::sum<float, float, 32>(acc);
                }

                outputData += expertOutputData * efficient;
            }

            slm_block_store<float, outputPerGroup>(sizeof(float) * hh * outputPerGroup, outputData);

            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> accOut = 0.0;
                for (int i = 0; i < (int)localThread; i++)
                {
                    accOut += slm_block_load<float, outputPerGroup>(sizeof(float) * i * outputPerGroup);
                }

                if (is_acc)
                {
                    accOut += block_load<OT, outputPerGroup>(outputs + t * output_len + h * outputPerGroup);
                }
                block_store<OT, outputPerGroup>(outputs + t * output_len + h * outputPerGroup, accOut);
            }

        });
    });
}


template <uint32_t OPG>
void runDownMergeQ4KL1_xve_typed(void *stream, const void *inputs, const uint8_t *downs, const uint32_t *indexes, const float *efficients, void *outputs, uint32_t input_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int is_acc, int input_precision, int output_precision)
{
    if (input_precision == 0 && output_precision == 0)
        runDownMergeQ4KL1_xve_impl<OPG>(stream, (const float *)inputs, downs, indexes, efficients, (float *)outputs, input_num, token_len, input_len, output_len, is_acc);
    else if (input_precision == 0 && output_precision == 1)
        runDownMergeQ4KL1_xve_impl<OPG>(stream, (const float *)inputs, downs, indexes, efficients, (fp16 *)outputs, input_num, token_len, input_len, output_len, is_acc);
    else if (input_precision == 1 && output_precision == 0)
        runDownMergeQ4KL1_xve_impl<OPG>(stream, (const fp16 *)inputs, downs, indexes, efficients, (float *)outputs, input_num, token_len, input_len, output_len, is_acc);
    else if (input_precision == 1 && output_precision == 1)
        runDownMergeQ4KL1_xve_impl<OPG>(stream, (const fp16 *)inputs, downs, indexes, efficients, (fp16 *)outputs, input_num, token_len, input_len, output_len, is_acc);
}

void runDownMergeQ4KL1_xve(void *stream, const void *inputs, const void *downs, const void *indexes, const void* efficients, void* outputs, uint32_t input_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int is_acc, int input_precision, int output_precision, ScratchBufferMgr *smgr)
{
    // Benchmark (input_num=8, input_len=512, output_len=2048): OPG=4 expected optimal (consistent with Q5K)
    runDownMergeQ4KL1_xve_typed<4>(stream, inputs, (const uint8_t *)downs, (const uint32_t *)indexes, (const float *)efficients, outputs, input_num, token_len, input_len, output_len, is_acc, input_precision, output_precision);
}


static RegisterGpuGemv g(GGML_TYPE_Q4_K, "GEMM.Q4_K.ESIMD.L1", ALL_PLATFORMS, runGemvQ4KL1_xve);
static RegisterGpuVariousGemv vv(GGML_TYPE_Q4_K, "GEMM.Q4_K.ESIMD.L1", ALL_PLATFORMS, runVariousGemmQ4KL1_xve);
static RegisterGpuGroupUpAndGateVec gg(GGML_TYPE_Q4_K, "GEMM.Q4_K.ESIMD.L1", ALL_PLATFORMS, runGroupUpAndGateVecQ4KL1_xve);
static RegisterGpuDownMerge gd(GGML_TYPE_Q4_K, "GEMM.Q4_K.ESIMD.L1", ALL_PLATFORMS, runDownMergeQ4KL1_xve);
