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
void runGemvQ5KL1_xve_xe2_impl(void *stream, IT *inputs, uint8_t *weights, OT* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;
    uint8_t *pqs = weights;
    uint8_t *pqh = pqs + num_blocks * 128;
    uint8_t *psl = pqh + num_blocks * 32;

    const uint32_t outputPerGroup = 16;
    uint32_t groups = (output_len + outputPerGroup - 1) / outputPerGroup;
    uint32_t localRange = num_blocks_per_row;
    sycl::nd_range<1> Range(groups * localRange, localRange);

    sycl::queue *q = (sycl::queue *)stream;
    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            slm_init(sizeof(float) * 64 * outputPerGroup);

            simd<uint32_t, 32> high_value_mask({0x00000001, 0x00000100, 0x00010000, 0x01000000, 0x00000002, 0x00000200, 0x00020000, 0x02000000,
                                          0x00000004, 0x00000400, 0x00040000, 0x04000000, 0x00000008, 0x00000800, 0x00080000, 0x08000000,
                                          0x00000010, 0x00001000, 0x00100000, 0x10000000, 0x00000020, 0x00002000, 0x00200000, 0x20000000,
                                          0x00000040, 0x00004000, 0x00400000, 0x40000000, 0x00000080, 0x00008000, 0x00800000, 0x80000000});
            simd<uint32_t, 8> low_mask({0x0000000f, 0x000000f0, 0x00000f00, 0x0000f000, 0x000f0000, 0x00f00000, 0x0f000000, 0xf0000000});
            simd<uint32_t, 8> low_shift({0, 4, 8, 12, 16, 20, 24, 28});
            simd<uint32_t, 8> high_mask({0x0003, 0x000c, 0x0030, 0x00c0, 0x0300, 0x0c00, 0x3000, 0xc000});
            simd<uint32_t, 8> high_shift({0, 2, 4, 6, 8, 10, 12, 14});

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);

            simd<float, 256> inputData = block_load<IT, 256>(inputs + 256 * hh);
            uint32_t blk_index = h * outputPerGroup * num_blocks_per_row + hh;
            uint32_t qs_offset = blk_index * 128;
            uint32_t qh_offset = blk_index * 32;
            uint32_t sl_offset = blk_index * 16;

            simd<float, outputPerGroup> outputData = 0.0;

    #pragma unroll
            for (int l = 0; l < outputPerGroup; l++)
            {
                simd<uint8_t, 128> qsData = block_load<uint8_t, 128>(pqs + qs_offset);
                simd<uint8_t, 32> qhData = block_load<uint8_t, 32>(pqh + qh_offset);
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
                qh_offset += num_blocks_per_row * 32;
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
                    uint32_t high = qhData.bit_cast_view<uint32_t>()[j];
                    simd<float, 32> highvalue = 0.0;
                    highvalue.merge(16.0, (high & high_value_mask) > 0);
                    weightData += highvalue;
                    
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


void runGemvQ5KL1_xve_xe2(void *stream, const void *inputs, const void *weights, const void *bias, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int bias_precision, int output_precision, ScratchBufferMgr *smgr)
{
    assert(bias == nullptr);
    if (input_precision == 0 && output_precision == 0)
    {
        runGemvQ5KL1_xve_xe2_impl(stream, (float *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 0 && output_precision == 1)
    {
        runGemvQ5KL1_xve_xe2_impl(stream, (float *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 1 && output_precision == 0)
    {
        runGemvQ5KL1_xve_xe2_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 1 && output_precision == 1)
    {
        runGemvQ5KL1_xve_xe2_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len);
    }
}

static RegisterGpuGemv g(GGML_TYPE_Q5_K, "GEMM.Q5_K.ESIMD.L1", PFLAG(PLATFORM_LNL) | PFLAG(PLATFORM_PTL) | PFLAG(PLATFORM_WCL), runGemvQ5KL1_xve_xe2);

template <uint32_t OPG, typename IT, typename OT>
void runDownMergeQ5KL1_xve_xe2_impl(void *stream, const IT *inputs, const uint8_t *downs, const uint32_t *indexes, const float* efficients, OT* outputs, uint32_t input_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int is_acc)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;
    uint32_t expert_size = num_blocks * 176;  // Q5K: 128 (qs) + 32 (qh) + 16 (scales)

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

            simd<uint32_t, 32> high_value_mask({0x00000001, 0x00000100, 0x00010000, 0x01000000, 0x00000002, 0x00000200, 0x00020000, 0x02000000,
                                          0x00000004, 0x00000400, 0x00040000, 0x04000000, 0x00000008, 0x00000800, 0x00080000, 0x08000000,
                                          0x00000010, 0x00001000, 0x00100000, 0x10000000, 0x00000020, 0x00002000, 0x00200000, 0x20000000,
                                          0x00000040, 0x00004000, 0x00400000, 0x40000000, 0x00000080, 0x00008000, 0x00800000, 0x80000000});
            simd<uint32_t, 8> low_mask({0x0000000f, 0x000000f0, 0x00000f00, 0x0000f000, 0x000f0000, 0x00f00000, 0x0f000000, 0xf0000000});
            simd<uint32_t, 8> low_shift({0, 4, 8, 12, 16, 20, 24, 28});
            simd<uint32_t, 8> high_mask({0x0003, 0x000c, 0x0030, 0x00c0, 0x0300, 0x0c00, 0x3000, 0xc000});
            simd<uint32_t, 8> high_shift({0, 2, 4, 6, 8, 10, 12, 14});

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);
            int t = ndi.get_group(1);

            simd<float, outputPerGroup> outputData = 0.0;

            for (int e = 0; e < input_num; e++)
            {
                uint32_t index = indexes ? indexes[t * aligned_input_num + e] : (uint32_t)e;
                float efficient = efficients[t * aligned_input_num + e];

                const uint8_t *weights = downs + index * expert_size;
                const uint8_t *pqs = weights;
                const uint8_t *pqh = pqs + num_blocks * 128;
                const uint8_t *psl = pqh + num_blocks * 32;

                simd<float, 256> inputData = block_load<IT, 256>(inputs + t * input_num * input_len + e * input_len + 256 * hh);

                // Load all OPG channels' data at once via 2D tile loads
                // qs surface: width=128B per block, pitch=num_blocks_per_row*128, height=output_len
                // Split qs into 2×64B tiles (load_2d max width = 64B for uint8_t)
                uint32_t qs_pitch = num_blocks_per_row * 128;
                simd<uint8_t, 64 * outputPerGroup> qsTileA = load_2d<uint8_t, 64, outputPerGroup, 1, false, false>(
                    pqs, qs_pitch - 1, output_len - 1, qs_pitch - 1,
                    hh * 128, h * outputPerGroup);
                simd<uint8_t, 64 * outputPerGroup> qsTileB = load_2d<uint8_t, 64, outputPerGroup, 1, false, false>(
                    pqs, qs_pitch - 1, output_len - 1, qs_pitch - 1,
                    hh * 128 + 64, h * outputPerGroup);

                // qh surface: width=32B, pitch=num_blocks_per_row*32
                uint32_t qh_pitch = num_blocks_per_row * 32;
                simd<uint8_t, 32 * outputPerGroup> qhTile = load_2d<uint8_t, 32, outputPerGroup, 1, false, false>(
                    pqh, qh_pitch - 1, output_len - 1, qh_pitch - 1,
                    hh * 32, h * outputPerGroup);

                // sl surface: width=16B, pitch=num_blocks_per_row*16
                uint32_t sl_pitch = num_blocks_per_row * 16;
                simd<uint8_t, 16 * outputPerGroup> slTile = load_2d<uint8_t, 16, outputPerGroup, 1, false, false>(
                    psl, sl_pitch - 1, output_len - 1, sl_pitch - 1,
                    hh * 16, h * outputPerGroup);

                simd<float, outputPerGroup> expertOutputData = 0.0;

    #pragma unroll
                for (int l = 0; l < (int)outputPerGroup; l++)
                {
                    // Extract this channel's data from tiles
                    simd<uint8_t, 128> qsData;
                    qsData.template select<64, 1>(0) = qsTileA.template select<64, 1>(l * 64);
                    qsData.template select<64, 1>(64) = qsTileB.template select<64, 1>(l * 64);
                    simd<uint8_t, 32> qhData = qhTile.template select<32, 1>(l * 32);
                    simd<uint8_t, 16> temp = slTile.template select<16, 1>(l * 16);

                    uint32_t slData = temp.bit_cast_view<uint32_t>()[0];
                    uint32_t mlData = temp.bit_cast_view<uint32_t>()[1];
                    uint16_t shData = temp.bit_cast_view<uint16_t>()[4];
                    uint16_t mhData = temp.bit_cast_view<uint16_t>()[5];
                    fp16 d_temp = temp.bit_cast_view<fp16>()[6];
                    fp16 dmin_temp = temp.bit_cast_view<fp16>()[7];
                    float d = (float)d_temp;
                    float dmin = (float)dmin_temp;

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
                        uint32_t high = qhData.bit_cast_view<uint32_t>()[j];
                        simd<float, 32> highvalue = 0.0;
                        highvalue.merge(16.0, (high & high_value_mask) > 0);
                        weightData += highvalue;

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
                for (int i = 0; i < num_blocks_per_row; i++)
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
void runDownMergeQ5KL1_xve_xe2_typed(void *stream, const void *inputs, const uint8_t *downs, const uint32_t *indexes, const float *efficients, void *outputs, uint32_t input_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int is_acc, int input_precision, int output_precision)
{
    if (input_precision == 0 && output_precision == 0)
        runDownMergeQ5KL1_xve_xe2_impl<OPG>(stream, (const float *)inputs, downs, indexes, efficients, (float *)outputs, input_num, token_len, input_len, output_len, is_acc);
    else if (input_precision == 0 && output_precision == 1)
        runDownMergeQ5KL1_xve_xe2_impl<OPG>(stream, (const float *)inputs, downs, indexes, efficients, (fp16 *)outputs, input_num, token_len, input_len, output_len, is_acc);
    else if (input_precision == 1 && output_precision == 0)
        runDownMergeQ5KL1_xve_xe2_impl<OPG>(stream, (const fp16 *)inputs, downs, indexes, efficients, (float *)outputs, input_num, token_len, input_len, output_len, is_acc);
    else if (input_precision == 1 && output_precision == 1)
        runDownMergeQ5KL1_xve_xe2_impl<OPG>(stream, (const fp16 *)inputs, downs, indexes, efficients, (fp16 *)outputs, input_num, token_len, input_len, output_len, is_acc);
}

void runDownMergeQ5KL1_xve_xe2(void *stream, const void *inputs, const void *downs, const void *indexes, const void* efficients, void* outputs, uint32_t input_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int is_acc, int input_precision, int output_precision, ScratchBufferMgr *smgr)
{
    // Benchmark (input_num=8, input_len=512, output_len=2048):
    //   OPG=16: ~81 us, OPG=4: ~63.5 us, OPG=8: ~65 us
    runDownMergeQ5KL1_xve_xe2_typed<4>(stream, inputs, (const uint8_t *)downs, (const uint32_t *)indexes, (const float *)efficients, outputs, input_num, token_len, input_len, output_len, is_acc, input_precision, output_precision);
}


static RegisterGpuDownMerge gd(GGML_TYPE_Q5_K, "GEMM.Q5_K.ESIMD.L1", PFLAG(PLATFORM_LNL) | PFLAG(PLATFORM_PTL) | PFLAG(PLATFORM_WCL), runDownMergeQ5KL1_xve_xe2);
