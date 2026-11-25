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

// Slim GEMM for small token batches (token_len <= 16).
// Same workgroup layout as GEMV, but each thread processes all tokens,
// sharing dequantized weight data across tokens to reduce memory bandwidth.
template <typename IT, typename OT>
void runGemmQ5KL1_slim_xve_impl(void *stream, IT *inputs, uint8_t *weights, OT* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;
    uint8_t *pqs = weights;
    uint8_t *pqh = pqs + num_blocks * 128;
    uint8_t *psl = pqh + num_blocks * 32;

    const uint32_t outputPerGroup = 16;
    uint32_t groups = (output_len + outputPerGroup - 1) / outputPerGroup;
    uint32_t localRange = num_blocks_per_row;

    // Same workgroup structure as GEMV — one workgroup per output group
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

            // Process each token sequentially, reusing weight data
            for (uint32_t v = 0; v < token_len; v++)
            {
                simd<float, 256> inputData = block_load<IT, 256>(inputs + v * input_len + 256 * hh);
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
                    for (uint32_t i = 0; i < num_blocks_per_row; i++)
                    {
                        accOut += slm_block_load<float, outputPerGroup>(sizeof(float) * i * outputPerGroup);
                    }

                    block_store<OT, outputPerGroup>(outputs + v * output_len + h * outputPerGroup, accOut);
                }

                barrier();
            }
        });
    });
}


void runGemmQ5KL1_slim_xve(void *stream, const void *inputs, const void *weights, const void *bias, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int bias_precision, int output_precision, ScratchBufferMgr *smgr)
{
    assert(bias == nullptr);
    if (input_precision == 0 && output_precision == 0)
    {
        runGemmQ5KL1_slim_xve_impl(stream, (float *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 0 && output_precision == 1)
    {
        runGemmQ5KL1_slim_xve_impl(stream, (float *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 1 && output_precision == 0)
    {
        runGemmQ5KL1_slim_xve_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 1 && output_precision == 1)
    {
        runGemmQ5KL1_slim_xve_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len);
    }
}

static RegisterGpuSlimGemm g(GGML_TYPE_Q5_K, "GEMM.Q5_K.ESIMD.L1", ALL_PLATFORMS, runGemmQ5KL1_slim_xve);  // superseded by XMX slim
