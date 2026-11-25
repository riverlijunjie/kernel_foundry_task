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

#define FP32_MIN (-1.7e+38)
#define FP16_MIN (-65504.0f)

// OPG: outputs per work-group. BS: Q6K blocks per thread (each block = 256 inputs).
template <uint32_t OPG, uint32_t BS, typename IT, typename OT>
void runGemvQ6KL1_xve_impl(void *stream, IT *inputs, uint8_t *weights, OT* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;
    uint8_t *pql = weights;
    uint8_t *pqh = pql + num_blocks * 128;
    int8_t *ps = (int8_t*)(pqh + num_blocks * 64);
    fp16 *pd = (fp16*)(ps + num_blocks * 16);

    constexpr uint32_t outputPerGroup = OPG;
    constexpr uint32_t numChunks = BS;
    uint32_t groups = (output_len + outputPerGroup - 1) / outputPerGroup;
    uint32_t localRange = num_blocks_per_row / numChunks;
    sycl::nd_range<1> Range(groups * localRange, localRange);

    sycl::queue *q = (sycl::queue *)stream;

    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            slm_init(sizeof(float) * 64 * outputPerGroup);

            simd<uint32_t, 16> shift({0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30});

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);

            simd<float, outputPerGroup> outputData = 0.0;

            for (uint32_t c = 0; c < numChunks; c++)
            {
                uint32_t blockIdx = hh * numChunks + c;
                simd<float, 256> inputData = block_load<IT, 256>(inputs + 256 * blockIdx);
                uint32_t blk_index = h * outputPerGroup * num_blocks_per_row + blockIdx;

    #pragma unroll
                for (int l = 0; l < (int)outputPerGroup; l++)
                {
                    simd<uint8_t, 128> qlData = block_load<uint8_t, 128>(pql + blk_index * 128);
                    simd<uint8_t, 64> qhData = block_load<uint8_t, 64>(pqh + blk_index * 64);
                    simd<int8_t, 16> sData = block_load<int8_t, 16>(ps + blk_index * 16);
                    float d = (float)(pd[blk_index]);
                    blk_index += num_blocks_per_row;

                    simd<float, 32> acc = 0.0;

    #pragma unroll
                    for (int j = 0; j < 8; j++)
                    {
                        // Pre-compute combined scale: d * scale for each half
                        float combined_lo = d * (float)sData[2 * j];
                        float combined_hi = d * (float)sData[2 * j + 1];

                        simd<float, 32> weightData;
                        simd<uint8_t, 16> temp;

                        // Lower 16: ql low nibble + qh bits
                        uint32_t s = qhData.bit_cast_view<uint32_t>()[2 * j];
                        temp = s >> shift;
                        temp = temp & 0x03;
                        temp = temp << 4;
                        temp = (qlData.select<16, 1>(16 * j) & 0x0f) + temp;
                        weightData.select<16, 1>(0) = (temp - 32) * combined_lo;

                        // Upper 16: ql high nibble + qh bits
                        s = qhData.bit_cast_view<uint32_t>()[2 * j + 1];
                        temp = s >> shift;
                        temp = temp & 0x03;
                        temp = temp << 4;
                        temp = ((qlData.select<16, 1>(16 * j) >> 4) & 0x0f) + temp;
                        weightData.select<16, 1>(16) = (temp - 32) * combined_hi;

                        acc += inputData.select<32, 1>(j * 32) * weightData;
                    }

                    outputData[l] += sycl::ext::intel::esimd::detail::sum<float, float, 32>(acc);
                }
            } // end chunk loop

            slm_block_store<float, outputPerGroup>(sizeof(float) * hh * outputPerGroup, outputData);

            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> accOut = 0.0;
                for (int i = 0; i < (int)localRange; i++)
                {
                    accOut += slm_block_load<float, outputPerGroup>(sizeof(float) * i * outputPerGroup);
                }

                block_store<OT, outputPerGroup>(outputs + h * outputPerGroup, accOut);
            }

        });
    });

}

template <uint32_t OPG, uint32_t BS>
void runGemvQ6KL1_xve_typed(void *stream, const void *inputs, uint8_t *weights, void *outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int output_precision)
{
    if (input_precision == 0 && output_precision == 0)
        runGemvQ6KL1_xve_impl<OPG, BS>(stream, (float *)inputs, weights, (float *)outputs, token_len, input_len, output_len);
    else if (input_precision == 0 && output_precision == 1)
        runGemvQ6KL1_xve_impl<OPG, BS>(stream, (float *)inputs, weights, (fp16 *)outputs, token_len, input_len, output_len);
    else if (input_precision == 1 && output_precision == 0)
        runGemvQ6KL1_xve_impl<OPG, BS>(stream, (fp16 *)inputs, weights, (float *)outputs, token_len, input_len, output_len);
    else if (input_precision == 1 && output_precision == 1)
        runGemvQ6KL1_xve_impl<OPG, BS>(stream, (fp16 *)inputs, weights, (fp16 *)outputs, token_len, input_len, output_len);
}

void runGemvQ6KL1_xve(void *stream, const void *inputs, const void *weights, const void *bias, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int bias_precision, int output_precision, ScratchBufferMgr *smgr)
{
    assert(bias == nullptr);
    // Benchmark (input_len=2048, output_len=248320):
    //   OPG=4/BS=1: 3730.6, OPG=8/BS=1: 3689.7, OPG=16/BS=1: 3686.8, OPG=32/BS=1: 3679.7
    //   OPG=4/BS=2: 3689.0, OPG=16/BS=2: 3665.4, OPG=32/BS=2: 3702.7
    //   OPG=4/BS=4: 3672.6, OPG=32/BS=4: 3678.1
    //   load_2d tiles: slightly slower (~3730 us) due to descriptor overhead.
    //   Scale fusion applied (d*scale pre-computed). OPG=16 BS=2 is marginally best.
    runGemvQ6KL1_xve_typed<16, 2>(stream, inputs, (uint8_t *)weights, outputs, token_len, input_len, output_len, input_precision, output_precision);
}


// OPG: outputs per work-group. BS: Q6K blocks per thread (each block = 256 inputs).
// index_len: number of valid output positions to compute (length of `indexes`).
// output_len: full width of the output buffer (positions not in `indexes` are filled with -inf by the caller wrapper).
template <uint32_t OPG, uint32_t BS, typename IT, typename OT>
void runSparseGemvQ6KL1_xve_impl(void *stream, IT *inputs, uint8_t *weights, OT* outputs, uint32_t *indexes, uint32_t index_len, uint32_t token_len, uint32_t input_len, uint32_t output_len)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;
    uint8_t *pql = weights;
    uint8_t *pqh = pql + num_blocks * 128;
    int8_t *ps = (int8_t*)(pqh + num_blocks * 64);
    fp16 *pd = (fp16*)(ps + num_blocks * 16);

    constexpr uint32_t outputPerGroup = OPG;
    constexpr uint32_t numChunks = BS;
    uint32_t groups = (index_len + outputPerGroup - 1) / outputPerGroup;
    uint32_t localRange = num_blocks_per_row / numChunks;
    sycl::nd_range<1> Range(groups * localRange, localRange);

    sycl::queue *q = (sycl::queue *)stream;

    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            slm_init(sizeof(float) * 64 * outputPerGroup);

            simd<uint32_t, 16> shift({0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30});

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);

            simd<float, outputPerGroup> outputData = 0.0;

            simd<uint32_t, outputPerGroup> indexData = block_load<uint32_t, outputPerGroup>(indexes + h * outputPerGroup);

            for (uint32_t c = 0; c < numChunks; c++)
            {
                uint32_t blockIdx = hh * numChunks + c;
                simd<float, 256> inputData = block_load<IT, 256>(inputs + 256 * blockIdx);

    #pragma unroll
                for (int l = 0; l < (int)outputPerGroup; l++)
                {
                    uint32_t row_idx = indexData[l];
                    uint32_t blk_index = row_idx * num_blocks_per_row + blockIdx;
                    simd<uint8_t, 128> qlData = block_load<uint8_t, 128>(pql + blk_index * 128);
                    simd<uint8_t, 64> qhData = block_load<uint8_t, 64>(pqh + blk_index * 64);
                    simd<int8_t, 16> sData = block_load<int8_t, 16>(ps + blk_index * 16);
                    float d = (float)(pd[blk_index]);

                    simd<float, 32> acc = 0.0;

    #pragma unroll
                    for (int j = 0; j < 8; j++)
                    {
                        // Pre-compute combined scale: d * scale for each half
                        float combined_lo = d * (float)sData[2 * j];
                        float combined_hi = d * (float)sData[2 * j + 1];

                        simd<float, 32> weightData;
                        simd<uint8_t, 16> temp;

                        // Lower 16: ql low nibble + qh bits
                        uint32_t s = qhData.bit_cast_view<uint32_t>()[2 * j];
                        temp = s >> shift;
                        temp = temp & 0x03;
                        temp = temp << 4;
                        temp = (qlData.select<16, 1>(16 * j) & 0x0f) + temp;
                        weightData.select<16, 1>(0) = (temp - 32) * combined_lo;

                        // Upper 16: ql high nibble + qh bits
                        s = qhData.bit_cast_view<uint32_t>()[2 * j + 1];
                        temp = s >> shift;
                        temp = temp & 0x03;
                        temp = temp << 4;
                        temp = ((qlData.select<16, 1>(16 * j) >> 4) & 0x0f) + temp;
                        weightData.select<16, 1>(16) = (temp - 32) * combined_hi;

                        acc += inputData.select<32, 1>(j * 32) * weightData;
                    }

                    outputData[l] += sycl::ext::intel::esimd::detail::sum<float, float, 32>(acc);
                }
            } // end chunk loop

            slm_block_store<float, outputPerGroup>(sizeof(float) * hh * outputPerGroup, outputData);

            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> accOut = 0.0;
                for (int i = 0; i < (int)localRange; i++)
                {
                    accOut += slm_block_load<float, outputPerGroup>(sizeof(float) * i * outputPerGroup);
                }

                //block_store<OT, outputPerGroup>(outputs + h * outputPerGroup, accOut);
                scatter<OT, outputPerGroup>(outputs, indexData * sizeof(OT), accOut);
            }

        });
    });

}

template<typename OT>
void setOutputBuffer(void *stream, OT *outputs, uint32_t output_len, OT fill_value)
{
    sycl::queue *q = (sycl::queue *)stream;
    q->submit([&](handler& cgh) {
        cgh.parallel_for(sycl::range<1>(output_len), [=](sycl::id<1> idx) SYCL_ESIMD_KERNEL {
            outputs[idx] = fill_value;
        });
    });
}

template <uint32_t OPG, uint32_t BS>
void runSparseGemvQ6KL1_xve_typed(void *stream, const void *inputs, uint8_t *weights, void *outputs, uint32_t *indexes, uint32_t index_len, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int output_precision)
{
    if (input_precision == 0 && output_precision == 0)
        runSparseGemvQ6KL1_xve_impl<OPG, BS>(stream, (float *)inputs, weights, (float *)outputs, indexes, index_len, token_len, input_len, output_len);
    else if (input_precision == 0 && output_precision == 1)
        runSparseGemvQ6KL1_xve_impl<OPG, BS>(stream, (float *)inputs, weights, (fp16 *)outputs, indexes, index_len, token_len, input_len, output_len);
    else if (input_precision == 1 && output_precision == 0)
        runSparseGemvQ6KL1_xve_impl<OPG, BS>(stream, (fp16 *)inputs, weights, (float *)outputs, indexes, index_len, token_len, input_len, output_len);
    else if (input_precision == 1 && output_precision == 1)
        runSparseGemvQ6KL1_xve_impl<OPG, BS>(stream, (fp16 *)inputs, weights, (fp16 *)outputs, indexes, index_len, token_len, input_len, output_len);
}

void runSparseGemvQ6KL1_xve(void *stream, const void *inputs, const void *weights, const void *bias, void* outputs, uint32_t *indexes, uint32_t index_len, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int bias_precision, int output_precision, ScratchBufferMgr *smgr)
{
    assert(bias == nullptr);
    assert(index_len % 16 == 0);  // block_load<uint32_t, OPG=16> requires alignment
    assert(index_len <= output_len);

    // Fill the full output buffer with -inf so positions not in `indexes` get -inf.
    uint32_t total_outputs = token_len * output_len;
    if (output_precision == 0)
        setOutputBuffer<float>(stream, (float *)outputs, total_outputs, (float)FP32_MIN);
    else
        setOutputBuffer<fp16>(stream, (fp16 *)outputs, total_outputs, (fp16)FP16_MIN);

    runSparseGemvQ6KL1_xve_typed<16, 2>(stream, inputs, (uint8_t *)weights, outputs, indexes, index_len, token_len, input_len, output_len, input_precision, output_precision);
}


template <uint32_t OPG, typename IT, typename OT>
void runDownMergeQ6KL1_xve_impl(void *stream, const IT *inputs, const uint8_t *downs, const uint32_t *indexes, const float* efficients, OT* outputs, uint32_t input_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int is_acc)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint32_t num_blocks_per_row = input_len / 256;
    uint32_t expert_size = num_blocks * 210;  // Q6K: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)

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

            simd<uint32_t, 16> shift({0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30});

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);
            int t = ndi.get_group(1);

            simd<float, outputPerGroup> outputData = 0.0;

            for (int e = 0; e < (int)input_num; e++)
            {
                uint32_t index = indexes ? indexes[t * aligned_input_num + e] : (uint32_t)e;
                float efficient = efficients[t * aligned_input_num + e];

                const uint8_t *weights = downs + index * expert_size;
                const uint8_t *pql = weights;
                const uint8_t *pqh = pql + num_blocks * 128;
                const int8_t  *ps  = (const int8_t *)(pqh + num_blocks * 64);
                const fp16    *pd  = (const fp16 *)(ps + num_blocks * 16);

                simd<float, 256> inputData = block_load<IT, 256>(inputs + t * input_num * input_len + e * input_len + 256 * hh);

                uint32_t blk_index = h * outputPerGroup * num_blocks_per_row + hh;

                simd<float, outputPerGroup> expertOutputData = 0.0;

    #pragma unroll
                for (int l = 0; l < (int)outputPerGroup; l++)
                {
                    simd<uint8_t, 128> qlData = block_load<uint8_t, 128>(pql + blk_index * 128);
                    simd<uint8_t, 64>  qhData = block_load<uint8_t, 64>(pqh + blk_index * 64);
                    simd<int8_t, 16>   sData  = block_load<int8_t, 16>(ps + blk_index * 16);
                    float d = (float)(pd[blk_index]);
                    blk_index += num_blocks_per_row;

                    simd<float, 32> acc = 0.0;

    #pragma unroll
                    for (int j = 0; j < 8; j++)
                    {
                        float combined_lo = d * (float)sData[2 * j];
                        float combined_hi = d * (float)sData[2 * j + 1];

                        simd<float, 32> weightData;
                        simd<uint8_t, 16> temp;

                        uint32_t s = qhData.bit_cast_view<uint32_t>()[2 * j];
                        temp = s >> shift;
                        temp = temp & 0x03;
                        temp = temp << 4;
                        temp = (qlData.select<16, 1>(16 * j) & 0x0f) + temp;
                        weightData.select<16, 1>(0) = (temp - 32) * combined_lo;

                        s = qhData.bit_cast_view<uint32_t>()[2 * j + 1];
                        temp = s >> shift;
                        temp = temp & 0x03;
                        temp = temp << 4;
                        temp = ((qlData.select<16, 1>(16 * j) >> 4) & 0x0f) + temp;
                        weightData.select<16, 1>(16) = (temp - 32) * combined_hi;

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
void runDownMergeQ6KL1_xve_typed(void *stream, const void *inputs, const uint8_t *downs, const uint32_t *indexes, const float *efficients, void *outputs, uint32_t input_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int is_acc, int input_precision, int output_precision)
{
    if (input_precision == 0 && output_precision == 0)
        runDownMergeQ6KL1_xve_impl<OPG>(stream, (const float *)inputs, downs, indexes, efficients, (float *)outputs, input_num, token_len, input_len, output_len, is_acc);
    else if (input_precision == 0 && output_precision == 1)
        runDownMergeQ6KL1_xve_impl<OPG>(stream, (const float *)inputs, downs, indexes, efficients, (fp16 *)outputs, input_num, token_len, input_len, output_len, is_acc);
    else if (input_precision == 1 && output_precision == 0)
        runDownMergeQ6KL1_xve_impl<OPG>(stream, (const fp16 *)inputs, downs, indexes, efficients, (float *)outputs, input_num, token_len, input_len, output_len, is_acc);
    else if (input_precision == 1 && output_precision == 1)
        runDownMergeQ6KL1_xve_impl<OPG>(stream, (const fp16 *)inputs, downs, indexes, efficients, (fp16 *)outputs, input_num, token_len, input_len, output_len, is_acc);
}

void runDownMergeQ6KL1_xve(void *stream, const void *inputs, const void *downs, const void *indexes, const void* efficients, void* outputs, uint32_t input_num, uint32_t token_len, uint32_t input_len, uint32_t output_len, int is_acc, int input_precision, int output_precision, ScratchBufferMgr *smgr)
{
    runDownMergeQ6KL1_xve_typed<4>(stream, inputs, (const uint8_t *)downs, (const uint32_t *)indexes, (const float *)efficients, outputs, input_num, token_len, input_len, output_len, is_acc, input_precision, output_precision);
}



static RegisterGpuGemv g(GGML_TYPE_Q6_K, "GEMM.Q6_K.ESIMD.L1", ALL_PLATFORMS, runGemvQ6KL1_xve);
static RegisterGpuSparseGemv gs(GGML_TYPE_Q6_K, "GEMM.Q6_K.ESIMD.L1", ALL_PLATFORMS, runSparseGemvQ6KL1_xve);
static RegisterGpuDownMerge gd(GGML_TYPE_Q6_K, "GEMM.Q6_K.ESIMD.L1", ALL_PLATFORMS, runDownMergeQ6KL1_xve);
