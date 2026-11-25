#include <stdint.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <algorithm>

#include "scratch_buffer.h"
#include "ggml_types.h"
#include "gemm_manager.h"
#include "types.h"

typedef sycl::half fp16;

using namespace sycl;
using namespace sycl::ext::intel::esimd;

// outputs should be pre-allocated with expert count aligned to 16
template<typename IT, typename OT>
static void runRouter(sycl::queue* q, IT *inputs, float *weights, OT *outputs, uint32_t token_len, uint32_t input_len, uint32_t expert_num, int is_sigmod)
{
    uint32_t outputPerGroup = 16;
    if (expert_num < outputPerGroup)
    {
        outputPerGroup = expert_num;
    }

    int globalGroup = (expert_num + outputPerGroup - 1) / outputPerGroup;
    int localThread = input_len / 128;
    sycl::range<2> GlobalRange(token_len * localThread, globalGroup);
    sycl::range<2> LocalRange(localThread, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    q->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            slm_init(sizeof(float) * 16 * 64);

            int v = ndi.get_group(0);
            int h = ndi.get_group(1);
            int hh = ndi.get_local_id(0);
            
            simd<float, 128> in = block_load<IT, 128>((inputs + hh * 128) + v * input_len);
            simd<float, 128> wei;

            simd<float, 16> out = 0.0;

            uint32_t weights_offset = h * outputPerGroup * input_len + hh * 128;

            for (int j = 0; j < outputPerGroup; j++)
            {
                wei = block_load<float, 128>(weights + weights_offset);
                weights_offset += input_len;

                simd<float, 32> temp = 0;
        #pragma unroll
                for (int k = 0; k < 4; k++)
                {
                    temp += in.template select<32, 1>(32 * k) * wei.select<32, 1>(32 * k);
                }

                out[j] = sycl::ext::intel::esimd::detail::sum<float, float, 32>(temp);
            }
            slm_block_store<float, 16>(sizeof(float) * hh * 16, out);
            barrier();

            if (hh == 0)
            {
                simd<float, 16> acc = 0.0;
        #pragma unroll
                for (int j = 0; j < localThread; j++)
                {
                    acc += slm_block_load<float, 16>(sizeof(float) * j * 16);
                }

                if (is_sigmod)
                {
                    acc = exp(-acc);
                    acc = acc + 1.0;
                    acc = 1.0/acc;
                }

                // For expert_num >= 16, outputPerGroup == 16 and the 16-float block_store
                // lands cleanly between tokens (token stride == expert_num >= 16).
                // For expert_num < 16 (e.g., shared_router with shared_experts == 1),
                // token stride is expert_num so a 16-float store would clobber adjacent
                // tokens' slots with the garbage 0.5 values in acc[outputPerGroup..15].
                // Use a masked scatter that touches only the valid lanes.
                if (outputPerGroup == 16)
                {
                    block_store<float, 16>(outputs + h * 16 + v * expert_num, acc);
                }
                else
                {
                    simd<uint32_t, 16> store_off;
                    simd_mask<16> store_mask = 0;
                    for (uint32_t i = 0; i < 16; ++i)
                    {
                        store_off[i] = (v * expert_num + h * outputPerGroup + i) * sizeof(float);
                        if (i < outputPerGroup) store_mask[i] = 1;
                    }
                    scatter<float, 16, 1>(outputs, store_off, acc, store_mask);
                }
            }

        });
    });
}

// The weights is actually fp16 after the shuffle
template<typename IT, typename OT>
static void runRouterBf16(sycl::queue* q, IT *inputs, fp16 *weights, OT *outputs, uint32_t token_len, uint32_t input_len, uint32_t expert_num, int is_sigmod)
{
    uint32_t outputPerGroup = 16;
    if (expert_num < outputPerGroup)
    {
        outputPerGroup = expert_num;
    }

    int globalGroup = (expert_num + outputPerGroup - 1) / outputPerGroup;
    int localThread = input_len / 128;
    sycl::range<2> GlobalRange(token_len * localThread, globalGroup);
    sycl::range<2> LocalRange(localThread, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    q->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            slm_init(sizeof(float) * 16 * 64);

            int v = ndi.get_group(0);
            int h = ndi.get_group(1);
            int hh = ndi.get_local_id(0);
            
            simd<float, 128> in = block_load<IT, 128>((inputs + hh * 128) + v * input_len);
            simd<fp16, 128> wei;

            simd<float, 16> out = 0.0;

            uint32_t weights_offset = h * outputPerGroup * input_len + hh * 128;

            for (int j = 0; j < outputPerGroup; j++)
            {
                wei = block_load<fp16, 128>(weights + weights_offset);
                weights_offset += input_len;

                simd<float, 32> temp = 0;
        #pragma unroll
                for (int k = 0; k < 4; k++)
                {
                    temp += in.template select<32, 1>(32 * k) * wei.select<32, 1>(32 * k);
                }

                out[j] = sycl::ext::intel::esimd::detail::sum<float, float, 32>(temp);
            }
            slm_block_store<float, 16>(sizeof(float) * hh * 16, out);
            barrier();

            if (hh == 0)
            {
                simd<float, 16> acc = 0.0;
        #pragma unroll
                for (int j = 0; j < localThread; j++)
                {
                    acc += slm_block_load<float, 16>(sizeof(float) * j * 16);
                }

                acc = acc * 4.0; // in previous shuffle, weights are deviced by 4 to convert from bf16 to fp16

                if (is_sigmod)
                {
                    acc = exp(-acc);
                    acc = acc + 1.0;
                    acc = 1.0/acc;
                }

                // For expert_num >= 16, outputPerGroup == 16 and the 16-float block_store
                // lands cleanly between tokens (token stride == expert_num >= 16).
                // For expert_num < 16 (e.g., shared_router with shared_experts == 1),
                // token stride is expert_num so a 16-float store would clobber adjacent
                // tokens' slots with the garbage 0.5 values in acc[outputPerGroup..15].
                // Use a masked scatter that touches only the valid lanes.
                if (outputPerGroup == 16)
                {
                    block_store<float, 16>(outputs + h * 16 + v * expert_num, acc);
                }
                else
                {
                    simd<uint32_t, 16> store_off;
                    simd_mask<16> store_mask = 0;
                    for (uint32_t i = 0; i < 16; ++i)
                    {
                        store_off[i] = (v * expert_num + h * outputPerGroup + i) * sizeof(float);
                        if (i < outputPerGroup) store_mask[i] = 1;
                    }
                    scatter<float, 16, 1>(outputs, store_off, acc, store_mask);
                }
            }

        });
    });
}


#define FP32_MIN (-1.7e+38)

// topKAndWeightsImpl – ESIMD kernel template for top-K selection and softmax weight
// computation over MoE router logits.
//
// Template parameters
//   CHUNK_SIZE  : experts handled per work-item; must be a power-of-2 multiple of 8
//   NUM_THREADS : work-items per work-group; CHUNK_SIZE * NUM_THREADS == total_experts
//   SELECTED    : number of experts to select
//
// Algorithm
//   Phase 1 – each work-item finds its local top-SELECTED candidates and deposits them
//             into SLM together with per-chunk log-sum-exp numerics.
//   Phase 2 – all work-items (redundantly, avoiding extra synchronisation) reduce the
//             NUM_THREADS * CANDS_PER_THREAD SLM candidates to the global top-SELECTED,
//             compute renormalised softmax weights, and work-item 0 scatter-writes the
//             result.
//
// Output stride per token is CANDS_PER_THREAD = ceil16(SELECTED).
// Slots [SELECTED, CANDS_PER_THREAD) are masked off in the scatter so callers may
// allocate exactly selected_experts elements per token when token_len == 1.
template<uint32_t CHUNK_SIZE, uint32_t NUM_THREADS, uint32_t SELECTED>
static void topKAndWeightsImpl(sycl::queue& q, float* inputs, uint32_t* indexes, float* weights, uint32_t token_len)
{
    constexpr uint32_t TOTAL_EXPERTS    = CHUNK_SIZE * NUM_THREADS;
    constexpr uint32_t CANDS_PER_THREAD = ((SELECTED + 15u) / 16u) * 16u; // next multiple of 16
    constexpr uint32_t TOTAL_CANDS      = NUM_THREADS * CANDS_PER_THREAD;

    // SLM layout (byte offsets):
    //   [0,                        TOTAL_CANDS*4)             – candidate float values
    //   [TOTAL_CANDS*4,            TOTAL_CANDS*8)             – candidate int32 expert indexes
    //   [TOTAL_CANDS*8,            TOTAL_CANDS*8+NUM_THREADS*4) – per-thread softmax max
    //   [TOTAL_CANDS*8+NT*4,       TOTAL_CANDS*8+NUM_THREADS*8) – per-thread softmax sum
    constexpr uint32_t SLM_CAND_VALUES  = 0u;
    constexpr uint32_t SLM_CAND_INDEXES = TOTAL_CANDS * sizeof(float);
    constexpr uint32_t SLM_LOCAL_MAX    = SLM_CAND_INDEXES + TOTAL_CANDS * sizeof(int);
    constexpr uint32_t SLM_LOCAL_SUM    = SLM_LOCAL_MAX + NUM_THREADS * sizeof(float);
    constexpr uint32_t SLM_TOTAL_BYTES  = SLM_LOCAL_SUM + NUM_THREADS * sizeof(float);

    sycl::range<1> GlobalRange(token_len * NUM_THREADS);
    sycl::range<1> LocalRange(NUM_THREADS);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            slm_init(SLM_TOTAL_BYTES);

            const int h  = (int)ndi.get_group(0);
            const int hh = (int)ndi.get_local_id(0);

            // ------------------------------------------------------------------
            // Phase 1: per-thread top-SELECTED selection within CHUNK_SIZE experts
            // ------------------------------------------------------------------
            simd<float, CHUNK_SIZE> data = block_load<float, CHUNK_SIZE>(
                inputs + h * (int)TOTAL_EXPERTS + hh * (int)CHUNK_SIZE);

            simd<float,   CHUNK_SIZE> buf = data;
            simd<uint8_t, CHUNK_SIZE> p   = 0;
            simd_mask<CHUNK_SIZE>     m;
            int count = 0;

            while (count < (int)SELECTED)
            {
                float mx = hmax<float, float, CHUNK_SIZE>(buf);
                m = (buf > mx - 0.000001f);
                buf.merge(FP32_MIN, m);
                p.merge(1, m);
                count = sycl::ext::intel::esimd::detail::sum<uint8_t, uint8_t, CHUNK_SIZE>(p);
            }

            int count2 = 0;
            if (count > (int)SELECTED)
            {
                p.merge(2, m);
                count2 = sycl::ext::intel::esimd::detail::sum<int, uint16_t, CHUNK_SIZE>(m);
                count2 = count2 + (int)SELECTED - count;
            }

            simd<float, CANDS_PER_THREAD> cand_values  = FP32_MIN;
            simd<int,   CANDS_PER_THREAD> cand_indexes = -1;
            {
                int idx = 0, idx2 = 0;
                for (int i = 0; i < (int)CHUNK_SIZE && idx < (int)SELECTED; ++i)
                {
                    if (p[i] == 1)
                    {
                        cand_values[idx]  = data[i];
                        cand_indexes[idx] = i + hh * (int)CHUNK_SIZE;
                        ++idx;
                    }
                    else if (p[i] == 2 && idx2 < count2)
                    {
                        cand_values[idx]  = data[i];
                        cand_indexes[idx] = i + hh * (int)CHUNK_SIZE;
                        ++idx;
                        ++idx2;
                    }
                }
            }

            slm_block_store<float, CANDS_PER_THREAD>(SLM_CAND_VALUES  + hh * CANDS_PER_THREAD * sizeof(float), cand_values);
            slm_block_store<int,   CANDS_PER_THREAD>(SLM_CAND_INDEXES + hh * CANDS_PER_THREAD * sizeof(int),   cand_indexes);

            // Per-thread log-sum-exp numerics for numerically stable global softmax
            float local_max = hmax<float, float, CHUNK_SIZE>(data);
            simd<float, CHUNK_SIZE> shifted = data - local_max;
            simd<float, CHUNK_SIZE> exps    = exp(shifted);
            float local_sum = sycl::ext::intel::esimd::detail::sum<float, float, CHUNK_SIZE>(exps);

            slm_block_store<float, 1>(SLM_LOCAL_MAX + hh * sizeof(float), local_max);
            slm_block_store<float, 1>(SLM_LOCAL_SUM + hh * sizeof(float), local_sum);

            barrier();

            // ------------------------------------------------------------------
            // Phase 2: global top-SELECTED + renormalized softmax
            //          All threads run this redundantly; only thread 0 writes output.
            // ------------------------------------------------------------------
            {
                simd<float, NUM_THREADS> all_max = slm_block_load<float, NUM_THREADS>(SLM_LOCAL_MAX);
                simd<float, NUM_THREADS> all_sum = slm_block_load<float, NUM_THREADS>(SLM_LOCAL_SUM);

                float global_max = hmax<float, float, NUM_THREADS>(all_max);
                simd<float, NUM_THREADS> compensate = exp(all_max - global_max);
                float global_sum = sycl::ext::intel::esimd::detail::sum<float, float, NUM_THREADS>(all_sum * compensate);

                simd<float, TOTAL_CANDS> all_cval = slm_block_load<float, TOTAL_CANDS>(SLM_CAND_VALUES);
                simd<int,   TOTAL_CANDS> all_cidx = slm_block_load<int,   TOTAL_CANDS>(SLM_CAND_INDEXES);

                simd<float,   TOTAL_CANDS> buf2 = all_cval;
                simd<uint8_t, TOTAL_CANDS> p2   = 0;
                simd_mask<TOTAL_CANDS>     m2;
                int cnt = 0;

                while (cnt < (int)SELECTED)
                {
                    float mx = hmax<float, float, TOTAL_CANDS>(buf2);
                    m2 = (buf2 > mx - 0.000001f);
                    buf2.merge(FP32_MIN, m2);
                    p2.merge(1, m2);
                    cnt = sycl::ext::intel::esimd::detail::sum<uint8_t, uint8_t, TOTAL_CANDS>(p2);
                }

                int cnt2 = 0;
                if (cnt > (int)SELECTED)
                {
                    p2.merge(2, m2);
                    cnt2 = sycl::ext::intel::esimd::detail::sum<int, uint16_t, TOTAL_CANDS>(m2);
                    cnt2 = cnt2 + (int)SELECTED - cnt;
                }

                simd<uint32_t,    CANDS_PER_THREAD> out_indexes = 0u;
                simd<float,       CANDS_PER_THREAD> out_values  = FP32_MIN;
                {
                    int oi = 0, oi2 = 0;
                    for (int i = 0; i < (int)TOTAL_CANDS && oi < (int)SELECTED; ++i)
                    {
                        if (p2[i] == 1)
                        {
                            out_indexes[oi] = (uint32_t)all_cidx[i];
                            out_values[oi]  = all_cval[i];
                            ++oi;
                        }
                        else if (p2[i] == 2 && oi2 < cnt2)
                        {
                            out_indexes[oi] = (uint32_t)all_cidx[i];
                            out_values[oi]  = all_cval[i];
                            ++oi;
                            ++oi2;
                        }
                    }
                }

                // Scatter mask: valid for [0, SELECTED), zeroed for padding [SELECTED, CANDS_PER_THREAD)
                simd_mask<CANDS_PER_THREAD> scatter_mask = 1;
                for (uint32_t i = SELECTED; i < CANDS_PER_THREAD; ++i)
                    scatter_mask[i] = 0;

                // Renormalized softmax weights over the selected logits
                simd<float, CANDS_PER_THREAD> idata2   = out_values - global_max;
                simd<float, CANDS_PER_THREAD> edata    = exp(idata2);
                edata.merge(0.0f, scatter_mask == 0);   // zero padding slots before sum
                edata = edata / global_sum;
                float norm_sum = sycl::ext::intel::esimd::detail::sum<float, float, CANDS_PER_THREAD>(edata);
                simd<float, CANDS_PER_THREAD> out_weights = edata / norm_sum;
                out_weights.merge(0.0f, scatter_mask == 0);

                if (hh == 0)
                {
                    // Output stride = CANDS_PER_THREAD per token.
                    // Masked scatter ensures only the SELECTED valid positions are written.
                    simd<uint32_t, CANDS_PER_THREAD> scatter_off;
                    for (uint32_t i = 0; i < CANDS_PER_THREAD; ++i)
                        scatter_off[i] = ((uint32_t)h * CANDS_PER_THREAD + i) * sizeof(float);

                    scatter<float,    CANDS_PER_THREAD, 1>(weights, scatter_off, out_weights,  scatter_mask);
                    scatter<uint32_t, CANDS_PER_THREAD, 1>(indexes, scatter_off, out_indexes,  scatter_mask);
                }
            }
        });
    });
}

// Dispatch topKAndWeightsImpl to the right compile-time specialisation.
// CHUNK_SIZE is capped at 128 for register pressure; larger expert counts
// increase NUM_THREADS instead.  Add further rows to the table as needed.
static void topKAndWeights(sycl::queue& q, float* inputs, uint32_t* indexes, float* weights,
                           uint32_t token_len, uint32_t total_experts, uint32_t selected_experts)
{
#define MOE_DISPATCH_SEL(CHUNK, NT)                                                                     \
    switch (selected_experts) {                                                                          \
        case  1: topKAndWeightsImpl<CHUNK, NT,  1>(q, inputs, indexes, weights, token_len); return;     \
        case  2: topKAndWeightsImpl<CHUNK, NT,  2>(q, inputs, indexes, weights, token_len); return;     \
        case  4: topKAndWeightsImpl<CHUNK, NT,  4>(q, inputs, indexes, weights, token_len); return;     \
        case  6: topKAndWeightsImpl<CHUNK, NT,  6>(q, inputs, indexes, weights, token_len); return;     \
        case  8: topKAndWeightsImpl<CHUNK, NT,  8>(q, inputs, indexes, weights, token_len); return;     \
        case 10: topKAndWeightsImpl<CHUNK, NT, 10>(q, inputs, indexes, weights, token_len); return;     \
        case 12: topKAndWeightsImpl<CHUNK, NT, 12>(q, inputs, indexes, weights, token_len); return;     \
        case 16: topKAndWeightsImpl<CHUNK, NT, 16>(q, inputs, indexes, weights, token_len); return;     \
        default: break;                                                                                  \
    }

    switch (total_experts) {
        case   16: MOE_DISPATCH_SEL( 16, 1); break;
        case   32: MOE_DISPATCH_SEL( 32, 1); break;
        case   64: MOE_DISPATCH_SEL( 64, 1); break;
        case  128: MOE_DISPATCH_SEL(128, 1); break;
        case  256: MOE_DISPATCH_SEL(128, 2); break;
        case  512: MOE_DISPATCH_SEL(128, 4); break;
        case 1024: MOE_DISPATCH_SEL(128, 8); break;
        default: break;
    }

#undef MOE_DISPATCH_SEL

    assert(false && "topKAndWeights: unsupported total_experts/selected_experts combination");
}


static void MergeOutputs(sycl::queue *q, float* inputs, float* scales, float *outputs, uint32_t token_len, uint32_t output_len)
{
        sycl::range<2> GlobalRange(token_len, output_len / 128);
        sycl::range<2> LocalRange(1, 1);
        sycl::nd_range<2> Range(GlobalRange, LocalRange);
    
        q->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(Range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                int t = ndi.get_group(0);
                int h = ndi.get_group(1);

                float s = scales[t];
                simd<float, 128> inData = block_load<float, 128>(inputs + t * output_len + h * 128);
                simd<float, 128> outData = block_load<float, 128>(outputs + t * output_len + h * 128);

                outData = outData + inData * s;
                block_store<float, 128>(outputs + t * output_len + h * 128, outData);
            });
        });
}

void RunFfnMoeFusion_vec(void *stream, const void *inputs, LlamaEsimdWeightsBuffer *router, LlamaEsimdWeightsBuffer *up, LlamaEsimdWeightsBuffer *gate, LlamaEsimdWeightsBuffer *down, LlamaEsimdWeightsBuffer *shared_router, LlamaEsimdWeightsBuffer *shared_up, LlamaEsimdWeightsBuffer *shared_gate, LlamaEsimdWeightsBuffer *shared_down, void *outputs, uint32_t total_experts, uint32_t selected_experts, uint32_t shared_experts, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, unsigned input_precision, unsigned output_precision, ScratchBufferMgr *smgr)
{
    assert(token_len <= 16); // This function is only for the token_len <= 16 case; otherwise, use the mat version.
    assert(router->format == 0 || router->format == GGML_TYPE_BF16); // fp32 only

    if (selected_experts == 256) // for warmup, non-sense selected_experts set from llama.cpp
    {
        selected_experts = 8;
    }

    smgr->Lock();

    sycl::queue *q = (sycl::queue *)stream;

    float *router_output = (float *)smgr->Allocate((sycl::queue *)stream, token_len * total_experts * sizeof(float));
    if (input_precision == 0 && router->format == 0)
    {
        runRouter(q, (float *)inputs, (float *)router->data, router_output, token_len, input_len, total_experts, 0);
    }
    else if (input_precision == 1 && router->format == 0)
    {
        runRouter(q, (fp16 *)inputs, (float *)router->data, router_output, token_len, input_len, total_experts, 0);
    }
    else if (input_precision == 0 && router->format == GGML_TYPE_BF16)
    {
        runRouterBf16(q, (float *)inputs, (fp16 *)router->data, router_output, token_len, input_len, total_experts, 0);
    }
    else if (input_precision == 1 && router->format == GGML_TYPE_BF16)
    {
        runRouterBf16(q, (fp16 *)inputs, (fp16 *)router->data, router_output, token_len, input_len, total_experts, 0);
    }


    uint32_t aligned_selected_experts = ((selected_experts + 15) / 16) * 16;

    uint32_t *indexes = (uint32_t *)smgr->Allocate((sycl::queue *)stream, token_len * aligned_selected_experts * sizeof(uint32_t));
    float *weights = (float *)smgr->Allocate((sycl::queue *)stream, token_len * aligned_selected_experts * sizeof(float));

    topKAndWeights(*q, router_output, indexes, weights, token_len, total_experts, selected_experts);

    float *silu_result = (float *)smgr->Allocate((sycl::queue *)stream, token_len * selected_experts * hidden_len * sizeof(float));

    auto fp_UpAndGate = GemmManager::GetInstance().GetGpuGroupUpAndGateVecFunc(up->format, GetCurrentPlatform());
    auto fp_Down = GemmManager::GetInstance().GetGpuDownMergeFunc(down->format, GetCurrentPlatform());

    if (fp_UpAndGate == nullptr || fp_Down == nullptr)
    {
        fprintf(stderr, "Error: no GPU GroupUpAndGateVec or DownMerge for expert up format %d or down format %d - aborting, UpAndGate %p, Down %p\n", up->format, down->format, fp_UpAndGate, fp_Down);
        smgr->Unlock();
        return;
    }
    int input_element_size = (input_precision == 0) ? sizeof(float) : sizeof(fp16);
    int output_element_size = (output_precision == 0) ? sizeof(float) : sizeof(fp16);

    fp_UpAndGate(q, inputs, up->data, gate->data, indexes, silu_result, selected_experts, token_len, input_len, hidden_len, input_precision, 0, smgr);
    fp_Down(q, silu_result, down->data, indexes, weights, outputs, selected_experts, token_len, hidden_len, input_len, 0, 0, output_precision, smgr);
    // for (int t = 0; t < token_len; t++)
    // {
    //     fp_UpAndGate(q, (uint8_t *)inputs + t * input_len * input_element_size, up->data, gate->data, indexes + t * aligned_selected_experts, silu_result + t * selected_experts * hidden_len, selected_experts, 1, input_len, hidden_len, input_precision, 0, smgr);
    //     fp_Down(q, silu_result + t * selected_experts * hidden_len, down->data, indexes + t * aligned_selected_experts, weights + t * aligned_selected_experts, (uint8_t *)outputs + t * input_len * output_element_size, selected_experts, 1, hidden_len, input_len, 0, 0, output_precision, smgr);
    // }

    if (shared_experts == 0)
    {
        smgr->Unlock();
        return;
    }

    if (input_precision == 0 && shared_router->format == 0)
    {
        runRouter(q, (float *)inputs, (float *)shared_router->data, router_output, token_len, input_len, shared_experts, 1);
    }
    else if (input_precision == 1 && shared_router->format == 0)
    {
        runRouter(q, (fp16 *)inputs, (float *)shared_router->data, router_output, token_len, input_len, shared_experts, 1);
    }
    else if (input_precision == 0 && shared_router->format == GGML_TYPE_BF16)
    {
        runRouterBf16(q, (float *)inputs, (fp16 *)shared_router->data, router_output, token_len, input_len, shared_experts, 1);
    }
    else if (input_precision == 1 && shared_router->format == GGML_TYPE_BF16)
    {
        runRouterBf16(q, (fp16 *)inputs, (fp16 *)shared_router->data, router_output, token_len, input_len, shared_experts, 1);
    }

    if (shared_experts == 1 && token_len > 1)
    {
        auto fp_SlimUpAndGate = GemmManager::GetInstance().GetGpuSlimUpAndGateFunc(shared_up->format, GetCurrentPlatform());
        auto fp_SlimDown = GemmManager::GetInstance().GetGpuSlimGemmFunc(shared_down->format, GetCurrentPlatform());
        if (fp_SlimUpAndGate != nullptr && fp_SlimDown != nullptr)
        {
            float* shared_output = (float *)smgr->Allocate((sycl::queue *)stream, token_len * input_len * sizeof(float));
            fp_SlimUpAndGate(q, (uint8_t *)inputs, shared_up->data, shared_gate->data, silu_result, token_len, input_len, hidden_len, input_precision, 0, smgr);
            fp_SlimDown(q, silu_result, shared_down->data, nullptr, shared_output, token_len, hidden_len, input_len, 0, 0, output_precision, smgr);

            MergeOutputs(q, shared_output, router_output, (float *)outputs, token_len, input_len);
        }
        else
        {
            fprintf(stderr, "Warning: no GPU SlimUpAndGateVec or SlimDownMerge for shared expert format %d - skipping\n", shared_up->format);
        }
    }
    else
    {
        fp_UpAndGate = GemmManager::GetInstance().GetGpuGroupUpAndGateVecFunc(shared_up->format, GetCurrentPlatform());
        fp_Down = GemmManager::GetInstance().GetGpuDownMergeFunc(shared_down->format, GetCurrentPlatform());

        if (fp_UpAndGate != nullptr && fp_Down != nullptr)
        {
            for (int t = 0; t < token_len; t++)
            {
                fp_UpAndGate(q, (uint8_t *)inputs + t * input_len * input_element_size, shared_up->data, shared_gate->data, nullptr, silu_result + t * shared_experts * hidden_len, shared_experts, token_len, input_len, hidden_len, input_precision, 0, smgr);
                fp_Down(q, silu_result + t * shared_experts * hidden_len, shared_down->data, nullptr, router_output + t * shared_experts, (uint8_t *)outputs + t * input_len * output_element_size, shared_experts, token_len, hidden_len, input_len, 1, 0, output_precision, smgr);
            }

        }
        else
        {
            fprintf(stderr, "Warning: no GPU GroupUpAndGateVec or DownMerge for shared expert format %d - skipping\n", shared_up->format);
        }
    }

    smgr->Unlock();
}
