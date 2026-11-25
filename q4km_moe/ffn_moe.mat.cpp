#include <stdint.h>
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <algorithm>
#include <vector>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp> 

#define NOMINMAX
#include <windows.h>
#include "scratch_buffer.h"
#include "ggml_types.h"
#include "gemm_manager.h"
#include "types.h"
#include "moe_router.h"

typedef sycl::half fp16;

// Dynamic loading of LGR functions from platform-specific DLLs
typedef void fpt_RunGemmLgr(void *stream, const void *inputs, const void *weights, const void *bias, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int weight_format, int bias_precision, int output_precision, ScratchBufferMgr *smgr);
typedef fpt_runmatupandgate* fpt_GetGpuMatUpAndGateFuncLgr(int format);

struct LgrMoeFuncs {
    fpt_RunGemmLgr *pRunGemmLgr = nullptr;
    fpt_GetGpuMatUpAndGateFuncLgr *pGetGpuMatUpAndGateFuncLgr = nullptr;
};

static LgrMoeFuncs& GetLgrMoeFuncs() {
    static LgrMoeFuncs funcs = [] {
        LgrMoeFuncs f;
        PLATFORM plt = GetCurrentPlatform();
        const char* dll = nullptr;
        switch (plt) {
            case PLATFORM_ARL_H: dll = "llama.opt.lgr.xe1_5.dll"; break;
            case PLATFORM_LNL: case PLATFORM_PTL: case PLATFORM_WCL: dll = "llama.opt.lgr.xe2.dll"; break;
            default: return f;
        }
        HMODULE h = LoadLibraryA(dll);
        if (!h) {
            printf("Error: Failed to load %s (error %lu)\n", dll, GetLastError());
            return f;
        }
        f.pRunGemmLgr = (fpt_RunGemmLgr *)GetProcAddress(h, "RunGemmLgr");
        if (!f.pRunGemmLgr)
            printf("Error: GetProcAddress failed for RunGemmLgr in %s (error %lu)\n", dll, GetLastError());
        f.pGetGpuMatUpAndGateFuncLgr = (fpt_GetGpuMatUpAndGateFuncLgr *)GetProcAddress(h, "GetGpuMatUpAndGateFuncLgr");
        if (!f.pGetGpuMatUpAndGateFuncLgr)
            printf("Error: GetProcAddress failed for GetGpuMatUpAndGateFuncLgr in %s (error %lu)\n", dll, GetLastError());
        return f;
    }();
    return funcs;
}

// Dispatch a Up Gate Silu fusion kernel
bool up_gate_silu_fusion(void *stream, const void *inputs, const void *up, const void *gate, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int format, ScratchBufferMgr *smgr)
{
    bool used_fused = false;
    if (token_len == 1)
    {
        fpt_rungroupupandgatevec *fpn = GemmManager::GetInstance().GetGpuGroupUpAndGateVecFunc(format, GetCurrentPlatform());
        if (fpn != nullptr)
        {
            fpn(stream, inputs, up, gate, nullptr, outputs, 1, token_len, input_len, output_len, 1, 1, smgr);
            used_fused = true;
        }
    }
    else if (token_len > 1 && token_len <= 16) {
        fpt_runslimupandgate *fpn = GemmManager::GetInstance().GetGpuSlimUpAndGateFunc(format, GetCurrentPlatform());
        if (fpn != nullptr) {
            fpn(stream, inputs, up, gate, outputs, token_len, input_len, output_len, 1, 1, smgr);
            used_fused = true;
        }
    }
    else if (token_len > 16) {
        fpt_runmatupandgate *fpn = GetLgrMoeFuncs().pGetGpuMatUpAndGateFuncLgr ? GetLgrMoeFuncs().pGetGpuMatUpAndGateFuncLgr(format) : nullptr;
        if (fpn != nullptr) {
            fpn(stream, inputs, up, gate, outputs, token_len, input_len, output_len, 1, 1, smgr);
            used_fused = true;
        }
        else{
            fpn = GemmManager::GetInstance().GetGpuMatUpAndGateFunc(format, GetCurrentPlatform());
            if (fpn != nullptr) {
                fpn(stream, inputs, up, gate, outputs, token_len, input_len, output_len, 1, 1, smgr);
                used_fused = true;
            }
        }
    }
    return used_fused;

}

// Dispatch a GPU GEMM (engine=1, fp16 in/out) using RunGemm.
static inline void mat_run_gpu_gemm(void *stream, const void *inputs, const void *weights,
                                    void *outputs, uint32_t token_len, uint32_t input_len,
                                    uint32_t output_len, int weight_format, ScratchBufferMgr *smgr)
{
    if (token_len > 1 && token_len <= 16)
    {
        fpt_rungemm *fpn = GemmManager::GetInstance().GetGpuSlimGemmFunc(weight_format, GetCurrentPlatform());
        if (fpn != nullptr)
        {
            fpn(stream, inputs, weights, nullptr, outputs, token_len, input_len, output_len, 1, -1, 1, smgr);
            return;
        }
    }
    // GEMM sub-functions use Lock/Unlock to manage scratch allocations correctly.
    if (token_len > 1)
    {
        fpt_rungemm *fpn = GemmManager::GetInstance().GetGpuGemmFunc(weight_format, GetCurrentPlatform());
        if (fpn == nullptr)
        {
            if (GetLgrMoeFuncs().pRunGemmLgr)
                GetLgrMoeFuncs().pRunGemmLgr(stream, inputs, weights, nullptr, outputs, token_len, input_len, output_len, 1, weight_format, -1, 1, smgr);
            return;
        }
        fpn(stream, inputs, weights, nullptr, outputs, token_len, input_len, output_len, 1, -1, 1, smgr);
    }
    else
    {
        fpt_rungemm *fpn = GemmManager::GetInstance().GetGpuGemvFunc(weight_format, GetCurrentPlatform());
        if (fpn == nullptr)
        {
            if (GetLgrMoeFuncs().pRunGemmLgr)
                GetLgrMoeFuncs().pRunGemmLgr(stream, inputs, weights, nullptr, outputs, token_len, input_len, output_len, 1, weight_format, -1, 1, smgr);
            return;
        }
        fpn(stream, inputs, weights, nullptr, outputs, token_len, input_len, output_len, 1, -1, 1, smgr);
    }

}

// ----------------------------------------------------------------
// RAII device buffer
// ----------------------------------------------------------------
struct MatDevBuf
{
    void        *ptr = nullptr;
    sycl::queue *q   = nullptr;
    MatDevBuf(size_t bytes, sycl::queue &queue)
        : q(&queue), ptr(sycl::malloc_device(bytes, queue)) {}
    ~MatDevBuf() { if (ptr && q) sycl::free(ptr, *q); }
    template<typename T> T* as() const { return reinterpret_cast<T*>(ptr); }
    operator bool() const { return ptr != nullptr; }
private:
    MatDevBuf(const MatDevBuf &) = delete;
    MatDevBuf &operator=(const MatDevBuf &) = delete;
};

// ----------------------------------------------------------------
// Weight byte stride per expert (rows x cols in the given format)
// ----------------------------------------------------------------
static size_t MatGetExpertStride(int format, uint32_t rows, uint32_t cols)
{
    uint64_t n = (uint64_t)rows * cols;
    switch (format) {
    case GGML_TYPE_Q4_0:    return (size_t)(n / 32  * 18);
    case GGML_TYPE_Q4_1:    return (size_t)(n / 32  * 20);
    case GGML_TYPE_Q8_0:    return (size_t)(n / 32  * 34);
    case GGML_TYPE_MXFP4:   return (size_t)(n / 32  * 17);
    case GGML_TYPE_Q2_K:    return (size_t)(n / 256 * 84);
    case GGML_TYPE_Q3_K:    return (size_t)(n / 256 * 110);
    case GGML_TYPE_Q4_K:    return (size_t)(n / 256 * 144);
    case GGML_TYPE_Q5_K:    return (size_t)(n / 256 * 176);
    case GGML_TYPE_Q6_K:    return (size_t)(n / 256 * 210);
    case GGML_TYPE_IQ1_S:   return (size_t)(n / 256 * 50);
    case GGML_TYPE_IQ2_XXS: return (size_t)(n / 256 * 66);
    case GGML_TYPE_IQ2_XS:  return (size_t)(n / 256 * 74);
    case GGML_TYPE_IQ2_S:   return (size_t)(n / 256 * 82);
    case GGML_TYPE_IQ3_XXS: return (size_t)(n / 256 * 98);
    case GGML_TYPE_IQ3_S:   return (size_t)(n / 256 * 110);
    case GGML_TYPE_IQ4_XS:  return (size_t)(n / 256 * 136);
    case GGML_TYPE_IQ4_NL:  return (size_t)(n / 32  * 18);
    default:                return 0;
    }
}

// ----------------------------------------------------------------
// GPU helper kernels (plain SYCL parallel_for, no ESIMD needed)
// ----------------------------------------------------------------
static void scatterInputs(sycl::queue& q, fp16 *inputs, uint32_t* scattered_offsets, fp16 *outputs, uint32_t token_len, uint32_t input_len, uint32_t selected_experts)
{
    // if (selected_experts != 8)
    // {
    //     printf("Warning! only supports 8 selected experts by now! %d is provided\n", selected_experts);
    // }
    int globalGroup = token_len;
    int localThread = (input_len + 127) / 128;
    sycl::range<1> GlobalRange(globalGroup * localThread);
    sycl::range<1> LocalRange(localThread);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    uint32_t aligned_selected_experts = (selected_experts + 15) / 16 * 16; // align to 16 for easier indexing in the kernel

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);

            simd<uint32_t, 16> indexes = block_load<uint32_t, 16>(scattered_offsets + h * aligned_selected_experts);
            uint32_t input_offset = h * input_len + hh * 128;
            simd<fp16, 128> data = block_load<fp16, 128>(inputs + input_offset);
        
            for (int j = 0; j < selected_experts; j ++)
            {
                uint32_t output_offset = indexes[j] * input_len + hh * 128;
                block_store<fp16, 128>(outputs + output_offset, data);
            }
        });
    });

}

template <typename OT>
static void mergeOutputs(sycl::queue& q, fp16 *inputs, uint32_t* scattered_offsets, float* weights, fp16 *shared_inputs, float *sh_scores, OT *outputs, uint32_t token_len, uint32_t input_len, uint32_t selected_experts, uint32_t shared_experts)
{
    int globalGroup = token_len;
    int localThread = (input_len + 127) / 128;
    sycl::range<1> GlobalRange(globalGroup * localThread);
    sycl::range<1> LocalRange(localThread);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    uint32_t aligned_selected_experts = (selected_experts + 15) / 16 * 16; // align to 16 for easier indexing in the kernel
    uint32_t sh_score_stride = (shared_experts + 15) / 16 * 16; // align to 16 for easier indexing in the kernel

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);

            simd<uint32_t, 16> indexes = block_load<uint32_t, 16>(scattered_offsets + h * aligned_selected_experts);
            simd<float, 16> w = block_load<float, 16>(weights + h * aligned_selected_experts);
            uint32_t output_offset = h * input_len + hh * 128;
            simd<float, 128> outputdata = 0.0;
        
        #pragma unroll
            for (int j = 0; j < selected_experts; j ++)
            {
                uint32_t input_offset = indexes[j] * input_len + hh * 128;
                simd<float, 128> data = block_load<fp16, 128>(inputs + input_offset);
                float wei = w[j];
                outputdata = outputdata + data * wei;
            }
            if (shared_experts > 0)
            {
                for (int k = 0; k < shared_experts; k++)
                {
                    uint32_t sh_input_offset = k * token_len * input_len + h * input_len + hh * 128;
                    simd<float, 128> sh_data = block_load<fp16, 128>(shared_inputs + sh_input_offset);
                    float sh_wei = sh_scores[h * sh_score_stride + k];
                    outputdata = outputdata + sh_data * sh_wei;
                }
            }
            block_store<OT, 128>(outputs + output_offset, outputdata);
        });
    });

}


// SiLU gate: out[i] = up[i] * gate[i] / (1 + exp(-gate[i]))
static void gpu_silu(sycl::queue *q, const fp16 *up, const fp16 *gate,
                     fp16 *out, size_t count)
{
    q->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            float g = gate[id];
            float u = up[id];
            out[id] = u * g / (1.0f + sycl::exp(-g));
        });
    });
}

// Gather + weighted merge:
//   out[t*L+f] = sum_e{ weights[t*ws+e] * scattered[to_off[t*SE+e]*L+f] }
// ws  = weights buffer stride per token (cands_per_token)
// SE  = selected_experts (stride for to_off)
static void gpu_gather(sycl::queue *q,
                       const float    *scattered,
                       const float    *weights,
                       const uint32_t *to_off,
                       float          *out,
                       uint32_t T, uint32_t L, uint32_t SE, uint32_t ws)
{
    q->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<2>(T, L), [=](sycl::id<2> id) {
            uint32_t t = id[0], f = id[1];
            float v = 0.0f;
            for (uint32_t e = 0; e < SE; e++)
                v += weights[t * ws + e] * scattered[(size_t)to_off[t * ws + e] * L + f];
            out[t * L + f] = v;
        });
    });
}

// Accumulate scaled: out[t*L+f] += scores[t * score_stride + score_idx] * add[t*L+f]
static void gpu_acc_scaled(sycl::queue *q,
                           const fp16 *add, const float *scores,
                           uint32_t score_stride, uint32_t score_idx,
                           float *out, uint32_t T, uint32_t L)
{
    q->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<2>(T, L), [=](sycl::id<2> id) {
            uint32_t t = id[0], f = id[1];
            out[t * L + f] += scores[t * score_stride + score_idx] * add[t * L + f];
        });
    });
}

static void Fp32ToFp16(sycl::queue *q, const float *input, fp16 *output, uint32_t count)
{
    int globalThread = (count + 127) / 128;
    sycl::nd_range<1> Range(globalThread, 1);

    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            int h = ndi.get_group(0);
            simd<float, 128> inputData = block_load<float, 128>(input + h * 128);
            simd<fp16, 128> outputData = inputData;
            block_store<fp16, 128>(output + h * 128, outputData);
        });
    });
    
}


// slm_init requires a compile-time constant, so LocalRange is a template parameter.
// MaxLocalRange = ceil(input_len / 128); dispatch table handles common input_len values.
template <uint32_t SharedExperts, uint32_t LocalRange>
static void runSharedRouterMatImpl(sycl::queue& q, fp16 *inputs, float *weights,
    float *outputs, uint32_t token_len, uint32_t input_len, uint32_t score_stride)
{
    int globalGroup = (token_len + 15) / 16;
    sycl::range<1> GlobalRange(globalGroup * LocalRange);
    sycl::range<1> LocalRange_r(LocalRange);
    sycl::nd_range<1> Range(GlobalRange, LocalRange_r);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            // slm_init argument must be a compile-time constant — LocalRange is a template param.
            slm_init(sizeof(float) * SharedExperts * 16 * LocalRange);

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);

            simd<fp16, 128> in;
            simd<float, 128 * SharedExperts> wei;
            simd<float, SharedExperts * 16> out = 0.0;

            uint32_t input_offset = hh * 128 + h * 16 * input_len;
            uint32_t weights_offset = hh * 128;

            #pragma unroll
            for (uint32_t j = 0; j < SharedExperts; j++) {
                wei.template select<128, 1>(j * 128) = block_load<float, 128>(weights + weights_offset);
                weights_offset += input_len;
            }

            #pragma unroll
            for (int t = 0; t < 16 && h * 16 + t < token_len; t++) {
                in = block_load<fp16, 128>(inputs + input_offset);
                input_offset += input_len;
                #pragma unroll
                for (uint32_t j = 0; j < SharedExperts; j++) {
                    simd<float, 32> temp = 0;
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        temp += in.template select<32, 1>(32 * k) * wei.template select<32, 1>(32 * k + 128 * j);
                    }
                    out[j * 16 + t] = sycl::ext::intel::esimd::detail::sum<float, float, 32>(temp);
                }
            }

            // Store each expert's 16 partial sums to SLM (16 floats = 64 bytes per call)
            #pragma unroll
            for (uint32_t i = 0; i < SharedExperts; i++) {
                slm_block_store<float, 16>(
                    sizeof(float) * (hh * SharedExperts * 16 + i * 16),
                    out.template select<16, 1>(i * 16));
            }
            barrier();

            if (hh < SharedExperts) {
                simd<float, 16> acc = 0.0;
                // LocalRange is a compile-time template param — loop is unrollable
                #pragma unroll
                for (uint32_t j = 0; j < LocalRange; j++) {
                    acc += slm_block_load<float, 16>(sizeof(float) * 16 * (j * SharedExperts + hh));
                }

                acc = -1.0 * acc;
                acc = sycl::ext::intel::esimd::pow<float, 16, float>(2.718f, acc);
                acc = acc + 1.0;
                acc = 1.0 / acc;

                simd<uint32_t, 16> output_offsets({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
                output_offsets = output_offsets + h * 16;
                output_offsets = output_offsets * score_stride + hh;
                output_offsets *= sizeof(float);

                scatter<float, 16>(outputs, output_offsets, acc);
            }
        });
    });
}

static void runSharedRouterMat(sycl::queue& q, fp16 *inputs, float *weights,
    float *outputs, uint32_t token_len, uint32_t input_len,
    uint32_t shared_experts, uint32_t score_stride)
{
    uint32_t localRange = (input_len + 127) / 128;
    // Dispatch on both shared_experts and localRange so slm_init gets a compile-time constant.
    // Add more cases here as needed for other input_len values.
    if (shared_experts == 1) {
        switch (localRange) {
            case  8: runSharedRouterMatImpl<1,  8>(q, inputs, weights, outputs, token_len, input_len, score_stride); return;
            case 16: runSharedRouterMatImpl<1, 16>(q, inputs, weights, outputs, token_len, input_len, score_stride); return;
            case 32: runSharedRouterMatImpl<1, 32>(q, inputs, weights, outputs, token_len, input_len, score_stride); return;
            case 56: runSharedRouterMatImpl<1, 56>(q, inputs, weights, outputs, token_len, input_len, score_stride); return;
            case 64: runSharedRouterMatImpl<1, 64>(q, inputs, weights, outputs, token_len, input_len, score_stride); return;
        }
    } else if (shared_experts == 2) {
        switch (localRange) {
            case  8: runSharedRouterMatImpl<2,  8>(q, inputs, weights, outputs, token_len, input_len, score_stride); return;
            case 16: runSharedRouterMatImpl<2, 16>(q, inputs, weights, outputs, token_len, input_len, score_stride); return;
            case 32: runSharedRouterMatImpl<2, 32>(q, inputs, weights, outputs, token_len, input_len, score_stride); return;
            case 56: runSharedRouterMatImpl<2, 56>(q, inputs, weights, outputs, token_len, input_len, score_stride); return;
            case 64: runSharedRouterMatImpl<2, 64>(q, inputs, weights, outputs, token_len, input_len, score_stride); return;
        }
    }
    assert(false && "runSharedRouterMat: unsupported shared_experts/input_len combination");
}

void RunFfnMoeFusion_mat(void *stream, const void *inputs,
    LlamaEsimdWeightsBuffer *router, LlamaEsimdWeightsBuffer *up,
    LlamaEsimdWeightsBuffer *gate, LlamaEsimdWeightsBuffer *down,
    LlamaEsimdWeightsBuffer *shared_router, LlamaEsimdWeightsBuffer *shared_up,
    LlamaEsimdWeightsBuffer *shared_gate, LlamaEsimdWeightsBuffer *shared_down,
    void *outputs, uint32_t total_experts, uint32_t selected_experts, uint32_t shared_experts,
    uint32_t token_len, uint32_t input_len, uint32_t hidden_len,
    unsigned input_precision, unsigned output_precision, ScratchBufferMgr *smgr)
{
    sycl::queue *q = (sycl::queue *)stream;

    if (selected_experts == 256) // for warmup, non-sense selected_experts set from llama.cpp
    {
        selected_experts = 8;
    }

    smgr->Lock();

    fp16 *inputs_fp16 = (fp16 *)inputs;
    if (input_precision == 0)
    {
        inputs_fp16 = (fp16 *)smgr->Allocate(q, token_len * input_len * sizeof(fp16));
        Fp32ToFp16(q, (const float *)inputs, (fp16 *)inputs_fp16, token_len * input_len);
    }

    // ================================================================
    // Step 1: Router – compute dot products for all tokens at once.
    //   out[j * total_experts + e] = dot(inputs[j], router_weights[e])
    //   Uses a plain parallel_for (no ESIMD) for correctness across all token counts.
    // ================================================================
    assert(router->format == GGML_TYPE_F32);
    fp16 *router_weights_fp16 = (fp16 *)smgr->Allocate(q, total_experts * input_len * sizeof(fp16));
    Fp32ToFp16(q, (const float *)router->data, router_weights_fp16, total_experts * input_len);

    float *router_outputs = (float *)smgr->Allocate(q, token_len * total_experts * sizeof(float));
    {
        dnnl::engine eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
        dnnl::stream s = dnnl::sycl_interop::make_stream(eng, *q);

        dnnl::memory::desc input_f16_md({ token_len, input_len }, dnnl::memory::data_type::f16, { input_len, 1 }); // M x K layout
        dnnl::memory::desc router_f16_md({ input_len, total_experts }, dnnl::memory::data_type::f16, { 1, input_len}); // M x K layout
        dnnl::memory::desc router_output_md({ token_len, total_experts }, dnnl::memory::data_type::f32, { total_experts, 1}); // M x K layout

        dnnl::memory input_fp16_mem = dnnl::memory(input_f16_md, eng, inputs_fp16);
        dnnl::memory router_fp16_mem = dnnl::memory(router_f16_md, eng, router_weights_fp16);
        dnnl::memory router_output_mem = dnnl::memory(router_output_md, eng, router_outputs);

        dnnl::primitive_attr attr;
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
        dnnl::matmul::primitive_desc router_matmul_pd = dnnl::matmul::primitive_desc(eng, input_f16_md, router_f16_md, router_output_md, attr);
        dnnl::matmul router_matmul_p(router_matmul_pd);

        router_matmul_p.execute(s, { {DNNL_ARG_SRC, input_fp16_mem}, {DNNL_ARG_WEIGHTS, router_fp16_mem}, {DNNL_ARG_DST, router_output_mem} });
    }

    // topKAndWeightsImpl writes CANDS_PER_THREAD = ceil16(selected_experts) entries per token.
    const uint32_t aligned_selected_experts = ((selected_experts + 15) / 16) * 16;

    // ================================================================
    // Step 2: For each token, select top-K experts and compute softmax weights
    // ================================================================
    uint32_t *indexes = (uint32_t *)smgr->Allocate(q, token_len * aligned_selected_experts * sizeof(uint32_t));
    float *weights = (float *)smgr->Allocate(q, token_len * aligned_selected_experts * sizeof(float));
    topKAndWeights(*q, router_outputs, indexes, weights, token_len, total_experts, selected_experts);
    q->wait();

    std::vector<uint32_t> indexes_cpu(token_len * aligned_selected_experts);
    q->memcpy(indexes_cpu.data(), indexes, token_len * aligned_selected_experts * sizeof(uint32_t)).wait();

    fp16 *share_results = nullptr;
    float *sh_scores = nullptr;
    // shared experts
    if (shared_experts > 0 && shared_router && shared_up && shared_gate && shared_down) {
        const size_t sh_up_str   = MatGetExpertStride(shared_up->format,   hidden_len, input_len);
        const size_t sh_gate_str = MatGetExpertStride(shared_gate->format, hidden_len, input_len);
        const size_t sh_down_str = MatGetExpertStride(shared_down->format, input_len,  hidden_len);

        // runRouter block_store<float,16> always writes 16 floats per group;
        // allocate ceil16(shared_experts) per token to match that granularity.
        const uint32_t sh_score_stride = ((shared_experts + 15u) / 16u) * 16u;
        sh_scores = (float *)smgr->Allocate(q, token_len * sh_score_stride * sizeof(float));
        runSharedRouterMat(*q, inputs_fp16, (float *)shared_router->data, sh_scores, token_len, input_len, shared_experts, sh_score_stride);
        fp16 *silu_result = (fp16 *)smgr->Allocate(q, token_len * hidden_len * sizeof(fp16));
        share_results = (fp16 *)smgr->Allocate(q, shared_experts * token_len * input_len * sizeof(fp16));

        for (uint32_t s = 0; s < shared_experts; s++) {
            const void *sh_up_w   = (const uint8_t *)shared_up->data   + s * sh_up_str;
            const void *sh_gate_w = (const uint8_t *)shared_gate->data + s * sh_gate_str;
            const void *sh_down_w = (const uint8_t *)shared_down->data + s * sh_down_str;

            auto fpn = GemmManager::GetInstance().GetGpuMatUpAndGateFunc(shared_up->format, GetCurrentPlatform());
            fpn(q, inputs_fp16, sh_up_w, sh_gate_w, silu_result, token_len, input_len, hidden_len, 1, 1, smgr);
            mat_run_gpu_gemm(q, silu_result, sh_down_w, share_results + s * token_len * input_len, token_len, hidden_len, input_len, shared_down->format, smgr);
        }

    }

    // ================================================================
    // Step 4: Build scatter plan on host (same logic as CPU path)
    //   from_tok[pos]              = which token owns scatter slot pos
    //   to_off[j * SE + k]         = scatter slot for token j's k-th expert
    // ================================================================
    const uint32_t scatter_total = token_len * selected_experts;
    std::vector<uint32_t> counts  (total_experts, 0);
    std::vector<uint32_t> starts  (total_experts, 0);
    std::vector<uint32_t> to_off  (token_len * aligned_selected_experts, 0);

    {
        uint32_t pos = 0;
        for (uint32_t t = 0; t < total_experts; t++) {
            starts[t] = pos;
            for (uint32_t j = 0; j < token_len; j++)
                for (uint32_t k = 0; k < selected_experts; k++)
                    if (indexes_cpu[j * aligned_selected_experts + k] == t) {
                        to_off[j * aligned_selected_experts + k] = pos;
                        ++pos;
                    }
        }
    }

    // ================================================================
    // Step5: Scatter
    // ================================================================
    uint32_t *to_offset = (uint32_t *)smgr->Allocate(q, token_len * aligned_selected_experts * sizeof(uint32_t));
    q->memcpy(to_offset, to_off.data(), token_len * aligned_selected_experts * sizeof(uint32_t)).wait();
    fp16 *scattered_inputs = (fp16 *)smgr->Allocate(q, token_len * selected_experts * input_len * sizeof(fp16));
    scatterInputs(*q, inputs_fp16, to_offset, scattered_inputs, token_len, input_len, selected_experts);

    for (uint32_t t = 0; t < total_experts - 1; t++)
        counts[t] = starts[t + 1] - starts[t];
    counts[total_experts - 1] = token_len * selected_experts - starts[total_experts - 1];
    
    uint32_t aligned_token_len = ((token_len + 15) / 16) * 16;
    MatDevBuf scattered_out_dev(scatter_total * input_len * sizeof(float),  *q);


    // ================================================================
    // Step 6: Per-expert up / gate / SiLU / down
    //   GEMMs are dispatched via mat_run_gpu_gemm which checks the main DLL's
    //   GemmManager first (DNNL/XVE types like Q8_0) then falls back to the lgr
    //   DLL (XMX types like Q4_K, Q5_K).
    // ================================================================
    const size_t up_stride   = MatGetExpertStride(up->format,   hidden_len, input_len);
    const size_t gate_stride = MatGetExpertStride(gate->format, hidden_len, input_len);
    const size_t down_stride = MatGetExpertStride(down->format, input_len,  hidden_len);

    // Temp buffers sized for the largest possible batch (token_len rows)
    fp16 *silu_output = (fp16 *)smgr->Allocate(q, aligned_token_len * hidden_len * sizeof(fp16));

    fp16 *scattered_output = (fp16 *)smgr->Allocate(q, token_len * selected_experts * input_len * sizeof(fp16));

    // The SYCL queue is in-order: every kernel submission executes strictly after
    // the one before it.  GEMM sub-functions use Lock/Unlock on the bump pointer
    // only – they do not cancel any in-flight GPU work.  Because the queue
    // is in-order, by the time any kernel starts executing on the GPU all previously
    // submitted kernels have already finished, so re-using the same scratch address
    // range across consecutive GEMMs is safe without explicit q->wait() calls.
    for (uint32_t t = 0; t < total_experts; t++) {
        if (counts[t] == 0) continue;

        fp16      *ip     = scattered_inputs  + (size_t)starts[t] * input_len;
        fp16      *op     = scattered_output + (size_t)starts[t] * input_len;
        const void *w_up   = (const uint8_t *)up->data   + t * up_stride;
        const void *w_gate = (const uint8_t *)gate->data + t * gate_stride;
        const void *w_down = (const uint8_t *)down->data + t * down_stride;

        bool used_fused = up_gate_silu_fusion(q, ip, w_up, w_gate, silu_output, counts[t], input_len, hidden_len, up->format, smgr);

        if (!used_fused) {
            MatDevBuf temp_up_dev  (token_len * hidden_len * sizeof(fp16), *q);
            MatDevBuf temp_gate_dev(token_len * hidden_len * sizeof(fp16), *q);
            mat_run_gpu_gemm(q, ip, w_up,   temp_up_dev.ptr,   counts[t], input_len, hidden_len, up->format,   smgr);
            mat_run_gpu_gemm(q, ip, w_gate, temp_gate_dev.ptr, counts[t], input_len, hidden_len, gate->format, smgr);

            gpu_silu(q, temp_up_dev.as<fp16>(), temp_gate_dev.as<fp16>(), silu_output, (size_t)counts[t] * hidden_len);

            q->wait(); // ensure computing finished before free temp_up_dev and temp_gate_dev
        }

        mat_run_gpu_gemm(q, silu_output, w_down, op,
                        counts[t], hidden_len, input_len, down->format, smgr);
    }

    if (output_precision == 0)
    {
        mergeOutputs(*q, scattered_output, to_offset, (float *)weights, share_results, sh_scores, (float *)outputs, token_len, input_len, selected_experts, shared_experts);
    }
    else
    {
        mergeOutputs(*q, scattered_output, to_offset, (float *)weights, share_results, sh_scores, (fp16 *)outputs, token_len, input_len, selected_experts, shared_experts);
    }

    smgr->Unlock(); // outputs is ready, can unlock now.
}

