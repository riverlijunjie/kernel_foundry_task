#include <stdint.h>
#include <sycl/sycl.hpp>

#include "scratch_buffer.h"
#include "ggml_types.h"
#include "gemm_manager.h"
#include <cassert>

typedef sycl::half fp16;

typedef struct {
    uint8_t ql[256/2];      // quants, lower 4 bits
    uint8_t qh[256/4];      // quants, upper 2 bits
    int8_t  scales[256/16]; // scales, quantized with 8 bits
    fp16 d;             // super-block scale
} block_q6_K;

void ShuffleQ6KL1(const void *inputs, void *outputs, uint32_t input_len, uint32_t output_len)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint8_t *pql = (uint8_t *)outputs;
    uint8_t *pqh = pql + num_blocks * 128;
    int8_t *ps = (int8_t*)(pqh + num_blocks * 64);
    fp16 *pd = (fp16*)(ps + num_blocks * 16);

    block_q6_K *blocks = (block_q6_K *)inputs;

    uint8_t *raw_weights = new uint8_t[256];
    int8_t *raw_scales = new int8_t[16];
    
    for (int b = 0; b < num_blocks; b++)
    {
        for (int j = 0; j < 2; j ++)
        {
            for (int k = 0; k < 64; k++)
            {
                raw_weights[j * 128 + k] = blocks[b].ql[j * 64 + k] & 0x0F;
                raw_weights[j * 128 + 64 + k] = (blocks[b].ql[j * 64 + k] >> 4) & 0x0F;
            }

            for (int k = 0; k < 32; k++)
            {
                raw_weights[j * 128 + k] += ((blocks[b].qh[j * 32 + k] & 0x03) << 4);
                raw_weights[j * 128 + 32 + k] += ((blocks[b].qh[j * 32 + k] & 0x0c) << 2);
                raw_weights[j * 128 + 64 + k] += ((blocks[b].qh[j * 32 + k] & 0x30));
                raw_weights[j * 128 + 96 + k] += ((blocks[b].qh[j * 32 + k] & 0xc0) >> 2);
            }
        }

        for (int j = 0; j < 16; j++)
        {
            raw_scales[j] = blocks[b].scales[j];
        }

        uint8_t *opql = pql + b * 128;
        uint8_t *opqh = pqh + b * 64;
        int8_t *ops = ps + b * 16;
        fp16 *opd = pd + b;

        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 16; k ++)
            {
                opql[j * 16 + k] = (raw_weights[j * 32 + k] & 0x0F) | ((raw_weights[j * 32 + 16 + k] & 0x0F) << 4);
            }
        }
        for (int j = 0; j < 16; j ++)
        {
            for (int k = 0; k < 4; k ++)
            {
                opqh[j * 4 + k] = ((raw_weights[j * 16 + k] & 0x30) >> 4) | ((raw_weights[j * 16 + 4 + k] & 0x30) >> 2) | ((raw_weights[j * 16 + 8 + k] & 0x30)) | ((raw_weights[j * 16 + 12 + k] & 0x30) << 2);
            }
        }

        for (int k = 0; k < 16; k++)
        {
            ops[k] = raw_scales[k];
        }

        opd[0] = blocks[b].d;
    }

    delete[] raw_weights;
    delete[] raw_scales;
}

static void dequantQ6KL1(uint8_t *inputs, float *outputs, uint32_t input_len, uint32_t output_len)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint8_t *pql = inputs;
    uint8_t *pqh = pql + num_blocks * 128;
    int8_t *ps = (int8_t *)(pqh + num_blocks * 64);
    fp16 *pd = (fp16 *)(ps + num_blocks * 16);

    for (int b = 0; b < num_blocks; b++)
    {
        uint8_t *ipql = pql + b * 128;
        uint8_t *ipqh = pqh + b * 64;
        int8_t *ips = ps + b * 16;
        fp16* ipd = pd + b;
        float global_d = (float)(*ipd);
        float* poutputs = outputs + b * 256;


        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 16; k++)
            {
                uint8_t temp = (ipql[j * 16 + k] & 0x0F) | (((ipqh[j * 8 + k%4] >> ((k/4) * 2)) & 0x03) << 4);
                poutputs[j * 32 + k] = (temp - 32)  * global_d * ips[2 * j];
                temp = ((ipql[j * 16 + k] >> 4) & 0x0F) | (((ipqh[j * 8 + 4 + k%4] >> ((k/4) * 2)) & 0x03) << 4);
                poutputs[j * 32 + 16 + k] = (temp - 32) * global_d * ips[2 * j + 1];
            }
        }
    }
}

static void runGemm_cpu(float *inputs, float *weights, float* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len)
{
    for (int j = 0; j < token_len; j++)
    {
        for (int k = 0; k < output_len; k++)
        {
            float acc = 0.0;
            for (int i = 0; i < input_len; i++)
            {
                acc += inputs[j * input_len + i] * weights[k * input_len + i];
            }
            outputs[j * output_len + k] = acc;
        }
    }
}

static void dump(void* data, size_t size, const char *filename)
{
    FILE* fp = nullptr;
    fopen_s(&fp, filename, "wb");
    fwrite(data, 1, size, fp);
    fclose(fp);
}

template <typename IT, typename OT>
void runGemmQ6KL1_cpu_impl(void *stream, IT *inputs, uint8_t *weights, OT* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len)
{
    IT *inputs_cpu = new IT[token_len * input_len];
    OT *outputs_cpu = new OT[token_len * output_len];
    uint32_t weight_size = input_len * output_len / 256 * sizeof (block_q6_K);
    uint8_t *weights_cpu = new uint8_t[weight_size];
    float *weights_fp32 = new float[input_len * output_len];
    float *inputs_fp32 = new float[token_len * input_len];
    float *outputs_fp32 = new float[token_len * output_len];

    sycl::queue *q = (sycl::queue *)stream;
    q->wait();
    q->memcpy(inputs_cpu, inputs, token_len * input_len * sizeof(IT)).wait();
    q->memcpy(weights_cpu, weights, weight_size).wait();

    dequantQ6KL1(weights_cpu, weights_fp32, input_len, output_len);
    dump(weights_fp32, input_len * output_len * sizeof(float), "fp32_weights.bin");
    for (int j = 0; j < token_len * input_len; j++)
    {
        inputs_fp32[j] = (float)inputs_cpu[j];
    }

    runGemm_cpu(inputs_fp32, weights_fp32, outputs_fp32, token_len, input_len, output_len);
    for (int j = 0; j < token_len * output_len; j++)
    {
        outputs_cpu[j] = (OT)outputs_fp32[j];
    }

    q->memcpy(outputs, outputs_cpu, token_len * output_len * sizeof(OT)).wait();

    delete[] weights_fp32;
    delete[] inputs_fp32;
    delete[] outputs_fp32;
    delete[] inputs_cpu;
    delete[] outputs_cpu;
    delete[] weights_cpu;
}

void runGemmQ6KL1_cpu(void *stream, const void *inputs, const void *weights, const void *bias, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int bias_precision, int output_precision, ScratchBufferMgr *smgr)
{
    assert(bias == nullptr);
    if (input_precision == 0 && output_precision == 0)
    {
        runGemmQ6KL1_cpu_impl(stream, (float *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 0 && output_precision == 1)
    {
        runGemmQ6KL1_cpu_impl(stream, (float *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 1 && output_precision == 0)
    {
        runGemmQ6KL1_cpu_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 1 && output_precision == 1)
    {
        runGemmQ6KL1_cpu_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len);
    }
}

static RegisterCpuGemm g(GGML_TYPE_Q6_K, "GEMM.Q6_K.ESIMD.L1", ALL_PLATFORMS, runGemmQ6KL1_cpu);
static RegisterShuffle s(GGML_TYPE_Q6_K, "GEMM.Q6_K.ESIMD.L1", ALL_PLATFORMS, ShuffleQ6KL1);