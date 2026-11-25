#include <stdint.h>
#include <sycl/sycl.hpp>

#include "scratch_buffer.h"
#include "ggml_types.h"
#include "gemm_manager.h"
#include <cassert>

typedef sycl::half fp16;

typedef struct {
    fp16 d;    // super-block scale for quantized scales
    fp16 dmin; // super-block scale for quantized mins
    uint8_t scales[12]; // scales and mins, quantized with 6 bits
    uint8_t qh[32];           // quants, high bit
    uint8_t qs[128];           // quants, low 4 bits
} block_q5_K;

void ShuffleQ5KL1(const void *inputs, void *outputs, uint32_t input_len, uint32_t output_len)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint8_t *pqs = (uint8_t *)outputs;
    uint8_t *pqh = pqs + num_blocks * 128;
    uint8_t *psl = pqh + num_blocks * 32;

    block_q5_K *blocks = (block_q5_K *)inputs;

    uint8_t *raw_weights = new uint8_t[256];
    uint8_t *raw_scales = new uint8_t[8];
    uint8_t *raw_mins = new uint8_t[8];
    
    for (int b = 0; b < num_blocks; b++)
    {
        for (int j = 0; j < 4; j ++)
        {
            for (int k = 0; k < 32; k++)
            {
                raw_weights[j * 64 + k] = blocks[b].qs[j * 32 + k] & 0x0F;
                raw_weights[j * 64 + 32 + k] = (blocks[b].qs[j * 32 + k] >> 4) & 0x0F;
            }
        }

        for (int s = 0; s < 8; s ++)
        {
            for (int j = 0; j < 32; j++)
            {
                raw_weights[s * 32 + j] += ((blocks[b].qh[j] & (1 << s)) ? 16 : 0);
            }
        }

        raw_scales[0] = blocks[b].scales[0] & 0x3F;
        raw_scales[1] = blocks[b].scales[1] & 0x3F;
        raw_scales[2] = blocks[b].scales[2] & 0x3F;
        raw_scales[3] = blocks[b].scales[3] & 0x3F;
        raw_scales[4] = (blocks[b].scales[8] & 0x0F) | ((blocks[b].scales[0] >> 2) & 0x30);
        raw_scales[5] = (blocks[b].scales[9] & 0x0F) | ((blocks[b].scales[1] >> 2) & 0x30);
        raw_scales[6] = (blocks[b].scales[10] & 0x0F) | ((blocks[b].scales[2] >> 2) & 0x30);
        raw_scales[7] = (blocks[b].scales[11] & 0x0F) | ((blocks[b].scales[3] >> 2) & 0x30);

        raw_mins[0] = blocks[b].scales[4] & 0x3F;
        raw_mins[1] = blocks[b].scales[5] & 0x3F;
        raw_mins[2] = blocks[b].scales[6] & 0x3F;
        raw_mins[3] = blocks[b].scales[7] & 0x3F;
        raw_mins[4] = ((blocks[b].scales[8] >> 4) & 0x0F) | ((blocks[b].scales[4] >> 2) & 0x30);
        raw_mins[5] = ((blocks[b].scales[9] >> 4) & 0x0F) | ((blocks[b].scales[5] >> 2) & 0x30);
        raw_mins[6] = ((blocks[b].scales[10] >> 4) & 0x0F) | ((blocks[b].scales[6] >> 2) & 0x30);
        raw_mins[7] = ((blocks[b].scales[11] >> 4) & 0x0F) | ((blocks[b].scales[7] >> 2) & 0x30);

        uint8_t *opqs = pqs + b * 128;
        uint8_t *opqh = pqh + b * 32;
        uint8_t *opsl = psl + b * 16;
        uint8_t *opml = psl + b * 16 + 4;
        uint8_t *opsh = psl + b * 16 + 8;
        uint8_t *opmh = psl + b * 16 + 10;
        fp16 *opd = (fp16 *)(psl + b * 16 + 12);
        fp16 *opdmin = (fp16 *)(psl + b * 16 + 14);

        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 16; k ++)
            {
                opqs[j * 16 + k] = (raw_weights[j * 32 + k] & 0x0F) | ((raw_weights[j * 32 + 16 + k] & 0x0F) << 4);
            }
        }

        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 4; k ++)
            {
                opqh[j * 4 + k] = ((raw_weights[j * 32 + k] & 0x10) >> 4)
                                | ((raw_weights[j * 32 + k + 4] & 0x10) >> 3)
                                | ((raw_weights[j * 32 + k + 8] & 0x10) >> 2)
                                | ((raw_weights[j * 32 + k + 12] & 0x10) >> 1)
                                | ((raw_weights[j * 32 + k + 16] & 0x10) >> 0)
                                | ((raw_weights[j * 32 + k + 20] & 0x10) << 1)
                                | ((raw_weights[j * 32 + k + 24] & 0x10) << 2)
                                | ((raw_weights[j * 32 + k + 28] & 0x10) << 3);
            }
        }

        opsl[0] = (raw_scales[0] & 0x0F) | ((raw_scales[1] & 0x0F) << 4);
        opsl[1] = (raw_scales[2] & 0x0F) | ((raw_scales[3] & 0x0F) << 4);
        opsl[2] = (raw_scales[4] & 0x0F) | ((raw_scales[5] & 0x0F) << 4);
        opsl[3] = (raw_scales[6] & 0x0F) | ((raw_scales[7] & 0x0F) << 4);

        opsh[0] = ((raw_scales[0] & 0x30) >> 4) | ((raw_scales[1] & 0x30) >> 2) | (raw_scales[2] & 0x30) | ((raw_scales[3] & 0x30) << 2);
        opsh[1] = ((raw_scales[4] & 0x30) >> 4) | ((raw_scales[5] & 0x30) >> 2) | (raw_scales[6] & 0x30) | ((raw_scales[7] & 0x30) << 2);

        opml[0] = (raw_mins[0] & 0x0F) | ((raw_mins[1] & 0x0F) << 4);
        opml[1] = (raw_mins[2] & 0x0F) | ((raw_mins[3] & 0x0F) << 4);
        opml[2] = (raw_mins[4] & 0x0F) | ((raw_mins[5] & 0x0F) << 4);
        opml[3] = (raw_mins[6] & 0x0F) | ((raw_mins[7] & 0x0F) << 4);

        opmh[0] = ((raw_mins[0] & 0x30) >> 4) | ((raw_mins[1] & 0x30) >> 2) | (raw_mins[2] & 0x30) | ((raw_mins[3] & 0x30) << 2);
        opmh[1] = ((raw_mins[4] & 0x30) >> 4) | ((raw_mins[5] & 0x30) >> 2) | (raw_mins[6] & 0x30) | ((raw_mins[7] & 0x30) << 2);

        opd[0] = blocks[b].d;
        opdmin[0] = blocks[b].dmin;
    }

    delete[] raw_weights;
    delete[] raw_scales;
    delete[] raw_mins;
}

static void dequantQ5KL1(uint8_t *inputs, float *outputs, uint32_t input_len, uint32_t output_len)
{
    uint32_t num_blocks = input_len * output_len / 256;
    uint8_t *pqs = inputs;
    uint8_t *pqh = pqs + num_blocks * 128;
    uint8_t *psl = pqh + num_blocks * 32;

    float *scales = new float[8];
    float *mins = new float[8];

    uint8_t *quants = new uint8_t[256];

    for (int b = 0; b < num_blocks; b++)
    {
        uint8_t *ipqs = pqs + b * 128;
        uint8_t *ipqh = pqh + b * 32;
        float* poutputs = outputs + b * 256;

        uint8_t *ipsl = psl + b * 16;
        uint8_t *ipml = psl + b * 16 + 4;
        uint8_t *ipsh = psl + b * 16 + 8;
        uint8_t *ipmh = psl + b * 16 + 10;
        fp16 *opd = (fp16 *)(psl + b * 16 + 12);
        fp16 *opdmin = (fp16 *)(psl + b * 16 + 14);
        float global_scale = (float)(*opd);
        float global_min = (float)(*opdmin);
        
        scales[0] = (uint8_t)((ipsl[0] & 0x0F) | ((ipsh[0] & 0x03) << 4)) * global_scale;
        scales[1] = (uint8_t)(((ipsl[0] >> 4) & 0x0F) | ((ipsh[0] & 0x0C) << 2)) * global_scale;
        scales[2] = (uint8_t)((ipsl[1] & 0x0F) | (ipsh[0] & 0x30)) * global_scale;
        scales[3] = (uint8_t)(((ipsl[1] >> 4) & 0x0F) | ((ipsh[0] & 0xC0) >> 2)) * global_scale;
        scales[4] = (uint8_t)((ipsl[2] & 0x0F) | ((ipsh[1] & 0x03) << 4)) * global_scale;
        scales[5] = (uint8_t)(((ipsl[2] >> 4) & 0x0F) | ((ipsh[1] & 0x0C) << 2)) * global_scale;
        scales[6] = (uint8_t)((ipsl[3] & 0x0F) | (ipsh[1] & 0x30)) * global_scale;
        scales[7] = (uint8_t)(((ipsl[3] >> 4) & 0x0F) | ((ipsh[1] & 0xC0) >> 2)) * global_scale;

        mins[0] = (uint8_t)((ipml[0] & 0x0F) | ((ipmh[0] & 0x03) << 4)) * global_min;
        mins[1] = (uint8_t)(((ipml[0] >> 4) & 0x0F) | ((ipmh[0] & 0x0C) << 2)) * global_min;
        mins[2] = (uint8_t)((ipml[1] & 0x0F) | (ipmh[0] & 0x30)) * global_min;
        mins[3] = (uint8_t)(((ipml[1] >> 4) & 0x0F) | ((ipmh[0] & 0xC0) >> 2)) * global_min;
        mins[4] = (uint8_t)((ipml[2] & 0x0F) | ((ipmh[1] & 0x03) << 4)) * global_min;
        mins[5] = (uint8_t)(((ipml[2] >> 4) & 0x0F) | ((ipmh[1] & 0x0C) << 2)) * global_min;
        mins[6] = (uint8_t)((ipml[3] & 0x0F) | (ipmh[1] & 0x30)) * global_min;
        mins[7] = (uint8_t)(((ipml[3] >> 4) & 0x0F) | ((ipmh[1] & 0xC0) >> 2)) * global_min;

        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 16; k++)
            {
                quants[j * 32 + k] = ipqs[j * 16 + k] & 0x0F;
                quants[j * 32 + 16 + k] = (ipqs[j * 16 + k] >> 4) & 0x0F;
            }
        }
        for (int j = 0; j < 8; j++)
        {
            for (int s = 0; s < 8; s++)
            {
                for (int k = 0; k < 4; k++)
                {
                    quants[j * 32 + s * 4 + k] += ((ipqh[j * 4 + k] & (1 << s)) ? 16 : 0);
                }
            }
        }
        for (int j = 0; j < 8; j ++)
        {
            for (int k = 0; k < 32; k++)
            {
                poutputs[j * 32 + k] = quants[j * 32 + k] * scales[j] - mins[j];
            }
        }
    }

    delete[] scales;
    delete[] mins;
    delete[] quants;
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
void runGemmQ5KL1_cpu_impl(void *stream, IT *inputs, uint8_t *weights, OT* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len)
{
    IT *inputs_cpu = new IT[token_len * input_len];
    OT *outputs_cpu = new OT[token_len * output_len];
    uint32_t weight_size = input_len * output_len / 256 * sizeof (block_q5_K);
    uint8_t *weights_cpu = new uint8_t[weight_size];
    float *weights_fp32 = new float[input_len * output_len];
    float *inputs_fp32 = new float[token_len * input_len];
    float *outputs_fp32 = new float[token_len * output_len];

    sycl::queue *q = (sycl::queue *)stream;
    q->wait();
    q->memcpy(inputs_cpu, inputs, token_len * input_len * sizeof(IT)).wait();
    q->memcpy(weights_cpu, weights, weight_size).wait();

    dequantQ5KL1(weights_cpu, weights_fp32, input_len, output_len);
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

void runGemmQ5KL1_cpu(void *stream, const void *inputs, const void *weights, const void *bias, void* outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, int input_precision, int bias_precision, int output_precision, ScratchBufferMgr *smgr)
{
    assert(bias == nullptr);
    if (input_precision == 0 && output_precision == 0)
    {
        runGemmQ5KL1_cpu_impl(stream, (float *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 0 && output_precision == 1)
    {
        runGemmQ5KL1_cpu_impl(stream, (float *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 1 && output_precision == 0)
    {
        runGemmQ5KL1_cpu_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (float *)outputs, token_len, input_len, output_len);
    }
    else if (input_precision == 1 && output_precision == 1)
    {
        runGemmQ5KL1_cpu_impl(stream, (fp16 *)inputs, (uint8_t *)weights, (fp16 *)outputs, token_len, input_len, output_len);
    }
}

static RegisterCpuGemm g(GGML_TYPE_Q5_K, "GEMM.Q5_K.ESIMD.L1", ALL_PLATFORMS, runGemmQ5KL1_cpu);
static RegisterShuffle s(GGML_TYPE_Q5_K, "GEMM.Q5_K.ESIMD.L1", ALL_PLATFORMS, ShuffleQ5KL1);