// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// KernelFoundry harness for gr_up_micro.cl (UP_PROJ micro-GEMM of the GatedResidual oneDNN READ path).
// The READ pipeline is RBAR -> oneDNN DOWN matmul -> gr_up_micro; the tests force it and check that gr_up_micro ran.
//   correctness: fp32 reference on a subset of token rows (first / last rows, tile boundaries and a stride over the rest).
//   benchmark:   prints "KF_READY", waits for "RUN" on stdin, then prints one "KF_RUNTIME_MS <ms>" line per iteration and "KF_DONE".
//                The time is the GPU-side event time of the gated_residual primitive.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gated_residual.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/runtime/profiling.hpp>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "gated_residual_inst.h"
#include "intel_gpu/op/gated_residual.hpp"
#include "intel_gpu/primitives/gated_residual_caps.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;
namespace caps = cldnn::gated_residual_caps;

namespace {

constexpr size_t kBranches = 4;
constexpr size_t kHidden = 640;                       // D
constexpr size_t kLowrank = 320;                      // L: K of the UP_PROJ micro-GEMM
constexpr size_t kBranchedWidth = kBranches * kHidden;  // C * D: N of the UP_PROJ micro-GEMM
constexpr float kEpsilon = 1e-6f;
constexpr float kTolerance = 2e-2f;

struct gr_weights {
    std::vector<float> gain;    // [C*D]
    std::vector<float> w_down;  // [L, C*D]
    std::vector<float> w_up;    // [C*D, L]
};

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// tests::random_generator only takes integer bounds: draw from [-1, 1] with 1/16 resolution, then scale and shift.
std::vector<float> random_vec(tests::random_generator& rg, size_t count, float scale, float bias = 0.0f) {
    auto v = rg.generate_random_1d<float>(count, -1, 1, 16);
    std::vector<float> out(v.begin(), v.end());
    for (auto& x : out) {
        x = x * scale + bias;
    }
    return out;
}

std::vector<ov::float16> to_f16(const std::vector<float>& src) {
    std::vector<ov::float16> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = ov::float16(src[i]);
    }
    return dst;
}

memory::ptr make_f16(engine& eng, const ov::PartialShape& shape, const std::vector<float>& data) {
    auto mem = eng.allocate_memory(layout{shape, data_types::f16, format::bfyx});
    set_values(mem, to_f16(data));
    return mem;
}

gr_weights make_weights(tests::random_generator& rg) {
    gr_weights w;
    w.gain = random_vec(rg, kBranchedWidth, 0.5f, 1.0f);
    w.w_down = random_vec(rg, kLowrank * kBranchedWidth, 0.1f);
    w.w_up = random_vec(rg, kBranchedWidth * kLowrank, 0.1f);
    return w;
}

bool gr_available() {
    return caps::is_device_supported(get_test_engine().get_device_info());
}

// GPU_GR_KERNEL_SELECTION=onednn needs the debug caps; without them `auto` routes the oneDNN path only on some devices / row counts.
ExecutionConfig make_config(bool profiling) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
#ifdef GPU_DEBUG_CONFIG
    config.set_property(ov::intel_gpu::gr_kernel_selection("onednn"));
#endif
    if (profiling) {
        config.set_property(ov::enable_profiling(true));
    }
    return config;
}

topology make_topology(const memory::ptr& residual_mem, const gr_weights& w, bool with_sink) {
    auto& engine = get_test_engine();
    topology topo;
    topo.add(input_layout("residual", residual_mem->get_layout()));
    topo.add(data("gain", make_f16(engine, {static_cast<int64_t>(kBranchedWidth)}, w.gain)));
    topo.add(data("w_down", make_f16(engine, {static_cast<int64_t>(kLowrank), static_cast<int64_t>(kBranchedWidth)}, w.w_down)));
    topo.add(data("w_up", make_f16(engine, {static_cast<int64_t>(kBranchedWidth), static_cast<int64_t>(kLowrank)}, w.w_up)));
    topo.add(gated_residual("gr_read",
                            {input_info("residual"), input_info("gain"), input_info("w_down"), input_info("w_up")},
                            gated_residual::Mode::READ,
                            kBranches,
                            kHidden,
                            kLowrank,
                            kEpsilon,
                            /*norm_gain_plus_one=*/false,
                            /*produce_write_gate=*/false));
    if (with_sink) {
        // A network output's event is a marker; with a consumer the event of the multi-dispatch READ groups its kernels.
        topo.add(activation("sink", input_info("gr_read"), activation_func::abs));
    }
    return topo;
}

std::string executed_kernels(network& net) {
    const auto inst = net.get_primitive("gr_read");
    return inst->get_impl()->get_kernels_dump_info(*inst->get_impl_params()).get_entries();
}

void check_up_micro_ran(network& net) {
    const auto kernels = executed_kernels(net);
    ASSERT_NE(kernels.find("gr_up_micro_"), std::string::npos) << "gr_up_micro did not run, executed kernels: " << kernels;
}

// fp32 RBAR -> DOWN (+ SiLU) -> UP (+ sigmoid, Rbar product, branch mean) for one token row.
void reference_row(const float* residual, const gr_weights& w, std::vector<float>& mixed) {
    std::vector<float> rbar(kBranchedWidth);
    for (size_t c = 0; c < kBranches; c++) {
        double sumsq = 0.0;
        for (size_t d = 0; d < kHidden; d++) {
            const float v = residual[c * kHidden + d];
            sumsq += static_cast<double>(v) * v;
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(sumsq / kHidden) + kEpsilon);
        for (size_t d = 0; d < kHidden; d++) {
            const size_t idx = c * kHidden + d;
            rbar[idx] = residual[idx] * inv * w.gain[idx];
        }
    }

    const float inv_c = 1.0f / static_cast<float>(kBranches);
    std::vector<float> t(kLowrank);
    for (size_t l = 0; l < kLowrank; l++) {
        double acc = 0.0;
        for (size_t k = 0; k < kBranchedWidth; k++) {
            acc += static_cast<double>(w.w_down[l * kBranchedWidth + k]) * rbar[k];
        }
        const float x = static_cast<float>(acc) * inv_c;
        t[l] = x * sigmoidf(x);
    }

    mixed.assign(kHidden, 0.0f);
    for (size_t d = 0; d < kHidden; d++) {
        double acc = 0.0;
        for (size_t c = 0; c < kBranches; c++) {
            double logit = 0.0;
            for (size_t l = 0; l < kLowrank; l++) {
                logit += static_cast<double>(w.w_up[(c * kHidden + d) * kLowrank + l]) * t[l];
            }
            acc += static_cast<double>(sigmoidf(static_cast<float>(logit))) * rbar[c * kHidden + d];
        }
        mixed[d] = static_cast<float>(acc) * inv_c;
    }
}

std::vector<size_t> rows_to_check(size_t rows) {
    std::set<size_t> picked;
    for (const size_t r : {0, 1, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 1023, 1024}) {
        if (r < rows) {
            picked.insert(r);
        }
    }
    for (size_t r = 0; r < 8 && r < rows; r++) {
        picked.insert(rows - 1 - r);
    }
    const size_t stride = std::max<size_t>(1, rows / 96);
    for (size_t r = 3; r < rows; r += stride) {
        picked.insert(r);
    }
    return {picked.begin(), picked.end()};
}

double executing_ms(const event::ptr& ev) {
    if (ev == nullptr) {
        return -1.0;
    }
    for (const auto& interval : ev->get_profiling_info()) {
        if (interval.stage == instrumentation::profiling_stage::executing && interval.value != nullptr) {
            return static_cast<double>(interval.value->value().count()) / 1.0e6;
        }
    }
    return -1.0;
}

size_t benchmark_iterations() {
    const char* value = std::getenv("KF_GR_ITERS");
    const long parsed = value != nullptr ? std::strtol(value, nullptr, 10) : 0;
    return parsed > 0 ? static_cast<size_t>(parsed) : 100;
}

class gr_up_micro_kf_test : public ::testing::TestWithParam<size_t> {};

TEST_P(gr_up_micro_kf_test, correctness) {
    ASSERT_TRUE(gr_available()) << "no GatedResidual implementation is available on this device";
    const size_t rows = GetParam();

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    const auto residual = random_vec(rg, rows * kBranchedWidth, 1.0f);
    const auto weights = make_weights(rg);

    auto& engine = get_test_engine();
    auto residual_mem = make_f16(engine, {static_cast<int64_t>(rows), static_cast<int64_t>(kBranchedWidth)}, residual);
    network net(engine, make_topology(residual_mem, weights, /*with_sink=*/false), make_config(/*profiling=*/false));
    net.set_input_data("residual", residual_mem);
    auto outputs = net.execute();
    ASSERT_TRUE(outputs.count("gr_read") > 0);
    ASSERT_NO_FATAL_FAILURE(check_up_micro_ran(net));

    cldnn::mem_lock<ov::float16> out(outputs.at("gr_read").get_memory(), get_test_stream());
    ASSERT_EQ(out.size(), rows * kHidden);
    for (size_t i = 0; i < out.size(); i++) {
        ASSERT_TRUE(std::isfinite(static_cast<float>(out[i]))) << "non-finite output at " << i;
    }

    float max_err = 0.0f;
    std::vector<float> expected;
    for (const size_t row : rows_to_check(rows)) {
        reference_row(residual.data() + row * kBranchedWidth, weights, expected);
        for (size_t d = 0; d < kHidden; d++) {
            const float got = static_cast<float>(out[row * kHidden + d]);
            ASSERT_NEAR(got, expected[d], kTolerance) << "mismatch at row " << row << " d " << d;
            max_err = std::max(max_err, std::abs(got - expected[d]));
        }
    }
    std::cout << "GR_UP_MICRO rows=" << rows << " max_err_vs_fp32_ref=" << max_err << std::endl;
}

TEST_P(gr_up_micro_kf_test, benchmark) {
    ASSERT_TRUE(gr_available()) << "no GatedResidual implementation is available on this device";
    const size_t rows = GetParam();
    const size_t iterations = benchmark_iterations();
    constexpr size_t warmup = 20;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    const auto residual = random_vec(rg, rows * kBranchedWidth, 1.0f);
    const auto weights = make_weights(rg);

    auto& engine = get_test_engine();
    // Device-resident like real activations: the default lockable allocation can be usm_shared and would make RBAR measure PCIe reads.
    auto residual_mem = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(rows), static_cast<int64_t>(kBranchedWidth)}, data_types::f16, format::bfyx},
                                               allocation_type::usm_device);
    const auto residual_f16 = to_f16(residual);
    residual_mem->copy_from(get_test_stream(), residual_f16.data(), true);

    network net(engine, make_topology(residual_mem, weights, /*with_sink=*/true), make_config(/*profiling=*/true));
    net.set_input_data("residual", residual_mem);
    for (size_t i = 0; i < warmup; i++) {
        net.execute().at("sink").get_memory();
    }
    ASSERT_NO_FATAL_FAILURE(check_up_micro_ran(net));

    std::cout << "KF_READY" << std::endl;
    std::string command;
    ASSERT_TRUE(std::getline(std::cin, command) && command == "RUN") << "expected RUN on stdin after KF_READY";

    // Iterations are enqueued back-to-back and synchronized at the end: a host wait per iteration lets the GPU clock drop.
    std::vector<event::ptr> events;
    events.reserve(iterations);
    for (size_t i = 0; i < iterations; i++) {
        net.execute();
        events.push_back(net.get_executed_primitives().at("gr_read"));
    }
    std::vector<double> runtimes;
    runtimes.reserve(iterations);
    for (const auto& ev : events) {
        ev->wait();
        const double ms = executing_ms(ev);
        ASSERT_GT(ms, 0.0) << "no 'executing' profiling interval was captured for gr_read";
        runtimes.push_back(ms);
    }
    for (const double ms : runtimes) {
        std::cout << "KF_RUNTIME_MS " << std::setprecision(9) << ms << '\n';
    }
    std::cout << "KF_DONE" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(kf,
                         gr_up_micro_kf_test,
                         ::testing::Values(size_t{1024}, size_t{2048}, size_t{4096}, size_t{8192}),
                         [](const ::testing::TestParamInfo<size_t>& info) {
                             return "M" + std::to_string(info.param);
                         });

}  // namespace
