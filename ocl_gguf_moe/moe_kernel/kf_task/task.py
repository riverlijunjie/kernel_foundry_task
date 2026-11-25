# [USER_INSTRUCTIONS_START]
# Optimization target: MoE 3-GEMM SwiGLU MLP kernel (moe_3gemm_swiglu_mlp.cl), GGUF-native
# int8-activation dp4a decode path used by Qwen3.6-35B-A3B decode on Intel Arc / Xe3.
#
# REQUIRED — must be preserved exactly (host code depends on these):
#   1. ALL three KERNEL() entry points and their full parameter signatures (unchanged):
#        KERNEL(mlp_gate_up)(expert_list, gate_weight_addr, gate_scale_addr, gate_zp_addr,
#                            up_weight_addr, up_scale_addr, up_zp_addr, x_addr, y_addr)
#        KERNEL(mlp_down)(expert_list, down_weight_addr, down_scale_addr, down_zp_addr,
#                         x_addr, routing_weights_addr, y_addr)
#        KERNEL(mlp_reduce)(x, y)
#   2. The compile-time dispatch guards:
#        #if GATE_UP_ENABLE ... #elif DOWN_ENABLE ... #else (mlp_reduce) ... #endif
#      (the host selects which kernel to build by setting exactly one of these to 1)
#   3. The FIRST line of the EVOLVE block — the include directive — MUST be kept:
#        #include "include/batch_headers/sub_group_block_read.cl"
#      This pulls in common.cl which defines:
#        unroll_for  =>  __attribute__((opencl_unroll_hint)) for
#      DO NOT remove or move this include. Without it, unroll_for is undefined.
#   4. FUNC() / FUNC_CALL() usage rules (CRITICAL — breaks compilation if violated):
#      - EVERY helper function called via FUNC_CALL(name)(...) MUST be defined as:
#            inline <type> FUNC(name)(<params>) { ... }
#        NOT as:  inline <type> name(<params>)   <- causes "undeclared identifier" errors
#      - Example correct definitions (preserve or replicate exactly):
#            inline half FUNC(moe_gguf_load_f16)(const __global uchar* p) { ... }
#            inline void FUNC(moe_gguf_get_scale_min_k4)(int j, const __global uchar* q,
#                                                        uchar* d, uchar* m) { ... }
#            inline float FUNC(moe_dp_dot_q4_k)(...) { ... }
#      - If you inline or remove a helper, also remove all its FUNC_CALL() call sites.
#   5. MOE_GGUF_DP4A / MOE_GGUF_LISA conditional guards for the two decode paths.
#   6. The EVOLVE block MUST define ALL THREE of the following dp4a driver functions
#      (the KERNEL body outside the EVOLVE block calls them; if any is missing the
#       build fails with "undeclared identifier"):
#
#      a) Activation quantizer — 5 parameters including xsum (for Q4_K/Q5_K min-fold):
#            inline void FUNC(moe_gguf_quant_act_i8)(
#                const __local half* x2,   // input half activation [K]
#                __local char*  xq,        // output int8 quantized activation [K]
#                __local float* xsc,       // output per-32 f32 scale [K/32]
#                __local int*   xsum,      // output per-32 activation sum [K/32]
#                int K)
#
#      b) Gate/up dp4a GEMV driver — 10 parameters including xsum:
#            inline void FUNC(moe_gguf_gate_up_gemv_dp4a)(
#                int code,
#                const __global uchar* row_base, __global half* y, int N, int K,
#                const __local char* xq, const __local float* xsc,
#                const __local int* xsum,   // <-- required 8th parameter
#                const __local half* x2,
#                const bool silu)
#
#      c) Down dp4a GEMV driver — 10 parameters including xsum:
#            inline void FUNC(moe_gguf_down_gemv_dp4a)(
#                int code,
#                const __global uchar* row_base, MOE_DTYPE routing_weight_val,
#                __global half* y, int N, int K,
#                const __local char* xq, const __local float* xsc,
#                const __local int* xsum,   // <-- required 9th parameter
#                const __local half* x2)
#
# OPTIMIZATION targets (improve freely):
#   - The dp4a decode inner loops (vectorize using uchar4/int4/dot4, unroll, pipeline)
#   - SLM layout for xq/xsc/xsum activation requantization
#   - Register blocking and accumulation in the GEMV loops
#   - Memory coalescing and vectorized weight loads
#   - Work-item granularity and sub-group utilization
# [USER_INSTRUCTIONS_END]

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import pyopencl as cl

from kernelfoundry import TestBase  # Tests must derive from TestBase

# --------------------------------------------------------------------------------------
# Reuse the standalone harness (OV-faithful JIT preamble + NumPy GGUF-decode oracle) that
# ships next to this task. These modules are the single source of truth for how OpenVINO
# builds/dispatches the extracted MoE 3-GEMM SwiGLU kernel, so a kernel optimized here
# transfers 1:1 back into OV. Mirrors the sibling gguf_kernels task layout.
# --------------------------------------------------------------------------------------
TASK_DIR = Path(__file__).parent.resolve()
HARNESS_DIR = TASK_DIR / "harness"
KDIR = str(TASK_DIR / "kernels")  # include/ headers (common.cl, batch_headers) live here
sys.path.insert(0, str(HARNESS_DIR))

import moe_jit  # noqa: E402
import moe_ref as mr  # noqa: E402

# --------------------------------------------------------------------------------------
# Target problem: moe_3gemm_swiglu_qwen36_H2048_I512_E256
#   MoE 3-GEMM SwiGLU MLP, GGUF-native block decode via the int8-activation dp4a path
#   (MOE_GGUF_DP4A=1) used by Qwen3.6-35B-A3B decode. The single .cl produces two entry
#   points (gate_up, down) selected by JIT constants. Benchmarked on the three per-expert
#   decode formats that actually appear in the model:
#       gate_up : Q4_K   (silu(gate) * up)
#       down    : Q5_K   (37/40 layers)
#       down    : Q6_K   ( 3/40 layers)
#   M=1 decode, top-k=8 routed experts, f16 output. Matches configs/qwen36.json.
# --------------------------------------------------------------------------------------
HIDDEN = 2048
INTER = 512
EXPERTS = 256
TOPK = 8
SEED = 1234

# (stage kind, GGUF weight format) — the three benchmarked cases (== configs/qwen36.json)
CASES = [
    ("gate_up", "gguf_q4_k"),
    ("down", "gguf_q5_k"),
    ("down", "gguf_q6_k"),
]

VERIFY_SLOTS = 2   # expert slots to check against the (slow Python) NumPy oracle
VERIFY_ROWS = 48   # output rows per slot to check

KERNEL_FILE = "kernel_moe.cl"
REFERENCE_FILE = "reference_moe.cl"

SG = moe_jit.SUBGROUP_SIZE   # 32 (Xe2+)
NB = moe_jit.N_BLOCK         # 4
SGN = moe_jit.SUBGROUP_NUM   # 8


# --------------------------------------------------------------------------------------
# OpenCL context / build helpers
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ocl_queue():
    """First GPU device across all platforms, profiling-enabled command queue."""
    device = None
    for platform in cl.get_platforms():
        devs = platform.get_devices(cl.device_type.GPU)
        if devs:
            device = devs[0]
            break
    assert device is not None, "No OpenCL GPU device found"
    ctx = cl.Context([device])
    return cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


def _build(ctx, cl_path, entry_point, consts):
    """Prepend the OV-faithful JIT preamble to a .cl source and build the entry point."""
    with open(cl_path) as f:
        body = f.read()
    src = moe_jit.build_preamble(entry_point, consts) + "\n" + body
    opts = ["-I", KDIR] + moe_jit.OV_BUILD_OPTIONS
    prog = cl.Program(ctx, src).build(options=opts)
    return cl.Kernel(prog, entry_point)


def _dummy(ctx, nbytes=256):
    """Unused scale/zero-point args (the GGUF native-decode path reads the raw block)."""
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=nbytes)


def _el_pool(experts, topk):
    """Rotating contiguous windows of `topk` experts sweeping all `experts`.

    The per-launch top-k weight working set (~5-9 MiB) fits in the 16 MiB L2, so a naive
    repeated benchmark goes L2-hot and stops reflecting the real DRAM-bound decode. Cycling
    these expert-list buffers across benchmark iterations (via measure_runtime's args list)
    makes successive launches read *different* experts, so the union spans the full weight
    (>> L2) and every launch streams cold from DRAM — no explicit L2 flush needed.
    """
    n = max(1, experts // topk)
    return [np.asarray([(j * topk + t) % experts for t in range(topk)], np.int32) for j in range(n)]


def _rel_l2(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a - b))


# --------------------------------------------------------------------------------------
# Per-case setup: build the GEMV kernel + device buffers ONCE. The full E=256 expert
# weights are resident (>> L2) so cycling `d_el_pool` streams the weight cold. Returns a
# uniform `state` interface: enqueue(queue, d_el) dispatches ONLY the GEMV (the evolve
# target); the MoE kernel entry does its own in-SLM activation int8-requant, so there is
# no separate prequant kernel to isolate (unlike the gguf gemv task).
# --------------------------------------------------------------------------------------
def _setup_gate_up(ctx, fmt, cl_path):
    mf = cl.mem_flags
    rng = np.random.default_rng(SEED)
    be, bb = moe_jit.GGUF_BLK[fmt]
    row_bytes = (HIDDEN // be) * bb
    # SwiGLU squares the projection magnitude (silu(gate)*up); keep weights/activation small
    # so the f16 output stays well inside HALF_MAX (timing is data-independent).
    gate_w = mr.gen_expert_weights(fmt, EXPERTS, INTER, HIDDEN, seed=SEED, scale_amp=0.004)
    up_w = mr.gen_expert_weights(fmt, EXPERTS, INTER, HIDDEN, seed=SEED + 7, scale_amp=0.004)
    x = (rng.standard_normal((1, HIDDEN)) * 0.05).astype(np.float16)
    y = np.zeros((TOPK, INTER), np.float16)

    pools = _el_pool(EXPERTS, TOPK)
    d_gw = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(gate_w))
    d_uw = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(up_w))
    d_x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    d_y = cl.Buffer(ctx, mf.WRITE_ONLY, size=y.nbytes)
    d_el_pool = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p) for p in pools]
    d_gs, d_gz, d_us, d_uz = _dummy(ctx), _dummy(ctx), _dummy(ctx), _dummy(ctx)

    ep, consts = moe_jit.gate_up_jit(fmt, HIDDEN, INTER, EXPERTS, TOPK)
    krn = _build(ctx, cl_path, ep, consts)
    gsz = (TOPK, SG, INTER // NB)
    lsz = (1, SG, SGN)

    def enqueue(queue, d_el):
        return krn(queue, gsz, lsz, d_el, d_gw, d_gs, d_gz, d_uw, d_us, d_uz, d_x, d_y)

    def read_output(queue):
        cl.enqueue_copy(queue, y, d_y)
        queue.finish()
        return y.astype(np.float32)

    def oracle(el):
        """silu(x @ gate_e) * (x @ up_e) for the first VERIFY_SLOTS experts, VERIFY_ROWS rows."""
        xdeq = mr.act_for(fmt, x[0])
        V = min(VERIFY_ROWS, INTER)
        S = min(VERIFY_SLOTS, TOPK)
        ref = np.zeros((S, V), np.float32)
        for s in range(S):
            e = int(el[s])
            ge = gate_w[e].reshape(INTER, row_bytes)
            ue = up_w[e].reshape(INTER, row_bytes)
            for n in range(V):
                gW = mr.deq_row(fmt, ge[n], HIDDEN)
                uW = mr.deq_row(fmt, ue[n], HIDDEN)
                ref[s, n] = mr.silu(float(xdeq @ gW)) * float(xdeq @ uW)
        return ref, V

    return dict(enqueue=enqueue, read_output=read_output, oracle=oracle,
                el_pool=pools, d_el_pool=d_el_pool)


def _setup_down(ctx, fmt, cl_path):
    mf = cl.mem_flags
    rng = np.random.default_rng(SEED)
    be, bb = moe_jit.GGUF_BLK[fmt]
    row_bytes = (INTER // be) * bb
    down_w = mr.gen_expert_weights(fmt, EXPERTS, HIDDEN, INTER, seed=SEED + 13, scale_amp=0.01)
    x = (rng.standard_normal((TOPK, INTER)) * 0.2).astype(np.float16)      # intermediate acts
    routing = (rng.random(TOPK) * 0.5 + 0.1).astype(np.float16)           # per-slot router weight
    y = np.zeros((TOPK, HIDDEN), np.float16)

    pools = _el_pool(EXPERTS, TOPK)
    d_dw = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(down_w))
    d_x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(x))
    d_rw = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=routing)
    d_y = cl.Buffer(ctx, mf.WRITE_ONLY, size=y.nbytes)
    d_el_pool = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p) for p in pools]
    d_ds, d_dz = _dummy(ctx), _dummy(ctx)

    ep, consts = moe_jit.down_jit(fmt, HIDDEN, INTER, EXPERTS, TOPK)
    krn = _build(ctx, cl_path, ep, consts)
    gsz = (TOPK, SG, HIDDEN // NB)
    lsz = (1, SG, SGN)

    def enqueue(queue, d_el):
        return krn(queue, gsz, lsz, d_el, d_dw, d_ds, d_dz, d_x, d_rw, d_y)

    def read_output(queue):
        cl.enqueue_copy(queue, y, d_y)
        queue.finish()
        return y.astype(np.float32)

    def oracle(el):
        """(x_s @ down_e) * router_s for the first VERIFY_SLOTS slots, VERIFY_ROWS rows."""
        V = min(VERIFY_ROWS, HIDDEN)
        S = min(VERIFY_SLOTS, TOPK)
        ref = np.zeros((S, V), np.float32)
        for s in range(S):
            e = int(el[s])
            de = down_w[e].reshape(HIDDEN, row_bytes)
            xdeq = mr.act_for(fmt, x[s])
            for n in range(V):
                dW = mr.deq_row(fmt, de[n], INTER)
                ref[s, n] = float(xdeq @ dW) * float(routing[s])
        return ref, V

    return dict(enqueue=enqueue, read_output=read_output, oracle=oracle,
                el_pool=pools, d_el_pool=d_el_pool)


def _setup(ctx, kind, fmt, cl_path):
    return _setup_gate_up(ctx, fmt, cl_path) if kind == "gate_up" else _setup_down(ctx, fmt, cl_path)


@pytest.fixture(scope="session")
def get_state(ocl_queue, use_reference):
    """Session-cached per-case runner built from the evolve kernel (or the reference under --ref)."""
    cache = {}
    cl_path = str(TASK_DIR / (REFERENCE_FILE if use_reference else KERNEL_FILE))

    def _get(kind, fmt):
        key = (kind, fmt)
        if key not in cache:
            cache[key] = _setup(ocl_queue.context, kind, fmt, cl_path)
        return cache[key]

    return _get


# --------------------------------------------------------------------------------------
# Test class
# --------------------------------------------------------------------------------------
class TestMoE3GemmSwigluDp4a(TestBase):
    """MoE 3-GEMM SwiGLU MLP, GGUF int8-activation dp4a decode (moe_3gemm_swiglu_qwen36_H2048_I512_E256)."""

    def build(self, gpu_arch) -> list[str]:
        """Best-effort compile check for both entry points. The kernel needs GPU-only
        OpenCL features (cl_khr_integer_dot_product, intel_reqd_sub_group_size), so real
        validation happens when the test compiles it on the GPU. On a GPU-less build node
        this is a no-op; on a GPU node it compiles the evolve kernel to catch syntax errors."""
        device = None
        for platform in cl.get_platforms():
            devs = platform.get_devices(cl.device_type.GPU)
            if devs:
                device = devs[0]
                break
        if device is None:
            return  # no GPU on this node -> defer compilation to the GPU test
        ctx = cl.Context([device])
        cl_path = str(TASK_DIR / KERNEL_FILE)
        for kind, fmt in CASES:
            if kind == "gate_up":
                ep, consts = moe_jit.gate_up_jit(fmt, HIDDEN, INTER, EXPERTS, TOPK)
            else:
                ep, consts = moe_jit.down_jit(fmt, HIDDEN, INTER, EXPERTS, TOPK)
            _build(ctx, cl_path, ep, consts)

    @pytest.mark.parametrize("kind,fmt", CASES)
    def test_correctness_vs_numpy(self, get_state, ocl_queue, kind, fmt):
        """GEMV output vs the NumPy GGUF-decode oracle on a subset of experts/rows."""
        st = get_state(kind, fmt)
        st["enqueue"](ocl_queue, st["d_el_pool"][0]).wait()
        C = st["read_output"](ocl_queue)
        ref, V = st["oracle"](st["el_pool"][0])
        S = ref.shape[0]
        err = _rel_l2(C[:S, :V], ref)
        assert err < 3e-2, f"[{kind}/{fmt}] rel-L2 vs numpy {err:.4g} too high"

    @pytest.mark.parametrize("kind,fmt", CASES)
    def test_correctness_vs_reference(self, get_state, ocl_queue, kind, fmt):
        """GEMV output vs the reference .cl over the full output (cheap GPU-vs-GPU compare)."""
        st = get_state(kind, fmt)
        st["enqueue"](ocl_queue, st["d_el_pool"][0]).wait()
        C = st["read_output"](ocl_queue).copy()

        ref_st = _setup(ocl_queue.context, kind, fmt, str(TASK_DIR / REFERENCE_FILE))
        ref_st["enqueue"](ocl_queue, ref_st["d_el_pool"][0]).wait()
        C_ref = ref_st["read_output"](ocl_queue)
        err = _rel_l2(C, C_ref)
        assert err < 1e-2, f"[{kind}/{fmt}] rel-L2 vs reference {err:.4g} too high"

    @pytest.mark.parametrize("kind,fmt", CASES)
    @pytest.mark.performance
    def test_benchmark(self, get_state, ocl_queue, measure_runtime, kind, fmt):
        """Times ONLY the GEMV kernel, cycling expert-list buffers so the weight stays
        DRAM-cold across iterations (the top-k working set alone would go L2-resident)."""
        st = get_state(kind, fmt)
        args = [(d_el,) for d_el in st["d_el_pool"]]
        kwargs = [{}] * len(args)
        measure_runtime(
            lambda d_el: st["enqueue"](ocl_queue, d_el),
            sync_fn=ocl_queue.finish,
            args=args,
            kwargs=kwargs,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestMoE3GemmSwigluDp4a()
    task.build(TestBase.get_machine_gpu_arch())
