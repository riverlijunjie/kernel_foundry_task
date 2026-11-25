# [USER_INSTRUCTIONS_START]
# Optimization target: fc_gguf_gemm_opt.cl  (OpenVINO intel_gpu, graph/impls/ocl_v2/fc_gguf_gemm_opt.cl)
#   GGUF prefill GEMM  C[M,N] = A[M,K] x deq(W)[N,K]^T  on Intel Xe2/Xe3 (BMG / PTL) using DPAS
#   (intel_sub_group_f16_f16_matrix_mad_k16), 2D block IO and an SLM-shared decoded weight tile.
#   One source serves six GGUF formats (Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0): the host defines exactly one
#   GG_FMT_* macro per build and a per-format entry point (fc_gguf_gemm_<fmt>); every build uses the same
#   KERNEL(fc_gguf_gemm) body, so all six formats are tested and timed.
#   fc_gguf_gemm_common.cl and fc_gguf_gemm_body.cl are INLINED into the single EVOLVE block of this file.
#
# This task is self-contained: the OV-faithful JIT preamble / tile config / dispatch, the SG-shuffle weight
# repack and the NumPy dequant oracle live in harness/gguf_gemm_ref.py; the shared headers in kernels/.
#
# REQUIRED - must be preserved exactly (host code and the weight repack depend on these):
#   1. The entry point and parameter list (the host binds them positionally):
#        KERNEL(fc_gguf_gemm)(OPTIONAL_SHAPE_INFO_ARG
#            const __global INPUT0_TYPE* A, const __global uchar* W, __global OUTPUT_TYPE* C,
#            uint M, uint K, uint N [, FUSED_OPS_DECLS])
#   2. The two include directives at the top of the EVOLVE block:
#        #include "include/batch_headers/common.cl"
#        #include "include/batch_headers/fetch_data.cl"
#   3. FUNC() / FUNC_CALL() usage: every helper called via FUNC_CALL(name)(...) MUST be defined as
#        inline <type> FUNC(name)(<params>) { ... }
#   4. The six per-format sections selected by GG_FMT_Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / Q8_0 and the
#      `#error` fallback. The host sets exactly one of them.
#   5. The kernel attributes: intel_reqd_sub_group_size(16) and reqd_work_group_size(16, TOKEN_LOCAL, 1).
#      Dispatch is fixed by the host:
#        local  = [16, TOKEN_LOCAL, 1]
#        global = [N / (16*ROW_GROUPS*ROW_BLOCKS) * 16, ceil(ceil(M/(8*TOKEN_GROUPS)) / (TOKEN_LOCAL/ROW_BLOCKS)) * TOKEN_LOCAL, 1]
#   6. The JIT constants (ROW_GROUPS, TOKEN_GROUPS, TOKEN_LOCAL, ROW_BLOCKS, SLM_JBLK, PF_A, GG_N_BAND,
#      GG_A_F32, GG_OUT_F32, INPUT0_TYPE, OUTPUT_TYPE) are produced by the host for each shape / arch; keep
#      the kernel valid for every combination (BMG: RG=2 TG=4 TL=8 RB=1; PTL: RG=4 TG=4 TL=8/16 RB=1/2 with
#      256 GRF), not only the one measured.
#   7. The weight is read in the SG-shuffled layout produced by RepackGGUFWeightsShuffle (same buffer the
#      decode GEMV reads) - the layout cannot change, only how it is fetched and decoded.
#   8. The output layout C[bm*N + n] (f16, or f32 with GG_OUT_F32); the HAS_FUSED_OPS path stays guarded by
#      #if HAS_FUSED_OPS.
#
# Accuracy: fp32 accumulation; output must match A @ deq(W)^T within rel-L2 2e-2 (f16 or f32 output).
#
# OPTIMIZATION ideas:
#   - Cheaper dequant of the SLM weight tile (fewer ALU ops per element, packed f16 math, LUT/bit tricks)
#   - Better overlap of the next phase's global weight fetch + decode with the DPAS of the current phase
#   - Activation prefetch distance / 2D block-load shape, SLM bank layout and barrier count
#   - Different SLM ring depth, work-group token/row split, K-loop unrolling, register pressure vs 256 GRF
# [USER_INSTRUCTIONS_END]

from kernelfoundry.eval_pipeline.utils.performance import detect_profiler, profiler_session

import functools
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import pyopencl as cl

from kernelfoundry import TestBase  # Tests must derive from TestBase

TASK_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(TASK_DIR / "harness"))
KDIR = str(TASK_DIR / "kernels")  # include/batch_headers/*.cl

import gguf_gemm_ref as gr  # noqa: E402

KERNEL_FILE = str(TASK_DIR / "fc_gguf_gemm_opt.cl")
REFERENCE_FILE = str(TASK_DIR / "fc_gguf_gemm_reference.cl")

SEED = 1234
BENCH = dict(flush_mib=128, iters=30, warmup=5, flush_l2=True)


def _case(fmt, M, K, N, in_dt="f16", out_dt="f16"):
    return dict(fmt=fmt, M=M, K=K, N=N, in_dt=in_dt, out_dt=out_dt)


def _cid(c):
    return "%s_M%d_K%d_N%d_%s_%s" % (c["fmt"], c["M"], c["K"], c["N"], c["in_dt"], c["out_dt"])


# Small shapes checked against the NumPy oracle (token tail M=77, multi-K-block, f32 in/out variants).
CORR_CASES = [_case(f, 77, 1024, 256) for f in gr.FORMATS] + [
    _case("q4_k", 130, 768, 512, "f32", "f32"),
    _case("q6_k", 130, 768, 512, "f32", "f32"),
    _case("q8_0", 41, 512, 128, "f16", "f32"),
]
# Prefill-sized shapes: GPU-vs-reference compare and timing (K=N=4096, f16 in/out, as in the OV perf test).
BENCH_CASES = [_case(f, M, 4096, 4096) for f in gr.FORMATS for M in (1024, 4096, 8192)]


# --------------------------------------------------------------------------------------
# Device / build helpers
# --------------------------------------------------------------------------------------
def _gpu_device():
    for platform in cl.get_platforms():
        devs = platform.get_devices(cl.device_type.GPU)
        if devs:
            return devs[0]
    return None


def _is_xe3(device):
    """Xe3 (PTL, Arc B390) uses the 256-GRF tile config; Xe2 (BMG) the small one. GGUF_GEMM_ARCH=xe2|xe3 overrides."""
    forced = os.environ.get("GGUF_GEMM_ARCH", "").lower()
    if forced in ("xe2", "bmg"):
        return False
    if forced in ("xe3", "ptl"):
        return True
    name = device.name.upper()
    return "B390" in name or "B370" in name or "PANTHER" in name


@pytest.fixture(scope="session")
def ocl_queue():
    device = _gpu_device()
    assert device is not None, "No OpenCL GPU device found"
    ctx = cl.Context([device])
    return cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


@pytest.fixture(scope="session")
def l2_flush(ocl_queue):
    """Created once, before any benchmark: scratch pages committed, flush kernel warm."""
    ctx = ocl_queue.context
    elems = (BENCH["flush_mib"] * 1024 * 1024) // 4
    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=elems * 4)
    sink = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=4096 * 4)
    cl.enqueue_copy(ocl_queue, buf, np.zeros(elems, np.float32))
    krn = cl.Kernel(cl.Program(ctx, gr.CACHE_FLUSH_CL).build(options=gr.OV_BUILD_OPTIONS), "cache_flush")
    ocl_queue.finish()
    flush = dict(krn=krn, buf=buf, sink=sink, elems=elems)
    _flush_enqueue(ocl_queue, flush)
    ocl_queue.finish()
    return flush


def _flush_enqueue(queue, flush, sweeps=2):
    """Sweep a scratch buffer >> L2 so the next launch reads its weight cold from DRAM (in-order queue)."""
    gsz = (min(flush["elems"], 1024 * 1024),)
    for _ in range(sweeps):
        flush["krn"](queue, gsz, None, flush["buf"], np.uint32(flush["elems"]), flush["sink"])


def _build_kernel(ctx, path, c):
    cfg = gr.pick_config(c["fmt"], c["N"], c["K"], _is_xe3(ctx.devices[0]))
    entry = gr.ENTRY_POINT[c["fmt"]]
    with open(path) as f:
        body = f.read()
    src = gr.build_preamble(entry, gr.jit_constants(c["fmt"], cfg, c["in_dt"], c["out_dt"])) + "\n" + body
    prog = cl.Program(ctx, src).build(options=["-I", KDIR] + gr.build_options(cfg))
    return cl.Kernel(prog, entry), cfg


@functools.lru_cache(maxsize=None)
def _weights(fmt, N, K):
    W = gr.gen_weight_bytes(fmt, N, K, seed=SEED)
    return W, np.ascontiguousarray(gr.shuffle_weights(fmt, W))


def _make_runner(ctx, c, path):
    fmt, M, K, N = c["fmt"], c["M"], c["K"], c["N"]
    krn, cfg = _build_kernel(ctx, path, c)
    assert N % gr.n_align(cfg) == 0, "N=%d must be a multiple of %d for %s" % (N, gr.n_align(cfg), cfg)
    Wnat, Wsh = _weights(fmt, N, K)
    rng = np.random.default_rng(SEED + 1)
    a_np = np.float16 if c["in_dt"] == "f16" else np.float32
    c_np = np.float16 if c["out_dt"] == "f16" else np.float32
    A = (rng.standard_normal((M, K)) * 0.5).astype(a_np)
    mf = cl.mem_flags
    d_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    d_W = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Wsh)
    C = np.zeros((M, N), c_np)
    d_C = cl.Buffer(ctx, mf.WRITE_ONLY, size=C.nbytes)
    gsz, lsz = gr.dispatch(cfg, M, N)
    scalars = (np.uint32(M), np.uint32(K), np.uint32(N))

    def run(queue):
        return krn(queue, gsz[:2], lsz[:2], d_A, d_W, d_C, *scalars)

    return dict(c=c, A=A, Wnat=Wnat, C=C, d_C=d_C, run=run)


def _read_output(state, queue):
    cl.enqueue_copy(queue, state["C"], state["d_C"])
    queue.finish()
    return state["C"].astype(np.float32)


def _rel_l2(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a - b))


def _cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _run_once(ctx_queue, c, path):
    state = _make_runner(ctx_queue.context, c, path)
    state["run"](ctx_queue).wait()
    return state, _read_output(state, ctx_queue)


ALL_CASES = CORR_CASES + BENCH_CASES


class TestGgufGemm(TestBase):
    """GGUF prefill GEMM (Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0) swept over small + prefill-sized shapes."""

    def build(self, gpu_arch) -> list[str]:
        """Compile every format / dtype combination on any available GPU."""
        device = _gpu_device()
        if device is None:
            return  # no GPU on this node -> defer to the GPU test
        ctx = cl.Context([device])
        for c in CORR_CASES + [x for x in BENCH_CASES if x["M"] == 1024]:
            _build_kernel(ctx, KERNEL_FILE, c)

    @pytest.mark.parametrize("case", CORR_CASES, ids=[_cid(c) for c in CORR_CASES])
    def test_correctness_vs_numpy(self, case, ocl_queue):
        """Kernel output vs A @ deq(W)^T computed with the NumPy oracle on the native GGUF bytes."""
        state, C = _run_once(ocl_queue, case, KERNEL_FILE)
        assert np.all(np.isfinite(C)), "non-finite values in output"
        C_ref = state["A"].astype(np.float32) @ gr.dequantize(case["fmt"], state["Wnat"]).T
        err, cos = _rel_l2(C, C_ref), _cosine(C, C_ref)
        assert err < 2e-2, f"rel-L2 error {err:.4g} too high (cosine={cos:.4f})"
        assert cos > 0.999, f"cosine {cos:.4f} too low (rel-L2={err:.4g})"

    @pytest.mark.parametrize("case", ALL_CASES, ids=[_cid(c) for c in ALL_CASES])
    def test_correctness_vs_reference(self, case, ocl_queue):
        """Kernel output vs the reference .cl over the full output (GPU-vs-GPU compare)."""
        _, C = _run_once(ocl_queue, case, KERNEL_FILE)
        _, C_ref = _run_once(ocl_queue, case, REFERENCE_FILE)
        assert np.all(np.isfinite(C)), "non-finite values in output"
        err = _rel_l2(C, C_ref)
        assert err < 5e-3, f"rel-L2 error vs reference {err:.4g} too high"

    @pytest.mark.performance
    @pytest.mark.parametrize("case", BENCH_CASES, ids=[_cid(c) for c in BENCH_CASES])
    def test_benchmark(self, case, use_reference, ocl_queue, l2_flush, request, profile_store):
        """Device time (CL profiling events) of ONE launch, preceded by an untimed L2 flush."""
        state = _make_runner(ocl_queue.context, case, REFERENCE_FILE if use_reference else KERNEL_FILE)

        ext_profiler = detect_profiler()

        run = state["run"]
        for _ in range(BENCH["warmup"]):
            run(ocl_queue)
            ocl_queue.finish()

        profiler_name = detect_profiler()
        with profiler_session(profiler_name):
            times_ms = []
            for _ in range(BENCH["iters"]):
                if BENCH["flush_l2"]:
                    _flush_enqueue(ocl_queue, l2_flush)
                ev = run(ocl_queue)
                ev.wait()
                times_ms.append((ev.profile.end - ev.profile.start) * 1e-6)

        profile_store[request.node.nodeid] = times_ms
        logging.info(
            "[Performance test] %s iters=%d  Median: %.4f ms, Min: %.4f ms, Max: %.4f ms",
            request.node.nodeid, len(times_ms), float(np.median(times_ms)), float(np.min(times_ms)), float(np.max(times_ms)),
        )


if __name__ == "__main__":
    TestGgufGemm().build(TestBase.get_machine_gpu_arch())
