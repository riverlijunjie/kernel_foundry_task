# [USER_INSTRUCTIONS_START]
# Optimization target: kernel_dp4a_q8_0_packed.cl
#   Q8_0 coalesced byte-transposed (LISA/packed) int8-activation dp4a GEMV kernel.
#   Used by attention/GDN weight projections in Qwen3-class models on Intel Arc / Xe3.
#
# This task is fully self-contained: the OV-faithful JIT preamble (harness/ov_jit.py),
# the NumPy GGUF decode oracle (harness/gguf_ref.py), the shared .cl kernels + headers
# (kernels/), and the benchmark case list (configs/q8_0_packed.json) all live inside this
# kf_task directory. It sweeps ALL cases listed in configs/q8_0_packed.json.
#
# REQUIRED — must be preserved exactly (host code depends on these):
#   1. The KERNEL entry point and its full parameter signature (unchanged):
#        KERNEL(fc_gguf_dp4a_q8_0_packed)(
#            OPTIONAL_SHAPE_INFO_ARG
#            const __global char*        Aq,
#            const __global float*       Asc,
#            const __global uchar*       W,
#                  __global OUTPUT_TYPE* C)
#   2. The two include directives at the top of the EVOLVE block:
#        #include "include/batch_headers/common.cl"
#        #include "include/batch_headers/fetch_data.cl"
#      These pull in unroll_for / KERNEL / FUNC / FUNC_CALL macros.
#      DO NOT remove or move these includes.
#   3. FUNC() / FUNC_CALL() usage rules (CRITICAL):
#      - Every helper called via FUNC_CALL(name)(...) MUST be defined as:
#            inline <type> FUNC(name)(<params>) { ... }
#   4. The pragma: #pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#   5. The sub-group attribute: __attribute__((intel_reqd_sub_group_size(SG_SIZE)))
#   6. The NROW / Q8E_BB / GGUF_G compile-time constants (JIT-defined; keep uses).
#   7. Output must be written as:
#        C[(uint)bm * (uint)N_SIZE + n] = TO_OUTPUT_TYPE(total);
#      (the HAS_FUSED_OPS path is optional but must stay guarded by #if HAS_FUSED_OPS)
#
# Weight layout (SIZE-EXACT byte-transposed — zero expansion over native Q8_0):
#   group g (SG=16 blocks), byte i of block-lane L:
#     dst_offset = (n * GGUF_G + g) * (SG * Q8E_BB) + i * SG + L
#   Each subgroup lane reads 34 consecutive strides of 16 — a single coalesced burst.
#
# OPTIMIZATION targets (improve freely):
#   - Vectorize the inner byte-gather loop (gather 2 or 4 bytes per step using uchar2/4)
#   - Unroll the dp4a dot loop more aggressively (currently 8×4-wide)
#   - Increase NROW > 1 (more output rows per work-item; reuses the activation load)
#   - Overlap the byte-gather and dp4a computation (software pipelining)
#   - Exploit sub-group shuffles to share loaded activation across lanes
# [USER_INSTRUCTIONS_END]

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import pyopencl as cl

from kernelfoundry import TestBase  # Tests must derive from TestBase

# --------------------------------------------------------------------------------------
# Self-contained layout: everything the task needs lives under this kf_task directory.
# --------------------------------------------------------------------------------------
TASK_DIR = Path(__file__).parent.resolve()
HARNESS_DIR = TASK_DIR / "harness"
KDIR = str(TASK_DIR / "kernels")           # shared .cl + include/ headers live here
CONFIG_JSON = TASK_DIR / "configs" / "q8_0_packed.json"
sys.path.insert(0, str(HARNESS_DIR))

import ov_jit   # noqa: E402
import gguf_ref as gr  # noqa: E402

# --------------------------------------------------------------------------------------
# Test cases — read straight from configs/q8_0_packed.json (the single source of truth).
# Every case is the Q8_0 coalesced byte-transposed dp4a GEMV at a different (K, N).
# --------------------------------------------------------------------------------------
FORMAT   = "gguf_q8_0"
SEED     = 1234
SG       = ov_jit.GGUF_GEMV_SG_SIZE
G        = gr.GGUF_REQUANT_GROUP    # 32
_NP_OUT  = {"f16": np.float16, "f32": np.float32}

KERNEL_FILE    = "kernel_dp4a_q8_0_packed.cl"
REFERENCE_FILE = "reference_dp4a_q8_0_packed.cl"


def _load_cases():
    """Load the benchmark cases + bench settings from configs/q8_0_packed.json."""
    with open(CONFIG_JSON) as f:
        cfg = json.load(f)
    cases = []
    for c in cfg["cases"]:
        assert c["kernel"] == "dp4a_q8_0_packed" and c["format"] == FORMAT, c
        cases.append(dict(K=int(c["K"]), N=int(c["N"]),
                          M=int(c.get("M", 1)), out_dt=c.get("out_dt", "f16")))
    bench = dict(
        flush_mib=int(cfg.get("flush_mib", 128)),
        iters=int(cfg.get("iters", 200)),
        warmup=int(cfg.get("warmup", 15)),
        flush_l2=bool(cfg.get("flush_l2", True)),
    )
    return cases, bench


CASES, BENCH = _load_cases()
CASE_IDS = ["K%d_N%d" % (c["K"], c["N"]) for c in CASES]


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


@pytest.fixture(scope="session")
def l2_flush(ocl_queue):
    """Session-scoped L2 flush state: created ONCE at session start, before any test runs.

    Creating it early ensures:
    - The 128 MiB scratch buffer is page-committed (no lazy-fault overhead during timed runs)
    - The flush kernel instruction cache is warm
    - The GPU has been active and is in its operating P-state when test_benchmark runs
    All benchmark tests share this single flush state."""
    return _setup_l2_flush(ocl_queue.context, ocl_queue, BENCH["flush_mib"])


def _build(ctx, fname_or_path, entry_point, consts):
    """Prepend the OV-faithful JIT preamble to a .cl source and build the entry point."""
    path = fname_or_path if os.path.isabs(fname_or_path) else os.path.join(KDIR, fname_or_path)
    with open(path) as f:
        body = f.read()
    src = ov_jit.build_preamble(entry_point, consts) + "\n" + body
    opts = ["-I", KDIR] + ov_jit.OV_BUILD_OPTIONS
    prog = cl.Program(ctx, src).build(options=opts)
    return cl.Kernel(prog, entry_point)


def _repack_q8_0(Wbytes, N, K):
    """Size-exact byte-transpose of a Q8_0 weight (matches RepackGGUFWeightsQ6K Q8_0 branch
    and fc_gguf_dp4a_q8_0_packed.cl):
        group = SG (16) blocks; byte i of block-lane L stored at
        (n*GGUF_G + g)*SG*bb + i*SG + L.
    Returns (packed_bytes [N, GGUF_G*SG*bb], GGUF_G).  Zero byte expansion."""
    be, bb = gr.GGUF_BLOCK_GEOM[FORMAT]       # 32, 34
    bpr = K // be
    gG = (bpr + SG - 1) // SG
    B = Wbytes.reshape(N, bpr, bb)
    out = np.zeros((N, gG * SG * bb), np.uint8)
    lane = np.arange(bpr) % SG
    grp  = np.arange(bpr) // SG
    for i in range(bb):
        out[:, grp * (SG * bb) + i * SG + lane] = B[:, :, i]
    return out, gG


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


def _read_output(state, queue):
    cl.enqueue_copy(queue, state["C"], state["d_C"])
    queue.finish()
    return state["C"].astype(np.float32)


# --------------------------------------------------------------------------------------
# Build the prequant + dp4a pipeline for one (K, N, M, out_dt) case. The weight is
# byte-transposed (packed LISA layout) before upload. Returns a state dict whose
# state["run_dp4a"] dispatches ONLY the dp4a kernel (the evolve target).
# --------------------------------------------------------------------------------------
def _make_dp4a_runner(ctx, K, N, M, out_dt, dp4a_path):
    np_out = _NP_OUT[out_dt]
    be, bb = gr.GGUF_BLOCK_GEOM[FORMAT]       # 32, 34

    Wbytes = gr.gen_weight_bytes(FORMAT, N, K, seed=SEED)    # [N, bpr*bb] uint8
    Wp, _  = _repack_q8_0(Wbytes, N, K)                      # byte-transposed layout
    Wp = np.ascontiguousarray(Wp)
    rng = np.random.default_rng(SEED + 1)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)

    mf = cl.mem_flags
    d_A   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(A))
    d_W   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Wp)
    d_Aq  = cl.Buffer(ctx, mf.READ_WRITE, size=M * K)                  # int8 [M,K]
    d_Asc = cl.Buffer(ctx, mf.READ_WRITE, size=M * (K // G) * 4)       # f32  [M,K/32]
    C     = np.zeros((M, N), np_out)
    d_C   = cl.Buffer(ctx, mf.WRITE_ONLY, size=C.nbytes)

    pq_ep, pq_consts    = ov_jit.prequant_jit(FORMAT, K, N, in0_dt="f16")
    pq = _build(ctx, "fc_gguf_prequant.cl", pq_ep, pq_consts)
    dp_ep, dp_consts, _ = ov_jit.dp4a_q8_0_packed_jit(K, N, out_dt=out_dt)
    dp = _build(ctx, dp4a_path, dp_ep, dp_consts)

    pq_g = (K // G, M, 1);  pq_l = (1, 1, 1)
    dp_g = (N * SG,  M, 1); dp_l = (SG, 1, 1)   # NROW=1

    def run_prequant(queue):
        return pq(queue, pq_g, pq_l, d_A, d_Aq, d_Asc)

    def run_dp4a(queue):
        return dp(queue, dp_g, dp_l, d_Aq, d_Asc, d_W, d_C)

    return dict(
        K=K, N=N, M=M, out_dt=out_dt,
        Wbytes=Wbytes, A=A, C=C, d_Aq=d_Aq, d_Asc=d_Asc, d_C=d_C,
        run_prequant=run_prequant, run_dp4a=run_dp4a,
    )


# --------------------------------------------------------------------------------------
# L2 cache flush (mirrors bench.py Gpu.setup_flush / flush_l2). Sweeping a scratch buffer
# several times larger than L2 evicts the previous launch's weight, so the next timed dp4a
# re-reads its weight COLD from DRAM — modelling real inference where OTHER kernels run
# between two invocations of this kernel and pollute the cache (never a back-to-back
# L2-resident replay).
# --------------------------------------------------------------------------------------
def _setup_l2_flush(ctx, queue, mib):
    """Mirrors bench.py Gpu.setup_flush exactly:
    1. Allocate scratch (READ_WRITE) + sink (WRITE_ONLY, 16 KiB).
    2. Pre-fill scratch with zeros via host-to-device DMA → commits all GPU pages NOW
       (no lazy-allocation page faults on the first timed flush sweep).
    3. Build the cache_flush kernel.
    4. queue.finish() — everything is resident before any warmup/benchmark starts.
    5. Run one untimed pre-flush sweep to warm the flush kernel instruction cache."""
    mf = cl.mem_flags
    elems = (mib * 1024 * 1024) // 4
    buf  = cl.Buffer(ctx, mf.READ_WRITE, size=elems * 4)
    sink = cl.Buffer(ctx, mf.WRITE_ONLY, size=4096 * 4)
    zeros = np.zeros(elems, np.float32)
    cl.enqueue_copy(queue, buf, zeros)           # commit all GPU pages
    with open(os.path.join(KDIR, "cache_flush.cl")) as f:
        src = f.read()
    prog = cl.Program(ctx, src).build(options=ov_jit.OV_BUILD_OPTIONS)
    krn = cl.Kernel(prog, "cache_flush")
    queue.finish()                               # ensure pages + kernel are fully ready
    flush = dict(krn=krn, buf=buf, sink=sink, elems=elems)
    _flush_l2(queue, flush)                      # one untimed pre-run to warm i-cache
    return flush


def _flush_l2(queue, flush, sweeps=2):
    """Full synchronised flush — used during setup (pre-run) and warmup phases."""
    gsz = (min(flush["elems"], 1024 * 1024),)
    for _ in range(sweeps):
        flush["krn"](queue, gsz, None, flush["buf"], np.uint32(flush["elems"]), flush["sink"])
    queue.finish()


def _flush_l2_enqueue(queue, flush, sweeps=2):
    """Enqueue-only flush — used inside the timed iteration loop.

    Does NOT call queue.finish(). Since the OpenCL queue is in-order, the dp4a kernel
    submitted immediately after will still wait for these flush kernels to finish before
    starting. Skipping queue.finish() keeps the GPU continuously active (no CPU-GPU sync
    point), preventing the Intel Arc power manager from dropping the clock frequency during
    the brief CPU overhead between flush completion and dp4a submission.
    The CL profiling event on the dp4a still measures only its own device execution time."""
    gsz = (min(flush["elems"], 1024 * 1024),)
    for _ in range(sweeps):
        flush["krn"](queue, gsz, None, flush["buf"], np.uint32(flush["elems"]), flush["sink"])


def _numpy_reference(state):
    """(int8-dequant activation) @ deq(W)^T over all N rows, using the ORIGINAL (non-packed)
    weight bytes — this is the correctness oracle."""
    K, N, M = state["K"], state["N"], state["M"]
    Aq  = np.zeros(M * K, np.int8)
    Asc = np.zeros(M * (K // G), np.float32)
    cl.enqueue_copy(_QUEUE[0], Aq,  state["d_Aq"])
    cl.enqueue_copy(_QUEUE[0], Asc, state["d_Asc"])
    _QUEUE[0].finish()
    Aq  = Aq.reshape(M, K)
    Asc = Asc.reshape(M, K // G)
    Aq_deq = (Aq.astype(np.float32).reshape(M, K // G, G) * Asc[:, :, None]).reshape(M, K)
    Wf = gr.dequantize(FORMAT, state["Wbytes"])   # [N, K] f32
    return (Aq_deq @ Wf.T).astype(np.float32)


_QUEUE = [None]  # tiny holder so the oracle helper can reach the session queue


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
class TestGgufDp4aQ8_0Packed(TestBase):
    """Q8_0 coalesced byte-transposed dp4a GEMV, swept over configs/q8_0_packed.json."""

    def build(self, gpu_arch) -> list[str]:
        """Best-effort compile check for every case on any available GPU device."""
        device = None
        for platform in cl.get_platforms():
            devs = platform.get_devices(cl.device_type.GPU)
            if devs:
                device = devs[0]
                break
        if device is None:
            return  # no GPU on this node -> defer to the GPU test
        ctx = cl.Context([device])
        for c in CASES:
            dp_ep, dp_consts, _ = ov_jit.dp4a_q8_0_packed_jit(c["K"], c["N"], out_dt=c["out_dt"])
            _build(ctx, os.path.join(str(TASK_DIR), KERNEL_FILE), dp_ep, dp_consts)

    def test_correctness_vs_numpy(self, case, ocl_queue):
        """dp4a output vs the NumPy decode oracle (all N output rows)."""
        _QUEUE[0] = ocl_queue
        state = _make_dp4a_runner(ocl_queue.context, case["K"], case["N"], case["M"],
                                  case["out_dt"], os.path.join(str(TASK_DIR), KERNEL_FILE))
        state["run_prequant"](ocl_queue).wait()
        state["run_dp4a"](ocl_queue).wait()
        C = _read_output(state, ocl_queue)
        C_ref = _numpy_reference(state)
        err = _rel_l2(C, C_ref)
        cos = _cosine(C, C_ref)
        assert err < 2e-2, f"rel-L2 error {err:.4g} too high (cosine={cos:.4f})"
        assert cos > 0.999, f"cosine {cos:.4f} too low (rel-L2={err:.4g})"

    def test_correctness_vs_reference(self, case, ocl_queue):
        """dp4a output vs the reference .cl over the full N rows (GPU-vs-GPU compare)."""
        kern_state = _make_dp4a_runner(ocl_queue.context, case["K"], case["N"], case["M"],
                                       case["out_dt"], os.path.join(str(TASK_DIR), KERNEL_FILE))
        kern_state["run_prequant"](ocl_queue).wait()
        kern_state["run_dp4a"](ocl_queue).wait()
        C = _read_output(kern_state, ocl_queue).copy()

        ref_state = _make_dp4a_runner(ocl_queue.context, case["K"], case["N"], case["M"],
                                      case["out_dt"], os.path.join(str(TASK_DIR), REFERENCE_FILE))
        ref_state["run_prequant"](ocl_queue).wait()
        ref_state["run_dp4a"](ocl_queue).wait()
        C_ref = _read_output(ref_state, ocl_queue)
        err = _rel_l2(C, C_ref)
        assert err < 1e-2, f"rel-L2 error vs reference {err:.4g} too high"

    @pytest.mark.performance
    def test_benchmark(self, case, use_reference, ocl_queue, l2_flush, request, profile_store):
        """Times ONLY the dp4a decode kernel with an L2 flush before EACH timed launch.

        This models real inference: between two invocations of this kernel, OTHER kernels
        run and evict its weight from L2, so every launch re-reads its weight COLD from DRAM
        (never a back-to-back L2-resident replay). Timing uses CL profiling events so only
        the dp4a device time is recorded — the flush runs untimed. Mirrors bench.py's
        flush_l2 + time_event path; results go to KF's profile_store for scoring.
        l2_flush is session-scoped: pages are committed and kernel is warm before this runs."""
        cl_file = REFERENCE_FILE if use_reference else KERNEL_FILE
        state = _make_dp4a_runner(ocl_queue.context, case["K"], case["N"], case["M"],
                                  case["out_dt"], os.path.join(str(TASK_DIR), cl_file))
        state["run_prequant"](ocl_queue).wait()
        ocl_queue.finish()
        run_dp4a = state["run_dp4a"]

        # Warmup (untimed) to page in the weight / stabilise clocks.
        for _ in range(BENCH["warmup"]):
            run_dp4a(ocl_queue)
            ocl_queue.finish()

        # Timed trials: flush L2 (untimed) then time ONE cold dp4a launch via CL events.
        # _flush_l2_enqueue omits queue.finish() so the GPU stays continuously active and
        # the Intel Arc power manager cannot drop frequency during the CPU sync overhead.
        times_ms = []
        for _ in range(BENCH["iters"]):
            if BENCH["flush_l2"]:
                _flush_l2_enqueue(ocl_queue, l2_flush)
            ev = run_dp4a(ocl_queue)
            ev.wait()
            times_ms.append((ev.profile.end - ev.profile.start) * 1e-6)  # ns -> ms

        profile_store[request.node.nodeid] = times_ms
        logging.info(
            "[Performance test] %s flush_l2=%s iters=%d  Median: %.4f ms, Min: %.4f ms, Max: %.4f ms",
            request.node.nodeid, BENCH["flush_l2"], len(times_ms),
            float(np.median(times_ms)), float(np.min(times_ms)), float(np.max(times_ms)),
        )


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestGgufDp4aQ8_0Packed()
    task.build(TestBase.get_machine_gpu_arch())
