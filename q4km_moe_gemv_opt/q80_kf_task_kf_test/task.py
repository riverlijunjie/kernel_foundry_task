# [USER_INSTRUCTIONS_START]
# Optimization target: fc_gguf_q8_0_sg in kernel_q80_sg.cl
#
# This is a GEMV kernel computing C[bm, n] = sum_k A[bm, k] * W[n, k]
# for a Q8_0 quantized weight matrix in the SG-transposed layout.
# Activation A is fp16 [BM, K], output C is fp32 [BM, N].
#
# Q8_0 format: block_q8_0 = { half d; int8_t qs[32]; }
#   dequant: w = q * d  (signed int8, no zero-point)
#   Eight consecutive blocks form a super-block (256 weights):
#     pqs_T: 256 bytes of SG-transposed int8 weights
#     pd_T:  16 bytes of SG-transposed fp16 scales (8 sub-block scales)
#
# The weight buffer W = [pqs_T | pd_T] is SG column-major transposed so that
# intel_sub_group_block_read8 delivers 32 weights (8 chunks x 4 bytes) to all
# 16 lanes in a single 512-byte transaction.
#
# Shapes (BM=1 token):
#   K2048_N512  : K=2048, N=512
#   K2048_N4096 : K=2048, N=4096
#   K2048_N8192 : K=2048, N=8192
#
# REQUIRED constraints (host dispatch in task.py depends on them):
#   1. Entry-point name must remain EXACTLY: fc_gguf_q8_0_sg
#   2. Argument order must remain EXACTLY: (A, W, C)
#   3. OPG (sub_group_size) must remain 16.
#   4. The kernel is built with -DKSPLIT=<n> -DK_SIZE=<K> -DN_SIZE=<N>;
#      it must produce correct results for any KSPLIT value that task.py
#      selects (1, 2, 4, 8, ...).
#   5. Dispatch is fixed by task.py:
#        global=(N_SIZE, BM, KSPLIT)   local=(OPG, 1, KSPLIT)
#
# Optimization ideas to explore:
#   - Replace the SLM-based activation distribution with a register-shuffle
#     approach using intel_sub_group_broadcast (avoids per-K-block SLM
#     scatter/gather and the sub_group_barrier), as done in the q80_se kernel.
#   - Explore half-precision accumulation (fp16 multiply-add) for this
#     bandwidth-bound workload.
#   - Reduce live register pressure so IGC can keep more weight block-reads
#     in flight; consider restructuring the 8-subblock loop.
#   - Tune the KSPLIT-dependent code paths for different shapes.
# [USER_INSTRUCTIONS_END]

import logging
from pathlib import Path

import numpy as np
import pytest
import pyopencl as cl

from kernelfoundry import TestBase  # Tests must derive from TestBase
from kernelfoundry.eval_pipeline.utils.performance import detect_profiler, profiler_session

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TASK_DIR       = Path(__file__).parent.resolve()
KERNEL_FILE    = str(TASK_DIR / "kernel_q80_sg.cl")
REFERENCE_FILE = str(TASK_DIR / "reference_q80_sg.cl")

OPG  = 16    # sub_group_size (fixed by the kernel contract)
BM   = 1     # batch size (number of activation vectors)
SEED = 1234

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
BENCH = dict(flush_mib=128, iters=1000, warmup=100, flush_l2=True)

# ---------------------------------------------------------------------------
# Test cases: (K, N) shapes
# ---------------------------------------------------------------------------
TEST_CASES = [
    {"K": 2048, "N": 512,  "label": "K2048_N512"},
    {"K": 2048, "N": 4096, "label": "K2048_N4096"},
    {"K": 2048, "N": 8192, "label": "K2048_N8192"},
]
CASE_IDS = [c["label"] for c in TEST_CASES]

# ---------------------------------------------------------------------------
# L3 flush kernel (inline — no extra file needed)
# ---------------------------------------------------------------------------
_FLUSH_SRC = r"""
__kernel void cache_flush(__global float* buf, uint n, __global float* sink) {
    uint gid    = get_global_id(0);
    uint stride = get_global_size(0);
    float acc   = 0.0f;
    for (uint i = gid; i < n; i += stride)
        acc += buf[i];
    if (gid < 4096u) sink[gid] += acc;
}
"""

# ---------------------------------------------------------------------------
# Q8_0 weight generation + SG packing
# ---------------------------------------------------------------------------
def _ascon(a):
    return np.ascontiguousarray(a)


def _pack_q80_pqs_sg(pqs, K, N, opg=OPG):
    """256-byte per-row-per-256-block Q8_0 weight section (8 sub-blocks x
    8 chunks x 4 int8 bytes), row-major -> SG column-major."""
    nbpr = K // 256
    nrg  = N // opg
    src = pqs.reshape(nrg, opg, nbpr, 8, 8, 4)   # [h,lid,bid,j,c,b]
    dst = src.transpose(0, 2, 3, 4, 1, 5)         # [h,bid,j,c,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def _pack_q80_pd_sg(pd, K, N, opg=OPG):
    """16-byte per-row-per-256-block SoA fp16 scale section (8 sub-block
    scales), row-major -> SG column-major."""
    nbpr = K // 256
    nrg  = N // opg
    src = pd.reshape(nrg, opg, nbpr, 8, 2)        # [h,lid,bid,j,b(fp16)]
    dst = src.transpose(0, 2, 3, 1, 4)            # [h,bid,j,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def _gen_q80_row_major(N, K, seed=SEED):
    """Generate a [N, K] weight matrix and quantize to GGUF Q8_0 (row-major).

    Returns:
        ref_W   : (N, K) float32 dequantized reference weights
        q_bytes : (N, K) uint8  (view of int8 quantized weights)
        d_bytes : (N, K//32 * 2) uint8  fp16 scale bytes, row-major
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, K)).astype(np.float32) * 0.05
    nblk = K // 32
    Wb = W.reshape(N, nblk, 32)
    amax = np.max(np.abs(Wb), axis=2)
    d = (amax / 127.0).astype(np.float32)
    d_safe = np.where(d == 0.0, 1.0, d)
    q = np.clip(np.round(Wb / d_safe[:, :, None]), -127, 127).astype(np.int8)
    d16 = d.astype(np.float16)
    ref_W = (q.astype(np.float32) * d16.astype(np.float32)[:, :, None]).reshape(N, K)
    q_bytes = q.view(np.uint8).reshape(N, K)
    d_bytes = d16.view(np.uint8).reshape(N, nblk * 2)
    return ref_W, q_bytes, d_bytes


def _build_q80_sg_weight(N, K, seed=SEED):
    """Build one SG-packed Q8_0 weight matrix [N, K].

    Returns:
        flat    : uint8 flat buffer = [pqs_T | pd_T]
        ref_W   : (N, K) float32 reference dequantization
    """
    assert K % 256 == 0 and N % OPG == 0
    ref_W, q_bytes, d_bytes = _gen_q80_row_major(N, K, seed=seed)
    pqs_T = _pack_q80_pqs_sg(q_bytes.reshape(-1), K, N)
    pd_T  = _pack_q80_pd_sg(d_bytes.reshape(-1), K, N)
    flat  = np.concatenate([pqs_T, pd_T])
    return flat, ref_W


def _choose_ksplit(base_groups, split_space, target=2048, maxk=8):
    import os
    force = os.environ.get("Q80_KSPLIT", "")
    if force:
        return int(force)
    if base_groups >= 512:
        return 1
    ks = max(1, round(target / max(base_groups, 1)))
    ks = min(ks, split_space, maxk)
    p = 1
    while p * 2 <= ks:
        p *= 2
    return p


def _cbuf(ctx, arr):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=_ascon(arr))


# ---------------------------------------------------------------------------
# Build helper: compile with shape + KSPLIT defines
# ---------------------------------------------------------------------------
def _build_prog(ctx, cl_file, K, N, ksplit):
    src  = Path(cl_file).read_text()
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit} -DK_SIZE={K} -DN_SIZE={N}"
    return cl.Program(ctx, src).build(options=opts)


# ---------------------------------------------------------------------------
# L3 flush helpers
# ---------------------------------------------------------------------------
def _setup_l2_flush(ctx, queue, mib):
    mf    = cl.mem_flags
    elems = (mib * 1024 * 1024) // 4
    buf   = cl.Buffer(ctx, mf.READ_WRITE, size=elems * 4)
    sink  = cl.Buffer(ctx, mf.WRITE_ONLY, size=4096 * 4)
    cl.enqueue_copy(queue, buf, np.zeros(elems, np.float32))
    prog  = cl.Program(ctx, _FLUSH_SRC).build()
    krn   = cl.Kernel(prog, "cache_flush")
    queue.finish()
    flush = dict(krn=krn, buf=buf, sink=sink, elems=elems)
    _flush_l2(queue, flush)
    return flush


def _flush_l2(queue, flush, sweeps=2):
    gsz = (min(flush["elems"], 1 << 20),)
    for _ in range(sweeps):
        flush["krn"](queue, gsz, None, flush["buf"], np.uint32(flush["elems"]), flush["sink"])
    queue.finish()


def _flush_l2_enqueue(queue, flush, sweeps=2):
    gsz = (min(flush["elems"], 1 << 20),)
    for _ in range(sweeps):
        flush["krn"](queue, gsz, None, flush["buf"], np.uint32(flush["elems"]), flush["sink"])


# ---------------------------------------------------------------------------
# Dispatch-state builder
# ---------------------------------------------------------------------------
def _make_state(ctx, case, seed=SEED, need_ref=True):
    K = case["K"]
    N = case["N"]

    w_flat, ref_W = _build_q80_sg_weight(N, K, seed=seed)

    rng = np.random.default_rng(seed ^ 0xBEEF_1234)
    x = rng.standard_normal((BM, K)).astype(np.float16)

    y_ref = None
    if need_ref:
        y_ref = (x.astype(np.float32) @ ref_W.T).astype(np.float32)  # [BM, N]

    base_groups = (N // OPG) * BM
    ksplit = _choose_ksplit(base_groups, K // 256)

    A_b = _cbuf(ctx, x)
    W_b = _cbuf(ctx, w_flat)
    C_b = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=BM * N * 4)  # fp32 output

    gsize = (N, BM, ksplit)
    lsize = (OPG, 1, ksplit)

    return dict(K=K, N=N, ksplit=ksplit, gsize=gsize, lsize=lsize,
                A_b=A_b, W_b=W_b, C_b=C_b, y_ref=y_ref)


def _run(queue, prog, st):
    krn = cl.Kernel(prog, "fc_gguf_q8_0_sg")
    krn.set_args(st["A_b"], st["W_b"], st["C_b"])

    def enq():
        return cl.enqueue_nd_range_kernel(queue, krn, st["gsize"], st["lsize"])

    ev = enq()
    ev.wait()
    got = np.empty(BM * st["N"], dtype=np.float32)
    cl.enqueue_copy(queue, got, st["C_b"])
    queue.finish()
    got = got.reshape(BM, st["N"])
    return got, enq


# ---------------------------------------------------------------------------
# OpenCL fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ocl_queue():
    """First GPU device across all platforms, profiling-enabled."""
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
    """Session-scoped L3 flush state."""
    return _setup_l2_flush(ocl_queue.context, ocl_queue, BENCH["flush_mib"])


# ---------------------------------------------------------------------------
# Tests — parametrized over Q8_0 GEMV shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", TEST_CASES, ids=CASE_IDS)
class TestQ80GemvSG(TestBase):
    """Q8_0 SG (sub-group block-read, SLM-prestaged) GEMV kernel."""

    def build(self, gpu_arch) -> list[str]:
        """Compile-check: build for each shape and verify the entry point exists."""
        device = None
        for p in cl.get_platforms():
            devs = p.get_devices(cl.device_type.GPU)
            if devs:
                device = devs[0]
                break
        if device is None:
            return
        ctx = cl.Context([device])
        for case in TEST_CASES:
            prog = _build_prog(ctx, KERNEL_FILE, case["K"], case["N"], ksplit=1)
            cl.Kernel(prog, "fc_gguf_q8_0_sg")   # raises if missing

    def test_correctness_vs_numpy(self, case, ocl_queue):
        """GPU output must match pure-Python dequantize+GEMV reference."""
        ctx  = ocl_queue.context
        st   = _make_state(ctx, case, need_ref=True)
        prog = _build_prog(ctx, KERNEL_FILE, st["K"], st["N"], st["ksplit"])
        got, _ = _run(ocl_queue, prog, st)

        y_ref   = st["y_ref"]
        max_abs = float(np.max(np.abs(y_ref.astype(np.float64) - got.astype(np.float64))))
        denom   = float(np.max(np.abs(y_ref))) + 1e-9
        rel_err = max_abs / denom
        assert rel_err < 2e-2, (
            f"{case['label']}: max relative error {rel_err:.4g} exceeds 2% threshold"
        )

    def test_correctness_vs_reference(self, case, ocl_queue):
        """GPU kernel output must match reference .cl output (GPU-vs-GPU)."""
        ctx = ocl_queue.context
        st  = _make_state(ctx, case, need_ref=False)

        kern_prog = _build_prog(ctx, KERNEL_FILE,    st["K"], st["N"], st["ksplit"])
        ref_prog  = _build_prog(ctx, REFERENCE_FILE, st["K"], st["N"], st["ksplit"])

        y_kern, _ = _run(ocl_queue, kern_prog, st)
        y_ref,  _ = _run(ocl_queue, ref_prog,  st)

        norm_ref = float(np.linalg.norm(y_ref)) + 1e-9
        rel_err  = float(np.linalg.norm(
            y_kern.astype(np.float64) - y_ref.astype(np.float64)
        )) / norm_ref
        assert rel_err < 1e-2, (
            f"{case['label']}: rel-L2 vs reference {rel_err:.4g} exceeds 1% threshold"
        )

    @pytest.mark.performance
    def test_benchmark(self, case, use_reference, ocl_queue, l2_flush, request, profile_store):
        """
        Benchmark with per-iteration L3 flush (cold-weight DRAM read).
        Results stored in profile_store for KernelFoundry speedup computation.
        """
        ctx     = ocl_queue.context
        cl_file = REFERENCE_FILE if use_reference else KERNEL_FILE
        st      = _make_state(ctx, case, need_ref=False)
        prog    = _build_prog(ctx, cl_file, st["K"], st["N"], st["ksplit"])
        _, enq  = _run(ocl_queue, prog, st)

        # Warmup — page in weights, stabilise GPU clocks.
        for _ in range(BENCH["warmup"]):
            enq()
            ocl_queue.finish()

        # Timed trials: flush L3 (untimed) then one cold kernel dispatch.
        times_ms = []
        profiler_name = detect_profiler()
        with profiler_session(profiler_name):
            for _ in range(BENCH["iters"]):
                if BENCH["flush_l2"]:
                    _flush_l2_enqueue(ocl_queue, l2_flush)
                    ocl_queue.finish()
                ev = enq()
                ev.wait()
                times_ms.append((ev.profile.end - ev.profile.start) * 1e-6)

        profile_store[request.node.nodeid] = times_ms
        logging.info(
            "[Performance] %s  flush_l2=%s  iters=%d  "
            "Median: %.4f ms  Min: %.4f ms  Max: %.4f ms",
            request.node.nodeid, BENCH["flush_l2"], len(times_ms),
            float(np.median(times_ms)), float(np.min(times_ms)), float(np.max(times_ms)),
        )


if __name__ == "__main__":
    """Optional build validation when running this file directly."""
    task = TestQ80GemvSG()
    task.build(TestBase.get_machine_gpu_arch())
