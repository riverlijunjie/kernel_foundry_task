# [USER_INSTRUCTIONS_START]
# Optimization target: shared_gate_up_q8_0 and shared_down_merge_q8_0 in
# kernel_moe_q80_shared_sg.cl
#
# These are the two GEMV kernels that implement one "shared expert" MoE-FFN
# block for GGML Q8_0 quantized weights, using the bandwidth-optimal "SG"
# (sub-group block-read, transposed weight layout) scheme with SLM-free
# register-shuffle activation redistribution (no per-K-block SLM stage or
# barrier):
#   shared_gate_up_q8_0    : fused gate-proj + up-proj GEMV + SiLU(gate)*up
#   shared_down_merge_q8_0 : down-proj GEMV, scaled by a per-token scalar
#                            gate (optionally accumulated into outputs)
#
# Unlike the routed Q4_K/Q5_K/Q6_K MoE kernels elsewhere in this workspace,
# there is only ONE (shared) expert weight matrix per projection -- no
# expert-pool / routing-index indirection -- and the down-projection is
# scaled by a single per-token scalar ("shared_gate") instead of a
# weighted sum over active experts.
#
# Shapes (token_len=4):
#   GateUp_H2048_I512   : gate_up, hidden_size=2048, intermediate_size=512
#   DownMerge_I512_H2048: down,    intermediate_size=512, hidden_size=2048
#
# REQUIRED constraints (host dispatch in task.py depends on them):
#   1. Entry-point names and argument order must remain EXACTLY:
#        shared_gate_up_q8_0(gate_weights, up_weights, hidden_states,
#                            gate_up_out, hidden_size, intermediate_size)
#        shared_down_merge_q8_0(gate_up_in, down_weights, shared_gate,
#                                final_out, intermediate_size, hidden_size,
#                                is_acc)
#   2. OPG (sub_group_size) must remain 16.
#   3. The kernel is built with -DKSPLIT=<n> for varying n (1,2,4,8,...);
#      it must produce correct results for any such n.
#   4. Dispatch is fixed by task.py:
#        shared_gate_up_q8_0:    global=(intermediate_size,KSPLIT,token_len)
#                                local =(16,KSPLIT,1)
#        shared_down_merge_q8_0: global=(hidden_size,KSPLIT,token_len)
#                                local =(16,KSPLIT,1)
#
# Optimization ideas to explore:
#   - Reduce the shuffle count in sh_q8_0_bcast_subblock (currently 16
#     sub_group_broadcast calls per 32-activation sub-block) without
#     regressing correctness.
#   - Explore the Q80_HALF_MUL half-precision multiply-accumulate path
#     already stubbed in the kernel for this bandwidth-bound workload.
#   - Reuse the broadcast activation registers across BOTH the gate and up
#     projections in shared_gate_up_q8_0 (already done in the seed kernel);
#     keep exploring ways to cut live registers further so IGC can keep more
#     weight block-reads in flight per matrix.
#   - Tune the KSPLIT-dependent code paths without breaking any KSPLIT value.
#   - Wide sub-group block reads for weight bytes and the packed fp16 scale
#     record (avoid many narrow global loads); recover per-lane scalar
#     fields from one wide read via shuffles instead of separate loads.
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
KERNEL_FILE    = str(TASK_DIR / "kernel_moe_q80_shared_sg.cl")
REFERENCE_FILE = str(TASK_DIR / "reference_moe_q80_shared_sg.cl")

OPG = 16  # sub_group_size (fixed by the kernel contract)

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
SEED = 1234
BENCH = dict(flush_mib=256, iters=300, warmup=30, flush_l2=True)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    {"kind": "shared_gate_up", "hidden_size": 2048, "intermediate_size": 512,
     "token_len": 4, "label": "GateUp_H2048_I512"},
    {"kind": "shared_down_merge", "intermediate_size": 512, "hidden_size": 2048,
     "token_len": 4, "label": "DownMerge_I512_H2048"},
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
# Q8_0 shared-expert weight generation + SG packing (row-major GGML Q8_0 ->
# SG column-major, identical scheme to the routed Q4_K/Q5_K/Q6_K packers
# elsewhere in this workspace, but with byte-per-weight int8 chunks and no
# nibble-splitting -- see the header comment in
# moe_gemv_q80_shared_sg.cl / kernel_moe_q80_shared_sg.cl for the exact
# byte-offset derivation).
#
#   pqs section (256 B/row/256-block): 8 sub-blocks (32 weights) x 8 chunks
#     (4 int8 weights) x OPG lanes x 4 bytes.
#   pd  section (16 B/row/256-block, SoA): 8 sub-block fp16 scales x OPG
#     lanes x 2 bytes.
# ---------------------------------------------------------------------------
def _ascon(a):
    return np.ascontiguousarray(a)


def _pack_q80_pqs_sg(pqs, K, N, opg=OPG):
    """256-byte per-row-per-256-block Q8_0 weight section (8 sub-blocks x 8
    chunks x 4 int8 bytes), row-major -> SG column-major."""
    nbpr = K // 256
    nrg  = N // opg
    src = pqs.reshape(nrg, opg, nbpr, 8, 8, 4)          # [h,lid,bid,j,c,b]
    dst = src.transpose(0, 2, 3, 4, 1, 5)               # [h,bid,j,c,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def _pack_q80_pd_sg(pd, K, N, opg=OPG):
    """16-byte per-row-per-256-block SoA fp16 scale section (8 sub-block
    scales), row-major -> SG column-major."""
    nbpr = K // 256
    nrg  = N // opg
    src = pd.reshape(nrg, opg, nbpr, 8, 2)              # [h,lid,bid,j,b(fp16)]
    dst = src.transpose(0, 2, 3, 1, 4)                  # [h,bid,j,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def _gen_q80_row_major(N, K, seed=SEED):
    """Generate a [N, K] float weight matrix and its raw, row-major GGML
    Q8_0 quantization (blocks of 32 int8 weights + one fp32-derived fp16
    scale per block).  Returns (ref_W[N,K] float32, q_bytes[N,K] uint8,
    d_bytes[N, K//32 * 2] uint8 row-major fp16 scale bytes)."""
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((N, K)).astype(np.float32) * 0.05)
    nblk = K // 32
    Wb = W.reshape(N, nblk, 32)
    amax = np.max(np.abs(Wb), axis=2)
    d = (amax / 127.0).astype(np.float32)
    d_safe = np.where(d == 0.0, 1.0, d)
    q = np.round(Wb / d_safe[:, :, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    d16 = d.astype(np.float16)
    ref_W = (q.astype(np.float32) * d16.astype(np.float32)[:, :, None]).reshape(N, K)
    q_bytes = q.view(np.uint8).reshape(N, K)
    d_bytes = d16.view(np.uint8).reshape(N, nblk * 2)
    return ref_W, q_bytes, d_bytes


def _build_q80_shared_weight(N, K, seed=SEED):
    """Build one SG-packed Q8_0 weight matrix [N, K] (a single "shared
    expert" weight, no expert pool/index indirection) and its float32
    reference dequantization."""
    assert K % 256 == 0 and N % OPG == 0
    ref_W, q_bytes, d_bytes = _gen_q80_row_major(N, K, seed=seed)
    pqs_T = _pack_q80_pqs_sg(q_bytes.reshape(-1), K, N)
    pd_T  = _pack_q80_pd_sg(d_bytes.reshape(-1), K, N)
    flat = np.concatenate([pqs_T, pd_T])
    return flat, ref_W


def _silu(x):
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    sig = np.empty_like(x)
    sig[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    sig[~pos] = ex / (1.0 + ex)
    return (x * sig).astype(np.float32)


def _choose_ksplit(base_groups, split_space, target=2048, maxk=8):
    def _pow2_floor(x):
        p = 1
        while p * 2 <= x:
            p *= 2
        return p
    import os
    force = os.environ.get("MOE_KSPLIT", "")
    if force:
        return int(force)
    if base_groups >= 512:
        return 1
    ks = max(1, round(target / max(base_groups, 1)))
    ks = min(ks, split_space, maxk)
    return _pow2_floor(ks)


def _cbuf(ctx, arr):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=np.ascontiguousarray(arr))


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------
def _build_prog(ctx, cl_file, ksplit):
    src = Path(cl_file).read_text()
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit}"
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
# Dispatch-state builders
# ---------------------------------------------------------------------------
def _make_gate_up_state(ctx, case, seed=SEED, need_ref=True):
    hidden_size       = case["hidden_size"]
    intermediate_size = case["intermediate_size"]
    token_len         = case["token_len"]

    gate_flat, gate_ref = _build_q80_shared_weight(intermediate_size, hidden_size, seed=seed)
    up_flat,   up_ref   = _build_q80_shared_weight(intermediate_size, hidden_size, seed=seed + 999)

    rng = np.random.default_rng(seed ^ 0xBEEF_1234)
    x = rng.standard_normal((token_len, hidden_size)).astype(np.float16)

    y_ref = None
    if need_ref:
        x_f32 = x.astype(np.float32)
        y_ref = np.zeros((token_len, intermediate_size), dtype=np.float32)
        for t in range(token_len):
            gate = gate_ref @ x_f32[t]
            up   = up_ref @ x_f32[t]
            y_ref[t] = _silu(gate) * up

    base_groups = (intermediate_size // OPG) * token_len
    ksplit = _choose_ksplit(base_groups, hidden_size // 256)

    x_b    = _cbuf(ctx, x)
    gate_b = _cbuf(ctx, gate_flat)
    up_b   = _cbuf(ctx, up_flat)
    out_b  = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=token_len * intermediate_size * 2)  # half

    gsize = (intermediate_size, ksplit, token_len)
    lsize = (OPG, ksplit, 1)

    return dict(hidden_size=hidden_size, intermediate_size=intermediate_size,
                token_len=token_len, ksplit=ksplit, gsize=gsize, lsize=lsize,
                x_b=x_b, gate_b=gate_b, up_b=up_b, out_b=out_b, y_ref=y_ref)


def _run_gate_up(queue, prog, st):
    krn = cl.Kernel(prog, "shared_gate_up_q8_0")
    krn.set_args(st["gate_b"], st["up_b"], st["x_b"], st["out_b"],
                 np.uint32(st["hidden_size"]), np.uint32(st["intermediate_size"]))

    def enq():
        return cl.enqueue_nd_range_kernel(queue, krn, st["gsize"], st["lsize"])

    ev = enq()
    ev.wait()
    got = np.empty(st["token_len"] * st["intermediate_size"], dtype=np.float16)
    cl.enqueue_copy(queue, got, st["out_b"])
    queue.finish()
    got = got.astype(np.float32).reshape(st["token_len"], st["intermediate_size"])
    return got, enq


def _make_down_merge_state(ctx, case, seed=SEED, need_ref=True, is_acc=False):
    intermediate_size = case["intermediate_size"]
    hidden_size       = case["hidden_size"]
    token_len         = case["token_len"]

    down_flat, down_ref = _build_q80_shared_weight(hidden_size, intermediate_size, seed=seed)

    rng = np.random.default_rng(seed ^ 0xBEEF_1234)
    gate_up_in   = rng.standard_normal((token_len, intermediate_size)).astype(np.float16)
    shared_gate  = rng.uniform(0.1, 1.0, token_len).astype(np.float16)
    prev_out     = rng.standard_normal((token_len, hidden_size)).astype(np.float32)

    y_ref = None
    if need_ref:
        gate_up_f32 = gate_up_in.astype(np.float32)
        shared_gate_f32 = shared_gate.astype(np.float32)
        y_ref = np.zeros((token_len, hidden_size), dtype=np.float32)
        for t in range(token_len):
            acc = shared_gate_f32[t] * (down_ref @ gate_up_f32[t])
            if is_acc:
                acc = acc + prev_out[t]
            y_ref[t] = acc.astype(np.float32)

    base_groups = (hidden_size // OPG) * token_len
    ksplit = _choose_ksplit(base_groups, intermediate_size // 256)

    in_b   = _cbuf(ctx, gate_up_in)
    down_b = _cbuf(ctx, down_flat)
    gate_b = _cbuf(ctx, shared_gate)
    out_b  = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=token_len * hidden_size * 2)  # half
    if is_acc:
        cl.enqueue_copy(cl.CommandQueue(ctx), out_b, prev_out.astype(np.float16).ravel())

    gsize = (hidden_size, ksplit, token_len)
    lsize = (OPG, ksplit, 1)

    return dict(intermediate_size=intermediate_size, hidden_size=hidden_size,
                token_len=token_len, ksplit=ksplit, gsize=gsize, lsize=lsize,
                in_b=in_b, down_b=down_b, gate_b=gate_b, out_b=out_b,
                prev_out=prev_out, is_acc=is_acc, y_ref=y_ref)


def _run_down_merge(queue, prog, st):
    if st["is_acc"]:
        cl.enqueue_copy(queue, st["out_b"], st["prev_out"].astype(np.float16).ravel())
        queue.finish()

    krn = cl.Kernel(prog, "shared_down_merge_q8_0")
    krn.set_args(st["in_b"], st["down_b"], st["gate_b"], st["out_b"],
                 np.uint32(st["intermediate_size"]), np.uint32(st["hidden_size"]),
                 np.int32(1 if st["is_acc"] else 0))

    def enq():
        return cl.enqueue_nd_range_kernel(queue, krn, st["gsize"], st["lsize"])

    ev = enq()
    ev.wait()
    got = np.empty(st["token_len"] * st["hidden_size"], dtype=np.float16)
    cl.enqueue_copy(queue, got, st["out_b"])
    queue.finish()
    got = got.astype(np.float32).reshape(st["token_len"], st["hidden_size"])
    return got, enq


# ---------------------------------------------------------------------------
# OpenCL fixtures
# ---------------------------------------------------------------------------
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
    """Session-scoped L3 flush state. Created once before any benchmark runs."""
    return _setup_l2_flush(ocl_queue.context, ocl_queue, BENCH["flush_mib"])


# ---------------------------------------------------------------------------
# Tests — parametrized over both shared-expert Q8_0 GEMV kernels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", TEST_CASES, ids=CASE_IDS)
class TestMoeGemvQ80SharedSG(TestBase):
    """Shared-expert Q8_0 'SG' (sub-group block-read, SLM-free) GEMV kernels, gate_up + down_merge."""

    def build(self, gpu_arch) -> list[str]:
        """Compile-check: build the kernel file and verify both entry points exist."""
        device = None
        for p in cl.get_platforms():
            devs = p.get_devices(cl.device_type.GPU)
            if devs:
                device = devs[0]
                break
        if device is None:
            return
        ctx  = cl.Context([device])
        prog = _build_prog(ctx, KERNEL_FILE, ksplit=1)
        cl.Kernel(prog, "shared_gate_up_q8_0")     # raises if missing
        cl.Kernel(prog, "shared_down_merge_q8_0")  # raises if missing

    def test_correctness_vs_numpy(self, case, ocl_queue):
        """GPU output must match pure-Python dequantize+GEMV(+SiLU) reference."""
        ctx = ocl_queue.context
        if case["kind"] == "shared_gate_up":
            st   = _make_gate_up_state(ctx, case, need_ref=True)
            prog = _build_prog(ctx, KERNEL_FILE, st["ksplit"])
            got, _ = _run_gate_up(ocl_queue, prog, st)
        else:
            st   = _make_down_merge_state(ctx, case, need_ref=True)
            prog = _build_prog(ctx, KERNEL_FILE, st["ksplit"])
            got, _ = _run_down_merge(ocl_queue, prog, st)

        y_ref = st["y_ref"]
        max_abs = float(np.max(np.abs(y_ref.astype(np.float64) - got.astype(np.float64))))
        denom   = float(np.max(np.abs(y_ref))) + 1e-9
        rel_err = max_abs / denom
        assert rel_err < 2e-2, (
            f"{case['label']}: max relative error {rel_err:.4g} exceeds 2% threshold"
        )

    def test_correctness_vs_reference(self, case, ocl_queue):
        """GPU kernel output must match reference .cl output (GPU-vs-GPU)."""
        ctx = ocl_queue.context
        if case["kind"] == "shared_gate_up":
            st        = _make_gate_up_state(ctx, case, need_ref=False)
            kern_prog = _build_prog(ctx, KERNEL_FILE, st["ksplit"])
            ref_prog  = _build_prog(ctx, REFERENCE_FILE, st["ksplit"])
            y_kern, _ = _run_gate_up(ocl_queue, kern_prog, st)
            y_ref,  _ = _run_gate_up(ocl_queue, ref_prog,  st)
        else:
            st        = _make_down_merge_state(ctx, case, need_ref=False)
            kern_prog = _build_prog(ctx, KERNEL_FILE, st["ksplit"])
            ref_prog  = _build_prog(ctx, REFERENCE_FILE, st["ksplit"])
            y_kern, _ = _run_down_merge(ocl_queue, kern_prog, st)
            y_ref,  _ = _run_down_merge(ocl_queue, ref_prog,  st)

        norm_ref = float(np.linalg.norm(y_ref)) + 1e-9
        rel_err  = float(np.linalg.norm(y_kern.astype(np.float64) - y_ref.astype(np.float64))) / norm_ref
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

        if case["kind"] == "shared_gate_up":
            st   = _make_gate_up_state(ctx, case, need_ref=False)
            prog = _build_prog(ctx, cl_file, st["ksplit"])
            _, enq = _run_gate_up(ocl_queue, prog, st)
        else:
            st   = _make_down_merge_state(ctx, case, need_ref=False)
            prog = _build_prog(ctx, cl_file, st["ksplit"])
            _, enq = _run_down_merge(ocl_queue, prog, st)

        # Warmup — page in weights, stabilise GPU clocks
        for _ in range(BENCH["warmup"]):
            enq()
            ocl_queue.finish()

        # Timed trials: flush L3 (untimed) then one cold kernel dispatch via CL events.
        times_ms = []
        profiler_name = detect_profiler()
        with profiler_session(profiler_name):
            for _ in range(BENCH["iters"]):
                if BENCH["flush_l2"]:
                    _flush_l2_enqueue(ocl_queue, l2_flush)
                    ocl_queue.finish()
                ev = enq()
                ev.wait()
                times_ms.append((ev.profile.end - ev.profile.start) * 1e-6)  # ns -> ms

        profile_store[request.node.nodeid] = times_ms
        logging.info(
            "[Performance] %s  flush_l2=%s  iters=%d  "
            "Median: %.4f ms  Min: %.4f ms  Max: %.4f ms",
            request.node.nodeid, BENCH["flush_l2"], len(times_ms),
            float(np.median(times_ms)), float(np.min(times_ms)), float(np.max(times_ms)),
        )


if __name__ == "__main__":
    """Optional build validation when running this file directly."""
    task = TestMoeGemvQ80SharedSG()
    task.build(TestBase.get_machine_gpu_arch())
