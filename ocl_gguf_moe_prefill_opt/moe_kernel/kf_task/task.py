import sys
from pathlib import Path

import numpy as np
import pytest
import pyopencl as cl

from kernelfoundry import TestBase  # Tests must derive from TestBase

# --------------------------------------------------------------------------------------
# Reuse the standalone harness (OV-faithful JIT preamble + NumPy GGUF-transcode oracle)
# that ships next to this task. These modules are the single source of truth for how
# OpenVINO builds/dispatches the extracted MoE GGUF weight-transcode kernel, so a kernel
# optimized here transfers 1:1 back into OV. Mirrors the sibling moe_kernel/kf_task layout.
# --------------------------------------------------------------------------------------
TASK_DIR = Path(__file__).parent.resolve()
HARNESS_DIR = TASK_DIR / "harness"
KDIR = str(TASK_DIR / "kernels")  # include/ headers (common.cl, batch_headers) live here
sys.path.insert(0, str(HARNESS_DIR))

import moe_jit  # noqa: E402
import moe_ref as mr  # noqa: E402

# --------------------------------------------------------------------------------------
# Target problem: moe_gguf_transcode_prefill_E256
#   GGUF MoE weight transcode (prefill / compute-bound path, token > 32). Decodes one
#   projection's raw GGUF block-quantised expert weights W[E, N, K] into the oneDNN-WOQ
#   native low-bit layout consumed by the per-expert oneDNN prefill loop:
#       * packed weight scratch WQ : i4 (TRANSCODE_TO_I4=1) or i8, physical [E, N, K]
#       * per-group f16 scale SC   : [E, K/REQUANT_GROUP, N]
#   One work-item owns one (expert e, output row n, GGUF block blk).
#   Benchmarked on the two per-projection formats that appear in Qwen3.6-35B-A3B prefill
#   (== configs/transcode.json / results/transcode_report.json):
#       gate_up projection : Q4_K -> i4  (N=512,  K=2048)
#       down    projection : Q5_K -> i8  (N=2048, K=512)
# --------------------------------------------------------------------------------------
EXPERTS = 256
SEED = 1234

# (GGUF weight format, N, K) — the two benchmarked cases (== configs/transcode.json)
CASES = [
    ("gguf_q4_k", 512, 2048),
    ("gguf_q5_k", 2048, 512),
]

VERIFY_SLOTS = 2   # expert slots to check against the (slow Python) NumPy oracle
VERIFY_ROWS = 48   # output rows per slot to check

KERNEL_FILE = "kernel_transcode.cl"
REFERENCE_FILE = "reference_transcode.cl"

GROUP = moe_jit.MOE_GGUF_REQUANT_GROUP     # 32
SGL = moe_jit.MOE_GGUF_TRANSCODE_LWS       # 16 (local x / subgroup size)


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


def _rel_l2(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a - b))


def _recon_subset(wq, sc, fmt, N, K, to_i4, group, slots, rows):
    """Reconstruct codes*scale for the first `slots` experts x `rows` rows from the packed
    low-bit scratch (WQ) + per-group f16 scale scratch (SC). Cheap GPU-vs-GPU compare basis."""
    S = min(slots, EXPERTS)
    R = min(rows, N)
    out = np.zeros((S, R, K), np.float32)
    for e in range(S):
        for n in range(R):
            codes = mr.unpack_wq_row(wq, e, n, N, K, to_i4).astype(np.float32)
            scl = mr.gather_sc_row(sc, e, n, N, K, group)
            out[e, n] = codes * np.repeat(scl, group)
    return out


# --------------------------------------------------------------------------------------
# Per-case setup: build the transcode kernel + device buffers ONCE. The full E=256 raw
# GGUF weight tensor W[E,N,K] (>> 16 MiB L2) is resident, so a single launch already
# streams every expert cold from DRAM — the transcode reads all of W each dispatch.
# `enqueue(queue)` dispatches ONLY the transcode kernel (the evolve target).
# --------------------------------------------------------------------------------------
def _setup(ctx, fmt, N, K, cl_path):
    mf = cl.mem_flags
    be, bb = moe_jit.GGUF_BLK[fmt]
    assert K % be == 0, "K must be a multiple of the GGUF block element count"
    to_i4, qmax = moe_jit.TRANSCODE_TARGET[fmt]

    # Raw GGUF expert weights [E, N, K] (per-expert contiguous, matching the kernel indexing).
    W = np.ascontiguousarray(mr.gen_expert_weights(fmt, EXPERTS, N, K, seed=SEED, scale_amp=0.02))
    wq_bytes = EXPERTS * N * ((K // 2) if to_i4 else K)   # i4 packed two-per-byte / i8 one-per-byte
    sc_elems = EXPERTS * (K // GROUP) * N                  # f16 per-group scale [E, K/32, N]

    d_W = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W)
    d_WQ = cl.Buffer(ctx, mf.WRITE_ONLY, size=wq_bytes)
    d_SC = cl.Buffer(ctx, mf.WRITE_ONLY, size=sc_elems * 2)

    ep, consts = moe_jit.transcode_jit(fmt, EXPERTS, N, K)
    krn = _build(ctx, cl_path, ep, consts)
    blocks_per_row = K // be
    n_global = ((N + SGL - 1) // SGL) * SGL
    gsz = (n_global, blocks_per_row, EXPERTS)
    lsz = (SGL, 1, 1)

    wq = np.empty(wq_bytes, np.uint8)
    sc = np.empty(sc_elems, np.float16)

    def enqueue(queue):
        return krn(queue, gsz, lsz, d_W, d_WQ, d_SC)

    def read_output(queue):
        cl.enqueue_copy(queue, wq, d_WQ)
        cl.enqueue_copy(queue, sc, d_SC)
        queue.finish()
        return wq, sc

    return dict(enqueue=enqueue, read_output=read_output,
                W=W, fmt=fmt, N=N, K=K, to_i4=to_i4, qmax=qmax)


@pytest.fixture(scope="session")
def get_state(ocl_queue, use_reference):
    """Session-cached per-case runner built from the evolve kernel (or the reference under --ref)."""
    cache = {}
    cl_path = str(TASK_DIR / (REFERENCE_FILE if use_reference else KERNEL_FILE))
    print("cl_path: ", cl_path)

    def _get(fmt, N, K):
        key = (fmt, N, K)
        if key not in cache:
            cache[key] = _setup(ocl_queue.context, fmt, N, K, cl_path)
        return cache[key]

    return _get


# --------------------------------------------------------------------------------------
# Test class
# --------------------------------------------------------------------------------------
class TestMoEGgufTranscode(TestBase):
    """GGUF MoE weight transcode, oneDNN-WOQ low-bit repack (moe_gguf_transcode_prefill_E256)."""

    def build(self, gpu_arch) -> list[str]:
        """Best-effort compile check for both cases. The kernel needs GPU-only OpenCL features
        (intel sub-groups, the OV JIT preamble), so real validation happens when the test
        compiles it on the GPU. On a GPU-less build node this is a no-op; on a GPU node it
        compiles the evolve kernel to catch syntax errors."""
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
        print("build cl_path: ", cl_path)
        for fmt, N, K in CASES:
            ep, consts = moe_jit.transcode_jit(fmt, EXPERTS, N, K)
            _build(ctx, cl_path, ep, consts)

    @pytest.mark.parametrize("fmt,N,K", CASES)
    def test_correctness_vs_numpy(self, get_state, ocl_queue, fmt, N, K):
        """Transcoded low-bit codes + f16 scales vs the NumPy GGUF-transcode oracle on a subset
        of experts/rows. Codes must match the reference quantizer within 1 LSB of the rounding
        ties, scales must match the f16-cast of the ideal per-group scale, and the reconstruction
        (codes*scale) must track the ideal decode within the group magnitude."""
        st = get_state(fmt, N, K)
        st["enqueue"](ocl_queue).wait()
        wq, sc = st["read_output"](ocl_queue)

        W = st["W"]
        to_i4, qmax = st["to_i4"], st["qmax"]
        be, bb = moe_jit.GGUF_BLK[fmt]
        row_bytes_len = (K // be) * bb
        Wr = W.reshape(EXPERTS, N, row_bytes_len)

        code_gt1 = code_total = 0
        max_code_diff = 0
        scale_relerr = 0.0
        recon_relerr = 0.0
        for e in range(min(VERIFY_SLOTS, EXPERTS)):
            for n in range(min(VERIFY_ROWS, N)):
                ref_codes, ref_sc = mr.transcode_row(fmt, Wr[e, n], K, to_i4, qmax, GROUP)
                gpu_codes = mr.unpack_wq_row(wq, e, n, N, K, to_i4)
                gpu_sc = mr.gather_sc_row(sc, e, n, N, K, GROUP)
                diff = np.abs(ref_codes - gpu_codes)
                code_gt1 += int((diff > 1).sum())
                code_total += diff.size
                max_code_diff = max(max_code_diff, int(diff.max()))
                denom = np.abs(ref_sc) + 1e-12
                scale_relerr = max(scale_relerr, float((np.abs(ref_sc - gpu_sc) / denom).max()))
                ref_val = mr.deq_row(fmt, Wr[e, n], K).astype(np.float32)
                gpu_val = gpu_codes.astype(np.float32) * np.repeat(gpu_sc, GROUP)
                mag = np.abs(ref_val.reshape(-1, GROUP)).max(axis=1) + 1e-12
                rel = np.abs((ref_val - gpu_val).reshape(-1, GROUP)) / mag[:, None]
                recon_relerr = max(recon_relerr, float(rel.max()))
        code_mismatch_frac = code_gt1 / max(code_total, 1)

        assert code_mismatch_frac == 0.0, \
            f"[{fmt}] {code_mismatch_frac:.4g} of codes differ by >1 LSB (max diff {max_code_diff})"
        assert scale_relerr < 1e-3, f"[{fmt}] scale rel-err {scale_relerr:.4g} too high"
        assert recon_relerr < 0.15, f"[{fmt}] reconstruction rel-err {recon_relerr:.4g} too high"

    @pytest.mark.parametrize("fmt,N,K", CASES)
    def test_correctness_vs_reference(self, get_state, ocl_queue, fmt, N, K):
        """Transcoded output vs the reference .cl (cheap GPU-vs-GPU compare on a subset of the
        reconstructed weight)."""
        st = get_state(fmt, N, K)
        st["enqueue"](ocl_queue).wait()
        wq, sc = st["read_output"](ocl_queue)
        rec = _recon_subset(wq, sc, fmt, N, K, st["to_i4"], GROUP, VERIFY_SLOTS, VERIFY_ROWS)

        ref_st = _setup(ocl_queue.context, fmt, N, K, str(TASK_DIR / REFERENCE_FILE))
        ref_st["enqueue"](ocl_queue).wait()
        wq_ref, sc_ref = ref_st["read_output"](ocl_queue)
        rec_ref = _recon_subset(wq_ref, sc_ref, fmt, N, K, ref_st["to_i4"], GROUP, VERIFY_SLOTS, VERIFY_ROWS)

        err = _rel_l2(rec, rec_ref)
        assert err < 1e-2, f"[{fmt}] rel-L2 vs reference {err:.4g} too high"

    @pytest.mark.parametrize("fmt,N,K", CASES)
    @pytest.mark.performance
    def test_benchmark(self, get_state, ocl_queue, measure_runtime, fmt, N, K):
        """Times ONLY the transcode kernel. The raw GGUF weight W[E,N,K] (>> 16 MiB L2) is read
        in full every launch, so successive dispatches stream cold from DRAM without any explicit
        L2 flush or input cycling."""
        st = get_state(fmt, N, K)
        st["enqueue"](ocl_queue).wait()  # warm build / first-touch allocation
        measure_runtime(
            lambda: st["enqueue"](ocl_queue),
            sync_fn=ocl_queue.finish,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestMoEGgufTranscode()
    task.build(TestBase.get_machine_gpu_arch())
