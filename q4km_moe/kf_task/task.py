# [USER_INSTRUCTIONS_START]
# Optimization target: gemv_q4k and gemv_q6k in kernel_q4k_q6k.cl
#
# These are plain GEMV kernels for GGUF quantized weight matrices using
# Q4_K and Q6_K formats with the shuffled layout (ShuffleQ4KL1 / ShuffleQ6KL1).
# They are used in MoE (Mixture-of-Experts) LLM inference for Q, K, V, up, gate,
# down projections.
#
# Test cases (token_len=1 GEMV, output = W[N,K] * x[K]):
#   Q6K_K12288_N4096   Q4K_K12288_N4096   Q4K_K4096_N12288
#   Q4K_K4096_N4096    Q6K_K4096_N1024    Q4K_K4096_N1024
#
# REQUIRED constraints (host dispatch in task.py depends on them):
#   1. Entry-point names and argument order must remain EXACTLY:
#        gemv_q4k(inputs, pqs, psl, outputs, input_len, output_len)
#        gemv_q6k(inputs, pql, pqh, ps, pd_buf, outputs, input_len, output_len)
#   2. #define OPG_Q4K, OPG_Q6K, BS_Q6K must remain in the file.
#      Their values may change; task.py parses them automatically so that
#      global/local sizes stay consistent. New values must satisfy:
#        output_len % OPG_Q4K == 0
#        output_len % OPG_Q6K == 0
#        (input_len / 256) % BS_Q6K == 0
#
# Optimization ideas to explore:
#   - Vectorize input load: float4/float8/float16 (inputs[] is 256-float aligned)
#   - Sub-group shuffles to share input vector across SIMD lanes
#   - Vectorize nibble unpack: uchar4/uchar8 reads on pqs/pql
#   - Increase OPG (outputs per work-group) to amortize input load cost
#   - Increase BS_Q6K (blocks per thread) to reduce local-memory traffic
#   - Unroll j (sub-block) and k (nibble) loops explicitly
#   - Replace private array inp[256] with direct reads from global in the inner loop
#     (avoids 1 KB private-memory spill on small register files)
#   - Software pipeline: overlap input load with MAC computation
# [USER_INSTRUCTIONS_END]

import logging
import re
import struct
from pathlib import Path

import numpy as np
import pytest
import pyopencl as cl

from kernelfoundry import TestBase  # Tests must derive from TestBase

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TASK_DIR       = Path(__file__).parent.resolve()
KERNEL_FILE    = str(TASK_DIR / "kernel_q4k_q6k.cl")
REFERENCE_FILE = str(TASK_DIR / "reference_q4k_q6k.cl")

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
SEED = 1234
BENCH = dict(flush_mib=256, iters=300, warmup=50, flush_l2=True)

# Correctness check: verify first VR output rows (enough to catch bugs, fast)
VR = 64

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    {"quant": "q6k", "K": 12288, "N":  4096, "label": "Q6K_K12288_N4096"},
    {"quant": "q4k", "K": 12288, "N":  4096, "label": "Q4K_K12288_N4096"},
    {"quant": "q4k", "K":  4096, "N": 12288, "label": "Q4K_K4096_N12288"},
    {"quant": "q4k", "K":  4096, "N":  4096, "label": "Q4K_K4096_N4096"},
    {"quant": "q6k", "K":  4096, "N":  1024, "label": "Q6K_K4096_N1024"},
    {"quant": "q4k", "K":  4096, "N":  1024, "label": "Q4K_K4096_N1024"},
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
# fp16 bytes -> float helper
# ---------------------------------------------------------------------------
def _fp16(b0, b1):
    return float(struct.unpack('e', struct.pack('H', int(b0) | (int(b1) << 8)))[0])


# ---------------------------------------------------------------------------
# Random GGUF block generators (structurally valid, random-valued)
# ---------------------------------------------------------------------------
def _gen_q4k_blocks(n, seed=SEED):
    rng = np.random.default_rng(seed)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm  = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc  = rng.integers(0, 64,  (n, 12),  dtype=np.uint8)
    qs  = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qs], axis=1).ravel()   # [n * 144]


def _gen_q6k_blocks(n, seed=SEED):
    rng = np.random.default_rng(seed)
    ql  = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    qh  = rng.integers(0, 256, (n,  64), dtype=np.uint8)
    sc  = rng.integers(-8, 8,  (n,  16), dtype=np.int8).view(np.uint8)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    return np.concatenate([ql, qh, sc, d], axis=1).ravel()   # [n * 210]


# ---------------------------------------------------------------------------
# Shuffle: standard GGUF layout -> separated-section layout used by kernels
# ---------------------------------------------------------------------------
def _shuffle_q4k(raw, K, N):
    """raw: [N * K/256 * 144] uint8  ->  pqs [NB*128], psl [NB*16]"""
    NB  = N * K // 256
    pqs = np.zeros(NB * 128, dtype=np.uint8)
    psl = np.zeros(NB * 16,  dtype=np.uint8)
    for b in range(NB):
        blk  = raw[b * 144:(b + 1) * 144]
        d    = _fp16(blk[0], blk[1])
        dmin = _fp16(blk[2], blk[3])
        sc   = blk[4:16]
        qs   = blk[16:144]

        rw = np.zeros(256, dtype=np.uint8)
        for j in range(4):
            for k in range(32):
                rw[j * 64 + k]      = qs[j * 32 + k] & 0x0F
                rw[j * 64 + 32 + k] = (qs[j * 32 + k] >> 4) & 0x0F

        rs = np.zeros(8, np.uint8)
        rm = np.zeros(8, np.uint8)
        rs[0] = sc[0] & 63;   rs[1] = sc[1] & 63
        rs[2] = sc[2] & 63;   rs[3] = sc[3] & 63
        rs[4] = (sc[8]  & 0x0F) | ((sc[0]  >> 2) & 0x30)
        rs[5] = (sc[9]  & 0x0F) | ((sc[1]  >> 2) & 0x30)
        rs[6] = (sc[10] & 0x0F) | ((sc[2]  >> 2) & 0x30)
        rs[7] = (sc[11] & 0x0F) | ((sc[3]  >> 2) & 0x30)
        rm[0] = sc[4] & 63;   rm[1] = sc[5] & 63
        rm[2] = sc[6] & 63;   rm[3] = sc[7] & 63
        rm[4] = ((sc[8]  >> 4) & 0x0F) | ((sc[4] >> 2) & 0x30)
        rm[5] = ((sc[9]  >> 4) & 0x0F) | ((sc[5] >> 2) & 0x30)
        rm[6] = ((sc[10] >> 4) & 0x0F) | ((sc[6] >> 2) & 0x30)
        rm[7] = ((sc[11] >> 4) & 0x0F) | ((sc[7] >> 2) & 0x30)

        opqs = pqs[b * 128:(b + 1) * 128]
        for j in range(8):
            for k in range(16):
                opqs[j * 16 + k] = (rw[j*32+k] & 0x0F) | ((rw[j*32+16+k] & 0x0F) << 4)

        opsl = psl[b * 16:(b + 1) * 16]
        opsl[0]  = (rs[0] & 0x0F) | ((rs[1] & 0x0F) << 4)
        opsl[1]  = (rs[2] & 0x0F) | ((rs[3] & 0x0F) << 4)
        opsl[2]  = (rs[4] & 0x0F) | ((rs[5] & 0x0F) << 4)
        opsl[3]  = (rs[6] & 0x0F) | ((rs[7] & 0x0F) << 4)
        opsl[4]  = (rm[0] & 0x0F) | ((rm[1] & 0x0F) << 4)
        opsl[5]  = (rm[2] & 0x0F) | ((rm[3] & 0x0F) << 4)
        opsl[6]  = (rm[4] & 0x0F) | ((rm[5] & 0x0F) << 4)
        opsl[7]  = (rm[6] & 0x0F) | ((rm[7] & 0x0F) << 4)
        opsl[8]  = ((rs[0]&0x30)>>4) | ((rs[1]&0x30)>>2) | (rs[2]&0x30) | ((rs[3]&0x30)<<2)
        opsl[9]  = ((rs[4]&0x30)>>4) | ((rs[5]&0x30)>>2) | (rs[6]&0x30) | ((rs[7]&0x30)<<2)
        opsl[10] = ((rm[0]&0x30)>>4) | ((rm[1]&0x30)>>2) | (rm[2]&0x30) | ((rm[3]&0x30)<<2)
        opsl[11] = ((rm[4]&0x30)>>4) | ((rm[5]&0x30)>>2) | (rm[6]&0x30) | ((rm[7]&0x30)<<2)
        opsl[12:14] = np.frombuffer(struct.pack('e', d),    dtype=np.uint8)
        opsl[14:16] = np.frombuffer(struct.pack('e', dmin), dtype=np.uint8)
    return pqs, psl


def _shuffle_q6k(raw, K, N):
    """raw: [N * K/256 * 210] uint8  ->  pql [NB*128], pqh [NB*64], ps_u8 [NB*16], pd [NB*2]"""
    NB  = N * K // 256
    pql = np.zeros(NB * 128, dtype=np.uint8)
    pqh = np.zeros(NB * 64,  dtype=np.uint8)
    ps  = np.zeros(NB * 16,  dtype=np.int8)
    pd  = np.zeros(NB * 2,   dtype=np.uint8)
    for b in range(NB):
        blk  = raw[b * 210:(b + 1) * 210]
        ql_r = blk[0:128]
        qh_r = blk[128:192]
        sc_r = blk[192:208].view(np.int8)
        d    = _fp16(blk[208], blk[209])

        rw = np.zeros(256, np.uint8)
        for j in range(2):
            for k in range(64):
                rw[j*128+k]       = ql_r[j*64+k] & 0x0F
                rw[j*128+64+k]    = (ql_r[j*64+k] >> 4) & 0x0F
            for k in range(32):
                rw[j*128+k]      += ((qh_r[j*32+k] & 0x03) << 4)
                rw[j*128+32+k]   += ((qh_r[j*32+k] & 0x0C) << 2)
                rw[j*128+64+k]   +=   qh_r[j*32+k] & 0x30
                rw[j*128+96+k]   += ((qh_r[j*32+k] & 0xC0) >> 2)

        opql = pql[b * 128:(b + 1) * 128]
        for j in range(8):
            for k in range(16):
                opql[j*16+k] = (rw[j*32+k] & 0x0F) | ((rw[j*32+16+k] & 0x0F) << 4)

        opqh = pqh[b * 64:(b + 1) * 64]
        for j in range(16):
            for k in range(4):
                opqh[j*4+k] = (((rw[j*16+k]   & 0x30) >> 4)
                             |  ((rw[j*16+k+4] & 0x30) >> 2)
                             |   (rw[j*16+k+8] & 0x30)
                             |  ((rw[j*16+k+12]& 0x30) << 2))

        ps[b * 16:(b + 1) * 16] = sc_r
        pd[b * 2: (b + 1) * 2]  = np.frombuffer(struct.pack('e', d), dtype=np.uint8)
    return pql, pqh, ps.view(np.uint8), pd


# ---------------------------------------------------------------------------
# Pure-Python dequantize + GEMV reference
# ---------------------------------------------------------------------------
def _deq_q4k(pqs, psl, K, N):
    """Decode shuffled Q4_K -> W[N, K] float32."""
    nprow = K // 256
    W = np.zeros((N, K), np.float32)
    for row in range(N):
        for bi in range(nprow):
            idx  = row * nprow + bi
            qs_b = pqs[idx * 128:(idx + 1) * 128]
            sl_b = psl[idx * 16: (idx + 1) * 16]
            sli  = int(sl_b[0])|(int(sl_b[1])<<8)|(int(sl_b[2])<<16)|(int(sl_b[3])<<24)
            mli  = int(sl_b[4])|(int(sl_b[5])<<8)|(int(sl_b[6])<<16)|(int(sl_b[7])<<24)
            shi  = int(sl_b[8]) | (int(sl_b[9]) <<8)
            mhi  = int(sl_b[10])| (int(sl_b[11])<<8)
            d    = _fp16(sl_b[12], sl_b[13])
            dmin = _fp16(sl_b[14], sl_b[15])
            for j in range(8):
                sq = ((sli >> (j*4)) & 0xF) | (((shi >> (j*2)) & 3) << 4)
                mq = ((mli >> (j*4)) & 0xF) | (((mhi >> (j*2)) & 3) << 4)
                sc = sq * d;  mv = mq * dmin
                for k in range(16):
                    byte = int(qs_b[j*16+k])
                    cb   = bi * 256 + j * 32
                    W[row, cb + k]      = (byte & 0x0F) * sc - mv
                    W[row, cb + 16 + k] = ((byte >> 4) & 0x0F) * sc - mv
    return W


def _deq_q6k(pql, pqh, ps_u8, pd_buf, K, N):
    """Decode shuffled Q6_K -> W[N, K] float32."""
    nprow = K // 256
    ps    = ps_u8.view(np.int8)
    SH    = [0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30]
    W     = np.zeros((N, K), np.float32)
    for row in range(N):
        for bi in range(nprow):
            idx  = row * nprow + bi
            ql_b = pql[idx*128:(idx+1)*128]
            qh_b = pqh[idx*64: (idx+1)*64]
            s_b  = ps [idx*16: (idx+1)*16]
            d    = _fp16(pd_buf[idx*2], pd_buf[idx*2+1])
            for j in range(8):
                clo = d * float(s_b[2*j]);  chi = d * float(s_b[2*j+1])
                hlo = (int(qh_b[j*8])  |(int(qh_b[j*8+1])<<8)
                     | int(qh_b[j*8+2])<<16|(int(qh_b[j*8+3])<<24))
                hhi = (int(qh_b[j*8+4])|(int(qh_b[j*8+5])<<8)
                     | int(qh_b[j*8+6])<<16|(int(qh_b[j*8+7])<<24))
                for k in range(16):
                    qlb = int(ql_b[j*16+k]);  sh = SH[k]
                    w0  = float((qlb&0x0F)|(((hlo>>sh)&3)<<4));  w0 = (w0-32.0)*clo
                    w1  = float(((qlb>>4)&0x0F)|(((hhi>>sh)&3)<<4));  w1 = (w1-32.0)*chi
                    cb  = bi * 256 + j * 32
                    W[row, cb + k]      = w0
                    W[row, cb + 16 + k] = w1
    return W


# ---------------------------------------------------------------------------
# GPU buffer creation helpers
# ---------------------------------------------------------------------------
def _ro_buf(ctx, arr):
    """Upload ndarray to a READ_ONLY GPU buffer."""
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=np.ascontiguousarray(arr))


def _make_q4k_state(ctx, K, N, vr=VR, seed=SEED):
    """
    Build GPU buffers for a Q4_K GEMV (N output rows, K input length).
    vr: number of rows for which a Python reference is computed (0 = skip).
    Returns a dict with keys: K, N, nprow, vr, pqs_b, psl_b, x_b, y_ref.
    """
    nprow  = K // 256
    total  = N * nprow
    vr     = min(vr, N)
    rng    = np.random.default_rng(seed ^ 0xBEEF_1234)

    pqs_f = np.empty(total * 128, dtype=np.uint8)
    psl_f = np.empty(total * 16,  dtype=np.uint8)

    pqs_v = psl_v = None
    if vr > 0:
        raw    = _gen_q4k_blocks(vr * nprow, seed=seed)
        pqs_v, psl_v = _shuffle_q4k(raw, K, vr)
        pqs_f[:vr * nprow * 128] = pqs_v
        psl_f[:vr * nprow * 16]  = psl_v
    rest = (N - vr) * nprow
    if rest > 0:
        pqs_f[vr * nprow * 128:] = rng.integers(0, 256, rest * 128, dtype=np.uint8)
        psl_f[vr * nprow * 16:]  = rng.integers(0, 256, rest * 16,  dtype=np.uint8)

    x     = rng.standard_normal(K).astype(np.float32)
    y_ref = (_deq_q4k(pqs_v, psl_v, K, vr) @ x).astype(np.float32) if vr > 0 else None

    return dict(K=K, N=N, nprow=nprow, vr=vr,
                pqs_b=_ro_buf(ctx, pqs_f), psl_b=_ro_buf(ctx, psl_f),
                x_b=_ro_buf(ctx, x), y_ref=y_ref)


def _make_q6k_state(ctx, K, N, vr=VR, seed=SEED):
    """
    Build GPU buffers for a Q6_K GEMV (N output rows, K input length).
    vr: number of rows for which a Python reference is computed (0 = skip).
    Returns a dict with keys: K, N, nprow, vr, pql_b, pqh_b, ps_b, pd_b, x_b, y_ref.
    """
    nprow  = K // 256
    total  = N * nprow
    vr     = min(vr, N)
    rng    = np.random.default_rng(seed ^ 0xBEEF_1234)

    pql_f = np.empty(total * 128, dtype=np.uint8)
    pqh_f = np.empty(total * 64,  dtype=np.uint8)
    ps_f  = np.empty(total * 16,  dtype=np.uint8)
    pd_f  = np.empty(total * 2,   dtype=np.uint8)

    pql_v = pqh_v = ps_v = pd_v = None
    if vr > 0:
        raw  = _gen_q6k_blocks(vr * nprow, seed=seed)
        pql_v, pqh_v, ps_v, pd_v = _shuffle_q6k(raw, K, vr)
        pql_f[:vr * nprow * 128] = pql_v
        pqh_f[:vr * nprow * 64]  = pqh_v
        ps_f [:vr * nprow * 16]  = ps_v
        pd_f [:vr * nprow * 2]   = pd_v
    rest = (N - vr) * nprow
    if rest > 0:
        pql_f[vr * nprow * 128:] = rng.integers(0, 256, rest * 128, dtype=np.uint8)
        pqh_f[vr * nprow * 64:]  = rng.integers(0, 256, rest * 64,  dtype=np.uint8)
        ps_f [vr * nprow * 16:]  = rng.integers(0, 256, rest * 16,  dtype=np.uint8)
        pd_f [vr * nprow * 2:]   = rng.integers(0, 256, rest * 2,   dtype=np.uint8)

    x     = rng.standard_normal(K).astype(np.float32)
    y_ref = (_deq_q6k(pql_v, pqh_v, ps_v, pd_v, K, vr) @ x).astype(np.float32) if vr > 0 else None

    return dict(K=K, N=N, nprow=nprow, vr=vr,
                pql_b=_ro_buf(ctx, pql_f), pqh_b=_ro_buf(ctx, pqh_f),
                ps_b=_ro_buf(ctx, ps_f),   pd_b=_ro_buf(ctx, pd_f),
                x_b=_ro_buf(ctx, x), y_ref=y_ref)


# ---------------------------------------------------------------------------
# OPG / BS constant parser
# Reads #define OPG_Q4K, OPG_Q6K, BS_Q6K from the compiled kernel source so
# that task.py always derives the correct dispatch dimensions even when the
# LLM changes those constants during optimization.
# ---------------------------------------------------------------------------
def _parse_dispatch_consts(cl_file):
    src = Path(cl_file).read_text()
    def _get(name, default):
        m = re.search(r'#define\s+' + name + r'\s+(\d+)', src)
        return int(m.group(1)) if m else default
    return _get("OPG_Q4K", 16), _get("OPG_Q6K", 16), _get("BS_Q6K", 2)


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------
def _build_prog(ctx, cl_file):
    src = Path(cl_file).read_text()
    return cl.Program(ctx, src).build()


# ---------------------------------------------------------------------------
# L3 flush helpers
# ---------------------------------------------------------------------------
def _setup_l2_flush(ctx, queue, mib):
    mf    = cl.mem_flags
    elems = (mib * 1024 * 1024) // 4
    buf   = cl.Buffer(ctx, mf.READ_WRITE, size=elems * 4)
    sink  = cl.Buffer(ctx, mf.WRITE_ONLY, size=4096 * 4)
    cl.enqueue_copy(queue, buf, np.zeros(elems, np.float32))   # commit all pages
    prog  = cl.Program(ctx, _FLUSH_SRC).build()
    krn   = cl.Kernel(prog, "cache_flush")
    queue.finish()
    flush = dict(krn=krn, buf=buf, sink=sink, elems=elems)
    # One untimed pre-run to warm instruction cache
    _flush_l2(queue, flush)
    return flush


def _flush_l2(queue, flush, sweeps=2):
    gsz = (min(flush["elems"], 1 << 20),)
    for _ in range(sweeps):
        flush["krn"](queue, gsz, None, flush["buf"], np.uint32(flush["elems"]), flush["sink"])
    queue.finish()


def _flush_l2_enqueue(queue, flush, sweeps=2):
    """Enqueue flush without queue.finish() to keep GPU continuously active."""
    gsz = (min(flush["elems"], 1 << 20),)
    for _ in range(sweeps):
        flush["krn"](queue, gsz, None, flush["buf"], np.uint32(flush["elems"]), flush["sink"])
    # queue.finish()


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
# Tests — parametrized over all 6 cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", TEST_CASES, ids=CASE_IDS)
class TestGemvQ4KQ6K(TestBase):
    """Q4_K and Q6_K shuffled-layout plain GEMV, swept over MoE projection shapes."""

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
        prog = _build_prog(ctx, KERNEL_FILE)
        cl.Kernel(prog, "gemv_q4k")   # raises if missing
        cl.Kernel(prog, "gemv_q6k")   # raises if missing

    def test_correctness_vs_numpy(self, case, ocl_queue):
        """GPU output must match pure-Python dequantize+GEMV reference (first VR rows)."""
        quant = case["quant"]
        K, N  = case["K"], case["N"]
        prog  = _build_prog(ocl_queue.context, KERNEL_FILE)
        opg_q4k, opg_q6k, bs_q6k = _parse_dispatch_consts(KERNEL_FILE)

        if quant == "q4k":
            st   = _make_q4k_state(ocl_queue.context, K, N, vr=VR)
            krn  = cl.Kernel(prog, "gemv_q4k")
            nprow = st["nprow"]
            y_b  = cl.Buffer(ocl_queue.context, cl.mem_flags.WRITE_ONLY, size=N * 4)
            krn.set_args(st["x_b"], st["pqs_b"], st["psl_b"],
                         y_b, np.uint32(K), np.uint32(N))
            gsize = (N // opg_q4k) * nprow
            lsize = nprow
        else:  # q6k
            st   = _make_q6k_state(ocl_queue.context, K, N, vr=VR)
            krn  = cl.Kernel(prog, "gemv_q6k")
            nprow = st["nprow"]
            local_range = nprow // bs_q6k
            y_b  = cl.Buffer(ocl_queue.context, cl.mem_flags.WRITE_ONLY, size=N * 4)
            krn.set_args(st["x_b"], st["pql_b"], st["pqh_b"],
                         st["ps_b"], st["pd_b"],
                         y_b, np.uint32(K), np.uint32(N))
            gsize = (N // opg_q6k) * local_range
            lsize = local_range

        ev = cl.enqueue_nd_range_kernel(ocl_queue, krn, (gsize,), (lsize,))
        ev.wait()
        y_gpu = np.empty(N, np.float32)
        cl.enqueue_copy(ocl_queue, y_gpu, y_b)
        ocl_queue.finish()

        y_ref = st["y_ref"]
        vr    = st["vr"]
        assert y_ref is not None, "Reference not computed (vr=0)"
        max_abs = float(np.max(np.abs(y_ref - y_gpu[:vr])))
        denom   = float(np.max(np.abs(y_ref))) + 1e-9
        rel_err = max_abs / denom
        assert rel_err < 2e-2, (
            f"{case['label']}: max relative error {rel_err:.4g} exceeds 2% threshold"
        )

    def test_correctness_vs_reference(self, case, ocl_queue):
        """GPU kernel output must match reference .cl output (first VR rows, GPU-vs-GPU)."""
        quant = case["quant"]
        K, N  = case["K"], case["N"]
        vr    = min(VR, N)
        opg_q4k, opg_q6k, bs_q6k = _parse_dispatch_consts(KERNEL_FILE)

        kern_prog = _build_prog(ocl_queue.context, KERNEL_FILE)
        ref_prog  = _build_prog(ocl_queue.context, REFERENCE_FILE)

        def _run(prog, st, quant):
            if quant == "q4k":
                krn   = cl.Kernel(prog, "gemv_q4k")
                nprow = st["nprow"]
                y_b   = cl.Buffer(ocl_queue.context, cl.mem_flags.WRITE_ONLY, size=N * 4)
                krn.set_args(st["x_b"], st["pqs_b"], st["psl_b"],
                             y_b, np.uint32(K), np.uint32(N))
                gsize = (N // opg_q4k) * nprow
                lsize = nprow
            else:
                krn   = cl.Kernel(prog, "gemv_q6k")
                nprow = st["nprow"]
                local_range = nprow // bs_q6k
                y_b   = cl.Buffer(ocl_queue.context, cl.mem_flags.WRITE_ONLY, size=N * 4)
                krn.set_args(st["x_b"], st["pql_b"], st["pqh_b"],
                             st["ps_b"], st["pd_b"],
                             y_b, np.uint32(K), np.uint32(N))
                gsize = (N // opg_q6k) * local_range
                lsize = local_range
            ev = cl.enqueue_nd_range_kernel(ocl_queue, krn, (gsize,), (lsize,))
            ev.wait()
            y = np.empty(N, np.float32)
            cl.enqueue_copy(ocl_queue, y, y_b)
            ocl_queue.finish()
            return y[:vr]

        if quant == "q4k":
            st = _make_q4k_state(ocl_queue.context, K, N, vr=vr)
        else:
            st = _make_q6k_state(ocl_queue.context, K, N, vr=vr)

        y_kern = _run(kern_prog, st, quant)
        y_ref  = _run(ref_prog,  st, quant)

        norm_ref = float(np.linalg.norm(y_ref)) + 1e-9
        rel_err  = float(np.linalg.norm(y_kern - y_ref)) / norm_ref
        assert rel_err < 1e-2, (
            f"{case['label']}: rel-L2 vs reference {rel_err:.4g} exceeds 1% threshold"
        )

    @pytest.mark.performance
    def test_benchmark(self, case, use_reference, ocl_queue, l2_flush, request, profile_store):
        """
        Benchmark with per-iteration L3 flush (cold-weight DRAM read).
        Results stored in profile_store for KernelFoundry speedup computation.
        """
        quant   = case["quant"]
        K, N    = case["K"], case["N"]
        cl_file = REFERENCE_FILE if use_reference else KERNEL_FILE
        prog    = _build_prog(ocl_queue.context, cl_file)
        opg_q4k, opg_q6k, bs_q6k = _parse_dispatch_consts(cl_file)

        ctx = ocl_queue.context
        if quant == "q4k":
            st    = _make_q4k_state(ctx, K, N, vr=0)
            krn   = cl.Kernel(prog, "gemv_q4k")
            nprow = st["nprow"]
            y_b   = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=N * 4)
            krn.set_args(st["x_b"], st["pqs_b"], st["psl_b"],
                         y_b, np.uint32(K), np.uint32(N))
            gsize = (N // opg_q4k) * nprow
            lsize = nprow
        else:  # q6k
            st    = _make_q6k_state(ctx, K, N, vr=0)
            krn   = cl.Kernel(prog, "gemv_q6k")
            nprow = st["nprow"]
            local_range = nprow // bs_q6k
            y_b   = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=N * 4)
            krn.set_args(st["x_b"], st["pql_b"], st["pqh_b"],
                         st["ps_b"], st["pd_b"],
                         y_b, np.uint32(K), np.uint32(N))
            gsize = (N // opg_q6k) * local_range
            lsize = local_range

        def _enq():
            return cl.enqueue_nd_range_kernel(ocl_queue, krn, (gsize,), (lsize,))

        # Warmup — page in weights, stabilise GPU clocks
        for _ in range(BENCH["warmup"]):
            _enq()
            ocl_queue.finish()

        # Timed trials: flush L3 (untimed) then one cold kernel dispatch via CL events.
        # _flush_l2_enqueue omits queue.finish() so the GPU stays continuously active and
        # the Intel Arc power manager cannot drop frequency during CPU overhead.
        times_ms = []
        for _ in range(BENCH["iters"]):
            if BENCH["flush_l2"]:
                _flush_l2_enqueue(ocl_queue, l2_flush)
                ocl_queue.finish()
            ev = _enq()
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
    task = TestGemvQ4KQ6K()
    task.build(TestBase.get_machine_gpu_arch())
