"""
test_moe_gemv_sg_kernels.py

Correctness + benchmark + roofline harness for the high-performance "SG"
(sub-group block-read) MoE OpenCL kernels:

    moe_gemv_q4k_sg.cl : group_up_gate_q4k_sg  down_merge_q4k_sg
    moe_gemv_q5k_sg.cl : group_up_gate_q5k_sg  down_merge_q5k_sg
    moe_gemv_q6k_sg.cl : group_up_gate_q6k_sg  down_merge_q6k_sg

These kernels transpose each expert's weight matrix into the "SG" column-major
layout (identical scheme to gemv_q4k_sg.cl / pack_q4k_sg) so that a single
intel_sub_group_block_read delivers 4 contiguous weight bytes to each of the 16
lanes, dramatically raising the width of the DRAM transactions for these
memory-bound GEMVs.  The on-device weight buffers are the SAME total size as
the moe_gemv_q4k_shuffle.cl layout -- only the byte ordering differs.

MoE-FFN shapes (same as test_moe_gemv_shuffle_kernels.py):
    gate_up : token_len=1, K=2048, output_len=512
    down    : token_len=1, K=512,  output_len=2048
    256 experts total, top-8-of-256 active routing per token.

Usage:
    python test_moe_gemv_sg_kernels.py [--device 0] [--iters 1000] [--warmup 100]
        [--flush_mib 256] [--peak_bw 512] [--peak_fp32 9800]
        [--verbose] [--no-bench] [--no-cache]
"""

import os
import struct
import statistics
import argparse
import hashlib

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    raise SystemExit("pyopencl is required: pip install pyopencl")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPG = 16   # sub_group_size = output rows per sub-group


def cl_src(name):
    return os.path.join(SCRIPT_DIR, name)


# ---------------------------------------------------------------------------
# Expert-pool disk cache (SG-packed pools)
# ---------------------------------------------------------------------------
_CACHE_VERSION = "sgv1"
_CACHE_ENABLED = os.environ.get("MOE_EXPERT_POOL_NO_CACHE", "0") not in ("1", "true", "True")
_CACHE_ENABLED = False  # disabled by default
_CACHE_DIR = os.environ.get(
    "MOE_EXPERT_POOL_CACHE_DIR",
    os.path.join(SCRIPT_DIR, ".expert_pool_cache"),
)


def _cache_path(quant, num_experts, K, N, seed):
    key = f"{quant}sg_{_CACHE_VERSION}_ne{num_experts}_K{K}_N{N}_seed{seed}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"{key}_{digest}.npz")


def _load_cached(quant, num_experts, K, N, seed):
    if not _CACHE_ENABLED:
        return None
    path = _cache_path(quant, num_experts, K, N, seed)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            flat = data["flat"]
            ref_W = data["ref_W"]
            expert_size = int(data["expert_size"])
        print(f"    [cache] loaded {quant}_sg pool (experts={num_experts} K={K} N={N} "
              f"seed={seed}) from {path}")
        return flat, ref_W, expert_size
    except Exception as exc:
        print(f"    [cache] failed to load {path} ({exc}); rebuilding")
        return None


def _save_cached(quant, num_experts, K, N, seed, flat, ref_W, expert_size):
    if not _CACHE_ENABLED:
        return
    path = _cache_path(quant, num_experts, K, N, seed)
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp_path = path + ".tmp.npz"
        np.savez(tmp_path, flat=flat, ref_W=ref_W, expert_size=np.int64(expert_size))
        os.replace(tmp_path, path)
        print(f"    [cache] saved {quant}_sg pool to {path}")
    except Exception as exc:
        print(f"    [cache] failed to save {path} ({exc})")


# ---------------------------------------------------------------------------
# Inline cache-flush OpenCL kernel
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
# Hardware roofline model
# ---------------------------------------------------------------------------
class HW:
    def __init__(self, peak_bw_gbps=512.0, fp32_gflops=9800.0):
        self.peak_bw_gbps = peak_bw_gbps
        self.fp32_gflops = fp32_gflops

    def roofline(self, flops, bytes_moved, seconds):
        achieved_gflops = flops / seconds / 1e9
        achieved_bw_gbps = bytes_moved / seconds / 1e9
        ai = flops / max(bytes_moved, 1)
        ridge = self.fp32_gflops / max(self.peak_bw_gbps, 1e-12)
        if ai < ridge:
            roof_gflops = ai * self.peak_bw_gbps
            bound = "memory"
        else:
            roof_gflops = self.fp32_gflops
            bound = "compute"
        return dict(
            achieved_gflops=achieved_gflops,
            achieved_bw_gbps=achieved_bw_gbps,
            arith_intensity=ai,
            ridge_point=ridge,
            roofline_gflops=roof_gflops,
            roofline_pct=achieved_gflops / max(roof_gflops, 1e-12) * 100.0,
            bw_util_pct=achieved_bw_gbps / max(self.peak_bw_gbps, 1e-12) * 100.0,
            bound_by=bound,
        )


# ---------------------------------------------------------------------------
# GPU context helper
# ---------------------------------------------------------------------------
class Gpu:
    def __init__(self, device_idx=0):
        platforms = cl.get_platforms()
        devices = [d for p in platforms for d in p.get_devices()]
        if not devices:
            raise RuntimeError("No OpenCL devices found")
        self.device = devices[device_idx % len(devices)]
        print(f"Device [{device_idx}]: {self.device.name}")
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(
            self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self._prog_cache = {}
        self._flush_krn = None
        self._flush_buf = None
        self._flush_sink = None
        self._flush_n = 0

    def build(self, src_file, build_opts=""):
        # Extra compile-time options can be injected for A/B experiments, e.g.
        #   MOE_CL_OPTS="-DACC_MODE=2 -DPRESTAGE=0" python test_moe_gemv_sg_kernels.py
        extra = os.environ.get("MOE_CL_OPTS", "").strip()
        if extra:
            build_opts = (build_opts + " " + extra).strip()
        key = (src_file, build_opts)
        if key not in self._prog_cache:
            with open(src_file) as f:
                src = f.read()
            self._prog_cache[key] = cl.Program(self.ctx, src).build(options=build_opts)
        return self._prog_cache[key]

    def _build_src(self, src, key):
        if key not in self._prog_cache:
            self._prog_cache[key] = cl.Program(self.ctx, src).build()
        return self._prog_cache[key]

    def setup_flush(self, mib=256):
        n = mib * 1024 * 1024 // 4
        self._flush_n = n
        mf = cl.mem_flags
        zeros = np.zeros(n, dtype=np.float32)
        self._flush_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self._flush_sink = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=4096 * 4)
        prog = self._build_src(_FLUSH_SRC, "_flush")
        self._flush_krn = cl.Kernel(prog, "cache_flush")
        self.queue.finish()

    def flush_l3(self, sweeps=2):
        if self._flush_krn is None:
            return
        gsz = (min(self._flush_n, 1 << 20),)
        for _ in range(sweeps):
            self._flush_krn(self.queue, gsz, None,
                            self._flush_buf, np.uint32(self._flush_n), self._flush_sink)
        self.queue.finish()

    def time_kernel(self, enq_fn, iters=1000, warmup=100, do_flush=True):
        for _ in range(warmup):
            enq_fn(self.queue)
            self.queue.finish()
        times = []
        for _ in range(iters):
            if do_flush:
                self.flush_l3()
            ev = enq_fn(self.queue)
            ev.wait()
            times.append((ev.profile.end - ev.profile.start) * 1e-9)
        return times


def stats(ts):
    return dict(
        mean_ms=statistics.mean(ts) * 1e3,
        median_ms=statistics.median(ts) * 1e3,
        min_ms=min(ts) * 1e3,
        max_ms=max(ts) * 1e3,
        iters=len(ts),
    )


def _cbuf(ctx, arr):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=np.ascontiguousarray(arr))


def check_close(label, ref, got, rtol=1e-2, atol=1e-3, verbose=False):
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    max_abs = float(np.max(np.abs(ref - got)))
    ref_scale = float(np.max(np.abs(ref)))
    rel_err = max_abs / (ref_scale + 1e-9)
    eff_atol = max(atol, ref_scale * 1e-6)
    ok = bool(np.allclose(ref, got, rtol=rtol, atol=eff_atol))
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}  max_abs={max_abs:.3e}  rel={rel_err:.3e}")
    if verbose and not ok:
        wi = int(np.argmax(np.abs(ref - got)))
        print(f"         worst idx={wi}  ref={ref.ravel()[wi]:.6f}  got={got.ravel()[wi]:.6f}")
    return ok


def _print_roofline(rl, moved):
    print(f"    BW={rl['achieved_bw_gbps']:.1f} GB/s ({rl['bw_util_pct']:.1f}% of peak)  "
          f"AI={rl['arith_intensity']:.2f} FLOP/B  "
          f"roofline={rl['roofline_pct']:.1f}%  bound={rl['bound_by']}  "
          f"GFLOPS={rl['achieved_gflops']:.1f}")
    print(f"    moved={moved / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Raw GGUF block generators
# ---------------------------------------------------------------------------
def gen_q4k_blocks(n, seed=1234):
    rng = np.random.default_rng(seed)
    d = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc = rng.integers(0, 64, (n, 12), dtype=np.uint8)
    qs = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qs], axis=1).ravel()


def gen_q5k_blocks(n, seed=1234):
    rng = np.random.default_rng(seed)
    d = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc = rng.integers(0, 64, (n, 12), dtype=np.uint8)
    qh = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    qs = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qh, qs], axis=1).ravel()


def gen_q6k_blocks(n, seed=1234):
    rng = np.random.default_rng(seed)
    ql = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    qh = rng.integers(0, 256, (n, 64), dtype=np.uint8)
    sc = rng.integers(-8, 8, (n, 16), dtype=np.int8).view(np.uint8)
    d = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    return np.concatenate([ql, qh, sc, d], axis=1).ravel()


# ---------------------------------------------------------------------------
# Shuffle: standard GGUF layout -> row-major separated-section kernel layout
# (vectorized over all NB = N*K//256 blocks; identical to
#  test_moe_gemv_shuffle_kernels.py)
# ---------------------------------------------------------------------------
def shuffle_q4k(raw, K, N):
    NB = N * K // 256
    blk = raw.reshape(NB, 144)
    d_bytes = blk[:, 0:2]
    dmin_bytes = blk[:, 2:4]
    sc = blk[:, 4:16]
    qs = blk[:, 16:144].reshape(NB, 4, 32)

    low = qs & 0x0F
    high = (qs >> 4) & 0x0F
    rw = np.empty((NB, 4, 64), dtype=np.uint8)
    rw[:, :, 0:32] = low
    rw[:, :, 32:64] = high
    rw = rw.reshape(NB, 256)

    (sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7,
     sc8, sc9, sc10, sc11) = (sc[:, i] for i in range(12))

    rs = np.empty((NB, 8), dtype=np.uint8)
    rm = np.empty((NB, 8), dtype=np.uint8)
    rs[:, 0] = sc0 & 63; rs[:, 1] = sc1 & 63; rs[:, 2] = sc2 & 63; rs[:, 3] = sc3 & 63
    rs[:, 4] = (sc8 & 0x0F) | ((sc0 >> 2) & 0x30)
    rs[:, 5] = (sc9 & 0x0F) | ((sc1 >> 2) & 0x30)
    rs[:, 6] = (sc10 & 0x0F) | ((sc2 >> 2) & 0x30)
    rs[:, 7] = (sc11 & 0x0F) | ((sc3 >> 2) & 0x30)
    rm[:, 0] = sc4 & 63; rm[:, 1] = sc5 & 63; rm[:, 2] = sc6 & 63; rm[:, 3] = sc7 & 63
    rm[:, 4] = ((sc8 >> 4) & 0x0F) | ((sc4 >> 2) & 0x30)
    rm[:, 5] = ((sc9 >> 4) & 0x0F) | ((sc5 >> 2) & 0x30)
    rm[:, 6] = ((sc10 >> 4) & 0x0F) | ((sc6 >> 2) & 0x30)
    rm[:, 7] = ((sc11 >> 4) & 0x0F) | ((sc7 >> 2) & 0x30)

    rw8 = rw.reshape(NB, 8, 32)
    lo16 = rw8[:, :, 0:16] & 0x0F
    hi16 = (rw8[:, :, 16:32] & 0x0F) << 4
    opqs = (lo16 | hi16).reshape(NB, 128).astype(np.uint8)

    opsl = np.zeros((NB, 16), dtype=np.uint8)
    opsl[:, 0] = (rs[:, 0] & 0x0F) | ((rs[:, 1] & 0x0F) << 4)
    opsl[:, 1] = (rs[:, 2] & 0x0F) | ((rs[:, 3] & 0x0F) << 4)
    opsl[:, 2] = (rs[:, 4] & 0x0F) | ((rs[:, 5] & 0x0F) << 4)
    opsl[:, 3] = (rs[:, 6] & 0x0F) | ((rs[:, 7] & 0x0F) << 4)
    opsl[:, 4] = (rm[:, 0] & 0x0F) | ((rm[:, 1] & 0x0F) << 4)
    opsl[:, 5] = (rm[:, 2] & 0x0F) | ((rm[:, 3] & 0x0F) << 4)
    opsl[:, 6] = (rm[:, 4] & 0x0F) | ((rm[:, 5] & 0x0F) << 4)
    opsl[:, 7] = (rm[:, 6] & 0x0F) | ((rm[:, 7] & 0x0F) << 4)
    opsl[:, 8] = ((rs[:, 0] & 0x30) >> 4) | ((rs[:, 1] & 0x30) >> 2) | (rs[:, 2] & 0x30) | ((rs[:, 3] & 0x30) << 2)
    opsl[:, 9] = ((rs[:, 4] & 0x30) >> 4) | ((rs[:, 5] & 0x30) >> 2) | (rs[:, 6] & 0x30) | ((rs[:, 7] & 0x30) << 2)
    opsl[:, 10] = ((rm[:, 0] & 0x30) >> 4) | ((rm[:, 1] & 0x30) >> 2) | (rm[:, 2] & 0x30) | ((rm[:, 3] & 0x30) << 2)
    opsl[:, 11] = ((rm[:, 4] & 0x30) >> 4) | ((rm[:, 5] & 0x30) >> 2) | (rm[:, 6] & 0x30) | ((rm[:, 7] & 0x30) << 2)
    opsl[:, 12:14] = d_bytes
    opsl[:, 14:16] = dmin_bytes

    return opqs.reshape(-1), opsl.reshape(-1)


def shuffle_q5k(raw, K, N):
    NB = N * K // 256
    blk = raw.reshape(NB, 176)
    d_bytes = blk[:, 0:2]
    dmin_bytes = blk[:, 2:4]
    sc = blk[:, 4:16]
    qh_r = blk[:, 16:48]
    qs_r = blk[:, 48:176].reshape(NB, 4, 32)

    g_arr = np.arange(8, dtype=np.uint8)
    rh = ((qh_r[:, None, :] >> g_arr[None, :, None]) & 1).astype(np.uint8)

    low = qs_r & 0x0F
    high = (qs_r >> 4) & 0x0F
    rw = np.empty((NB, 8, 32), dtype=np.uint8)
    rw[:, 0::2, :] = low
    rw[:, 1::2, :] = high

    (sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7,
     sc8, sc9, sc10, sc11) = (sc[:, i] for i in range(12))

    rs = np.empty((NB, 8), dtype=np.uint8)
    rm = np.empty((NB, 8), dtype=np.uint8)
    rs[:, 0] = sc0 & 63; rs[:, 1] = sc1 & 63; rs[:, 2] = sc2 & 63; rs[:, 3] = sc3 & 63
    rs[:, 4] = (sc8 & 0x0F) | ((sc0 >> 2) & 0x30)
    rs[:, 5] = (sc9 & 0x0F) | ((sc1 >> 2) & 0x30)
    rs[:, 6] = (sc10 & 0x0F) | ((sc2 >> 2) & 0x30)
    rs[:, 7] = (sc11 & 0x0F) | ((sc3 >> 2) & 0x30)
    rm[:, 0] = sc4 & 63; rm[:, 1] = sc5 & 63; rm[:, 2] = sc6 & 63; rm[:, 3] = sc7 & 63
    rm[:, 4] = ((sc8 >> 4) & 0x0F) | ((sc4 >> 2) & 0x30)
    rm[:, 5] = ((sc9 >> 4) & 0x0F) | ((sc5 >> 2) & 0x30)
    rm[:, 6] = ((sc10 >> 4) & 0x0F) | ((sc6 >> 2) & 0x30)
    rm[:, 7] = ((sc11 >> 4) & 0x0F) | ((sc7 >> 2) & 0x30)

    rw8 = rw.reshape(NB, 8, 32)
    lo16 = rw8[:, :, 0:16] & 0x0F
    hi16 = (rw8[:, :, 16:32] & 0x0F) << 4
    opqs = (lo16 | hi16).reshape(NB, 128).astype(np.uint8)

    opqh = np.zeros((NB, 8, 4), dtype=np.uint8)
    for s in range(8):
        opqh |= (rh[:, :, s * 4:s * 4 + 4] << s).astype(np.uint8)
    opqh = opqh.reshape(NB, 32)

    opsl = np.zeros((NB, 16), dtype=np.uint8)
    opsl[:, 0] = (rs[:, 0] & 0x0F) | ((rs[:, 1] & 0x0F) << 4)
    opsl[:, 1] = (rs[:, 2] & 0x0F) | ((rs[:, 3] & 0x0F) << 4)
    opsl[:, 2] = (rs[:, 4] & 0x0F) | ((rs[:, 5] & 0x0F) << 4)
    opsl[:, 3] = (rs[:, 6] & 0x0F) | ((rs[:, 7] & 0x0F) << 4)
    opsl[:, 4] = (rm[:, 0] & 0x0F) | ((rm[:, 1] & 0x0F) << 4)
    opsl[:, 5] = (rm[:, 2] & 0x0F) | ((rm[:, 3] & 0x0F) << 4)
    opsl[:, 6] = (rm[:, 4] & 0x0F) | ((rm[:, 5] & 0x0F) << 4)
    opsl[:, 7] = (rm[:, 6] & 0x0F) | ((rm[:, 7] & 0x0F) << 4)
    opsl[:, 8] = ((rs[:, 0] & 0x30) >> 4) | ((rs[:, 1] & 0x30) >> 2) | (rs[:, 2] & 0x30) | ((rs[:, 3] & 0x30) << 2)
    opsl[:, 9] = ((rs[:, 4] & 0x30) >> 4) | ((rs[:, 5] & 0x30) >> 2) | (rs[:, 6] & 0x30) | ((rs[:, 7] & 0x30) << 2)
    opsl[:, 10] = ((rm[:, 0] & 0x30) >> 4) | ((rm[:, 1] & 0x30) >> 2) | (rm[:, 2] & 0x30) | ((rm[:, 3] & 0x30) << 2)
    opsl[:, 11] = ((rm[:, 4] & 0x30) >> 4) | ((rm[:, 5] & 0x30) >> 2) | (rm[:, 6] & 0x30) | ((rm[:, 7] & 0x30) << 2)
    opsl[:, 12:14] = d_bytes
    opsl[:, 14:16] = dmin_bytes

    return opqs.reshape(-1), opqh.reshape(-1), opsl.reshape(-1)


def shuffle_q6k(raw, K, N):
    NB = N * K // 256
    blk = raw.reshape(NB, 210)
    ql_r = blk[:, 0:128].reshape(NB, 2, 64)
    qh_r = blk[:, 128:192].reshape(NB, 2, 32)
    sc_r = blk[:, 192:208]
    d_bytes = blk[:, 208:210]

    low = ql_r & 0x0F
    high = (ql_r >> 4) & 0x0F
    rw = np.empty((NB, 2, 128), dtype=np.uint8)
    rw[:, :, 0:64] = low
    rw[:, :, 64:128] = high

    rw[:, :, 0:32] = (rw[:, :, 0:32] + ((qh_r & 0x03) << 4)).astype(np.uint8)
    rw[:, :, 32:64] = (rw[:, :, 32:64] + ((qh_r & 0x0C) << 2)).astype(np.uint8)
    rw[:, :, 64:96] = (rw[:, :, 64:96] + (qh_r & 0x30)).astype(np.uint8)
    rw[:, :, 96:128] = (rw[:, :, 96:128] + ((qh_r & 0xC0) >> 2)).astype(np.uint8)

    rw_flat = rw.reshape(NB, 256)
    rw8 = rw_flat.reshape(NB, 8, 32)
    lo16 = rw8[:, :, 0:16] & 0x0F
    hi16 = (rw8[:, :, 16:32] & 0x0F) << 4
    opql = (lo16 | hi16).reshape(NB, 128).astype(np.uint8)

    rw16 = rw_flat.reshape(NB, 16, 16)
    a = rw16[:, :, 0:4] & 0x30
    b = rw16[:, :, 4:8] & 0x30
    c = rw16[:, :, 8:12] & 0x30
    e = rw16[:, :, 12:16] & 0x30
    opqh = ((a >> 4) | (b >> 2) | c | (e << 2)).reshape(NB, 64).astype(np.uint8)

    return opql.reshape(-1), opqh.reshape(-1), sc_r.reshape(-1).copy(), d_bytes.reshape(-1).copy()


# ---------------------------------------------------------------------------
# Reference dequantize (vectorized over all NB blocks) -> W[N, K]
# (operates on the row-major shuffled buffers, identical to shuffle test)
# ---------------------------------------------------------------------------
def deq_q4k(pqs, psl, K, N):
    nprow = K // 256
    NB = N * nprow
    qs_b = pqs.reshape(NB, 8, 16)
    sl_b = psl.reshape(NB, 16)

    sli = (sl_b[:, 0].astype(np.uint32) | (sl_b[:, 1].astype(np.uint32) << 8) |
           (sl_b[:, 2].astype(np.uint32) << 16) | (sl_b[:, 3].astype(np.uint32) << 24))
    mli = (sl_b[:, 4].astype(np.uint32) | (sl_b[:, 5].astype(np.uint32) << 8) |
           (sl_b[:, 6].astype(np.uint32) << 16) | (sl_b[:, 7].astype(np.uint32) << 24))
    shi = sl_b[:, 8].astype(np.uint32) | (sl_b[:, 9].astype(np.uint32) << 8)
    mhi = sl_b[:, 10].astype(np.uint32) | (sl_b[:, 11].astype(np.uint32) << 8)
    d = sl_b[:, 12:14].copy().view(np.float16).reshape(NB).astype(np.float32)
    dmin = sl_b[:, 14:16].copy().view(np.float16).reshape(NB).astype(np.float32)

    j_arr = np.arange(8, dtype=np.uint32)
    sq = ((sli[:, None] >> (j_arr * 4)) & 0xF) | (((shi[:, None] >> (j_arr * 2)) & 3) << 4)
    mq = ((mli[:, None] >> (j_arr * 4)) & 0xF) | (((mhi[:, None] >> (j_arr * 2)) & 3) << 4)
    sc = sq.astype(np.float32) * d[:, None]
    mv = mq.astype(np.float32) * dmin[:, None]

    lo = (qs_b & 0x0F).astype(np.float32)
    hi = ((qs_b >> 4) & 0x0F).astype(np.float32)

    Wblock = np.empty((NB, 8, 32), dtype=np.float32)
    Wblock[:, :, 0:16] = lo * sc[:, :, None] - mv[:, :, None]
    Wblock[:, :, 16:32] = hi * sc[:, :, None] - mv[:, :, None]
    return Wblock.reshape(N, nprow, 256).reshape(N, K)


def deq_q5k(pqs, pqh, psl, K, N):
    nprow = K // 256
    NB = N * nprow
    qs_b = pqs.reshape(NB, 8, 16)
    qh_b = pqh.reshape(NB, 8, 4)
    sl_b = psl.reshape(NB, 16)

    sli = (sl_b[:, 0].astype(np.uint32) | (sl_b[:, 1].astype(np.uint32) << 8) |
           (sl_b[:, 2].astype(np.uint32) << 16) | (sl_b[:, 3].astype(np.uint32) << 24))
    mli = (sl_b[:, 4].astype(np.uint32) | (sl_b[:, 5].astype(np.uint32) << 8) |
           (sl_b[:, 6].astype(np.uint32) << 16) | (sl_b[:, 7].astype(np.uint32) << 24))
    shi = sl_b[:, 8].astype(np.uint32) | (sl_b[:, 9].astype(np.uint32) << 8)
    mhi = sl_b[:, 10].astype(np.uint32) | (sl_b[:, 11].astype(np.uint32) << 8)
    d = sl_b[:, 12:14].copy().view(np.float16).reshape(NB).astype(np.float32)
    dmin = sl_b[:, 14:16].copy().view(np.float16).reshape(NB).astype(np.float32)

    j_arr = np.arange(8, dtype=np.uint32)
    sq = ((sli[:, None] >> (j_arr * 4)) & 0xF) | (((shi[:, None] >> (j_arr * 2)) & 3) << 4)
    mq = ((mli[:, None] >> (j_arr * 4)) & 0xF) | (((mhi[:, None] >> (j_arr * 2)) & 3) << 4)
    sc = sq.astype(np.float32) * d[:, None]
    mv = mq.astype(np.float32) * dmin[:, None]

    rh = np.zeros((NB, 8, 32), dtype=np.uint8)
    for s in range(8):
        rh[:, :, 4 * s:4 * s + 4] = (qh_b >> s) & 1

    lo_bit = rh[:, :, 0:16].astype(np.float32)
    hi_bit = rh[:, :, 16:32].astype(np.float32)
    w0 = (qs_b & 0x0F).astype(np.float32) + lo_bit * 16.0
    w1 = ((qs_b >> 4) & 0x0F).astype(np.float32) + hi_bit * 16.0

    Wblock = np.empty((NB, 8, 32), dtype=np.float32)
    Wblock[:, :, 0:16] = w0 * sc[:, :, None] - mv[:, :, None]
    Wblock[:, :, 16:32] = w1 * sc[:, :, None] - mv[:, :, None]
    return Wblock.reshape(N, nprow, 256).reshape(N, K)


def deq_q6k(pql, pqh, ps_u8, pd_buf, K, N):
    nprow = K // 256
    NB = N * nprow
    ql_b = pql.reshape(NB, 8, 16)
    qh_b = pqh.reshape(NB, 8, 8)
    s_b = ps_u8.reshape(NB, 16).view(np.int8)
    d = pd_buf.reshape(NB, 2).copy().view(np.float16).reshape(NB).astype(np.float32)

    clo = d[:, None] * s_b[:, 0::2].astype(np.float32)
    chi = d[:, None] * s_b[:, 1::2].astype(np.float32)

    hlo = (qh_b[:, :, 0].astype(np.uint32) | (qh_b[:, :, 1].astype(np.uint32) << 8) |
           (qh_b[:, :, 2].astype(np.uint32) << 16) | (qh_b[:, :, 3].astype(np.uint32) << 24))
    hhi = (qh_b[:, :, 4].astype(np.uint32) | (qh_b[:, :, 5].astype(np.uint32) << 8) |
           (qh_b[:, :, 6].astype(np.uint32) << 16) | (qh_b[:, :, 7].astype(np.uint32) << 24))

    SH = np.array([0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30], dtype=np.uint32)

    w0 = (ql_b & 0x0F).astype(np.float32)
    w0 = w0 + (((hlo[:, :, None] >> SH[None, None, :]) & 3).astype(np.float32) * 16.0)
    w0 = (w0 - 32.0) * clo[:, :, None]

    w1 = ((ql_b >> 4) & 0x0F).astype(np.float32)
    w1 = w1 + (((hhi[:, :, None] >> SH[None, None, :]) & 3).astype(np.float32) * 16.0)
    w1 = (w1 - 32.0) * chi[:, :, None]

    Wblock = np.empty((NB, 8, 32), dtype=np.float32)
    Wblock[:, :, 0:16] = w0
    Wblock[:, :, 16:32] = w1
    return Wblock.reshape(N, nprow, 256).reshape(N, K)


# ---------------------------------------------------------------------------
# SG transposed packing (vectorized).  Row-major shuffled section -> SG
# column-major section; bit-identical to the loop-based pack_q{4,5,6}k_sg in
# test_gemv_sg_kernels.py, but implemented as pure numpy reshape/transpose so
# it is fast enough to run over 256 experts.
#
# Block index blk = row*nbpr + bid,  row = h*OPG + lid.
# ---------------------------------------------------------------------------
def _ascon(a):
    return np.ascontiguousarray(a)


def pack_qk_pqs_sg(pqs, K, N, opg=OPG):
    """128-byte low-quant section (Q4_K pqs / Q5_K pqs / Q6_K pql)."""
    nbpr = K // 256
    nrg = N // opg
    src = pqs.reshape(nrg, opg, nbpr, 8, 4, 4)          # [h,lid,bid,j,c,b]
    dst = src.transpose(0, 2, 3, 4, 1, 5)               # [h,bid,j,c,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def pack_qk_psl_sg(psl, K, N, opg=OPG):
    """16-byte scale/min record -> SoA psl_T."""
    nbpr = K // 256
    nrg = N // opg
    p = _ascon(psl.reshape(nrg, opg, nbpr, 16).transpose(0, 2, 1, 3))  # [h,bid,lid,16]
    sl = _ascon(p[..., 0:4]).reshape(nrg, nbpr, opg * 4)
    ml = _ascon(p[..., 4:8]).reshape(nrg, nbpr, opg * 4)
    sh = _ascon(p[..., 8:10]).reshape(nrg, nbpr, opg * 2)
    mh = _ascon(p[..., 10:12]).reshape(nrg, nbpr, opg * 2)
    d = _ascon(p[..., 12:14]).reshape(nrg, nbpr, opg * 2)
    dm = _ascon(p[..., 14:16]).reshape(nrg, nbpr, opg * 2)
    return np.concatenate([sl, ml, sh, mh, d, dm], axis=2).reshape(-1).astype(np.uint8)


def pack_q5k_pqh_sg(pqh, K, N, opg=OPG):
    """32-byte qh section: one 4-byte word per sub-block per lane."""
    nbpr = K // 256
    nrg = N // opg
    src = pqh.reshape(nrg, opg, nbpr, 8, 4)             # [h,lid,bid,j,b]
    dst = src.transpose(0, 2, 3, 1, 4)                  # [h,bid,j,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def pack_q6k_pqh_sg(pqh, K, N, opg=OPG):
    """64-byte qh section: 8 sub-blocks x 2 chunks x 4 bytes per lane."""
    nbpr = K // 256
    nrg = N // opg
    src = pqh.reshape(nrg, opg, nbpr, 8, 2, 4)          # [h,lid,bid,j,c,b]
    dst = src.transpose(0, 2, 3, 4, 1, 5)               # [h,bid,j,c,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def pack_q6k_ps_sg(ps, K, N, opg=OPG):
    """16-byte int8 scales: scale si at si*OPG, 1 byte per lane."""
    nbpr = K // 256
    nrg = N // opg
    src = ps.reshape(nrg, opg, nbpr, 16)                # [h,lid,bid,si]
    dst = src.transpose(0, 2, 3, 1)                     # [h,bid,si,lid]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def pack_q6k_pd_sg(pd, K, N, opg=OPG):
    """2-byte fp16 super-scale: 1 ushort per lane."""
    nbpr = K // 256
    nrg = N // opg
    src = pd.reshape(nrg, opg, nbpr, 2)                 # [h,lid,bid,b]
    dst = src.transpose(0, 2, 1, 3)                     # [h,bid,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Q8_0 shared-expert SG packing (moe_gemv_q80_shared_sg.cl).  Unlike the
# Q4_K/Q5_K/Q6_K routed-expert kernels above (which pack pre-shuffled
# "row-major separated section" buffers), the Q8_0 shared-expert weights are
# packed directly from a standard row-major [N, K] GGML Q8_0 encoding (blocks
# of 32 int8 weights + one fp16 scale per block) -- see the header comment in
# moe_gemv_q80_shared_sg.cl for the exact byte-offset derivation.
#
#   pqs section (256 B/row/256-block): 8 sub-blocks (32 weights) x 8 chunks
#     (4 int8 weights) x OPG lanes x 4 bytes.
#   pd  section (16 B/row/256-block, SoA): 8 sub-block fp16 scales x OPG
#     lanes x 2 bytes.
# ---------------------------------------------------------------------------
def pack_q80_pqs_sg(pqs, K, N, opg=OPG):
    """256-byte per-row-per-256-block Q8_0 weight section (8 sub-blocks x 8
    chunks x 4 int8 bytes), row-major -> SG column-major."""
    nbpr = K // 256
    nrg = N // opg
    src = pqs.reshape(nrg, opg, nbpr, 8, 8, 4)          # [h,lid,bid,j,c,b]
    dst = src.transpose(0, 2, 3, 4, 1, 5)               # [h,bid,j,c,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def pack_q80_pd_sg(pd, K, N, opg=OPG):
    """16-byte per-row-per-256-block SoA fp16 scale section (8 sub-block
    scales), row-major -> SG column-major."""
    nbpr = K // 256
    nrg = N // opg
    src = pd.reshape(nrg, opg, nbpr, 8, 2)              # [h,lid,bid,j,b(fp16)]
    dst = src.transpose(0, 2, 3, 1, 4)                  # [h,bid,j,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def gen_q80_row_major(N, K, seed=1234):
    """Generate a [N, K] float weight matrix and its raw, row-major GGML
    Q8_0 quantization (blocks of 32 int8 weights + one fp32-derived fp16
    scale per block).  Returns (ref_W[N,K] float32, q_bytes[N,K] uint8,
    d_bytes[N, K//32 * 2] uint8 SoA-free / row-major fp16 scale bytes)."""
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


def build_q80_shared_weight(N, K, seed=1234):
    """Build one SG-packed Q8_0 weight matrix [N, K] (a single "shared
    expert" weight, no expert pool/index indirection) and its float32
    reference dequantization."""
    assert K % 256 == 0 and N % OPG == 0
    ref_W, q_bytes, d_bytes = gen_q80_row_major(N, K, seed=seed)
    pqs_T = pack_q80_pqs_sg(q_bytes.reshape(-1), K, N)
    pd_T = pack_q80_pd_sg(d_bytes.reshape(-1), K, N)
    flat = np.concatenate([pqs_T, pd_T])
    return flat, ref_W



# ---------------------------------------------------------------------------
# Expert-pool builders: generate `num_experts` weight matrices [N, K] each,
# shuffle -> SG-pack -> concatenate into one flat buffer indexable as
# `experts + index * expert_size` (SAME expert_size as the shuffle layout).
# ---------------------------------------------------------------------------
def _build_q4k_sg_pool_impl(num_experts, K, N, seed=1234):
    nprow = K // 256
    sz_pqs = nprow * N * 128
    sz_psl = nprow * N * 16
    expert_size = nprow * N * 144
    flat = np.zeros(num_experts * expert_size, dtype=np.uint8)
    ref_W = np.zeros((num_experts, N, K), dtype=np.float32)
    print("build_q4k_sg_pool: experts=%d K=%d N=%d nprow=%d" % (num_experts, K, N, nprow))
    for e in range(num_experts):
        raw = gen_q4k_blocks(nprow * N, seed=seed + e * 97)
        pqs, psl = shuffle_q4k(raw, K, N)
        ref_W[e] = deq_q4k(pqs, psl, K, N)
        pqs_T = pack_qk_pqs_sg(pqs, K, N)
        psl_T = pack_qk_psl_sg(psl, K, N)
        off = e * expert_size
        flat[off: off + sz_pqs] = pqs_T
        flat[off + sz_pqs: off + sz_pqs + sz_psl] = psl_T
    print("build_q4k_sg_pool: done")
    return flat, ref_W, expert_size


def _build_q5k_sg_pool_impl(num_experts, K, N, seed=1234):
    nprow = K // 256
    sz_pqs = nprow * N * 128
    sz_pqh = nprow * N * 32
    sz_psl = nprow * N * 16
    expert_size = nprow * N * 176
    flat = np.zeros(num_experts * expert_size, dtype=np.uint8)
    ref_W = np.zeros((num_experts, N, K), dtype=np.float32)
    print("build_q5k_sg_pool: experts=%d K=%d N=%d nprow=%d" % (num_experts, K, N, nprow))
    for e in range(num_experts):
        raw = gen_q5k_blocks(nprow * N, seed=seed + e * 97)
        pqs, pqh, psl = shuffle_q5k(raw, K, N)
        ref_W[e] = deq_q5k(pqs, pqh, psl, K, N)
        pqs_T = pack_qk_pqs_sg(pqs, K, N)
        pqh_T = pack_q5k_pqh_sg(pqh, K, N)
        psl_T = pack_qk_psl_sg(psl, K, N)
        off = e * expert_size
        flat[off: off + sz_pqs] = pqs_T
        flat[off + sz_pqs: off + sz_pqs + sz_pqh] = pqh_T
        flat[off + sz_pqs + sz_pqh: off + sz_pqs + sz_pqh + sz_psl] = psl_T
    print("build_q5k_sg_pool: done")
    return flat, ref_W, expert_size


def _build_q6k_sg_pool_impl(num_experts, K, N, seed=1234):
    nprow = K // 256
    sz_pql = nprow * N * 128
    sz_pqh = nprow * N * 64
    sz_ps = nprow * N * 16
    sz_pd = nprow * N * 2
    expert_size = nprow * N * 210
    flat = np.zeros(num_experts * expert_size, dtype=np.uint8)
    ref_W = np.zeros((num_experts, N, K), dtype=np.float32)
    print("build_q6k_sg_pool: experts=%d K=%d N=%d nprow=%d" % (num_experts, K, N, nprow))
    for e in range(num_experts):
        raw = gen_q6k_blocks(nprow * N, seed=seed + e * 97)
        pql, pqh, ps, pd = shuffle_q6k(raw, K, N)
        ref_W[e] = deq_q6k(pql, pqh, ps, pd, K, N)
        pql_T = pack_qk_pqs_sg(pql, K, N)
        pqh_T = pack_q6k_pqh_sg(pqh, K, N)
        ps_T = pack_q6k_ps_sg(ps, K, N)
        pd_T = pack_q6k_pd_sg(pd, K, N)
        off = e * expert_size
        flat[off: off + sz_pql] = pql_T
        flat[off + sz_pql: off + sz_pql + sz_pqh] = pqh_T
        flat[off + sz_pql + sz_pqh: off + sz_pql + sz_pqh + sz_ps] = ps_T
        flat[off + sz_pql + sz_pqh + sz_ps: off + expert_size] = pd_T
    print("build_q6k_sg_pool: done")
    return flat, ref_W, expert_size


def _cached_pool(quant, impl_fn):
    def wrapper(num_experts, K, N, seed=1234):
        cached = _load_cached(quant, num_experts, K, N, seed)
        if cached is not None:
            return cached
        flat, ref_W, expert_size = impl_fn(num_experts, K, N, seed=seed)
        _save_cached(quant, num_experts, K, N, seed, flat, ref_W, expert_size)
        return flat, ref_W, expert_size
    return wrapper


build_q4k_sg_pool = _cached_pool("q4k", _build_q4k_sg_pool_impl)
build_q5k_sg_pool = _cached_pool("q5k", _build_q5k_sg_pool_impl)
build_q6k_sg_pool = _cached_pool("q6k", _build_q6k_sg_pool_impl)


def silu(x):
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    sig = np.empty_like(x)
    sig[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    sig[~pos] = ex / (1.0 + ex)
    return (x * sig).astype(np.float32)


def make_routing_indexes(rng, token_len, num_experts, active, aligned_num):
    indexes = np.zeros((token_len, aligned_num), dtype=np.uint32)
    for t in range(token_len):
        indexes[t, :active] = rng.choice(num_experts, size=active, replace=False)
        if aligned_num > active:
            indexes[t, active:] = rng.integers(0, num_experts, aligned_num - active)
    return indexes


# ---------------------------------------------------------------------------
# K-split (occupancy) heuristic for the SG MoE dispatch.
#   base_groups  = number of work-groups at KSPLIT=1
#   split_space  = reduction iterations available to split across sub-groups
# ---------------------------------------------------------------------------
def _pow2_floor(x):
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def choose_ksplit(base_groups, split_space, target=2048, maxk=8):
    force = os.environ.get("MOE_KSPLIT", "")
    if force:
        return int(force)
    if base_groups >= 512:
        return 1
    ks = max(1, round(target / max(base_groups, 1)))
    ks = min(ks, split_space, maxk)
    return _pow2_floor(ks)


_CL_SRC = dict(q4k="moe_gemv_q4k_sg.cl", q5k="moe_gemv_q5k_sg.cl", q6k="moe_gemv_q6k_sg.cl")
_BUILD_POOL = dict(q4k=build_q4k_sg_pool, q5k=build_q5k_sg_pool, q6k=build_q6k_sg_pool)
_BYTES_PER_BLOCK = dict(q4k=144, q5k=176, q6k=210)


# ===========================================================================
# gate_up (group_up_gate) SG kernels: q4k / q5k / q6k
# ===========================================================================
def test_group_up_gate(gpu, hw, quant, K=2048, output_len=512, num_experts=256,
                       active=8, token_len=1, seed=1, verbose=False,
                       iters=1000, warmup=100, do_bench=True):
    label = (f"group_up_gate_{quant}_sg  K={K} output_len={output_len} "
             f"num_experts={num_experts} active={active} token_len={token_len}")
    print(f"\n  [{label}]")
    assert output_len % OPG == 0, "output_len must be a multiple of OPG(16)"

    build_pool = _BUILD_POOL[quant]
    rng = np.random.default_rng(seed)
    ups_flat, ups_ref, expert_size = build_pool(num_experts, K, output_len, seed=seed)
    gates_flat, gates_ref, _ = build_pool(num_experts, K, output_len, seed=seed + 999)

    output_num = active
    aligned_output_num = (output_num + 15) // 16 * 16
    indexes = make_routing_indexes(rng, token_len, num_experts, active, aligned_output_num)
    # Activations are fp16 on the device (kernels read them via
    # intel_sub_group_block_read_us8); reference uses the fp16-rounded values.
    x = rng.standard_normal((token_len, K)).astype(np.float16)
    x_f32 = x.astype(np.float32)

    # ---- reference ----
    ref = np.zeros((token_len, output_num, output_len), dtype=np.float32)
    for t in range(token_len):
        for v in range(output_num):
            idx = int(indexes[t, v])
            up = ups_ref[idx] @ x_f32[t]
            gate = gates_ref[idx] @ x_f32[t]
            ref[t, v] = silu(gate) * up

    # ---- GPU ----
    base_groups = (output_len // OPG) * output_num * token_len
    ksplit = choose_ksplit(base_groups, K // 256)
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit}"
    prog = gpu.build(cl_src(_CL_SRC[quant]), opts)
    krn = cl.Kernel(prog, f"group_up_gate_{quant}_sg")

    x_b = _cbuf(gpu.ctx, x)
    ups_b = _cbuf(gpu.ctx, ups_flat)
    gates_b = _cbuf(gpu.ctx, gates_flat)
    idx_b = _cbuf(gpu.ctx, indexes.ravel())
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY,
                      size=token_len * output_num * output_len * 4)

    krn.set_args(x_b, ups_b, gates_b, idx_b, out_b,
                 np.uint32(output_num), np.uint32(aligned_output_num),
                 np.uint32(token_len), np.uint32(K), np.uint32(output_len))

    gsize = (output_len, ksplit, output_num * token_len)
    lsize = (OPG, ksplit, 1)
    print(f"    KSPLIT={ksplit}  work-groups={base_groups}  live_subgroups={base_groups*ksplit}")

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue)
    ev.wait()
    got = np.empty(token_len * output_num * output_len, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.reshape(token_len, output_num, output_len)

    ok = check_close(f"group_up_gate_{quant}_sg", ref, got, verbose=verbose)
    timing = rl = moved = None
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        timing = stats(ts)
        flops = token_len * output_num * 2 * (2 * output_len * K)
        w_bytes = token_len * output_num * 2 * expert_size
        moved = w_bytes + token_len * K * 2 + token_len * output_num * output_len * 4
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  median={timing['median_ms']:.3f} ms  "
              f"min={timing['min_ms']:.3f} ms  max={timing['max_ms']:.3f} ms  (n={timing['iters']})")
        _print_roofline(rl, moved)
        print(f"    (bytes/block={_BYTES_PER_BLOCK[quant]}, expert_size={expert_size/1e6:.3f} MB, "
              f"weight_bytes={w_bytes/1e6:.1f} MB)")
    return ok, timing, rl, moved


# ===========================================================================
# down_merge SG kernels: q4k / q5k / q6k
# ===========================================================================
def test_down_merge(gpu, hw, quant, K=512, output_len=2048, num_experts=256,
                    active=8, token_len=1, is_acc=False, seed=2, verbose=False,
                    iters=1000, warmup=100, do_bench=True):
    label = (f"down_merge_{quant}_sg  K={K} output_len={output_len} "
             f"num_experts={num_experts} active={active} token_len={token_len} is_acc={is_acc}")
    print(f"\n  [{label}]")
    assert output_len % OPG == 0, "output_len must be a multiple of OPG(16)"

    build_pool = _BUILD_POOL[quant]
    rng = np.random.default_rng(seed)
    downs_flat, downs_ref, expert_size = build_pool(num_experts, K, output_len, seed=seed)

    input_num = active
    aligned_input_num = (input_num + 15) // 16 * 16
    indexes = make_routing_indexes(rng, token_len, num_experts, active, aligned_input_num)
    efficients = rng.uniform(0.1, 1.0, (token_len, aligned_input_num)).astype(np.float32)
    # Activations are fp16 on the device; reference uses the fp16-rounded values.
    inputs = rng.standard_normal((token_len, input_num, K)).astype(np.float16)
    inputs_f32 = inputs.astype(np.float32)
    prev_out = rng.standard_normal((token_len, output_len)).astype(np.float32)

    # ---- reference ----
    ref = np.zeros((token_len, output_len), dtype=np.float32)
    for t in range(token_len):
        acc = np.zeros(output_len, dtype=np.float64)
        for e in range(input_num):
            idx = int(indexes[t, e])
            eff = float(efficients[t, e])
            acc += eff * (downs_ref[idx] @ inputs_f32[t, e])
        if is_acc:
            acc += prev_out[t]
        ref[t] = acc.astype(np.float32)

    # ---- GPU ----
    base_groups = (output_len // OPG) * token_len
    # down_merge is latency-bound (hot==cold): the routed down GEMV only has
    # base_groups=output_len/OPG work-groups (token_len=1), so a KSPLIT of 8
    # leaves the GPU with too few resident sub-groups to hide load/barrier
    # latency.  Allowing KSPLIT up to 16 (maxk=16) doubles the live sub-groups
    # (1024 -> 2048) and is ~15-27% faster here; the single-buffer streaming SLM
    # in the .cl keeps the work-group SLM flat so occupancy is preserved.
    ksplit = choose_ksplit(base_groups, input_num * (K // 256), maxk=16)
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit}"
    prog = gpu.build(cl_src(_CL_SRC[quant]), opts)
    krn = cl.Kernel(prog, f"down_merge_{quant}_sg")

    in_b = _cbuf(gpu.ctx, inputs)
    downs_b = _cbuf(gpu.ctx, downs_flat)
    idx_b = _cbuf(gpu.ctx, indexes.ravel())
    eff_b = _cbuf(gpu.ctx, efficients.ravel())
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_WRITE, size=token_len * output_len * 4)
    if is_acc:
        cl.enqueue_copy(gpu.queue, out_b, prev_out.ravel())
        gpu.queue.finish()

    krn.set_args(in_b, downs_b, idx_b, eff_b, out_b,
                 np.uint32(input_num), np.uint32(aligned_input_num),
                 np.uint32(token_len), np.uint32(K), np.uint32(output_len),
                 np.int32(1 if is_acc else 0))

    gsize = (output_len, ksplit, token_len)
    lsize = (OPG, ksplit, 1)
    print(f"    KSPLIT={ksplit}  work-groups={base_groups}  live_subgroups={base_groups*ksplit}")

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    if is_acc:
        cl.enqueue_copy(gpu.queue, out_b, prev_out.ravel())
        gpu.queue.finish()
    ev = enq(gpu.queue)
    ev.wait()
    got = np.empty(token_len * output_len, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.reshape(token_len, output_len)

    ok = check_close(f"down_merge_{quant}_sg", ref, got, verbose=verbose)
    timing = rl = moved = None
    if do_bench:
        # For is_acc benchmarking, reset prev_out each timed iter is not needed
        # for timing (only correctness), so time the plain enqueue.
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        timing = stats(ts)
        flops = token_len * input_num * 2 * output_len * K
        w_bytes = token_len * input_num * expert_size
        moved = w_bytes + token_len * input_num * K * 2 + token_len * output_len * 4
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  median={timing['median_ms']:.3f} ms  "
              f"min={timing['min_ms']:.3f} ms  max={timing['max_ms']:.3f} ms  (n={timing['iters']})")
        _print_roofline(rl, moved)
        print(f"    (bytes/block={_BYTES_PER_BLOCK[quant]}, expert_size={expert_size/1e6:.3f} MB, "
              f"weight_bytes={w_bytes/1e6:.1f} MB)")
    return ok, timing, rl, moved




# ===========================================================================
# Shared-expert Q8_0 SG kernels (moe_gemv_q80_shared_sg.cl):
#   shared_gate_up_q8_0      shared_down_merge_q8_0
# Single ("shared") expert -- no expert_list/routing-index indirection, only
# a per-token scalar routing weight (`shared_gate`) for the down-merge stage.
# ===========================================================================
_Q80_SHARED_CL_SRC = "moe_gemv_q80_shared_sg.cl"


def test_shared_gate_up_q8_0(gpu, hw, hidden_size=2048, intermediate_size=512,
                             token_len=4, seed=201, verbose=False,
                             iters=1000, warmup=100, do_bench=True):
    label = (f"shared_gate_up_q8_0_sg  hidden_size={hidden_size} "
             f"intermediate_size={intermediate_size} token_len={token_len}")
    print(f"\n  [{label}]")
    assert intermediate_size % OPG == 0, "intermediate_size must be a multiple of OPG(16)"
    assert hidden_size % 256 == 0, "hidden_size must be a multiple of 256"

    gate_flat, gate_ref = build_q80_shared_weight(intermediate_size, hidden_size, seed=seed)
    up_flat, up_ref = build_q80_shared_weight(intermediate_size, hidden_size, seed=seed + 999)

    rng = np.random.default_rng(seed + 1)
    # Activations are fp16 on the device; reference uses the fp16-rounded values.
    x = rng.standard_normal((token_len, hidden_size)).astype(np.float16)
    x_f32 = x.astype(np.float32)

    # ---- reference ----
    ref = np.zeros((token_len, intermediate_size), dtype=np.float32)
    for t in range(token_len):
        gate = gate_ref @ x_f32[t]
        up = up_ref @ x_f32[t]
        ref[t] = silu(gate) * up

    # ---- GPU ----
    base_groups = (intermediate_size // OPG) * token_len
    ksplit = choose_ksplit(base_groups, hidden_size // 256)
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit}"
    prog = gpu.build(cl_src(_Q80_SHARED_CL_SRC), opts)
    krn = cl.Kernel(prog, "shared_gate_up_q8_0")

    x_b = _cbuf(gpu.ctx, x)
    gate_b = _cbuf(gpu.ctx, gate_flat)
    up_b = _cbuf(gpu.ctx, up_flat)
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY,
                      size=token_len * intermediate_size * 2)  # half

    krn.set_args(gate_b, up_b, x_b, out_b,
                 np.uint32(hidden_size), np.uint32(intermediate_size))

    gsize = (intermediate_size, ksplit, token_len)
    lsize = (OPG, ksplit, 1)
    print(f"    KSPLIT={ksplit}  work-groups={base_groups}  live_subgroups={base_groups*ksplit}")

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue)
    ev.wait()
    got = np.empty(token_len * intermediate_size, dtype=np.float16)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.astype(np.float32).reshape(token_len, intermediate_size)

    ok = check_close("shared_gate_up_q8_0_sg", ref, got, verbose=verbose)
    timing = rl = moved = None
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        timing = stats(ts)
        flops = token_len * 2 * (2 * intermediate_size * hidden_size)
        w_bytes = gate_flat.nbytes + up_flat.nbytes
        moved = w_bytes + token_len * hidden_size * 2 + token_len * intermediate_size * 2
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  median={timing['median_ms']:.3f} ms  "
              f"min={timing['min_ms']:.3f} ms  max={timing['max_ms']:.3f} ms  (n={timing['iters']})")
        _print_roofline(rl, moved)
        print(f"    (expert_size={gate_flat.nbytes/1e6:.3f} MB per matrix, "
              f"weight_bytes={w_bytes/1e6:.1f} MB)")
    return ok, timing, rl, moved


def test_shared_down_merge_q8_0(gpu, hw, intermediate_size=512, hidden_size=2048,
                                token_len=4, is_acc=False, seed=202, verbose=False,
                                iters=1000, warmup=100, do_bench=True):
    label = (f"shared_down_merge_q8_0_sg  intermediate_size={intermediate_size} "
             f"hidden_size={hidden_size} token_len={token_len} is_acc={is_acc}")
    print(f"\n  [{label}]")
    assert hidden_size % OPG == 0, "hidden_size must be a multiple of OPG(16)"
    assert intermediate_size % 256 == 0, "intermediate_size must be a multiple of 256"

    down_flat, down_ref = build_q80_shared_weight(hidden_size, intermediate_size, seed=seed)

    rng = np.random.default_rng(seed + 1)
    # Activations are fp16 on the device; reference uses the fp16-rounded values.
    gate_up_in = rng.standard_normal((token_len, intermediate_size)).astype(np.float16)
    gate_up_f32 = gate_up_in.astype(np.float32)
    shared_gate = rng.uniform(0.1, 1.0, token_len).astype(np.float16)
    shared_gate_f32 = shared_gate.astype(np.float32)
    prev_out = rng.standard_normal((token_len, hidden_size)).astype(np.float32)

    # ---- reference ----
    ref = np.zeros((token_len, hidden_size), dtype=np.float32)
    for t in range(token_len):
        acc = shared_gate_f32[t] * (down_ref @ gate_up_f32[t])
        if is_acc:
            acc = acc + prev_out[t]
        ref[t] = acc.astype(np.float32)

    # ---- GPU ----
    base_groups = (hidden_size // OPG) * token_len
    ksplit = choose_ksplit(base_groups, intermediate_size // 256)
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit}"
    prog = gpu.build(cl_src(_Q80_SHARED_CL_SRC), opts)
    krn = cl.Kernel(prog, "shared_down_merge_q8_0")

    in_b = _cbuf(gpu.ctx, gate_up_in)
    down_b = _cbuf(gpu.ctx, down_flat)
    gate_b = _cbuf(gpu.ctx, shared_gate)
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_WRITE, size=token_len * hidden_size * 2)  # half
    if is_acc:
        cl.enqueue_copy(gpu.queue, out_b, prev_out.astype(np.float16).ravel())
        gpu.queue.finish()

    krn.set_args(in_b, down_b, gate_b, out_b,
                 np.uint32(intermediate_size), np.uint32(hidden_size),
                 np.int32(1 if is_acc else 0))

    gsize = (hidden_size, ksplit, token_len)
    lsize = (OPG, ksplit, 1)
    print(f"    KSPLIT={ksplit}  work-groups={base_groups}  live_subgroups={base_groups*ksplit}")

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    if is_acc:
        cl.enqueue_copy(gpu.queue, out_b, prev_out.astype(np.float16).ravel())
        gpu.queue.finish()
    ev = enq(gpu.queue)
    ev.wait()
    got = np.empty(token_len * hidden_size, dtype=np.float16)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.astype(np.float32).reshape(token_len, hidden_size)

    ok = check_close("shared_down_merge_q8_0_sg", ref, got, verbose=verbose)
    timing = rl = moved = None
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        timing = stats(ts)
        flops = token_len * 2 * hidden_size * intermediate_size
        w_bytes = down_flat.nbytes
        moved = w_bytes + token_len * intermediate_size * 2 + token_len * hidden_size * 2
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  median={timing['median_ms']:.3f} ms  "
              f"min={timing['min_ms']:.3f} ms  max={timing['max_ms']:.3f} ms  (n={timing['iters']})")
        _print_roofline(rl, moved)
        print(f"    (expert_size={down_flat.nbytes/1e6:.3f} MB, weight_bytes={w_bytes/1e6:.1f} MB)")
    return ok, timing, rl, moved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--flush_mib", type=int, default=128)
    ap.add_argument("--peak_bw", type=float, default=456.0)
    ap.add_argument("--peak_fp32", type=float, default=9800.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-bench", action="store_true", help="skip timing, only correctness")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the on-disk expert-pool cache (always regenerate)")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="ignore any existing cached expert pools and regenerate + re-save them")
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated substrings; only run matching kernels")
    args = ap.parse_args()

    global _CACHE_ENABLED, _CACHE_DIR
    if args.no_cache:
        _CACHE_ENABLED = False
    if args.cache_dir:
        _CACHE_DIR = args.cache_dir
    if args.refresh_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        for fn in os.listdir(_CACHE_DIR):
            if fn.endswith(".npz") and "sg" in fn:
                os.remove(os.path.join(_CACHE_DIR, fn))
        print(f"  [cache] cleared SG pools in {_CACHE_DIR}")
    print(f"  Expert-pool cache: {'ENABLED dir='+_CACHE_DIR if _CACHE_ENABLED else 'DISABLED'}")

    hw = HW(peak_bw_gbps=args.peak_bw, fp32_gflops=args.peak_fp32)
    gpu = Gpu(args.device)
    if not args.no_bench:
        gpu.setup_flush(args.flush_mib)

    SEP = "=" * 100
    print(SEP)
    print("  MoE OpenCL SG (sub-group block-read) kernel correctness + timing + roofline")
    print(f"  gate_up[token_len=1, K=2048, output_len=512]   down[token_len=1, K=512, output_len=2048]")
    print(f"  256 experts total, top-8-of-256 active routing per token")
    if not args.no_bench:
        print(f"  Peak FP32 : {hw.fp32_gflops:.0f} GFLOPS")
        print(f"  Peak BW   : {hw.peak_bw_gbps:.1f} GB/s   "
              f"(ridge point {hw.fp32_gflops / hw.peak_bw_gbps:.2f} FLOP/B)")
        print(f"  L3 flush  : {args.flush_mib} MiB buffer, 2 sweeps per iteration")
    print(SEP)

    GU_KW = dict(K=2048, output_len=512, token_len=1, num_experts=256, active=8)
    DM_KW = dict(K=512, output_len=2048, token_len=1, num_experts=256, active=8)

    results = []

    only = [f.strip() for f in args.only.split(",") if f.strip()]

    def run(name, fn, **kw):
        if only and not any(f in name for f in only):
            return
        try:
            ok, timing, rl, moved = fn(gpu, hw, verbose=args.verbose, iters=args.iters,
                                       warmup=args.warmup, do_bench=not args.no_bench, **kw)
            results.append((name, ok, timing, rl, moved))
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {name}: {exc}")
            traceback.print_exc()
            results.append((name, False, None, None, None))

    run("group_up_gate_q4k_sg [256,2048,512]", test_group_up_gate, quant="q4k", seed=101, **GU_KW)
    # run("group_up_gate_q5k_sg [256,2048,512]", test_group_up_gate, quant="q5k", seed=102, **GU_KW)
    # run("group_up_gate_q6k_sg [256,2048,512]", test_group_up_gate, quant="q6k", seed=103, **GU_KW)
    run("down_merge_q4k_sg [256,512,2048]", test_down_merge, quant="q4k", is_acc=False, seed=104, **DM_KW)
    # run("down_merge_q5k_sg [256,512,2048]", test_down_merge, quant="q5k", is_acc=False, seed=105, **DM_KW)
    # run("down_merge_q6k_sg [256,512,2048]", test_down_merge, quant="q6k", is_acc=False, seed=106, **DM_KW)

    SH_GU_KW = dict(hidden_size=2048, intermediate_size=512, token_len=4)
    SH_DM_KW = dict(intermediate_size=512, hidden_size=2048, token_len=4)
    # run("shared_gate_up_q8_0_sg [2048->512]", test_shared_gate_up_q8_0, seed=201, **SH_GU_KW)
    # run("shared_down_merge_q8_0_sg [512->2048]", test_shared_down_merge_q8_0, is_acc=False, seed=202, **SH_DM_KW)
    # run("shared_down_merge_q8_0_sg(is_acc) [512->2048]", test_shared_down_merge_q8_0, is_acc=True, seed=203, **SH_DM_KW)

    W = 122
    print("\n" + "=" * W)
    print(f"  {'Kernel':<38} {'Corr':>5}  {'mean ms':>9}  {'BW GB/s':>9}  "
          f"{'BW%':>6}  {'GFLOPS':>8}  {'Roof%':>7}  {'Bound':>7}  {'moved MB':>9}")
    print("-" * W)
    for name, ok, timing, rl, moved in results:
        mean = f"{timing['mean_ms']:.3f}" if timing else "-"
        bw = f"{rl['achieved_bw_gbps']:.1f}" if rl else "-"
        bwp = f"{rl['bw_util_pct']:.1f}%" if rl else "-"
        gf = f"{rl['achieved_gflops']:.1f}" if rl else "-"
        rp = f"{rl['roofline_pct']:.1f}%" if rl else "-"
        bd = rl['bound_by'] if rl else "-"
        mm = f"{moved / 1e6:.1f}" if moved else "-"
        print(f"  {name:<38} {'OK' if ok else 'FAIL':>5}  {mean:>9}  {bw:>9}  "
              f"{bwp:>6}  {gf:>8}  {rp:>7}  {bd:>7}  {mm:>9}")
    print("=" * W)

    n_ok = sum(1 for _, ok, *_ in results if ok)
    print(f"\n  {n_ok}/{len(results)} kernels passed correctness.\n")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
