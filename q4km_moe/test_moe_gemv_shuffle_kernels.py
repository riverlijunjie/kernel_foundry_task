"""
test_moe_gemv_shuffle_kernels.py

Correctness + benchmark + roofline harness for the MoE OpenCL kernels
translated from the SYCL/ESIMD MoE GEMV sources, restricted to the
"real-world" MoE-FFN shapes:

    gate_up projection : token_len=1, K(hidden)=2048, output_len(intermediate)=512
    down    projection : token_len=1, K(intermediate)=512, output_len(hidden)=2048

with 256 experts total and 8 experts selected (activated) per token
(top-8-of-256 routing), matching a typical MoE-FFN configuration.

    moe_gemv_q4k_shuffle.cl
        group_up_gate_q4k  <- runGroupUpAndGateVecQ4KL1_xve_impl  (gemv_q4k_L1.xve.cpp)
        down_merge_q4k     <- runDownMergeQ4KL1_xve_impl          (gemv_q4k_L1.xve.cpp)

    moe_gemv_q5k_shuffle.cl
        group_up_gate_q5k  <- (Q5_K adaptation of runGroupUpAndGateVecQ4KL1_xve_impl,
                                no direct SYCL source; mirrors group_up_gate_q4k structure)
        down_merge_q5k     <- runDownMergeQ5KL1_xve_xe2_impl      (gemv_q5k_L1.xve-xe2.cpp)

    moe_gemv_q6k_shuffle.cl
        group_up_gate_q6k  <- (Q6_K adaptation of runSparseGemvQ6KL1_xve_impl's
                                sub-block dequant/dot machinery, restructured into
                                the fused up+gate/SiLU MoE shape, replacing the
                                plain sparse_gemv_q6k kernel)
        down_merge_q6k     <- runDownMergeQ6KL1_xve_impl          (gemv_q6k_L1.xve.cpp)

For each kernel this script:
  1. Generates a pool of 256 structurally-valid GGUF quant blocks (one weight
     matrix per expert), shuffles them into the kernel's expected on-device
     layout, and uploads them to a single flat `experts` buffer (indexable by
     `index * expert_size`).
  2. Builds inputs / top-8 routing indexes / efficients exactly as the SYCL
     kernels expect.
  3. Runs the OpenCL kernel once and compares against a pure NumPy /
     pure-Python reference implementation (bit-compatible dequantization).
  4. Times `iters` repeated invocations, flushing the L3 cache with a
     dedicated flush kernel between every timed run (so expert weights must
     be re-fetched from DRAM each time, exactly like test_gemv_shuffle_kernels.py),
     and reports timing stats plus a roofline analysis (achieved GB/s vs
     peak, arithmetic intensity, and % of the roofline bound achieved).

Usage:
    python test_moe_gemv_shuffle_kernels.py [--device 0] [--iters 50]
        [--warmup 5] [--flush_mib 256] [--peak_bw 512] [--peak_fp32 9800]
        [--verbose] [--no-bench]
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


def cl_src(name):
    return os.path.join(SCRIPT_DIR, name)


# ---------------------------------------------------------------------------
# Expert-pool disk cache
#
# build_q{4,5,6}k_expert_pool() are expensive (pure-Python per-block
# shuffle/dequant loops over up to 256 experts x thousands of GGUF blocks
# each) and dominate the wall-clock time of this script (>99% in practice).
# Since the pools are fully determined by (quant, num_experts, K, N, seed)
# and the shuffle/reference-dequant code, they can safely be cached to disk
# on first generation and reloaded (near-instant memcpy/np.load) on every
# subsequent run with the same parameters.
#
# Set MOE_EXPERT_POOL_NO_CACHE=1 (or pass --no-cache) to bypass the cache
# entirely, and MOE_EXPERT_POOL_CACHE_DIR (or --cache-dir) to relocate it.
# ---------------------------------------------------------------------------
_CACHE_VERSION = "v1"  # bump if the block layout / shuffle logic changes
_CACHE_ENABLED = os.environ.get("MOE_EXPERT_POOL_NO_CACHE", "0") not in ("1", "true", "True")
_CACHE_DIR = os.environ.get(
    "MOE_EXPERT_POOL_CACHE_DIR",
    os.path.join(SCRIPT_DIR, ".expert_pool_cache"),
)


def _expert_pool_cache_path(quant, num_experts, K, N, seed):
    key = f"{quant}_{_CACHE_VERSION}_ne{num_experts}_K{K}_N{N}_seed{seed}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    fname = f"{key}_{digest}.npz"
    return os.path.join(_CACHE_DIR, fname)


def _load_cached_expert_pool(quant, num_experts, K, N, seed):
    if not _CACHE_ENABLED:
        return None
    path = _expert_pool_cache_path(quant, num_experts, K, N, seed)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            flat = data["flat"]
            ref_W = data["ref_W"]
            expert_size = int(data["expert_size"])
        print(f"    [cache] loaded {quant} expert pool (experts={num_experts} K={K} N={N} "
              f"seed={seed}) from {path}")
        return flat, ref_W, expert_size
    except Exception as exc:
        print(f"    [cache] failed to load {path} ({exc}); rebuilding")
        return None


def _save_cached_expert_pool(quant, num_experts, K, N, seed, flat, ref_W, expert_size):
    if not _CACHE_ENABLED:
        return
    path = _expert_pool_cache_path(quant, num_experts, K, N, seed)
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        # np.savez() appends a ".npz" suffix unless the filename already ends
        # with one, so the temp file itself must end in ".npz" or the
        # subsequent os.replace() would silently target the wrong path.
        tmp_path = path + ".tmp.npz"
        np.savez(tmp_path, flat=flat, ref_W=ref_W, expert_size=np.int64(expert_size))
        os.replace(tmp_path, path)
        print(f"    [cache] saved {quant} expert pool (experts={num_experts} K={K} N={N} "
              f"seed={seed}) to {path}")
    except Exception as exc:
        print(f"    [cache] failed to save {path} ({exc})")


def _cached_expert_pool_builder(quant, impl_fn):
    """Wrap a build_q{4,5,6}k_expert_pool() impl with a transparent disk cache."""
    def wrapper(num_experts, K, N, seed=1234):
        cached = _load_cached_expert_pool(quant, num_experts, K, N, seed)
        if cached is not None:
            return cached
        flat, ref_W, expert_size = impl_fn(num_experts, K, N, seed=seed)
        _save_cached_expert_pool(quant, num_experts, K, N, seed, flat, ref_W, expert_size)
        return flat, ref_W, expert_size
    wrapper.__name__ = impl_fn.__name__.replace("_impl", "")
    return wrapper


# ---------------------------------------------------------------------------
# Inline cache-flush OpenCL kernel (same as test_gemv_shuffle_kernels.py)
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
# Hardware roofline model (identical model to test_gemv_shuffle_kernels.py)
# ---------------------------------------------------------------------------
class HW:
    """Single-precision / DRAM-bandwidth roofline model."""

    def __init__(self, peak_bw_gbps=512.0, fp32_gflops=9800.0):
        self.peak_bw_gbps = peak_bw_gbps
        self.fp32_gflops = fp32_gflops

    def roofline(self, flops, bytes_moved, seconds):
        achieved_gflops = flops / seconds / 1e9
        achieved_bw_gbps = bytes_moved / seconds / 1e9
        ai = flops / max(bytes_moved, 1)          # FLOP / byte
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

    def build(self, src_file):
        if src_file not in self._prog_cache:
            with open(src_file) as f:
                src = f.read()
            self._prog_cache[src_file] = cl.Program(self.ctx, src).build()
        return self._prog_cache[src_file]

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

    def time_kernel(self, enq_fn, iters=50, warmup=5, do_flush=True):
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
    # For large-magnitude accumulations (e.g. K=2048+ MoE-FFN sized reductions)
    # float32 summation-order differences between the GPU kernel and the
    # float64 NumPy reference can produce absolute errors that scale with the
    # result magnitude even though the *relative* error stays tiny. Scale the
    # absolute tolerance with the reference magnitude to avoid false FAILs on
    # those large-scale benchmarks while leaving small-scale tests unaffected.
    eff_atol = max(atol, ref_scale * 1e-6)
    ok = bool(np.allclose(ref, got, rtol=rtol, atol=eff_atol))
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}  max_abs={max_abs:.3e}  rel={rel_err:.3e}")
    if verbose and not ok:
        wi = int(np.argmax(np.abs(ref - got)))
        flat_ref = ref.ravel()
        flat_got = got.ravel()
        print(f"         worst idx={wi}  ref={flat_ref[wi]:.6f}  got={flat_got[wi]:.6f}")
    return ok


def _print_roofline(rl, moved):
    print(f"    BW={rl['achieved_bw_gbps']:.1f} GB/s ({rl['bw_util_pct']:.1f}% of peak)  "
          f"AI={rl['arith_intensity']:.2f} FLOP/B  "
          f"roofline={rl['roofline_pct']:.1f}%  bound={rl['bound_by']}  "
          f"GFLOPS={rl['achieved_gflops']:.1f}")
    print(f"    moved={moved / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# fp16 helper
# ---------------------------------------------------------------------------
def fp16_to_float(b0, b1):
    return float(struct.unpack('e', struct.pack('H', int(b0) | (int(b1) << 8)))[0])


# ---------------------------------------------------------------------------
# Raw GGUF block generators
# ---------------------------------------------------------------------------
def gen_q4k_blocks(n, seed=1234):
    """n x 144-byte block_q4_K layout: d(2) dmin(2) scales(12) qs(128)."""
    rng = np.random.default_rng(seed)
    d = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc = rng.integers(0, 64, (n, 12), dtype=np.uint8)
    qs = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qs], axis=1).ravel()


def gen_q5k_blocks(n, seed=1234):
    """n x 176-byte block_q5_K layout: d(2) dmin(2) scales(12) qh(32) qs(128)."""
    rng = np.random.default_rng(seed)
    d = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc = rng.integers(0, 64, (n, 12), dtype=np.uint8)
    qh = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    qs = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qh, qs], axis=1).ravel()


def gen_q6k_blocks(n, seed=1234):
    """n x 210-byte block_q6_K layout: ql(128) qh(64) scales(16) d(2)."""
    rng = np.random.default_rng(seed)
    ql = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    qh = rng.integers(0, 256, (n, 64), dtype=np.uint8)
    sc = rng.integers(-8, 8, (n, 16), dtype=np.int8).view(np.uint8)
    d = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    return np.concatenate([ql, qh, sc, d], axis=1).ravel()


# ---------------------------------------------------------------------------
# Shuffle: standard GGUF layout -> separated-section kernel layout

# ---------------------------------------------------------------------------
# Shuffle: standard GGUF layout -> separated-section kernel layout
# (identical transforms to test_gemv_shuffle_kernels.py / ShuffleQxKL1)
#
# Vectorized over all NB = N*K//256 blocks at once with NumPy (no per-block
# Python loops), since the raw per-block byte-shuffle math is purely
# elementwise / small-fixed-size and maps cleanly onto batched NumPy ops.
# This is what makes build_q{4,5,6}k_expert_pool() fast (was previously
# >99% of this script's wall-clock time due to Python-level looping over
# up to 256 experts x thousands of blocks each x per-bit operations).
# ---------------------------------------------------------------------------
def shuffle_q4k(raw, K, N):
    NB = N * K // 256
    blk = raw.reshape(NB, 144)
    d_bytes = blk[:, 0:2]
    dmin_bytes = blk[:, 2:4]
    sc = blk[:, 4:16]                     # (NB, 12) uint8
    qs = blk[:, 16:144].reshape(NB, 4, 32)  # (NB, 4, 32) uint8

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
    # d / dmin round-trip through fp16 unchanged, so just copy the raw bytes.
    opsl[:, 12:14] = d_bytes
    opsl[:, 14:16] = dmin_bytes

    return opqs.reshape(-1), opsl.reshape(-1)


def shuffle_q5k(raw, K, N):
    NB = N * K // 256
    blk = raw.reshape(NB, 176)
    d_bytes = blk[:, 0:2]
    dmin_bytes = blk[:, 2:4]
    sc = blk[:, 4:16]                          # (NB, 12) uint8
    qh_r = blk[:, 16:48]                        # (NB, 32) uint8
    qs_r = blk[:, 48:176].reshape(NB, 4, 32)    # (NB, 4, 32) uint8

    # rh[:, g, l] = (qh_r[:, l] >> g) & 1   (derivation: 2*m+bit_pos == g)
    g_arr = np.arange(8, dtype=np.uint8)
    rh = ((qh_r[:, None, :] >> g_arr[None, :, None]) & 1).astype(np.uint8)  # (NB, 8, 32)

    # rw[:, g, l] with g even -> low nibble of qs_r[:, g//2, l]; g odd -> high nibble
    low = qs_r & 0x0F        # (NB, 4, 32) -> g = 0,2,4,6
    high = (qs_r >> 4) & 0x0F  # (NB, 4, 32) -> g = 1,3,5,7
    rw = np.empty((NB, 8, 32), dtype=np.uint8)
    rw[:, 0::2, :] = low
    rw[:, 1::2, :] = high
    rw_flat = rw.reshape(NB, 256)

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

    rw8 = rw_flat.reshape(NB, 8, 32)
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
    sc_r = blk[:, 192:208]           # (NB, 16) raw int8 bytes, copied verbatim
    d_bytes = blk[:, 208:210]        # (NB, 2) raw fp16 bytes, copied verbatim

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
# Reference dequantize (vectorized over all NB blocks at once, NumPy),
# used to build reference weight matrices.
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

    # Reconstruct the 1-bit-per-position high-bit array (NB, 8, 32) from the
    # packed qh bytes: qh_b[:, j, k] bit s corresponds to rh[:, j, k + 4*s].
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

    clo = d[:, None] * s_b[:, 0::2].astype(np.float32)   # (NB, 8)
    chi = d[:, None] * s_b[:, 1::2].astype(np.float32)   # (NB, 8)

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
# Expert-pool builders: generate `num_experts` weight matrices [N, K] each,
# shuffle each into its kernel layout, and concatenate them into a single
# flat buffer indexable as `experts + index * expert_size`.
#
# The per-expert shuffle/dequant calls above are now fully vectorized (all
# NB = N*K//256 blocks processed at once with NumPy), so the remaining
# per-expert Python loop below (<=256 iterations) is cheap.
# ---------------------------------------------------------------------------
def _build_q4k_expert_pool_impl(num_experts, K, N, seed=1234):
    nprow = K // 256
    expert_size_pqs = nprow * N * 128
    expert_size_psl = nprow * N * 16
    all_pqs = np.zeros(num_experts * expert_size_pqs, dtype=np.uint8)
    all_psl = np.zeros(num_experts * expert_size_psl, dtype=np.uint8)
    ref_W = np.zeros((num_experts, N, K), dtype=np.float32)
    print("build_q4k_expert_pool: experts=%d  K=%d  N=%d  nprow=%d" % (num_experts, K, N, nprow))
    for e in range(num_experts):
        raw = gen_q4k_blocks(nprow * N, seed=seed + e * 97)
        pqs, psl = shuffle_q4k(raw, K, N)
        all_pqs[e * expert_size_pqs:(e + 1) * expert_size_pqs] = pqs
        all_psl[e * expert_size_psl:(e + 1) * expert_size_psl] = psl
        ref_W[e] = deq_q4k(pqs, psl, K, N)
    expert_size = nprow * N * 144
    flat = np.zeros(num_experts * expert_size, dtype=np.uint8)
    for e in range(num_experts):
        off = e * expert_size
        flat[off: off + expert_size_pqs] = all_pqs[e * expert_size_pqs:(e + 1) * expert_size_pqs]
        flat[off + expert_size_pqs: off + expert_size_pqs + expert_size_psl] = \
            all_psl[e * expert_size_psl:(e + 1) * expert_size_psl]
    print("build_q4k_expert_pool: experts=%d  K=%d  N=%d  nprow=%d done" % (num_experts, K, N, nprow))
    return flat, ref_W, expert_size


def _build_q5k_expert_pool_impl(num_experts, K, N, seed=1234):
    nprow = K // 256
    sz_pqs = nprow * N * 128
    sz_pqh = nprow * N * 32
    sz_psl = nprow * N * 16
    expert_size = nprow * N * 176
    flat = np.zeros(num_experts * expert_size, dtype=np.uint8)
    ref_W = np.zeros((num_experts, N, K), dtype=np.float32)
    print("build_q5k_expert_pool: experts=%d  K=%d  N=%d  nprow=%d" % (num_experts, K, N, nprow))
    for e in range(num_experts):
        raw = gen_q5k_blocks(nprow * N, seed=seed + e * 97)
        pqs, pqh, psl = shuffle_q5k(raw, K, N)
        off = e * expert_size
        flat[off: off + sz_pqs] = pqs
        flat[off + sz_pqs: off + sz_pqs + sz_pqh] = pqh
        flat[off + sz_pqs + sz_pqh: off + sz_pqs + sz_pqh + sz_psl] = psl
        ref_W[e] = deq_q5k(pqs, pqh, psl, K, N)
    print("build_q5k_expert_pool: experts=%d  K=%d  N=%d  nprow=%d done" % (num_experts, K, N, nprow))
    return flat, ref_W, expert_size


def _build_q6k_expert_pool_impl(num_experts, K, N, seed=1234):
    nprow = K // 256
    sz_pql = nprow * N * 128
    sz_pqh = nprow * N * 64
    sz_ps = nprow * N * 16
    sz_pd = nprow * N * 2
    expert_size = nprow * N * 210
    flat = np.zeros(num_experts * expert_size, dtype=np.uint8)
    ref_W = np.zeros((num_experts, N, K), dtype=np.float32)
    print("build_q6k_expert_pool: experts=%d  K=%d  N=%d  nprow=%d" % (num_experts, K, N, nprow))
    for e in range(num_experts):
        raw = gen_q6k_blocks(nprow * N, seed=seed + e * 97)
        pql, pqh, ps, pd = shuffle_q6k(raw, K, N)
        off = e * expert_size
        flat[off: off + sz_pql] = pql
        flat[off + sz_pql: off + sz_pql + sz_pqh] = pqh
        flat[off + sz_pql + sz_pqh: off + sz_pql + sz_pqh + sz_ps] = ps
        flat[off + sz_pql + sz_pqh + sz_ps: off + expert_size] = pd
        ref_W[e] = deq_q6k(pql, pqh, ps, pd, K, N)
    print("build_q6k_expert_pool: experts=%d  K=%d  N=%d  nprow=%d done" % (num_experts, K, N, nprow))
    return flat, ref_W, expert_size


build_q4k_expert_pool = _cached_expert_pool_builder("q4k", _build_q4k_expert_pool_impl)
build_q5k_expert_pool = _cached_expert_pool_builder("q5k", _build_q5k_expert_pool_impl)
build_q6k_expert_pool = _cached_expert_pool_builder("q6k", _build_q6k_expert_pool_impl)


def silu(x):
    # Numerically-stable SiLU: silu(x) = x * sigmoid(x)
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    sig = np.empty_like(x)
    sig[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    sig[~pos] = ex / (1.0 + ex)
    return (x * sig).astype(np.float32)


# ---------------------------------------------------------------------------
# Top-K-of-N routing index generator: each token independently selects
# `active` distinct experts out of `num_experts` (without replacement),
# matching typical MoE top-k routing.  Padding slots (aligned_num beyond
# `active`) are filled with additional random (possibly repeated) indices
# so the buffer is fully populated, but the kernels are only launched over
# the first `active` groups.
# ---------------------------------------------------------------------------
def make_routing_indexes(rng, token_len, num_experts, active, aligned_num):
    indexes = np.zeros((token_len, aligned_num), dtype=np.uint32)
    for t in range(token_len):
        indexes[t, :active] = rng.choice(num_experts, size=active, replace=False)
        if aligned_num > active:
            indexes[t, active:] = rng.integers(0, num_experts, aligned_num - active)
    return indexes


# ===========================================================================
# gate_up (group_up_gate) kernels: q4k / q5k / q6k
# ===========================================================================
_OPG_UG = 8
_BS_UG = 1
_BYTES_PER_BLOCK = dict(q4k=144, q5k=176, q6k=210)
_CL_SRC = dict(q4k="moe_gemv_q4k_shuffle.cl", q5k="moe_gemv_q5k_shuffle.cl", q6k="moe_gemv_q6k_shuffle.cl")
_BUILD_POOL = dict(q4k=build_q4k_expert_pool, q5k=build_q5k_expert_pool, q6k=build_q6k_expert_pool)


def test_group_up_gate(gpu, hw, quant, K=2048, output_len=512, num_experts=256,
                        active=8, token_len=1, seed=1, verbose=False,
                        iters=50, warmup=5, do_bench=True):
    label = (f"group_up_gate_{quant}  K={K} output_len={output_len} "
             f"num_experts={num_experts} active={active} token_len={token_len}")
    print(f"\n  [{label}]")
    nprow = K // 256
    assert output_len % _OPG_UG == 0
    assert nprow % _BS_UG == 0

    build_pool = _BUILD_POOL[quant]
    rng = np.random.default_rng(seed)
    ups_flat, ups_ref, expert_size = build_pool(num_experts, K, output_len, seed=seed)
    gates_flat, gates_ref, _ = build_pool(num_experts, K, output_len, seed=seed + 999)

    output_num = active
    aligned_output_num = (output_num + 15) // 16 * 16
    indexes = make_routing_indexes(rng, token_len, num_experts, active, aligned_output_num)
    x = rng.standard_normal((token_len, K)).astype(np.float32)

    # ---- reference ----
    ref = np.zeros((token_len, output_num, output_len), dtype=np.float32)
    for t in range(token_len):
        for v in range(output_num):
            idx = int(indexes[t, v])
            up = ups_ref[idx] @ x[t]
            gate = gates_ref[idx] @ x[t]
            ref[t, v] = silu(gate) * up

    # ---- GPU ----
    prog = gpu.build(cl_src(_CL_SRC[quant]))
    krn = cl.Kernel(prog, f"group_up_gate_{quant}")

    x_b = _cbuf(gpu.ctx, x)
    ups_b = _cbuf(gpu.ctx, ups_flat)
    gates_b = _cbuf(gpu.ctx, gates_flat)
    idx_b = _cbuf(gpu.ctx, indexes.ravel())
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY,
                       size=token_len * output_num * output_len * 4)

    krn.set_args(x_b, ups_b, gates_b, idx_b, out_b,
                 np.uint32(output_num), np.uint32(aligned_output_num),
                 np.uint32(token_len), np.uint32(K), np.uint32(output_len))

    local_thread = nprow // _BS_UG
    gsize = ((output_len // _OPG_UG) * local_thread, output_num, token_len)
    lsize = (local_thread, 1, 1)

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue)
    ev.wait()
    got = np.empty(token_len * output_num * output_len, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.reshape(token_len, output_num, output_len)

    ok = check_close(f"group_up_gate_{quant}", ref, got, verbose=verbose)
    timing = None
    rl = None
    moved = None
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        timing = stats(ts)

        # ---- roofline accounting ----
        # flops: for each (token, active-expert) pair, two GEMVs (up + gate)
        #        of shape [output_len, K], each 2*output_len*K FLOPs (mul+add).
        flops = token_len * output_num * 2 * (2 * output_len * K)
        # bytes: naive worst-case DRAM traffic assuming each selected expert's
        #        up+gate weight matrix must be re-fetched per (token,expert)
        #        pair (post L3-flush benchmark methodology), plus reading
        #        the input activations once per token and writing the output.
        bpb = _BYTES_PER_BLOCK[quant]
        w_bytes = token_len * output_num * 2 * expert_size
        moved = w_bytes + token_len * K * 4 + token_len * output_num * output_len * 4
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  median={timing['median_ms']:.3f} ms  "
              f"min={timing['min_ms']:.3f} ms  max={timing['max_ms']:.3f} ms  (n={timing['iters']})")
        _print_roofline(rl, moved)
        print(f"    (bytes/block={bpb}, expert_size={expert_size / 1e6:.3f} MB, "
              f"weight_bytes={w_bytes / 1e6:.1f} MB)")
    return ok, timing, rl, moved


# ===========================================================================
# down_merge kernels: q4k / q5k / q6k
# ===========================================================================
_OPG_DM = 4


def test_down_merge(gpu, hw, quant, K=512, output_len=2048, num_experts=256,
                     active=8, token_len=1, is_acc=False, seed=2, verbose=False,
                     iters=50, warmup=5, do_bench=True):
    label = (f"down_merge_{quant}  K={K} output_len={output_len} "
             f"num_experts={num_experts} active={active} token_len={token_len} is_acc={is_acc}")
    print(f"\n  [{label}]")
    nprow = K // 256
    assert output_len % _OPG_DM == 0

    build_pool = _BUILD_POOL[quant]
    rng = np.random.default_rng(seed)
    downs_flat, downs_ref, expert_size = build_pool(num_experts, K, output_len, seed=seed)

    input_num = active
    aligned_input_num = (input_num + 15) // 16 * 16
    indexes = make_routing_indexes(rng, token_len, num_experts, active, aligned_input_num)
    efficients = rng.uniform(0.1, 1.0, (token_len, aligned_input_num)).astype(np.float32)
    inputs = rng.standard_normal((token_len, input_num, K)).astype(np.float32)
    prev_out = rng.standard_normal((token_len, output_len)).astype(np.float32)

    # ---- reference ----
    ref = np.zeros((token_len, output_len), dtype=np.float32)
    for t in range(token_len):
        acc = np.zeros(output_len, dtype=np.float64)
        for e in range(input_num):
            idx = int(indexes[t, e])
            eff = float(efficients[t, e])
            acc += eff * (downs_ref[idx] @ inputs[t, e])
        if is_acc:
            acc += prev_out[t]
        ref[t] = acc.astype(np.float32)

    # ---- GPU ----
    prog = gpu.build(cl_src(_CL_SRC[quant]))
    krn = cl.Kernel(prog, f"down_merge_{quant}")

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

    gsize = ((output_len // _OPG_DM) * nprow, token_len)
    lsize = (nprow, 1)

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue)
    ev.wait()
    got = np.empty(token_len * output_len, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.reshape(token_len, output_len)

    ok = check_close(f"down_merge_{quant}", ref, got, verbose=verbose)
    timing = None
    rl = None
    moved = None
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        timing = stats(ts)

        # ---- roofline accounting ----
        # flops: for each (token, active-expert) pair, one GEMV [output_len, K]
        flops = token_len * input_num * 2 * output_len * K
        # bytes: naive worst-case DRAM traffic -- one expert weight matrix
        #        fetched per (token, expert) pair, plus input activations and
        #        output writes.
        bpb = _BYTES_PER_BLOCK[quant]
        w_bytes = token_len * input_num * expert_size
        moved = w_bytes + token_len * input_num * K * 4 + token_len * output_len * 4
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  median={timing['median_ms']:.3f} ms  "
              f"min={timing['min_ms']:.3f} ms  max={timing['max_ms']:.3f} ms  (n={timing['iters']})")
        _print_roofline(rl, moved)
        print(f"    (bytes/block={bpb}, expert_size={expert_size / 1e6:.3f} MB, "
              f"weight_bytes={w_bytes / 1e6:.1f} MB)")
    return ok, timing, rl, moved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--flush_mib", type=int, default=256)
    ap.add_argument("--peak_bw", type=float, default=512.0,
                    help="Peak DRAM bandwidth in GB/s (default: 512)")
    ap.add_argument("--peak_fp32", type=float, default=9800.0,
                    help="Peak FP32 throughput in GFLOPS (default: 9800)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-bench", action="store_true", help="skip timing, only correctness")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the on-disk expert-pool cache (always regenerate)")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="ignore any existing cached expert pools and regenerate + re-save them")
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="directory for cached expert pools "
                         "(default: %(default)s, or MOE_EXPERT_POOL_CACHE_DIR / "
                         "<script_dir>/.expert_pool_cache)")
    args = ap.parse_args()

    global _CACHE_ENABLED, _CACHE_DIR
    if args.no_cache:
        _CACHE_ENABLED = False
    if args.cache_dir:
        _CACHE_DIR = args.cache_dir
    if args.refresh_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        for fn in os.listdir(_CACHE_DIR):
            if fn.endswith(".npz"):
                os.remove(os.path.join(_CACHE_DIR, fn))
        print(f"  [cache] cleared {_CACHE_DIR}")
    if _CACHE_ENABLED:
        print(f"  Expert-pool cache: ENABLED  dir={_CACHE_DIR}")
    else:
        print("  Expert-pool cache: DISABLED")

    hw = HW(peak_bw_gbps=args.peak_bw, fp32_gflops=args.peak_fp32)
    gpu = Gpu(args.device)
    if not args.no_bench:
        gpu.setup_flush(args.flush_mib)

    SEP = "=" * 100
    print(SEP)
    print("  MoE OpenCL shuffle-kernel correctness + timing + roofline")
    print(f"  gate_up[token_len=1, K=2048, output_len=512]   "
          f"down[token_len=1, K=512, output_len=2048]")
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

    def run(name, fn, **kw):
        try:
            ok, timing, rl, moved = fn(gpu, hw, verbose=args.verbose, iters=args.iters,
                                        warmup=args.warmup, do_bench=not args.no_bench, **kw)
            results.append((name, ok, timing, rl, moved))
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {name}: {exc}")
            traceback.print_exc()
            results.append((name, False, None, None, None))

    run("group_up_gate_q4k [256,2048,512]", test_group_up_gate, quant="q4k", seed=101, **GU_KW)
    run("group_up_gate_q5k [256,2048,512]", test_group_up_gate, quant="q5k", seed=102, **GU_KW)
    run("group_up_gate_q6k [256,2048,512]", test_group_up_gate, quant="q6k", seed=103, **GU_KW)
    run("down_merge_q4k [256,512,2048]", test_down_merge, quant="q4k", is_acc=False, seed=104, **DM_KW)
    run("down_merge_q5k [256,512,2048]", test_down_merge, quant="q5k", is_acc=False, seed=105, **DM_KW)
    run("down_merge_q6k [256,512,2048]", test_down_merge, quant="q6k", is_acc=False, seed=106, **DM_KW)

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
