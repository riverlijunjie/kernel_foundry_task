"""
test_gemv_shuffle_kernels.py

Benchmark + correctness harness for Q4_K, Q5_K and Q6_K shuffled-layout OpenCL GEMV kernels.

Benchmark cases (token_len=1 GEMV):
    Q6K_K4096_N151936  Q6K_K12288_N4096  Q4K_K12288_N4096
    Q4K_K4096_N12288   Q4K_K4096_N4096   Q6K_K4096_N1024  Q4K_K4096_N1024
    Q5K_K4096_N8192    Q5K_K8192_N4096

Per case:
  1. Builds shuffled weight buffers on GPU (proper GGUF blocks for the first
     `verify_rows` output rows, random bytes for the rest).
  2. Runs one kernel invocation and verifies the first `verify_rows` outputs
     against a pure-Python reference dequantizer.
  3. Runs `iters` (default 500) timed iterations using CL profiling events.
     A cache-flush kernel (reads a buffer >> L3 size) is issued between every
     timed run so weights must be re-fetched from DRAM each time.
  4. Reports timing statistics (mean/median/min/max) and a roofline analysis
     (achieved BW vs peak, arithmetic intensity, roofline % vs memory or
     compute bound).

Usage:
    python test_gemv_shuffle_kernels.py [--device 0]
        [--iters 500] [--warmup 10] [--flush_mib 256]
        [--peak_bw 512] [--peak_fp32 9800]
        [--verify_rows 64] [--verbose]
"""

import os
import struct
import statistics
import time
import argparse

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    raise SystemExit("pyopencl is required: pip install pyopencl")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def cl_src(name):
    return os.path.join(SCRIPT_DIR, name)


# ---------------------------------------------------------------------------
# Inline cache-flush OpenCL kernel
# Reads through a large buffer to evict L3/LLC cache lines.
# ---------------------------------------------------------------------------
_FLUSH_SRC = r"""
__kernel void cache_flush(__global float* buf, uint n, __global float* sink) {
    uint gid    = get_global_id(0);
    uint stride = get_global_size(0);
    float acc   = 0.0f;
    for (uint i = gid; i < n; i += stride)
        acc += buf[i];
    /* write to prevent DCE while keeping output traffic tiny */
    if (gid < 4096u) sink[gid] += acc;
}
"""

# ---------------------------------------------------------------------------
# Hardware roofline model
# ---------------------------------------------------------------------------
class HW:
    """Single-precision / DRAM-bandwidth roofline model."""

    def __init__(self, peak_bw_gbps=512.0, fp32_gflops=9800.0):
        self.peak_bw_gbps = peak_bw_gbps
        self.fp32_gflops  = fp32_gflops

    def roofline(self, flops, bytes_moved, seconds):
        achieved_gflops  = flops       / seconds / 1e9
        achieved_bw_gbps = bytes_moved / seconds / 1e9
        ai               = flops / max(bytes_moved, 1)         # FLOP / byte
        ridge            = self.fp32_gflops / max(self.peak_bw_gbps, 1e-12)
        if ai < ridge:
            roof_gflops = ai * self.peak_bw_gbps
            bound       = "memory"
        else:
            roof_gflops = self.fp32_gflops
            bound       = "compute"
        return dict(
            achieved_gflops  = achieved_gflops,
            achieved_bw_gbps = achieved_bw_gbps,
            arith_intensity  = ai,
            ridge_point      = ridge,
            roofline_gflops  = roof_gflops,
            roofline_pct     = achieved_gflops / max(roof_gflops, 1e-12) * 100.0,
            bw_util_pct      = achieved_bw_gbps / max(self.peak_bw_gbps, 1e-12) * 100.0,
            bound_by         = bound,
        )


# ---------------------------------------------------------------------------
# GPU context: build cache, L3 flush, CL-profiling-event timer
# ---------------------------------------------------------------------------
class Gpu:
    def __init__(self, device_idx=0):
        platforms = cl.get_platforms()
        devices   = [d for p in platforms for d in p.get_devices()]
        if not devices:
            raise RuntimeError("No OpenCL devices found")
        self.device = devices[device_idx % len(devices)]
        print(f"Device [{device_idx}]: {self.device.name}")
        self.ctx   = cl.Context([self.device])
        self.queue = cl.CommandQueue(
            self.ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        self._prog_cache = {}
        self._flush_krn  = None
        self._flush_buf  = None
        self._flush_sink = None
        self._flush_n    = 0

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
        """Allocate a flush buffer larger than the GPU L3 cache."""
        n    = mib * 1024 * 1024 // 4
        self._flush_n = n
        mf   = cl.mem_flags
        zeros = np.zeros(n, dtype=np.float32)
        self._flush_buf  = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self._flush_sink = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=4096 * 4)
        prog = self._build_src(_FLUSH_SRC, "_flush")
        self._flush_krn  = cl.Kernel(prog, "cache_flush")
        self.queue.finish()

    def flush_l3(self, sweeps=2):
        """Read through the flush buffer to evict all L3 lines."""
        if self._flush_krn is None:
            return
        gsz = (min(self._flush_n, 1 << 20),)
        for _ in range(sweeps):
            self._flush_krn(self.queue, gsz, None,
                            self._flush_buf,
                            np.uint32(self._flush_n),
                            self._flush_sink)
        self.queue.finish()

    def time_kernel(self, enq_fn, iters=500, warmup=10, do_flush=True):
        """
        enq_fn(queue) must return the cl.Event of the primary kernel.
        Returns a list of device-side execution times in seconds
        (CL profiling: END - START nanoseconds).
        """
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
        mean_ms   = statistics.mean(ts)   * 1e3,
        median_ms = statistics.median(ts) * 1e3,
        min_ms    = min(ts) * 1e3,
        max_ms    = max(ts) * 1e3,
        iters     = len(ts),
    )


# ---------------------------------------------------------------------------
# fp16 helper
# ---------------------------------------------------------------------------
def fp16_to_float(b0, b1):
    return float(struct.unpack('e', struct.pack('H', int(b0) | (int(b1) << 8)))[0])


# ---------------------------------------------------------------------------
# Fast numpy-vectorised block generators
# (produce raw GGUF bytes that are structurally valid but random-valued)
# ---------------------------------------------------------------------------
def gen_q4k_blocks(n, seed=1234):
    """n x 144-byte block_q4_K layout: d(2) dmin(2) scales(12) qs(128)."""
    rng = np.random.default_rng(seed)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm  = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc  = rng.integers(0, 64,  (n, 12),  dtype=np.uint8)
    qs  = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qs], axis=1).ravel()   # [n * 144]


def gen_q5k_blocks(n, seed=1234):
    """n x 176-byte block_q5_K layout: d(2) dmin(2) scales(12) qh(32) qs(128)."""
    rng = np.random.default_rng(seed)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm  = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc  = rng.integers(0, 64,  (n, 12),  dtype=np.uint8)
    qh  = rng.integers(0, 256, (n,  32), dtype=np.uint8)
    qs  = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qh, qs], axis=1).ravel()   # [n * 176]


def gen_q6k_blocks(n, seed=1234):
    """n x 210-byte block_q6_K layout: ql(128) qh(64) scales(16) d(2)."""
    rng = np.random.default_rng(seed)
    ql  = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    qh  = rng.integers(0, 256, (n,  64), dtype=np.uint8)
    sc  = rng.integers(-8, 8,  (n,  16), dtype=np.int8).view(np.uint8)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    return np.concatenate([ql, qh, sc, d], axis=1).ravel()   # [n * 210]


# ---------------------------------------------------------------------------
# Shuffle: standard GGUF layout -> separated-section kernel layout
# ---------------------------------------------------------------------------
def shuffle_q4k(raw, K, N):
    """
    ShuffleQ4KL1.
    Input : raw bytes [N*K/256 * 144] (block_q4_K blocks)
    Output: pqs [NB*128], psl [NB*16]  where NB = N*K/256
    """
    NB  = N * K // 256
    pqs = np.zeros(NB * 128, dtype=np.uint8)
    psl = np.zeros(NB * 16,  dtype=np.uint8)
    for b in range(NB):
        blk  = raw[b * 144:(b + 1) * 144]
        d    = fp16_to_float(blk[0], blk[1])
        dmin = fp16_to_float(blk[2], blk[3])
        sc   = blk[4:16];   qs = blk[16:144]

        rw = np.zeros(256, dtype=np.uint8)
        for j in range(4):
            for k in range(32):
                rw[j * 64 + k]      = qs[j * 32 + k] & 0x0F
                rw[j * 64 + 32 + k] = (qs[j * 32 + k] >> 4) & 0x0F

        rs = np.zeros(8, np.uint8);  rm = np.zeros(8, np.uint8)
        rs[0] = sc[0] & 63;  rs[1] = sc[1] & 63;  rs[2] = sc[2] & 63;  rs[3] = sc[3] & 63
        rs[4] = (sc[8]  & 0x0F) | ((sc[0]  >> 2) & 0x30)
        rs[5] = (sc[9]  & 0x0F) | ((sc[1]  >> 2) & 0x30)
        rs[6] = (sc[10] & 0x0F) | ((sc[2]  >> 2) & 0x30)
        rs[7] = (sc[11] & 0x0F) | ((sc[3]  >> 2) & 0x30)
        rm[0] = sc[4] & 63;  rm[1] = sc[5] & 63;  rm[2] = sc[6] & 63;  rm[3] = sc[7] & 63
        rm[4] = ((sc[8]  >> 4) & 0x0F) | ((sc[4] >> 2) & 0x30)
        rm[5] = ((sc[9]  >> 4) & 0x0F) | ((sc[5] >> 2) & 0x30)
        rm[6] = ((sc[10] >> 4) & 0x0F) | ((sc[6] >> 2) & 0x30)
        rm[7] = ((sc[11] >> 4) & 0x0F) | ((sc[7] >> 2) & 0x30)

        opqs = pqs[b * 128:(b + 1) * 128]
        for j in range(8):
            for k in range(16):
                opqs[j * 16 + k] = (rw[j * 32 + k] & 0x0F) | ((rw[j * 32 + 16 + k] & 0x0F) << 4)

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


def shuffle_q5k(raw, K, N):
    """
    ShuffleQ5KL1.
    Input : raw bytes [N*K/256 * 176] (block_q5_K blocks: d(2) dmin(2) scales(12) qh(32) qs(128))
    Output: pqs [NB*128], pqh [NB*32], psl [NB*16]  where NB = N*K/256

    Reference dequant (llama.cpp block_q5_K), for 32-weight group g=0..7:
        m = g // 2 ; bit_pos = g % 2
        for l in 0..31:
            nibble  = (qs_r[m*32 + l] >> (4*bit_pos)) & 0xF
            hi_bit  = (qh_r[l] >> (2*m + bit_pos)) & 1
            weight[g*32 + l] = nibble + (hi_bit << 4)
        scale/min index "is" == g  (get_scale_min_k4 identical to Q4_K)

    Shuffled layout (see gemv_q5k_shuffle.cl header):
        pqs[j*16+k] = low4(w[j*32+k]) | (low4(w[j*32+16+k]) << 4)     j=0..7, k=0..15
        bit s of pqh[j*4+k] = high_bit(w[j*32 + k + s*4])             j=0..7, k=0..3, s=0..7
        psl: identical 16-byte scale/min/d/dmin packing as Q4_K
    """
    NB  = N * K // 256
    pqs = np.zeros(NB * 128, dtype=np.uint8)
    pqh = np.zeros(NB * 32,  dtype=np.uint8)
    psl = np.zeros(NB * 16,  dtype=np.uint8)
    for b in range(NB):
        blk   = raw[b * 176:(b + 1) * 176]
        d     = fp16_to_float(blk[0], blk[1])
        dmin  = fp16_to_float(blk[2], blk[3])
        sc    = blk[4:16]
        qh_r  = blk[16:48]
        qs_r  = blk[48:176]

        rw = np.zeros(256, dtype=np.uint8)   # low nibble (0..15) per weight
        rh = np.zeros(256, dtype=np.uint8)   # high bit   (0/1)   per weight
        for g in range(8):
            m = g // 2
            bit_pos = g % 2
            for l in range(32):
                rw[g * 32 + l] = (qs_r[m * 32 + l] >> (4 * bit_pos)) & 0x0F
                rh[g * 32 + l] = (qh_r[l] >> (2 * m + bit_pos)) & 0x01

        rs = np.zeros(8, np.uint8);  rm = np.zeros(8, np.uint8)
        rs[0] = sc[0] & 63;  rs[1] = sc[1] & 63;  rs[2] = sc[2] & 63;  rs[3] = sc[3] & 63
        rs[4] = (sc[8]  & 0x0F) | ((sc[0]  >> 2) & 0x30)
        rs[5] = (sc[9]  & 0x0F) | ((sc[1]  >> 2) & 0x30)
        rs[6] = (sc[10] & 0x0F) | ((sc[2]  >> 2) & 0x30)
        rs[7] = (sc[11] & 0x0F) | ((sc[3]  >> 2) & 0x30)
        rm[0] = sc[4] & 63;  rm[1] = sc[5] & 63;  rm[2] = sc[6] & 63;  rm[3] = sc[7] & 63
        rm[4] = ((sc[8]  >> 4) & 0x0F) | ((sc[4] >> 2) & 0x30)
        rm[5] = ((sc[9]  >> 4) & 0x0F) | ((sc[5] >> 2) & 0x30)
        rm[6] = ((sc[10] >> 4) & 0x0F) | ((sc[6] >> 2) & 0x30)
        rm[7] = ((sc[11] >> 4) & 0x0F) | ((sc[7] >> 2) & 0x30)

        opqs = pqs[b * 128:(b + 1) * 128]
        for j in range(8):
            for k in range(16):
                opqs[j * 16 + k] = (rw[j * 32 + k] & 0x0F) | ((rw[j * 32 + 16 + k] & 0x0F) << 4)

        opqh = pqh[b * 32:(b + 1) * 32]
        for j in range(8):
            for k in range(4):
                byte = 0
                for s in range(8):
                    wi = k + s * 4
                    byte |= int(rh[j * 32 + wi]) << s
                opqh[j * 4 + k] = byte

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
    return pqs, pqh, psl


def shuffle_q6k(raw, K, N):
    """
    ShuffleQ6KL1.
    Input : raw bytes [N*K/256 * 210] (block_q6_K blocks)
    Output: pql [NB*128], pqh [NB*64], ps_u8 [NB*16], pd [NB*2]
    """
    NB  = N * K // 256
    pql = np.zeros(NB * 128, dtype=np.uint8)
    pqh = np.zeros(NB * 64,  dtype=np.uint8)
    ps  = np.zeros(NB * 16,  dtype=np.int8)
    pd  = np.zeros(NB * 2,   dtype=np.uint8)
    for b in range(NB):
        blk  = raw[b * 210:(b + 1) * 210]
        ql_r = blk[0:128];  qh_r = blk[128:192]
        sc_r = blk[192:208].view(np.int8)
        d    = fp16_to_float(blk[208], blk[209])

        rw = np.zeros(256, np.uint8)
        for j in range(2):
            for k in range(64):
                rw[j * 128 + k]      = ql_r[j * 64 + k] & 0x0F
                rw[j * 128 + 64 + k] = (ql_r[j * 64 + k] >> 4) & 0x0F
            for k in range(32):
                rw[j * 128 + k]       += ((qh_r[j * 32 + k] & 0x03) << 4)
                rw[j * 128 + 32 + k]  += ((qh_r[j * 32 + k] & 0x0C) << 2)
                rw[j * 128 + 64 + k]  +=   qh_r[j * 32 + k] & 0x30
                rw[j * 128 + 96 + k]  += ((qh_r[j * 32 + k] & 0xC0) >> 2)

        opql = pql[b * 128:(b + 1) * 128]
        for j in range(8):
            for k in range(16):
                opql[j * 16 + k] = (rw[j * 32 + k] & 0x0F) | ((rw[j * 32 + 16 + k] & 0x0F) << 4)

        opqh = pqh[b * 64:(b + 1) * 64]
        for j in range(16):
            for k in range(4):
                opqh[j * 4 + k] = (((rw[j*16+k]    & 0x30) >> 4)
                                 |  ((rw[j*16+k+4]  & 0x30) >> 2)
                                 |   (rw[j*16+k+8]  & 0x30)
                                 |  ((rw[j*16+k+12] & 0x30) << 2))

        ps[b * 16:(b + 1) * 16] = sc_r
        pd[b * 2:(b + 1) * 2]   = np.frombuffer(struct.pack('e', d), dtype=np.uint8)
    return pql, pqh, ps.view(np.uint8), pd


# ---------------------------------------------------------------------------
# Reference dequantize (pure Python, used for correctness only)
# ---------------------------------------------------------------------------
def deq_q4k(pqs, psl, K, N):
    """Decode shuffled Q4_K -> W[N, K] float32."""
    nprow = K // 256
    W = np.zeros((N, K), np.float32)
    for row in range(N):
        for bi in range(nprow):
            idx  = row * nprow + bi
            qs_b = pqs[idx * 128:(idx + 1) * 128]
            sl_b = psl[idx * 16: (idx + 1) * 16]
            sli  = int(sl_b[0]) | (int(sl_b[1])<<8) | (int(sl_b[2])<<16) | (int(sl_b[3])<<24)
            mli  = int(sl_b[4]) | (int(sl_b[5])<<8) | (int(sl_b[6])<<16) | (int(sl_b[7])<<24)
            shi  = int(sl_b[8])  | (int(sl_b[9])<<8)
            mhi  = int(sl_b[10]) | (int(sl_b[11])<<8)
            d    = fp16_to_float(sl_b[12], sl_b[13])
            dmin = fp16_to_float(sl_b[14], sl_b[15])
            for j in range(8):
                sq = ((sli >> (j * 4)) & 0xF) | (((shi >> (j * 2)) & 3) << 4)
                mq = ((mli >> (j * 4)) & 0xF) | (((mhi >> (j * 2)) & 3) << 4)
                sc = sq * d;  mv = mq * dmin
                for k in range(16):
                    byte = int(qs_b[j * 16 + k])
                    cb   = bi * 256 + j * 32
                    W[row, cb + k]      = (byte & 0x0F) * sc - mv
                    W[row, cb + 16 + k] = ((byte >> 4) & 0x0F) * sc - mv
    return W


def deq_q5k(pqs, pqh, psl, K, N):
    """Decode shuffled Q5_K -> W[N, K] float32."""
    nprow = K // 256
    W = np.zeros((N, K), np.float32)
    for row in range(N):
        for bi in range(nprow):
            idx  = row * nprow + bi
            qs_b = pqs[idx * 128:(idx + 1) * 128]
            qh_b = pqh[idx * 32: (idx + 1) * 32]
            sl_b = psl[idx * 16: (idx + 1) * 16]
            sli  = int(sl_b[0]) | (int(sl_b[1])<<8) | (int(sl_b[2])<<16) | (int(sl_b[3])<<24)
            mli  = int(sl_b[4]) | (int(sl_b[5])<<8) | (int(sl_b[6])<<16) | (int(sl_b[7])<<24)
            shi  = int(sl_b[8])  | (int(sl_b[9])<<8)
            mhi  = int(sl_b[10]) | (int(sl_b[11])<<8)
            d    = fp16_to_float(sl_b[12], sl_b[13])
            dmin = fp16_to_float(sl_b[14], sl_b[15])
            for j in range(8):
                sq = ((sli >> (j * 4)) & 0xF) | (((shi >> (j * 2)) & 3) << 4)
                mq = ((mli >> (j * 4)) & 0xF) | (((mhi >> (j * 2)) & 3) << 4)
                sc = sq * d;  mv = mq * dmin
                qh_word = (int(qh_b[j*4]) | (int(qh_b[j*4+1])<<8)
                         | (int(qh_b[j*4+2])<<16) | (int(qh_b[j*4+3])<<24))
                cb = bi * 256 + j * 32
                for k in range(16):
                    byte = int(qs_b[j * 16 + k])

                    w0 = float(byte & 0x0F)
                    bidx0 = k % 4; bpos0 = k // 4
                    if (qh_word >> (bidx0 * 8 + bpos0)) & 1:
                        w0 += 16.0
                    W[row, cb + k] = w0 * sc - mv

                    w1 = float((byte >> 4) & 0x0F)
                    bidx1 = (k + 16) % 4; bpos1 = (k + 16) // 4
                    if (qh_word >> (bidx1 * 8 + bpos1)) & 1:
                        w1 += 16.0
                    W[row, cb + 16 + k] = w1 * sc - mv
    return W


def deq_q6k(pql, pqh, ps_u8, pd_buf, K, N):
    """Decode shuffled Q6_K -> W[N, K] float32."""
    nprow = K // 256
    ps    = ps_u8.view(np.int8)
    SH    = [0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30]
    W     = np.zeros((N, K), np.float32)
    for row in range(N):
        for bi in range(nprow):
            idx  = row * nprow + bi
            ql_b = pql[idx * 128:(idx + 1) * 128]
            qh_b = pqh[idx * 64: (idx + 1) * 64]
            s_b  = ps [idx * 16: (idx + 1) * 16]
            d    = fp16_to_float(pd_buf[idx * 2], pd_buf[idx * 2 + 1])
            for j in range(8):
                clo = d * float(s_b[2 * j]);  chi = d * float(s_b[2 * j + 1])
                hlo = (int(qh_b[j*8])   | (int(qh_b[j*8+1])<<8)
                     | (int(qh_b[j*8+2])<<16) | (int(qh_b[j*8+3])<<24))
                hhi = (int(qh_b[j*8+4]) | (int(qh_b[j*8+5])<<8)
                     | (int(qh_b[j*8+6])<<16) | (int(qh_b[j*8+7])<<24))
                for k in range(16):
                    qlb = int(ql_b[j * 16 + k]);  sh = SH[k]
                    w0  = float((qlb & 0x0F) | (((hlo >> sh) & 3) << 4)); w0 = (w0 - 32.0) * clo
                    w1  = float(((qlb >> 4) & 0x0F) | (((hhi >> sh) & 3) << 4)); w1 = (w1 - 32.0) * chi
                    cb  = bi * 256 + j * 32
                    W[row, cb + k]      = w0
                    W[row, cb + 16 + k] = w1
    return W


# ---------------------------------------------------------------------------
# GPU buffer builders
# Generates proper shuffled blocks for the first `vr` output rows;
# fills the remaining rows with random bytes (structurally irrelevant for
# timing, but required for a full-size DRAM-read benchmark).
# ---------------------------------------------------------------------------
def _cbuf(ctx, arr):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=np.ascontiguousarray(arr))


def make_q4k_bufs(gpu, K, N, vr=64, seed=1234):
    """
    Returns: pqs_buf, psl_buf, x_buf, y_ref[vr] (or None), vr
    """
    nprow = K // 256
    total = N * nprow
    vr    = min(vr, N)
    rng   = np.random.default_rng(seed ^ 0xCAFE_1234)

    pqs_f = np.empty(total * 128, dtype=np.uint8)
    psl_f = np.empty(total * 16,  dtype=np.uint8)

    pqs_v = psl_v = None
    if vr > 0:
        raw   = gen_q4k_blocks(vr * nprow, seed=seed)
        pqs_v, psl_v = shuffle_q4k(raw, K, vr)
        pqs_f[:vr * nprow * 128] = pqs_v
        psl_f[:vr * nprow * 16]  = psl_v

    rest = (N - vr) * nprow
    if rest > 0:
        pqs_f[vr * nprow * 128:] = rng.integers(0, 256, rest * 128, dtype=np.uint8)
        psl_f[vr * nprow * 16:]  = rng.integers(0, 256, rest * 16,  dtype=np.uint8)

    x     = rng.standard_normal(K).astype(np.float32)
    y_ref = (deq_q4k(pqs_v, psl_v, K, vr) @ x) if vr > 0 else None

    return (_cbuf(gpu.ctx, pqs_f), _cbuf(gpu.ctx, psl_f),
            _cbuf(gpu.ctx, x), y_ref, vr)


def make_q5k_bufs(gpu, K, N, vr=64, seed=1234):
    """
    Returns: pqs_buf, pqh_buf, psl_buf, x_buf, y_ref[vr] (or None), vr
    """
    nprow = K // 256
    total = N * nprow
    vr    = min(vr, N)
    rng   = np.random.default_rng(seed ^ 0xCAFE_1234)

    pqs_f = np.empty(total * 128, dtype=np.uint8)
    pqh_f = np.empty(total * 32,  dtype=np.uint8)
    psl_f = np.empty(total * 16,  dtype=np.uint8)

    pqs_v = pqh_v = psl_v = None
    if vr > 0:
        raw = gen_q5k_blocks(vr * nprow, seed=seed)
        pqs_v, pqh_v, psl_v = shuffle_q5k(raw, K, vr)
        pqs_f[:vr * nprow * 128] = pqs_v
        pqh_f[:vr * nprow * 32]  = pqh_v
        psl_f[:vr * nprow * 16]  = psl_v

    rest = (N - vr) * nprow
    if rest > 0:
        pqs_f[vr * nprow * 128:] = rng.integers(0, 256, rest * 128, dtype=np.uint8)
        pqh_f[vr * nprow * 32:]  = rng.integers(0, 256, rest * 32,  dtype=np.uint8)
        psl_f[vr * nprow * 16:]  = rng.integers(0, 256, rest * 16,  dtype=np.uint8)

    x     = rng.standard_normal(K).astype(np.float32)
    y_ref = (deq_q5k(pqs_v, pqh_v, psl_v, K, vr) @ x) if vr > 0 else None

    return (_cbuf(gpu.ctx, pqs_f), _cbuf(gpu.ctx, pqh_f), _cbuf(gpu.ctx, psl_f),
            _cbuf(gpu.ctx, x), y_ref, vr)


def make_q6k_bufs(gpu, K, N, vr=64, seed=1234):
    """
    Returns: pql_buf, pqh_buf, ps_buf, pd_buf, x_buf, y_ref[vr] (or None), vr
    """
    nprow = K // 256
    total = N * nprow
    vr    = min(vr, N)
    rng   = np.random.default_rng(seed ^ 0xCAFE_1234)

    pql_f = np.empty(total * 128, dtype=np.uint8)
    pqh_f = np.empty(total * 64,  dtype=np.uint8)
    ps_f  = np.empty(total * 16,  dtype=np.uint8)
    pd_f  = np.empty(total * 2,   dtype=np.uint8)

    pql_v = pqh_v = ps_v = pd_v = None
    if vr > 0:
        raw  = gen_q6k_blocks(vr * nprow, seed=seed)
        pql_v, pqh_v, ps_v, pd_v = shuffle_q6k(raw, K, vr)
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
    y_ref = (deq_q6k(pql_v, pqh_v, ps_v, pd_v, K, vr) @ x) if vr > 0 else None

    return (_cbuf(gpu.ctx, pql_f), _cbuf(gpu.ctx, pqh_f),
            _cbuf(gpu.ctx, ps_f),  _cbuf(gpu.ctx, pd_f),
            _cbuf(gpu.ctx, x), y_ref, vr)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def check_close(label, ref, got, rtol=1e-2, atol=1e-3, verbose=False):
    max_abs = float(np.max(np.abs(ref - got)))
    rel_err = max_abs / (float(np.max(np.abs(ref))) + 1e-9)
    ok      = bool(np.allclose(ref, got, rtol=rtol, atol=atol))
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}  "
          f"max_abs={max_abs:.3e}  rel={rel_err:.3e}")
    if verbose and not ok:
        wi = int(np.argmax(np.abs(ref - got)))
        print(f"         worst idx={wi}  ref={ref[wi]:.6f}  got={got[wi]:.6f}")
    return ok


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
_OPG_Q4K = 16   # outputs per work-group (must match kernel #define)
_OPG_Q5K = 16   # outputs per work-group (must match OPG_Q5K in gemv_q5k_shuffle.cl)
_OPG_Q6K = 16
_BS_Q6K  = 2    # blocks per thread     (must match kernel #define)


def _print_result(st, rl, moved):
    print(f"    mean={st['mean_ms']:.3f} ms  median={st['median_ms']:.3f} ms  "
          f"min={st['min_ms']:.3f} ms  max={st['max_ms']:.3f} ms  (n={st['iters']})")
    print(f"    BW={rl['achieved_bw_gbps']:.1f} GB/s ({rl['bw_util_pct']:.1f}% of peak)  "
          f"AI={rl['arith_intensity']:.2f} FLOP/B  "
          f"roofline={rl['roofline_pct']:.1f}%  bound={rl['bound_by']}  "
          f"GFLOPS={rl['achieved_gflops']:.1f}")
    print(f"    moved={moved / 1e6:.1f} MB")


def bench_q4k(gpu, hw, K, N, iters, warmup, vr=64, seed=1234, verbose=False):
    """
    Benchmark gemv_q4k on a [N, K] weight matrix.
    Dispatch: global=(N/OPG * nprow), local=(nprow)  where nprow=K/256.
    """
    label = f"Q4K  K={K:>6}  N={N:>7}"
    print(f"\n  [{label}]")
    nprow = K // 256

    prog = gpu.build(cl_src("gemv_q4k_shuffle.cl"))
    krn  = cl.Kernel(prog, "gemv_q4k")

    t0 = time.perf_counter()
    pqs_b, psl_b, x_b, y_ref, vr_actual = make_q4k_bufs(gpu, K, N, vr, seed)
    y_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=N * 4)
    print(f"    Buffer build: {(time.perf_counter() - t0) * 1e3:.0f} ms  |  "
          f"weight={N * nprow * 144 / 1e6:.1f} MB")

    # kernel args: inputs, pqs, psl, outputs, input_len, output_len
    krn.set_args(x_b, pqs_b, psl_b, y_b, np.uint32(K), np.uint32(N))

    gsize = (N // _OPG_Q4K) * nprow
    lsize = nprow

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, (gsize,), (lsize,))

    # correctness
    enq(gpu.queue);  gpu.queue.finish()
    y_ocl = np.empty(N, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, y_ocl, y_b);  gpu.queue.finish()
    corr_ok = True
    if y_ref is not None:
        corr_ok = check_close(f"first {vr_actual} rows", y_ref, y_ocl[:vr_actual], verbose=verbose)

    # benchmark
    print(f"    Benchmarking {iters} iters + {warmup} warmup, L3 flush each ...")
    ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
    st = stats(ts)

    flops   = 2 * N * K
    w_bytes = N * nprow * 144          # Q4K shuffled: 128 qs + 16 sl per block
    moved   = w_bytes + K * 4 + N * 4
    rl = hw.roofline(flops, moved, st["mean_ms"] * 1e-3)
    _print_result(st, rl, moved)

    return dict(label=label.strip(), corr_ok=corr_ok, timing=st, roofline=rl,
                weight_mb=w_bytes / 1e6, moved_mb=moved / 1e6)


def bench_q5k(gpu, hw, K, N, iters, warmup, vr=64, seed=1234, verbose=False):
    """
    Benchmark gemv_q5k on a [N, K] weight matrix.
    Dispatch: global=(N/OPG * nprow), local=(nprow)  where nprow=K/256.
    """
    label = f"Q5K  K={K:>6}  N={N:>7}"
    print(f"\n  [{label}]")
    nprow = K // 256

    prog = gpu.build(cl_src("gemv_q5k_shuffle.cl"))
    krn  = cl.Kernel(prog, "gemv_q5k")

    t0 = time.perf_counter()
    pqs_b, pqh_b, psl_b, x_b, y_ref, vr_actual = make_q5k_bufs(gpu, K, N, vr, seed)
    y_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=N * 4)
    print(f"    Buffer build: {(time.perf_counter() - t0) * 1e3:.0f} ms  |  "
          f"weight={N * nprow * 176 / 1e6:.1f} MB")

    # kernel args: inputs, pqs, pqh, psl, outputs, input_len, output_len
    krn.set_args(x_b, pqs_b, pqh_b, psl_b, y_b, np.uint32(K), np.uint32(N))

    gsize = (N // _OPG_Q5K) * nprow
    lsize = nprow

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, (gsize,), (lsize,))

    # correctness
    enq(gpu.queue);  gpu.queue.finish()
    y_ocl = np.empty(N, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, y_ocl, y_b);  gpu.queue.finish()
    corr_ok = True
    if y_ref is not None:
        corr_ok = check_close(f"first {vr_actual} rows", y_ref, y_ocl[:vr_actual], verbose=verbose)

    # benchmark
    print(f"    Benchmarking {iters} iters + {warmup} warmup, L3 flush each ...")
    ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
    st = stats(ts)

    # Q5K shuffled block = 128 (qs) + 32 (qh) + 16 (sl) = 176 bytes
    flops   = 2 * N * K
    w_bytes = N * nprow * 176
    moved   = w_bytes + K * 4 + N * 4
    rl = hw.roofline(flops, moved, st["mean_ms"] * 1e-3)
    _print_result(st, rl, moved)

    return dict(label=label.strip(), corr_ok=corr_ok, timing=st, roofline=rl,
                weight_mb=w_bytes / 1e6, moved_mb=moved / 1e6)


def bench_q6k(gpu, hw, K, N, iters, warmup, vr=64, seed=1234, verbose=False):
    """
    Benchmark gemv_q6k on a [N, K] weight matrix.
    Dispatch: local_range=nprow/BS_Q6K, global=(N/OPG*local_range), local=(local_range).
    """
    label = f"Q6K  K={K:>6}  N={N:>7}"
    print(f"\n  [{label}]")
    nprow     = K // 256
    loc_range = nprow // _BS_Q6K

    prog = gpu.build(cl_src("gemv_q6k_shuffle.cl"))
    krn  = cl.Kernel(prog, "gemv_q6k")

    t0 = time.perf_counter()
    pql_b, pqh_b, ps_b, pd_b, x_b, y_ref, vr_actual = make_q6k_bufs(gpu, K, N, vr, seed)
    y_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=N * 4)
    print(f"    Buffer build: {(time.perf_counter() - t0) * 1e3:.0f} ms  |  "
          f"weight={N * nprow * 210 / 1e6:.1f} MB")

    # kernel args: inputs, pql, pqh, ps, pd_buf, outputs, input_len, output_len
    krn.set_args(x_b, pql_b, pqh_b, ps_b, pd_b, y_b, np.uint32(K), np.uint32(N))

    gsize = (N // _OPG_Q6K) * loc_range
    lsize = loc_range

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, (gsize,), (lsize,))

    # correctness
    enq(gpu.queue);  gpu.queue.finish()
    y_ocl = np.empty(N, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, y_ocl, y_b);  gpu.queue.finish()
    corr_ok = True
    if y_ref is not None:
        corr_ok = check_close(f"first {vr_actual} rows", y_ref, y_ocl[:vr_actual], verbose=verbose)

    # benchmark
    print(f"    Benchmarking {iters} iters + {warmup} warmup, L3 flush each ...")
    ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
    st = stats(ts)

    # Q6K shuffled block = 128 (ql) + 64 (qh) + 16 (ps) + 2 (pd) = 210 bytes
    flops   = 2 * N * K
    w_bytes = N * nprow * 210
    moved   = w_bytes + K * 4 + N * 4
    rl = hw.roofline(flops, moved, st["mean_ms"] * 1e-3)
    _print_result(st, rl, moved)

    return dict(label=label.strip(), corr_ok=corr_ok, timing=st, roofline=rl,
                weight_mb=w_bytes / 1e6, moved_mb=moved / 1e6)


# ---------------------------------------------------------------------------
# Test-case table
# ---------------------------------------------------------------------------
TEST_CASES = [
    #{"quant": "q6k", "K":  4096, "N": 151936, "label": "Q6K_K4096_N151936"},
    {"quant": "q6k", "K": 12288, "N":   4096, "label": "Q6K_K12288_N4096"},
    {"quant": "q4k", "K": 12288, "N":   4096, "label": "Q4K_K12288_N4096"},
    {"quant": "q4k", "K":  4096, "N":  12288, "label": "Q4K_K4096_N12288"},
    {"quant": "q4k", "K":  4096, "N":   4096, "label": "Q4K_K4096_N4096"},
    {"quant": "q6k", "K":  4096, "N":   1024, "label": "Q6K_K4096_N1024"},
    {"quant": "q4k", "K":  4096, "N":   1024, "label": "Q4K_K4096_N1024"},
    {"quant": "q5k", "K":  4096, "N":   8192, "label": "Q5K_K4096_N8192"},
    {"quant": "q5k", "K":  8192, "N":   4096, "label": "Q5K_K8192_N4096"},
]

_BENCH_FN = {
    "q4k": bench_q4k,
    "q5k": bench_q5k,
    "q6k": bench_q6k,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",      type=int,   default=0)
    ap.add_argument("--iters",       type=int,   default=500,
                    help="timed iterations per case (default: 500)")
    ap.add_argument("--warmup",      type=int,   default=10)
    ap.add_argument("--flush_mib",   type=int,   default=256,
                    help="L3 flush buffer size in MiB (default: 64)")
    ap.add_argument("--peak_bw",     type=float, default=512.0,
                    help="Peak DRAM bandwidth in GB/s (default: 512)")
    ap.add_argument("--peak_fp32",   type=float, default=9800.0,
                    help="Peak FP32 throughput in GFLOPS (default: 9800)")
    ap.add_argument("--verify_rows", type=int,   default=64,
                    help="Output rows to check for correctness (default: 64)")
    ap.add_argument("--verbose",     action="store_true")
    args = ap.parse_args()

    hw  = HW(peak_bw_gbps=args.peak_bw, fp32_gflops=args.peak_fp32)
    gpu = Gpu(args.device)
    gpu.setup_flush(args.flush_mib)

    SEP = "=" * 100
    print(SEP)
    print(f"  Peak FP32 : {hw.fp32_gflops:.0f} GFLOPS")
    print(f"  Peak BW   : {hw.peak_bw_gbps:.1f} GB/s   "
          f"(ridge point {hw.fp32_gflops / hw.peak_bw_gbps:.2f} FLOP/B)")
    print(f"  L3 flush  : {args.flush_mib} MiB buffer, 2 sweeps per iteration")
    print(f"  Iters     : {args.iters} timed + {args.warmup} warmup  |  "
          f"verify_rows={args.verify_rows}")
    print(SEP)

    results = []
    for case in TEST_CASES:
        try:
            fn  = _BENCH_FN[case["quant"]]
            rec = fn(gpu, hw, case["K"], case["N"],
                     args.iters, args.warmup, args.verify_rows,
                     verbose=args.verbose)
            rec["label"] = case["label"]
            results.append(rec)
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {case['label']}: {exc}")
            traceback.print_exc()
            results.append({"label": case["label"], "corr_ok": False,
                            "timing": {}, "roofline": {}})

    # ---- summary table ----
    W = 112
    print("\n" + "=" * W)
    print(f"  {'Case':<26} {'Corr':>5}  {'mean ms':>9}  {'BW GB/s':>9}  "
          f"{'BW%':>6}  {'GFLOPS':>8}  {'Roof%':>7}  {'Bound':>7}  {'moved MB':>9}")
    print("-" * W)
    for r in results:
        rl = r.get("roofline", {})
        st = r.get("timing",   {})
        print(f"  {r['label']:<26} {'OK' if r.get('corr_ok') else 'FAIL':>5}"
              f"  {st.get('mean_ms',          float('nan')):>9.3f}"
              f"  {rl.get('achieved_bw_gbps', float('nan')):>9.1f}"
              f"  {rl.get('bw_util_pct',      float('nan')):>6.1f}%"
              f"  {rl.get('achieved_gflops',  float('nan')):>8.1f}"
              f"  {rl.get('roofline_pct',     float('nan')):>7.1f}%"
              f"  {rl.get('bound_by', '-'):>7}"
              f"  {r.get('moved_mb', float('nan')):>9.1f}")
    print("=" * W)
    n_ok = sum(1 for r in results if r.get("corr_ok"))
    print(f"\n  {n_ok}/{len(results)} cases passed correctness.\n")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
