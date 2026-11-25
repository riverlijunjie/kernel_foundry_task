"""
test_gemv_sg_kernels.py

Correctness + benchmark for the high-performance "SG" (sub-group block read)
OpenCL GEMV kernels:
    gemv_q4k_sg_v3  (gemv_q4k_sg.cl)
    gemv_q6k_sg     (gemv_q6k_sg.cl)

These kernels use intel_sub_group_block_read for coalesced weight loads.
The weight data must be pre-packed into the "SG transposed" column-major layout
by the pack_q4k_sg() / pack_q6k_sg() functions below.

Usage:
    python test_gemv_sg_kernels.py [--device 0]
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

OPG = 16   # sub_group_size = outputs per work-group

# ---------------------------------------------------------------------------
# K-split (occupancy) heuristic
# ---------------------------------------------------------------------------
# The SG kernels launch one work-group per OPG output rows, and (with KSPLIT>1)
# KSPLIT sub-groups per work-group, each covering a strided subset of the
# K-blocks.  Total live hardware threads = (N/OPG) * KSPLIT.  For small N the
# baseline (KSPLIT=1) leaves the GPU badly under-occupied; we raise KSPLIT so
# that the total sub-group (thread) count approaches TARGET_THREADS, capped by
# the number of K-blocks (can't split more finely) and a work-group limit.
import os as _os
# --- occupancy tuning knobs (env-overridable) -----------------------------
TARGET_THREADS = int(_os.environ.get("GEMV_TARGET",    "4096"))  # threads we aim for
BASE_KEEP      = int(_os.environ.get("GEMV_BASEKEEP",  "512"))   # keep KSPLIT=1 above this many WGs
MAX_KSPLIT     = int(_os.environ.get("GEMV_MAXKSPLIT", "16"))    # WG / block-count safety cap
MIN_BLK_PER_SG = int(_os.environ.get("GEMV_MINBLK",    "2"))     # keep >= this many K-blocks/sub-group
MIN_THREADS    = int(_os.environ.get("GEMV_MINTHREADS","1024"))  # ... unless still thread-starved

def _pow2_floor(x):
    p = 1
    while p * 2 <= x:
        p *= 2
    return p

def choose_ksplit(N, K, target=None, max_ksplit=None):
    """Pick #sub-groups per work-group (K-split factor) to raise occupancy.

    Total live hardware threads = (N/OPG) * KSPLIT.  We split the K dimension
    across sub-groups so small-N GEMVs fill the GPU, but avoid over-splitting
    (which shrinks the per-sub-group work below what amortises the reduction).
    """
    target     = TARGET_THREADS if target is None else target
    max_ksplit = MAX_KSPLIT if max_ksplit is None else max_ksplit
    force = _os.environ.get("GEMV_KSPLIT", "")
    if force:
        return int(force)

    nbpr = K // 256                      # K-blocks available to split
    base = max(1, N // OPG)              # work-groups (== threads at KSPLIT=1)
    if base >= BASE_KEEP:                # already enough occupancy -> no split
        return 1

    ks = _pow2_floor(min(max(1, round(target / base)), nbpr, max_ksplit))

    # Back off if each sub-group would get too few K-blocks (reduction/SLM
    # overhead dominates), but only while we still have plenty of threads.
    while ks > 1 and (nbpr // ks) < MIN_BLK_PER_SG and base * (ks // 2) >= MIN_THREADS:
        ks //= 2
    return ks

# ---------------------------------------------------------------------------
# Cache flush kernel (same as test_gemv_shuffle_kernels.py)
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
        self.fp32_gflops  = fp32_gflops

    def roofline(self, flops, bytes_moved, seconds):
        gflops = flops / seconds / 1e9
        bw     = bytes_moved / seconds / 1e9
        ai     = flops / max(bytes_moved, 1)
        ridge  = self.fp32_gflops / max(self.peak_bw_gbps, 1e-12)
        roof   = ai * self.peak_bw_gbps if ai < ridge else self.fp32_gflops
        bound  = "memory" if ai < ridge else "compute"
        return dict(
            achieved_gflops  = gflops,
            achieved_bw_gbps = bw,
            arith_intensity  = ai,
            roofline_pct     = gflops / max(roof, 1e-12) * 100,
            bw_util_pct      = bw / max(self.peak_bw_gbps, 1e-12) * 100,
            bound_by         = bound,
        )

# ---------------------------------------------------------------------------
# GPU context
# ---------------------------------------------------------------------------
class Gpu:
    def __init__(self, device_idx=0):
        platforms = cl.get_platforms()
        devices   = [d for p in platforms for d in p.get_devices()]
        self.device = devices[device_idx % len(devices)]
        print(f"Device [{device_idx}]: {self.device.name}")
        self.ctx   = cl.Context([self.device])
        self.queue = cl.CommandQueue(
            self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self._prog = {}
        self._flush_krn = self._flush_buf = self._flush_sink = None
        self._flush_n = 0

    def build(self, src_file, build_opts=""):
        key = (src_file, build_opts)
        if key not in self._prog:
            with open(src_file) as f:
                src = f.read()
            self._prog[key] = cl.Program(self.ctx, src).build(options=build_opts)
        return self._prog[key]

    def _build_src(self, src, key):
        if key not in self._prog:
            self._prog[key] = cl.Program(self.ctx, src).build()
        return self._prog[key]

    def setup_flush(self, mib=256):
        n    = mib * 1024 * 1024 // 4
        self._flush_n = n
        mf   = cl.mem_flags
        zeros = np.zeros(n, dtype=np.float32)
        self._flush_buf  = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
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

    def time_kernel(self, enq_fn, iters=500, warmup=10, do_flush=True):
        for _ in range(warmup):
            enq_fn(self.queue);  self.queue.finish()
        times = []
        for _ in range(iters):
            if do_flush: self.flush_l3()
            ev = enq_fn(self.queue)
            ev.wait()
            times.append((ev.profile.end - ev.profile.start) * 1e-9)
        return times

def stats(ts):
    return dict(mean_ms=statistics.mean(ts)*1e3, median_ms=statistics.median(ts)*1e3,
                min_ms=min(ts)*1e3, max_ms=max(ts)*1e3, iters=len(ts))

# ---------------------------------------------------------------------------
# fp16 helper
# ---------------------------------------------------------------------------
def fp16_to_float(b0, b1):
    return float(struct.unpack('e', struct.pack('H', int(b0) | (int(b1) << 8)))[0])

# ---------------------------------------------------------------------------
# Fast block generators (same as test_gemv_shuffle_kernels.py)
# ---------------------------------------------------------------------------
def gen_q4k_blocks(n, seed=1234):
    rng = np.random.default_rng(seed)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm  = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc  = rng.integers(0, 64,  (n, 12), dtype=np.uint8)
    qs  = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qs], axis=1).ravel()

def gen_q6k_blocks(n, seed=1234):
    rng = np.random.default_rng(seed)
    ql  = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    qh  = rng.integers(0, 256, (n,  64), dtype=np.uint8)
    sc  = rng.integers(-8, 8,  (n,  16), dtype=np.int8).view(np.uint8)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    return np.concatenate([ql, qh, sc, d], axis=1).ravel()

def gen_q5k_blocks(n, seed=1234):
    """n x 176-byte block_q5_K layout: d(2) dmin(2) scales(12) qh(32) qs(128)."""
    rng = np.random.default_rng(seed)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm  = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc  = rng.integers(0, 64,  (n, 12),  dtype=np.uint8)
    qh  = rng.integers(0, 256, (n,  32), dtype=np.uint8)
    qs  = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qh, qs], axis=1).ravel()   # [n * 176]

# ---------------------------------------------------------------------------
# Standard shuffle (row-major) — needed to build the reference pqs/psl
# ---------------------------------------------------------------------------
def shuffle_q4k(raw, K, N):
    NB  = N * K // 256
    pqs = np.zeros(NB * 128, dtype=np.uint8)
    psl = np.zeros(NB * 16,  dtype=np.uint8)
    for b in range(NB):
        blk  = raw[b * 144:(b + 1) * 144]
        d    = fp16_to_float(blk[0], blk[1])
        dmin = fp16_to_float(blk[2], blk[3])
        sc   = blk[4:16];  qs = blk[16:144]
        rw = np.zeros(256, dtype=np.uint8)
        for j in range(4):
            for k in range(32):
                rw[j*64+k]      = qs[j*32+k] & 0x0F
                rw[j*64+32+k]   = (qs[j*32+k] >> 4) & 0x0F
        rs = np.zeros(8, np.uint8);  rm = np.zeros(8, np.uint8)
        rs[0]=sc[0]&63; rs[1]=sc[1]&63; rs[2]=sc[2]&63; rs[3]=sc[3]&63
        rs[4]=(sc[8]&0xF)|((sc[0]>>2)&0x30); rs[5]=(sc[9]&0xF)|((sc[1]>>2)&0x30)
        rs[6]=(sc[10]&0xF)|((sc[2]>>2)&0x30); rs[7]=(sc[11]&0xF)|((sc[3]>>2)&0x30)
        rm[0]=sc[4]&63; rm[1]=sc[5]&63; rm[2]=sc[6]&63; rm[3]=sc[7]&63
        rm[4]=((sc[8]>>4)&0xF)|((sc[4]>>2)&0x30); rm[5]=((sc[9]>>4)&0xF)|((sc[5]>>2)&0x30)
        rm[6]=((sc[10]>>4)&0xF)|((sc[6]>>2)&0x30); rm[7]=((sc[11]>>4)&0xF)|((sc[7]>>2)&0x30)
        opqs = pqs[b*128:(b+1)*128]
        for j in range(8):
            for k in range(16):
                opqs[j*16+k] = (rw[j*32+k]&0xF) | ((rw[j*32+16+k]&0xF) << 4)
        opsl = psl[b*16:(b+1)*16]
        opsl[0]=(rs[0]&0xF)|((rs[1]&0xF)<<4); opsl[1]=(rs[2]&0xF)|((rs[3]&0xF)<<4)
        opsl[2]=(rs[4]&0xF)|((rs[5]&0xF)<<4); opsl[3]=(rs[6]&0xF)|((rs[7]&0xF)<<4)
        opsl[4]=(rm[0]&0xF)|((rm[1]&0xF)<<4); opsl[5]=(rm[2]&0xF)|((rm[3]&0xF)<<4)
        opsl[6]=(rm[4]&0xF)|((rm[5]&0xF)<<4); opsl[7]=(rm[6]&0xF)|((rm[7]&0xF)<<4)
        opsl[8] =((rs[0]&0x30)>>4)|((rs[1]&0x30)>>2)|(rs[2]&0x30)|((rs[3]&0x30)<<2)
        opsl[9] =((rs[4]&0x30)>>4)|((rs[5]&0x30)>>2)|(rs[6]&0x30)|((rs[7]&0x30)<<2)
        opsl[10]=((rm[0]&0x30)>>4)|((rm[1]&0x30)>>2)|(rm[2]&0x30)|((rm[3]&0x30)<<2)
        opsl[11]=((rm[4]&0x30)>>4)|((rm[5]&0x30)>>2)|(rm[6]&0x30)|((rm[7]&0x30)<<2)
        opsl[12:14] = np.frombuffer(struct.pack('e', d),    dtype=np.uint8)
        opsl[14:16] = np.frombuffer(struct.pack('e', dmin), dtype=np.uint8)
    return pqs, psl

def shuffle_q6k(raw, K, N):
    NB  = N * K // 256
    pql = np.zeros(NB * 128, dtype=np.uint8)
    pqh = np.zeros(NB * 64,  dtype=np.uint8)
    ps  = np.zeros(NB * 16,  dtype=np.int8)
    pd  = np.zeros(NB * 2,   dtype=np.uint8)
    for b in range(NB):
        blk  = raw[b*210:(b+1)*210]
        ql_r = blk[0:128];  qh_r = blk[128:192]
        sc_r = blk[192:208].view(np.int8)
        d    = fp16_to_float(blk[208], blk[209])
        rw = np.zeros(256, np.uint8)
        for j in range(2):
            for k in range(64):
                rw[j*128+k]      = ql_r[j*64+k] & 0x0F
                rw[j*128+64+k]   = (ql_r[j*64+k] >> 4) & 0x0F
            for k in range(32):
                rw[j*128+k]      += ((qh_r[j*32+k] & 0x03) << 4)
                rw[j*128+32+k]   += ((qh_r[j*32+k] & 0x0C) << 2)
                rw[j*128+64+k]   +=   qh_r[j*32+k] & 0x30
                rw[j*128+96+k]   += ((qh_r[j*32+k] & 0xC0) >> 2)
        opql = pql[b*128:(b+1)*128]
        for j in range(8):
            for k in range(16):
                opql[j*16+k] = (rw[j*32+k]&0xF) | ((rw[j*32+16+k]&0xF) << 4)
        opqh = pqh[b*64:(b+1)*64]
        for j in range(16):
            for k in range(4):
                opqh[j*4+k] = (((rw[j*16+k]   &0x30)>>4)|((rw[j*16+k+4] &0x30)>>2)
                               |(rw[j*16+k+8]  &0x30)    |((rw[j*16+k+12]&0x30)<<2))
        ps[b*16:(b+1)*16] = sc_r
        pd[b*2:(b+1)*2] = np.frombuffer(struct.pack('e', d), dtype=np.uint8)
    return pql, pqh, ps.view(np.uint8), pd

def shuffle_q5k(raw, K, N):
    """
    ShuffleQ5KL1.
    Input : raw bytes [N*K/256 * 176] (block_q5_K: d(2) dmin(2) scales(12) qh(32) qs(128))
    Output: pqs [NB*128], pqh [NB*32], psl [NB*16]  where NB = N*K/256

    Reference dequant (llama.cpp block_q5_K), for 32-weight group g=0..7:
        m = g // 2 ; bit_pos = g % 2
        for l in 0..31:
            nibble  = (qs_r[m*32 + l] >> (4*bit_pos)) & 0xF
            hi_bit  = (qh_r[l] >> (2*m + bit_pos)) & 1
            weight[g*32 + l] = nibble + (hi_bit << 4)
        scale/min index "is" == g  (get_scale_min_k4 identical to Q4_K)
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

# ---------------------------------------------------------------------------
# SG transposed packing functions
# ---------------------------------------------------------------------------

def pack_q4k_sg(pqs, psl, K, N, opg=OPG):
    """
    Convert row-major shuffled Q4K layout (pqs, psl) to SG column-major format.
    SAME total size as input arrays.

    pqs_T layout (total_blk * opg * 128 = total_blk * 128 bytes — same as pqs):
      Entry (h, bid) at (h*nbpr + bid) * opg * 128  (= 2048 bytes)
      Sub-block j at j * opg * 16 within entry
      Chunk c (0..3) at j*opg*16 + c*opg*4 within entry
      Lane lid gets: pqs[blk*128 + j*16 + c*4 .. +3]  blk=(h*opg+lid)*nbpr+bid

    psl_T layout (total_blk * opg * 16 = total_blk * 16 bytes — same as psl):
      Entry (h, bid) at (h*nbpr + bid) * opg * 16  (= 256 bytes)
      SoA fields: sl(64B) ml(64B) sh(32B) mh(32B) d(32B) dmin(32B)
      Lane lid's sl at: entry + lid*4
      Lane lid's d  at: entry + 192 + lid*2
    """
    nbpr      = K // 256
    if N % opg != 0:
        raise ValueError(f"N ({N}) must be divisible by opg ({opg}) for SG packing")
    nrg       = N // opg
    total_blk = N * nbpr
    entries   = nrg * nbpr

    # Same total size as originals
    pqs_T = np.zeros(entries * opg * 128, dtype=np.uint8)  # == total_blk * 128
    psl_T = np.zeros(entries * opg * 16,  dtype=np.uint8)  # == total_blk * 16
    assert pqs_T.nbytes == pqs.nbytes, f"pqs_T size mismatch: {pqs_T.nbytes} vs {pqs.nbytes}"
    assert psl_T.nbytes == psl.nbytes, f"psl_T size mismatch: {psl_T.nbytes} vs {psl.nbytes}"

    for h in range(nrg):
        for bid in range(nbpr):
            psl_entry = (h * nbpr + bid) * opg * 16   # 256 bytes
            pqs_entry = (h * nbpr + bid) * opg * 128  # 2048 bytes

            for lid in range(opg):
                row = h * opg + lid
                blk = row * nbpr + bid

                # ---- psl_T: SoA fields ----
                src = psl[blk * 16:(blk + 1) * 16]
                e = psl_entry
                psl_T[e +   0 + lid*4 : e +   0 + lid*4 + 4] = src[0:4]   # sl
                psl_T[e +  64 + lid*4 : e +  64 + lid*4 + 4] = src[4:8]   # ml
                psl_T[e + 128 + lid*2 : e + 128 + lid*2 + 2] = src[8:10]  # sh
                psl_T[e + 160 + lid*2 : e + 160 + lid*2 + 2] = src[10:12] # mh
                psl_T[e + 192 + lid*2 : e + 192 + lid*2 + 2] = src[12:14] # d
                psl_T[e + 224 + lid*2 : e + 224 + lid*2 + 2] = src[14:16] # dmin

                # ---- pqs_T: 8 sub-blocks × 4 chunks ----
                for j in range(8):
                    src_sub = pqs[blk * 128 + j * 16 : blk * 128 + j * 16 + 16]
                    for chunk in range(4):
                        dst = pqs_entry + j * opg * 16 + chunk * opg * 4 + lid * 4
                        pqs_T[dst:dst + 4] = src_sub[chunk * 4: chunk * 4 + 4]

    return pqs_T, psl_T


def pack_q6k_sg(pql, pqh, ps_u8, pd_buf, K, N, opg=OPG):
    """
    Convert row-major shuffled Q6K layout to SG column-major format.
    SAME total size as input arrays.

    pql_T (total_blk * opg * 128 = total_blk * 128 bytes — same as pql):
      Entry (h,bid) at (h*nbpr+bid)*opg*128, sub-block j at j*opg*16,
      chunk c (0..3) at j*opg*16 + c*opg*4, lane lid gets 4 bytes.

    pqh_T (total_blk * opg * 64 = total_blk * 64 bytes — same as pqh):
      Entry at (h*nbpr+bid)*opg*64, sub-block j at j*opg*8,
      chunk c (0..1) at j*opg*8 + c*opg*4.

    ps_T  (total_blk * opg * 16 = total_blk * 16 bytes — same as ps):
      Entry at (h*nbpr+bid)*opg*16, scale si at si*opg, 1 byte per lane.

    pd_T  (total_blk * opg * 2 = total_blk * 2 bytes — same as pd):
      Entry at (h*nbpr+bid)*opg*2, 1 ushort (fp16) per lane.
    """
    nbpr      = K // 256
    if N % opg != 0:
        raise ValueError(f"N ({N}) must be divisible by opg ({opg}) for SG packing")
    nrg       = N // opg
    total_blk = N * nbpr
    entries   = nrg * nbpr
    ps        = ps_u8.view(np.int8)

    # Same total sizes as originals
    pql_T = np.zeros(entries * opg * 128, dtype=np.uint8)  # == total_blk * 128
    pqh_T = np.zeros(entries * opg * 64,  dtype=np.uint8)  # == total_blk * 64
    ps_T  = np.zeros(entries * opg * 16,  dtype=np.int8)   # == total_blk * 16
    pd_T  = np.zeros(entries * opg * 2,   dtype=np.uint8)  # == total_blk * 2
    assert pql_T.nbytes == pql.nbytes
    assert pqh_T.nbytes == pqh.nbytes
    assert ps_T.nbytes  == ps_u8.nbytes
    assert pd_T.nbytes  == pd_buf.nbytes

    for h in range(nrg):
        for bid in range(nbpr):
            pql_entry = (h * nbpr + bid) * opg * 128
            pqh_entry = (h * nbpr + bid) * opg * 64
            ps_entry  = (h * nbpr + bid) * opg * 16
            pd_entry  = (h * nbpr + bid) * opg * 2

            for lid in range(opg):
                row = h * opg + lid
                blk = row * nbpr + bid

                # ---- pd_T ----
                pd_T[pd_entry + lid*2 : pd_entry + lid*2 + 2] = pd_buf[blk*2:(blk+1)*2]

                # ---- ps_T: scale si at si*opg within entry ----
                for si in range(16):
                    ps_T[ps_entry + si * opg + lid] = ps[blk * 16 + si]

                # ---- pql_T: 8 sub-blocks × 4 chunks ----
                for j in range(8):
                    src_ql = pql[blk * 128 + j * 16 : blk * 128 + j * 16 + 16]
                    for chunk in range(4):
                        dst = pql_entry + j * opg * 16 + chunk * opg * 4 + lid * 4
                        pql_T[dst:dst + 4] = src_ql[chunk*4:chunk*4+4]

                # ---- pqh_T: 8 sub-blocks × 2 chunks ----
                for j in range(8):
                    src_qh = pqh[blk * 64 + j * 8 : blk * 64 + j * 8 + 8]
                    for chunk in range(2):
                        dst = pqh_entry + j * opg * 8 + chunk * opg * 4 + lid * 4
                        pqh_T[dst:dst + 4] = src_qh[chunk*4:chunk*4+4]

    return pql_T, pqh_T, ps_T.view(np.uint8), pd_T


def pack_q5k_sg(pqs, pqh, psl, K, N, opg=OPG):
    """
    Convert row-major shuffled Q5K layout (pqs, pqh, psl) to SG column-major
    format.  SAME total size as input arrays.

    pqs_T layout (total_blk * opg * 128 = total_blk * 128 bytes — same as pqs):
      Entry (h, bid) at (h*nbpr + bid) * opg * 128  (= 2048 bytes)
      Sub-block j at j * opg * 16 within entry
      Chunk c (0..3) at j*opg*16 + c*opg*4 within entry
      Lane lid gets: pqs[blk*128 + j*16 + c*4 .. +3]  blk=(h*opg+lid)*nbpr+bid

    pqh_T layout (total_blk * opg * 32 = total_blk * 32 bytes — same as pqh):
      Entry (h, bid) at (h*nbpr + bid) * opg * 32  (= 512 bytes)
      Sub-block j's 4-byte qh word for all opg lanes at j * opg * 4
      Lane lid's word at: entry + j*opg*4 + lid*4

    psl_T layout (total_blk * opg * 16 = total_blk * 16 bytes — same as psl):
      Entry (h, bid) at (h*nbpr + bid) * opg * 16  (= 256 bytes)
      SoA fields: sl(64B) ml(64B) sh(32B) mh(32B) d(32B) dmin(32B)
      Lane lid's sl at: entry + lid*4
      Lane lid's d  at: entry + 192 + lid*2
    """
    nbpr      = K // 256
    if N % opg != 0:
        raise ValueError(f"N ({N}) must be divisible by opg ({opg}) for SG packing")
    nrg       = N // opg
    total_blk = N * nbpr
    entries   = nrg * nbpr

    # Same total size as originals
    pqs_T = np.zeros(entries * opg * 128, dtype=np.uint8)  # == total_blk * 128
    pqh_T = np.zeros(entries * opg * 32,  dtype=np.uint8)  # == total_blk * 32
    psl_T = np.zeros(entries * opg * 16,  dtype=np.uint8)  # == total_blk * 16
    assert pqs_T.nbytes == pqs.nbytes, f"pqs_T size mismatch: {pqs_T.nbytes} vs {pqs.nbytes}"
    assert pqh_T.nbytes == pqh.nbytes, f"pqh_T size mismatch: {pqh_T.nbytes} vs {pqh.nbytes}"
    assert psl_T.nbytes == psl.nbytes, f"psl_T size mismatch: {psl_T.nbytes} vs {psl.nbytes}"

    for h in range(nrg):
        for bid in range(nbpr):
            psl_entry = (h * nbpr + bid) * opg * 16   # 256 bytes
            pqs_entry = (h * nbpr + bid) * opg * 128  # 2048 bytes
            pqh_entry = (h * nbpr + bid) * opg * 32   # 512 bytes

            for lid in range(opg):
                row = h * opg + lid
                blk = row * nbpr + bid

                # ---- psl_T: SoA fields ----
                src = psl[blk * 16:(blk + 1) * 16]
                e = psl_entry
                psl_T[e +   0 + lid*4 : e +   0 + lid*4 + 4] = src[0:4]   # sl
                psl_T[e +  64 + lid*4 : e +  64 + lid*4 + 4] = src[4:8]   # ml
                psl_T[e + 128 + lid*2 : e + 128 + lid*2 + 2] = src[8:10]  # sh
                psl_T[e + 160 + lid*2 : e + 160 + lid*2 + 2] = src[10:12] # mh
                psl_T[e + 192 + lid*2 : e + 192 + lid*2 + 2] = src[12:14] # d
                psl_T[e + 224 + lid*2 : e + 224 + lid*2 + 2] = src[14:16] # dmin

                # ---- pqs_T: 8 sub-blocks × 4 chunks ----
                for j in range(8):
                    src_sub = pqs[blk * 128 + j * 16 : blk * 128 + j * 16 + 16]
                    for chunk in range(4):
                        dst = pqs_entry + j * opg * 16 + chunk * opg * 4 + lid * 4
                        pqs_T[dst:dst + 4] = src_sub[chunk * 4: chunk * 4 + 4]

                # ---- pqh_T: 8 sub-blocks × 1 word (4 bytes) ----
                for j in range(8):
                    src_qh = pqh[blk * 32 + j * 4 : blk * 32 + j * 4 + 4]
                    dst = pqh_entry + j * opg * 4 + lid * 4
                    pqh_T[dst:dst + 4] = src_qh

    return pqs_T, pqh_T, psl_T


# ---------------------------------------------------------------------------
# Reference dequantizers (same as test_gemv_shuffle_kernels.py)
# ---------------------------------------------------------------------------
def deq_q4k(pqs, psl, K, N):
    nprow = K // 256
    W = np.zeros((N, K), np.float32)
    for row in range(N):
        for bi in range(nprow):
            idx  = row * nprow + bi
            qs_b = pqs[idx*128:(idx+1)*128]
            sl_b = psl[idx*16:(idx+1)*16]
            sli  = int(sl_b[0])|(int(sl_b[1])<<8)|(int(sl_b[2])<<16)|(int(sl_b[3])<<24)
            mli  = int(sl_b[4])|(int(sl_b[5])<<8)|(int(sl_b[6])<<16)|(int(sl_b[7])<<24)
            shi  = int(sl_b[8])|(int(sl_b[9])<<8)
            mhi  = int(sl_b[10])|(int(sl_b[11])<<8)
            d    = fp16_to_float(sl_b[12], sl_b[13])
            dmin = fp16_to_float(sl_b[14], sl_b[15])
            for j in range(8):
                sq = ((sli>>(j*4))&0xF)|(((shi>>(j*2))&3)<<4)
                mq = ((mli>>(j*4))&0xF)|(((mhi>>(j*2))&3)<<4)
                sc = sq*d;  mv = mq*dmin
                for k in range(16):
                    byte = int(qs_b[j*16+k]); cb = bi*256+j*32
                    W[row, cb+k]    = (byte&0xF)*sc - mv
                    W[row, cb+16+k] = ((byte>>4)&0xF)*sc - mv
    return W

def deq_q6k(pql, pqh, ps_u8, pd_buf, K, N):
    nprow = K // 256; ps = ps_u8.view(np.int8)
    SH = [0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30]
    W  = np.zeros((N, K), np.float32)
    for row in range(N):
        for bi in range(nprow):
            idx  = row * nprow + bi
            ql_b = pql[idx*128:(idx+1)*128]; qh_b = pqh[idx*64:(idx+1)*64]
            s_b  = ps [idx*16:(idx+1)*16];   d = fp16_to_float(pd_buf[idx*2], pd_buf[idx*2+1])
            for j in range(8):
                clo = d*float(s_b[2*j]);  chi = d*float(s_b[2*j+1])
                hlo = int(qh_b[j*8])|(int(qh_b[j*8+1])<<8)|(int(qh_b[j*8+2])<<16)|(int(qh_b[j*8+3])<<24)
                hhi = int(qh_b[j*8+4])|(int(qh_b[j*8+5])<<8)|(int(qh_b[j*8+6])<<16)|(int(qh_b[j*8+7])<<24)
                for k in range(16):
                    qlb=int(ql_b[j*16+k]); sh=SH[k]; cb=bi*256+j*32
                    W[row,cb+k]    = (float((qlb&0xF)|(((hlo>>sh)&3)<<4))-32.0)*clo
                    W[row,cb+16+k] = (float(((qlb>>4)&0xF)|(((hhi>>sh)&3)<<4))-32.0)*chi
    return W


def deq_q5k(pqs, pqh, psl, K, N):
    """Decode shuffled Q5_K -> W[N, K] float32 (matches gemv_q5k_shuffle.cl)."""
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

# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------
def _cbuf(ctx, arr):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=np.ascontiguousarray(arr))

def make_q4k_sg_bufs(gpu, K, N, vr=64, seed=1234):
    """Build SG-packed Q4K weight buffers + reference output for first vr rows."""
    nbpr = K // 256;  vr = min(vr, N)
    rng  = np.random.default_rng(seed ^ 0xABCD)

    # Proper blocks for first vr rows
    pqs_v = psl_v = None
    if vr > 0:
        raw = gen_q4k_blocks(vr * nbpr, seed=seed)
        pqs_v, psl_v = shuffle_q4k(raw, K, vr)

    # Build full row-major pqs/psl (vr rows + random rest)
    total_blk = N * nbpr
    pqs_f = np.empty(total_blk * 128, dtype=np.uint8)
    psl_f = np.empty(total_blk * 16,  dtype=np.uint8)
    if vr > 0:
        pqs_f[:vr*nbpr*128] = pqs_v
        psl_f[:vr*nbpr*16]  = psl_v
    rest = (N - vr) * nbpr
    if rest > 0:
        pqs_f[vr*nbpr*128:] = rng.integers(0, 256, rest*128, dtype=np.uint8)
        psl_f[vr*nbpr*16:]  = rng.integers(0, 256, rest*16,  dtype=np.uint8)

    # Pack to SG format
    t0 = time.perf_counter()
    pqs_sg, psl_sg = pack_q4k_sg(pqs_f, psl_f, K, N)
    print(f"    SG pack time: {(time.perf_counter()-t0)*1e3:.0f} ms")

    x     = rng.standard_normal(K).astype(np.float16)
    y_ref = (deq_q4k(pqs_v, psl_v, K, vr) @ x.astype(np.float32)) if vr > 0 else None

    return (_cbuf(gpu.ctx, pqs_sg), _cbuf(gpu.ctx, psl_sg),
            _cbuf(gpu.ctx, x), y_ref, vr,
            pqs_sg.nbytes + psl_sg.nbytes)

def make_q6k_sg_bufs(gpu, K, N, vr=64, seed=1234):
    """Build SG-packed Q6K weight buffers + reference output."""
    nbpr = K // 256;  vr = min(vr, N)
    rng  = np.random.default_rng(seed ^ 0xABCD)

    pql_v = pqh_v = ps_v = pd_v = None
    if vr > 0:
        raw = gen_q6k_blocks(vr * nbpr, seed=seed)
        pql_v, pqh_v, ps_v, pd_v = shuffle_q6k(raw, K, vr)

    total_blk = N * nbpr
    pql_f = np.empty(total_blk*128, dtype=np.uint8)
    pqh_f = np.empty(total_blk*64,  dtype=np.uint8)
    ps_f  = np.empty(total_blk*16,  dtype=np.uint8)
    pd_f  = np.empty(total_blk*2,   dtype=np.uint8)
    if vr > 0:
        pql_f[:vr*nbpr*128]=pql_v; pqh_f[:vr*nbpr*64]=pqh_v
        ps_f[:vr*nbpr*16]=ps_v;    pd_f[:vr*nbpr*2]=pd_v
    rest = (N - vr) * nbpr
    if rest > 0:
        pql_f[vr*nbpr*128:] = rng.integers(0, 256, rest*128, dtype=np.uint8)
        pqh_f[vr*nbpr*64:]  = rng.integers(0, 256, rest*64,  dtype=np.uint8)
        ps_f [vr*nbpr*16:]  = rng.integers(0, 256, rest*16,  dtype=np.uint8)
        pd_f [vr*nbpr*2:]   = rng.integers(0, 256, rest*2,   dtype=np.uint8)

    t0 = time.perf_counter()
    pql_sg, pqh_sg, ps_sg, pd_sg = pack_q6k_sg(pql_f, pqh_f, ps_f, pd_f, K, N)
    print(f"    SG pack time: {(time.perf_counter()-t0)*1e3:.0f} ms")

    x     = rng.standard_normal(K).astype(np.float16)
    y_ref = (deq_q6k(pql_v, pqh_v, ps_v, pd_v, K, vr) @ x.astype(np.float32)) if vr > 0 else None

    total_bytes = pql_sg.nbytes + pqh_sg.nbytes + ps_sg.nbytes + pd_sg.nbytes
    return (_cbuf(gpu.ctx, pql_sg), _cbuf(gpu.ctx, pqh_sg),
            _cbuf(gpu.ctx, ps_sg),  _cbuf(gpu.ctx, pd_sg),
            _cbuf(gpu.ctx, x), y_ref, vr, total_bytes)

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def check_close(label, ref, got, rtol=2e-2, atol=0.5, verbose=False):
    max_abs = float(np.max(np.abs(ref - got)))
    rel_err = max_abs / (float(np.max(np.abs(ref))) + 1e-9)
    ok      = bool(np.allclose(ref, got, rtol=rtol, atol=atol))
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}  max_abs={max_abs:.3e}  rel={rel_err:.3e}")
    if verbose and not ok:
        wi = int(np.argmax(np.abs(ref - got)))
        print(f"         worst idx={wi}  ref={ref[wi]:.6f}  got={got[wi]:.6f}")
    return ok

# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
def _print_result(st, rl, moved):
    print(f"    mean={st['mean_ms']:.3f} ms  median={st['median_ms']:.3f} ms  "
          f"min={st['min_ms']:.3f} ms  max={st['max_ms']:.3f} ms  (n={st['iters']})")
    print(f"    BW={rl['achieved_bw_gbps']:.1f} GB/s ({rl['bw_util_pct']:.1f}% of peak)  "
          f"AI={rl['arith_intensity']:.2f} FLOP/B  "
          f"roofline={rl['roofline_pct']:.1f}%  bound={rl['bound_by']}  "
          f"GFLOPS={rl['achieved_gflops']:.1f}")
    print(f"    moved={moved/1e6:.1f} MB")


def bench_q4k_sg(gpu, hw, K, N, iters, warmup, vr=64, seed=1234, verbose=False):
    label = f"Q4K_SG K={K} N={N}"
    print(f"\n  [{label}]")
    nbpr = K // 256

    t0 = time.perf_counter()
    pqs_b, psl_b, x_b, y_ref, vr_act, sg_bytes = make_q4k_sg_bufs(gpu, K, N, vr, seed)
    y_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=N * 2)
    print(f"    Buffer total: {(time.perf_counter()-t0)*1e3:.0f} ms  "
          f"sg_weight={sg_bytes/1e6:.1f} MB  orig={N * nbpr * 144 / 1e6:.1f} MB  "
          f"(SG weights same size as orig; only layout differs)")

    # Build options must enable sub-groups
    ksplit = choose_ksplit(N, K)
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit}"
    prog = gpu.build(cl_src("gemv_q4k_sg.cl"), opts)
    krn  = cl.Kernel(prog, "gemv_q4k_sg")

    # global = (output_len, KSPLIT), local = (OPG, KSPLIT)  -> KSPLIT sub-groups
    # per work-group, each covering a strided subset of the K-blocks.
    print(f"    KSPLIT={ksplit}  work-groups={N//OPG}  live_subgroups={(N//OPG)*ksplit}")
    gsize = (N, ksplit)
    lsize = (OPG, ksplit)
    krn.set_args(x_b, pqs_b, psl_b, y_b, np.uint32(K), np.uint32(N))

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    # correctness
    enq(gpu.queue);  gpu.queue.finish()
    y_ocl = np.empty(N, dtype=np.float16)
    cl.enqueue_copy(gpu.queue, y_ocl, y_b);  gpu.queue.finish()
    corr_ok = True
    if y_ref is not None:
        y_ref_f16 = y_ref.astype(np.float16).astype(np.float32)
        corr_ok = check_close(f"first {vr_act} rows", y_ref_f16, y_ocl[:vr_act].astype(np.float32), verbose=verbose)

    # benchmark
    print(f"    Benchmarking {iters} iters + {warmup} warmup, L3 flush each ...")
    ts  = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
    st  = stats(ts)

    # Roofline: actual bytes moved = SG weight size + input + output
    flops   = 2 * N * K
    w_bytes = N * nbpr * 144   # use original weight bytes for fair BW comparison
    moved   = w_bytes + K * 2 + N * 2
    rl = hw.roofline(flops, moved, st["mean_ms"] * 1e-3)
    _print_result(st, rl, moved)

    return dict(label=label, corr_ok=corr_ok, timing=st, roofline=rl,
                weight_mb=w_bytes/1e6, sg_weight_mb=sg_bytes/1e6, moved_mb=moved/1e6)


def bench_q6k_sg(gpu, hw, K, N, iters, warmup, vr=64, seed=1234, verbose=False):
    label = f"Q6K_SG K={K} N={N}"
    print(f"\n  [{label}]")
    nbpr = K // 256

    t0 = time.perf_counter()
    pql_b, pqh_b, ps_b, pd_b, x_b, y_ref, vr_act, sg_bytes = make_q6k_sg_bufs(gpu, K, N, vr, seed)
    y_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=N * 2)
    print(f"    Buffer total: {(time.perf_counter()-t0)*1e3:.0f} ms  "
          f"sg_weight={sg_bytes/1e6:.1f} MB  orig={N * nbpr * 210 / 1e6:.1f} MB  "
          f"(SG weights same size as orig; only layout differs)")

    ksplit = choose_ksplit(N, K)
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit}"
    prog = gpu.build(cl_src("gemv_q6k_sg.cl"), opts)
    krn  = cl.Kernel(prog, "gemv_q6k_sg")

    print(f"    KSPLIT={ksplit}  work-groups={N//OPG}  live_subgroups={(N//OPG)*ksplit}")
    gsize = (N, ksplit)
    lsize = (OPG, ksplit)
    # args: inputs, pql_sg, pqh_sg, ps_sg, pd_sg, outputs, input_len, output_len
    krn.set_args(x_b, pql_b, pqh_b, ps_b, pd_b, y_b, np.uint32(K), np.uint32(N))

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    # correctness
    enq(gpu.queue);  gpu.queue.finish()
    y_ocl = np.empty(N, dtype=np.float16)
    cl.enqueue_copy(gpu.queue, y_ocl, y_b);  gpu.queue.finish()
    corr_ok = True
    if y_ref is not None:
        y_ref_f16 = y_ref.astype(np.float16).astype(np.float32)
        corr_ok = check_close(f"first {vr_act} rows", y_ref_f16, y_ocl[:vr_act].astype(np.float32), verbose=verbose)

    # benchmark
    print(f"    Benchmarking {iters} iters + {warmup} warmup, L3 flush each ...")
    ts  = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
    st  = stats(ts)

    flops   = 2 * N * K
    w_bytes = N * nbpr * 210
    moved   = w_bytes + K * 2 + N * 2
    rl = hw.roofline(flops, moved, st["mean_ms"] * 1e-3)
    _print_result(st, rl, moved)

    return dict(label=label, corr_ok=corr_ok, timing=st, roofline=rl,
                weight_mb=w_bytes/1e6, sg_weight_mb=sg_bytes/1e6, moved_mb=moved/1e6)


# ---------------------------------------------------------------------------
# Q4_0 / Q8_0 (GGUF "legacy" quants) — SG transposed packing + reference
# ---------------------------------------------------------------------------
# Each GGUF block covers QK=32 weights.  We group EIGHT consecutive blocks
# into one 256-weight "super-block" so the same OPG/sub-block machinery used
# by the K-quant SG kernels applies unchanged.  Per super-block, per row:
#     Q4_0:  pqs 128 bytes (8*16 qs) + pd 16 bytes (8 fp16 d) = 144 == 8*18
#     Q8_0:  pqs 256 bytes (8*32 qs) + pd 16 bytes (8 fp16 d) = 272 == 8*34
# The Q4_0 qs nibble layout matches a Q4_K sub-block exactly, and dequant
# w=(nibble-8)*d is expressed as acc += d*dot - (8*d)*isum (scale/min combine).
# Q8_0 is signed int8 with w=q*d, i.e. acc += d*dot (no min term).

def gen_q4_0_blocks(n, seed=1234):
    """n x 18-byte block_q4_0: d(2) qs(16)."""
    rng = np.random.default_rng(seed)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    qs  = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    return np.concatenate([d, qs], axis=1).ravel()            # [n * 18]

def gen_q8_0_blocks(n, seed=1234):
    """n x 34-byte block_q8_0: d(2) qs(32, int8)."""
    rng = np.random.default_rng(seed)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    qs  = rng.integers(-128, 128, (n, 32), dtype=np.int8).view(np.uint8).reshape(n, 32)
    return np.concatenate([d, qs], axis=1).ravel()            # [n * 34]

def shuffle_q4_0(raw, K, N):
    """Row-major: gather 8 Q4_0 blocks -> super-block pqs[NB*128], pd[NB*16]."""
    NB  = N * K // 256
    pqs = np.zeros(NB * 128, dtype=np.uint8)
    pd  = np.zeros(NB * 16,  dtype=np.uint8)
    for sb in range(NB):
        for j in range(8):
            blk = raw[(sb * 8 + j) * 18:(sb * 8 + j) * 18 + 18]
            pd[sb * 16 + j * 2 : sb * 16 + j * 2 + 2]        = blk[0:2]
            pqs[sb * 128 + j * 16 : sb * 128 + j * 16 + 16]  = blk[2:18]
    return pqs, pd

def shuffle_q8_0(raw, K, N):
    """Row-major: gather 8 Q8_0 blocks -> super-block pqs[NB*256], pd[NB*16]."""
    NB  = N * K // 256
    pqs = np.zeros(NB * 256, dtype=np.uint8)
    pd  = np.zeros(NB * 16,  dtype=np.uint8)
    for sb in range(NB):
        for j in range(8):
            blk = raw[(sb * 8 + j) * 34:(sb * 8 + j) * 34 + 34]
            pd[sb * 16 + j * 2 : sb * 16 + j * 2 + 2]        = blk[0:2]
            pqs[sb * 256 + j * 32 : sb * 256 + j * 32 + 32]  = blk[2:34]
    return pqs, pd

def pack_q4_0_sg(pqs, pd, K, N, opg=OPG):
    """Row-major Q4_0 (pqs,pd) -> SG transposed (pqs_T, pd_T). Same total size.

    pqs_T (== pqs size): entry (h,bid) at (h*nbpr+bid)*opg*128, sub-block j at
        j*opg*16, chunk c(0..3) at j*opg*16 + c*opg*4, lane lid gets 4 bytes.
    pd_T  (== pd  size): entry (h,bid) at (h*nbpr+bid)*opg*16, SoA field j at
        j*opg*2, lane lid's fp16 at j*opg*2 + lid*2 (read via block_read_us8).
    """
    nbpr = K // 256
    if N % opg != 0:
        raise ValueError(f"N ({N}) must be divisible by opg ({opg}) for SG packing")
    nrg     = N // opg
    entries = nrg * nbpr
    pqs_T = np.zeros(entries * opg * 128, dtype=np.uint8)
    pd_T  = np.zeros(entries * opg * 16,  dtype=np.uint8)
    assert pqs_T.nbytes == pqs.nbytes, f"pqs_T size mismatch: {pqs_T.nbytes} vs {pqs.nbytes}"
    assert pd_T.nbytes  == pd.nbytes,  f"pd_T size mismatch: {pd_T.nbytes} vs {pd.nbytes}"
    for h in range(nrg):
        for bid in range(nbpr):
            pqs_entry = (h * nbpr + bid) * opg * 128
            pd_entry  = (h * nbpr + bid) * opg * 16
            for lid in range(opg):
                blk = (h * opg + lid) * nbpr + bid
                for j in range(8):
                    src = pqs[blk * 128 + j * 16 : blk * 128 + j * 16 + 16]
                    for c in range(4):
                        dst = pqs_entry + j * opg * 16 + c * opg * 4 + lid * 4
                        pqs_T[dst:dst + 4] = src[c * 4:c * 4 + 4]
                for j in range(8):
                    dst = pd_entry + j * opg * 2 + lid * 2
                    pd_T[dst:dst + 2] = pd[blk * 16 + j * 2 : blk * 16 + j * 2 + 2]
    return pqs_T, pd_T

def pack_q8_0_sg(pqs, pd, K, N, opg=OPG):
    """Row-major Q8_0 (pqs,pd) -> SG transposed (pqs_T, pd_T). Same total size.

    pqs_T (== pqs size): entry (h,bid) at (h*nbpr+bid)*opg*256, sub-block j at
        j*opg*32, chunk c(0..7) at j*opg*32 + c*opg*4, lane lid gets 4 bytes.
    pd_T  (== pd  size): identical SoA layout to pack_q4_0_sg.
    """
    nbpr = K // 256
    if N % opg != 0:
        raise ValueError(f"N ({N}) must be divisible by opg ({opg}) for SG packing")
    nrg     = N // opg
    entries = nrg * nbpr
    pqs_T = np.zeros(entries * opg * 256, dtype=np.uint8)
    pd_T  = np.zeros(entries * opg * 16,  dtype=np.uint8)
    assert pqs_T.nbytes == pqs.nbytes, f"pqs_T size mismatch: {pqs_T.nbytes} vs {pqs.nbytes}"
    assert pd_T.nbytes  == pd.nbytes,  f"pd_T size mismatch: {pd_T.nbytes} vs {pd.nbytes}"
    for h in range(nrg):
        for bid in range(nbpr):
            pqs_entry = (h * nbpr + bid) * opg * 256
            pd_entry  = (h * nbpr + bid) * opg * 16
            for lid in range(opg):
                blk = (h * opg + lid) * nbpr + bid
                for j in range(8):
                    src = pqs[blk * 256 + j * 32 : blk * 256 + j * 32 + 32]
                    for c in range(8):
                        dst = pqs_entry + j * opg * 32 + c * opg * 4 + lid * 4
                        pqs_T[dst:dst + 4] = src[c * 4:c * 4 + 4]
                for j in range(8):
                    dst = pd_entry + j * opg * 2 + lid * 2
                    pd_T[dst:dst + 2] = pd[blk * 16 + j * 2 : blk * 16 + j * 2 + 2]
    return pqs_T, pd_T

def deq_q4_0(pqs, pd, K, N):
    """Decode row-major shuffled Q4_0 -> W[N,K] float32.  w=(nibble-8)*d."""
    nprow = K // 256
    W = np.zeros((N, K), np.float32)
    for row in range(N):
        for bi in range(nprow):
            idx  = row * nprow + bi
            qs_b = pqs[idx * 128:(idx + 1) * 128]
            d_b  = pd [idx * 16 :(idx + 1) * 16]
            for j in range(8):
                d  = fp16_to_float(d_b[j * 2], d_b[j * 2 + 1])
                cb = bi * 256 + j * 32
                for k in range(16):
                    byte = int(qs_b[j * 16 + k])
                    W[row, cb + k]      = ((byte & 0xF) - 8) * d
                    W[row, cb + 16 + k] = (((byte >> 4) & 0xF) - 8) * d
    return W

def deq_q8_0(pqs, pd, K, N):
    """Decode row-major shuffled Q8_0 -> W[N,K] float32.  w=q*d (signed)."""
    nprow = K // 256
    qs_i8 = pqs.view(np.int8)
    W = np.zeros((N, K), np.float32)
    for row in range(N):
        for bi in range(nprow):
            idx  = row * nprow + bi
            qs_b = qs_i8[idx * 256:(idx + 1) * 256]
            d_b  = pd  [idx * 16 :(idx + 1) * 16]
            for j in range(8):
                d  = fp16_to_float(d_b[j * 2], d_b[j * 2 + 1])
                cb = bi * 256 + j * 32
                for i in range(32):
                    W[row, cb + i] = int(qs_b[j * 32 + i]) * d
    return W

def make_q4_0_sg_bufs(gpu, K, N, vr=64, seed=1234):
    """Build SG-packed Q4_0 weight buffers + reference output for first vr rows."""
    nbpr = K // 256;  vr = min(vr, N)
    rng  = np.random.default_rng(seed ^ 0xABCD)
    pqs_v = pd_v = None
    if vr > 0:
        raw = gen_q4_0_blocks(vr * nbpr * 8, seed=seed)
        pqs_v, pd_v = shuffle_q4_0(raw, K, vr)
    total_blk = N * nbpr
    pqs_f = np.empty(total_blk * 128, dtype=np.uint8)
    pd_f  = np.empty(total_blk * 16,  dtype=np.uint8)
    if vr > 0:
        pqs_f[:vr*nbpr*128] = pqs_v
        pd_f [:vr*nbpr*16]  = pd_v
    rest = (N - vr) * nbpr
    if rest > 0:
        pqs_f[vr*nbpr*128:] = rng.integers(0, 256, rest*128, dtype=np.uint8)
        pd_f [vr*nbpr*16:]  = rng.integers(0, 256, rest*16,  dtype=np.uint8)
    t0 = time.perf_counter()
    pqs_sg, pd_sg = pack_q4_0_sg(pqs_f, pd_f, K, N)
    print(f"    SG pack time: {(time.perf_counter()-t0)*1e3:.0f} ms")
    x     = rng.standard_normal(K).astype(np.float16)
    y_ref = (deq_q4_0(pqs_v, pd_v, K, vr) @ x.astype(np.float32)) if vr > 0 else None
    return (_cbuf(gpu.ctx, pqs_sg), _cbuf(gpu.ctx, pd_sg),
            _cbuf(gpu.ctx, x), y_ref, vr, pqs_sg.nbytes + pd_sg.nbytes)

def make_q8_0_sg_bufs(gpu, K, N, vr=64, seed=1234):
    """Build SG-packed Q8_0 weight buffers + reference output for first vr rows."""
    nbpr = K // 256;  vr = min(vr, N)
    rng  = np.random.default_rng(seed ^ 0xABCD)
    pqs_v = pd_v = None
    if vr > 0:
        raw = gen_q8_0_blocks(vr * nbpr * 8, seed=seed)
        pqs_v, pd_v = shuffle_q8_0(raw, K, vr)
    total_blk = N * nbpr
    pqs_f = np.empty(total_blk * 256, dtype=np.uint8)
    pd_f  = np.empty(total_blk * 16,  dtype=np.uint8)
    if vr > 0:
        pqs_f[:vr*nbpr*256] = pqs_v
        pd_f [:vr*nbpr*16]  = pd_v
    rest = (N - vr) * nbpr
    if rest > 0:
        pqs_f[vr*nbpr*256:] = rng.integers(0, 256, rest*256, dtype=np.uint8)
        pd_f [vr*nbpr*16:]  = rng.integers(0, 256, rest*16,  dtype=np.uint8)
    t0 = time.perf_counter()
    pqs_sg, pd_sg = pack_q8_0_sg(pqs_f, pd_f, K, N)
    print(f"    SG pack time: {(time.perf_counter()-t0)*1e3:.0f} ms")
    x     = rng.standard_normal(K).astype(np.float16)
    y_ref = (deq_q8_0(pqs_v, pd_v, K, vr) @ x.astype(np.float32)) if vr > 0 else None
    return (_cbuf(gpu.ctx, pqs_sg), _cbuf(gpu.ctx, pd_sg),
            _cbuf(gpu.ctx, x), y_ref, vr, pqs_sg.nbytes + pd_sg.nbytes)

def bench_q4_0_sg(gpu, hw, K, N, iters, warmup, vr=64, seed=1234, verbose=False):
    label = f"Q4_0_SG K={K} N={N}"
    print(f"\n  [{label}]")
    nbpr = K // 256
    t0 = time.perf_counter()
    pqs_b, pd_b, x_b, y_ref, vr_act, sg_bytes = make_q4_0_sg_bufs(gpu, K, N, vr, seed)
    y_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=N * 2)
    print(f"    Buffer total: {(time.perf_counter()-t0)*1e3:.0f} ms  "
          f"sg_weight={sg_bytes/1e6:.1f} MB  orig={N * nbpr * 144 / 1e6:.1f} MB  "
          f"(SG weights same size as orig; only layout differs)")
    ksplit = choose_ksplit(N, K)
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit}"
    prog = gpu.build(cl_src("gemv_q4_0_sg.cl"), opts)
    krn  = cl.Kernel(prog, "gemv_q4_0_sg")
    print(f"    KSPLIT={ksplit}  work-groups={N//OPG}  live_subgroups={(N//OPG)*ksplit}")
    gsize = (N, ksplit)
    lsize = (OPG, ksplit)
    krn.set_args(x_b, pqs_b, pd_b, y_b, np.uint32(K), np.uint32(N))
    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)
    enq(gpu.queue);  gpu.queue.finish()
    y_ocl = np.empty(N, dtype=np.float16)
    cl.enqueue_copy(gpu.queue, y_ocl, y_b);  gpu.queue.finish()
    corr_ok = True
    if y_ref is not None:
        y_ref_f16 = y_ref.astype(np.float16).astype(np.float32)
        corr_ok = check_close(f"first {vr_act} rows", y_ref_f16, y_ocl[:vr_act].astype(np.float32), verbose=verbose)
    print(f"    Benchmarking {iters} iters + {warmup} warmup, L3 flush each ...")
    ts  = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
    st  = stats(ts)
    flops   = 2 * N * K
    w_bytes = N * nbpr * 144
    moved   = w_bytes + K * 2 + N * 2
    rl = hw.roofline(flops, moved, st["mean_ms"] * 1e-3)
    _print_result(st, rl, moved)
    return dict(label=label, corr_ok=corr_ok, timing=st, roofline=rl,
                weight_mb=w_bytes/1e6, sg_weight_mb=sg_bytes/1e6, moved_mb=moved/1e6)

def bench_q8_0_sg(gpu, hw, K, N, iters, warmup, vr=64, seed=1234, verbose=False):
    label = f"Q8_0_SG K={K} N={N}"
    print(f"\n  [{label}]")
    nbpr = K // 256
    t0 = time.perf_counter()
    pqs_b, pd_b, x_b, y_ref, vr_act, sg_bytes = make_q8_0_sg_bufs(gpu, K, N, vr, seed)
    y_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=N * 2)
    print(f"    Buffer total: {(time.perf_counter()-t0)*1e3:.0f} ms  "
          f"sg_weight={sg_bytes/1e6:.1f} MB  orig={N * nbpr * 272 / 1e6:.1f} MB  "
          f"(SG weights same size as orig; only layout differs)")
    ksplit = choose_ksplit(N, K)
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit}"
    prog = gpu.build(cl_src("gemv_q8_0_sg.cl"), opts)
    krn  = cl.Kernel(prog, "gemv_q8_0_sg")
    print(f"    KSPLIT={ksplit}  work-groups={N//OPG}  live_subgroups={(N//OPG)*ksplit}")
    gsize = (N, ksplit)
    lsize = (OPG, ksplit)
    krn.set_args(x_b, pqs_b, pd_b, y_b, np.uint32(K), np.uint32(N))
    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)
    enq(gpu.queue);  gpu.queue.finish()
    y_ocl = np.empty(N, dtype=np.float16)
    cl.enqueue_copy(gpu.queue, y_ocl, y_b);  gpu.queue.finish()
    corr_ok = True
    if y_ref is not None:
        y_ref_f16 = y_ref.astype(np.float16).astype(np.float32)
        corr_ok = check_close(f"first {vr_act} rows", y_ref_f16, y_ocl[:vr_act].astype(np.float32), verbose=verbose)
    print(f"    Benchmarking {iters} iters + {warmup} warmup, L3 flush each ...")
    ts  = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
    st  = stats(ts)
    flops   = 2 * N * K
    w_bytes = N * nbpr * 272
    moved   = w_bytes + K * 2 + N * 2
    rl = hw.roofline(flops, moved, st["mean_ms"] * 1e-3)
    _print_result(st, rl, moved)
    return dict(label=label, corr_ok=corr_ok, timing=st, roofline=rl,
                weight_mb=w_bytes/1e6, sg_weight_mb=sg_bytes/1e6, moved_mb=moved/1e6)


# ---------------------------------------------------------------------------
# Q4_1 (GGUF "legacy" quant) — SG transposed packing + reference
# ---------------------------------------------------------------------------
# block_q4_1 = { half d; half m; uint8_t qs[16]; }  = 20 bytes, QK=32 weights.
# We group EIGHT consecutive blocks into one 256-weight super-block, mirroring
# gemv_q4_0_sg.  Per super-block, per row:
#     pqs 128 bytes (8*16 qs) + pd 16 bytes (8 fp16 d) + pm 16 bytes (8 fp16 m)
#     = 160 == 8*20  (pure reorder — same total size).
# Q4_1 dequant is w = nibble*d + m (nibble used DIRECTLY, no -8 offset), so the
# GEMV combine is acc += d*dot + m*isum (a per-block min term, "+" sign) —
# vs Q4_0's acc += d*dot - (8*d)*isum.

def gen_q4_1_blocks(n, seed=1234):
    """n x 20-byte block_q4_1: d(2) m(2) qs(16)."""
    rng = np.random.default_rng(seed)
    d   = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    m   = np.float16(rng.uniform(-0.05, 0.05, n)).view(np.uint8).reshape(n, 2)
    qs  = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    return np.concatenate([d, m, qs], axis=1).ravel()         # [n * 20]

def shuffle_q4_1(raw, K, N):
    """Row-major: gather 8 Q4_1 blocks -> pqs[NB*128], pd[NB*16], pm[NB*16]."""
    NB  = N * K // 256
    pqs = np.zeros(NB * 128, dtype=np.uint8)
    pd  = np.zeros(NB * 16,  dtype=np.uint8)
    pm  = np.zeros(NB * 16,  dtype=np.uint8)
    for sb in range(NB):
        for j in range(8):
            blk = raw[(sb * 8 + j) * 20:(sb * 8 + j) * 20 + 20]
            pd[sb * 16 + j * 2 : sb * 16 + j * 2 + 2]        = blk[0:2]
            pm[sb * 16 + j * 2 : sb * 16 + j * 2 + 2]        = blk[2:4]
            pqs[sb * 128 + j * 16 : sb * 128 + j * 16 + 16]  = blk[4:20]
    return pqs, pd, pm

def pack_q4_1_sg(pqs, pd, pm, K, N, opg=OPG):
    """Row-major Q4_1 (pqs,pd,pm) -> SG transposed (pqs_T,pd_T,pm_T). Same size.

    pqs_T (== pqs size): entry (h,bid) at (h*nbpr+bid)*opg*128, sub-block j at
        j*opg*16, chunk c(0..3) at j*opg*16 + c*opg*4, lane lid gets 4 bytes.
    pd_T/pm_T (== pd/pm size): entry (h,bid) at (h*nbpr+bid)*opg*16, SoA field j
        at j*opg*2, lane lid's fp16 at j*opg*2 + lid*2 (read via block_read_us8).
    """
    nbpr = K // 256
    if N % opg != 0:
        raise ValueError(f"N ({N}) must be divisible by opg ({opg}) for SG packing")
    nrg     = N // opg
    entries = nrg * nbpr
    pqs_T = np.zeros(entries * opg * 128, dtype=np.uint8)
    pd_T  = np.zeros(entries * opg * 16,  dtype=np.uint8)
    pm_T  = np.zeros(entries * opg * 16,  dtype=np.uint8)
    assert pqs_T.nbytes == pqs.nbytes, f"pqs_T size mismatch: {pqs_T.nbytes} vs {pqs.nbytes}"
    assert pd_T.nbytes  == pd.nbytes,  f"pd_T size mismatch: {pd_T.nbytes} vs {pd.nbytes}"
    assert pm_T.nbytes  == pm.nbytes,  f"pm_T size mismatch: {pm_T.nbytes} vs {pm.nbytes}"
    for h in range(nrg):
        for bid in range(nbpr):
            pqs_entry = (h * nbpr + bid) * opg * 128
            pd_entry  = (h * nbpr + bid) * opg * 16
            for lid in range(opg):
                blk = (h * opg + lid) * nbpr + bid
                for j in range(8):
                    src = pqs[blk * 128 + j * 16 : blk * 128 + j * 16 + 16]
                    for c in range(4):
                        dst = pqs_entry + j * opg * 16 + c * opg * 4 + lid * 4
                        pqs_T[dst:dst + 4] = src[c * 4:c * 4 + 4]
                for j in range(8):
                    dst = pd_entry + j * opg * 2 + lid * 2
                    pd_T[dst:dst + 2] = pd[blk * 16 + j * 2 : blk * 16 + j * 2 + 2]
                    pm_T[dst:dst + 2] = pm[blk * 16 + j * 2 : blk * 16 + j * 2 + 2]
    return pqs_T, pd_T, pm_T

def deq_q4_1(pqs, pd, pm, K, N):
    """Decode row-major shuffled Q4_1 -> W[N,K] float32.  w = nibble*d + m."""
    nprow = K // 256
    W = np.zeros((N, K), np.float32)
    for row in range(N):
        for bi in range(nprow):
            idx  = row * nprow + bi
            qs_b = pqs[idx * 128:(idx + 1) * 128]
            d_b  = pd [idx * 16 :(idx + 1) * 16]
            m_b  = pm [idx * 16 :(idx + 1) * 16]
            for j in range(8):
                d  = fp16_to_float(d_b[j * 2], d_b[j * 2 + 1])
                m  = fp16_to_float(m_b[j * 2], m_b[j * 2 + 1])
                cb = bi * 256 + j * 32
                for k in range(16):
                    byte = int(qs_b[j * 16 + k])
                    W[row, cb + k]      = (byte & 0xF) * d + m
                    W[row, cb + 16 + k] = ((byte >> 4) & 0xF) * d + m
    return W

def make_q4_1_sg_bufs(gpu, K, N, vr=64, seed=1234):
    """Build SG-packed Q4_1 weight buffers + reference output for first vr rows."""
    nbpr = K // 256;  vr = min(vr, N)
    rng  = np.random.default_rng(seed ^ 0xABCD)
    pqs_v = pd_v = pm_v = None
    if vr > 0:
        raw = gen_q4_1_blocks(vr * nbpr * 8, seed=seed)
        pqs_v, pd_v, pm_v = shuffle_q4_1(raw, K, vr)
    total_blk = N * nbpr
    pqs_f = np.empty(total_blk * 128, dtype=np.uint8)
    pd_f  = np.empty(total_blk * 16,  dtype=np.uint8)
    pm_f  = np.empty(total_blk * 16,  dtype=np.uint8)
    if vr > 0:
        pqs_f[:vr*nbpr*128] = pqs_v
        pd_f [:vr*nbpr*16]  = pd_v
        pm_f [:vr*nbpr*16]  = pm_v
    rest = (N - vr) * nbpr
    if rest > 0:
        pqs_f[vr*nbpr*128:] = rng.integers(0, 256, rest*128, dtype=np.uint8)
        pd_f [vr*nbpr*16:]  = rng.integers(0, 256, rest*16,  dtype=np.uint8)
        pm_f [vr*nbpr*16:]  = rng.integers(0, 256, rest*16,  dtype=np.uint8)
    t0 = time.perf_counter()
    pqs_sg, pd_sg, pm_sg = pack_q4_1_sg(pqs_f, pd_f, pm_f, K, N)
    print(f"    SG pack time: {(time.perf_counter()-t0)*1e3:.0f} ms")
    x     = rng.standard_normal(K).astype(np.float16)
    y_ref = (deq_q4_1(pqs_v, pd_v, pm_v, K, vr) @ x.astype(np.float32)) if vr > 0 else None
    return (_cbuf(gpu.ctx, pqs_sg), _cbuf(gpu.ctx, pd_sg), _cbuf(gpu.ctx, pm_sg),
            _cbuf(gpu.ctx, x), y_ref, vr, pqs_sg.nbytes + pd_sg.nbytes + pm_sg.nbytes)

def bench_q4_1_sg(gpu, hw, K, N, iters, warmup, vr=64, seed=1234, verbose=False):
    label = f"Q4_1_SG K={K} N={N}"
    print(f"\n  [{label}]")
    nbpr = K // 256
    t0 = time.perf_counter()
    pqs_b, pd_b, pm_b, x_b, y_ref, vr_act, sg_bytes = make_q4_1_sg_bufs(gpu, K, N, vr, seed)
    y_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=N * 2)
    print(f"    Buffer total: {(time.perf_counter()-t0)*1e3:.0f} ms  "
          f"sg_weight={sg_bytes/1e6:.1f} MB  orig={N * nbpr * 160 / 1e6:.1f} MB  "
          f"(SG weights same size as orig; only layout differs)")
    ksplit = choose_ksplit(N, K)
    opts = f"-cl-std=CL2.0 -DKSPLIT={ksplit}"
    prog = gpu.build(cl_src("gemv_q4_1_sg.cl"), opts)
    krn  = cl.Kernel(prog, "gemv_q4_1_sg")
    print(f"    KSPLIT={ksplit}  work-groups={N//OPG}  live_subgroups={(N//OPG)*ksplit}")
    gsize = (N, ksplit)
    lsize = (OPG, ksplit)
    krn.set_args(x_b, pqs_b, pd_b, pm_b, y_b, np.uint32(K), np.uint32(N))
    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)
    enq(gpu.queue);  gpu.queue.finish()
    y_ocl = np.empty(N, dtype=np.float16)
    cl.enqueue_copy(gpu.queue, y_ocl, y_b);  gpu.queue.finish()
    corr_ok = True
    if y_ref is not None:
        y_ref_f16 = y_ref.astype(np.float16).astype(np.float32)
        corr_ok = check_close(f"first {vr_act} rows", y_ref_f16, y_ocl[:vr_act].astype(np.float32), verbose=verbose)
    print(f"    Benchmarking {iters} iters + {warmup} warmup, L3 flush each ...")
    ts  = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
    st  = stats(ts)
    flops   = 2 * N * K
    w_bytes = N * nbpr * 160
    moved   = w_bytes + K * 2 + N * 2
    rl = hw.roofline(flops, moved, st["mean_ms"] * 1e-3)
    _print_result(st, rl, moved)
    return dict(label=label, corr_ok=corr_ok, timing=st, roofline=rl,
                weight_mb=w_bytes/1e6, sg_weight_mb=sg_bytes/1e6, moved_mb=moved/1e6)



# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    {"quant": "q4_0", "K": 12288, "N":   4096, "label": "Q4_0_SG_K12288_N4096"},
    {"quant": "q8_0", "K": 12288, "N":   4096, "label": "Q8_0_SG_K12288_N4096"},
    {"quant": "q4_0", "K":  4096, "N":   4096, "label": "Q4_0_SG_K4096_N4096"},
    {"quant": "q8_0", "K":  4096, "N":   4096, "label": "Q8_0_SG_K4096_N4096"},
    {"quant": "q4_0", "K":  4096, "N":   1024, "label": "Q4_0_SG_K4096_N1024"},
    {"quant": "q8_0", "K":  4096, "N":   1024, "label": "Q8_0_SG_K4096_N1024"},
    {"quant": "q4_1", "K": 12288, "N":   4096, "label": "Q4_1_SG_K12288_N4096"},
    {"quant": "q4_1", "K":  4096, "N":   4096, "label": "Q4_1_SG_K4096_N4096"},
    {"quant": "q4_1", "K":  4096, "N":   1024, "label": "Q4_1_SG_K4096_N1024"},
    #{"quant": "q6k", "K":  4096, "N": 151936, "label": "Q6K_SG_K4096_N151936"},
    {"quant": "q6k", "K": 12288, "N":   4096, "label": "Q6K_SG_K12288_N4096"},
    {"quant": "q4k", "K": 12288, "N":   4096, "label": "Q4K_SG_K12288_N4096"},
    {"quant": "q4k", "K":  4096, "N":  12288, "label": "Q4K_SG_K4096_N12288"},
    {"quant": "q4k", "K":  4096, "N":   4096, "label": "Q4K_SG_K4096_N4096"},
    {"quant": "q6k", "K":  4096, "N":   1024, "label": "Q6K_SG_K4096_N1024"},
    {"quant": "q4k", "K":  4096, "N":   1024, "label": "Q4K_SG_K4096_N1024"},
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",      type=int,   default=0)
    ap.add_argument("--iters",       type=int,   default=500)
    ap.add_argument("--warmup",      type=int,   default=10)
    ap.add_argument("--flush_mib",   type=int,   default=256)
    ap.add_argument("--peak_bw",     type=float, default=512.0)
    ap.add_argument("--peak_fp32",   type=float, default=9800.0)
    ap.add_argument("--verify_rows", type=int,   default=64)
    ap.add_argument("--verbose",     action="store_true")
    args = ap.parse_args()

    hw  = HW(peak_bw_gbps=args.peak_bw, fp32_gflops=args.peak_fp32)
    gpu = Gpu(args.device)
    gpu.setup_flush(args.flush_mib)

    SEP = "=" * 100
    print(SEP)
    print(f"  Peak FP32: {hw.fp32_gflops:.0f} GFLOPS | Peak BW: {hw.peak_bw_gbps:.1f} GB/s | "
          f"L3 flush: {args.flush_mib} MiB")
    print(f"  Iters: {args.iters} + {args.warmup} warmup | verify_rows={args.verify_rows}")
    print(f"  SG transposed weight layout: SAME total bytes as original (pure reorder)")
    print(f"  intel_sub_group_block_read path enabled (device semantics verified)")
    print(SEP)

    results = []
    for case in TEST_CASES:
        try:
            _DISPATCH = {"q4k": bench_q4k_sg,  "q6k": bench_q6k_sg,
                         "q4_0": bench_q4_0_sg, "q8_0": bench_q8_0_sg,
                         "q4_1": bench_q4_1_sg}
            fn  = _DISPATCH[case["quant"]]
            rec = fn(gpu, hw, case["K"], case["N"],
                     args.iters, args.warmup, args.verify_rows, verbose=args.verbose)
            rec["label"] = case["label"]
            results.append(rec)
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {case['label']}: {exc}")
            traceback.print_exc()
            results.append({"label": case["label"], "corr_ok": False,
                            "timing": {}, "roofline": {}})

    W = 112
    print("\n" + "=" * W)
    print(f"  {'Case':<30} {'Corr':>5}  {'mean ms':>9}  {'BW GB/s':>9}  "
          f"{'BW%':>6}  {'GFLOPS':>8}  {'Roof%':>7}  {'Bound':>7}  {'moved MB':>9}")
    print("-" * W)
    for r in results:
        rl = r.get("roofline", {});  st = r.get("timing", {})
        print(f"  {r['label']:<30} {'OK' if r.get('corr_ok') else 'FAIL':>5}"
              f"  {st.get('mean_ms',float('nan')):>9.3f}"
              f"  {rl.get('achieved_bw_gbps',float('nan')):>9.1f}"
              f"  {rl.get('bw_util_pct',float('nan')):>6.1f}%"
              f"  {rl.get('achieved_gflops',float('nan')):>8.1f}"
              f"  {rl.get('roofline_pct',float('nan')):>7.1f}%"
              f"  {rl.get('bound_by','-'):>7}"
              f"  {r.get('moved_mb',float('nan')):>9.1f}")
    print("=" * W)
    n_ok = sum(1 for r in results if r.get("corr_ok"))
    print(f"\n  {n_ok}/{len(results)} cases passed correctness.\n")
    return 0 if n_ok == len(results) else 1

if __name__ == "__main__":
    raise SystemExit(main())
