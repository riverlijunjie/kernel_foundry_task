"""
test_q80_gemv_sg.py

Correctness + benchmark + roofline harness for the high-performance "SG"
(sub-group block-read) Q8_0 GEMV OpenCL kernel:

    fc_gguf_q8_0_sg.cl : fc_gguf_q8_0_sg

The kernel transposes the weight matrix into the "SG" column-major layout so
that a single intel_sub_group_block_read8 delivers 32 consecutive int8 weights
(8 chunks of 4 bytes) to all 16 lanes simultaneously, maximising DRAM
transaction width for this memory-bound GEMV.

== Q8_0 recap ==
  GGUF block_q8_0 = { half d; int8_t qs[32]; }  (34 bytes)
  Eight consecutive blocks form a super-block (256 weights, 272 bytes):
    pqs: 8 * 32 = 256 bytes  (signed int8 weights)
    pd:  8 * 2  =  16 bytes  (fp16 scales, one per sub-block of 32)
  dequant: w = q * d  (no zero-point)

== Test shapes ==
    K=2048, N=512   -- small output (e.g. attention projection)
    K=2048, N=4096  -- mid-size output
    K=2048, N=8192  -- large output (e.g. FFN up-projection)

Usage:
    python test_q80_gemv_sg.py [--device 0] [--iters 50] [--warmup 5]
        [--flush_mib 256] [--peak_bw 512] [--peak_fp32 9800]
        [--verbose] [--no-bench]
"""

import os
import statistics
import argparse

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    raise SystemExit("pyopencl is required: pip install pyopencl")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPG = 16   # sub_group_size = output rows per sub-group
KERNEL_FILE = os.path.join(SCRIPT_DIR, "fc_gguf_q8_0_sg.cl")


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
# Raw GGUF Q8_0 block generator
# ---------------------------------------------------------------------------
def gen_q80_blocks(n, seed=1234):
    """Generate n Q8_0 GGUF blocks: { half d; int8_t qs[32]; } = 34 bytes each."""
    rng = np.random.default_rng(seed)
    d = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    qs = rng.integers(-128, 127, (n, 32), dtype=np.int8).view(np.uint8)
    return np.concatenate([d, qs], axis=1).ravel()


# ---------------------------------------------------------------------------
# GGUF layout -> row-major separated-section layout
#
# Q8_0 "pure reorder" (no bit manipulation, just field separation):
#   Input:  N * K // 32 Q8_0 blocks of 34 bytes each
#   Output: pqs (N * K // 256 super-blocks * 256 bytes) + pd (... * 16 bytes)
# ---------------------------------------------------------------------------
def shuffle_q80(raw, K, N):
    """Separate GGUF Q8_0 blocks into pqs (weights) and pd (scales).

    Groups 8 consecutive Q8_0 blocks into one super-block.
    Returns:
        pqs: flat uint8 array of shape (N*K//256, 256)  (signed int8 weights)
        pd:  flat uint8 array of shape (N*K//256, 16)   (8 fp16 scales per super-block)
    """
    n_sub = N * K // 32           # total number of Q8_0 blocks
    n_super = n_sub // 8          # = N * K // 256 super-blocks
    blk = raw.reshape(n_sub, 34)  # [n_sub, 34]:  2 bytes d + 32 bytes qs
    d_bytes = blk[:, 0:2]         # fp16 scale bytes per sub-block
    qs_bytes = blk[:, 2:34]       # 32 signed int8 weights per sub-block

    # Group 8 sub-blocks into super-blocks
    d_super = d_bytes.reshape(n_super, 8, 2)    # [NB, 8, 2]
    qs_super = qs_bytes.reshape(n_super, 8, 32)  # [NB, 8, 32]

    pqs = qs_super.reshape(n_super, 256).astype(np.uint8)  # [NB, 256]
    pd = d_super.reshape(n_super, 16).astype(np.uint8)     # [NB, 16]
    return pqs.ravel(), pd.ravel()


# ---------------------------------------------------------------------------
# Reference dequantize: pqs + pd -> W[N, K] float32
# ---------------------------------------------------------------------------
def deq_q80(pqs, pd_bytes, K, N):
    """Dequantize Q8_0 shuffled layout to float32.

    pqs: flat uint8 (N*K//256 super-blocks * 256 int8 weights each)
    pd_bytes: flat uint8 (N*K//256 super-blocks * 16 bytes = 8 fp16 scales each)
    Returns W: (N, K) float32
    """
    nbpr = K // 256
    NB = N * nbpr
    qs_b = pqs.reshape(NB, 8, 32).view(np.int8).astype(np.float32)  # [NB, 8, 32]
    # Interpret pairs of bytes as fp16 -> extract 8 scales per super-block
    d_f32 = pd_bytes.reshape(NB, 8, 2).copy().view(np.float16).reshape(NB, 8).astype(np.float32)
    # w = q * d  (no zero-point)
    W_block = qs_b * d_f32[:, :, np.newaxis]  # [NB, 8, 32]
    return W_block.reshape(N, nbpr, 256).reshape(N, K)


# ---------------------------------------------------------------------------
# SG-transposed packing: row-major shuffled section -> SG column-major section
#
# Block index = row*nbpr + bid,  row = h*OPG + lid.
#
# pqs_T layout: for SG block (h, bid), sub-block j, chunk c, lane l:
#   byte offset = ((h*nbpr+bid)*OPG*64 + j*OPG*8 + c*OPG + l) * 4
#   which matches intel_sub_group_block_read8 access pattern.
#
# pd_T layout: for SG block (h, bid), sub-block j, lane l:
#   ushort offset = (h*nbpr+bid)*OPG*8 + j*OPG + l
#   which matches intel_sub_group_block_read_us8 access pattern.
# ---------------------------------------------------------------------------
def _ascon(a):
    return np.ascontiguousarray(a)


def pack_q80_pqs_sg(pqs, K, N, opg=OPG):
    """Pack Q8_0 weights into SG column-major layout.

    pqs: flat uint8 of shape (N*K//256, 256) = 8 sub-blocks × 32 bytes = 8×8 chunks×4 bytes
    Returns SG-packed uint8 flat array.
    """
    nbpr = K // 256
    nrg = N // opg
    # src[h, l, bid, j, c, b]  where row = h*opg + l, chunk c (0..7), 4 bytes
    src = pqs.reshape(nrg, opg, nbpr, 8, 8, 4)
    # dst[h, bid, j, c, l, b] — interleave lanes for block-read
    dst = src.transpose(0, 2, 3, 4, 1, 5)
    return _ascon(dst).reshape(-1).astype(np.uint8)


def pack_q80_pd_sg(pd, K, N, opg=OPG):
    """Pack Q8_0 fp16 scales into SG SoA layout.

    pd: flat uint8 of shape (N*K//256, 16) = 8 sub-blocks × 2 fp16 bytes
    Returns SG-packed uint8 flat array.
    """
    nbpr = K // 256
    nrg = N // opg
    # src[h, l, bid, j, b]  (2 bytes per fp16 scale)
    src = pd.reshape(nrg, opg, nbpr, 8, 2)
    # dst[h, bid, j, l, b] — SoA: sub-block j's scales for all lanes contiguous
    dst = src.transpose(0, 2, 3, 1, 4)
    return _ascon(dst).reshape(-1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Build the SG-packed weight buffer W = [pqs_T | pd_T]
# ---------------------------------------------------------------------------
def build_q80_sg_weight(K, N, seed=1234):
    """Generate a random Q8_0 weight matrix and pack into SG layout.

    Returns:
        W_buf:  flat uint8 array = [pqs_T | pd_T]
        ref_W:  (N, K) float32 reference weight matrix
        w_bytes: total weight buffer size in bytes
    """
    n_sub = N * K // 32
    raw = gen_q80_blocks(n_sub, seed=seed)
    pqs, pd = shuffle_q80(raw, K, N)
    ref_W = deq_q80(pqs, pd, K, N)
    pqs_T = pack_q80_pqs_sg(pqs, K, N)
    pd_T = pack_q80_pd_sg(pd, K, N)
    W_buf = np.concatenate([pqs_T, pd_T]).astype(np.uint8)
    # Sanity: buffer size should equal (N*K//256) * 272
    expected = (N * K // 256) * (256 + 16)
    assert W_buf.nbytes == expected, f"buffer size mismatch: {W_buf.nbytes} != {expected}"
    return W_buf, ref_W, W_buf.nbytes


# ---------------------------------------------------------------------------
# KSPLIT heuristic
# ---------------------------------------------------------------------------
def _pow2_floor(x):
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def choose_ksplit(base_groups, split_space, target=2048, maxk=8):
    force = os.environ.get("Q80_KSPLIT", "")
    if force:
        return int(force)
    if base_groups >= 512:
        return 1
    ks = max(1, round(target / max(base_groups, 1)))
    ks = min(ks, split_space, maxk)
    return _pow2_floor(ks)


def ksplit_candidates(split_space, maxk=16):
    cands = []
    ks = 1
    limit = max(1, min(split_space, maxk))
    while ks <= limit:
        cands.append(ks)
        ks *= 2
    return cands


def autotune_ksplit(gpu, candidates, make_dispatch, iters=15, warmup=4):
    """Empirically pick the fastest KSPLIT value."""
    force = os.environ.get("Q80_KSPLIT", "")
    if force:
        ks = int(force)
        return ks, {ks: 0.0}

    results = {}
    best_ks, best_t = candidates[0], float("inf")
    for ks in candidates:
        opts = f"-cl-std=CL2.0 -DKSPLIT={ks}"
        # Build opts are embedded in make_dispatch via the ks argument
        args, gsize, lsize, build_opts = make_dispatch(ks)
        prog = gpu.build(KERNEL_FILE, build_opts)
        krn = cl.Kernel(prog, "fc_gguf_q8_0_sg")
        krn.set_args(*args)

        def enq(q, _krn=krn, _g=gsize, _l=lsize):
            return cl.enqueue_nd_range_kernel(q, _krn, _g, _l)

        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        t = statistics.median(ts)
        results[ks] = t
        if t < best_t:
            best_t, best_ks = t, ks
    return best_ks, results


# ===========================================================================
# Main test function
# ===========================================================================
def test_q80_gemv_sg(gpu, hw, K=2048, N=512, BM=1, seed=1, verbose=False,
                     iters=50, warmup=5, do_bench=True,
                     autotune=False, autotune_iters=15, autotune_warmup=4):
    """Test fc_gguf_q8_0_sg for one (K, N, BM) shape.

    Args:
        K: number of input features (must be multiple of 256)
        N: number of output features (must be multiple of OPG=16)
        BM: batch size (number of activation vectors)
    """
    label = f"fc_gguf_q8_0_sg  K={K} N={N} BM={BM}"
    print(f"\n  [{label}]")
    assert K % 256 == 0, "K must be a multiple of 256"
    assert N % OPG == 0, "N must be a multiple of OPG (16)"

    # ---- Build weight buffer ----
    print(f"    Building Q8_0 SG-packed weight buffer (K={K} N={N}) ...")
    W_buf, ref_W, w_bytes = build_q80_sg_weight(K, N, seed=seed)

    # ---- Generate activations (fp16) ----
    rng = np.random.default_rng(seed + 777)
    x = rng.standard_normal((BM, K)).astype(np.float16)
    x_f32 = x.astype(np.float32)

    # ---- Reference: C[bm, n] = sum_k x[bm, k] * W[n, k] ----
    ref = (x_f32 @ ref_W.T).astype(np.float32)   # [BM, N]

    # ---- GPU buffers ----
    A_b = _cbuf(gpu.ctx, x)
    W_b = _cbuf(gpu.ctx, W_buf)
    C_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=BM * N * 4)

    # ---- KSPLIT selection ----
    base_groups = (N // OPG) * BM
    split_space = K // 256

    def make_dispatch(ks):
        build_opts = f"-cl-std=CL2.0 -DKSPLIT={ks} -DK_SIZE={K} -DN_SIZE={N}"
        gsize = (N, BM, ks)
        lsize = (OPG, 1, ks)
        args = (A_b, W_b, C_b)
        return args, gsize, lsize, build_opts

    if autotune and do_bench:
        candidates = ksplit_candidates(split_space)
        ksplit, at_res = autotune_ksplit(gpu, candidates, make_dispatch,
                                         iters=autotune_iters, warmup=autotune_warmup)
        pretty = {k: round(v * 1e3, 4) for k, v in at_res.items()}
        print(f"    [autotune] KSPLIT candidates {candidates} -> chose {ksplit}  "
              f"(median ms: {pretty})")
    else:
        ksplit = choose_ksplit(base_groups, split_space)

    args, gsize, lsize, build_opts = make_dispatch(ksplit)
    prog = gpu.build(KERNEL_FILE, build_opts)
    krn = cl.Kernel(prog, "fc_gguf_q8_0_sg")
    krn.set_args(*args)

    print(f"    KSPLIT={ksplit}  gsize={gsize}  lsize={lsize}")
    print(f"    work-groups={base_groups}  live_subgroups={base_groups * ksplit}")

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    # ---- Correctness check ----
    ev = enq(gpu.queue)
    ev.wait()
    got = np.empty(BM * N, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, C_b)
    gpu.queue.finish()
    got = got.reshape(BM, N)

    ok = check_close(f"fc_gguf_q8_0_sg K={K} N={N}", ref, got, verbose=verbose)

    # ---- Benchmark ----
    timing = rl = moved = None
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        timing = stats(ts)
        # FLOPs: BM * N * K multiply-adds
        flops = 2.0 * BM * N * K
        # bytes moved: weight buffer + activations (fp16) + output (fp32)
        moved = w_bytes + BM * K * 2 + BM * N * 4
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  median={timing['median_ms']:.3f} ms  "
              f"min={timing['min_ms']:.3f} ms  max={timing['max_ms']:.3f} ms  (n={timing['iters']})")
        _print_roofline(rl, moved)
        nbpr = K // 256
        q80_bytes_per_superblock = 272  # 256 weights + 16 scales
        total_superblocks = N * nbpr
        print(f"    weight_buf={w_bytes / 1e6:.3f} MB  "
              f"({total_superblocks} super-blocks x {q80_bytes_per_superblock} bytes)")
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
    ap.add_argument("--no-bench", action="store_true", help="skip timing, correctness only")
    ap.add_argument("--autotune-ksplit", action="store_true",
                    help="empirically pick KSPLIT per shape on this device")
    ap.add_argument("--autotune-iters", type=int, default=15)
    ap.add_argument("--autotune-warmup", type=int, default=4)
    args = ap.parse_args()

    hw = HW(peak_bw_gbps=args.peak_bw, fp32_gflops=args.peak_fp32)
    gpu = Gpu(args.device)
    if not args.no_bench:
        gpu.setup_flush(args.flush_mib)

    SEP = "=" * 100
    print(SEP)
    print("  Q8_0 GEMV SG (sub-group block-read) kernel — correctness + timing + roofline")
    if not args.no_bench:
        print(f"  Peak FP32 : {hw.fp32_gflops:.0f} GFLOPS")
        print(f"  Peak BW   : {hw.peak_bw_gbps:.1f} GB/s   "
              f"(ridge point {hw.fp32_gflops / hw.peak_bw_gbps:.2f} FLOP/B)")
        print(f"  L3 flush  : {args.flush_mib} MiB buffer, 2 sweeps per iteration")
    print(SEP)

    common_kw = dict(
        verbose=args.verbose,
        iters=args.iters,
        warmup=args.warmup,
        do_bench=not args.no_bench,
        autotune=args.autotune_ksplit,
        autotune_iters=args.autotune_iters,
        autotune_warmup=args.autotune_warmup,
        BM=1,
    )

    results = []

    def run(name, **kw):
        try:
            ok, timing, rl, moved = test_q80_gemv_sg(gpu, hw, **kw, **common_kw)
            results.append((name, ok, timing, rl, moved))
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {name}: {exc}")
            traceback.print_exc()
            results.append((name, False, None, None, None))

    # ---- Test cases ----
    run("fc_gguf_q8_0_sg [K=2048, N=512]",  K=2048, N=512,  seed=101)
    run("fc_gguf_q8_0_sg [K=2048, N=4096]", K=2048, N=4096, seed=102)
    run("fc_gguf_q8_0_sg [K=2048, N=8192]", K=2048, N=8192, seed=103)

    # ---- Summary table ----
    W = 122
    print("\n" + "=" * W)
    print(f"  {'Kernel':<42} {'Corr':>5}  {'mean ms':>9}  {'BW GB/s':>9}  "
          f"{'BW%':>6}  {'GFLOPS':>8}  {'Roof%':>7}  {'Bound':>7}  {'moved MB':>9}")
    print("-" * W)
    for name, ok, timing, rl, moved in results:
        mean  = f"{timing['mean_ms']:.3f}" if timing else "-"
        bw    = f"{rl['achieved_bw_gbps']:.1f}" if rl else "-"
        bwp   = f"{rl['bw_util_pct']:.1f}%" if rl else "-"
        gf    = f"{rl['achieved_gflops']:.1f}" if rl else "-"
        rp    = f"{rl['roofline_pct']:.1f}%" if rl else "-"
        bd    = rl['bound_by'] if rl else "-"
        mm    = f"{moved / 1e6:.1f}" if moved else "-"
        print(f"  {name:<42} {'OK' if ok else 'FAIL':>5}  {mean:>9}  {bw:>9}  "
              f"{bwp:>6}  {gf:>8}  {rp:>7}  {bd:>7}  {mm:>9}")
    print("=" * W)

    n_ok = sum(1 for _, ok, *_ in results if ok)
    print(f"\n  {n_ok}/{len(results)} kernels passed correctness.\n")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
