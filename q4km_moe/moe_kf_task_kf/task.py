# [USER_INSTRUCTIONS_START]
# Optimization target: group_up_gate_q4k_sg and down_merge_q4k_sg in
# kernel_moe_q4k_sg.cl
#
# These are the two GEMV kernels that implement one MoE-FFN block for
# GGUF Q4_K quantized expert weights, using the bandwidth-optimal "SG"
# (sub-group block-read, transposed weight layout) scheme:
#   group_up_gate_q4k_sg : fused up-proj + gate-proj GEMV + SiLU(gate)*up
#   down_merge_q4k_sg    : down-proj GEMV, weighted-sum merged over experts
#
# MoE-FFN shapes (top-8-of-256 routing, token_len=1):
#   GroupUpGate_K2048_N512  : gate_up, K=2048, output_len=512
#   DownMerge_K512_N2048    : down,    K=512,  output_len=2048
#
# REQUIRED constraints (host dispatch in task.py depends on them):
#   1. Entry-point names and argument order must remain EXACTLY:
#        group_up_gate_q4k_sg(inputs, ups, gates, indexes, outputs,
#                              output_num, aligned_output_num, token_len,
#                              input_len, output_len)
#        down_merge_q4k_sg(inputs, downs, indexes, efficients, outputs,
#                           input_num, aligned_input_num, token_len,
#                           input_len, output_len, is_acc)
#   2. OPG (sub_group_size) must remain 16.
#   3. The kernel is built with -DKSPLIT=<n> for varying n (1,2,4,8,...);
#      it must produce correct results for any such n.
#   4. Dispatch is fixed by task.py:
#        group_up_gate_q4k_sg: global=(output_len,KSPLIT,output_num*token_len)
#                              local =(16,KSPLIT,1)
#        down_merge_q4k_sg:    global=(output_len,KSPLIT,token_len)
#                              local =(16,KSPLIT,1)
#
# Optimization ideas to explore:
#   - Wide sub-group block reads for weight bytes and the packed scale/min
#     record (avoid many narrow global loads).
#   - Sub-group shuffles to recover per-lane scalar fields from one wide read.
#   - Fold the Q4_K activation-sum term into the same wide activation load
#     used for the dot product (sub_group_reduce_add), avoiding a second pass.
#   - Split up/gate reduction work across sub-groups to halve live weight
#     registers and avoid spilling.
#   - Reduce SLM traffic / bank conflicts in the staged-activation buffer.
#   - Hoist routing-table (indexes/efficients) loads out of the per-iteration
#     dependency chain with one wide read + shuffles.
#   - Tune the KSPLIT-dependent code paths (e.g. contiguous vs round-robin
#     down_merge work distribution) without breaking any KSPLIT value.
# [USER_INSTRUCTIONS_END]

import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np
import pytest
import pyopencl as cl

from kernelfoundry import TestBase  # Tests must derive from TestBase

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TASK_DIR       = Path(__file__).parent.resolve()
KERNEL_FILE    = str(TASK_DIR / "kernel_moe_q4k_sg.cl")
REFERENCE_FILE = str(TASK_DIR / "reference_moe_q4k_sg.cl")

OPG = 16  # sub_group_size = output rows per sub-group (fixed by the kernel contract)

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
SEED = 1234
BENCH = dict(flush_mib=256, iters=300, warmup=30, flush_l2=True)

# ---------------------------------------------------------------------------
# Expert-pool disk cache (OFF by default).
#
# NOTE: a full expert pool for these MoE-FFN shapes (256 experts, K=2048/512,
# N=512/2048) stores not just the packed weights but also a full fp32
# dequantized reference tensor (ref_W) for every expert -- on the order of
# 1 GB *per pool*, and each test case builds 2-3 pools (ups/gates or downs).
# Caching that to disk by default silently fills the workspace with several
# GB of temporary .npz files across a handful of pytest invocations (this bit
# test_moe_gemv_sg_kernels.py, whose standalone --no-cache flag defaults to
# "cache enabled" and must be passed explicitly to avoid it).
#
# To avoid that trap here, the on-disk cache defaults to DISABLED: every
# pytest run rebuilds the pool from the RNG seed in memory (still fast enough
# for these shapes) and nothing large is left behind on disk. Set
# MOE_KF_EXPERT_POOL_CACHE=1 in the environment to opt back into disk caching
# (e.g. for fast repeated local iteration), and MOE_KF_EXPERT_POOL_CACHE_DIR
# to redirect where it is written.
# ---------------------------------------------------------------------------
_CACHE_VERSION = "kfv1"
_CACHE_ENABLED = os.environ.get("MOE_KF_EXPERT_POOL_CACHE", "0") in ("1", "true", "True")
_CACHE_ENABLED = False
_CACHE_DIR = Path(os.environ.get(
    "MOE_KF_EXPERT_POOL_CACHE_DIR", str(TASK_DIR / ".expert_pool_cache")))


def _cache_path(num_experts, K, N, seed):
    key = f"q4ksg_{_CACHE_VERSION}_ne{num_experts}_K{K}_N{N}_seed{seed}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return _CACHE_DIR / f"{key}_{digest}.npz"


def _load_cached(num_experts, K, N, seed):
    if not _CACHE_ENABLED:
        return None
    path = _cache_path(num_experts, K, N, seed)
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            flat = data["flat"]
            ref_W = data["ref_W"]
            expert_size = int(data["expert_size"])
        return flat, ref_W, expert_size
    except Exception as exc:
        logging.warning("expert-pool cache load failed (%s); rebuilding", exc)
        return None


def _save_cached(num_experts, K, N, seed, flat, ref_W, expert_size):
    if not _CACHE_ENABLED:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(num_experts, K, N, seed)
        tmp = str(path) + ".tmp.npz"
        np.savez(tmp, flat=flat, ref_W=ref_W, expert_size=np.int64(expert_size))
        os.replace(tmp, path)
    except Exception as exc:
        logging.warning("expert-pool cache save failed (%s)", exc)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    {"kind": "group_up_gate", "K": 2048, "output_len": 512,
     "num_experts": 256, "active": 8, "token_len": 1,
     "label": "GroupUpGate_K2048_N512"},
    {"kind": "down_merge", "K": 512, "output_len": 2048,
     "num_experts": 256, "active": 8, "token_len": 1,
     "label": "DownMerge_K512_N2048"},
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
# Random GGUF Q4_K block generator (structurally valid, random-valued)
# ---------------------------------------------------------------------------
def _gen_q4k_blocks(n, seed=SEED):
    rng = np.random.default_rng(seed)
    d  = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc = rng.integers(0, 64,  (n, 12),  dtype=np.uint8)
    qs = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qs], axis=1).ravel()  # [n * 144]


# ---------------------------------------------------------------------------
# Shuffle: standard GGUF layout -> row-major separated-section layout
# (vectorized over all NB = N*K//256 blocks)
# ---------------------------------------------------------------------------
def _shuffle_q4k(raw, K, N):
    NB  = N * K // 256
    blk = raw.reshape(NB, 144)
    d_bytes    = blk[:, 0:2]
    dmin_bytes = blk[:, 2:4]
    sc = blk[:, 4:16]
    qs = blk[:, 16:144].reshape(NB, 4, 32)

    low  = qs & 0x0F
    high = (qs >> 4) & 0x0F
    rw = np.empty((NB, 4, 64), dtype=np.uint8)
    rw[:, :, 0:32]  = low
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

    rw8  = rw.reshape(NB, 8, 32)
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
    opsl[:, 8]  = ((rs[:, 0] & 0x30) >> 4) | ((rs[:, 1] & 0x30) >> 2) | (rs[:, 2] & 0x30) | ((rs[:, 3] & 0x30) << 2)
    opsl[:, 9]  = ((rs[:, 4] & 0x30) >> 4) | ((rs[:, 5] & 0x30) >> 2) | (rs[:, 6] & 0x30) | ((rs[:, 7] & 0x30) << 2)
    opsl[:, 10] = ((rm[:, 0] & 0x30) >> 4) | ((rm[:, 1] & 0x30) >> 2) | (rm[:, 2] & 0x30) | ((rm[:, 3] & 0x30) << 2)
    opsl[:, 11] = ((rm[:, 4] & 0x30) >> 4) | ((rm[:, 5] & 0x30) >> 2) | (rm[:, 6] & 0x30) | ((rm[:, 7] & 0x30) << 2)
    opsl[:, 12:14] = d_bytes
    opsl[:, 14:16] = dmin_bytes

    return opqs.reshape(-1), opsl.reshape(-1)


def _deq_q4k(pqs, psl, K, N):
    """Decode row-major shuffled Q4_K -> W[N, K] float32 (vectorized)."""
    nprow = K // 256
    NB    = N * nprow
    qs_b  = pqs.reshape(NB, 8, 16)
    sl_b  = psl.reshape(NB, 16)

    sli = (sl_b[:, 0].astype(np.uint32) | (sl_b[:, 1].astype(np.uint32) << 8) |
           (sl_b[:, 2].astype(np.uint32) << 16) | (sl_b[:, 3].astype(np.uint32) << 24))
    mli = (sl_b[:, 4].astype(np.uint32) | (sl_b[:, 5].astype(np.uint32) << 8) |
           (sl_b[:, 6].astype(np.uint32) << 16) | (sl_b[:, 7].astype(np.uint32) << 24))
    shi = sl_b[:, 8].astype(np.uint32) | (sl_b[:, 9].astype(np.uint32) << 8)
    mhi = sl_b[:, 10].astype(np.uint32) | (sl_b[:, 11].astype(np.uint32) << 8)
    d    = sl_b[:, 12:14].copy().view(np.float16).reshape(NB).astype(np.float32)
    dmin = sl_b[:, 14:16].copy().view(np.float16).reshape(NB).astype(np.float32)

    j_arr = np.arange(8, dtype=np.uint32)
    sq = ((sli[:, None] >> (j_arr * 4)) & 0xF) | (((shi[:, None] >> (j_arr * 2)) & 3) << 4)
    mq = ((mli[:, None] >> (j_arr * 4)) & 0xF) | (((mhi[:, None] >> (j_arr * 2)) & 3) << 4)
    sc = sq.astype(np.float32) * d[:, None]
    mv = mq.astype(np.float32) * dmin[:, None]

    lo = (qs_b & 0x0F).astype(np.float32)
    hi = ((qs_b >> 4) & 0x0F).astype(np.float32)

    Wblock = np.empty((NB, 8, 32), dtype=np.float32)
    Wblock[:, :, 0:16]  = lo * sc[:, :, None] - mv[:, :, None]
    Wblock[:, :, 16:32] = hi * sc[:, :, None] - mv[:, :, None]
    return Wblock.reshape(N, nprow, 256).reshape(N, K)


# ---------------------------------------------------------------------------
# SG transposed packing (vectorized): row-major shuffled section -> SG
# column-major section.  Block index blk = row*nbpr + bid, row = h*OPG + lid.
# ---------------------------------------------------------------------------
def _ascon(a):
    return np.ascontiguousarray(a)


def _pack_pqs_sg(pqs, K, N, opg=OPG):
    """128-byte low-quant section (Q4_K pqs)."""
    nbpr = K // 256
    nrg  = N // opg
    src = pqs.reshape(nrg, opg, nbpr, 8, 4, 4)   # [h,lid,bid,j,c,b]
    dst = src.transpose(0, 2, 3, 4, 1, 5)        # [h,bid,j,c,lid,b]
    return _ascon(dst).reshape(-1).astype(np.uint8)


def _pack_psl_sg(psl, K, N, opg=OPG):
    """16-byte scale/min record -> SoA psl_T."""
    nbpr = K // 256
    nrg  = N // opg
    p = _ascon(psl.reshape(nrg, opg, nbpr, 16).transpose(0, 2, 1, 3))  # [h,bid,lid,16]
    sl = _ascon(p[..., 0:4]).reshape(nrg, nbpr, opg * 4)
    ml = _ascon(p[..., 4:8]).reshape(nrg, nbpr, opg * 4)
    sh = _ascon(p[..., 8:10]).reshape(nrg, nbpr, opg * 2)
    mh = _ascon(p[..., 10:12]).reshape(nrg, nbpr, opg * 2)
    d  = _ascon(p[..., 12:14]).reshape(nrg, nbpr, opg * 2)
    dm = _ascon(p[..., 14:16]).reshape(nrg, nbpr, opg * 2)
    return np.concatenate([sl, ml, sh, mh, d, dm], axis=2).reshape(-1).astype(np.uint8)


def _build_q4k_sg_pool_impl(num_experts, K, N, seed=SEED):
    nprow  = K // 256
    sz_pqs = nprow * N * 128
    sz_psl = nprow * N * 16
    expert_size = nprow * N * 144
    flat  = np.zeros(num_experts * expert_size, dtype=np.uint8)
    ref_W = np.zeros((num_experts, N, K), dtype=np.float32)
    for e in range(num_experts):
        raw = _gen_q4k_blocks(nprow * N, seed=seed + e * 97)
        pqs, psl = _shuffle_q4k(raw, K, N)
        ref_W[e] = _deq_q4k(pqs, psl, K, N)
        pqs_T = _pack_pqs_sg(pqs, K, N)
        psl_T = _pack_psl_sg(psl, K, N)
        off = e * expert_size
        flat[off: off + sz_pqs] = pqs_T
        flat[off + sz_pqs: off + sz_pqs + sz_psl] = psl_T
    return flat, ref_W, expert_size


def _build_q4k_sg_pool(num_experts, K, N, seed=SEED):
    cached = _load_cached(num_experts, K, N, seed)
    if cached is not None:
        return cached
    flat, ref_W, expert_size = _build_q4k_sg_pool_impl(num_experts, K, N, seed=seed)
    _save_cached(num_experts, K, N, seed, flat, ref_W, expert_size)
    return flat, ref_W, expert_size


def _silu(x):
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    sig = np.empty_like(x)
    sig[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    sig[~pos] = ex / (1.0 + ex)
    return (x * sig).astype(np.float32)


def _make_routing_indexes(rng, token_len, num_experts, active, aligned_num):
    indexes = np.zeros((token_len, aligned_num), dtype=np.uint32)
    for t in range(token_len):
        indexes[t, :active] = rng.choice(num_experts, size=active, replace=False)
        if aligned_num > active:
            indexes[t, active:] = rng.integers(0, num_experts, aligned_num - active)
    return indexes


def _choose_ksplit(base_groups, split_space, target=2048, maxk=8):
    def _pow2_floor(x):
        p = 1
        while p * 2 <= x:
            p *= 2
        return p
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
def _make_group_up_gate_state(ctx, case, seed=SEED, need_ref=True):
    K, N          = case["K"], case["output_len"]
    num_experts   = case["num_experts"]
    active        = case["active"]
    token_len     = case["token_len"]

    ups_flat, ups_ref, expert_size     = _build_q4k_sg_pool(num_experts, K, N, seed=seed)
    gates_flat, gates_ref, _           = _build_q4k_sg_pool(num_experts, K, N, seed=seed + 999)

    output_num = active
    aligned_output_num = (output_num + 15) // 16 * 16
    rng = np.random.default_rng(seed ^ 0xBEEF_1234)
    indexes = _make_routing_indexes(rng, token_len, num_experts, active, aligned_output_num)
    x = rng.standard_normal((token_len, K)).astype(np.float16)

    y_ref = None
    if need_ref:
        x_f32 = x.astype(np.float32)
        y_ref = np.zeros((token_len, output_num, N), dtype=np.float32)
        for t in range(token_len):
            for v in range(output_num):
                idx  = int(indexes[t, v])
                up   = ups_ref[idx] @ x_f32[t]
                gate = gates_ref[idx] @ x_f32[t]
                y_ref[t, v] = _silu(gate) * up

    base_groups = (N // OPG) * output_num * token_len
    ksplit = _choose_ksplit(base_groups, K // 256)

    x_b     = _cbuf(ctx, x)
    ups_b   = _cbuf(ctx, ups_flat)
    gates_b = _cbuf(ctx, gates_flat)
    idx_b   = _cbuf(ctx, indexes.ravel())
    out_b   = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY,
                        size=token_len * output_num * N * 4)

    gsize = (N, ksplit, output_num * token_len)
    lsize = (OPG, ksplit, 1)

    return dict(K=K, N=N, output_num=output_num, aligned_output_num=aligned_output_num,
                token_len=token_len, ksplit=ksplit, gsize=gsize, lsize=lsize,
                x_b=x_b, ups_b=ups_b, gates_b=gates_b, idx_b=idx_b, out_b=out_b,
                y_ref=y_ref, expert_size=expert_size)


def _run_group_up_gate(queue, prog, st):
    krn = cl.Kernel(prog, "group_up_gate_q4k_sg")
    krn.set_args(st["x_b"], st["ups_b"], st["gates_b"], st["idx_b"], st["out_b"],
                 np.uint32(st["output_num"]), np.uint32(st["aligned_output_num"]),
                 np.uint32(st["token_len"]), np.uint32(st["K"]), np.uint32(st["N"]))

    def enq():
        return cl.enqueue_nd_range_kernel(queue, krn, st["gsize"], st["lsize"])

    ev = enq()
    ev.wait()
    got = np.empty(st["token_len"] * st["output_num"] * st["N"], dtype=np.float32)
    cl.enqueue_copy(queue, got, st["out_b"])
    queue.finish()
    return got.reshape(st["token_len"], st["output_num"], st["N"]), enq


def _make_down_merge_state(ctx, case, seed=SEED, need_ref=True):
    K, N          = case["K"], case["output_len"]
    num_experts   = case["num_experts"]
    active        = case["active"]
    token_len     = case["token_len"]

    downs_flat, downs_ref, expert_size = _build_q4k_sg_pool(num_experts, K, N, seed=seed)

    input_num = active
    aligned_input_num = (input_num + 15) // 16 * 16
    rng = np.random.default_rng(seed ^ 0xBEEF_1234)
    indexes     = _make_routing_indexes(rng, token_len, num_experts, active, aligned_input_num)
    efficients  = rng.uniform(0.1, 1.0, (token_len, aligned_input_num)).astype(np.float32)
    inputs      = rng.standard_normal((token_len, input_num, K)).astype(np.float16)

    y_ref = None
    if need_ref:
        inputs_f32 = inputs.astype(np.float32)
        y_ref = np.zeros((token_len, N), dtype=np.float32)
        for t in range(token_len):
            acc = np.zeros(N, dtype=np.float64)
            for e in range(input_num):
                idx = int(indexes[t, e])
                eff = float(efficients[t, e])
                acc += eff * (downs_ref[idx] @ inputs_f32[t, e])
            y_ref[t] = acc.astype(np.float32)

    base_groups = (N // OPG) * token_len
    ksplit = _choose_ksplit(base_groups, input_num * (K // 256))

    in_b     = _cbuf(ctx, inputs)
    downs_b  = _cbuf(ctx, downs_flat)
    idx_b    = _cbuf(ctx, indexes.ravel())
    eff_b    = _cbuf(ctx, efficients.ravel())
    out_b    = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=token_len * N * 4)

    gsize = (N, ksplit, token_len)
    lsize = (OPG, ksplit, 1)

    return dict(K=K, N=N, input_num=input_num, aligned_input_num=aligned_input_num,
                token_len=token_len, ksplit=ksplit, gsize=gsize, lsize=lsize,
                in_b=in_b, downs_b=downs_b, idx_b=idx_b, eff_b=eff_b, out_b=out_b,
                y_ref=y_ref, expert_size=expert_size)


def _run_down_merge(queue, prog, st):
    krn = cl.Kernel(prog, "down_merge_q4k_sg")
    krn.set_args(st["in_b"], st["downs_b"], st["idx_b"], st["eff_b"], st["out_b"],
                 np.uint32(st["input_num"]), np.uint32(st["aligned_input_num"]),
                 np.uint32(st["token_len"]), np.uint32(st["K"]), np.uint32(st["N"]),
                 np.int32(0))

    def enq():
        return cl.enqueue_nd_range_kernel(queue, krn, st["gsize"], st["lsize"])

    ev = enq()
    ev.wait()
    got = np.empty(st["token_len"] * st["N"], dtype=np.float32)
    cl.enqueue_copy(queue, got, st["out_b"])
    queue.finish()
    return got.reshape(st["token_len"], st["N"]), enq


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
# Tests — parametrized over both MoE-FFN GEMV kernels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", TEST_CASES, ids=CASE_IDS)
class TestMoeGemvQ4KSG(TestBase):
    """Q4_K MoE 'SG' (sub-group block-read) GEMV kernels, gate_up + down_merge."""

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
        cl.Kernel(prog, "group_up_gate_q4k_sg")  # raises if missing
        cl.Kernel(prog, "down_merge_q4k_sg")     # raises if missing

    def test_correctness_vs_numpy(self, case, ocl_queue):
        """GPU output must match pure-Python dequantize+GEMV(+SiLU) reference."""
        ctx = ocl_queue.context
        if case["kind"] == "group_up_gate":
            st   = _make_group_up_gate_state(ctx, case, need_ref=True)
            prog = _build_prog(ctx, KERNEL_FILE, st["ksplit"])
            got, _ = _run_group_up_gate(ocl_queue, prog, st)
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
        if case["kind"] == "group_up_gate":
            st        = _make_group_up_gate_state(ctx, case, need_ref=False)
            kern_prog = _build_prog(ctx, KERNEL_FILE, st["ksplit"])
            ref_prog  = _build_prog(ctx, REFERENCE_FILE, st["ksplit"])
            y_kern, _ = _run_group_up_gate(ocl_queue, kern_prog, st)
            y_ref,  _ = _run_group_up_gate(ocl_queue, ref_prog,  st)
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

        if case["kind"] == "group_up_gate":
            st   = _make_group_up_gate_state(ctx, case, need_ref=False)
            prog = _build_prog(ctx, cl_file, st["ksplit"])
            _, enq = _run_group_up_gate(ocl_queue, prog, st)
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
    task = TestMoeGemvQ4KSG()
    task.build(TestBase.get_machine_gpu_arch())
