"""In-process KSPLIT / variant tuner: identical conditions for every config."""
import os, sys, statistics, numpy as np, pyopencl as cl
import test_moe_gemv_sg_kernels as T

gpu = T.Gpu(0)
gpu.setup_flush(128)
mf = cl.mem_flags
OPG = T.OPG
KSPLITS = [int(x) for x in os.environ.get("KS", "2,4,8,16").split(",")]
QUANTS = os.environ.get("QUANTS", "q4k,q5k,q6k").split(",")
ITERS = int(os.environ.get("ITERS", "300"))


def bench(krn, gs, ls):
    enq = lambda q: cl.enqueue_nd_range_kernel(q, krn, gs, ls)
    ts = gpu.time_kernel(enq, iters=ITERS, warmup=30, do_flush=True)
    return statistics.median(ts) * 1e3, min(ts) * 1e3


def run_gu(quant, K=2048, N=512, ne=256, active=8):
    ups, _, esz = T._BUILD_POOL[quant](ne, K, N, seed=101)
    gates, _, _ = T._BUILD_POOL[quant](ne, K, N, seed=1100)
    rng = np.random.default_rng(1)
    idxs = T.make_routing_indexes(rng, 1, ne, active, 16)
    x = rng.standard_normal((1, K)).astype(np.float16)
    xb, ub, gb = T._cbuf(gpu.ctx, x), T._cbuf(gpu.ctx, ups), T._cbuf(gpu.ctx, gates)
    ib = T._cbuf(gpu.ctx, idxs.ravel())
    ob = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=active * N * 4)
    out = []
    for ks in KSPLITS:
        prog = gpu.build(T.cl_src(T._CL_SRC[quant]), f"-cl-std=CL2.0 -DKSPLIT={ks} " + os.environ.get("EXTRA",""))
        k = cl.Kernel(prog, f"group_up_gate_{quant}_sg")
        k.set_args(xb, ub, gb, ib, ob, np.uint32(active), np.uint32(16),
                   np.uint32(1), np.uint32(K), np.uint32(N))
        out.append((ks,) + bench(k, (N, ks, active), (OPG, ks, 1)))
    return out, active * 2 * esz


def run_dm(quant, K=512, N=2048, ne=256, active=8):
    downs, _, esz = T._BUILD_POOL[quant](ne, K, N, seed=104)
    rng = np.random.default_rng(2)
    idxs = T.make_routing_indexes(rng, 1, ne, active, 16)
    eff = rng.uniform(0.1, 1.0, (1, 16)).astype(np.float32)
    inp = rng.standard_normal((1, active, K)).astype(np.float16)
    inb, db = T._cbuf(gpu.ctx, inp), T._cbuf(gpu.ctx, downs)
    ib, eb = T._cbuf(gpu.ctx, idxs.ravel()), T._cbuf(gpu.ctx, eff.ravel())
    ob = cl.Buffer(gpu.ctx, mf.READ_WRITE, size=N * 4)
    out = []
    for ks in KSPLITS:
        prog = gpu.build(T.cl_src(T._CL_SRC[quant]), f"-cl-std=CL2.0 -DKSPLIT={ks} " + os.environ.get("EXTRA",""))
        k = cl.Kernel(prog, f"down_merge_{quant}_sg")
        k.set_args(inb, db, ib, eb, ob, np.uint32(active), np.uint32(16),
                   np.uint32(1), np.uint32(K), np.uint32(N), np.int32(0))
        out.append((ks,) + bench(k, (N, ks, 1), (OPG, ks, 1)))
    return out, active * esz


for quant in QUANTS:
    for name, fn in (("gate_up", run_gu), ("down_merge", run_dm)):
        res, wbytes = fn(quant)
        best = min(r[1] for r in res)
        print(f"{name:11s} {quant}: " + "  ".join(
            f"K{ks}={med:.4f}{'*' if med == best else ' '}({wbytes/med/1e6:.0f}GB/s)"
            for ks, med, mn in res))
