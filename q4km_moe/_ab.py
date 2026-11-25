"""Same-process A/B of baseline vs optimized kernels (alternating, so GPU state
and clocks affect both equally)."""
import os, statistics, numpy as np, pyopencl as cl
import test_moe_gemv_sg_kernels as T

gpu = T.Gpu(0); gpu.setup_flush(128)
mf = cl.mem_flags; OPG = T.OPG
KSPLITS = [int(x) for x in os.environ.get("KS", "4,8").split(",")]
ITERS = int(os.environ.get("ITERS", "300"))
REPS = int(os.environ.get("REPS", "3"))


def timeit(krn, gs, ls):
    enq = lambda q: cl.enqueue_nd_range_kernel(q, krn, gs, ls)
    return statistics.median(gpu.time_kernel(enq, iters=ITERS, warmup=20, do_flush=True)) * 1e3


def setup(quant, kind):
    rng = np.random.default_rng(1)
    if kind == "gate_up":
        K, N, ne, active = 2048, 512, 256, 8
        ups, _, esz = T._BUILD_POOL[quant](ne, K, N, seed=101)
        gts, _, _ = T._BUILD_POOL[quant](ne, K, N, seed=1100)
        idxs = T.make_routing_indexes(rng, 1, ne, active, 16)
        x = rng.standard_normal((1, K)).astype(np.float16)
        args = [T._cbuf(gpu.ctx, x), T._cbuf(gpu.ctx, ups), T._cbuf(gpu.ctx, gts),
                T._cbuf(gpu.ctx, idxs.ravel()),
                cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=active * N * 4),
                np.uint32(active), np.uint32(16), np.uint32(1), np.uint32(K), np.uint32(N)]
        return args, (N, active), active * 2 * esz
    K, N, ne, active = 512, 2048, 256, 8
    downs, _, esz = T._BUILD_POOL[quant](ne, K, N, seed=104)
    idxs = T.make_routing_indexes(rng, 1, ne, active, 16)
    eff = rng.uniform(0.1, 1.0, (1, 16)).astype(np.float32)
    inp = rng.standard_normal((1, active, K)).astype(np.float16)
    args = [T._cbuf(gpu.ctx, inp), T._cbuf(gpu.ctx, downs), T._cbuf(gpu.ctx, idxs.ravel()),
            T._cbuf(gpu.ctx, eff.ravel()), cl.Buffer(gpu.ctx, mf.READ_WRITE, size=N * 4),
            np.uint32(active), np.uint32(16), np.uint32(1), np.uint32(K), np.uint32(N),
            np.int32(0)]
    return args, (N, 1), active * esz


for quant in ("q4k", "q5k", "q6k"):
    for kind, kname in (("gate_up", "group_up_gate_%s_sg"), ("down_merge", "down_merge_%s_sg")):
        args, (N, z), wb = setup(quant, kind)
        for ks in KSPLITS:
            res = {}
            for rep in range(REPS):
                for tag, src in (("base", T.cl_src("moe_gemv_%s_sg.cl.baseline" % quant)),
                                 ("opt ", T.cl_src("moe_gemv_%s_sg.cl" % quant))):
                    prog = gpu.build(src, "-cl-std=CL2.0 -DKSPLIT=%d" % ks)
                    k = cl.Kernel(prog, kname % quant)
                    k.set_args(*args)
                    t = timeit(k, (N, ks, z), (OPG, ks, 1))
                    res.setdefault(tag, []).append(t)
            b, o = min(res["base"]), min(res["opt "])
            print("%-10s %s K%-2d : base %.4f ms (%3.0f GB/s)  opt %.4f ms (%3.0f GB/s)  "
                  "speedup %.2fx" % (kind, quant, ks, b, wb/b/1e6, o, wb/o/1e6, b/o))
