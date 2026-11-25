#!/usr/bin/env python
"""Standalone bench harness for matmul_opt.cl. Usage: python bench.py [warmup] [trials]"""
import sys, time, math, re
import numpy as np
import pyopencl as cl
from pathlib import Path

M, K, N = 2048, 2560, 2048

HERE = Path(__file__).parent

def build_kernel(src_text, ctx):
    return cl.Program(ctx, src_text).build()

def parse_define(src, name):
    m = re.search(rf'#define\s+{name}\s+(\d+)', src)
    return int(m.group(1)) if m else None

def parse_lws(src):
    m = re.search(r'reqd_work_group_size\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', src)
    if m: return int(m.group(1))*int(m.group(2))*int(m.group(3))
    return None

def make_runner(filename):
    src = (HERE/filename).read_text()
    src_active = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src_active = re.sub(r"//.*", "", src_active)

    plat = cl.get_platforms()[0]
    dev = plat.get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prog = build_kernel(src, ctx)
    knl = prog.matmul

    rng = np.random.default_rng(0)
    a = rng.standard_normal((M,K), dtype=np.float32).astype(np.float16)
    b = rng.standard_normal((K,N), dtype=np.float32).astype(np.float16)

    a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=a.ravel())
    b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=b.ravel())
    c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=M*N*2)

    tile_m = parse_define(src_active, "TILE_M")
    tile_n = parse_define(src_active, "TILE_N")
    wg_m = parse_define(src_active, "WG_M")
    wg_n = parse_define(src_active, "WG_N")
    sg = parse_define(src_active, "SG_SIZE")

    if all(v is not None for v in [tile_m,tile_n,wg_m,wg_n,sg]):
        wg_tile_m = wg_m*tile_m
        wg_tile_n = wg_n*tile_n
        wg_size = sg*wg_m*wg_n
        grid_m = math.ceil(M/wg_tile_m)
        grid_n = math.ceil(N/wg_tile_n)
        gws = (grid_m*wg_size, grid_n)
        lws = (wg_size, 1)
    else:
        # Naive fallback
        gws = (N, M)
        lws = None

    def run():
        ev = knl(q, gws, lws, a_buf, b_buf, c_buf, np.int32(M), np.int32(K), np.int32(N))
        ev.wait()
        return (ev.profile.end - ev.profile.start) * 1e-6  # ms

    def verify():
        run()
        out = np.empty(M*N, dtype=np.float16)
        cl.enqueue_copy(q, out, c_buf)
        out = out.reshape(M,N).astype(np.float32)
        ref = (a.astype(np.float32) @ b.astype(np.float32))
        ok = np.allclose(out, ref, rtol=2e-2, atol=2e-2)
        max_abs = np.max(np.abs(out-ref))
        max_rel = np.max(np.abs((out-ref)/(np.abs(ref)+1e-3)))
        return ok, max_abs, max_rel

    return run, verify, q, gws, lws

def bench(filename="matmul_opt.cl", warmup=5, trials=20):
    run, verify, q, gws, lws = make_runner(filename)
    print(f"GWS={gws}  LWS={lws}")
    ok, ma, mr = verify()
    print(f"correctness: {'OK' if ok else 'FAIL'}  max_abs={ma:.4f}  max_rel={mr:.4f}")
    if not ok:
        return None
    for _ in range(warmup): run()
    times = [run() for _ in range(trials)]
    times.sort()
    PEAK = 116e12  # B580 fp16 XMX TFLOPS
    flops = 2*M*K*N
    min_ms = times[0]
    avg_ms = sum(times)/len(times)
    util = (flops/(min_ms*1e-3)) / PEAK * 100
    print(f"min={min_ms:.4f} ms  avg={avg_ms:.4f} ms  util={util:.1f}%  ({flops/(min_ms*1e-3)/1e12:.1f} TFLOPS)")
    return min_ms

if __name__ == "__main__":
    fn = sys.argv[1] if len(sys.argv)>1 else "matmul_opt.cl"
    w = int(sys.argv[2]) if len(sys.argv)>2 else 5
    t = int(sys.argv[3]) if len(sys.argv)>3 else 20
    bench(fn, w, t)
