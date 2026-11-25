# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Standalone micro-benchmark + correctness harness for the extracted OpenVINO MoE 3-GEMM SwiGLU
kernel (moe_3gemm_swiglu_mlp.cl), GGUF Q4_K native-decode + MOE_GGUF_DP4A path used by
Qwen3.6-35B-A3B decode. Two stages are profiled independently: `gate_up` and `down`.

The kernel is the VERBATIM OV .cl, compiled with the OV-faithful JIT preamble + JIT constants
(moe_jit.py) and dispatched with OV's exact batched-GEMV geometry (from moe/moe_3gemm_swiglu_opt.cpp
execute path): one workgroup per (token, expert-slot):
    gate_up: global=(token*topk, SUBGROUP_SIZE, INTERMEDIATE/N_BLOCK), local=(1, SUBGROUP_SIZE, SUBGROUP_NUM)
    down:    global=(token*topk, SUBGROUP_SIZE, HIDDEN/N_BLOCK),       local=(1, SUBGROUP_SIZE, SUBGROUP_NUM)
Built with SHARED_EXPERT_ENABLE=0 (routed experts only — the dominant GEMV cost; per-expert decode
is identical to the shared expert). Roofline uses the top_k experts actually read from DRAM.

Usage:
    python3 moe_bench.py --config ../configs/qwen36.json --out ../results/qwen36_report.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import pyopencl as cl

import moe_jit
import moe_ref as mr
from hw import HW

KDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels"))


class Gpu:
    def __init__(self, device_index=0, platform_index=0):
        self.platform = cl.get_platforms()[platform_index]
        self.device = self.platform.get_devices()[device_index]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self._flush_kernel = None

    def build(self, fname, entry_point, consts):
        with open(os.path.join(KDIR, fname)) as f:
            body = f.read()
        src = moe_jit.build_preamble(entry_point, consts) + "\n" + body
        opts = ["-I", KDIR] + moe_jit.OV_BUILD_OPTIONS
        prog = cl.Program(self.ctx, src).build(options=opts)
        return cl.Kernel(prog, entry_point)

    def setup_flush(self, mib):
        self._flush_elems = (mib * 1024 * 1024) // 4
        self._flush_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self._flush_elems * 4)
        self._flush_sink = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=4096 * 4)
        cl.enqueue_copy(self.queue, self._flush_buf, np.zeros(self._flush_elems, np.float32))
        with open(os.path.join(KDIR, "cache_flush.cl")) as f:
            src = f.read()
        prog = cl.Program(self.ctx, src).build(options=moe_jit.OV_BUILD_OPTIONS)
        self._flush_kernel = cl.Kernel(prog, "cache_flush")
        self.queue.finish()

    def flush_l2(self, sweeps=2):
        gsz = (min(self._flush_elems, 1024 * 1024),)
        for _ in range(sweeps):
            self._flush_kernel(self.queue, gsz, None, self._flush_buf,
                               np.uint32(self._flush_elems), self._flush_sink)
        self.queue.finish()

    def time_event(self, enqueue_fn, iters, warmup, do_flush):
        for _ in range(warmup):
            enqueue_fn(self.queue); self.queue.finish()
        times = []
        for _ in range(iters):
            if do_flush:
                self.flush_l2()
            ev = enqueue_fn(self.queue)
            ev.wait()
            times.append((ev.profile.end - ev.profile.start) * 1e-9)
        return times


def stats(t):
    return {"median_ms": statistics.median(t) * 1e3, "min_ms": min(t) * 1e3,
            "mean_ms": statistics.mean(t) * 1e3, "iters": len(t)}


def rel_l2(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a - b))


def cosine(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


SG = moe_jit.SUBGROUP_SIZE
NB = moe_jit.N_BLOCK
SGN = moe_jit.SUBGROUP_NUM


def _rng(seed):
    return np.random.default_rng(seed)


def _dummy(gpu, nbytes=256):
    return cl.Buffer(gpu.ctx, cl.mem_flags.READ_ONLY, size=nbytes)


def run_gate_up(gpu, hw, fmt, hidden, inter, experts, topk, iters, warmup, do_flush,
                seed=1234, verify_slots=2, verify_rows=64):
    be, bb = moe_jit.GGUF_BLK[fmt]
    row_bytes = (hidden // be) * bb                     # bytes per output row (K=hidden)
    expert_bytes = inter * row_bytes                    # per-expert gate (or up) weight bytes
    mf = cl.mem_flags
    rng = _rng(seed)
    # SwiGLU squares the projection magnitude (silu(gate)*up), so keep weights/activation small
    # enough that the f16 output stays well inside HALF_MAX=65504 (timing is data-independent).
    gate_w = mr.gen_expert_weights(fmt, experts, inter, hidden, seed=seed, scale_amp=0.004)
    up_w = mr.gen_expert_weights(fmt, experts, inter, hidden, seed=seed + 7, scale_amp=0.004)
    expert_list = rng.permutation(experts)[:topk].astype(np.int32)             # distinct topk experts
    x = (rng.standard_normal((1, hidden)) * 0.05).astype(np.float16)
    y = np.zeros((topk, inter), np.float16)

    d_el = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=expert_list)
    d_gw = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(gate_w))
    d_uw = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(up_w))
    d_x = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    d_y = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=y.nbytes)
    d_gs, d_gz, d_us, d_uz = _dummy(gpu), _dummy(gpu), _dummy(gpu), _dummy(gpu)

    ep, consts = moe_jit.gate_up_jit(fmt, hidden, inter, experts, topk)
    krn = gpu.build("moe_3gemm_swiglu_mlp.cl", ep, consts)
    gsz = (topk, SG, inter // NB)
    lsz = (1, SG, SGN)

    def enq(q):
        return krn(q, gsz, lsz, d_el, d_gw, d_gs, d_gz, d_uw, d_us, d_uz, d_x, d_y)

    enq(gpu.queue).wait()
    cl.enqueue_copy(gpu.queue, y, d_y); gpu.queue.finish()

    # correctness on a subset (verify_slots experts x verify_rows output rows)
    xdeq = mr.act_for(fmt, x[0])                                 # [hidden]
    V = min(verify_rows, inter)
    errs = []
    for s in range(min(verify_slots, topk)):
        e = int(expert_list[s])
        ge = gate_w[e].reshape(inter, row_bytes)
        ue = up_w[e].reshape(inter, row_bytes)
        ref = np.zeros(V, np.float32)
        for n in range(V):
            gW = mr.deq_row(fmt, ge[n], hidden)
            uW = mr.deq_row(fmt, ue[n], hidden)
            ref[n] = mr.silu(float(xdeq @ gW)) * float(xdeq @ uW)
        errs.append(rel_l2(y[s, :V].astype(np.float32), ref))
    err = max(errs); passed = err < 3e-2

    times = gpu.time_event(enq, iters, warmup, do_flush)
    st = stats(times)
    weight_bytes = topk * 2 * expert_bytes              # gate + up for the topk read experts
    bytes_moved = weight_bytes + hidden * 2 + y.nbytes
    flops = 2 * topk * inter * hidden * 2               # gate + up GEMVs
    rl = hw.roofline(flops, bytes_moved, st["median_ms"] * 1e-3, compute_domain="int8")
    return _rec("moe_gate_up", fmt, dict(H=hidden, I=inter, E=experts, topk=topk),
                st, rl, dict(rel_l2_err=err, passed=passed),
                dict(weight_MiB=weight_bytes / 1048576, read_experts=topk))


def run_down(gpu, hw, fmt, hidden, inter, experts, topk, iters, warmup, do_flush,
             seed=1234, verify_slots=2, verify_rows=64):
    be, bb = moe_jit.GGUF_BLK[fmt]
    row_bytes = (inter // be) * bb                      # bytes per output row (K=inter)
    expert_bytes = hidden * row_bytes
    mf = cl.mem_flags
    rng = _rng(seed)
    # Down experts decode Q5_K/Q6_K via int8-activation dp4a; keep magnitudes modest so the f16
    # output stays inside HALF_MAX (timing is data-independent).
    down_w = mr.gen_expert_weights(fmt, experts, hidden, inter, seed=seed + 13, scale_amp=0.01)
    expert_list = rng.permutation(experts)[:topk].astype(np.int32)
    x = (rng.standard_normal((topk, inter)) * 0.2).astype(np.float16)           # intermediate acts
    routing = (rng.random(topk) * 0.5 + 0.1).astype(np.float16)                 # per-slot router weight
    y = np.zeros((topk, hidden), np.float16)

    d_el = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=expert_list)
    d_dw = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(down_w))
    d_x = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(x))
    d_rw = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=routing)
    d_y = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=y.nbytes)
    d_ds, d_dz = _dummy(gpu), _dummy(gpu)

    ep, consts = moe_jit.down_jit(fmt, hidden, inter, experts, topk)
    krn = gpu.build("moe_3gemm_swiglu_mlp.cl", ep, consts)
    gsz = (topk, SG, hidden // NB)
    lsz = (1, SG, SGN)

    def enq(q):
        return krn(q, gsz, lsz, d_el, d_dw, d_ds, d_dz, d_x, d_rw, d_y)

    enq(gpu.queue).wait()
    cl.enqueue_copy(gpu.queue, y, d_y); gpu.queue.finish()

    V = min(verify_rows, hidden)
    errs = []
    for s in range(min(verify_slots, topk)):
        e = int(expert_list[s])
        de = down_w[e].reshape(hidden, row_bytes)
        xdeq = mr.act_for(fmt, x[s])                             # [inter]
        ref = np.zeros(V, np.float32)
        for n in range(V):
            dW = mr.deq_row(fmt, de[n], inter)
            ref[n] = float(xdeq @ dW) * float(routing[s])
        errs.append(rel_l2(y[s, :V].astype(np.float32), ref))
    err = max(errs); passed = err < 3e-2

    times = gpu.time_event(enq, iters, warmup, do_flush)
    st = stats(times)
    weight_bytes = topk * expert_bytes
    bytes_moved = weight_bytes + x.nbytes + routing.nbytes + y.nbytes
    flops = 2 * topk * hidden * inter
    rl = hw.roofline(flops, bytes_moved, st["median_ms"] * 1e-3, compute_domain="int8")
    return _rec("moe_down", fmt, dict(H=hidden, I=inter, E=experts, topk=topk),
                st, rl, dict(rel_l2_err=err, passed=passed),
                dict(weight_MiB=weight_bytes / 1048576, read_experts=topk))


def _rec(kernel, fmt, params, st, rl, correctness, extra):
    return {"kernel": kernel, "format": fmt, "params": params, "timing": st,
            "roofline": rl, "correctness": correctness, "data": extra}


def print_rec(r):
    p = " ".join("%s=%s" % (k, v) for k, v in r["params"].items())
    ok = "PASS" if r["correctness"].get("passed") else "FAIL"
    rl = r["roofline"]
    print("  [%s] %-13s %-10s | %s" % (ok, r["kernel"], r["format"], p))
    print("        time=%.4f ms  BW=%.1f GB/s (%.1f%% of peak)  roofline=%.1f%%  bound=%s  wt=%.1f MiB"
          % (r["timing"]["median_ms"], rl["achieved_bw_gbps"], rl["bw_util_pct"],
             rl["roofline_pct"], rl["bound_by"], r["data"]["weight_MiB"]))
    print("        correctness: rel_l2_err=%.4g" % r["correctness"]["rel_l2_err"])


def run_case(gpu, hw, case, glob):
    kind = case["kernel"]
    fmt = case.get("format", "gguf_q4_k")
    a = (gpu, hw, fmt, case["hidden"], case["inter"], case["experts"], case["topk"],
         case.get("iters", glob["iters"]), case.get("warmup", glob["warmup"]),
         case.get("flush_l2", glob["flush_l2"]), case.get("seed", 1234),
         case.get("verify_slots", 2), case.get("verify_rows", 64))
    if kind == "gate_up":
        return run_gate_up(*a)
    if kind == "down":
        return run_down(*a)
    raise ValueError("unknown moe kernel kind %r" % kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--platform", type=int, default=0)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    glob = {"iters": cfg.get("iters", 40), "warmup": cfg.get("warmup", 10), "flush_l2": cfg.get("flush_l2", True)}
    hw = HW(cfg.get("hw"))
    gpu = Gpu(args.device, args.platform)
    gpu.setup_flush(cfg.get("flush_mib", 128))
    print("=" * 92)
    print("Device : %s" % gpu.device.name)
    print("HW peak: INT8(dp4a) %.2f TOPS | BW %.0f GB/s | L2 %.1f MiB"
          % (hw.int8_tops, hw.peak_bw_gbps, hw.l2_cache_bytes / 1048576))
    print("Bench  : iters=%d warmup=%d flush_l2=%s" % (glob["iters"], glob["warmup"], glob["flush_l2"]))
    print("=" * 92)
    records = []
    for case in cfg["cases"]:
        try:
            r = run_case(gpu, hw, case, glob)
            records.append(r); print_rec(r)
        except Exception as e:
            print("  [ERR ] %s: %s" % (case.get("kernel"), e))
            import traceback; traceback.print_exc()
    out = args.out or cfg.get("out")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"device": gpu.device.name, "hw": hw.summary(), "records": records,
                       "timestamp_unix": int(time.time())}, f, indent=2)
        print("\nWrote report -> %s" % out)
    npass = sum(1 for r in records if r["correctness"].get("passed"))
    print("\n%d/%d cases passed correctness." % (npass, len(records)))
    return 0 if npass == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
