"""Standalone gr_read / gr_write (Gated Residual, hyper-connection) CM kernels:
build -> optional correctness check vs FP64 numpy reference -> roofline sweep.

Shapes default to the production hyper-connection width used by
`gated_residual_gpu_test.cpp::run_point`: C=4 branches, hidden_size=2560 (D=640
per branch), low-rank width L=320, eps=1e-6.

Run on the remote GPU only (see cm_kernel/README.md):
    export CM_FE_DIR=/mnt/river   # (Linux BMG) or set on the PTL box's env
    python test_gr_gated_residual.py --check --rows 1 8 64 512 2048 4096
"""
from __future__ import annotations

import argparse

import numpy as np

import gr_ref
import qsa_harness as H
from gr_read_dpas import GRReadDPAS

GR_WG_SIZE = 16
DPAS_TOL = 6e-2  # matches the repo's scalar-vs-dpas tolerance convention (fp16 rounding
                 # through the extra materialized Rbar/T/gate scratch tensors)


def make_gr_read_case(rows: int, C: int, D: int, L: int, eps: float,
                      norm_gain_plus_one: bool, produce_write_gate: bool, kern):
    from clops import cl

    CD = C * D
    rng = np.random.default_rng(0)
    residual = H.rand_f16(rng, rows, CD, scale=1.0)
    gain = H.rand_f16(rng, CD, scale=0.5, bias=1.0)
    w_down = H.rand_f16(rng, L, CD, scale=0.1)
    w_up = H.rand_f16(rng, CD, L, scale=0.1)
    w_inject = H.rand_f16(rng, C, CD, scale=0.1) if produce_write_gate else None

    t_residual = cl.tensor(residual.reshape(-1))
    t_gain = cl.tensor(gain.reshape(-1))
    t_wdown = cl.tensor(w_down.reshape(-1))
    t_wup = cl.tensor(w_up.reshape(-1))
    t_winject = cl.tensor(w_inject.reshape(-1)) if produce_write_gate else None
    t_mixed = cl.tensor(np.zeros((rows, D), np.float16))
    t_gate = cl.tensor(np.zeros((rows, C), np.float16)) if produce_write_gate else None

    GWS, LWS = [GR_WG_SIZE, rows, 1], [GR_WG_SIZE, 1, 1]

    def enqueue():
        args = [t_residual, t_gain, t_wdown, t_wup]
        if produce_write_gate:
            args.append(t_winject)
        args.append(t_mixed)
        if produce_write_gate:
            args.append(t_gate)
        args.append(rows)
        kern.enqueue("gr_read", GWS, LWS, *args)

    def check():
        enqueue()
        cl.finish()
        gain_eff = gain.astype(np.float32)
        if norm_gain_plus_one:
            gain_eff = gain_eff + 1.0
        mixed_ref, gate_ref = gr_ref.gr_read_ref(
            residual.astype(np.float32), gain_eff, w_down.astype(np.float32), w_up.astype(np.float32),
            w_inject.astype(np.float32) if produce_write_gate else None, C, D, L, eps)
        mixed = t_mixed.numpy().reshape(rows, D).astype(np.float64)
        merr = float(np.abs(mixed - mixed_ref).max())
        ok = merr < 2e-2
        msg = (f"[GR-read] rows={rows} D={D} L={L} wgate={produce_write_gate} "
               f"mixed_err={merr:.4f} -> {'OK' if ok else 'FAIL'}")
        if produce_write_gate:
            gate = t_gate.numpy().reshape(rows, C).astype(np.float64)
            gerr = float(np.abs(gate - gate_ref).max())
            in_range = bool(gate.min() > 0.0 and gate.max() < 2.0)
            ok = ok and gerr < 2e-2 and in_range
            msg += f" gate_err={gerr:.4f} in_range={in_range}"
        print(msg)
        return ok

    # Minimal-unique DRAM traffic: weights counted once, residual/output per row.
    weight_bytes = (L * CD + CD * L + (C * CD if produce_write_gate else 0)) * 2
    per_row_bytes = CD * 2 + D * 2 + (C * 2 if produce_write_gate else 0)
    total_bytes = weight_bytes + rows * per_row_bytes
    # down-proj + up-proj dot products dominate; write gate adds a small C x CD term.
    flops = rows * (4.0 * L * CD + (2.0 * C * CD if produce_write_gate else 0.0))

    return dict(name="GR-read", enqueue=enqueue, flops=flops, bytes=total_bytes, check=check)


def make_gr_read_dpas_case(rows: int, C: int, D: int, L: int, eps: float,
                          norm_gain_plus_one: bool, produce_write_gate: bool, pipeline: GRReadDPAS):
    from clops import cl

    CD = C * D
    rng = np.random.default_rng(0)
    residual = H.rand_f16(rng, rows, CD, scale=1.0)
    gain = H.rand_f16(rng, CD, scale=0.5, bias=1.0)
    w_down = H.rand_f16(rng, L, CD, scale=0.1)
    w_up = H.rand_f16(rng, CD, L, scale=0.1)
    w_inject = H.rand_f16(rng, C, CD, scale=0.1) if produce_write_gate else None

    t_residual = cl.tensor(residual.reshape(-1))
    t_gain = cl.tensor(gain.reshape(-1))
    t_wdown = cl.tensor(w_down.reshape(-1))
    t_wup = cl.tensor(w_up.reshape(-1))
    t_winject = cl.tensor(w_inject.reshape(-1)) if produce_write_gate else None
    t_rbar = cl.tensor(np.zeros((rows, CD), np.float16))
    t_t = cl.tensor(np.zeros((rows, L), np.float16))
    t_gate = cl.tensor(np.zeros((rows, CD), np.float16))
    t_mixed = cl.tensor(np.zeros((rows, D), np.float16))
    t_wgate_out = cl.tensor(np.zeros((rows, C), np.float16)) if produce_write_gate else None

    def enqueue():
        pipeline.enqueue(rows, t_residual, t_gain, t_wdown, t_wup, t_winject,
                         t_rbar, t_t, t_gate, t_mixed, t_wgate_out)

    def check():
        enqueue()
        cl.finish()
        gain_eff = gain.astype(np.float32)
        if norm_gain_plus_one:
            gain_eff = gain_eff + 1.0
        mixed_ref, gate_ref = gr_ref.gr_read_ref(
            residual.astype(np.float32), gain_eff, w_down.astype(np.float32), w_up.astype(np.float32),
            w_inject.astype(np.float32) if produce_write_gate else None, C, D, L, eps)
        mixed = t_mixed.numpy().reshape(rows, D).astype(np.float64)
        merr = float(np.abs(mixed - mixed_ref).max())
        ok = merr < DPAS_TOL
        msg = (f"[GR-read-dpas] rows={rows} D={D} L={L} wgate={produce_write_gate} "
               f"mixed_err={merr:.4f} -> {'OK' if ok else 'FAIL'}")
        if produce_write_gate:
            gate = t_wgate_out.numpy().reshape(rows, C).astype(np.float64)
            gerr = float(np.abs(gate - gate_ref).max())
            in_range = bool(gate.min() > 0.0 and gate.max() < 2.0)
            ok = ok and gerr < DPAS_TOL and in_range
            msg += f" gate_err={gerr:.4f} in_range={in_range}"
        print(msg)
        return ok

    weight_bytes = (L * CD + CD * L + (C * CD if produce_write_gate else 0)) * 2
    # Extra scratch traffic vs the fused scalar kernel: Rbar/gate are each written once
    # and read TWICE (down-proj + combine, resp. combine + write-gate), T is written and
    # read once. This is the real, honest bandwidth cost of materializing the pipeline.
    scratch_bytes = rows * (CD * 2 * (3 if produce_write_gate else 2) + L * 2 * 2)
    per_row_bytes = CD * 2 + D * 2 + (C * 2 if produce_write_gate else 0)
    total_bytes = weight_bytes + rows * per_row_bytes + scratch_bytes
    flops = rows * (4.0 * L * CD + (2.0 * C * CD if produce_write_gate else 0.0))

    return dict(name="GR-read-dpas", enqueue=enqueue, flops=flops, bytes=total_bytes, check=check)


def make_gr_write_case(rows: int, C: int, D: int, kern):
    from clops import cl

    CD = C * D
    rng = np.random.default_rng(1)
    residual = H.rand_f16(rng, rows, CD, scale=1.0)
    y = H.rand_f16(rng, rows, D, scale=1.0)
    gate = H.f16(rng.uniform(0.0, 2.0, size=(rows, C)))

    t_residual = cl.tensor(residual.reshape(-1))
    t_y = cl.tensor(y.reshape(-1))
    t_gate = cl.tensor(gate.reshape(-1))
    t_out = cl.tensor(np.zeros((rows, CD), np.float16))

    GWS, LWS = [D // 16, C, rows], [1, 1, 1]

    def enqueue():
        kern.enqueue("gr_write", GWS, LWS, t_residual, t_y, t_gate, t_out, rows)

    def check():
        enqueue()
        cl.finish()
        ref = gr_ref.gr_write_ref(residual.astype(np.float32), y.astype(np.float32), gate.astype(np.float32), C, D)
        out = t_out.numpy().reshape(rows, CD).astype(np.float64)
        err = float(np.abs(out - ref).max())
        ok = err < 1e-2
        print(f"[GR-write] rows={rows} D={D} err={err:.4f} -> {'OK' if ok else 'FAIL'}")
        return ok

    flops = rows * CD * 2.0  # one multiply-add per output element
    bytes_ = rows * (CD * 2 + D * 2 + C * 2 + CD * 2)  # residual + y + gate in, residual_out
    return dict(name="GR-write", enqueue=enqueue, flops=flops, bytes=bytes_, check=check)


def make_gr_write_wide_case(rows: int, C: int, D: int, kern, tile: int):
    from clops import cl

    CD = C * D
    rng = np.random.default_rng(1)
    residual = H.rand_f16(rng, rows, CD, scale=1.0)
    y = H.rand_f16(rng, rows, D, scale=1.0)
    gate = H.f16(rng.uniform(0.0, 2.0, size=(rows, C)))

    t_residual = cl.tensor(residual.reshape(-1))
    t_y = cl.tensor(y.reshape(-1))
    t_gate = cl.tensor(gate.reshape(-1))
    t_out = cl.tensor(np.zeros((rows, CD), np.float16))

    GWS, LWS = [(D + tile - 1) // tile, rows, 1], [1, 1, 1]

    def enqueue():
        kern.enqueue("gr_write_wide", GWS, LWS, t_residual, t_y, t_gate, t_out, rows)

    def check():
        enqueue()
        cl.finish()
        ref = gr_ref.gr_write_ref(residual.astype(np.float32), y.astype(np.float32), gate.astype(np.float32), C, D)
        out = t_out.numpy().reshape(rows, CD).astype(np.float64)
        err = float(np.abs(out - ref).max())
        ok = err < 1e-2
        print(f"[GR-write-wide] rows={rows} D={D} tile={tile} err={err:.4f} -> {'OK' if ok else 'FAIL'}")
        return ok

    flops = rows * CD * 2.0
    bytes_ = rows * (CD * 2 + D * 2 + C * 2 + CD * 2)
    return dict(name="GR-write-wide", enqueue=enqueue, flops=flops, bytes=bytes_, check=check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 8, 64, 512, 2048, 4096],
                     help="token rows per call (1 ~= decode, larger ~= prefill)")
    ap.add_argument("--branches", type=int, default=4, help="C, hyper-connection branch count")
    ap.add_argument("--hidden", type=int, default=640, help="D, per-branch channel count")
    ap.add_argument("--lowrank", type=int, default=320, help="L, read-gate low rank")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--norm-gain-plus-one", action="store_true")
    ap.add_argument("--produce-write-gate", action="store_true", default=True)
    ap.add_argument("--no-write-gate", dest="produce_write_gate", action="store_false")
    ap.add_argument("--mode", choices=["read", "write", "both"], default="both")
    ap.add_argument("--read-variant", choices=["scalar", "dpas", "both"], default="both")
    ap.add_argument("--write-variant", choices=["scalar", "wide", "both"], default="both")
    ap.add_argument("--combine-tile", type=int, default=64, help="gr_combine elements/thread (dpas read pipeline)")
    ap.add_argument("--write-tile", type=int, default=64, help="gr_write_wide elements/thread")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    C, D, L = args.branches, args.hidden, args.lowrank

    if args.mode in ("read", "both"):
        if args.read_variant in ("scalar", "both"):
            read_kern = H.build_kernel("gr_read.cm", dict(
                KERNEL_NAME="gr_read",
                GR_BRANCHES=C, GR_HIDDEN=D, GR_LOWRANK=L, GR_EPS=args.eps,
                GR_NORM_GAIN_PLUS_ONE=1 if args.norm_gain_plus_one else 0,
                GR_PRODUCE_WRITE_GATE=1 if args.produce_write_gate else 0,
                GR_WG_SIZE=GR_WG_SIZE))

            def make_case_read(n):
                return make_gr_read_case(n, C, D, L, args.eps, args.norm_gain_plus_one,
                                          args.produce_write_gate, read_kern)

            H.sweep(args.rows, make_case_read, phase="prefill", iters=args.iters,
                    flush=not args.no_flush, do_check=args.check)

        if args.read_variant in ("dpas", "both"):
            pipeline = GRReadDPAS(C, D, L, args.eps, args.norm_gain_plus_one,
                                  args.produce_write_gate, wg=GR_WG_SIZE,
                                  combine_tile=args.combine_tile)

            def make_case_read_dpas(n):
                return make_gr_read_dpas_case(n, C, D, L, args.eps, args.norm_gain_plus_one,
                                              args.produce_write_gate, pipeline)

            H.sweep(args.rows, make_case_read_dpas, phase="prefill", iters=args.iters,
                    flush=not args.no_flush, do_check=args.check)

    if args.mode in ("write", "both"):
        if args.write_variant in ("scalar", "both"):
            write_kern = H.build_kernel("gr_write.cm", dict(
                KERNEL_NAME="gr_write", GR_BRANCHES=C, GR_HIDDEN=D))

            def make_case_write(n):
                return make_gr_write_case(n, C, D, write_kern)

            H.sweep(args.rows, make_case_write, phase="prefill", iters=args.iters,
                    flush=not args.no_flush, do_check=args.check)

        if args.write_variant in ("wide", "both"):
            write_wide_kern = H.build_kernel("gr_write_wide.cm", dict(
                KERNEL_NAME="gr_write_wide", GR_BRANCHES=C, GR_HIDDEN=D, GR_WRITE_TILE=args.write_tile))

            def make_case_write_wide(n):
                return make_gr_write_wide_case(n, C, D, write_wide_kern, args.write_tile)

            H.sweep(args.rows, make_case_write_wide, phase="prefill", iters=args.iters,
                    flush=not args.no_flush, do_check=args.check)


if __name__ == "__main__":
    main()
