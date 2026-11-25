"""Strict Stage2 Q1 checks and alternating A/B; run GPU work on remote BMG only.

Default runs correctness, including captured scalar FP32 projection. --benchmark
adds 100 warmups / 20 samples by default; --flush evicts 96 MiB BEFORE each whole
Q1, never between projection and prepare. GPU event sum excludes inter-event gaps;
host enqueue-to-finish timing also includes both launches and synchronization.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time

import numpy as np

import qsa_config
import qsa_harness as H
import qsa_ref as R
from qsa_prepare_dpas import PrepareDPAS

WG = 16
GUARD = -777.0


class Kernels:
    def __init__(self, cfg):
        from clops import cl

        self.base = H.build_kernel("qsa_prepare.cm", cfg.jit("qsa_prepare", QSA_PREPARE_WG=WG))
        self.dpas = PrepareDPAS(cfg, WG)
        # Test-only capture from the ACTUAL scalar baseline, before half conversion.
        # Production source/ABI and performance runs have no capture stores.
        src = H.flatten_source("qsa_prepare.cm", cfg.jit("qsa_prepare_capture", QSA_PREPARE_WG=WG))
        signature = "uint32_t num_wg) {"
        store = "cm_slm_block_write(slm_proj, j0 * (uint)sizeof(float), out);"
        assert src.count(signature) == 1 and src.count(store) == 2
        src = src.replace(signature, "uint32_t num_wg, float* capture ATTR) {")
        src = src.replace(store, store + "\ncm_ptr_store<float, 16>(capture, "
                          "(token_idx * J + j0) * (uint)sizeof(float), out);")
        self.capture = cl.kernels(src, H.QSA_BUILD_OPTIONS)


class Case:
    def __init__(self, cfg, seqs, seed=13, guard=True, distribution="random"):
        from clops import cl

        self.cfg = cfg
        self.batch = b = H.make_batch(seqs, cfg.pa_block_size, seed=seed)
        self.nt, self.j = b.num_tokens, (cfg.indexer_heads + 1) * cfg.indexer_head_dim
        rng = np.random.default_rng(seed)
        # Continuous half inputs (not only exactly representable 1/16 steps).
        self.x = H.f16(rng.normal(0, 0.25, (self.nt, cfg.hidden_size)))
        self.w = H.f16(rng.normal(0, 0.1, (self.j, cfg.hidden_size)))
        if distribution == "zero":
            self.x.fill(0)
        self.qn = H.f16(rng.uniform(0.5, 1.5, cfg.indexer_head_dim))
        self.kn = H.f16(rng.uniform(0.5, 1.5, cfg.indexer_head_dim))
        # Nontrivial position_ids, deliberately different from absolute cache positions.
        self.pos = np.array([b.seq_of_token(t)[1] + 7 for t in range(self.nt)], np.int32)
        max_pos = max([p + q + 16 for p, q in seqs] + [16])
        angles = rng.uniform(-3, 3, (max_pos, cfg.indexer_rotary_dim // 2))
        self.rot = H.f16(np.stack((np.cos(angles), np.sin(angles)), axis=-1).reshape(max_pos, -1))
        self.raw = H.f16(rng.normal(0, 0.4, (b.num_blocks, cfg.pa_block_size, cfg.indexer_head_dim)))
        self.summ = H.f16(rng.normal(0, 0.4, (b.num_blocks, cfg.summaries_per_page, cfg.indexer_head_dim)))
        # Avoid make_prepare_map's empty-q/nonaligned-past behavior without changing a shared helper.
        rows = [(s, block) for s, (p, q) in enumerate(seqs) if q
                for block in range(p // cfg.compress_ratio, (p + q - 1) // cfg.compress_ratio + 1)]
        self.nwg = len(rows)
        # clops cannot copy zero bytes; dummy storage is never dispatched/read.
        self.pmap = cl.tensor(np.array(rows or [(0, 0)], np.int32).reshape(-1, 2))
        self.inputs = [cl.tensor(a if a.size else np.zeros(1, a.dtype)) for a in (self.x, self.w, self.qn, self.kn, self.rot, self.pos,
                       b.past_lens, b.subsequence_begins, b.block_indices, b.block_indices_begins)]
        self.guard = guard
        self.gws, self.lws = [1, 1, (self.nwg + int(guard)) * WG], [1, 1, WG]

    def reference(self, proj, half_pool=False):
        c, b = self.cfg, self.batch
        di, hi, r = c.indexer_head_dim, c.indexer_heads, c.compress_ratio
        iq = np.zeros((self.nt, hi, di), np.float32)
        raw, summ = self.raw.copy(), self.summ.copy()
        raw_mask, sum_mask = np.zeros(raw.shape, bool), np.zeros(summ.shape, bool)
        table = self.rot.astype(np.float32).reshape(-1)

        def norm_rope(v, weight, pos):
            return R._rope(R._rms_norm(v, weight.astype(np.float32), c.indexer_norm_eps),
                           table, pos, c.indexer_rotary_dim)

        for t in range(self.nt):
            for h in range(hi):
                iq[t, h] = norm_rope(proj[t, h * di:(h + 1) * di], self.qn, int(self.pos[t]))
        for s, (past, q) in enumerate(b.seqs):
            if not q:
                continue
            begin, pb = int(b.subsequence_begins[s]), int(b.block_indices_begins[s])

            def page(p):
                return int(b.block_indices[pb + p // c.pa_block_size])

            for p in range(past, past + q):
                raw[page(p), p % c.pa_block_size] = proj[begin + p - past, hi * di:]
                raw_mask[page(p), p % c.pa_block_size] = True
            for block in range(past // r, (past + q) // r):
                pooled = np.zeros(di, np.float32)
                for p in range(block * r, (block + 1) * r):
                    if p < past:
                        v = self.raw[page(p), p % c.pa_block_size].astype(np.float32)
                    else:
                        v = proj[begin + p - past, hi * di:]
                        if half_pool:
                            v = v.astype(np.float16).astype(np.float32)
                    pooled += v
                pooled *= np.float32(1.0 / r)
                pg, slot = page(block * r), block % c.summaries_per_page
                summ[pg, slot] = norm_rope(pooled, self.kn, block * r)
                sum_mask[pg, slot] = True
        return (iq.astype(np.float16), raw, summ), (raw_mask, sum_mask)


class Pipeline:
    def __init__(self, case, kernels, dpas, capture=False):
        from clops import cl

        self.c, self.k, self.dpas, self.capture = case, kernels, dpas, capture
        self.shapes = [(case.nt, case.cfg.indexer_heads, case.cfg.indexer_head_dim),
                       case.raw.shape, case.summ.shape]
        self.out = [cl.tensor(np.r_[a.reshape(-1), np.full(32, GUARD, np.float16)])
                    for a in (np.full(self.shapes[0], np.nan, np.float16), case.raw, case.summ)]
        self.proj = cl.tensor(np.r_[np.full(case.nt * case.j, np.nan, np.float32),
                                   np.full(32, GUARD, np.float32)])

    def enqueue(self):
        c = self.c
        if c.nt == 0:
            return
        args = (*c.inputs, self.out[1], self.out[2], self.out[0], c.pmap, c.nwg)
        if self.dpas:
            self.k.dpas.enqueue(self.proj, c.nt, c.gws, c.lws, *args, guard_tiles=c.guard)
        elif self.capture:
            self.k.capture.enqueue("qsa_prepare_capture", c.gws, c.lws, *args, self.proj)
        else:
            self.k.base.enqueue("qsa_prepare", c.gws, c.lws, *args)

    def outputs(self):
        result = []
        for tensor, shape in zip(self.out, self.shapes):
            a = tensor.numpy().reshape(-1)
            np.testing.assert_array_equal(a[-32:], GUARD)
            result.append(a[:-32].reshape(shape))
        return result

    def projection(self):
        a = self.proj.numpy().reshape(-1)
        np.testing.assert_array_equal(a[-32:], GUARD)
        return a[:-32].reshape(self.c.nt, self.c.j)


def compare(got, want, *, atol=8e-3, rtol=2e-3):
    np.testing.assert_allclose(got, want, atol=atol, rtol=rtol)
    delta = np.abs(got.astype(np.float64) - want.astype(np.float64))
    return {"max_abs": float(delta.max(initial=0)), "rms": float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.0}


def validate_case(c, kernels, label, capture=True):
    from clops import cl

    pair = [Pipeline(c, kernels, False, capture), Pipeline(c, kernels, True)]
    for p in pair:
        p.enqueue()
    cl.finish()
    # FP64 oracle for small edge tests; FP32 BLAS for large benchmark-sized validation.
    dtype = np.float64 if capture else np.float32
    ref_proj = (c.x.astype(dtype) @ c.w.astype(dtype).T).astype(np.float32)
    projected = pair[1].projection()
    row = {"case": label, "tokens": c.nt,
           "projection_ref": compare(projected, ref_proj, atol=3e-5, rtol=2e-5)}
    if capture:
        scalar = pair[0].projection()
        row["scalar_projection_ref"] = compare(scalar, ref_proj, atol=3e-5, rtol=2e-5)
        row["projection_scalar"] = compare(projected, scalar, atol=3e-5, rtol=2e-5)
    ref, masks = c.reference(ref_proj)
    a, b = [p.outputs() for p in pair]
    for i, name in enumerate(("iq", "raw", "summary")):
        row[name + "_ref"] = compare(b[i], ref[i])
        row[name + "_scalar"] = compare(b[i], a[i])
        compare(a[i], ref[i])
        if i:
            # Past cache and incomplete summaries must be bitwise untouched.
            for out in (a[i], b[i]):
                np.testing.assert_array_equal(out[~masks[i - 1]], (c.raw, c.summ)[i - 1][~masks[i - 1]])
    # Isolate postprocessing from accumulation order using actual DPAS FP32 rows.
    own_ref, _ = c.reference(projected)
    for out, expected in zip(b, own_ref):
        compare(out, expected)
    print("PASS " + json.dumps(row), flush=True)
    return pair


def pooling_precision(cfg, kernels):
    from clops import cl

    c = Case(cfg, [(0, cfg.compress_ratio)])
    p = Pipeline(c, kernels, True)
    proj = np.ones((c.nt, c.j), np.float32)
    di = cfg.indexer_head_dim
    proj[:, -di:] = 0
    proj[0, -di:] = 1.0001
    proj[1, -di:] = -1.0
    p.proj = cl.tensor(np.r_[proj.reshape(-1), np.full(32, GUARD, np.float32)])
    kernels.dpas.prepare.enqueue("qsa_prepare_projected", c.gws, c.lws, p.proj,
                                 *c.inputs, p.out[1], p.out[2], p.out[0], c.pmap, c.nwg)
    cl.finish()
    expected, _ = c.reference(proj)
    wrong, _ = c.reference(proj, half_pool=True)
    actual = p.outputs()
    for a, b in zip(actual, expected):
        compare(a, b, atol=1e-3, rtol=1e-3)
    gap = float(np.max(np.abs(expected[2].astype(np.float32) - wrong[2].astype(np.float32))))
    assert gap > 1e-2, gap
    print(f"PASS FP32 pooling cancellation: half-reload alternative differs by {gap:.8g}", flush=True)


def stats(values):
    a = np.asarray(values)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95))}


def benchmark(cfg, kernels, args):
    from clops import cl

    flusher = H.CacheFlusher() if args.flush else None
    for mode in args.modes:
        for n in args.sizes:
            c = Case(cfg, [(0, n)] if mode == "prefill" else [(n, 1)], guard=False)
            pair = validate_case(c, kernels, f"bench-{mode}-{n}", capture=False)
            cl.profiling(True)
            for i in range(args.warmup):
                for j in (i % 2, 1 - i % 2):
                    pair[j].enqueue()
                cl.finish()
            events, wall = [[], []], [[], []]
            for i in range(args.samples):
                for j in (i % 2, 1 - i % 2):
                    if flusher:
                        flusher.flush()
                    start = time.perf_counter_ns()
                    pair[j].enqueue()
                    ns = cl.finish()
                    wall[j].append((time.perf_counter_ns() - start) * 1e-6)
                    ns = np.asarray(ns, np.float64).reshape(-1)
                    assert len(ns) == (2 if j else 1), ns
                    events[j].append(ns * 1e-6)
            row = {"mode": mode, "size": n, "tokens": c.nt, "flush": args.flush,
                   "warmup": args.warmup, "samples": args.samples}
            for j, name in enumerate(("baseline", "dpas")):
                a = np.asarray(events[j])
                row[name] = {"gpu_sum": stats(a.sum(axis=1)), "stage_mean_ms": a.mean(axis=0).tolist(),
                             "host_finish": stats(wall[j])}
            row["gpu_ratio"] = row["baseline"]["gpu_sum"]["mean_ms"] / row["dpas"]["gpu_sum"]["mean_ms"]
            print("BENCH " + json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--modes", choices=["prefill", "decode"], nargs="+", default=["prefill", "decode"])
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--flush", action="store_true")
    args = ap.parse_args()
    if args.warmup < 0 or args.samples < 1 or any(n < 1 for n in args.sizes):
        ap.error("warmup >= 0, samples >= 1, sizes >= 1 required")
    from clops import cl

    print("DEVICE", cl.dev_info(), flush=True)
    cfg = qsa_config.load()
    kernels = Kernels(cfg)
    cases = [([(0, 1)], "single"), ([(3, 1)], "decode-complete"),
             ([(0, 15)], "tail15"), ([(0, 16)], "full16"), ([(0, 17)], "tail17"),
             ([(0, 0), (3, 6), (15, 19), (34, 1), (65, 0), (0, 33)], "mixed"),
             ([(0, 0), (7, 0)], "empty")]
    for seed in (13, 29):
        for seqs, label in cases:
            validate_case(Case(cfg, seqs, seed), kernels, f"{label}-seed{seed}")
    validate_case(Case(cfg, [(3, 6), (15, 17)], distribution="zero"), kernels, "zero")
    pooling_precision(cfg, kernels)
    # Alternate valid geometry (one K step wider than 32, non-power-of-two Di).
    alt = replace(cfg, hidden_size=96, indexer_heads=2, indexer_head_dim=48,
                  indexer_rotary_dim=16, compress_ratio=2, pa_block_size=32)
    alt_kernels = Kernels(alt)
    validate_case(Case(alt, [(1, 18), (31, 3)]), alt_kernels, "D96-Di48-H2-r2-p32")
    for bad in (replace(cfg, hidden_size=48), replace(cfg, indexer_head_dim=127)):
        try:
            PrepareDPAS(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported shape was not rejected")
    if args.benchmark:
        benchmark(cfg, kernels, args)
    print("ALL Q1 DPAS CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()