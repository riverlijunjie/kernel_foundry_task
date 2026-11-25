"""Strict remote-only Stage5 tests and same-input optimized-DPAS/GQA A/B.

Default runs edge tests. --benchmark-only checks every timed shape against FP32,
scalar, baseline DPAS and optimized DPAS before timing only optimized DPAS/GQA.
All selections are synthetic, not model traces. No Q1/Q2 or automatic dispatch.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from time import perf_counter

import numpy as np

import qsa_config
import qsa_harness as H
from qsa_attention_gqa import Q3GQAAttention
from test_q3_sparse_attention_dpas import heads_per_tile, workers
from validate_q3_dpas import build
from validate_q3_split import Case as BaseCase, tensor

COMPILED = {}
WORST = {"fp32": 0.0, "scalar": 0.0, "optimized": 0.0}


class Case(BaseCase):
    def __init__(self, cfg, seqs, *, pattern="independent", padded=True, seed=17,
                 zero_query=False):
        super().__init__(cfg, seqs, padded=padded, seed=seed, zero_query=zero_query)
        self.pattern = pattern
        if pattern != "independent":
            self.sel.fill(-1)
            self.cnt.fill(0)
            for t in range(self.nt):
                nc = (self.batch.seq_of_token(t)[1] + 1) // cfg.compress_ratio
                if pattern == "empty":
                    ids = np.empty(0, np.int32)
                elif pattern == "overlap":
                    ids = np.arange(min(nc, cfg.block_topk), dtype=np.int32)
                elif pattern == "disjoint":
                    # Four adjacent token lists have no common blocks. This is
                    # intentionally NOT dense semantics when the budget exceeds nc.
                    ids = np.arange(t % 4, nc, 4, dtype=np.int32)[:cfg.block_topk]
                elif pattern == "patterns":
                    p = ([0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4],
                         [4, 5, 6, 7], [], [0], [0, 1], [0, 1, 2])[t % 8]
                    ids = np.asarray([i for i in p if i < nc][:cfg.block_topk], np.int32)
                else:
                    raise ValueError(pattern)
                self.cnt[t] = len(ids)
                self.sel[t, :len(ids)] = ids
            self.inputs[7:9] = [tensor(self.sel), tensor(self.cnt)]
        for t in range(self.nt):
            ids = self.sel[t, :int(self.cnt[t])]
            nc = (self.batch.seq_of_token(t)[1] + 1) // cfg.compress_ratio
            assert 0 <= self.cnt[t] <= cfg.block_topk
            assert ((ids >= 0) & (ids < nc)).all() and (np.diff(ids) > 0).all()
        self.tm, self.nm = H.make_score_tile_map(self.batch)
        self.token_map, n = H.make_score_tile_map(self.batch, tile=1)
        assert n == self.nt
        self.device_tm, self.device_tokens = tensor(self.tm), tensor(self.token_map)

    def reference(self):
        """FP32 full stable softmax, matrix-vectorized across each GQA group.

        Independent of GPU online recurrence/DPAS half probabilities/reduction.
        Gather KV once per group to make full 8K validation practical on one CPU.
        """
        cfg, b = self.cfg, self.batch
        out = np.zeros((self.nt, self.hq, self.dv), np.float32)
        for t in range(self.nt):
            pos = self.positions(t)
            if not len(pos):
                continue
            seq, _ = b.seq_of_token(t)
            pages = b.block_indices[int(b.block_indices_begins[seq]) + pos // cfg.pa_block_size]
            tips = pos % cfg.pa_block_size
            qrow = self.query[t].reshape(self.hq, self.qhs).astype(np.float32)
            for kh in range(cfg.kv_heads_num):
                hs = slice(kh * cfg.n_rep, (kh + 1) * cfg.n_rep)
                k = self.kc[pages, kh, tips].astype(np.float32)
                v = self.vc[pages, kh, tips].astype(np.float32)
                scores = (qrow[hs, :cfg.k_head_size] * np.float32(cfg.scale)) @ k.T
                e = np.exp(scores - scores.max(axis=1, keepdims=True))
                value = (e @ v) / e.sum(axis=1, keepdims=True, dtype=np.float32)
                if cfg.has_output_gate:
                    value *= np.float32(1) / (np.float32(1) + np.exp(-qrow[hs, self.gd:self.gd + self.dv]))
                out[t, hs] = value
        return out.reshape(self.nt, self.hq * self.dv)


class Call:
    def __init__(self, c, variant, *, guard=False, geometry=None):
        self.c, self.variant, self.guard = c, variant, guard
        self.w, self.ht = geometry or (workers(c.cfg), heads_per_tile(c.cfg))
        key = (tuple(c.cfg.jit("unused").items()), variant, self.w, self.ht)
        if key not in COMPILED:
            if variant in ("baseline", "optimized"):
                if not self.w:
                    raise ValueError("existing DPAS geometry unsupported")
                COMPILED[key] = build(c.cfg, self.w, self.ht, variant)
            else:
                COMPILED[key] = Q3GQAAttention(c.cfg, enabled=variant == "gqa", fallback="scalar")
        self.impl = COMPILED[key]
        self.out = tensor(np.full(c.output_size + 64, np.nan, np.float16))

    def enqueue(self):
        c = self.c
        if not c.nt:
            return
        if self.variant in ("baseline", "optimized"):
            g = int(self.guard)
            self.impl.enqueue("qsa_sparse_attention_dpas",
                [(c.hq // self.ht + g) * self.w, (c.nm + g) * self.ht, 1], [self.w, 1, 1],
                *c.inputs, c.device_tm, self.out, c.nm, c.qbase, c.pitch, c.qhs, c.gd)
        else:
            self.impl.enqueue(c.inputs, self.out, c.nt, c.batch.num_seqs,
                c.qbase, c.pitch, c.qhs, c.gd, token_map=c.device_tokens, guard_items=self.guard)

    def result(self):
        c = self.c
        out = self.out.numpy().reshape(-1)
        assert np.isnan(out[c.output_size:]).all(), "output guard overwritten"
        got = out[:c.output_size].reshape(c.nt, c.hq * c.dv).astype(np.float32)
        assert np.isfinite(got).all(), "missing or nonfinite output"
        return got


def check(c, *, label="case", geometry=None, compare_old_dpas=True):
    from clops import cl

    ref = c.reference()
    variants = ("scalar", "baseline", "optimized", "gqa") if compare_old_dpas else ("scalar", "gqa")
    calls = {v: Call(c, v, guard=True, geometry=geometry)
             for v in variants}
    outputs, errors = {}, {}
    for name, call in calls.items():
        call.enqueue()
        cl.finish()
        got = call.result()
        np.testing.assert_allclose(got, ref, atol=2e-3, rtol=2e-3, err_msg=f"{label}/{name}/FP32")
        outputs[name] = got
        errors[name] = float(np.max(np.abs(got - ref), initial=0))
        for t in range(c.nt):
            if not len(c.positions(t)):
                np.testing.assert_array_equal(got[t], 0)
    if compare_old_dpas:
        np.testing.assert_array_equal(outputs["baseline"], outputs["optimized"])
    for name in (("scalar", "optimized") if compare_old_dpas else ("scalar",)):
        np.testing.assert_allclose(outputs["gqa"], outputs[name], atol=2e-3, rtol=2e-3)
        WORST[name] = max(WORST[name], float(np.max(np.abs(outputs["gqa"] - outputs[name]), initial=0)))
    WORST["fp32"] = max(WORST["fp32"], errors["gqa"])
    status = "gqa" if calls["gqa"].impl.enabled else "scalar-fallback"
    print("PASS " + json.dumps(dict(case=label, tokens=c.nt, path=status, errors=errors)), flush=True)
    # Timed launches must NOT contain the validation-only extra work groups.
    for call in calls.values():
        call.guard = False
    return calls, errors


def validate():
    from clops import cl

    cfg = qsa_config.load()
    mixed = [(0, 0), (0, 1), (1, 3), (3, 5), (14, 7), (15, 17), (31, 19), (63, 1)]
    cases = [("dense-partial", cfg, [(0, 129)], "independent", False, None),
             ("chunked-pitches", cfg, mixed, "independent", True, None),
             ("page-crossing", replace(cfg, block_topk=4), [(96, 33)], "patterns", True, None),
             ("empty-tail", cfg, mixed, "empty", True, None),
             ("K0", replace(cfg, block_topk=0), mixed, "empty", True, None),
             ("K1", replace(cfg, block_topk=1), mixed, "independent", False, None),
             ("ungated", replace(cfg, has_output_gate=False), mixed, "independent", True, None),
             ("Dh128-Dv256", replace(cfg, k_head_size=128, scale=128**-0.5), mixed, "independent", True, None),
             ("all-empty", cfg, [(0, 0), (16, 0)], "empty", True, None),
             ("no-sequences", cfg, [], "empty", True, None)]
    for pattern in ("independent", "overlap", "disjoint"):
        cases.append(("sparse-" + pattern, replace(cfg, block_topk=7), [(256, 33)], pattern, True, None))
    for page in (8, 32):
        cases.append((f"page{page}", replace(cfg, pa_block_size=page), mixed, "independent", True, None))
    for r in (2, 8, 16):
        cases.append((f"r{r}-all-tail-remainders", replace(cfg, compress_ratio=r, block_topk=4),
                      [(64 + rem, 1) for rem in range(r)] + [(33, 17)], "independent", True, None))
    for rep in (1, 3, 6, 12, 16):
        conf = replace(cfg, heads_num=rep * 2, k_head_size=128, v_head_size=128,
                       scale=128**-0.5, block_topk=4)
        # Old HT16/TT1 load_vnni_cols<1> fails CM type checking (8 vs16 uints).
        # Use the existing legal HT8/TT2 comparator; GQA still fills all16 heads.
        geo = (workers(conf), 8) if rep == 16 else None
        cases.append((f"padded-heads-{rep}", conf, mixed, "independent", True, geo))
    # Original geometry variants still compare old kernels; NEW kernel stays W4.
    for geo in ((8, 4), (4, 2), (4, 1)):
        cases.append((f"old-geometry-{geo}", cfg, mixed, "independent", True, geo))
    for heads, geo in ((24, (1, 4)), (16, (1, 8))):
        conf = replace(cfg, heads_num=heads, k_head_size=64, v_head_size=64, scale=0.125)
        cases.append((f"explicit-fallback-D64-H{heads}", conf, mixed, "independent", True, geo))
    for name, conf, seqs, pattern, padded, geo in cases:
        # Reproduced B580 IGC Floating point exception compiling OLD DPAS at
        # Dh128/Dv256/W4. New GQA and scalar compile and pass this shape.
        # Explicitly omit only that old comparator; never swallow build failures.
        old_dpas = name != "Dh128-Dv256"
        if not old_dpas:
            print("UNSUPPORTED OLD DPAS COMPARISON Dh128/Dv256/W4: IGC internal compiler error; checking scalar + FP32", flush=True)
        check(Case(conf, seqs, pattern=pattern, padded=padded), label=name, geometry=geo,
              compare_old_dpas=old_dpas)
    check(Case(cfg, mixed, zero_query=True, seed=29), label="zero-query-second-seed")
    # Compile-free host rejection BEFORE device access. No hidden auto fallback.
    for bad in (replace(cfg, heads_num=34), replace(cfg, heads_num=25),
                replace(cfg, k_head_size=64), replace(cfg, pa_block_size=15),
                replace(cfg, compress_ratio=3), replace(cfg, is_causal=False)):
        try:
            Q3GQAAttention(bad, enabled=True)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted unsupported geometry")
    c = Case(cfg, [(0, 1)])
    call = Call(c, "gqa")
    for layout in ((2, c.pitch, c.qhs, c.gd), (0, c.pitch + 2, c.qhs, c.gd),
                   (0, c.pitch, c.qhs + 2, c.gd), (0, c.pitch, c.qhs, c.gd + 1)):
        try:
            call.impl.enqueue(c.inputs, call.out, 1, 1, *layout, token_map=c.device_tokens)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted misaligned layout")
    call.impl.enqueue([], None, 0, 0, 0, 0, 0, 0)
    assert not Q3GQAAttention(cfg).enabled
    cl.finish()
    print(f"ALL {len(cases) + 1} STAGE5 CASES PASSED; WORST {json.dumps(WORST)}", flush=True)


def benchmark(args):
    from clops import cl

    cfg = qsa_config.load()
    flusher = H.CacheFlusher() if args.flush else None
    cl.profiling(True)
    for past in args.pasts:
        for n in args.sizes:
            for pattern in args.patterns:
                c = Case(cfg, [(past, n)], pattern=pattern, padded=False)
                calls, errors = check(c, label=f"bench-p{past}-q{n}-{pattern}")
                names = ("optimized", "gqa")
                for i in range(args.warmup):
                    for name in (names if i % 2 == 0 else names[::-1]):
                        calls[name].enqueue()
                    cl.finish()
                samples, hosts = {v: [] for v in names}, {v: [] for v in names}
                for i in range(args.samples):
                    for name in (names if i % 2 == 0 else names[::-1]):
                        if flusher:
                            flusher.flush()
                        t0 = perf_counter()
                        calls[name].enqueue()
                        ns = cl.finish()
                        hosts[name].append((perf_counter() - t0) * 1000)
                        assert len(ns) == 1, ns
                        samples[name].append(float(ns[0]) * 1e-6)
                row = dict(past=past, query=n, pattern=pattern, topk=cfg.block_topk,
                           cache="cold96MiB" if args.flush else "no-flush",
                           warmup=args.warmup, samples=args.samples, errors=errors, variants={})
                for name in names:
                    a = np.asarray(samples[name])
                    row["variants"][name] = dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                        p95_ms=float(np.percentile(a, 95)), host_mean_ms=float(np.mean(hosts[name])),
                        samples_ms=samples[name], host_samples_ms=hosts[name])
                    np.testing.assert_allclose(calls[name].result(), calls["scalar"].result(), atol=2e-3, rtol=2e-3)
                print("BENCH " + json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--flush", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--pasts", type=int, nargs="+", default=[0])
    ap.add_argument("--patterns", nargs="+", choices=["independent", "overlap", "disjoint"],
                    default=["independent", "overlap"])
    args = ap.parse_args()
    if args.warmup < 0 or min(args.samples, *args.sizes) < 1 or min(args.pasts) < 0:
        ap.error("invalid sample/shape arguments")
    from clops import cl

    print("DEVICE", cl.dev_info(), flush=True)
    if args.benchmark_only:
        benchmark(args)
    else:
        validate()


if __name__ == "__main__":
    main()