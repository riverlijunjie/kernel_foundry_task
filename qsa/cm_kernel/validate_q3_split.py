"""Strict standalone Stage4 Q3 tests and remote-only alternating GPU A/B.

Run on the B580 with CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1.
Default: edge suite. --benchmark-only still checks each timed shape against FP32
NumPy AND the unchanged scalar kernel. No Q1/Q2 integration, no default routing.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import qsa_config
import qsa_harness as H
from qsa_attention_split import Q3Attention

PARTITIONS = (1, 2, 4, 8, 16)
COMPILED = {}
WORST = {"numpy": 0.0, "scalar": 0.0}


def implementation(cfg, partitions):
    key = (tuple(cfg.jit("unused").items()), partitions)
    if key not in COMPILED:
        COMPILED[key] = Q3Attention(cfg, partitions=partitions)
    return COMPILED[key]


def tensor(a):
    from clops import cl

    a = np.ascontiguousarray(a)
    return cl.tensor(a if a.size else np.zeros(1, dtype=a.dtype))


class Case:
    def __init__(self, cfg, seqs, *, selection="random", padded=True, seed=7,
                 zero_query=False):
        self.cfg = cfg
        self.batch = b = H.make_batch(seqs, cfg.pa_block_size, seed=seed)
        self.nt, self.hq, self.dv = b.num_tokens, cfg.heads_num, cfg.v_head_size
        dh, dv = cfg.k_head_size, cfg.v_head_size
        self.gd = dh + (16 if padded else 0) if cfg.has_output_gate else 0
        self.qhs = (self.gd + dv if cfg.has_output_gate else dh) + (16 if padded else 0)
        self.qbase = 32 if padded else 0
        self.pitch = self.hq * self.qhs + (32 if padded else 0)
        rng = np.random.default_rng(seed)
        self.query = H.rand_f16(rng, self.nt, self.hq * self.qhs)
        if zero_query:
            for h in range(self.hq):
                self.query[:, h * self.qhs:h * self.qhs + dh] = 0
        self.storage = np.full(self.qbase + self.nt * self.pitch, np.nan, np.float16)
        self.storage[self.qbase:].reshape(self.nt, self.pitch)[:, :self.hq * self.qhs] = self.query
        self.kc = H.rand_f16(rng, b.num_blocks, cfg.kv_heads_num, cfg.pa_block_size, dh)
        self.vc = H.rand_f16(rng, b.num_blocks, cfg.kv_heads_num, cfg.pa_block_size, dv)
        self.sel, self.cnt = H.make_selection(b, cfg.compress_ratio, cfg.block_topk, seed=seed)
        if selection == "empty":
            self.sel.fill(-1)
            self.cnt.fill(0)
        elif selection == "disjoint":
            self.sel.fill(-1)
            for t in range(self.nt):
                nc = (b.seq_of_token(t)[1] + 1) // cfg.compress_ratio
                ids = np.arange(t % 4, nc, 4, dtype=np.int32)[:cfg.block_topk]
                self.cnt[t] = len(ids)
                self.sel[t, :len(ids)] = ids
        elif selection == "holes":
            # Negative entries INSIDE count must not form addresses; invalid padding
            # OUTSIDE count deliberately contains huge positive IDs to catch overreads.
            for t in range(self.nt):
                count = int(self.cnt[t])
                self.sel[t, :count:2] = -1
                self.sel[t, count:] = np.iinfo(np.int32).max
        elif selection == "overcount":
            self.cnt[:] = cfg.block_topk + 3  # out-of-contract robustness, bounded like scalar
        elif selection != "random":
            raise ValueError(selection)
        self.inputs = [tensor(a) for a in (self.storage, self.kc.reshape(-1), self.vc.reshape(-1),
            b.past_lens, b.subsequence_begins, b.block_indices, b.block_indices_begins,
            self.sel, self.cnt)]
        self.output_size = self.nt * self.hq * dv

    def positions(self, t):
        r = self.cfg.compress_ratio
        abs_pos = self.batch.seq_of_token(t)[1]
        blocks = self.sel[t, :min(int(self.cnt[t]), self.cfg.block_topk)]
        return np.asarray([int(b) * r + j for b in blocks if b >= 0 for j in range(r)] +
                          list(range((abs_pos + 1) // r * r, abs_pos + 1)), dtype=np.int32)

    def reference(self):
        """Independent stable batch softmax: all arithmetic arrays explicitly FP32.

        Unlike the GPU, no online recurrence and no partition reduction. Vectorized
        selected positions keep B4/past16K CPU checks within the remote time limit.
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
            for h in range(self.hq):
                kh = h // cfg.n_rep
                k = self.kc[pages, kh, tips].astype(np.float32)
                v = self.vc[pages, kh, tips].astype(np.float32)
                q = self.query[t, h * self.qhs:h * self.qhs + cfg.k_head_size].astype(np.float32)
                scores = k @ (q * np.float32(cfg.scale))
                e = np.exp(scores - scores.max())
                value = (e @ v) / e.sum(dtype=np.float32)
                if cfg.has_output_gate:
                    gate = self.query[t, h * self.qhs + self.gd:h * self.qhs + self.gd + self.dv].astype(np.float32)
                    value *= np.float32(1) / (np.float32(1) + np.exp(-gate))
                out[t, h] = value
        return out.reshape(self.nt, self.hq * self.dv)


class Call:
    def __init__(self, c, partitions, *, guard=False):
        self.c, self.guard = c, guard
        self.impl = implementation(c.cfg, partitions)
        self.out = tensor(np.full(c.output_size + 64, np.nan, np.float16))
        self.na, self.nm = self.impl.scratch_elements(c.nt)
        self.acc = tensor(np.full(self.na + 64, np.nan, np.float32))
        self.ml = tensor(np.full(self.nm + 64, np.nan, np.float32))

    def enqueue(self):
        c = self.c
        self.impl.enqueue(c.inputs, self.out, c.nt, c.batch.num_seqs, c.qbase, c.pitch,
                          c.qhs, c.gd, partial_acc=self.acc, partial_ml=self.ml,
                          guard_items=self.guard)

    def result(self):
        c = self.c
        out = self.out.numpy().reshape(-1)
        assert np.isnan(out[c.output_size:]).all(), "output guard overwritten"
        if self.impl.partitions is not None:
            acc, ml = self.acc.numpy().reshape(-1), self.ml.numpy().reshape(-1)
            assert np.isnan(acc[self.na:]).all() and np.isnan(ml[self.nm:]).all(), "scratch guard overwritten"
            assert np.isfinite(acc[:self.na]).all() and np.isfinite(ml[:self.nm]).all(), "missing/nonfinite partial"
            rows = ml[:self.nm].reshape(-1, 2)
            assert (rows[:, 1] >= 0).all()
            empty = rows[:, 1] == 0
            np.testing.assert_array_equal(acc[:self.na].reshape(-1, c.dv)[empty], 0)
        return out[:c.output_size].reshape(c.nt, c.hq * c.dv).astype(np.float32)


def check(c, *, guard=False, label="case"):
    from clops import cl

    ref = c.reference()
    calls = [Call(c, p, guard=guard) for p in (None, *PARTITIONS)]
    baseline = None
    errors = {}
    for p, call in zip((None, *PARTITIONS), calls):
        call.enqueue()
        cl.finish()
        got = call.result()
        if label == "all-zero-query" and p is not None:
            # With Q=0, exp(score)=1: l is EXACTLY the number of positions.
            # This independently checks slot partition boundaries and tail ownership.
            counts = call.ml.numpy()[:call.nm].reshape(c.nt, c.hq, p, 2)[..., 1]
            for t in range(c.nt):
                selected = min(int(c.cnt[t]), c.cfg.block_topk)
                tail = (c.batch.seq_of_token(t)[1] + 1) % c.cfg.compress_ratio
                for part in range(p):
                    slots = c.sel[t, selected * part // p:selected * (part + 1) // p]
                    n = int((slots >= 0).sum()) * c.cfg.compress_ratio + (tail if part == p - 1 else 0)
                    np.testing.assert_array_equal(counts[t, :, part], n)
        assert np.isfinite(got).all(), (label, p)
        if baseline is None:
            baseline = got
        er = float(np.max(np.abs(got - ref), initial=0))
        eb = float(np.max(np.abs(got - baseline), initial=0))
        np.testing.assert_allclose(got, ref, atol=2e-3, rtol=2e-3, err_msg=f"{label}/P{p}/NumPy")
        np.testing.assert_allclose(got, baseline, atol=2e-3, rtol=2e-3, err_msg=f"{label}/P{p}/scalar")
        for t in range(c.nt):
            if not len(c.positions(t)):
                np.testing.assert_array_equal(got[t], 0)
        errors[str(p)] = {"numpy": er, "scalar": eb}
        WORST["numpy"], WORST["scalar"] = max(WORST["numpy"], er), max(WORST["scalar"], eb)
    print("PASS " + json.dumps(dict(case=label, tokens=c.nt, errors=errors)), flush=True)
    return calls, errors


def check_source_copy():
    root = Path(__file__).parent
    source = (root / "kernels/qsa_sparse_attention.cm").read_text()
    copied = (root / "include/qsa_split_span.hpp").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == "a731a8ac4367513101b13ec4a3b96a63d4f9898463dfc2328a75f36c45b15081"
    for fn in ("qsa_softmax_step", "qsa_attend_span"):
        def body(text):
            start = text.index("CM_INLINE void " + fn)
            opening = text.index("{", start)
            depth = 1
            end = opening + 1
            while depth:
                depth += (text[end] == "{") - (text[end] == "}")
                end += 1
            return text[start:end]
        assert body(source) == body(copied), f"scalar helper drift: {fn}"
    print("PASS scalar SHA256 and private helper exact source identity", flush=True)


def check_reducer(cfg):
    from clops import cl

    c = Case(cfg, [(0, 1)])
    call = Call(c, 4, guard=True)
    # Synthetic unnormalized states: huge maxima need subtraction; empty NaN
    # states MUST be skipped before exp/acc reads. Gate is still taken from Q.
    for all_empty in (False, True):
        a = np.full((c.hq, 4, c.dv), np.nan, np.float32)
        ml = np.full((c.hq, 4, 2), np.nan, np.float32)
        ml[:, :, 1] = 0
        ml[:, 3, 0] = -np.inf
        expected = np.zeros((c.hq, c.dv), np.float32)
        if not all_empty:
            a[:, 1], a[:, 2] = 2, 9
            ml[:, 1] = [1000, 2]
            ml[:, 2] = [999, 3]
            expected[:] = (2 + 9 * np.exp(np.float32(-1))) / (2 + 3 * np.exp(np.float32(-1)))
            if cfg.has_output_gate:
                for h in range(c.hq):
                    g = c.query[0, h * c.qhs + c.gd:h * c.qhs + c.gd + c.dv].astype(np.float32)
                    expected[h] *= 1 / (1 + np.exp(-g))
        # Retain explicit ownership until finish rather than relying on clops' pool.
        device_a, device_ml = tensor(a), tensor(ml)
        call.impl.reducer.enqueue("qsa_sparse_attention_split_reduce", [c.hq + 1, 2, 1], [1, 1, 1],
            c.inputs[0], device_a, device_ml, call.out, 1, c.qbase, c.pitch, c.qhs, c.gd)
        cl.finish()
        got = call.out.numpy().reshape(-1)
        assert np.isnan(got[c.output_size:]).all()
        np.testing.assert_allclose(got[:c.output_size].reshape(c.hq, c.dv), expected, atol=1e-3, rtol=1e-3)
        if all_empty:
            np.testing.assert_array_equal(got[:c.output_size], 0)
    print("PASS reducer large maxima, NaN/-inf empty states, all-empty zero and one gate", flush=True)


def validate():
    from clops import cl

    check_source_copy()
    cfg = qsa_config.load()
    mixed = [(0, 0), (0, 1), (1, 3), (3, 5), (14, 7), (15, 17), (31, 3), (63, 1), (0, 0)]
    cases = [("mixed-layout", cfg, mixed, "random", False),
             ("disjoint", replace(cfg, block_topk=4), [(96, 9)], "disjoint", False),
             ("invalid-slots-and-padding", replace(cfg, block_topk=8), mixed, "holes", False),
             ("tail-only-empty", cfg, mixed, "empty", False),
             ("K0", replace(cfg, block_topk=0), mixed, "empty", False),
             ("K1-P-greater-count", replace(cfg, block_topk=1), mixed, "random", False),
             ("all-zero-query", cfg, mixed, "random", True),
             ("ungated", replace(cfg, has_output_gate=False), mixed, "random", False),
             ("different-Dh-Dv", replace(cfg, k_head_size=64, v_head_size=128, scale=0.125), mixed, "random", False),
             ("all-zero-q-lengths", cfg, [(0, 0), (16, 0)], "random", False),
             ("no-sequences", cfg, [], "random", False),
             ("count-bound-robustness", replace(cfg, block_topk=4), [(96, 2)], "overcount", False)]
    for r, page in ((2, 8), (4, 16), (8, 32), (16, 64)):
        conf = replace(cfg, compress_ratio=r, pa_block_size=page, block_topk=4)
        cases.append((f"r{r}-page{page}-all-remainders", conf,
                      [(page * 2 + rem, 1) for rem in range(r)] + [(0, 0)], "random", False))
    for batch in (1, 4, 16):
        for rem in range(4):
            cases.append((f"B{batch}-pastmod{rem}", cfg, [(128 + rem, 1)] * batch, "random", False))
    for name, conf, seqs, selection, zero in cases:
        check(Case(conf, seqs, selection=selection, zero_query=zero), guard=True, label=name)
    check_reducer(cfg)
    # Non-power-of-two P and uneven count/P division are supported, not timed.
    c = Case(replace(cfg, block_topk=7), [(64, 3), (0, 0), (67, 1)], seed=29)
    base = Call(c, None, guard=True)
    base.enqueue()
    cl.finish()
    scalar, ref = base.result(), c.reference()
    for p in (3, 15):
        call = Call(c, p, guard=True)
        call.enqueue()
        cl.finish()
        got = call.result()
        np.testing.assert_allclose(got, ref, atol=2e-3, rtol=2e-3)
        np.testing.assert_allclose(got, scalar, atol=2e-3, rtol=2e-3)
        er, eb = float(np.abs(got - ref).max()), float(np.abs(got - scalar).max())
        WORST["numpy"], WORST["scalar"] = max(WORST["numpy"], er), max(WORST["scalar"], eb)
        print(f"PASS uneven K7/P{p} numpy={er:.9g} scalar={eb:.9g}", flush=True)
    # Runtime helper rejection and no-op checks, not just source inspection.
    impl = implementation(cfg, 4)
    impl.enqueue([], None, 0, 0, 0, 0, 0, 0)
    assert implementation(cfg, None).partitions is None
    for p in (0, -1, 17, True, 1.5):
        try:
            Q3Attention(cfg, partitions=p)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid partitions {p}")
    for bad in (replace(cfg, pa_block_size=15), replace(cfg, k_head_size=63)):
        try:
            Q3Attention(bad, partitions=4)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid geometry")
    print(f"ALL {len(cases)} Q3 SPLIT CASES + P3/P15 PASSED; WORST {json.dumps(WORST)}", flush=True)


def benchmark(args):
    from clops import cl

    cfg = qsa_config.load()
    flusher = H.CacheFlusher() if args.flush else None
    cl.profiling(True)
    for batch in args.batches:
        for past in args.sizes:
            c = Case(cfg, [(past, 1)] * batch, padded=False)
            calls, errors = check(c, label=f"bench-B{batch}-past{past}")
            for i in range(args.warmup):
                for call in (calls if i % 2 == 0 else calls[::-1]):
                    call.enqueue()
                cl.finish()
            events, hosts = [[] for _ in calls], [[] for _ in calls]
            for i in range(args.samples):
                for j in (range(len(calls)) if i % 2 == 0 else reversed(range(len(calls)))):
                    if flusher:
                        flusher.flush()  # outside timing; NEVER between the two Q3 dispatches
                    t0 = perf_counter()
                    calls[j].enqueue()
                    ns = cl.finish()
                    hosts[j].append((perf_counter() - t0) * 1000)
                    assert len(ns) == (1 if j == 0 else 2), ns
                    events[j].append(np.asarray(ns, np.float64) * 1e-6)
            row = dict(batch=batch, past=past, cache="cold96MiB" if args.flush else "no-flush",
                       warmup=args.warmup, samples=args.samples, errors=errors, variants={})
            for j, name in enumerate(("scalar", *(f"P{p}" for p in PARTITIONS))):
                a = np.asarray(events[j])
                total = a.sum(axis=1)
                row["variants"][name] = dict(mean_ms=float(total.mean()),
                    median_ms=float(np.median(total)), p95_ms=float(np.percentile(total, 95)),
                    stage_mean_ms=a.mean(axis=0).tolist(), host_mean_ms=float(np.mean(hosts[j])),
                    samples_ms=total.tolist(), host_samples_ms=hosts[j],
                    scratch_bytes=(calls[j].na + calls[j].nm) * 4)
                calls[j].result()  # guards remain intact after all warmups/samples
            print("BENCH " + json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--flush", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 16384])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    args = ap.parse_args()
    if args.warmup < 0 or min(args.samples, *args.sizes, *args.batches) < 1:
        ap.error("warmup >= 0, samples/sizes/batches >= 1 required")
    from clops import cl

    print("DEVICE", cl.dev_info(), flush=True)
    if not args.benchmark_only:
        validate()
    if args.benchmark or args.benchmark_only:
        benchmark(args)


if __name__ == "__main__":
    main()