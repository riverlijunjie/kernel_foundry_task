"""Strict standalone Q2 dense-bypass regression; GPU execution is remote-only.

Default: small mixed batches, exact metadata/padding, score-only contracts and
instrumented finalizer score-read counts. --benchmark adds alternating A/B GPU
event timing (100 warmups + 20 samples per variant, no cache flush by default).
No Q1/Q3 dispatch or model/integrated speedup is implied by these Q2 timings.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json

import numpy as np

import qsa_config
import qsa_harness as H


CONFIGS = {
    "r4-p16-k512": (4, 16, 512),
    "r4-p16-k1": (4, 16, 1),
    "r4-p16-k0": (4, 16, 0),
    "r2-p16-k1": (2, 16, 1),
    "r8-p32-k512": (8, 32, 512),
    "r4-p64-k512": (4, 64, 512),
}
NAMES = {"partition": "qsa_score_partition", "dpas": "qsa_score_tile_dpas",
         "fused": "qsa_score_topk_fused"}
GUARD = -777777
WG = 16
PARTITION = 512
_KERNELS = {}


def defines(cfg, name, variant):
    extra = {"QSA_SCORE_WG": WG, "QSA_PARTITION_BLOCKS": PARTITION}
    if variant != "default":
        extra.update(QSA_DENSE_BYPASS=int(variant == "optimized"),
                     QSA_DENSE_SCORE_BYPASS=int(variant == "optimized"))
    return cfg.jit(name, **extra)


def replace_once(src, old, new):
    if src.count(old) != 1:
        raise AssertionError(f"Instrumentation source drift: {old!r}")
    return src.replace(old, new, 1)


def kernel(cfg, name, variant, instrument=False):
    from clops import cl

    jit = defines(cfg, name, variant)
    key = (tuple(jit.items()), instrument)
    if key not in _KERNELS:
        if not instrument:
            _KERNELS[key] = H.build_kernel(name + ".cm", jit)
        else:
            # Instrument the ACTUAL flattened radix implementation, not a copy
            # of the algorithm. Only test builds gain an extra argument. Every
            # original scores[i] load increments this row's private counter;
            # one CM work-item owns a row, so no atomics are required.
            src = H.flatten_source(name + ".cm", jit)
            if src.count("scores[i]") != 2:
                raise AssertionError("Radix score-load sites changed")
            src = src.replace("scores[i]", "qsa_counted_score(scores, i, reads)")
            marker = "CM_INLINE uint radix_select_topk("
            helper = """CM_INLINE float qsa_counted_score(const float* scores ATTR,
                    uint i, int32_t* reads ATTR) {
                reads[0] += 1;
                return scores[i];
            }
            """
            src = replace_once(src, marker, helper + marker)
            src = replace_once(src, "uint k, int32_t* out ATTR)",
                               "uint k, int32_t* out ATTR, int32_t* reads ATTR)")
            src = replace_once(src, "n_complete, K, out);",
                               "n_complete, K, out, score_reads + token_idx);")
            src = replace_once(src, "uint32_t max_blocks) {",
                               "uint32_t max_blocks, int32_t* score_reads ATTR) {")
            _KERNELS[key] = cl.kernels(src, H.QSA_BUILD_OPTIONS)
    return _KERNELS[key]


class Case:
    def __init__(self, cfg, seqs, ties=False):
        from clops import cl

        self.cfg = cfg
        self.batch = H.make_batch(seqs, cfg.pa_block_size, seed=13)
        self.nt = self.batch.num_tokens
        self.k = cfg.block_topk
        iq, summ, self.mb, self.nc = H.make_indexer_inputs(cfg, self.batch, seed=7)
        if ties:
            summ.fill(0)  # Exact ReLU ties, including the sparse threshold.
        self.dense = self.nc <= self.k
        self.inputs = [cl.tensor(a) for a in (
            iq.reshape(-1), summ, self.batch.past_lens, self.batch.subsequence_begins,
            self.batch.block_indices, self.batch.block_indices_begins)]
        tile_map, self.tiles = H.make_score_tile_map(self.batch)
        self.tile_map = cl.tensor(tile_map)
        # Distinct poison rows: no score bypass is allowed to touch dense rows.
        self.poison = np.full((self.nt, self.mb), np.nan, np.float32)
        self.poison[1::2] = -12345.0


class Pipeline:
    def __init__(self, case, backend, variant):
        from clops import cl

        self.c, self.backend, self.variant = case, backend, variant
        self.scorer = kernel(case.cfg, NAMES[backend], variant)
        self.finalizer = kernel(case.cfg, "qsa_topk_finalization", variant)
        self.scores = cl.tensor(case.poison)
        self.sel = cl.tensor(np.full(case.nt * case.k + 16, GUARD, np.int32))
        self.cnt = cl.tensor(np.full(case.nt + 16, GUARD, np.int32))

    def score(self):
        c = self.c
        if self.backend == "dpas":
            self.scorer.enqueue(NAMES[self.backend], [(c.tiles + 1) * WG, 1, 1], [WG, 1, 1],
                                *c.inputs, c.tile_map, self.scores, c.tiles, c.mb)
        else:
            parts = max(1, (c.mb + PARTITION - 1) // PARTITION)
            extra = [self.sel, self.cnt] if self.backend == "fused" else []
            self.scorer.enqueue(NAMES[self.backend],
                                [c.nt + 3, 1, parts if self.backend == "partition" else 1],
                                [1, 1, 1], *c.inputs, self.scores, *extra,
                                c.nt, c.batch.num_seqs, c.mb)

    def finalize(self, instrument=False):
        from clops import cl

        c = self.c
        extra = []
        finalizer = self.finalizer
        if instrument:
            self.reads = cl.tensor(np.zeros(c.nt, np.int32))
            extra = [self.reads]
            finalizer = kernel(c.cfg, "qsa_topk_finalization", self.variant, True)
        finalizer.enqueue("qsa_topk_finalization", [c.nt + 3, 1, 1], [1, 1, 1],
                          self.scores, c.inputs[2], c.inputs[3], self.sel, self.cnt,
                          c.nt, c.batch.num_seqs, c.mb, *extra)

    def enqueue(self):
        self.score()
        if self.backend != "fused":
            self.finalize()

    def metadata(self):
        c = self.c
        sel, cnt = self.sel.numpy(), self.cnt.numpy()
        np.testing.assert_array_equal(sel[c.nt * c.k:], GUARD)
        np.testing.assert_array_equal(cnt[c.nt:], GUARD)
        return sel[:c.nt * c.k].reshape(c.nt, c.k), cnt[:c.nt]


def expected(scores, nc, k):
    out = np.full((len(nc), k), -1, np.int32)
    counts = np.minimum(nc, k).astype(np.int32)
    for t, n in enumerate(nc):
        ids = np.arange(int(n))
        # Deterministic score-descending, ID-ascending tie-break; then ascending
        # output IDs, as required by Q3. Handles K=0 without a [-0:] slice.
        chosen = np.lexsort((ids, -scores[t, :n]))[:counts[t]]
        out[t, :counts[t]] = np.sort(chosen)
    return out, counts


def assert_metadata(pipeline, ref):
    for got, want in zip(pipeline.metadata(), ref):
        np.testing.assert_array_equal(got, want)


def validate(cfg, label):
    from clops import cl

    r, k = cfg.compress_ratio, cfg.block_topk
    edge = (k + 1) * r
    # Empty sequences, zero complete blocks, wholly dense/sparse tiles, a
    # sequence-local mixed tile, partial tiles, and the exact dense boundary.
    seqs = [(0, 0), (0, 1), (0, 19), (max(0, edge - 10), 33),
            ((k + 3) * r, 3), (edge - 2, 1), (edge - 1, 1), (9, 0)]
    for ties in (False, True):
        c = Case(cfg, seqs, ties)
        if r == 4 and k == 512:
            assert c.nc[-2:].tolist() == [512, 513]  # Visible 2051 / 2052.
            assert c.dense[-2:].tolist() == [True, False]
        for backend in NAMES:
            base = Pipeline(c, backend, "baseline")
            base.enqueue()
            cl.finish()
            scores = base.scores.numpy()
            ref = expected(scores, c.nc, k)
            assert_metadata(base, ref)
            for variant in ("default", "optimized"):
                p = Pipeline(c, backend, variant)
                p.enqueue()
                cl.finish()
                assert_metadata(p, ref)
                got = p.scores.numpy()
                bypass_scores = variant == "optimized" or backend == "fused"
                if bypass_scores:
                    np.testing.assert_array_equal(got[c.dense].view(np.uint32),
                                                  c.poison[c.dense].view(np.uint32))
                    np.testing.assert_array_equal(got[~c.dense], scores[~c.dense])
                else:
                    # No JIT flags: standalone score-output contract, including
                    # DPAS's existing stores into invalid/padded score columns.
                    np.testing.assert_array_equal(got, scores)
            if backend == "partition":
                # Count every actual radix score load for the poisoned pipeline.
                for p in (base, Pipeline(c, backend, "optimized")):
                    if p is not base:
                        p.score()
                    p.finalize(instrument=True)
                    cl.finish()
                    assert_metadata(p, ref)
                    reads = p.reads.numpy()
                    needs_reads = (c.nc > 0) & (k > 0)
                    if p.variant == "optimized":
                        needs_reads &= ~c.dense
                    np.testing.assert_array_equal(reads[~needs_reads], 0)
                    assert np.all(reads[needs_reads] > 0), reads
                    print(f"READS {label} ties={ties} {p.variant}: "
                          f"dense={int(reads[c.dense].sum())} "
                          f"sparse={int(reads[~c.dense].sum())}", flush=True)
        print(f"PASS pipeline {label} ties={ties} rows={c.nt} "
              f"dense={int(c.dense.sum())} backends=partition,dpas,fused", flush=True)

    # Additional finalizer-only tests avoid assuming all score distributions
    # come from random indexer tensors: negative values, ties and near-ties.
    c = Case(cfg, seqs)
    for distribution in ("zero", "equal", "ties", "near"):
        rng = np.random.default_rng(27)
        scores = rng.integers(-2, 3, (c.nt, c.mb)).astype(np.float32)
        if distribution == "zero":
            scores.fill(0)
        elif distribution == "equal":
            scores.fill(0.5)
        elif distribution == "near":
            scores = np.float32(1) + scores * np.spacing(np.float32(1))
        ref = expected(scores, c.nc, k)
        for variant in ("baseline", "optimized"):
            p = Pipeline(c, "partition", variant)
            supplied = scores.copy()
            if variant == "optimized":
                supplied[c.dense] = c.poison[c.dense]
            p.scores = cl.tensor(supplied)
            p.finalize()
            cl.finish()
            assert_metadata(p, ref)
        print(f"PASS finalizer {label} distribution={distribution}", flush=True)


def benchmark(cfg, args):
    from clops import cl

    flusher = H.CacheFlusher() if args.flush else None
    for mode in ("prefill", "decode"):
        for size in args.sizes:
            seqs = [(0, size)] if mode == "prefill" else [(size, 1)]
            c = Case(cfg, seqs)
            for backend in (("dpas", "fused") if mode == "prefill" else ("partition", "fused")):
                pair = [Pipeline(c, backend, v) for v in ("baseline", "optimized")]
                for p in pair:
                    p.enqueue()
                cl.finish()
                ref = pair[0].metadata()
                assert_metadata(pair[1], ref)
                cl.profiling(True)
                for i in range(args.warmup):
                    for j in (i % 2, 1 - i % 2):
                        pair[j].enqueue()
                    cl.finish()
                events = [[], []]
                for i in range(args.samples):
                    for j in (i % 2, 1 - i % 2):
                        if flusher:
                            flusher.flush()
                        pair[j].enqueue()
                        ns = np.asarray(cl.finish(), dtype=np.float64).reshape(-1)
                        assert len(ns) == (1 if backend == "fused" else 2), ns
                        events[j].append(ns * 1e-6)
                row = {"mode": mode, "size": size, "backend": backend,
                       "dense_rows": int(c.dense.sum()), "rows": c.nt,
                       "warmup": args.warmup, "samples": args.samples, "flush": args.flush}
                for j, variant in enumerate(("baseline", "optimized")):
                    arr = np.asarray(events[j])
                    total = arr.sum(axis=1)
                    row[variant] = {"mean_ms": float(total.mean()),
                                    "median_ms": float(np.median(total)),
                                    "p95_ms": float(np.percentile(total, 95)),
                                    "stage_mean_ms": arr.mean(axis=0).tolist()}
                row["ratio"] = row["baseline"]["mean_ms"] / row["optimized"]["mean_ms"]
                print("BENCH " + json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", choices=list(CONFIGS), nargs="+", default=list(CONFIGS))
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--flush", action="store_true", help="flush 96 MiB before each timed pipeline")
    args = ap.parse_args()
    if args.warmup < 0 or args.samples < 1 or any(n < 1 for n in args.sizes):
        ap.error("warmup >= 0, samples >= 1 and sizes >= 1 required")
    from clops import cl

    print("DEVICE", cl.dev_info(), flush=True)
    for label in args.configs:
        r, page, k = CONFIGS[label]
        cfg = replace(qsa_config.load(pa_block_size=page, block_topk=k), compress_ratio=r)
        validate(cfg, label)
    if args.benchmark:
        benchmark(qsa_config.load(), args)
    print("ALL DENSE BYPASS CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()