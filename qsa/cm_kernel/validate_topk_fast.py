"""Remote B580/PTL B390 SAME-score finalizer tests and alternating event benchmarks.

Finite sparse inputs only; ordered uint oracle intentionally distinguishes -0/+0.
No model/plugin accuracy claim. All allocations, references, reads and flushes
are outside timing. --benchmark-only still validates every timed input/output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from operator import index
from pathlib import Path

import numpy as np

import qsa_config
import qsa_harness as H
from qsa_score_chunked import topk_cache_capacity
from test_qsa_pipeline import Buffer, tensor, stats

CACHE = {}
NAMES = {"scalar": "qsa_topk_finalization", "coop": "qsa_topk_cooperative",
         "fast": "qsa_topk_radix_fast", "fast-wg": "qsa_topk_radix_wg",
         "fast-wg16-cached": "qsa_topk_radix_wg"}


def oracle(scores, nc, k):
    """Independent CPU key sort; no GPU threshold/cache/worker algorithm reuse."""
    out = np.full((len(nc), k), -1, np.int32)
    for t, n in enumerate(nc):
        count = min(int(n), k)
        if not count:
            continue
        if count == n:
            out[t, :count] = np.arange(count)
            continue
        a = scores[t, :n]
        if not np.isfinite(a).all():
            raise ValueError("sparse top-k requires finite scores (NaN/Inf rejected)")
        bits = a.view(np.uint32)
        keys = bits ^ np.where(bits & np.uint32(0x80000000),
                              np.uint32(0xffffffff), np.uint32(0x80000000))
        ids = np.arange(n)
        out[t, :count] = np.sort(np.lexsort((ids, ~keys))[:count])
    return out, np.minimum(nc, k).astype(np.int32)


class Finalizers:
    def __init__(self, scores, seqs, k=512, *, chunked=True, variants=tuple(NAMES),
                 capacity=256, force_compact=False, wg=16, key_capacity=None):
        self.cfg = qsa_config.load(block_topk=k)
        self.seqs, self.k, self.chunked = seqs, k, chunked
        self.wg = wg
        self.nt, self.mb = scores.shape
        # Independent of the single-work-item fast finalizer's compact capacity.
        cache_cap = topk_cache_capacity(self.mb) if key_capacity is None else index(key_capacity)
        if not 0 <= cache_cap <= 2048 or cache_cap % 64:
            raise ValueError("key_capacity must be 0..2048, multiple of 64")
        begins = np.cumsum([0] + [q for _, q in seqs]).astype(np.int32)
        assert begins[-1] == self.nt
        self.nc = np.concatenate([((p + np.arange(q) + 1) // self.cfg.compress_ratio)
                                  for p, q in seqs]).astype(np.int32)
        assert np.all(self.nc <= self.mb)
        self.want = oracle(scores, self.nc, k)
        self.host_scores = scores.copy()
        self.scores = Buffer(scores)
        self.meta = [tensor(np.asarray([p for p, _ in seqs], np.int32)), tensor(begins)]
        self.variants = variants
        self.widths = {v: 16 if v == "fast-wg16-cached" else
                   (wg if v in ("coop", "fast-wg") else 1) for v in variants}
        self.key_capacities = {v: cache_cap if v == "fast-wg16-cached" else 0 for v in variants}
        self.outputs = {v: (Buffer(np.full((self.nt, k), -777, np.int32)),
                            Buffer(np.full(self.nt, -777, np.int32))) for v in variants}
        self.kernels = {}
        for v in variants:
            width = self.widths[v]
            key = (v, k, chunked, capacity, force_compact, width, self.key_capacities[v])
            if key not in CACHE:
                name = NAMES[v]
                cache_jit = ({"QSA_TOPK_CACHE_CAPACITY": cache_cap}
                             if v == "fast-wg16-cached" else {})
                CACHE[key] = H.build_kernel(name + ".cm", self.cfg.jit(name,
                    QSA_SCORE_ROW_CHUNK=int(chunked), QSA_TOPK_WG=width, QSA_DENSE_BYPASS=1,
                    QSA_TOPK_CAPACITY=capacity, QSA_TOPK_FORCE_COMPACT=int(force_compact), **cache_jit))
            self.kernels[v] = CACHE[key]

    def enqueue(self, v, begin=0, end=None, scores=None, guard=0):
        end = self.nt if end is None else end
        wg = self.widths[v]
        sel, cnt = self.outputs[v]
        self.kernels[v].enqueue(NAMES[v], [(end - begin + guard) * wg, 1, 1], [wg, 1, 1],
            self.scores.device if scores is None else scores, *self.meta, sel.device, cnt.device,
            self.nt, len(self.seqs), self.mb, *((begin, end) if self.chunked else ()))

    def check(self):
        for v, (sel, cnt) in self.outputs.items():
            np.testing.assert_array_equal(sel.read(), self.want[0], err_msg=v)
            np.testing.assert_array_equal(cnt.read(), self.want[1], err_msg=v)
        np.testing.assert_array_equal(self.scores.read().view(np.uint32), self.host_scores.view(np.uint32))

    def validate(self, cap=None):
        from clops import cl
        for begin in range(0, self.nt, cap or self.nt):
            end = min(self.nt, begin + (cap or self.nt))
            local = Buffer(self.host_scores[begin:end].copy()) if cap else self.scores
            before = {v: (s.read(), c.read()) for v, (s, c) in self.outputs.items()}
            for v in self.variants:
                self.enqueue(v, begin, end, local.device, guard=2)
            cl.finish()
            np.testing.assert_array_equal(local.read().view(np.uint32),
                                          self.host_scores[begin:end].view(np.uint32))
            for v, (s, c) in self.outputs.items():
                for got, want, old in zip((s.read(), c.read()), self.want, before[v]):
                    np.testing.assert_array_equal(got[begin:end], want[begin:end], err_msg=v)
                    np.testing.assert_array_equal(got[:begin], old[:begin])
                    np.testing.assert_array_equal(got[end:], old[end:])
        self.check()


def patterns(rng, n):
    neighbors = np.array([0x3f7fffff, 0x3f800000, 0x3f800001, 0xbf800000,
                          0x00000000, 0x80000000, 1, 0x80000001], np.uint32).view(np.float32)
    # A zero-only row puts signed-zero ordering at the actual kth boundary.
    zeros = np.zeros(n, np.float32)
    zeros[::2] = -0.0
    return [rng.normal(size=n).astype(np.float32), np.full(n, 3, np.float32),
            np.zeros(n, np.float32), -rng.uniform(0, 9, n).astype(np.float32),
            rng.choice(neighbors, n),
            (np.uint32(0x3f800000) + rng.integers(0, 1024, n, dtype=np.uint32)).view(np.float32), zeros]


def regression():
    from clops import cl
    rng, cases = np.random.default_rng(938), 0
    try:
        oracle(np.array([[np.nan, 1]], np.float32), np.array([2]), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN contract must be rejected")
    for k in (0, 1, 512):
        for n in (513, 1024, 4096, 8192, 16384, 32768, 32769):
            # Independent sparse rows; plus dense/K0 poisoned rows and empty sequences.
            data_rows = patterns(rng, n)
            seqs = [(7, 0)] + [(4 * n - 1, 1)] * len(data_rows) + [(0, 3), (3, 2)]
            if k:
                seqs += [(4 * k - 1, 1)]
            mb = H.ceil8(n) + 8  # deliberately 8 mod 16 for non-16 row alignment
            scores = np.full((sum(q for _, q in seqs), mb), np.nan, np.float32)
            for row, data in enumerate(data_rows):
                if k:
                    scores[row, :n] = data
            # Sparse K1 row at abs_pos=4 also has one complete block: still dense.
            for chunked in (False, True):
                f = Finalizers(scores, seqs, k, chunked=chunked)
                f.validate(cap=3 if chunked else None)
                f.validate(cap=5 if chunked else None)  # repeat, different partial boundaries
                cases += 1
            print(f"PASS finite/poison/guards K={k} n={n} both-ABI", flush=True)
    # Mixed causal rows with unequal valid counts, partial chunks and signed zero ties.
    seqs = [(0, 17), (2046, 19), (32767, 5), (1, 0), (65535, 3)]
    scores = rng.normal(size=(44, 16392)).astype(np.float32)
    scores[:, ::3] = -0.0
    scores[:, 1::3] = 0.0
    for k in (0, 1, 512):
        f = Finalizers(scores, seqs, k)
        f.validate(cap=17)
        cases += 1
    # Deliberately overflow the tiny cache on the first full interval; verify
    # the compiled recovery branch against the same scores, not a truncated oracle.
    for capacity, forced in ((0, False), (64, True), (256, True), (512, False)):
        f = Finalizers(scores, seqs, 512, capacity=capacity, force_compact=forced)
        f.validate(cap=3)
        cases += 1
    for k in (0, 1, 512):
        f = Finalizers(scores, seqs, k, wg=8)
        f.validate(cap=17)
        cases += 1
    cl.finish()
    print(f"ALL {cases} FAST FINALIZER CONFIGURATIONS PASSED", flush=True)


def cached_boundary_tests():
    """Same compiled cache sees fit/overflow rows, nonzero row origins and tails.

    K2049 on the large case forces tied selections across worker boundaries.
    CAP64 catches accidental non-64-aligned small-shape cache addressing too.
    These are correctness cases, not claims of CM compilation or performance.
    """
    cases = 0
    for capacity in (64, 2048):
        boundary = 16 * capacity
        ns = [63, 64, 65, boundary - 1, boundary, boundary + 1]
        k = 512 if capacity == 64 else 2049
        mb = H.ceil8(max(ns)) + 8
        seqs = [(7, 0)] + [(4 * n - 1, 1) for n in ns] + [(0, 0)]
        scores = np.full((len(ns), mb), np.nan, np.float32)
        for row, n in enumerate(ns):
            if n > k:
                # Highest ID must survive, but equal keys favor lowest IDs.
                scores[row, :n] = 0.0
                scores[row, :n:2] = -0.0
                scores[row, n - 1] = 1.0
        for chunked in (False, True):
            f = Finalizers(scores, seqs, k, chunked=chunked,
                           variants=("fast-wg", "fast-wg16-cached"), key_capacity=capacity)
            f.validate(cap=2 if chunked else None)
            f.validate(cap=4 if chunked else None)
            cases += 1
        print(f"PASS WG16 cache CAP={capacity} n={boundary}/{boundary + 1} fit/fallback both-ABI", flush=True)
    print(f"ALL {cases} WG16 CACHE BOUNDARY CONFIGURATIONS PASSED", flush=True)


def helper_tests():
    """Fast opt-in vs old helper on identical GPU producer scores, reuse + guards."""
    from clops import cl
    from qsa_score_chunked import ChunkedQ2
    cases = 0
    for k in (0, 1, 512):
        cfg = qsa_config.load(block_topk=k)
        batch = H.make_batch([(0, 17), (2046, 19), (65536, 3), (7, 0)], cfg.pa_block_size)
        iq, summary, mb, nc = H.make_indexer_inputs(cfg, batch, seed=341)
        for backend in ("partition", "dpas"):
            for bypass in (False, True):
                base = ChunkedQ2(cfg, batch, mb, backend=backend, dense_bypass=bypass,
                                 unchunked=True)
                for data in (summary, np.zeros_like(summary)):
                    inputs = [tensor(a) for a in (iq, data, batch.past_lens, batch.subsequence_begins,
                              batch.block_indices, batch.block_indices_begins)]
                    bs, bc = Buffer(np.full((batch.num_tokens, k), -777, np.int32)), Buffer(np.full(batch.num_tokens, -777, np.int32))
                    base.enqueue(inputs, bs.device, bc.device, guard_groups=2)
                    cl.finish()
                    # Independent ordered-key oracle on identical producer scores,
                    # not just agreement with another GPU finalizer.
                    want, counts = oracle(base.read(), nc, k)
                    np.testing.assert_array_equal(bs.read(), want)
                    np.testing.assert_array_equal(bc.read(), counts)
                    for mode in ("fast", "fast-wg", "fast-wg8", "fast-wg16", "fast-wg16-cached"):
                        for cap in (1, 3, 17):
                            q2 = ChunkedQ2(cfg, batch, mb, backend=backend, dense_bypass=bypass,
                                           finalizer=mode, row_cap=cap, wg=8 if backend == "dpas" else 16)
                            out, cnt = Buffer(np.full_like(want, -777)), Buffer(np.full_like(counts, -777))
                            for _ in range(2):
                                q2.enqueue(inputs, out.device, cnt.device, guard_groups=2)
                                cl.finish()
                                np.testing.assert_array_equal(out.read(), want)
                                np.testing.assert_array_equal(cnt.read(), counts)
                                q2.read()
                            cases += 1
        empty = H.make_batch([(0, 0), (7, 0)], cfg.pa_block_size)
        for mode in ("fast", "fast-wg", "fast-wg8", "fast-wg16", "fast-wg16-cached"):
            q2 = ChunkedQ2(cfg, empty, 8, finalizer=mode)
            q2.enqueue([], None, None)
            assert not cl.finish()
    print(f"ALL {cases} FAST HELPER CONFIGURATIONS PASSED (two replays each)", flush=True)


def real_pipeline_scores(size=4096):
    """Actual external x -> Q0/Q1 -> Q2 DPAS, not synthetic indexer inputs."""
    from clops import cl
    import test_qsa_pipeline as T
    f = T.Fixture(qsa_config.load(), [size])
    p = T.Pipeline(f, [(0, size)], T.State(f.empty_state()), optimized=True,
                   guard=True, score_options={"row_cap": 17})
    p.prepare()
    cl.finish()
    # Last partial chunk of real rows; chunk-local scores, global causal origin.
    chunk = p.q2.plan.chunks[-1]
    p.q2.score(chunk, (p.iq.device, p.state.dev("summary"), *p.meta), guard_groups=1)
    cl.finish()
    count = chunk.row_end - chunk.row_begin
    scores = p.q2.read()[:count].copy()
    f = Finalizers(scores, [(chunk.row_begin, count)])
    f.validate(cap=3)
    print("PASS actual Q0/Q1/DPAS scores " + json.dumps(dict(
        global_begin=chunk.row_begin, rows=count, max_candidates=int(f.nc.max()),
        sha256=hashlib.sha256(scores.tobytes()).hexdigest())), flush=True)
    return scores


def benchmark(args):
    from clops import cl
    rng = np.random.default_rng(991)
    flusher = H.CacheFlusher() if "cold96MiB" in args.cache else None
    for n in args.candidates:
        actual = real_pipeline_scores(n * 4)[-1, :n] if args.distribution == "pipeline-row" else None
        for rows in args.rows:
            # A contiguous causal prefill slice ENDING at n complete blocks.
            past = 4 * n - rows
            if past < 0:
                raise ValueError("rows must fit the requested candidate endpoint")
            scores = (rng.uniform(0, 10, (rows, n)).astype(np.float32) if actual is None else
                      np.broadcast_to(actual, (rows, n)).copy())
            f = Finalizers(scores, [(past, rows)], variants=args.variants)
            f.validate()
            for cache in args.cache:
                for i in range(args.warmup):
                    for v in (f.variants if i % 2 == 0 else f.variants[::-1]):
                        f.enqueue(v)
                    cl.finish()
                samples = {v: [] for v in f.variants}
                for i in range(args.samples):
                    for v in (f.variants if i % 2 == 0 else f.variants[::-1]):
                        if cache == "cold96MiB":
                            flusher.flush()
                        f.enqueue(v)
                        ns = cl.finish()
                        assert len(ns) == 1, ns
                        samples[v].append(ns[0] * 1e-6)
                f.check()
                print("BENCH " + json.dumps(dict(rows=rows, candidates_min=int(f.nc.min()),
                    candidates_max=int(f.nc.max()), distribution=("uniform[0,10), identical device scores"
                        if actual is None else "actual Q0/Q1/DPAS last-row replay (correlated rows, not full prefill)"),
                    scores_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
                    cache=cache, warmup=args.warmup, samples=args.samples,
                    key_cache_capacity=f.key_capacities,
                    variants={v: dict(**stats(a), samples_ms=a) for v, a in samples.items()})), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--rows", nargs="+", type=int, default=[1, 256, 4096])
    ap.add_argument("--candidates", nargs="+", type=int, default=[4096, 16384])
    ap.add_argument("--variants", nargs="+", choices=tuple(NAMES), default=list(NAMES),
                    help="timed finalizers; fast-wg vs fast-wg16-cached isolates WG16 caching")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--distribution", choices=("uniform", "pipeline-row"), default="uniform")
    ap.add_argument("--cache", nargs="+", choices=("no-flush", "cold96MiB"), default=["no-flush", "cold96MiB"])
    args = ap.parse_args()
    if args.warmup < 0 or args.samples < 1 or min(args.rows + args.candidates) < 1:
        ap.error("positive rows/candidates/samples and nonnegative warmup required")
    from clops import cl
    if not any(name in str(cl.dev_info()) for name in ("B580", "B390")):
        raise RuntimeError("Specified remote B580 or PTL B390 only; never run on the local GPU")
    cl.profiling(True)
    root = Path(__file__).resolve().parent
    print("ENV " + json.dumps(dict(device=cl.dev_info(), sources={str(p.relative_to(root)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
        *[root / "kernels" / (name + ".cm") for name in NAMES.values()],
        root / "include/qsa_common.hpp", root / "include/qsa_topk_fast.hpp"]})), flush=True)
    if not args.benchmark_only:
        regression()
        cached_boundary_tests()
        helper_tests()
        real_pipeline_scores()
    if args.benchmark or args.benchmark_only:
        benchmark(args)


if __name__ == "__main__":
    main()