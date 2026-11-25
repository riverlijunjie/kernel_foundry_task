"""Strict Stage3 Q2 regression and alternating remote-only GPU A/B.

Q1/Q3 are not dispatched. Each score backend is compared only with itself;
finalizer-only A/B uses the SAME baseline score buffer, never Q1 DPAS outputs.
Default runs correctness. --benchmark adds 100 warmups/20 samples per variant.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from time import perf_counter

import numpy as np

import qsa_config
import qsa_harness as H
from qsa_topk_cooperative import PartitionScorer, TopKFinalizer
import validate_dense_bypass as D

GUARD = D.GUARD
_FINAL = {}
_SCORE = {}


def finalizer(cfg, cooperative, wg, bypass=True):
    key = (tuple(cfg.jit("unused").items()), cooperative, wg, bypass)
    if key not in _FINAL:
        _FINAL[key] = TopKFinalizer(cfg, cooperative=cooperative, wg=wg,
                                  dense_bypass=bypass)
    return _FINAL[key]


class Pipeline:
    def __init__(self, case, backend="partition", *, cooperative=False, wg=16,
                 partition=512, bypass=True, guard=False):
        from clops import cl

        self.c, self.backend, self.guard = case, backend, guard
        self.final = finalizer(case.cfg, cooperative, wg, bypass)
        key = (tuple(case.cfg.jit("unused").items()), backend, partition, bypass)
        if key not in _SCORE:
            if backend == "partition":
                _SCORE[key] = PartitionScorer(case.cfg, partition_blocks=partition,
                                             dense_score_bypass=bypass)
            else:
                name = "qsa_score_tile_dpas"
                _SCORE[key] = H.build_kernel(name + ".cm", case.cfg.jit(
                    name, QSA_SCORE_WG=16, QSA_DENSE_BYPASS=int(bypass),
                    QSA_DENSE_SCORE_BYPASS=int(bypass)))
        self.scorer = _SCORE[key]
        self.scores = cl.tensor(case.poison)
        self.sel = cl.tensor(np.full(case.nt * case.k + 32, GUARD, np.int32))
        self.cnt = cl.tensor(np.full(case.nt + 32, GUARD, np.int32))

    def score(self):
        c = self.c
        if self.backend == "partition":
            self.scorer.enqueue(c.inputs, self.scores, c.nt, c.batch.num_seqs, c.mb)
        else:
            self.scorer.enqueue("qsa_score_tile_dpas", [c.tiles * 16, 1, 1], [16, 1, 1],
                                *c.inputs, c.tile_map, self.scores, c.tiles, c.mb)

    def finalize(self):
        c = self.c
        self.final.enqueue(self.scores, c.inputs[2], c.inputs[3], self.sel, self.cnt,
                           c.nt, c.batch.num_seqs, c.mb, guard_groups=3 if self.guard else 0)

    def enqueue(self):
        self.score()
        self.finalize()

    def metadata(self):
        c = self.c
        sel, cnt = self.sel.numpy(), self.cnt.numpy()
        np.testing.assert_array_equal(sel[c.nt * c.k:], GUARD)
        np.testing.assert_array_equal(cnt[c.nt:], GUARD)
        return sel[:c.nt * c.k].reshape(c.nt, c.k), cnt[:c.nt]


def compare(p, ref):
    for got, want in zip(p.metadata(), ref):
        np.testing.assert_array_equal(got, want)


def check_scores(p, base_scores, bypass):
    c = p.c
    got = p.scores.numpy().view(np.uint32)
    ref = base_scores.view(np.uint32)
    if bypass:
        np.testing.assert_array_equal(got[c.dense], c.poison.view(np.uint32)[c.dense])
        np.testing.assert_array_equal(got[~c.dense], ref[~c.dense])
    else:
        np.testing.assert_array_equal(got, ref)


def check_reads(c, scores, ref, wg):
    """Instrument all three actual cooperative score-load sites, per CM thread."""
    from clops import cl

    name = "qsa_topk_cooperative"
    src = H.flatten_source(name + ".cm", c.cfg.jit(name, QSA_TOPK_WG=wg))
    # The unused scalar helper also has scores[i]; restrict to the kernel body.
    head, body = src.split('extern "C" _GENX_MAIN_', 1)
    assert body.count("scores[i]") == 3
    body = body.replace("scores[i]", "(reads[token_idx * WG + tid] += 1, scores[i])")
    assert body.count("uint32_t max_blocks)") == 1
    body = body.replace("uint32_t max_blocks)", "uint32_t max_blocks, int32_t* reads ATTR)")
    kern = cl.kernels(head + 'extern "C" _GENX_MAIN_' + body, H.QSA_BUILD_OPTIONS)
    p = Pipeline(c, cooperative=True, wg=wg, guard=True)
    supplied = scores.copy()
    supplied[c.dense] = c.poison[c.dense]
    p.scores = cl.tensor(supplied)
    reads = cl.tensor(np.zeros((c.nt + 3, wg), np.int32))
    kern.enqueue(name, [(c.nt + 3) * wg, 1, 1], [wg, 1, 1], p.scores,
                 c.inputs[2], c.inputs[3], p.sel, p.cnt, c.nt, c.batch.num_seqs, c.mb, reads)
    cl.finish()
    compare(p, ref)
    counts = reads.numpy()
    np.testing.assert_array_equal(counts[c.nt:], 0)
    active = (c.nc > c.k) & (c.k > 0)
    np.testing.assert_array_equal(counts[:c.nt][~active], 0)
    # Four full histogram scans plus count and emit: exactly 6*n reads.
    np.testing.assert_array_equal(counts[:c.nt][active].sum(axis=1), 6 * c.nc[active])
    print(f"PASS reads k={c.k} WG={wg} dense=0 sparse={int(counts.sum())}", flush=True)


def validate():
    from clops import cl

    for k in (0, 1, 512):
        cfg = qsa_config.load(block_topk=k)
        # Both visible-token boundary 2051/2052 and n_complete 2051/2052.
        # 4097/8193 complete blocks exceed a would-be 4096 candidate cap.
        seqs = [(0, 0), (0, 1), (0, 17), (2046, 17), (2050, 1), (2051, 1),
                (2051 * 4 - 1, 1), (2052 * 4 - 1, 1), (4097 * 4 - 1, 1),
                (8193 * 4 - 1, 2), (100, 0)]
        for ties in (False, True):
            c = D.Case(cfg, seqs, ties)
            assert c.nc[-7:-5].tolist() == [512, 513]
            assert 2051 in c.nc and 2052 in c.nc and c.nc.max() > 4096
            for backend in ("partition", "dpas"):
                base = Pipeline(c, backend, bypass=False, guard=True)
                base.enqueue()
                cl.finish()
                scores = base.scores.numpy()
                ref = D.expected(scores, c.nc, k)
                compare(base, ref)
                for wg in (8, 16):
                    p = Pipeline(c, backend, cooperative=True, wg=wg, guard=True)
                    p.scores = base.scores  # EXACT same GPU-produced score input
                    p.finalize()
                    cl.finish()
                    compare(p, ref)
                    p = Pipeline(c, backend, cooperative=True, wg=wg, guard=True)
                    p.enqueue()
                    cl.finish()
                    compare(p, ref)
                    check_scores(p, scores, True)
                if backend == "partition":
                    for part in (16, 32, 64):
                        for bypass in (False, True):
                            p = Pipeline(c, backend, cooperative=True, partition=part,
                                         bypass=bypass, guard=True)
                            p.enqueue()
                            cl.finish()
                            compare(p, ref)
                            check_scores(p, scores, bypass)
            print(f"PASS pipeline k={k} ties={ties} rows={c.nt} WG8/16 "
                  "partition512/16/32/64,dpas exact", flush=True)

        c = D.Case(cfg, seqs)
        rng = np.random.default_rng(39)
        for distribution in ("random", "zero", "equal", "ties", "near", "threshold", "late"):
            scores = rng.integers(-4, 5, (c.nt, c.mb)).astype(np.float32)
            if distribution == "random":
                scores = rng.normal(size=scores.shape).astype(np.float32)
            elif distribution == "zero":
                scores.fill(0)
            elif distribution == "equal":
                scores.fill(0.5)
            elif distribution == "near":
                scores = np.float32(1) + scores * np.spacing(np.float32(1))
            elif distribution == "threshold":
                scores.fill(1)
                scores[:, 1:2 * max(0, k - 1):2] = 2  # k-1 higher; huge kth tie
            elif distribution == "late":
                scores[:] = np.arange(c.mb, dtype=np.float32)  # winners after 4096
            ref = D.expected(scores, c.nc, k)
            base = Pipeline(c, bypass=False, guard=True)
            base.scores = cl.tensor(scores)
            base.finalize()
            cl.finish()
            compare(base, ref)
            for wg in (8, 16):
                for bypass in (False, True):
                    p = Pipeline(c, cooperative=True, wg=wg, bypass=bypass, guard=True)
                    supplied = scores.copy()
                    if bypass:
                        supplied[c.dense] = c.poison[c.dense]
                    p.scores = cl.tensor(supplied)
                    p.finalize()
                    cl.finish()
                    compare(p, ref)
                if distribution == "threshold":
                    check_reads(c, scores, ref, wg)
            print(f"PASS finalizer k={k} distribution={distribution}", flush=True)

    # Alternative r/page, and empty host enqueue without allocating zero-size GPU tensors.
    cfg = replace(qsa_config.load(block_topk=1, pa_block_size=32), compress_ratio=8)
    c = D.Case(cfg, [(0, 3), (7, 2), (31, 17), (32800, 1)])
    base = Pipeline(c, bypass=False)
    base.enqueue()
    cl.finish()
    ref = D.expected(base.scores.numpy(), c.nc, c.k)
    compare(base, ref)
    p = Pipeline(c, cooperative=True, partition=16)
    p.enqueue()
    cl.finish()
    compare(p, ref)
    p.final.enqueue(None, None, None, None, None, 0, 0, 0)
    p.scorer.enqueue([], None, 0, 0, 0)
    print("ALL Q2 COOPERATIVE CHECKS PASSED", flush=True)


def measure(pipelines, names, scope, args, flusher, context):
    from clops import cl

    fns = [p.enqueue if scope == "pipeline" else p.finalize for p in pipelines]
    cl.profiling(True)
    for i in range(args.warmup):
        for j in (range(len(fns)) if i % 2 == 0 else reversed(range(len(fns)))):
            fns[j]()
        cl.finish()
    events, hosts = [[] for _ in fns], [[] for _ in fns]
    for i in range(args.samples):
        for j in (range(len(fns)) if i % 2 == 0 else reversed(range(len(fns)))):
            if flusher:
                flusher.flush()  # never between score and top-k
            t0 = perf_counter()
            fns[j]()
            ns = cl.finish()
            hosts[j].append((perf_counter() - t0) * 1000)
            assert len(ns) == (2 if scope == "pipeline" else 1), ns
            events[j].append(np.asarray(ns, dtype=np.float64) * 1e-6)
    row = dict(context, scope=scope, warmup=args.warmup, samples=args.samples,
               cache="cold96MiB" if args.flush else "no-flush", variants={})
    for j, name in enumerate(names):
        a = np.asarray(events[j])
        total = a.sum(axis=1)
        row["variants"][name] = dict(mean_ms=float(total.mean()),
            median_ms=float(np.median(total)), p95_ms=float(np.percentile(total, 95)),
            stage_mean_ms=a.mean(axis=0).tolist(), host_mean_ms=float(np.mean(hosts[j])),
            samples_ms=total.tolist())
    print("BENCH " + json.dumps(row), flush=True)


def benchmark(args):
    from clops import cl

    cfg = qsa_config.load()
    flusher = H.CacheFlusher() if args.flush else None
    for mode in args.modes:
        sizes = args.prefill_sizes if mode == "prefill" else args.decode_sizes
        for size in sizes:
            c = D.Case(cfg, [(0, size)] if mode == "prefill" else [(size, 1)])
            backend = "dpas" if mode == "prefill" else "partition"
            baseline = Pipeline(c, backend, bypass=False)
            baseline.enqueue()
            cl.finish()
            scores = baseline.scores.numpy()
            ref = D.expected(scores, c.nc, c.k)
            compare(baseline, ref)
            specs = [("scalar-p512", False, 16, 512),
                     ("wg8-p512", True, 8, 512), ("wg16-p512", True, 16, 512)]
            if mode == "decode":
                specs += [(f"{label}-p{part}", coop, 16, part)
                          for part in (16, 32, 64)
                          for label, coop in (("scalar", False), ("wg16", True))]
            ps = []
            for name, coop, wg, part in specs:
                p = Pipeline(c, backend, cooperative=coop, wg=wg, partition=part)
                p.enqueue()
                cl.finish()
                compare(p, ref)
                check_scores(p, scores, True)
                ps.append(p)
            context = dict(mode=mode, size=size, backend=backend, rows=c.nt,
                           dense_rows=int(c.dense.sum()))
            measure(ps, [s[0] for s in specs], "pipeline", args, flusher, context)
            # Isolate selector on one identical baseline-produced score tensor.
            for p in ps[:3]:
                p.scores = baseline.scores
                p.finalize()
            cl.finish()
            for p in ps[:3]:
                compare(p, ref)
            measure(ps[:3], [s[0] for s in specs[:3]], "finalizer", args, flusher, context)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--benchmark-only", action="store_true",
                    help="skip edge regression, but still check every benchmark variant")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--flush", action="store_true")
    ap.add_argument("--modes", nargs="+", choices=["prefill", "decode"], default=["prefill", "decode"])
    ap.add_argument("--prefill-sizes", nargs="+", type=int, default=[1024, 2048, 4096])
    ap.add_argument("--decode-sizes", nargs="+", type=int, default=[4096, 16384, 65536])
    args = ap.parse_args()
    if args.warmup < 0 or args.samples < 1 or min(args.prefill_sizes + args.decode_sizes) < 1:
        ap.error("warmup >= 0, samples >= 1, sizes >= 1 required")
    from clops import cl

    print("DEVICE", cl.dev_info(), flush=True)
    if not args.benchmark_only:
        validate()
    if args.benchmark or args.benchmark_only:
        benchmark(args)


if __name__ == "__main__":
    main()