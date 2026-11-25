"""Remote B580 Q3 regression and same-input, alternating-order A/B benchmark.

Default: validate both paths against NumPy, including paged/chunked edge cases.
--benchmark: compare variants on identical device buffers with cold-cache timing.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

import qsa_config
import qsa_harness as H
import qsa_ref
from test_q3_sparse_attention_dpas import heads_per_tile, workers


VARIANTS = {
    "baseline": (0, 0, 0),
    "coalesce": (1, 0, 0),
    "wide-tail": (1, 1, 0),
    "optimized": (1, 1, 1),
}


def build(cfg, w, ht, variant):
    coalesce, tail, heads = VARIANTS[variant]
    return H.build_kernel("qsa_sparse_attention_dpas.cm", cfg.jit(
        "qsa_sparse_attention_dpas", QSA_ATT_WORKERS=w, QSA_HEADS_PER_TILE=ht,
        QSA_COALESCE_KV=coalesce, QSA_WIDE_TAIL=tail, QSA_CACHE_HEADS=heads))


def make_case(cfg, seqs, *, selection="random", padded=False, seed=0):
    from clops import cl

    batch = H.make_batch(seqs, cfg.pa_block_size, seed=seed)
    tile_map, num_tiles = H.make_score_tile_map(batch)
    hq, hkv, dh, dv = cfg.heads_num, cfg.kv_heads_num, cfg.k_head_size, cfg.v_head_size
    qhs = dh + dv if cfg.has_output_gate else dh
    gd = dh if cfg.has_output_gate else 0
    q_base = 32 if padded else 0
    q_pitch = hq * qhs + (32 if padded else 0)
    rng = np.random.default_rng(seed)
    query = H.rand_f16(rng, batch.num_tokens, hq * qhs)
    storage = np.zeros(q_base + batch.num_tokens * q_pitch, np.float16)
    storage[q_base:].reshape(batch.num_tokens, q_pitch)[:, :hq * qhs] = query
    kc = H.rand_f16(rng, batch.num_blocks, hkv, cfg.pa_block_size, dh)
    vc = H.rand_f16(rng, batch.num_blocks, hkv, cfg.pa_block_size, dv)
    sel, cnt = H.make_selection(batch, cfg.compress_ratio, cfg.block_topk, seed=seed)
    if selection == "empty":
        sel.fill(-1)
        cnt.fill(0)
    elif selection == "patterns":
        # Consecutive blocks, gaps, page-crossing runs, empty columns and partial steps.
        patterns = ([0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4], [4, 5, 6, 7],
                    [], [0], [0, 1], [0, 1, 2])
        sel.fill(-1)
        cnt.fill(0)
        for t in range(batch.num_tokens):
            nc = (batch.seq_of_token(t)[1] + 1) // cfg.compress_ratio
            ids = [b for b in patterns[t % len(patterns)] if b < nc][:cfg.block_topk]
            cnt[t] = len(ids)
            sel[t, :len(ids)] = ids
    elif selection != "random":
        raise ValueError(selection)

    # OpenCL disallows a zero-byte upload; an unused sentinel backs topk=0.
    device_sel = sel if sel.size else np.full(1, -1, np.int32)
    args = [cl.tensor(storage), cl.tensor(kc.reshape(-1)), cl.tensor(vc.reshape(-1)),
            cl.tensor(batch.past_lens), cl.tensor(batch.subsequence_begins),
            cl.tensor(batch.block_indices), cl.tensor(batch.block_indices_begins),
            cl.tensor(device_sel), cl.tensor(cnt), cl.tensor(tile_map)]
    output_size = batch.num_tokens * hq * dv

    def dispatch(kern, w, ht):
        # NaNs catch missing writes, and the guard catches stores past the last token.
        out = cl.tensor(np.full(output_size + 64, np.nan, np.float16))

        def enqueue():
            kern.enqueue("qsa_sparse_attention_dpas", [(hq // ht) * w, num_tiles * ht, 1],
                         [w, 1, 1], *args, out, num_tiles, q_base, q_pitch, qhs, gd)

        def result():
            a = out.numpy().reshape(-1)
            assert np.isnan(a[output_size:]).all(), "output guard overwritten"
            return a[:output_size].reshape(batch.num_tokens, hq * dv).astype(np.float32)

        return enqueue, result

    def reference():
        return qsa_ref.sparse_attention_ref(cfg, batch, query, kc, vc, sel, cnt,
                                           q_head_stride=qhs, gate_delta=gd)

    return dispatch, reference


def regression():
    from clops import cl

    cfg = qsa_config.load()
    mixed = [(0, 0), (0, 1), (1, 3), (3, 5), (14, 7), (15, 17), (31, 19), (63, 1)]
    cases = [
        ("dense-partial-tile", cfg, [(0, 129)], "random", False, None),
        ("chunked-shuffled-pages", cfg, mixed, "random", True, None),
        ("sparse-independent-selections", replace(cfg, block_topk=4), mixed, "random", True, None),
        ("page-crossing-and-empty-columns", replace(cfg, block_topk=4), [(96, 33)], "patterns", False, None),
        ("tail-only-and-empty-attention", replace(cfg, block_topk=4), mixed, "empty", True, None),
        ("one-block-budget", replace(cfg, block_topk=1), mixed, "random", False, None),
        ("zero-block-budget", replace(cfg, block_topk=0), mixed, "empty", False, None),
        ("page-8-fallback", replace(cfg, pa_block_size=8), mixed, "random", True, None),
        ("page-32-coalescing", replace(cfg, pa_block_size=32), mixed, "random", True, None),
        ("ungated", replace(cfg, has_output_gate=False), mixed, "random", True, None),
        ("eight-workers", cfg, mixed, "random", False, (8, 4)),
        ("eight-token-tile", replace(cfg, block_topk=4), mixed, "random", False, (4, 2)),
        ("sixteen-token-tile", replace(cfg, block_topk=4), mixed, "random", False, (4, 1)),
        ("single-worker", replace(cfg, k_head_size=64, v_head_size=64, scale=0.125),
         mixed, "random", True, (1, 4)),
        ("eight-head-tile", replace(cfg, heads_num=16, k_head_size=64, v_head_size=64, scale=0.125),
         mixed, "random", True, (1, 8)),
    ]
    for r in (2, 8, 16):
        cases.append((f"compression-{r}", replace(cfg, compress_ratio=r, block_topk=4),
                      mixed, "random", True, None))

    compiled = {}
    worst = 0.0
    for name, conf, seqs, selection, padded, geometry in cases:
        w, ht = geometry or (workers(conf), heads_per_tile(conf))
        dispatch, reference = make_case(conf, seqs, selection=selection, padded=padded)
        ref = reference()
        results = []
        for variant in ("baseline", "optimized"):
            key = (tuple(conf.jit("qsa_sparse_attention_dpas").items()), w, ht, variant)
            if key not in compiled:
                compiled[key] = build(conf, w, ht, variant)
            enqueue, result = dispatch(compiled[key], w, ht)
            enqueue()
            cl.finish()
            got = result()
            error = float(np.abs(got - ref).max())
            assert np.isfinite(got).all() and error < 2e-2, f"{name}/{variant}: max_abs={error}"
            worst = max(worst, error)
            results.append(got)
        np.testing.assert_array_equal(results[0], results[1], err_msg=name)
        print(f"PASS {name}: W={w} HT={ht}, baseline == optimized, max_abs={error:.6g}", flush=True)
    print(f"PASS {len(cases)} regression cases; worst max_abs={worst:.6g}")


def benchmark(args):
    cfg = qsa_config.load(pa_block_size=args.pa_block, block_topk=args.block_topk)
    w, ht = workers(cfg), heads_per_tile(cfg)
    variants = list(VARIANTS) if args.ablation else ["baseline", "optimized"]
    kernels = {v: build(cfg, w, ht, v) for v in variants}
    flusher = H.CacheFlusher()
    print(f"B580 cold-cache A/B: W={w} HT={ht} topk={cfg.block_topk} page={cfg.pa_block_size}, "
          f"{args.rounds} rounds x {args.iters} samples, warmup={args.warmup}", flush=True)
    for n in args.sizes:
        dispatch, _ = make_case(cfg, [(0, n)])
        calls = {v: dispatch(kernels[v], w, ht) for v in variants}
        stats = {v: [] for v in variants}
        for round_id in range(args.rounds):
            order = variants if round_id % 2 == 0 else variants[::-1]
            for v in order:
                enqueue, _ = calls[v]
                s = H.bench(enqueue, iters=args.iters, warmup=args.warmup, flusher=flusher)
                stats[v].append(s)
                print(f"q={n} round={round_id + 1} {v}: mean={s['mean']:.6f} "
                      f"median={s['median']:.6f} std={s['std']:.6f} ms", flush=True)
        baseline = calls["baseline"][1]()
        for v in variants[1:]:
            np.testing.assert_array_equal(baseline, calls[v][1](), err_msg=f"q={n}/{v}")
        means = {v: float(np.mean([s['mean'] for s in stats[v]])) for v in variants}
        for v in variants:
            print(f"SUMMARY q={n} {v}: mean={means[v]:.6f} ms "
                  f"speedup={means['baseline'] / means[v]:.3f}x; output identical", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="also time intermediate optimizations")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--block-topk", type=int, default=None)
    ap.add_argument("--pa-block", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    if min(args.iters, args.warmup, args.rounds, *args.sizes) < 1:
        ap.error("sizes, iters, warmup and rounds must be positive")
    if args.benchmark:
        benchmark(args)
    else:
        regression()


if __name__ == "__main__":
    main()