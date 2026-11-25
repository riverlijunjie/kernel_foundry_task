"""Sequential standalone Q0 -> Q1 -> Q2 -> Q3 validation and benchmark.

Default is the explicitly labelled baseline (CURRENT optimized prefill Q3 DPAS,
not the historical unoptimized Q3). --optimized opts into the five-stage work;
--compare runs independent baseline/optimized states. No plugin propagation.
GPU execution: specified remote B580 only. See README for exact commands.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from time import perf_counter_ns

import numpy as np

import qsa_config
import qsa_harness as H
from qsa_prepare_dpas import PrepareDPAS
from qsa_score_chunked import ChunkedQ2, DEFAULT_SCORE_BYTES, FINALIZER_CHOICES
from qsa_attention_split import Q3Attention
from qsa_attention_gqa import Q3GQAAttention

GUARD = -777
COMPILED = {}
SCORE_OPTIONS = {}


def tensor(a):
    from clops import cl
    a = np.ascontiguousarray(a)
    return cl.tensor(a if a.size else np.zeros(1, a.dtype))


class Buffer:
    """Caller-owned payload plus a checked 64-element suffix (also K0 dummy)."""
    def __init__(self, a):
        self.shape, self.size = a.shape, a.size
        self.device = tensor(np.concatenate((a.reshape(-1), np.full(64, GUARD, a.dtype))))

    def read(self):
        a = self.device.numpy().reshape(-1)
        np.testing.assert_array_equal(a[self.size:], GUARD)
        return a[:self.size].reshape(self.shape).copy()


def compiled(cfg, tag, factory):
    key = (tuple(cfg.jit("unused").items()), tag)
    if key not in COMPILED:
        COMPILED[key] = factory()
    return COMPILED[key]


def kernel(cfg, name, **jit):
    return compiled(cfg, (name, tuple(jit.items())),
                    lambda: H.build_kernel(name + ".cm", cfg.jit(name, **jit)))


def check_config(c):
    # Deliberately not a general deployment policy. K0/K1 are safety tests only.
    fields = (c.hidden_size, c.heads_num, c.kv_heads_num, c.k_head_size,
              c.v_head_size, c.indexer_heads, c.indexer_kv_heads, c.indexer_head_dim,
              c.indexer_rotary_dim, c.compress_ratio, c.pa_block_size,
              c.has_output_gate, c.is_causal, c.budget)
    if fields != (2560, 24, 2, 256, 256, 4, 1, 128, 32, 4, 16, True, True, 2048):
        raise ValueError("integration is validated only for default model geometry/r4/PA16/gated causal")
    if c.block_topk not in (0, 1, 512):
        raise ValueError("only K512 and safety-test K0/K1 are validated")


class Fixture:
    """Real external Q/K/V/mixed inputs; NEVER synthesized indexer summaries/IDs.

    Fixed shuffled physical page allocation survives successive calls. Each
    sequence's inputs include its history and future appended tokens.
    """
    def __init__(self, cfg, capacities, seed=71, zero=False):
        check_config(cfg)
        self.cfg = cfg
        self.capacities = capacities
        self.pages = H.make_batch([(0, n) for n in capacities], cfg.pa_block_size, seed=seed)
        rng = np.random.default_rng(seed)
        j = (cfg.indexer_heads + 1) * cfg.indexer_head_dim
        self.w = H.f16(rng.normal(0, .1, (j, cfg.hidden_size)))
        self.qn = H.f16(rng.uniform(.5, 1.5, cfg.indexer_head_dim))
        self.kn = H.f16(rng.uniform(.5, 1.5, cfg.indexer_head_dim))
        angles = rng.uniform(-3, 3, (max(capacities, default=0) + 8, cfg.indexer_rotary_dim // 2))
        self.rot = H.f16(np.stack((np.cos(angles), np.sin(angles)), -1).reshape(len(angles), -1))
        self.data = []
        for n in capacities:
            self.data.append(dict(
                x=H.f16(rng.normal(0, .25, (n, cfg.hidden_size))),
                key=H.rand_f16(rng, n, cfg.kv_heads_num, cfg.k_head_size),
                value=H.rand_f16(rng, n, cfg.kv_heads_num, cfg.v_head_size)))
            if zero:
                self.data[-1]["x"].fill(0)
        self.constants = [tensor(a) for a in (self.w, self.qn, self.kn, self.rot)]
        self.seed = seed

    def batch(self, seqs):
        if len(seqs) != len(self.capacities) or any(
                p < 0 or q < 0 or p + q > cap for (p, q), cap in zip(seqs, self.capacities)):
            raise ValueError("invalid call range or page capacity")
        b = self.pages
        return H.PagedBatch(b.pa_block, H.i32([p for p, _ in seqs]),
            H.i32(np.cumsum([0] + [q for _, q in seqs])), b.block_indices,
            b.block_indices_begins, b.num_blocks, list(seqs))

    def empty_state(self):
        c, n = self.cfg, self.pages.num_blocks
        return dict(kc=np.zeros((n, c.kv_heads_num, c.pa_block_size, c.k_head_size), np.float16),
            vc=np.zeros((n, c.kv_heads_num, c.pa_block_size, c.v_head_size), np.float16),
            raw=np.zeros((n, c.pa_block_size, c.indexer_head_dim), np.float16),
            summary=np.zeros((n, c.summaries_per_page, c.indexer_head_dim), np.float16))


class State:
    def __init__(self, arrays):
        self.buffers = {name: Buffer(a) for name, a in arrays.items()}

    def dev(self, name):
        return self.buffers[name].device

    def snapshot(self):
        return {name: a.read() for name, a in self.buffers.items()}


class Pipeline:
    """Preallocated complete slice on clops' single IN-ORDER global queue.

    enqueue() never allocates device buffers, copies tensors, or synchronizes.
    Replaying a call is deterministic: it stores the same current tokens; Q1
    reads only raw slots strictly below this call's fixed past_len.
    State can be passed to a later Pipeline with advanced seqs (same Fixture).
    """
    def __init__(self, fixture, seqs, state, *, optimized=False, phase="prefill", guard=False,
                 score_options=None, attention_options=None):
        self.f, self.cfg, self.state = fixture, fixture.cfg, state
        self.batch = b = fixture.batch(seqs)
        self.nt, self.optimized, self.phase, self.guard = b.num_tokens, optimized, phase, guard
        if phase not in ("prefill", "decode") or (phase == "decode" and any(q > 1 for _, q in seqs)):
            raise ValueError("decode requires at most one token per sequence")
        c, nt = self.cfg, self.nt
        self.before = state.snapshot()  # validation oracle; OUTSIDE all timing
        self.host = {name: np.concatenate([d[name][p:p + q] for d, (p, q) in zip(fixture.data, seqs)], axis=0)
                     for name in ("x", "key", "value")}
        rng = np.random.default_rng(fixture.seed + nt + sum(p for p, _ in seqs))
        self.qhs = c.k_head_size + c.v_head_size
        self.pitch = c.heads_num * self.qhs
        self.query = H.rand_f16(rng, nt, c.heads_num, self.qhs)
        self.pos = H.i32([b.seq_of_token(t)[1] + 3 for t in range(nt)])
        self.nc = H.i32([(b.seq_of_token(t)[1] + 1) // c.compress_ratio for t in range(nt)])
        self.dense = self.nc <= c.block_topk
        self.mb = max(8, H.ceil8(int(self.nc.max(initial=0))))
        self.meta = [tensor(a) for a in (b.past_lens, b.subsequence_begins,
                                       b.block_indices, b.block_indices_begins)]
        self.current = {name: tensor(a) for name, a in self.host.items()}
        self.qdev, self.posdev = tensor(self.query), tensor(self.pos)
        pm = [(s, block) for s, (p, q) in enumerate(seqs) if q
              for block in range(p // c.compress_ratio, (p + q - 1) // c.compress_ratio + 1)]
        self.nwg, self.pmap = len(pm), tensor(H.i32(pm).reshape(-1, 2))
        tm, self.tiles = H.make_score_tile_map(b)
        self.tmap = tensor(tm)
        tokens, _ = H.make_score_tile_map(b, tile=1)
        self.token_map = tensor(tokens)
        j = (c.indexer_heads + 1) * c.indexer_head_dim
        self.iq = Buffer(np.full((nt, c.indexer_heads, c.indexer_head_dim), np.nan, np.float16))
        self.projected = Buffer(np.full((nt, j), np.nan, np.float32))
        self.sel = Buffer(np.full((nt, c.block_topk), GUARD, np.int32))
        self.count = Buffer(np.full(nt, GUARD, np.int32))
        self.out = Buffer(np.full((nt, c.heads_num, c.v_head_size), np.nan, np.float16))
        self.q0 = kernel(c, "qsa_kv_cache_update")
        self.q1 = (compiled(c, "prepare-dpas", lambda: PrepareDPAS(c)) if optimized else
                   kernel(c, "qsa_prepare", QSA_PREPARE_WG=16))
        self.partition = 16 if optimized else 512
        coop = optimized and (phase == "decode" or bool(self.dense.all()))
        wg = 16 if phase == "decode" else 8
        self.q2 = ChunkedQ2(c, b, self.mb, backend="dpas" if phase == "prefill" else "partition",
            cooperative=coop, wg=wg, partition_blocks=self.partition, dense_bypass=optimized,
            **(SCORE_OPTIONS if score_options is None else score_options))
        # Explicit integration-only conservative policy: full K512 at >=2048
        # past, B1/B4 decode only; no inferred crossover for other counts/batches.
        self.split = (optimized and phase == "decode" and c.block_topk == 512 and
                      b.num_seqs in (1, 4) and nt == b.num_seqs and
                      all(p >= 2048 for p, _ in seqs))
        if phase == "prefill" and optimized:
            att_opts = dict(attention_options or {})
            if set(att_opts) - {"head_major", "vector_meta"}:
                raise ValueError("unsupported GQA option")
            if any(type(value) is not bool for value in att_opts.values()):
                raise ValueError("GQA options must be bool")
            self.q3 = compiled(c, ("gqa", tuple(sorted(att_opts.items()))),
                               lambda: Q3GQAAttention(c, enabled=True, **att_opts))
            q3name = "gqa-W4" + ("-head-major" if att_opts.get("head_major") else "")
            q3name += "-vector-meta" if att_opts.get("vector_meta") else ""
        elif phase == "prefill":
            self.q3 = kernel(c, "qsa_sparse_attention_dpas", QSA_ATT_WORKERS=4,
                QSA_HEADS_PER_TILE=4, QSA_COALESCE_KV=1, QSA_WIDE_TAIL=1, QSA_CACHE_HEADS=1)
            q3name = "CURRENT-optimized-DPAS-W4-HT4"
        else:
            partitions = 16 if self.split else None
            self.q3 = compiled(c, ("attention", partitions), lambda: Q3Attention(c, partitions=partitions))
            q3name = "split-P16" if self.split else "scalar"
        na, nm = self.q3.scratch_elements(nt) if phase == "decode" else (0, 0)
        self.acc, self.ml = Buffer(np.full(na, np.nan, np.float32)), Buffer(np.full(nm, np.nan, np.float32))
        self.labels = (["Q0", "Q1-project", "Q1-prepare"] if optimized else ["Q0", "Q1-scalar"])
        self.q2_labels = ["Q2-dpas" if phase == "prefill" else f"Q2-partition{self.partition}",
                  f"Q2-WG{wg}" if coop else "Q2-scalar"]
        if self.q2.final_name == "qsa_topk_radix_fast":
            self.q2_labels[1] = "Q2-fast"
        elif self.q2.final_name == "qsa_topk_radix_wg":
            self.q2_labels[1] = f"Q2-fast-WG{self.q2.wg}"
            if self.q2.cache_capacity:
                self.q2_labels[1] += "-cached"
        elif self.q2.final_name == "qsa_topk_cooperative":
            self.q2_labels[1] = f"Q2-WG{self.q2.wg}"
        else:
            self.q2_labels[1] = "Q2-scalar"
        self.labels += self.q2_labels * len(self.q2.plan.chunks) + ["Q3-" + q3name]
        if self.split:
            self.labels.append("Q3-reduce")

    def prepare(self):
        if not self.nt:
            return
        c, b, s = self.cfg, self.batch, self.state
        past, begins, pages, pb = self.meta
        g = int(self.guard)
        self.q0.enqueue("qsa_kv_cache_update", [1, c.kv_heads_num, ((self.nt + 15) // 16 + g) * 16],
            [1, 1, 16], self.current["key"], self.current["value"], past, pages, pb, begins,
            s.dev("kc"), s.dev("vc"), c.kv_heads_num * c.k_head_size, 0,
            c.kv_heads_num * c.v_head_size, 0, b.num_seqs)
        args = (self.current["x"], *self.f.constants, self.posdev, *self.meta,
                s.dev("raw"), s.dev("summary"), self.iq.device, self.pmap, self.nwg)
        gws, lws = [1, 1, (self.nwg + g) * 16], [1, 1, 16]
        if self.optimized:
            self.q1.enqueue(self.projected.device, self.nt, gws, lws, *args, guard_tiles=self.guard)
        else:
            self.q1.enqueue("qsa_prepare", gws, lws, *args)

    def enqueue(self):
        if not self.nt:
            return
        self.prepare()
        c, b, s, g = self.cfg, self.batch, self.state, int(self.guard)
        inputs = (self.iq.device, s.dev("summary"), *self.meta)
        self.q2.enqueue(inputs, self.sel.device, self.count.device, guard_groups=g)
        att = (self.qdev, s.dev("kc"), s.dev("vc"), *self.meta, self.sel.device, self.count.device)
        layout = (self.nt, b.num_seqs, 0, self.pitch, self.qhs, c.k_head_size)
        if self.phase == "prefill" and self.optimized:
            self.q3.enqueue(att, self.out.device, *layout, token_map=self.token_map, guard_items=self.guard)
        elif self.phase == "prefill":
            self.q3.enqueue("qsa_sparse_attention_dpas", [(c.heads_num // 4 + g) * 4,
                (self.tiles + g) * 4, 1], [4, 1, 1], *att, self.tmap, self.out.device,
                self.tiles, 0, self.pitch, self.qhs, c.k_head_size)
        else:
            self.q3.enqueue(att, self.out.device, *layout, partial_acc=self.acc.device,
                            partial_ml=self.ml.device, guard_items=self.guard)

    def result(self):
        result = self.state.snapshot()
        for name in ("iq", "sel", "count", "out", "projected", "acc", "ml"):
            result[name] = getattr(self, name).read()
        # Only the final reusable scratch, NEVER a full-token score matrix.
        result["scores"] = self.q2.read()
        for name in ("iq", "out"):
            assert np.isfinite(result[name]).all(), f"missing/nonfinite {name}"
        if self.split:
            assert np.isfinite(result["acc"]).all() and np.isfinite(result["ml"]).all()
        return result


def error(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(d.max(initial=0))


def close(a, b, atol, rtol, label):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=label)
    return error(a, b)


def norm_rope(p, weight, positions, f):
    c = f.cfg
    p = p.astype(np.float32).copy()
    p *= (np.float32(1) / np.sqrt(np.mean(p * p, axis=-1, keepdims=True) + np.float32(c.indexer_norm_eps)))
    p *= weight.astype(np.float32)
    half = c.indexer_rotary_dim // 2
    angles = f.rot[positions].astype(np.float32)
    # positions can be [tokens,1] to broadcast over heads.
    lo, hi = p[..., :half].copy(), p[..., half:2 * half].copy()
    p[..., :half] = lo * angles[..., 0::2] - hi * angles[..., 1::2]
    p[..., half:2 * half] = hi * angles[..., 0::2] + lo * angles[..., 1::2]
    return p.astype(np.float16)


def prepare_reference(p):
    c, b = p.cfg, p.batch
    proj = p.host["x"].astype(np.float32) @ p.f.w.astype(np.float32).T
    di, hi, r = c.indexer_head_dim, c.indexer_heads, c.compress_ratio
    iq = norm_rope(proj[:, :hi * di].reshape(p.nt, hi, di), p.f.qn, p.pos[:, None], p.f)
    ref = {name: a.copy() for name, a in p.before.items()}
    masks = {name: np.zeros(a.shape[:-1], bool) for name, a in ref.items()}
    for s, (past, q) in enumerate(b.seqs):
        begin, pb = int(b.subsequence_begins[s]), int(b.block_indices_begins[s])
        def page(pos):
            return int(b.block_indices[pb + pos // c.pa_block_size])
        for pos in range(past, past + q):
            t, pg, tip = begin + pos - past, page(pos), pos % c.pa_block_size
            for name, src in (("kc", "key"), ("vc", "value")):
                ref[name][pg, :, tip] = p.host[src][t]
                masks[name][pg, :, tip] = True
            ref["raw"][pg, tip] = proj[t, hi * di:]
            masks["raw"][pg, tip] = True
        if not q:
            continue
        for block in range(past // r, (past + q) // r):
            pooled = np.zeros(di, np.float32)
            for pos in range(block * r, (block + 1) * r):
                pooled += (p.before["raw"][page(pos), pos % c.pa_block_size].astype(np.float32)
                           if pos < past else proj[begin + pos - past, hi * di:])
            pg, slot = page(block * r), block % c.summaries_per_page
            ref["summary"][pg, slot] = norm_rope(pooled / np.float32(r), p.f.kn, block * r, p.f)
            masks["summary"][pg, slot] = True
    return proj, iq, ref, masks


def validate_prepare(p, got):
    proj, iq, ref, masks = prepare_reference(p)
    errors = {"iq": close(got["iq"], iq, 8e-3, 2e-3, "Q1 iq")}
    for name in ref:
        if name in ("kc", "vc"):
            np.testing.assert_array_equal(got[name], ref[name])
        else:
            errors[name] = close(got[name], ref[name], 8e-3, 2e-3, "Q1 " + name)
        np.testing.assert_array_equal(got[name][~masks[name]], p.before[name][~masks[name]])
    if p.optimized:
        errors["projection"] = close(got["projected"], proj, 3e-5, 2e-5, "Q1 projection")
    return errors


def score_reference(p, got, row_begin, row_end):
    c, b = p.cfg, p.batch
    ref = np.full((row_end - row_begin, p.mb), np.nan, np.float32)
    for s, (_, q) in enumerate(b.seqs):
        begin, end = map(int, b.subsequence_begins[s:s + 2])
        begin, end = max(begin, row_begin), min(end, row_end)
        if not q or begin >= end:
            continue
        n = int(p.nc[end - 1])
        ids = np.arange(n)
        pages = b.block_indices[int(b.block_indices_begins[s]) + ids // c.summaries_per_page]
        summaries = got["summary"][pages, ids % c.summaries_per_page].astype(np.float32)
        scores = np.zeros((end - begin, n), np.float32)
        for h in range(c.indexer_heads):
            scores += np.maximum(got["iq"][begin:end, h].astype(np.float32) @ summaries.T, np.float32(0))
        ref[begin - row_begin:end - row_begin, :n] = scores * np.float32(c.indexer_scale)
    return ref


def validated_score_rows(p, got, errors):
    """Re-score immutable own Q1 inputs, validate/consume one chunk, then reuse.

    Saved selected/counts come from the real uninterrupted Q0->Q3 enqueue.
    No full score readback, even for large prefill. Re-scoring is outside timing.
    """
    from clops import cl
    inputs = (p.iq.device, p.state.dev("summary"), *p.meta)
    errors["scores_own_iq_summary"] = 0.0
    np.testing.assert_array_equal(got["count"], np.minimum(p.nc, p.cfg.block_topk))
    for chunk in p.q2.plan.chunks:
        before = p.q2.read()
        p.q2.score(chunk, inputs, guard_groups=1)
        cl.finish()
        scores = p.q2.read()
        size = chunk.row_end - chunk.row_begin
        np.testing.assert_array_equal(scores[size:].view(np.uint32), before[size:].view(np.uint32))
        sr = score_reference(p, got, chunk.row_begin, chunk.row_end)
        for local, t in enumerate(range(chunk.row_begin, chunk.row_end)):
            nc, count = int(p.nc[t]), int(got["count"][t])
            if p.optimized and p.dense[t]:
                np.testing.assert_array_equal(scores[local].view(np.uint32), before[local].view(np.uint32))
            else:
                e = close(scores[local, :nc], sr[local, :nc], 3e-5, 2e-5, "Q2 scores")
                errors["scores_own_iq_summary"] = max(errors["scores_own_iq_summary"], e)
            expected = (np.arange(count, dtype=np.int32) if nc <= p.cfg.block_topk else
                        exact_selection(scores[local], nc, p.cfg.block_topk))
            np.testing.assert_array_equal(got["sel"][t, :count], expected)
            np.testing.assert_array_equal(got["sel"][t, count:], -1)
            yield t, scores[local, :nc]


def exact_selection(scores, nc, k):
    ids = np.arange(int(nc))
    # Finite nonnegative Q2 scores: descending score, ascending ID for exact ties.
    return np.sort(np.lexsort((ids, -scores[:nc]))[:min(int(nc), k)]).astype(np.int32)


def attention_reference(p, got, rows):
    c, b = p.cfg, p.batch
    out = np.zeros((len(rows), c.heads_num, c.v_head_size), np.float32)
    for i, t in enumerate(rows):
        s, ap = b.seq_of_token(int(t))
        ids = got["sel"][t, :got["count"][t]]
        pos = np.concatenate(((ids[:, None] * c.compress_ratio + np.arange(c.compress_ratio)).reshape(-1),
                              np.arange(int(p.nc[t]) * c.compress_ratio, ap + 1)))
        if not len(pos):
            continue
        pages = b.block_indices[int(b.block_indices_begins[s]) + pos // c.pa_block_size]
        tips = pos % c.pa_block_size
        q = p.query[t].astype(np.float32)
        for kh in range(c.kv_heads_num):
            hs = slice(kh * c.n_rep, (kh + 1) * c.n_rep)
            k, v = got["kc"][pages, kh, tips].astype(np.float32), got["vc"][pages, kh, tips].astype(np.float32)
            scores = (q[hs, :c.k_head_size] * np.float32(c.scale)) @ k.T
            e = np.exp(scores - scores.max(axis=1, keepdims=True))
            value = (e @ v) / e.sum(axis=1, keepdims=True, dtype=np.float32)
            value *= np.float32(1) / (np.float32(1) + np.exp(-q[hs, c.k_head_size:]))
            out[i, hs] = value
    return out


def validate_pair(ps, label, *, full_ref=False):
    from clops import cl
    for p in ps:
        p.enqueue()
    cl.finish()
    results = [p.result() for p in ps]
    nt = ps[0].nt
    differing = (np.flatnonzero(np.any(results[0]["sel"] != results[1]["sel"], axis=1))
                 if len(ps) == 2 else np.empty(0, np.int64))
    rows = set(range(nt)) if nt <= 64 or full_ref else set(np.linspace(0, nt - 1, 32, dtype=int))
    rows.update(map(int, differing))  # ALL changed-selection outputs, not filtered away
    rows.update(t for t in range(nt) if ps[0].batch.seq_of_token(t)[1] in (2049, 2050, 2051, 2052))
    rows = np.asarray(sorted(rows), np.int64)
    row = dict(case=label, tokens=nt, dense=int(ps[0].dense.sum()), sparse=int((~ps[0].dense).sum()),
               attention_ref_rows=len(rows), metadata_exact_rows=nt * len(ps), variants={})
    score_errors = [{"scores_own_iq_summary": 0.0} for _ in ps]
    streams = [validated_score_rows(p, got, e) for p, got, e in zip(ps, results, score_errors)]
    differing_set, margins, all_margins, ab_scores = set(map(int, differing)), [], [], 0.0
    k = ps[0].cfg.block_topk
    # zip holds at most ONE chunk per variant, not all-token scores.
    for entries in zip(*streams):
        t = entries[0][0]
        assert all(entry[0] == t for entry in entries)
        if len(ps) == 2 and not ps[0].dense[t]:
            sa, sb = entries[0][1], entries[1][1]
            eps = error(sa, sb)
            ab_scores = max(ab_scores, eps)
            if k:
                va, vb = np.sort(sa)[::-1], np.sort(sb)[::-1]
                all_margins.append(float(va[k - 1] - va[k]))
            if t in differing_set:
                a_ids, b_ids = set(results[0]["sel"][t]), set(results[1]["sel"][t])
                item = dict(token=t, sequence_position=list(ps[0].batch.seq_of_token(t)),
                    removed=sorted(map(int, a_ids - b_ids)), added=sorted(map(int, b_ids - a_ids)),
                    score_delta_bound=eps, baseline_margin=float(va[k - 1] - va[k]),
                    optimized_margin=float(vb[k - 1] - vb[k]),
                    output_max_abs=error(results[0]["out"][t], results[1]["out"][t]))
                assert item["baseline_margin"] <= 2 * eps + 1e-7, item
                margins.append(item)
    for p, got, se in zip(ps, results, score_errors):
        errors = validate_prepare(p, got)
        errors.update(se)
        ref = attention_reference(p, got, rows)
        errors["attention_own_selection"] = close(got["out"][rows], ref, 2e-3, 2e-3, "Q3 OWN selection")
        empty = (got["count"] == 0) & np.array([
            (p.batch.seq_of_token(t)[1] + 1) % p.cfg.compress_ratio == 0
            for t in range(nt)], dtype=bool)
        np.testing.assert_array_equal(got["out"][empty], 0)
        row["variants"]["optimized" if p.optimized else "baseline"] = dict(
            stages=list(dict.fromkeys(p.labels)), errors=errors, score_chunks=len(p.q2.plan.chunks),
            score_scratch_bytes=p.q2.plan.allocation_bytes, score_rows=p.q2.plan.scratch_rows)
    if len(ps) == 2:
        a, b = results
        row["ab_max_abs"] = {name: error(a[name], b[name]) for name in ("iq", "raw", "summary", "out")}
        row["ab_max_abs"]["sparse_scores"] = ab_scores
        row["selection_different_rows"] = len(differing)
        row["baseline_sparse_margin_min"] = min(all_margins, default=None)
        row["selection_differences"] = margins
        same = np.ones(nt, bool)
        same[differing] = False
        close(a["out"][same], b["out"][same], 2e-3, 2e-3, "Q3 same selection A/B")
    # Replay on persistent state; ALL payloads/guards must remain bit-exact.
    for p in ps:
        p.enqueue()
    cl.finish()
    for p, before in zip(ps, results):
        for name, value in p.result().items():
            np.testing.assert_array_equal(value, before[name], err_msg="replay " + name)
    print("PASS " + json.dumps(row), flush=True)
    for p in ps:
        p.guard = False  # validation-only extra WGs never enter timed dispatch
    return results


def setup_history(f, pasts):
    """One real scalar Q0/Q1 history initialization, then independent clones."""
    from clops import cl
    state = State(f.empty_state())
    if any(pasts):
        # History needs only Q0/Q1; never honor the unchunked comparison opt-out
        # for its potentially huge, unused Q2 allocation.
        p = Pipeline(f, [(0, n) for n in pasts], state, guard=True,
                 score_options={"row_cap": 1})
        p.prepare()
        cl.finish()
        got = state.snapshot()
        got["iq"] = p.iq.read()
        errors = validate_prepare(p, got)
        print("HISTORY " + json.dumps(dict(tokens=sum(pasts), errors=errors, source="actual scalar Q0/Q1; outside timing")), flush=True)
    return state.snapshot()


def make_pair(f, seqs, variants, phase, initial=None):
    initial = setup_history(f, [p for p, _ in seqs]) if initial is None else initial
    return [Pipeline(f, seqs, State(initial), optimized=opt, phase=phase, guard=True) for opt in variants]


def regression(cfg, variants, full_ref):
    cases = [
        ("small-full", [(0, 17)], "prefill", 512, False),
        ("mixed-boundary", [(0, 0), (0, 1), (3, 5), (15, 17), (2046, 17), (2050, 1), (2051, 1), (4096, 3)], "prefill", 512, False),
        ("short-decode", [(1, 1), (3, 1), (15, 1), (31, 1)], "decode", 512, False),
        ("K0-mixed", [(0, 1), (3, 5), (15, 3)], "prefill", 0, False),
        ("K1-ties", [(0, 17), (31, 3)], "prefill", 1, True),
        ("empty", [(0, 0), (7, 0)], "prefill", 512, False),
    ]
    for label, seqs, phase, k, zero in cases:
        f = Fixture(replace(cfg, block_topk=k), [p + q for p, q in seqs], zero=zero)
        ps = make_pair(f, seqs, variants, phase)
        validate_pair(ps, label, full_ref=full_ref)
        if label == "mixed-boundary":
            t = int(ps[0].batch.subsequence_begins[5])
            np.testing.assert_array_equal(ps[0].nc[t:t + 2], [512, 513])
    # Actual persistent same-device caches across calls, including a partly filled
    # summary block and the exact visible2051 ->2052 dense/sparse transition.
    f = Fixture(cfg, [2053, 18], seed=83)
    ps = make_pair(f, [(0, 2051), (0, 15)], variants, "prefill")
    validate_pair(ps, "continuity-prefill", full_ref=full_ref)
    for step, seqs in enumerate(([(2051, 1), (15, 1)], [(2052, 1), (16, 1)])):
        ps = [Pipeline(f, seqs, old.state, optimized=old.optimized, phase="decode", guard=True) for old in ps]
        validate_pair(ps, f"continuity-decode-{step + 1}", full_ref=full_ref)
    print("ALL 9 INTEGRATED REGRESSION CALLS PASSED", flush=True)


def stats(values):
    a = np.asarray(values, np.float64)
    return dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)))


def benchmark(cfg, args, variants):
    from clops import cl
    flusher = H.CacheFlusher() if args.flush else None
    shapes = []
    if args.mode in ("prefill", "both"):
        shapes += [(f"prefill{n}", [(0, n)], "prefill") for n in args.sizes]
    if args.mode in ("decode", "both"):
        shapes += [(f"decode{args.decode_past}-B{b}", [(args.decode_past, 1)] * b, "decode") for b in args.decode_batches]
    for label, seqs, phase in shapes:
        f = Fixture(cfg, [p + q for p, q in seqs])
        ps = make_pair(f, seqs, variants, phase)
        validated = validate_pair(ps, label, full_ref=args.full_ref)
        if args.validate_shapes:
            continue
        for i in range(args.warmup):
            for p in (ps if i % 2 == 0 else ps[::-1]):
                p.enqueue()
            cl.finish()
        events, hosts = [[] for _ in ps], [[] for _ in ps]
        for i in range(args.samples):
            for j in (range(len(ps)) if i % 2 == 0 else reversed(range(len(ps)))):
                if flusher:
                    flusher.flush()  # before WHOLE slice, never between stages
                start = perf_counter_ns()
                ps[j].enqueue()
                ns = cl.finish()
                hosts[j].append((perf_counter_ns() - start) * 1e-6)
                assert len(ns) == len(ps[j].labels), (ns, ps[j].labels)
                events[j].append(np.asarray(ns, np.float64) * 1e-6)
        row = dict(case=label, cache="cold96MiB" if args.flush else "no-flush",
                   warmup=args.warmup, samples=args.samples, variants={})
        for j, p in enumerate(ps):
            for name, value in p.result().items():
                np.testing.assert_array_equal(value, validated[j][name], err_msg="post-benchmark " + name)
            e = np.asarray(events[j])
            row["variants"]["optimized" if p.optimized else "baseline"] = dict(
                gpu_sum=stats(e.sum(axis=1)), host_elapsed=stats(hosts[j]),
                stages={name: float(e[:, [i for i, label in enumerate(p.labels) if label == name]].sum(axis=1).mean())
                    for name in dict.fromkeys(p.labels)},
                samples_ms=e.sum(axis=1).tolist(), host_samples_ms=hosts[j],
                score_scratch_bytes=p.q2.plan.allocation_bytes, score_chunks=len(p.q2.plan.chunks),
                score_rows=p.q2.plan.scratch_rows, projection_scratch_bytes=p.projected.size * 4,
                split_scratch_bytes=(p.acc.size + p.ml.size) * 4)
        print("BENCH " + json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--optimized", action="store_true")
    mode.add_argument("--compare", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--benchmark-only", action="store_true", help="skip edge suite, NEVER skip timed-shape validation")
    ap.add_argument("--validate-shapes", action="store_true", help="validate requested prefill/decode shapes without timing or edge suite")
    ap.add_argument("--score-budget-mib", type=int, default=128, help="score allocation ceiling, including guards (1..128 MiB)")
    ap.add_argument("--score-row-cap", type=int, help="force smaller query-row chunks for validation")
    ap.add_argument("--score-unchunked", action="store_true", help="old ABI comparison only when full scores fit budget; incompatible with row cap")
    ap.add_argument("--topk-finalizer", choices=FINALIZER_CHOICES,
                    help="explicit Q2 override; legacy/omitted preserves policy; fast-wg16 pins WG16")
    ap.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--decode-past", type=int, default=4096)
    ap.add_argument("--decode-batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--flush", action="store_true")
    ap.add_argument("--full-ref", action="store_true", help="Q3 NumPy for every query, not just <=64 + sampled + changed-selection rows")
    args = ap.parse_args()
    if (args.warmup < 0 or args.samples < 1 or any(n < 1 or n > 131072 for n in args.sizes)
            or not 1 <= args.decode_past <= 131072 or any(b not in (1, 4) for b in args.decode_batches)
            or not 1 <= args.score_budget_mib <= DEFAULT_SCORE_BYTES // (1024 * 1024)
            or (args.score_row_cap is not None and args.score_row_cap < 1)
            or (args.score_unchunked and args.score_row_cap is not None)):
        ap.error("sizes/past=1..131072, B=1/4, budget=1..128MiB, row_cap>=1; unchunked excludes row_cap; warmup>=0,samples>=1")
    SCORE_OPTIONS.update(budget_bytes=args.score_budget_mib * 1024 * 1024,
                         row_cap=args.score_row_cap, unchunked=args.score_unchunked,
                         finalizer=args.topk_finalizer)
    from clops import cl
    device = cl.dev_info()
    # Only the two remote machines named in remote_machine.txt are authorized:
    # the Arc B580 (BMG) and the PTL Arc B390 iGPU. This still rejects any
    # other/local GPU by name.
    if not any(name in str(device) for name in ("B580", "B390")):
        raise RuntimeError("Run only on a specified remote machine (Arc B580 or PTL Arc B390), not the local GPU")
    print("DEVICE", device, flush=True)
    cl.profiling(True)  # creates an in-order queue; do not change it mid-pipeline
    cfg = qsa_config.load()
    check_config(cfg)
    variants = [False, True] if args.compare else [args.optimized]
    if not args.benchmark_only and not args.validate_shapes:
        regression(cfg, variants, args.full_ref)
    if args.benchmark or args.benchmark_only or args.validate_shapes:
        benchmark(cfg, args, variants)
    print("ALL REQUESTED INTEGRATED CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()