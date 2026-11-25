"""Exact SAME-kernel Q2 row-chunk regression; run GPU tests on remote B580 only.

No benchmark. --plan-only needs no NumPy/clops/GPU. Guards count in the budget.
"""
from __future__ import annotations

import argparse

from qsa_score_chunked import DEFAULT_SCORE_BYTES, ChunkedQ2, plan_score_chunks


def planner_tests():
    def fails(**kwargs):
        try:
            plan_score_chunks(**kwargs)
        except ValueError:
            return
        raise AssertionError(f"expected failure: {kwargs}")
    for cap in (1, 3, 15, 16, 17, 31, 2051, 2052):
        begins = (0, 0, 1, 18, 18, 51, 2086, 4113)
        p = plan_score_chunks(begins, 520, budget_bytes=cap * 520 * 4 + 256,
                              row_cap=cap, guard_elements=64)
        seen = []
        for c in p.chunks:
            assert c.row_end - c.row_begin <= cap
            covered = []
            for s, t in p.tile_map[c.tile_offset:c.tile_offset + c.tile_count]:
                assert begins[s] <= t < begins[s + 1] and c.row_begin <= t < c.row_end
                covered.extend(range(t, min(t + 16, begins[s + 1], c.row_end)))
            assert covered == list(range(c.row_begin, c.row_end))
            seen.extend(covered)
        assert seen == list(range(begins[-1]))
        assert p.allocation_bytes <= p.budget_bytes
    empty = plan_score_chunks((0, 0, 0), 16384, budget_bytes=4)
    assert empty.scratch_rows == 0 and not empty.chunks and empty.allocation_bytes == 4
    one = plan_score_chunks((0, 17), 8, budget_bytes=32)
    assert one.scratch_rows == 1 and one.scratch_bytes == 32
    assert plan_score_chunks((0, 17), 8, budget_bytes=95).scratch_rows == 2
    fails(subsequence_begins=(0, 1), max_blocks=8, budget_bytes=31)
    fails(subsequence_begins=(0, 1), max_blocks=8, budget_bytes=1)
    fails(subsequence_begins=(0, 1), max_blocks=8, budget_bytes=DEFAULT_SCORE_BYTES + 1)
    fails(subsequence_begins=(0, 1), max_blocks=8, budget_bytes=287, guard_elements=64)
    fails(subsequence_begins=(0, 0), max_blocks=8, budget_bytes=128, guard_elements=64)
    fails(subsequence_begins=(0, 2), max_blocks=8, budget_bytes=32, unchunked=True)
    fails(subsequence_begins=(0, 1), max_blocks=8, row_cap=0)
    fails(subsequence_begins=(0, 2, 1), max_blocks=8)
    fails(subsequence_begins=(0, 1), max_blocks=9)
    p = plan_score_chunks((0, 65536), 16384, guard_elements=64)
    assert 65536 * 16384 * 4 == 1 << 32
    assert p.scratch_rows == 2047 and p.allocation_bytes <= DEFAULT_SCORE_BYTES
    assert p.chunks[-1].row_end == 65536
    print("PASS pure planner: NT0, budgets, row/tile/sequence boundaries, 64K/4GiB", flush=True)


def gpu_tests(ks):
    import numpy as np
    import qsa_config
    import qsa_harness as H
    from clops import cl
    from test_qsa_pipeline import Buffer, tensor
    from validate_dense_bypass import expected

    if "B580" not in str(cl.dev_info()):
        raise RuntimeError("GPU tests require the specified remote B580")
    cl.profiling(True)
    cases = 0
    # Global query starts are deliberately NOT 16-aligned. Includes no-query
    # histories, visible2051/2052, complete-block2051/2052 and >4096 candidates.
    seqs = [(0, 0), (0, 1), (0, 17), (3, 5), (2046, 17), (2050, 1),
            (2051, 1), (2051 * 4 - 1, 1), (2052 * 4 - 1, 1), (65536, 3), (7, 0)]
    for k in ks:
        cfg = qsa_config.load(block_topk=k)
        batch = H.make_batch(seqs, cfg.pa_block_size, seed=19)
        iq, summ, mb, nc = H.make_indexer_inputs(cfg, batch, seed=83)
        assert 512 in nc and 513 in nc and 2051 in nc and 2052 in nc
        inputs = [[tensor(a) for a in (iq, data, batch.past_lens, batch.subsequence_begins,
                  batch.block_indices, batch.block_indices_begins)] for data in (summ, np.zeros_like(summ))]
        for backend in ("partition", "dpas"):
            for bypass in (False, True):
                for coop in (False, True):
                    opts = dict(backend=backend, cooperative=coop, wg=16, partition_blocks=16,
                                dense_bypass=bypass, guard_elements=64)
                    base = ChunkedQ2(cfg, batch, mb, unchunked=True, **opts)
                    bs, bc = Buffer(np.full((batch.num_tokens, k), -777, np.int32)), Buffer(np.full(batch.num_tokens, -777, np.int32))
                    refs = []
                    for inp in inputs:
                        base.enqueue(inp, bs.device, bc.device, guard_groups=2)
                        cl.finish()
                        scores, sel, cnt = base.read(), bs.read(), bc.read()
                        # Dense rows' expected IDs never depend on poison.
                        supplied = scores.copy()
                        if bypass:
                            supplied[nc <= k] = 0
                        want = expected(supplied, nc, k)
                        np.testing.assert_array_equal(sel, want[0])
                        np.testing.assert_array_equal(cnt, want[1])
                        refs.append((scores, sel, cnt))
                    for cap in (1, 3, 15, 16, 17, 31):
                        q2 = ChunkedQ2(cfg, batch, mb, row_cap=cap,
                            budget_bytes=cap * mb * 4 + 256, **opts)
                        sel = Buffer(np.full((batch.num_tokens, k), -777, np.int32))
                        cnt = Buffer(np.full(batch.num_tokens, -777, np.int32))
                        scratch = q2.scores
                        # Change inputs on the SAME scratch: random -> exact ties.
                        for inp, (scores, want_sel, want_cnt) in zip(inputs, refs):
                            previous_sel, previous_cnt = sel.read(), cnt.read()
                            for chunk in q2.plan.chunks:
                                before = q2.read()
                                q2.score(chunk, inp, guard_groups=2)
                                q2.finalize(chunk, inp[2], inp[3], sel.device, cnt.device, guard_groups=2)
                                cl.finish()
                                got = q2.read()
                                n = chunk.row_end - chunk.row_begin
                                np.testing.assert_array_equal(got[n:].view(np.uint32), before[n:].view(np.uint32))
                                for local, t in enumerate(range(chunk.row_begin, chunk.row_end)):
                                    if bypass and nc[t] <= k:
                                        np.testing.assert_array_equal(got[local].view(np.uint32), before[local].view(np.uint32))
                                    else:
                                        np.testing.assert_array_equal(got[local, :nc[t]].view(np.uint32), scores[t, :nc[t]].view(np.uint32))
                                now_sel, now_cnt = sel.read(), cnt.read()
                                sl = slice(chunk.row_begin, chunk.row_end)
                                np.testing.assert_array_equal(now_sel[sl], want_sel[sl])
                                np.testing.assert_array_equal(now_cnt[sl], want_cnt[sl])
                                outside = np.ones(batch.num_tokens, bool)
                                outside[sl] = False
                                np.testing.assert_array_equal(now_sel[outside], previous_sel[outside])
                                np.testing.assert_array_equal(now_cnt[outside], previous_cnt[outside])
                                previous_sel, previous_cnt = now_sel, now_cnt
                            # Real path: all chunks back-to-back, one finish, no poison resets.
                            q2.enqueue(inp, sel.device, cnt.device, guard_groups=2)
                            cl.finish()
                            np.testing.assert_array_equal(sel.read(), want_sel)
                            np.testing.assert_array_equal(cnt.read(), want_cnt)
                            q2.read()
                            assert q2.scores is scratch
                        cases += 1
                    print(f"PASS K{k} {backend} bypass={bypass} coop={coop}: chunks1/3/15/16/17/31, random->ties reuse", flush=True)
        # Empty device buffers use nonzero dummies and enqueue no kernels.
        empty = H.make_batch([(0, 0), (7, 0)], cfg.pa_block_size)
        q2 = ChunkedQ2(cfg, empty, 8)
        q2.enqueue([], None, None)
        assert not cl.finish()
        assert q2.read().size == 0
    # Actual query-chunk boundaries 2051/2052, not just absolute positions.
    cfg = qsa_config.load(block_topk=1)
    batch = H.make_batch([(0, 2051), (17, 36)], cfg.pa_block_size)
    iq, summ, mb, nc = H.make_indexer_inputs(cfg, batch, seed=91)
    inp = [tensor(a) for a in (iq, summ, batch.past_lens, batch.subsequence_begins,
                              batch.block_indices, batch.block_indices_begins)]
    for backend in ("partition", "dpas"):
        base = ChunkedQ2(cfg, batch, mb, backend=backend, dense_bypass=False, unchunked=True)
        bs, bc = Buffer(np.zeros((batch.num_tokens, 1), np.int32)), Buffer(np.zeros(batch.num_tokens, np.int32))
        base.enqueue(inp, bs.device, bc.device)
        cl.finish()
        ref, rs, rc = base.read(), bs.read(), bc.read()
        for cap in (2051, 2052):
            q2 = ChunkedQ2(cfg, batch, mb, backend=backend, dense_bypass=False, row_cap=cap)
            for chunk in q2.plan.chunks:
                q2.score(chunk, inp, guard_groups=1)
                q2.finalize(chunk, inp[2], inp[3], bs.device, bc.device, guard_groups=1)
                cl.finish()
                a = q2.read()
                for local, t in enumerate(range(chunk.row_begin, chunk.row_end)):
                    np.testing.assert_array_equal(a[local, :nc[t]].view(np.uint32), ref[t, :nc[t]].view(np.uint32))
            np.testing.assert_array_equal(bs.read(), rs)
            np.testing.assert_array_equal(bc.read(), rc)
            cases += 1
    print(f"ALL {cases} Q2 ROW-CHUNK CONFIGURATIONS PASSED (two reuse inputs per small case)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--k", nargs="+", type=int, choices=(0, 1, 512), default=[0, 1, 512])
    args = ap.parse_args()
    planner_tests()
    if not args.plan_only:
        gpu_tests(args.k)


if __name__ == "__main__":
    main()