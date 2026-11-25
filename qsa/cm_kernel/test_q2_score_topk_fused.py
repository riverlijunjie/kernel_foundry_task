"""Q2 qsa_score_topk_fused: prefill/decode sweep + roofline (scalar score + top-k)."""
from __future__ import annotations

import argparse
import numpy as np

import qsa_config
import qsa_harness as H
import qsa_ref
import qsa_work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--mode", choices=["prefill", "decode", "both"], default="prefill")
    ap.add_argument("--decode-batch", type=int, default=1)
    ap.add_argument("--pa-block", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    from clops import cl

    cfg = qsa_config.load(pa_block_size=args.pa_block)
    K = cfg.block_topk
    kern = H.build_kernel("qsa_score_topk_fused.cm", cfg.jit("qsa_score_topk_fused"))

    def run(mode):
        def make_case(n):
            batch = H.make_mode_batch(mode, n, cfg.pa_block_size, args.decode_batch)
            iq, summ, max_blocks, ncomp = H.make_indexer_inputs(cfg, batch)
            t_iq, t_summ = cl.tensor(iq.reshape(-1)), cl.tensor(summ)
            t_past, t_sub = cl.tensor(batch.past_lens), cl.tensor(batch.subsequence_begins)
            t_bidx, t_bib = cl.tensor(batch.block_indices), cl.tensor(batch.block_indices_begins)
            t_scores = cl.tensor(np.zeros((batch.num_tokens, max_blocks), np.float32))
            t_sel = cl.tensor(np.full((batch.num_tokens, K), -1, np.int32))
            t_cnt = cl.tensor(np.zeros(batch.num_tokens, np.int32))
            GWS, LWS = [batch.num_tokens, 1, 1], [1, 1, 1]

            def enqueue():
                kern.enqueue("qsa_score_topk_fused", GWS, LWS, t_iq, t_summ, t_past, t_sub,
                             t_bidx, t_bib, t_scores, t_sel, t_cnt, batch.num_tokens, batch.num_seqs, max_blocks)

            def check():
                enqueue(); cl.finish()
                ref, _ = qsa_ref.indexer_scores_ref(cfg, batch, iq, summ, max_blocks)
                sel, cnt = t_sel.numpy().reshape(batch.num_tokens, K), t_cnt.numpy().reshape(batch.num_tokens)
                ok = all(qsa_ref.selection_matches(sel[t], int(cnt[t]), ref[t], ncomp[t], K, tol=5e-2)
                         for t in range(batch.num_tokens))
                print(f"[Q2-fused {mode}] n={n} correctness: {'OK' if ok else 'FAIL'}")

            flops, byts = qsa_work.q2_score(cfg, batch, ncomp)
            return dict(name="Q2-score_fused", enqueue=enqueue, flops=flops, bytes=byts, check=check)

        H.sweep(args.sizes, make_case, phase=mode, iters=args.iters,
                flush=not args.no_flush, do_check=args.check)

    for m in (["prefill", "decode"] if args.mode == "both" else [args.mode]):
        run(m)


if __name__ == "__main__":
    main()
