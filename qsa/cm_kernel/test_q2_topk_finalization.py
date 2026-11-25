"""Q2 qsa_topk_finalization: prefill/decode sweep + roofline (radix top-k)."""
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
    ap.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    ap.add_argument("--decode-batch", type=int, default=1)
    ap.add_argument("--pa-block", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    from clops import cl

    cfg = qsa_config.load(pa_block_size=args.pa_block)
    K, r = cfg.block_topk, cfg.compress_ratio
    kern = H.build_kernel("qsa_topk_finalization.cm", cfg.jit("qsa_topk_finalization"))

    def run(mode):
        def make_case(n):
            batch = H.make_mode_batch(mode, n, cfg.pa_block_size, args.decode_batch)
            ncomp = qsa_work.n_complete(batch, r)
            max_blocks = max(8, H.ceil8(int(ncomp.max())))
            rng = np.random.default_rng(0)
            scores = rng.random((batch.num_tokens, max_blocks), dtype=np.float64).astype(np.float32)
            t_scores = cl.tensor(np.ascontiguousarray(scores))
            t_past, t_sub = cl.tensor(batch.past_lens), cl.tensor(batch.subsequence_begins)
            t_sel = cl.tensor(np.full((batch.num_tokens, K), -1, np.int32))
            t_cnt = cl.tensor(np.zeros(batch.num_tokens, np.int32))
            GWS, LWS = [batch.num_tokens, 1, 1], [1, 1, 1]

            def enqueue():
                kern.enqueue("qsa_topk_finalization", GWS, LWS, t_scores, t_past, t_sub,
                             t_sel, t_cnt, batch.num_tokens, batch.num_seqs, max_blocks)

            def check():
                enqueue(); cl.finish()
                sel, cnt = t_sel.numpy().reshape(batch.num_tokens, K), t_cnt.numpy().reshape(batch.num_tokens)
                ok = all(qsa_ref.selection_matches(sel[t], int(cnt[t]), scores[t], int(ncomp[t]), K)
                         for t in range(batch.num_tokens))
                print(f"[Q2-topk {mode}] n={n} correctness: {'OK' if ok else 'FAIL'}")

            flops, byts = qsa_work.q2_topk(cfg, batch, ncomp)
            return dict(name="Q2-topk_final", enqueue=enqueue, flops=flops, bytes=byts, check=check)

        H.sweep(args.sizes, make_case, phase=mode, iters=args.iters,
                flush=not args.no_flush, do_check=args.check)

    for m in (["prefill", "decode"] if args.mode == "both" else [args.mode]):
        run(m)


if __name__ == "__main__":
    main()
