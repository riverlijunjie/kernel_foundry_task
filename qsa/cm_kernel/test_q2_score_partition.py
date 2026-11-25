"""Q2 qsa_score_partition: decode sweep + roofline (split-block scoring, pass 1).

Decode-only in production: one new token over a KV history swept by --sizes.
"""
from __future__ import annotations

import argparse
import numpy as np

import qsa_config
import qsa_harness as H
import qsa_ref
import qsa_work

PARTITION_BLOCKS = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096],
                    help="decode: KV history length; prefill: query tokens")
    ap.add_argument("--mode", choices=["prefill", "decode", "both"], default="decode")
    ap.add_argument("--decode-batch", type=int, default=1)
    ap.add_argument("--pa-block", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    from clops import cl

    cfg = qsa_config.load(pa_block_size=args.pa_block)
    kern = H.build_kernel("qsa_score_partition.cm",
                          cfg.jit("qsa_score_partition", QSA_PARTITION_BLOCKS=PARTITION_BLOCKS))

    def run(mode):
        def make_case(n):
            batch = H.make_mode_batch(mode, n, cfg.pa_block_size, args.decode_batch)
            iq, summ, max_blocks, ncomp = H.make_indexer_inputs(cfg, batch)
            nparts = max(1, (max_blocks + PARTITION_BLOCKS - 1) // PARTITION_BLOCKS)
            t_iq, t_summ = cl.tensor(iq.reshape(-1)), cl.tensor(summ)
            t_past, t_sub = cl.tensor(batch.past_lens), cl.tensor(batch.subsequence_begins)
            t_bidx, t_bib = cl.tensor(batch.block_indices), cl.tensor(batch.block_indices_begins)
            t_scores = cl.tensor(np.full((batch.num_tokens, max_blocks), -np.inf, np.float32))
            GWS, LWS = [batch.num_tokens, 1, nparts], [1, 1, 1]

            def enqueue():
                kern.enqueue("qsa_score_partition", GWS, LWS, t_iq, t_summ, t_past, t_sub,
                             t_bidx, t_bib, t_scores, batch.num_tokens, batch.num_seqs, max_blocks)

            def check():
                enqueue(); cl.finish()
                ref, _ = qsa_ref.indexer_scores_ref(cfg, batch, iq, summ, max_blocks)
                got = t_scores.numpy().reshape(batch.num_tokens, max_blocks)
                ok = True
                for tt in range(batch.num_tokens):
                    nc = int(ncomp[tt])
                    if nc and np.abs(got[tt, :nc] - ref[tt, :nc]).max() > 2e-2:
                        ok = False; break
                print(f"[Q2-part {mode}] n={n} correctness: {'OK' if ok else 'FAIL'} (parts={nparts})")

            flops, byts = qsa_work.q2_score(cfg, batch, ncomp)
            return dict(name="Q2-score_part", enqueue=enqueue, flops=flops, bytes=byts, check=check)

        H.sweep(args.sizes, make_case, phase=mode, iters=args.iters,
                flush=not args.no_flush, do_check=args.check)

    for m in (["prefill", "decode"] if args.mode == "both" else [args.mode]):
        run(m)


if __name__ == "__main__":
    main()
