"""Q3 qsa_sparse_attention (scalar): prefill/decode sweep + roofline.

Scalar path is QSA's production DECODE kernel (and a prefill fallback); decode
sweeps the KV history length.
"""
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
    ap.add_argument("--block-topk", type=int, default=None)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    from clops import cl

    cfg = qsa_config.load(pa_block_size=args.pa_block, block_topk=args.block_topk)
    Hq, Hkv, Dh, Dv = cfg.heads_num, cfg.kv_heads_num, cfg.k_head_size, cfg.v_head_size
    gate = cfg.has_output_gate
    qhs = 2 * Dh if gate else Dh
    gd = Dh if gate else 0
    q_pitch = Hq * qhs
    kern = H.build_kernel("qsa_sparse_attention.cm", cfg.jit("qsa_sparse_attention"))

    def run(mode):
        def make_case(n):
            batch = H.make_mode_batch(mode, n, cfg.pa_block_size, args.decode_batch)
            rng = np.random.default_rng(0)
            query = H.rand_f16(rng, batch.num_tokens, q_pitch)
            kc = H.rand_f16(rng, batch.num_blocks, Hkv, cfg.pa_block_size, Dh)
            vc = H.rand_f16(rng, batch.num_blocks, Hkv, cfg.pa_block_size, Dv)
            sel, cnt = H.make_selection(batch, cfg.compress_ratio, cfg.block_topk)
            t_q = cl.tensor(query)
            t_kc, t_vc = cl.tensor(kc.reshape(-1)), cl.tensor(vc.reshape(-1))
            t_past, t_sub = cl.tensor(batch.past_lens), cl.tensor(batch.subsequence_begins)
            t_bidx, t_bib = cl.tensor(batch.block_indices), cl.tensor(batch.block_indices_begins)
            t_sel, t_cnt = cl.tensor(sel), cl.tensor(cnt)
            t_out = cl.tensor(np.zeros((batch.num_tokens, Hq * Dv), np.float16))
            GWS, LWS = [Hq, batch.num_tokens, 1], [1, 1, 1]

            def enqueue():
                kern.enqueue("qsa_sparse_attention", GWS, LWS, t_q, t_kc, t_vc, t_past, t_sub,
                             t_bidx, t_bib, t_sel, t_cnt, t_out,
                             batch.num_tokens, batch.num_seqs, 0, q_pitch, qhs, gd)

            def check():
                enqueue(); cl.finish()
                ref = qsa_ref.sparse_attention_ref(cfg, batch, query, kc, vc, sel, cnt,
                                                   q_head_stride=qhs, gate_delta=gd)
                got = t_out.numpy().reshape(ref.shape).astype(np.float32)
                print(f"[Q3-scalar {mode}] n={n} correctness: "
                      f"{'OK' if np.abs(got - ref).max() < 2e-2 else 'FAIL'}")

            flops, byts = qsa_work.q3(cfg, batch, cnt, q_head_stride=qhs, kv_reuse_heads=Hkv)
            return dict(name="Q3-attn_scalar", enqueue=enqueue, flops=flops, bytes=byts, check=check)

        H.sweep(args.sizes, make_case, phase=mode, iters=args.iters,
                flush=not args.no_flush, do_check=args.check)

    for m in (["prefill", "decode"] if args.mode == "both" else [args.mode]):
        run(m)


if __name__ == "__main__":
    main()
