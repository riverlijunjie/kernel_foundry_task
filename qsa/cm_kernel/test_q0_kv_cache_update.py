"""Q0 qsa_kv_cache_update: prefill/decode sweep + roofline (paged K/V scatter)."""
from __future__ import annotations

import argparse
import numpy as np

import qsa_config
import qsa_harness as H
import qsa_work


def reference(cfg, batch, key, value):
    Hkv, Dh, Dv, pab = cfg.kv_heads_num, cfg.k_head_size, cfg.v_head_size, cfg.pa_block_size
    kc = np.zeros((batch.num_blocks, Hkv, pab, Dh), dtype=np.float16)
    vc = np.zeros((batch.num_blocks, Hkv, pab, Dv), dtype=np.float16)
    for tok in range(batch.num_tokens):
        seq, ap = batch.seq_of_token(tok)
        page = int(batch.block_indices[int(batch.block_indices_begins[seq]) + ap // pab])
        tip = ap % pab
        for h in range(Hkv):
            kc[page, h, tip] = key[tok, h * Dh:(h + 1) * Dh]
            vc[page, h, tip] = value[tok, h * Dv:(h + 1) * Dv]
    return kc, vc


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
    Hkv, Dh, Dv = cfg.kv_heads_num, cfg.k_head_size, cfg.v_head_size
    kern = H.build_kernel("qsa_kv_cache_update.cm", cfg.jit("qsa_kv_cache_update"))

    def run(mode):
        def make_case(n):
            batch = H.make_mode_batch(mode, n, cfg.pa_block_size, args.decode_batch)
            rng = np.random.default_rng(0)
            key = H.rand_f16(rng, batch.num_tokens, Hkv * Dh)
            value = H.rand_f16(rng, batch.num_tokens, Hkv * Dv)
            t_key, t_value = cl.tensor(key), cl.tensor(value)
            t_past = cl.tensor(batch.past_lens)
            t_bidx, t_bib = cl.tensor(batch.block_indices), cl.tensor(batch.block_indices_begins)
            t_sub = cl.tensor(batch.subsequence_begins)
            t_kc = cl.tensor(np.zeros((batch.num_blocks, Hkv, cfg.pa_block_size, Dh), np.float16))
            t_vc = cl.tensor(np.zeros((batch.num_blocks, Hkv, cfg.pa_block_size, Dv), np.float16))
            wg = 16
            GWS = [1, Hkv, ((batch.num_tokens + wg - 1) // wg) * wg]
            LWS = [1, 1, wg]

            def enqueue():
                kern.enqueue("qsa_kv_cache_update", GWS, LWS, t_key, t_value, t_past, t_bidx,
                             t_bib, t_sub, t_kc, t_vc, Hkv * Dh, 0, Hkv * Dv, 0, batch.num_seqs)

            def check():
                enqueue(); cl.finish()
                kc_ref, vc_ref = reference(cfg, batch, key, value)
                ok = (np.array_equal(t_kc.numpy().reshape(kc_ref.shape), kc_ref)
                      and np.array_equal(t_vc.numpy().reshape(vc_ref.shape), vc_ref))
                print(f"[Q0 {mode}] n={n} correctness: {'OK' if ok else 'FAIL'}")

            flops, byts = qsa_work.q0(cfg, batch)
            return dict(name="Q0-kv_update", enqueue=enqueue, flops=flops, bytes=byts, check=check)

        H.sweep(args.sizes, make_case, phase=mode, iters=args.iters,
                flush=not args.no_flush, do_check=args.check)

    for m in (["prefill", "decode"] if args.mode == "both" else [args.mode]):
        run(m)


if __name__ == "__main__":
    main()
