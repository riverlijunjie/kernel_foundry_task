"""Q1 qsa_prepare: prefill/decode sweep + roofline (indexer proj + RMSNorm + RoPE)."""
from __future__ import annotations

import argparse
import numpy as np

import qsa_config
import qsa_harness as H
import qsa_ref
import qsa_work

PREPARE_WG = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    ap.add_argument("--decode-batch", type=int, default=1)
    ap.add_argument("--pa-block", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dpas", action="store_true", help="opt in to FP32 DPAS projection + prepare (two dispatches)")
    args = ap.parse_args()
    from clops import cl

    cfg = qsa_config.load(pa_block_size=args.pa_block)
    Hidx, Di, D = cfg.indexer_heads, cfg.indexer_head_dim, cfg.hidden_size
    ROT, r, spp = cfg.indexer_rotary_dim, cfg.compress_ratio, cfg.summaries_per_page
    J = (Hidx + 1) * Di
    if args.dpas:
        from qsa_prepare_dpas import PrepareDPAS
        kern = PrepareDPAS(cfg, PREPARE_WG)
    else:
        kern = H.build_kernel("qsa_prepare.cm", cfg.jit("qsa_prepare", QSA_PREPARE_WG=PREPARE_WG))

    def run(mode):
        def make_case(n):
            batch = H.make_mode_batch(mode, n, cfg.pa_block_size, args.decode_batch)
            pmap, num_wg = H.make_prepare_map(batch, r)
            rng = np.random.default_rng(0)
            max_pos = n + 8
            mixed = H.rand_f16(rng, batch.num_tokens, D, scale=0.25)
            weight = H.rand_f16(rng, J, D, scale=0.1)
            q_norm = H.rand_f16(rng, Di, scale=0.5, bias=1.0)
            k_norm = H.rand_f16(rng, Di, scale=0.5, bias=1.0)
            rotary = H.f16(rng.uniform(-1, 1, size=(max_pos, ROT)))
            pos = np.array([batch.seq_of_token(t)[1] for t in range(batch.num_tokens)], np.int32)
            t = dict(
                mixed=cl.tensor(mixed.reshape(-1)), w=cl.tensor(weight.reshape(-1)),
                qn=cl.tensor(q_norm), kn=cl.tensor(k_norm), rot=cl.tensor(rotary.reshape(-1)),
                pos=cl.tensor(pos), past=cl.tensor(batch.past_lens), sub=cl.tensor(batch.subsequence_begins),
                bidx=cl.tensor(batch.block_indices), bib=cl.tensor(batch.block_indices_begins),
                rawk=cl.tensor(np.zeros(batch.num_blocks * cfg.pa_block_size * Di, np.float16)),
                summ=cl.tensor(np.zeros(batch.num_blocks * spp * Di, np.float16)),
                iq=cl.tensor(np.zeros((batch.num_tokens, Hidx, Di), np.float16)),
                pmap=cl.tensor(pmap))
            GWS, LWS = [1, 1, num_wg * PREPARE_WG], [1, 1, PREPARE_WG]
            projected = cl.tensor(np.zeros((batch.num_tokens, J), np.float32)) if args.dpas else None

            def enqueue():
                inputs = (t["mixed"], t["w"], t["qn"], t["kn"], t["rot"], t["pos"], t["past"],
                          t["sub"], t["bidx"], t["bib"], t["rawk"], t["summ"], t["iq"], t["pmap"], num_wg)
                if args.dpas:
                    kern.enqueue(projected, batch.num_tokens, GWS, LWS, *inputs)
                else:
                    kern.enqueue("qsa_prepare", GWS, LWS, *inputs)

            def check():
                enqueue(); cl.finish()
                iq_ref, summ_ref = qsa_ref.prepare_ref(cfg, batch, mixed, weight, q_norm, k_norm, rotary, pos)
                iq = t["iq"].numpy().reshape(batch.num_tokens, Hidx, Di).astype(np.float32)
                summ = t["summ"].numpy().reshape(-1).astype(np.float32)
                ok = np.abs(iq - iq_ref).max() < 3e-2
                for (pg, sl), v in summ_ref.items():
                    off = (pg * spp + sl) * Di
                    if np.abs(summ[off:off + Di] - v).max() > 3e-2:
                        ok = False; break
                print(f"[Q1 {mode}] n={n} correctness: {'OK' if ok else 'FAIL'}")

            flops, byts = qsa_work.q1(cfg, batch)
            if args.dpas:
                byts += 2 * batch.num_tokens * J * 4  # FP32 scratch write + read
            return dict(name="Q1-DPAS" if args.dpas else "Q1-prepare", enqueue=enqueue,
                        flops=flops, bytes=byts, check=check)

        H.sweep(args.sizes, make_case, phase=mode, iters=args.iters,
                flush=not args.no_flush, do_check=args.check)

    for m in (["prefill", "decode"] if args.mode == "both" else [args.mode]):
        run(m)


if __name__ == "__main__":
    main()
