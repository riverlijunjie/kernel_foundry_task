# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Standalone numpy simulation of the NEW MoE Q6_K int8-activation dp4a decoder
(FUNC(moe_dp_dot_q6_k) in moe_3gemm_swiglu_mlp.cl). Replicates the exact SWAR
stream/scale indexing of the kernel and checks it against:
  (a) deq_row(q6k) . act_i8_dequant(x)   -- what dp4a MUST reproduce exactly
  (b) deq_row(q6k) . x_float             -- the ideal full-precision dot
The kernel cannot be compiled on this box (UHD 630 lacks dot_acc_sat_4x8packed),
so this validates the ALGORITHM (indexing + offset folding) end-to-end in numpy.
"""
import numpy as np
import moe_ref as mr

BE, BB = mr.GGUF_BLK["gguf_q6_k"]  # 256, 210


def sim_block_dp4a(blk, xq_i8, asc):
    """Mirror FUNC(moe_dp_dot_q6_k): one 256-elem Q6_K block, int8 act xq_i8[256], 8 scales asc[8]."""
    d = mr.load_f16_le(blk, 208)
    sc = blk[192:208].view(np.int8).astype(np.int64)  # 16 signed scales
    ql_all = blk[0:128].astype(np.int64)
    qh_all = blk[128:192].astype(np.int64)
    aq = xq_i8.astype(np.int64)
    acc = 0.0
    for hf in range(2):
        o = hf * 128
        g = hf * 8
        ag = hf * 4
        ql = ql_all[hf * 64: hf * 64 + 64]
        qh = qh_all[hf * 32: hf * 32 + 32]
        for pr in range(2):
            ql_off = pr * 32
            sh_lo = pr * 2
            sh_hi = pr * 2 + 4
            act_lo = aq[o + pr * 32: o + pr * 32 + 32]
            act_hi = aq[o + pr * 32 + 64: o + pr * 32 + 64 + 32]
            SloA = AloA = ShiA = AhiA = 0
            SloC = AloC = ShiC = AhiC = 0
            for l in range(32):
                wA = (ql[ql_off + l] & 0x0F) | (((qh[l] >> sh_lo) & 0x03) << 4)
                wC = (ql[ql_off + l] >> 4) | (((qh[l] >> sh_hi) & 0x03) << 4)
                if l < 16:
                    SloA += wA * act_lo[l]; AloA += act_lo[l]
                    SloC += wC * act_hi[l]; AloC += act_hi[l]
                else:
                    ShiA += wA * act_lo[l]; AhiA += act_lo[l]
                    ShiC += wC * act_hi[l]; AhiC += act_hi[l]
            acc += d * asc[ag + pr] * (sc[g + 2 * pr] * (SloA - 32 * AloA)
                                       + sc[g + 2 * pr + 1] * (ShiA - 32 * AhiA))
            acc += d * asc[ag + pr + 2] * (sc[g + 2 * pr + 4] * (SloC - 32 * AloC)
                                           + sc[g + 2 * pr + 5] * (ShiC - 32 * AhiC))
    return acc


def quant_act_i8(x, G=32):
    xg = x.astype(np.float32).reshape(-1, G)
    amax = np.abs(xg).max(axis=1)
    inv = np.where(amax > 0, 127.0 / amax, 0.0)
    sc = amax / 127.0
    xq = np.clip(np.round(xg * inv[:, None]), -127, 127).astype(np.int64)
    return xq.reshape(-1), sc  # int8 codes [K], per-32 scale [K/32]


def main():
    rng = np.random.default_rng(7)
    K = 512               # inter_size (down proj contraction) — 2 blocks
    N = 64                # output rows to check
    W = mr.gen_expert_weights("gguf_q6_k", 1, N, K, seed=99, scale_amp=0.01)[0]
    row_bytes = (K // BE) * BB
    Wrows = W.reshape(N, row_bytes)
    x = (rng.standard_normal(K) * 0.2).astype(np.float16)

    xq, xsc = quant_act_i8(x)                       # int8 codes + per-32 scales
    x_i8_deq = (xq.reshape(-1, 32) * xsc[:, None]).reshape(-1)  # dequantized activation

    errs_vs_i8 = []
    errs_vs_full = []
    for n in range(N):
        rb = Wrows[n]
        wdeq = mr.deq_row("gguf_q6_k", rb, K)       # true dequant weight row
        ref_i8 = float(wdeq @ x_i8_deq)             # what dp4a must equal
        ref_full = float(wdeq @ x.astype(np.float32))
        # simulate kernel over the 2 blocks
        sim = 0.0
        for b in range(K // BE):
            blk = rb.reshape(K // BE, BB)[b]
            sim += sim_block_dp4a(blk, xq[b * BE:(b + 1) * BE], xsc[b * (BE // 32):(b + 1) * (BE // 32)])
        errs_vs_i8.append(abs(sim - ref_i8) / (abs(ref_i8) + 1e-6))
        errs_vs_full.append(abs(sim - ref_full) / (abs(ref_full) + 1e-6))

    errs_vs_i8 = np.array(errs_vs_i8)
    errs_vs_full = np.array(errs_vs_full)
    print("Q6_K dp4a decoder simulation (K=%d, %d rows)" % (K, N))
    print("  vs int8-dequant dot (must be ~0): max_rel=%.3e  mean_rel=%.3e" %
          (errs_vs_i8.max(), errs_vs_i8.mean()))
    print("  vs full-precision dot (int8 err): max_rel=%.3e  mean_rel=%.3e" %
          (errs_vs_full.max(), errs_vs_full.mean()))
    ok_i8 = errs_vs_i8.max() < 1e-4          # exact match to int8 path (only fp summation order)
    ok_full = errs_vs_full.mean() < 3e-2     # within int8 quant tolerance of the ideal dot
    print("  RESULT: %s (algorithm) / %s (int8 tolerance)" %
          ("PASS" if ok_i8 else "FAIL", "PASS" if ok_full else "FAIL"))
    return 0 if (ok_i8 and ok_full) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
