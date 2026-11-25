# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
NumPy correctness oracle for the extracted MoE 3-GEMM SwiGLU kernel (GGUF native block decode,
MOE_GGUF_DP4A path). Mirrors the .cl field offsets / op order (not a from-scratch algorithm).

Supports the formats that actually appear in Qwen3.6-35B-A3B:
  * gate/up experts: Q4_K   -> int8-activation dp4a  (moe_dp_dot_q4_k)
  * down experts:    Q5_K   -> int8-activation dp4a  (moe_dp_dot_q5_k)   [37/40 layers]
                     Q6_K   -> int8-activation dp4a  (moe_dp_dot_q6_k)   [ 3/40 layers]
Every GGUF MoE decode requantizes the activation to per-32 int8 first (moe_gguf_quant_act_i8) —
the reference honours this (act_for) so numerics track each path.
"""
import numpy as np

# fmt -> (block_elem_count, block_byte_size)
GGUF_BLK = {"gguf_q4_k": (256, 144), "gguf_q5_k": (256, 176), "gguf_q6_k": (256, 210), "gguf_q8_0": (32, 34)}
# f16 scale-field byte offsets to seed small (rest stays random)
_D_OFFS = {"gguf_q4_k": [0, 2], "gguf_q5_k": [0, 2], "gguf_q6_k": [208], "gguf_q8_0": [0]}
ACT_GROUP = 32
# formats whose MoE decode requantizes the activation to int8 (all supported GGUF MoE formats)
INT8_ACT = {"gguf_q4_k", "gguf_q5_k", "gguf_q6_k", "gguf_q8_0"}


def load_f16_le(b, off):
    bits = np.uint16(b[off]) | (np.uint16(b[off + 1]) << np.uint16(8))
    return float(np.frombuffer(np.uint16(bits).tobytes(), dtype=np.float16)[0])


def _scale_min_k4(j, q):
    """Q4_K/Q5_K 6-bit packed sub-scale/min unpack (get_scale_min_k4)."""
    if j < 4:
        d = q[j] & 63
        m = q[j + 4] & 63
    else:
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return int(d), int(m)


def _deq_q4k_row(row_bytes, K):
    bpr = K // 256
    B = row_bytes.reshape(bpr, 144)
    out = np.zeros(K, np.float32)
    for kb in range(bpr):
        blk = B[kb]
        d = load_f16_le(blk, 0); dmin = load_f16_le(blk, 2)
        scales = blk[4:16]; qs = blk[16:144].astype(np.int32)
        base = kb * 256; o = 0; is_ = 0; qoff = 0
        for j in range(0, 256, 64):
            sc, m = _scale_min_k4(is_ + 0, scales); d1 = d * sc; m1 = dmin * m
            sc, m = _scale_min_k4(is_ + 1, scales); d2 = d * sc; m2 = dmin * m
            for l in range(32):
                out[base + o] = d1 * float(qs[qoff + l] & 0x0F) - m1; o += 1
            for l in range(32):
                out[base + o] = d2 * float(qs[qoff + l] >> 4) - m2; o += 1
            qoff += 32; is_ += 2
    return out


def _deq_q5k_row(row_bytes, K):
    bpr = K // 256
    B = row_bytes.reshape(bpr, 176)
    out = np.zeros(K, np.float32)
    for kb in range(bpr):
        blk = B[kb]
        d = load_f16_le(blk, 0); dmin = load_f16_le(blk, 2)
        scales = blk[4:16]
        qh = blk[16:48].astype(np.int32); ql = blk[48:176].astype(np.int32)
        base = kb * 256; o = 0; is_ = 0; qloff = 0; u1 = 1; u2 = 2
        for j in range(0, 256, 64):
            sc, m = _scale_min_k4(is_ + 0, scales); d1 = d * sc; m1 = dmin * m
            sc, m = _scale_min_k4(is_ + 1, scales); d2 = d * sc; m2 = dmin * m
            for l in range(32):
                q = (ql[qloff + l] & 0x0F) + (16 if (qh[l] & u1) else 0)
                out[base + o] = d1 * float(q) - m1; o += 1
            for l in range(32):
                q = (ql[qloff + l] >> 4) + (16 if (qh[l] & u2) else 0)
                out[base + o] = d2 * float(q) - m2; o += 1
            qloff += 32; is_ += 2; u1 <<= 2; u2 <<= 2
    return out


def _deq_q6k_row(row_bytes, K):
    bpr = K // 256
    B = row_bytes.reshape(bpr, 210)
    out = np.zeros(K, np.float32)
    for kb in range(bpr):
        blk = B[kb]
        d = load_f16_le(blk, 208)
        base = kb * 256; ql_o = 0; qh_o = 128; sc_o = 192
        for nn in range(0, 256, 128):
            ql = blk[ql_o:ql_o + 64].astype(np.int32)
            qh = blk[qh_o:qh_o + 32].astype(np.int32)
            sc = blk[sc_o:sc_o + 8].view(np.int8).astype(np.int32)
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
                out[base + nn + l + 0] = d * float(sc[is_ + 0]) * q1
                out[base + nn + l + 32] = d * float(sc[is_ + 2]) * q2
                out[base + nn + l + 64] = d * float(sc[is_ + 4]) * q3
                out[base + nn + l + 96] = d * float(sc[is_ + 6]) * q4
            ql_o += 64; qh_o += 32; sc_o += 8
    return out


_DEQ = {"gguf_q4_k": _deq_q4k_row, "gguf_q5_k": _deq_q5k_row, "gguf_q6_k": _deq_q6k_row}


def deq_row(fmt, row_bytes, K):
    """Decode one weight row of K elements from its K/256 contiguous blocks of format `fmt`."""
    return _DEQ[fmt](row_bytes, K)


def gen_expert_weights(fmt, E, N, K, seed=1234, scale_amp=0.02):
    """Valid random per-expert weights of format `fmt`. Returns uint8 [E, N*(K//be)*bb]
    (per-expert contiguous; output row n at n*(K//be)*bb, matching the kernel indexing)."""
    be, bb = GGUF_BLK[fmt]
    assert K % be == 0
    bpr = K // be
    rng = np.random.default_rng(seed)
    W = rng.integers(0, 256, size=(E, N, bpr, bb), dtype=np.uint8)
    for off in _D_OFFS[fmt]:
        vals = rng.uniform(-scale_amp, scale_amp, size=(E, N, bpr)).astype(np.float16)
        vals[vals == 0] = np.float16(scale_amp)
        vb = vals.view(np.uint8).reshape(E, N, bpr, 2)
        W[:, :, :, off:off + 2] = vb
    return W.reshape(E, N * bpr * bb)


def act_i8_dequant(x, G=ACT_GROUP):
    """Per-32 symmetric absmax/127 int8 requant->dequant (moe_gguf_quant_act_i8)."""
    shp = x.shape
    xg = x.astype(np.float32).reshape(-1, G)
    amax = np.abs(xg).max(axis=1)
    sc = amax / 127.0
    inv = np.where(amax > 0, 127.0 / amax, 0.0)
    xq = np.clip(np.round(xg * inv[:, None]), -127, 127)
    return (xq * sc[:, None]).reshape(shp)


def act_for(fmt, x):
    """Activation the MoE decode actually consumes: int8-requant for dp4a formats, raw f16->f32
    for the Q6_K float fallback."""
    return act_i8_dequant(x.astype(np.float32)) if fmt in INT8_ACT else x.astype(np.float32)


def silu(v):
    return v / (1.0 + np.exp(-v))
