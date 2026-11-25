"""Numpy references for the QSA attention kernels (fp32 math, f16 inputs)."""

from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sparse_attention_ref(cfg, batch, query, key_cache, value_cache,
                         selected_blocks, selected_counts, *, q_head_stride, gate_delta):
    """Reference for qsa_sparse_attention(.dpas): scaled-dot-product softmax over the
    selected complete QSA blocks plus the always-visible incomplete tail.

    query is [num_tokens, q_pitch] with per-head layout [q(Dh) | gate(Dv)] when the
    output gate is on. Returns output [num_tokens, Hq*Dv] float32.
    """
    Hq, Dh, Dv, r, pab = cfg.heads_num, cfg.k_head_size, cfg.v_head_size, cfg.compress_ratio, cfg.pa_block_size
    n_rep = cfg.n_rep
    K_TOPK = selected_blocks.shape[1]
    q = query.astype(np.float32).reshape(-1)
    kc = key_cache.astype(np.float32).reshape(batch.num_blocks, cfg.kv_heads_num, pab, Dh)
    vc = value_cache.astype(np.float32).reshape(batch.num_blocks, cfg.kv_heads_num, pab, Dv)
    out = np.zeros((batch.num_tokens, Hq * Dv), dtype=np.float32)

    for tok in range(batch.num_tokens):
        seq, abs_pos = batch.seq_of_token(tok)
        page_begin = int(batch.block_indices_begins[seq])
        n_complete = (abs_pos + 1) // r

        def kv_at(pos, kv_head):
            page = int(batch.block_indices[page_begin + pos // pab])
            tip = pos % pab
            return kc[page, kv_head, tip], vc[page, kv_head, tip]

        for head in range(Hq):
            kv_head = head // n_rep
            base = tok * (Hq * q_head_stride) + head * q_head_stride
            qh = q[base:base + Dh] * cfg.scale

            positions = []
            cnt = int(selected_counts[tok])
            for i in range(min(cnt, K_TOPK)):
                b = int(selected_blocks[tok, i])
                if b < 0:
                    continue
                positions.extend(range(b * r, b * r + r))
            positions.extend(range(n_complete * r, abs_pos + 1))

            m, s = -1e30, 0.0
            acc = np.zeros(Dv, dtype=np.float32)
            for pos in positions:
                k, v = kv_at(pos, kv_head)
                score = float(np.dot(qh, k))
                nm = max(m, score)
                corr = np.exp(m - nm)
                w = np.exp(score - nm)
                s = s * corr + w
                acc = acc * corr + v * w
                m = nm
            o = acc / s if s > 0 else acc
            if gate_delta:
                g = q[base + gate_delta: base + gate_delta + Dv]
                o = o * sigmoid(g)
            out[tok, head * Dv:(head + 1) * Dv] = o
    return out
