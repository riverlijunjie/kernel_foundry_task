"""QSA parameters, pinned from the model's config.json (Qwen4Exp text_config).

The plugin builds every QSA CM kernel with a set of JIT ``#define``s (see
``make_qsa_jit`` in ``qsa.cpp``). This module reproduces those constants as
plain literals so this standalone task package does not need to ship or parse
the (large, mostly-irrelevant) HuggingFace ``config.json`` at runtime - only the
handful of fields that actually shape the QSA GQA-DPAS kernel are kept here,
copied once from ``workspace/qsa/config.json``'s ``text_config``:

    num_attention_heads=24, num_key_value_heads=2, head_dim=256,
    hidden_size=2560, indexer_n_heads=4, indexer_kv_heads=1,
    indexer_head_dim=128, partial_rotary_factor=0.25,
    indexer_compress_ratio=4, indexer_budget=2048,
    output_gate_type="sigmoid", rms_norm_eps=1e-6.

``block_topk`` is not spelled out in config.json: the model keeps a token
``budget`` and selects whole QSA blocks of ``r`` tokens, so ``block_topk``
defaults to ``ceil(budget / r)``. Override it (and the sequence shape) from
each test's CLI/parametrization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- pinned from workspace/qsa/config.json (text_config) ---
_HEAD_DIM = 256
_INDEXER_HEAD_DIM = 128
_PARTIAL_ROTARY_FACTOR = 0.25
_COMPRESS_RATIO = 4
_BUDGET = 2048


@dataclass
class QSAConfig:
    # --- main attention ---
    heads_num: int          # Hq  (num_attention_heads)
    kv_heads_num: int       # Hkv (num_key_value_heads)
    k_head_size: int        # Dh  (head_dim)
    v_head_size: int        # Dv  (head_dim)
    hidden_size: int        # D   (hidden_size)
    # --- indexer ---
    indexer_heads: int      # Hidx (indexer_n_heads)
    indexer_kv_heads: int   # (indexer_kv_heads, always 1 here)
    indexer_head_dim: int   # Di  (indexer_head_dim)
    indexer_rotary_dim: int # rotary_dim = Di * partial_rotary_factor
    compress_ratio: int     # r   (indexer_compress_ratio)
    budget: int             # token budget (indexer_budget)
    block_topk: int         # blocks selected per query (derived: ceil(budget/r))
    # --- paged attention / misc ---
    pa_block_size: int      # PA page size (tokens per page)
    has_output_gate: bool   # output_gate_type == "sigmoid"
    is_causal: bool
    scale: float            # 1/sqrt(Dh)
    indexer_scale: float    # 1/sqrt(Di)
    indexer_norm_eps: float

    # derived JIT-only fields
    max_selected: int = field(default=0)

    @property
    def n_rep(self) -> int:
        return self.heads_num // self.kv_heads_num

    @property
    def summaries_per_page(self) -> int:
        return self.pa_block_size // self.compress_ratio

    def jit(self, kernel_name: str, **extra) -> dict:
        """The JIT #define set shared by every QSA kernel (mirrors make_qsa_jit)."""
        d = {
            "QSA_HEADS": self.heads_num,
            "QSA_KV_HEADS": self.kv_heads_num,
            "QSA_K_HEAD_SIZE": self.k_head_size,
            "QSA_V_HEAD_SIZE": self.v_head_size,
            "QSA_HIDDEN": self.hidden_size,
            "QSA_IDX_HEADS": self.indexer_heads,
            "QSA_IDX_HEAD_DIM": self.indexer_head_dim,
            "QSA_IDX_ROTARY_DIM": self.indexer_rotary_dim,
            "QSA_COMPRESS_RATIO": self.compress_ratio,
            "QSA_BLOCK_TOPK": self.block_topk,
            "QSA_BUDGET": self.budget,
            "QSA_MAX_SELECTED": self.max_selected or (self.block_topk * self.compress_ratio),
            "QSA_PA_BLOCK_SIZE": self.pa_block_size,
            "QSA_HAS_OUTPUT_GATE": 1 if self.has_output_gate else 0,
            "QSA_IS_CAUSAL": 1 if self.is_causal else 0,
            "QSA_SCALE": _fl(self.scale),
            "QSA_IDX_SCALE": _fl(self.indexer_scale),
            "QSA_IDX_NORM_EPS": _fl(self.indexer_norm_eps),
            "KERNEL_NAME": kernel_name,
        }
        d.update(extra)
        return d


def _fl(x: float) -> str:
    return f"{float(x):.8g}f"


def load(*, pa_block_size: int = 16, block_topk: int | None = None) -> QSAConfig:
    """Build the pinned QSA parameters (see module docstring for provenance).

    pa_block_size is the paged-attention page size; the OpenVINO default is 16
    (``cldnn::paged_attention::block_size``), which is what the production QSA
    tests use. block_topk overrides the ``ceil(budget/r)`` default.
    """
    topk = block_topk if block_topk is not None else math.ceil(_BUDGET / _COMPRESS_RATIO)

    return QSAConfig(
        heads_num=24,
        kv_heads_num=2,
        k_head_size=_HEAD_DIM,
        v_head_size=_HEAD_DIM,
        hidden_size=2560,
        indexer_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=_INDEXER_HEAD_DIM,
        indexer_rotary_dim=int(round(_INDEXER_HEAD_DIM * _PARTIAL_ROTARY_FACTOR)),
        compress_ratio=_COMPRESS_RATIO,
        budget=_BUDGET,
        block_topk=topk,
        pa_block_size=pa_block_size,
        has_output_gate=True,  # output_gate_type == "sigmoid"
        is_causal=True,
        scale=1.0 / math.sqrt(_HEAD_DIM),
        indexer_scale=1.0 / math.sqrt(_INDEXER_HEAD_DIM),
        indexer_norm_eps=1e-6,
    )


if __name__ == "__main__":
    c = load()
    import pprint
    pprint.pprint(c.__dict__)
    print("n_rep =", c.n_rep, "summaries_per_page =", c.summaries_per_page)
