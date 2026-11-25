"""KernelFoundry custom task: qsa_indexer_topk.cm (CM / Intel Xe GPU).

reference.cm is the unmodified OpenVINO intel_gpu kernel
(impls/cm/qsa_indexer/qsa_indexer_topk.cm) with its two shared helper headers inlined so the file
is standalone. kernel.cm starts as an identical copy and is the file to optimize; only the code
between may change.

The kernel selects, for every query token, the top-K QSA blocks (K = QSA_BLOCK_TOPK) out of the
token's visible blocks, from a float32 score row [max_blocks]. Contract:
  * ordering: score descending, lower block id wins ties;
  * selected_blocks[token, :k] are the chosen ids in ASCENDING order, padded with -1;
  * selected_counts[token] = k = min(n_visible_blocks, K).
One work-group of QSA_TOPK_WG threads handles one token (decode: 16, prefill: 8, as in OpenVINO).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from clops import cl
from kernelfoundry import TestBase  # Tests must derive from TestBase

CWD = Path(__file__).parent

KERNEL_NAME = "qsa_indexer_topk"
# Qwen4Exp config: indexer_compress_ratio=4, indexer_budget=2048 -> block_topk = 2048 / 4.
COMPRESS_RATIO = 4
BLOCK_TOPK = 512
# Same options the OpenVINO plugin uses for qsa_indexer kernels.
BUILD_OPTIONS = "-cmc -Qxcm_register_file_size=256"

DENSE, PAGED = 1, 0  # QSA_ADDRESS_MODE
DECODE_WG, PREFILL_WG = 16, 8  # work-group sizes used by the plugin (topk_wg_size)

SENTINEL = -7  # pre-fill of the output buffers, to detect rows/entries the kernel failed to write
PAD_SCORE = 1e30  # scores beyond the visible blocks; must never be read


def build_kernel(filename: str, address_mode: int, wg: int):
    """JIT-compile kernel.cm / reference.cm. Dispatch-relevant macros are pinned here, not in the kernel."""
    defines = dict(
        KERNEL_NAME=KERNEL_NAME,
        QSA_BLOCK_TOPK=BLOCK_TOPK,
        QSA_COMPRESS_RATIO=COMPRESS_RATIO,
        QSA_ADDRESS_MODE=address_mode,
        QSA_TOPK_WG=wg,
    )
    header = "".join(f"#define {k} {v}\n" for k, v in defines.items())
    return cl.kernels(header + (CWD / filename).read_text(), BUILD_OPTIONS)


@pytest.fixture(scope="session")
def kernel_factory(use_reference):
    """Returns get(address_mode, wg) -> compiled handle (reference.cm if use_reference else kernel.cm)."""
    filename = "reference.cm" if use_reference else "kernel.cm"
    cache = {}

    def get(address_mode: int, wg: int):
        key = (address_mode, wg)
        if key not in cache:
            cache[key] = build_kernel(filename, address_mode, wg)
        return cache[key]

    return get


class Case:
    """One top-k finalize problem: score rows + per-token visible-block counts.

    Dense mode: `past_lens` is [B], every batch element has `seq_len` query tokens (token = b * seq_len + l).
    Paged mode: `seqs` is a list of (past_len, num_new_tokens); tokens are concatenated.
    Only rows [row_begin, row_end) are scored (block_scores holds just those rows).
    """

    def __init__(self, *, mode=DENSE, past_lens=None, seq_len=1, seqs=None, row_begin=0, row_end=None,
                 ties=False, seed=0):
        self.mode = mode
        if mode == DENSE:
            self.past = np.asarray(past_lens, dtype=np.int32)
            self.seq_len = int(seq_len)
            b = np.arange(len(self.past) * self.seq_len) // self.seq_len
            abs_pos = self.past[b] + (np.arange(len(b)) - b * self.seq_len)
            self.sub_begins = None
        else:
            self.past = np.asarray([p for p, _ in seqs], dtype=np.int32)
            self.sub_begins = np.concatenate([[0], np.cumsum([q for _, q in seqs])]).astype(np.int32)
            abs_pos = np.concatenate([p + np.arange(q) for p, q in seqs])
            self.seq_len = 0
        self.num_seqs = len(self.past)
        self.num_tokens = int(len(abs_pos))
        self.n = ((abs_pos + 1) // COMPRESS_RATIO).astype(np.int64)  # visible blocks per token
        self.row_begin = int(row_begin)
        self.row_end = self.num_tokens if row_end is None else int(row_end)
        rows = self.row_end - self.row_begin
        self.max_blocks = int(((int(self.n.max()) + 7) // 8) * 8 + 8)

        rng = np.random.default_rng(seed)
        scores = rng.standard_normal((rows, self.max_blocks), dtype=np.float32)
        if ties:  # few distinct values -> exercises the "block id ascending" tie-break
            scores = np.floor(scores * 2) / np.float32(2)
        for r in range(rows):
            scores[r, self.n[self.row_begin + r]:] = PAD_SCORE
        self.scores = scores

        self.t_scores = cl.tensor(self.scores.reshape(-1))
        self.t_past = cl.tensor(self.past)
        self.t_sub = cl.tensor(self.sub_begins) if self.sub_begins is not None else None
        self.t_sel = self.t_cnt = None

    def _fresh_outputs(self):
        self.t_sel = cl.tensor(np.full(self.num_tokens * BLOCK_TOPK, SENTINEL, dtype=np.int32))
        self.t_cnt = cl.tensor(np.full(self.num_tokens, SENTINEL, dtype=np.int32))

    def enqueue(self, handle, wg: int):
        """Dispatch: one work-group of `wg` threads per row in [row_begin, row_end)."""
        if self.t_sel is None:
            self._fresh_outputs()
        rows = self.row_end - self.row_begin
        grid, local = [rows * wg, 1, 1], [wg, 1, 1]
        tail = (self.num_tokens, self.num_seqs if self.mode == PAGED else self.seq_len,
                self.max_blocks, self.row_begin, self.row_end)
        if self.mode == DENSE:
            handle.enqueue(KERNEL_NAME, grid, local, self.t_scores, self.t_past,
                           self.t_sel, self.t_cnt, *tail)
        else:
            handle.enqueue(KERNEL_NAME, grid, local, self.t_scores, self.t_past, self.t_sub,
                           self.t_sel, self.t_cnt, *tail)

    def run(self, handle, wg: int):
        """Run once on fresh outputs and return (selected_blocks, selected_counts) as numpy."""
        self._fresh_outputs()
        self.enqueue(handle, wg)
        cl.finish()
        return (self.t_sel.numpy().reshape(self.num_tokens, BLOCK_TOPK),
                self.t_cnt.numpy().reshape(self.num_tokens))

    def reference(self):
        """NumPy ground truth; rows outside [row_begin, row_end) stay untouched (SENTINEL)."""
        sel = np.full((self.num_tokens, BLOCK_TOPK), SENTINEL, dtype=np.int32)
        cnt = np.full(self.num_tokens, SENTINEL, dtype=np.int32)
        for t in range(self.row_begin, self.row_end):
            n = int(self.n[t])
            k = min(n, BLOCK_TOPK)
            cnt[t] = k
            sel[t, k:] = -1
            if k == n:
                sel[t, :k] = np.arange(k)
            else:
                s = self.scores[t - self.row_begin, :n]
                order = np.lexsort((np.arange(n), -s))  # score desc, id asc
                sel[t, :k] = np.sort(order[:k])
        return sel, cnt


# name -> (Case kwargs, work-group size)
CORRECTNESS_CASES = {
    # n = 0, 1, 256 (dense prefix), 513 (first row that really selects), 1500, 10000 blocks
    "decode_mixed": (dict(past_lens=[0, 5, 1023, 2051, 6000, 40000], seq_len=1), DECODE_WG),
    "decode_ties": (dict(past_lens=[2051, 6000, 40000], seq_len=1, ties=True), DECODE_WG),
    # n = 35000 blocks: per-worker chunk exceeds the 2048-key register cache (uncached rescan path)
    "decode_long": (dict(past_lens=[140000], seq_len=1), DECODE_WG),
    "decode_long_ties": (dict(past_lens=[140000], seq_len=1, ties=True), DECODE_WG),
    "prefill": (dict(past_lens=[0, 3000], seq_len=700), PREFILL_WG),
    "prefill_ties": (dict(past_lens=[0, 3000], seq_len=700, ties=True), PREFILL_WG),
    # bounded-scratch chunking: only rows [2, 6) are scored/written
    "row_chunk": (dict(past_lens=[1000, 5000, 9000, 20000, 30000, 2051, 0, 80000], seq_len=1,
                       row_begin=2, row_end=6), DECODE_WG),
    "paged": (dict(mode=PAGED, seqs=[(0, 50), (7000, 40), (30000, 1)]), PREFILL_WG),
}

# (mode, batch / num sequences, tokens): decode = `batch` sequences with `ctx` history tokens and one
# new token each; prefill = one sequence of `ctx` tokens without history.
BENCHMARKS = [
    ("decode", 1, 8192),
    ("decode", 1, 32768),
    ("decode", 1, 131072),
    ("decode", 8, 32768),
    ("prefill", 1, 4096),
    ("prefill", 1, 8192),
    ("prefill", 1, 16384),
]


class TestQSAIndexerTopk(TestBase):
    """Custom task for the QSA indexer top-k finalize CM kernel."""

    @pytest.mark.parametrize("case_name", list(CORRECTNESS_CASES))
    def test_correctness_wrt_numpy(self, kernel_factory, case_name):
        """Selected ids / counts must match the NumPy top-k exactly."""
        kwargs, wg = CORRECTNESS_CASES[case_name]
        case = Case(**kwargs, seed=len(case_name))
        sel, cnt = case.run(kernel_factory(case.mode, wg), wg)
        exp_sel, exp_cnt = case.reference()
        np.testing.assert_array_equal(cnt, exp_cnt, err_msg=f"selected_counts, case={case_name}")
        np.testing.assert_array_equal(sel, exp_sel, err_msg=f"selected_blocks, case={case_name}")

    @pytest.mark.parametrize("case_name", ["decode_mixed", "decode_ties", "prefill_ties", "paged"])
    def test_correctness_wrt_reference(self, kernel_factory, use_reference, case_name):
        """Optimized kernel output must be identical to reference.cm on the same inputs."""
        if use_reference:
            pytest.skip("Skipping reference-vs-reference comparison.")
        kwargs, wg = CORRECTNESS_CASES[case_name]
        case = Case(**kwargs, seed=123)
        sel, cnt = case.run(kernel_factory(case.mode, wg), wg)
        ref_handle = build_kernel("reference.cm", case.mode, wg)
        ref_sel, ref_cnt = case.run(ref_handle, wg)
        np.testing.assert_array_equal(cnt, ref_cnt)
        np.testing.assert_array_equal(sel, ref_sel)

    @pytest.mark.performance
    @pytest.mark.parametrize("mode, batch, size", BENCHMARKS)
    def test_benchmark(self, kernel_factory, measure_runtime, mode, batch, size):
        """Benchmark: time only the kernel dispatch (no host<->device copies)."""
        if mode == "decode":
            case, wg = Case(past_lens=[size] * batch, seq_len=1, seed=size), DECODE_WG
        else:
            case, wg = Case(past_lens=[0] * batch, seq_len=size, seed=size), PREFILL_WG
        handle = kernel_factory(case.mode, wg)
        case._fresh_outputs()

        def run():
            case.enqueue(handle, wg)

        measure_runtime(run, sync_fn=cl.finish, auto_replicate_inputs_size=0)


if __name__ == "__main__":
    """Quick smoke-test when running this file directly during development."""
    kwargs, wg = CORRECTNESS_CASES["decode_mixed"]
    case = Case(**kwargs, seed=1)
    sel, cnt = case.run(build_kernel("reference.cm", case.mode, wg), wg)
    exp_sel, exp_cnt = case.reference()
    print("counts equal:", np.array_equal(cnt, exp_cnt), " blocks equal:", np.array_equal(sel, exp_sel))
