"""KernelFoundry custom task: qsa_sparse_attention_gqa_dpas.cm (CM / Intel BMG).

reference.cm is an unmodified copy of
workspace/qsa/cm_kernel/kernels/qsa_sparse_attention_gqa_dpas.cm - a DPAS-based,
paged-attention, GQA (grouped-query attention) sparse-attention kernel used by the
QSA (query sparse attention) OpenVINO plugin. kernel.cm starts as an identical copy
and is the file to be optimized; only the code inside its markers may change.

Model shapes (heads, head_dim, GQA group size, compress ratio, top-k budget, output
gate, ...) are pinned constants in qsa_config.py, copied from the original model's
config.json (see that module's docstring for provenance); config.json itself is
not shipped in this task package since only a handful of its fields matter here.

Each work-group computes the attention output for ONE (token, KV head) pair: it
gathers the top-k selected compressed KV blocks (paged KV cache layout) plus the
always-visible causal tail, and performs a DPAS-based online-softmax attention over
them for all query heads that share that KV head (the GQA group).
"""

from __future__ import annotations

from pathlib import Path

import clops

import numpy as np
import pytest

import qsa_config
import qsa_harness as H
import qsa_ref
from kernelfoundry import TestBase

CWD = Path(__file__).parent
CFG = qsa_config.load()
KERNEL_NAME = "qsa_sparse_attention_gqa_dpas"

# Grid-layout macros locked to their validated-optimal values (see
# .agents/skills/qsa_opt/SKILL.md §3 "Validated optimizations"), not left to the
# kernel's own #ifndef defaults or to the evolve loop:
#   QSA_GQA_HEAD_MAJOR=1  swaps which NDRange axis (token/entry vs. kv_head) is
#     the fast-varying (group-id) one, so adjacent work-groups are more likely
#     to share a KV head - measured ~1.35x/1.23x on 64K/128K-token long-context
#     prefill; pure scheduling change, same addresses/arithmetic/results.
#   QSA_GQA_VECTOR_META=1 batches the per-step selected-block-ID/physical-
#     page-ID reads into masked vector gathers instead of one scalar load per
#     slot; small gain alone, multiplies with head-major when combined.
# Case.enqueue() below hardcodes the dispatch grid to match the
# QSA_GQA_HEAD_MAJOR=1 axis mapping (entry is the group-id axis, kv_head is the
# direct global-id axis) - if this value changes, the grid must change with it.
# Also, since these are passed as JIT #defines prepended before the flattened
# file body, they take effect before the in-file #ifndef guards are even
# reached, so editing those guarded defaults inside
# has no effect - these two knobs are simply not part of the evolvable code.
FIXED_JIT = dict(QSA_GQA_HEAD_MAJOR=1, QSA_GQA_VECTOR_META=1)


def load_qsa_kernel(filename: str):
    """Compile kernel.cm or reference.cm.

    Both files define the same KERNEL_NAME (qsa_sparse_attention_gqa_dpas) entry
    point via the JIT #define below, so either can be dispatched identically.
    """
    jit = CFG.jit(KERNEL_NAME, **FIXED_JIT)
    return H.build_kernel(filename, jit)


@pytest.fixture(scope="session")
def kernel(use_reference):
    """Compiled CM kernel handle: reference.cm if use_reference else kernel.cm."""
    return load_qsa_kernel("reference.cm" if use_reference else "kernel.cm")


class Case:
    """One paged-attention QSA/GQA test case.

    ``seqs`` is a list of (past_len, num_new_tokens) per sequence, matching
    qsa_harness.make_batch. Builds random Q/K/V and a plausible top-k block
    selection, and can enqueue the compiled CM kernel and/or compute the FP32
    NumPy ground truth for the same inputs.
    """

    def __init__(self, cfg, seqs, *, seed=0):
        from clops import cl

        self.cfg = cfg
        self.batch = H.make_batch(seqs, cfg.pa_block_size, seed=seed)
        self.token_map, self.num_entries = H.make_score_tile_map(self.batch, tile=1)
        assert self.num_entries == self.batch.num_tokens

        Hq, Hkv = cfg.heads_num, cfg.kv_heads_num
        Dh, Dv = cfg.k_head_size, cfg.v_head_size
        gate = cfg.has_output_gate
        self.qhs = 2 * Dh if gate else Dh
        self.gd = Dh if gate else 0
        self.q_pitch = Hq * self.qhs

        rng = np.random.default_rng(seed)
        self.query = H.rand_f16(rng, self.batch.num_tokens, self.q_pitch)
        self.kc = H.rand_f16(rng, self.batch.num_blocks, Hkv, cfg.pa_block_size, Dh)
        self.vc = H.rand_f16(rng, self.batch.num_blocks, Hkv, cfg.pa_block_size, Dv)
        self.sel, self.cnt = H.make_selection(self.batch, cfg.compress_ratio, cfg.block_topk, seed=seed)

        self.t_q = cl.tensor(self.query)
        self.t_kc, self.t_vc = cl.tensor(self.kc.reshape(-1)), cl.tensor(self.vc.reshape(-1))
        self.t_past = cl.tensor(self.batch.past_lens)
        self.t_sub = cl.tensor(self.batch.subsequence_begins)
        self.t_bidx = cl.tensor(self.batch.block_indices)
        self.t_bib = cl.tensor(self.batch.block_indices_begins)
        self.t_sel, self.t_cnt = cl.tensor(self.sel), cl.tensor(self.cnt)
        self.t_tile = cl.tensor(self.token_map)
        self.t_out = cl.tensor(np.zeros((self.batch.num_tokens, Hq * Dv), np.float16))

    def enqueue(self, kernel_handle, *, guard=False):
        """Dispatch the kernel. guard=True adds one extra kv_head/token work-group
        row/column (which the kernel bounds-checks and no-ops on) to catch
        out-of-bounds writes; benchmarks should use guard=False.
        """
        g = int(guard)
        # QSA_GQA_HEAD_MAJOR=1 (FIXED_JIT above): entry is the group-id axis
        # (axis 0, local size 4 for the GQA_W worker split) and kv_head is the
        # direct global-id axis (axis 1, local size 1) - the opposite of the
        # HEAD_MAJOR=0 mapping.
        grid = [(self.num_entries + g) * 4, self.cfg.kv_heads_num + g, 1]
        kernel_handle.enqueue(KERNEL_NAME, grid, [4, 1, 1],
            self.t_q, self.t_kc, self.t_vc, self.t_past, self.t_sub,
            self.t_bidx, self.t_bib, self.t_sel, self.t_cnt, self.t_tile,
            self.t_out, self.num_entries, 0, self.q_pitch, self.qhs, self.gd)

    def result(self):
        Hq, Dv = self.cfg.heads_num, self.cfg.v_head_size
        return self.t_out.numpy().reshape(self.batch.num_tokens, Hq * Dv).astype(np.float32)

    def reference(self):
        return qsa_ref.sparse_attention_ref(self.cfg, self.batch, self.query, self.kc, self.vc,
                                            self.sel, self.cnt, q_head_stride=self.qhs, gate_delta=self.gd)


# Correctness cases: kept small (unlike the benchmark sizes) so the pure-Python
# FP32 NumPy reference stays fast. Covers a plain dense prefill, a sequence that
# continues from existing KV history (nonzero past_lens), and a mixed batch of
# several sequences (subsequence_begins / block_indices bookkeeping).
CORRECTNESS_CASES = {
    "dense_prefill": [(0, 130)],
    "with_history": [(300, 64)],
    "multi_seq": [(0, 64), (128, 96), (500, 40)],
}

# Benchmark sizes: input token counts (single-sequence causal prefill, no KV
# history) as requested for this task, sweeping short to long context.
BENCHMARK_SIZES = [2048, 4096, 8192, 16384]


class TestQSASparseAttentionGQADPAS(TestBase):
    """Custom task for the QSA GQA-DPAS sparse-attention CM kernel."""

    @pytest.mark.parametrize("case_name", list(CORRECTNESS_CASES))
    def test_correctness_wrt_numpy(self, kernel, case_name):
        """Kernel output vs. FP32 NumPy scaled-dot-product-attention ground truth."""
        from clops import cl

        case = Case(CFG, CORRECTNESS_CASES[case_name], seed=hash(case_name) & 0xFFFF)
        case.enqueue(kernel, guard=True)
        cl.finish()

        got = case.result()
        expected = case.reference()
        assert np.isfinite(got).all(), "kernel produced non-finite output"
        np.testing.assert_allclose(got, expected, atol=2e-3, rtol=2e-3,
                                   err_msg=f"case={case_name}")

    def test_correctness_wrt_reference(self, kernel, use_reference):
        """Verify the optimized kernel output matches reference.cm bit-for-bit inputs."""
        if use_reference:
            pytest.skip("Skipping reference-vs-reference comparison.")
        from clops import cl

        seqs = CORRECTNESS_CASES["multi_seq"]
        case = Case(CFG, seqs, seed=123)
        case.enqueue(kernel, guard=True)
        cl.finish()
        got = case.result()

        ref_kernel = load_qsa_kernel("reference.cm")
        case_ref = Case(CFG, seqs, seed=123)
        case_ref.enqueue(ref_kernel, guard=True)
        cl.finish()
        expected = case_ref.result()

        np.testing.assert_allclose(got, expected, atol=2e-3, rtol=2e-3)

    @pytest.mark.performance
    @pytest.mark.parametrize("num_tokens", BENCHMARK_SIZES)
    def test_benchmark_prefill(self, kernel, measure_runtime, num_tokens):
        """Benchmark: single-sequence causal prefill sweep over context length.

        Only the kernel dispatch is timed (no host<->device copies); no KV
        history (past_len=0), i.e. num_tokens is the input/context token count.
        """
        from clops import cl

        case = Case(CFG, [(0, num_tokens)], seed=num_tokens)

        def enqueue():
            case.enqueue(kernel, guard=False)

        measure_runtime(enqueue, sync_fn=cl.finish, auto_replicate_inputs_size=0)


if __name__ == "__main__":
    """Quick smoke-test when running this file directly during development."""
    from clops import cl

    ref_kernel = load_qsa_kernel("reference.cm")
    case = Case(CFG, [(0, 256)], seed=1)
    case.enqueue(ref_kernel, guard=True)
    cl.finish()
    got = case.result()
    expected = case.reference()
    print("Max abs error:", np.abs(got - expected).max())
