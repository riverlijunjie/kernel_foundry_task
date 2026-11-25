"""KernelFoundry custom task: gr_read_dpas.cm (CM / Intel Xe2 GPU).

reference.cm is the unmodified OpenVINO intel_gpu kernel (impls/cm/gated_residual/gr_read_dpas.cm)
with its shared helpers inlined so the file is standalone. kernel.cm starts as an identical copy
and is the file to optimize; only the code between the EVOLVE markers may change.

Gated Residual READ (Qwen4-Exp), one fused systolic kernel, per token row (C = 4 branches,
D = 2560 hidden per branch, L = 320 low-rank width):
    rbar_c  = RMSNorm_D(R_c) * (1 + norm_weight_c)              branch-wise
    t       = SiLU(rbar @ W_down^T / C)                         [L]
    mixed   = (1/C) * sum_c sigmoid(t @ W_up^T)_c * rbar_c      [D]
    gate    = 2 * sigmoid(rbar @ W_inject^T / C)                [C]   (write gate)
One work-group of 16 threads owns 32 token rows (as in the OpenVINO plugin).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from clops import cl
from kernelfoundry import TestBase  # Tests must derive from TestBase

CWD = Path(__file__).parent


# [EVOLVE_START]
KERNEL_NAME = "gr_read_dpas"
C, D, L = 4, 2560, 320
CD = C * D
EPS = 1e-6
# Same options the OpenVINO plugin uses for the gated_residual kernels.
BUILD_OPTIONS = "-cmc -Qxcm_register_file_size=256"

WG = 16  # gr_wg_size
TM = 32  # gr_read_dpas_tm: token rows per work-group
NT = 2  # low-rank 16-wide tiles per phase-2 task

SENTINEL = 7.0  # pre-fill of the outputs, to detect rows the kernel failed to write
PAD_ROWS = 64  # extra sentinel rows behind the outputs, to detect out-of-range writes

TOKENS = [2048, 4096, 8192, 32768, 65536]
CORRECTNESS_ROWS = [1, 7, 32, 33, 100, 1025]


def build_kernel(filename: str):
    """JIT-compile kernel.cm / reference.cm. Dispatch-relevant macros are pinned here, not in the kernel."""
    defines = dict(
        KERNEL_NAME=KERNEL_NAME,
        GR_BRANCHES=C,
        GR_HIDDEN=D,
        GR_LOWRANK=L,
        GR_EPS=EPS,
        GR_NORM_GAIN_PLUS_ONE=1,
        GR_PRODUCE_WRITE_GATE=1,
        GR_WG_SIZE=WG,
        GR_RD_TM=TM,
        GR_RD_NT=NT,
    )
    header = "".join(f"#define {k} {v}\n" for k, v in defines.items())
    return cl.kernels(header + (CWD / filename).read_text(), BUILD_OPTIONS)


@pytest.fixture(scope="session")
def kernel(use_reference):
    """Compiled handle: reference.cm if use_reference else kernel.cm."""
    return build_kernel("reference.cm" if use_reference else "kernel.cm")


class Weights:
    """Model weights (fp16) shared by all cases; scales keep t / gates in a realistic range."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)

        def make(shape, std):
            return (rng.standard_normal(shape, dtype=np.float32) * std).astype(np.float16)

        self.norm_weight = make((CD,), 0.1)  # stored as w, effective gain = 1 + w
        self.w_down = make((L, CD), 0.02)
        self.w_up = make((CD, L), 0.1)
        self.w_inject = make((C, CD), 0.02)
        self.t_norm_weight = cl.tensor(self.norm_weight)
        self.t_w_down = cl.tensor(self.w_down.reshape(-1))
        self.t_w_up = cl.tensor(self.w_up.reshape(-1))
        self.t_w_inject = cl.tensor(self.w_inject.reshape(-1))


_WEIGHTS = None


def weights() -> Weights:
    global _WEIGHTS
    if _WEIGHTS is None:
        _WEIGHTS = Weights()
    return _WEIGHTS


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Case:
    """One problem: residual [rows, C*D] fp16 plus sentinel-filled outputs."""

    def __init__(self, rows: int, seed: int = 0, tile_rows: int | None = None):
        self.rows = rows
        self.w = weights()
        rng = np.random.default_rng(seed)
        if tile_rows is None:
            res = rng.standard_normal((rows, CD), dtype=np.float32)
            # per-row magnitude spread exercises the RMSNorm
            res *= np.exp(rng.uniform(-1.5, 1.5, size=(rows, 1))).astype(np.float32)
            self.residual = res.astype(np.float16)
        else:  # benchmark inputs: tile a small random block instead of generating GBs of randn
            base = rng.standard_normal((tile_rows, CD), dtype=np.float32).astype(np.float16)
            self.residual = np.tile(base, ((rows + tile_rows - 1) // tile_rows, 1))[:rows]
        self.t_residual = cl.tensor(self.residual.reshape(-1))
        self._fresh_outputs()

    def _fresh_outputs(self):
        n = self.rows + PAD_ROWS
        self.t_mixed = cl.tensor(np.full(n * D, SENTINEL, dtype=np.float16))
        self.t_gate = cl.tensor(np.full(n * C, SENTINEL, dtype=np.float16))

    def enqueue(self, handle):
        """Dispatch: gws = [WG, ceil(rows / TM), 1], lws = [WG, 1, 1]."""
        w = self.w
        handle.enqueue(KERNEL_NAME, [WG, (self.rows + TM - 1) // TM, 1], [WG, 1, 1],
                       self.t_residual, w.t_norm_weight, w.t_w_down, w.t_w_up, w.t_w_inject,
                       self.t_mixed, self.t_gate, int(self.rows))

    def run(self, handle):
        """Run once on fresh outputs; returns (mixed [rows, D], gate [rows, C]) as fp32 and checks sentinels."""
        self._fresh_outputs()
        self.enqueue(handle)
        cl.finish()
        mixed = self.t_mixed.numpy().reshape(-1, D)
        gate = self.t_gate.numpy().reshape(-1, C)
        assert (mixed[self.rows:] == SENTINEL).all(), "kernel wrote past the last row of `mixed`"
        assert (gate[self.rows:] == SENTINEL).all(), "kernel wrote past the last row of `write_gate`"
        return mixed[:self.rows].astype(np.float32), gate[:self.rows].astype(np.float32)

    def reference(self):
        """FP32 NumPy model of GatedResidual READ with write gate (inputs are the fp16 values)."""
        w = self.w
        r = self.residual.astype(np.float32).reshape(self.rows, C, D)
        gain = 1.0 + w.norm_weight.astype(np.float32).reshape(C, D)
        inv = 1.0 / np.sqrt((r * r).mean(axis=-1, keepdims=True) + EPS)
        rbar = r * inv * gain
        flat = rbar.reshape(self.rows, CD)
        x = flat @ w.w_down.astype(np.float32).T / C
        t = x * _sigmoid(x)
        g = _sigmoid(t @ w.w_up.astype(np.float32).T).reshape(self.rows, C, D)
        mixed = (g * rbar).sum(axis=1) / C
        gate = 2.0 * _sigmoid(flat @ w.w_inject.astype(np.float32).T / C)
        return mixed, gate


class TestGatedResidualReadDpas(TestBase):
    """Custom task for the Qwen4 Gated Residual READ single-kernel DPAS implementation."""

    @pytest.mark.parametrize("rows", CORRECTNESS_ROWS)
    def test_correctness_wrt_numpy(self, kernel, rows):
        """Merged stream and write gate must match the FP32 NumPy model."""
        case = Case(rows, seed=rows)
        mixed, gate = case.run(kernel)
        exp_mixed, exp_gate = case.reference()
        np.testing.assert_allclose(mixed, exp_mixed, rtol=2e-2, atol=5e-3, err_msg=f"mixed, rows={rows}")
        np.testing.assert_allclose(gate, exp_gate, rtol=1e-2, atol=5e-3, err_msg=f"write_gate, rows={rows}")

    @pytest.mark.parametrize("rows", [33, 2048])
    def test_correctness_wrt_reference(self, kernel, use_reference, rows):
        """Optimized kernel output must match reference.cm on the same inputs."""
        if use_reference:
            pytest.skip("Skipping reference-vs-reference comparison.")
        case = Case(rows, seed=123)
        mixed, gate = case.run(kernel)
        ref_mixed, ref_gate = case.run(build_kernel("reference.cm"))
        np.testing.assert_allclose(mixed, ref_mixed, rtol=1e-2, atol=2e-3)
        np.testing.assert_allclose(gate, ref_gate, rtol=1e-2, atol=2e-3)

    @pytest.mark.performance
    @pytest.mark.parametrize("tokens", TOKENS)
    def test_benchmark(self, kernel, measure_runtime, tokens):
        """Benchmark: time only the kernel dispatch (no host<->device copies)."""
        case = Case(tokens, seed=tokens, tile_rows=2048)

        def run():
            case.enqueue(kernel)

        measure_runtime(run, sync_fn=cl.finish, auto_replicate_inputs_size=0)


# [EVOLVE_END]

if __name__ == "__main__":
    """Quick smoke-test when running this file directly during development."""
    case = Case(100, seed=1)
    mixed, gate = case.run(build_kernel("reference.cm"))
    exp_mixed, exp_gate = case.reference()
    print("max |mixed err|:", np.abs(mixed - exp_mixed).max(), " max |gate err|:", np.abs(gate - exp_gate).max())
