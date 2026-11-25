"""KernelFoundry task: GatedResidual READ / MIX as one systolic kernel (gr_read_dpas) of the OpenVINO intel_gpu plugin.

kernel.cl / reference.cl are extracted from ocl_v2/gr_read_dpas.cl (f16 weights). Per token row, with C branches of D channels
and a low-rank width L (Qwen4-exp: C = 4, D = 2560, L = 320):
    rbar_c = RMSNorm_D(R_c) * norm_weight_c
    t      = SiLU(rbar @ W_down^T / C)                          [L]
    mixed  = (1/C) * sum_c sigmoid(t @ W_up^T)_c * rbar_c       [D]
    gate   = 2 * sigmoid(rbar @ W_inject^T / C)                 [C]     (write gate, GR_PRODUCE_WRITE_GATE = 1 only)

How the kernel is compiled and dispatched is declared by the kernel source in a [LAUNCH_START] / [LAUNCH_END] comment block
(extra -D macros, 256-GRF mode, build options), see parse_launch(). GR_WG_SIZE and GR_RD_TM of that block define the grid:
gws = [GR_WG_SIZE * 16, ceil(rows / GR_RD_TM), 1], lws = [GR_WG_SIZE * 16, 1, 1].
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyopencl as cl
import pytest
from kernelfoundry import TestBase  # Tests must derive from TestBase

# Fixtures provided via conftest.py from kernelfoundry: use_reference, profile_store, measure_runtime, request

HERE = Path(__file__).parent
KERNEL_FILE = "kernel.cl"
REFERENCE_FILE = "reference.cl"
ENTRY_POINT = "gr_read_dpas"

# ===================== Model geometry (Qwen4-exp) =====================
C, D, L = 4, 2560, 320
CD = C * D
EPS = 1e-6
SUBGROUP_SIZE = 16

SENTINEL = 7.0  # pre-fill of the outputs: detects rows the kernel failed to write
PAD_ROWS = 64  # sentinel rows behind the outputs: detects writes past the last row

BENCH_ROWS = [2048, 4096, 8192, 32768, 65536]
CORRECTNESS_ROWS = [1, 7, 32, 33, 100, 1025]
BENCH_TILE_ROWS = 2048  # benchmark inputs tile a random block instead of generating GBs of randn

DEFAULT_LAUNCH = {
    "defines": "GR_WG_SIZE=16 GR_RD_TM=32 GR_RD_NT=2 GR_NV_KD=8 GR_RD_SLM_GAIN=2",
    "grf_256": "1",
    "build_options": "",
}

_LAUNCH_BLOCK = re.compile(r"//\s*\[LAUNCH_START\](.*?)//\s*\[LAUNCH_END\]", re.S)
_LAUNCH_LINE = re.compile(r"^\s*//\s*(defines|grf_256|build_options)\s*=\s*(.*?)\s*$")
_DEFINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.+\-]+$")
_BUILD_OPTION = re.compile(r"^-cl-[A-Za-z0-9=.\-]+$")


@dataclass(frozen=True)
class LaunchConfig:
    defines: tuple  # extra (name, value) macros
    wg_size: int  # subgroups per work-group (GR_WG_SIZE)
    tile_rows: int  # token rows per work-group (GR_RD_TM)
    grf_256: bool
    build_options: tuple


def parse_launch(source: str) -> LaunchConfig:
    """Reads the `key = value` lines of the [LAUNCH_START] / [LAUNCH_END] comment block of a kernel source."""
    values = dict(DEFAULT_LAUNCH)
    block = _LAUNCH_BLOCK.search(source)
    if block:
        for line in block.group(1).splitlines():
            match = _LAUNCH_LINE.match(line)
            if match:
                values[match.group(1)] = match.group(2)

    defines = []
    for item in values["defines"].split():
        if not _DEFINE.match(item):
            raise ValueError(f"launch defines: {item!r} is not NAME=VALUE")
        defines.append(tuple(item.split("=", 1)))
    macros = dict(defines)
    for required in ("GR_WG_SIZE", "GR_RD_TM"):
        if required not in macros or not macros[required].isdigit() or int(macros[required]) <= 0:
            raise ValueError(f"launch defines: {required} must be given as a positive integer")
    options = tuple(values["build_options"].split())
    for option in options:
        if not _BUILD_OPTION.match(option):
            raise ValueError(f"launch build_options: {option!r} is not allowed (only -cl-* options)")
    return LaunchConfig(tuple(defines), int(macros["GR_WG_SIZE"]), int(macros["GR_RD_TM"]), values["grf_256"].strip().lower() in ("1", "true", "yes"), options)


def task_defines(write_gate: bool) -> dict:
    """Macros fixed by the task (the geometry of the op); the launch block adds the tuning macros."""
    return {
        "SUBGROUP_SIZE": SUBGROUP_SIZE,
        "GR_BRANCHES": C,
        "GR_HIDDEN": D,
        "GR_LOWRANK": L,
        "GR_EPS": f"{EPS:.9g}f",
        "GR_PRODUCE_WRITE_GATE": int(write_gate),
    }


def _build_program(ctx: cl.Context, source: str, write_gate: bool, launch: LaunchConfig) -> cl.Program:
    defines = {**task_defines(write_gate), **dict(launch.defines)}
    options = ["-cl-mad-enable"] + [f"-D{k}={v}" for k, v in defines.items()] + list(launch.build_options)
    if launch.grf_256:
        options.append("-cl-intel-256-GRF-per-thread")
    last_error = None
    for std in ("-cl-std=CL3.0", "-cl-std=CL2.0"):  # same selection as the plugin's KernelGenerator
        try:
            return cl.Program(ctx, source).build(options=options + [std])
        except cl.Error as e:
            last_error = e
    raise last_error


# ===================== Weights and test data =====================
class Weights:
    """Model weights (f16) shared by all cases; the scales keep t and the gates in a realistic range."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)

        def make(shape, std, mean=0.0):
            return (mean + rng.standard_normal(shape, dtype=np.float32) * std).astype(np.float16)

        self.norm_weight = make((CD,), 0.1, 1.0)  # used as the gain as is
        self.w_down = make((L, CD), 0.02)
        self.w_up = make((CD, L), 0.1)
        self.w_inject = make((C, CD), 0.02)


_WEIGHTS = None


def weights() -> Weights:
    global _WEIGHTS
    if _WEIGHTS is None:
        _WEIGHTS = Weights()
    return _WEIGHTS


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ro(ctx: cl.Context, arr: np.ndarray) -> cl.Buffer:
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(arr))


class Case:
    """One problem: residual [rows, C*D] f16 plus sentinel-filled outputs."""

    def __init__(self, queue: cl.CommandQueue, rows: int, seed: int = 0, tile_rows: int | None = None):
        self.queue = queue
        self.rows = rows
        self.w = weights()
        rng = np.random.default_rng(seed)
        if tile_rows is None:
            res = rng.standard_normal((rows, CD), dtype=np.float32)
            res *= np.exp(rng.uniform(-1.5, 1.5, size=(rows, 1))).astype(np.float32)  # per-row magnitude spread exercises the RMSNorm
            self.residual = res.astype(np.float16)
        else:
            base = rng.standard_normal((tile_rows, CD), dtype=np.float32).astype(np.float16)
            self.residual = np.tile(base, ((rows + tile_rows - 1) // tile_rows, 1))[:rows]
        ctx = queue.context
        self.b_residual = _ro(ctx, self.residual)
        self.b_norm = _ro(ctx, self.w.norm_weight)
        self.b_down = _ro(ctx, self.w.w_down)
        self.b_up = _ro(ctx, self.w.w_up)
        self.b_inject = _ro(ctx, self.w.w_inject)
        self.fresh_outputs()

    def fresh_outputs(self):
        ctx = self.queue.context
        n = self.rows + PAD_ROWS
        mf = cl.mem_flags
        self.b_mixed = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.full(n * D, SENTINEL, np.float16))
        self.b_gate = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.full(n * C, SENTINEL, np.float16))

    def read_outputs(self):
        """(mixed [rows, D], gate [rows, C]) as f32; checks that nothing was written past the last row."""
        n = self.rows + PAD_ROWS
        mixed = np.empty(n * D, np.float16)
        gate = np.empty(n * C, np.float16)
        cl.enqueue_copy(self.queue, mixed, self.b_mixed)
        cl.enqueue_copy(self.queue, gate, self.b_gate)
        self.queue.finish()
        mixed, gate = mixed.reshape(n, D), gate.reshape(n, C)
        assert (mixed[self.rows :] == SENTINEL).all(), "kernel wrote past the last row of `mixed`"
        assert (gate[self.rows :] == SENTINEL).all(), "kernel wrote past the last row of `write_gate`"
        return mixed[: self.rows].astype(np.float32), gate[: self.rows].astype(np.float32)

    def reference(self):
        """f32 NumPy model of GatedResidual READ with write gate (inputs are the f16 values)."""
        w = self.w
        r = self.residual.astype(np.float32).reshape(self.rows, C, D)
        gain = w.norm_weight.astype(np.float32).reshape(C, D)
        inv = 1.0 / np.sqrt((r * r).mean(axis=-1, keepdims=True) + EPS)
        rbar = r * inv * gain
        flat = rbar.reshape(self.rows, CD)
        x = flat @ w.w_down.astype(np.float32).T / C
        t = x * _sigmoid(x)
        g = _sigmoid(t @ w.w_up.astype(np.float32).T).reshape(self.rows, C, D)
        mixed = (g * rbar).sum(axis=1) / C
        gate = 2.0 * _sigmoid(flat @ w.w_inject.astype(np.float32).T / C)
        return mixed, gate


# ===================== Kernel launcher =====================
class GrOp:
    """Compiles one kernel file for one write-gate setting and launches it as described by its LAUNCH block."""

    def __init__(self, queue: cl.CommandQueue, filename: str, write_gate: bool):
        source = (HERE / filename).read_text()
        self.queue = queue
        self.write_gate = write_gate
        self.launch = parse_launch(source)
        program = _build_program(queue.context, source, write_gate, self.launch)
        try:
            self._kernel = getattr(program, ENTRY_POINT)
        except AttributeError as e:
            raise ValueError(f"entry point {ENTRY_POINT!r} not found in {filename}") from e

    def __call__(self, case: Case):
        launch = self.launch
        lws = (launch.wg_size * SUBGROUP_SIZE, 1, 1)
        gws = (lws[0], max(1, -(-case.rows // launch.tile_rows)), 1)
        args = [case.b_residual, case.b_norm, case.b_down, case.b_up]
        if self.write_gate:
            args.append(case.b_inject)
        args.append(case.b_mixed)
        if self.write_gate:
            args.append(case.b_gate)
        args.append(np.uint32(case.rows))
        return self._kernel(self.queue, gws, lws, *args)


_OPS: dict = {}


def build_op(queue: cl.CommandQueue, filename: str, write_gate: bool) -> GrOp:
    key = (filename, write_gate, queue.int_ptr)
    if key not in _OPS:
        _OPS[key] = GrOp(queue, filename, write_gate)
    return _OPS[key]


# ===================== Fixtures =====================
def _has_gpu() -> bool:
    try:
        return any(p.get_devices(cl.device_type.GPU) for p in cl.get_platforms())
    except cl.Error as e:
        print(f"[build] OpenCL GPU query failed: {type(e).__name__}: {e}")
        return False


def _gpu_queue() -> cl.CommandQueue:
    for platform in cl.get_platforms():
        devs = platform.get_devices(cl.device_type.GPU)
        if devs:
            return cl.CommandQueue(cl.Context([devs[0]]), properties=cl.command_queue_properties.PROFILING_ENABLE)
    raise RuntimeError("No OpenCL GPU device found")


@pytest.fixture(scope="session")
def ocl_queue():
    """Fixture to create an OpenCL command queue for GPU execution."""
    return _gpu_queue()


@pytest.fixture(scope="session")
def kernel_file(use_reference):
    return REFERENCE_FILE if use_reference else KERNEL_FILE


# ===================== Test class =====================
def _tokens_id(rows: int) -> str:
    return f"tokens={rows}"


class TestGatedResidualReadDpas(TestBase):
    """GatedResidual READ / MIX single systolic kernel, f16 weights."""

    def build(self, gpu_arch) -> list[str]:
        """Compiles kernel.cl for both write-gate settings of the tests."""
        if not _has_gpu():
            print("[build] No OpenCL GPU device found: skipping the kernel.cl compile check (it is compiled when the tests run on the GPU machine).")
            return
        queue = _gpu_queue()
        for write_gate in (True, False):
            build_op(queue, KERNEL_FILE, write_gate)

    @pytest.mark.parametrize("rows", CORRECTNESS_ROWS, ids=_tokens_id)
    def test_correctness_wrt_numpy(self, ocl_queue, kernel_file, rows):
        """Merged stream and write gate must match the f32 NumPy model."""
        case = Case(ocl_queue, rows, seed=rows)
        build_op(ocl_queue, kernel_file, True)(case)
        mixed, gate = case.read_outputs()
        exp_mixed, exp_gate = case.reference()
        np.testing.assert_allclose(mixed, exp_mixed, rtol=2e-2, atol=5e-3, err_msg=f"mixed, rows={rows}")
        np.testing.assert_allclose(gate, exp_gate, rtol=1e-2, atol=5e-3, err_msg=f"write_gate, rows={rows}")

    @pytest.mark.parametrize("rows", [33, 1025], ids=_tokens_id)
    def test_correctness_without_write_gate(self, ocl_queue, kernel_file, rows):
        """GR_PRODUCE_WRITE_GATE = 0: the merged stream must match the NumPy model and the gate buffer stays untouched."""
        case = Case(ocl_queue, rows, seed=rows + 1)
        build_op(ocl_queue, kernel_file, False)(case)
        mixed, gate = case.read_outputs()
        exp_mixed, _ = case.reference()
        np.testing.assert_allclose(mixed, exp_mixed, rtol=2e-2, atol=5e-3, err_msg=f"mixed, rows={rows}")
        assert (gate == SENTINEL).all(), "kernel wrote `write_gate` although GR_PRODUCE_WRITE_GATE = 0"

    @pytest.mark.parametrize("rows", [33, 2048], ids=_tokens_id)
    def test_correctness_wrt_reference(self, ocl_queue, kernel_file, use_reference, rows):
        """Optimized kernel output must match reference.cl on the same inputs."""
        if use_reference:
            pytest.skip("Skipping reference-vs-reference comparison.")
        case = Case(ocl_queue, rows, seed=123)
        build_op(ocl_queue, kernel_file, True)(case)
        mixed, gate = case.read_outputs()
        case.fresh_outputs()
        build_op(ocl_queue, REFERENCE_FILE, True)(case)
        ref_mixed, ref_gate = case.read_outputs()
        np.testing.assert_allclose(mixed, ref_mixed, rtol=1e-2, atol=2e-3)
        np.testing.assert_allclose(gate, ref_gate, rtol=1e-2, atol=2e-3)

    @pytest.mark.performance
    @pytest.mark.parametrize("rows", BENCH_ROWS, ids=_tokens_id)
    def test_benchmark(self, ocl_queue, kernel_file, measure_runtime, rows):
        """Time only the kernel dispatch (no host<->device copies)."""
        op = build_op(ocl_queue, kernel_file, True)
        case = Case(ocl_queue, rows, seed=rows, tile_rows=BENCH_TILE_ROWS)
        measure_runtime(op, args=(case,), sync_fn=ocl_queue.finish, auto_replicate_inputs_size=0)


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestGatedResidualReadDpas()
    task.build(TestBase.get_machine_gpu_arch())
