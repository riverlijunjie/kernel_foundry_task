import math
import pytest
import numpy as np
import pyopencl as cl
from pathlib import Path
from kernelfoundry.custom_test import CustomTest  # Tests must derive from CustomTask

# Fixtures provided via conftest.py from kernelfoundry: use_reference, ocl_queue, profile_store, request

# ===================== Workload parameters (must match kernel macros) =====================
# SeedVR2 VAE decoder mid-attention workload
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1088
OUTPUT_FRAMES = 5
TEMPORAL_DOWNSAMPLE = 4
SPATIAL_DOWNSAMPLE = 8

T_LAT = (OUTPUT_FRAMES - 1) // TEMPORAL_DOWNSAMPLE + 1            # 2
H_LAT = OUTPUT_HEIGHT // SPATIAL_DOWNSAMPLE                       # 136
W_LAT = OUTPUT_WIDTH // SPATIAL_DOWNSAMPLE                        # 240
SEQ_LEN = T_LAT * H_LAT * W_LAT                                   # 65280
ATTENTION_WINDOW = H_LAT * W_LAT                                  # 32640
NUM_HEADS = 1
HEAD_DIM = 512                                                    # MAX_HEAD_DIM in kernel

# Launch geometry — must match kernel's REDUCE_LANES=32, KEY_TILE=16
REDUCE_LANES = 32
KEY_TILE = 16
QUERY_TILE = 1

KERNEL_NAME = "vision_attention_tiled_f16"


@pytest.fixture(scope="session")
def ocl_queue():
    """Fixture to create an OpenCL command queue for GPU execution."""
    device = None
    for platform in cl.get_platforms():
        devs = platform.get_devices(cl.device_type.GPU)
        if devs:
            device = devs[0]
            break
    assert device is not None, "No OpenCL GPU device found"
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    return queue


# ===================== Reference / data generation =====================

def generate_test_data():
    """
    Build Q, K, V inputs for the vision attention kernel.

    Mirrors pyopencl_runner.make_inputs: Q=K=0 means softmax weights collapse to a
    uniform 1/attention_window distribution per attention block, so the kernel output
    must equal the per-frame mean of V over the attention window. expected_mean is
    that closed-form reference, shape [t_lat, head_dim], float32.
    """
    elements = SEQ_LEN * NUM_HEADS * HEAD_DIM
    q = np.zeros(elements, dtype=np.float16)
    k = np.zeros(elements, dtype=np.float16)
    v = np.empty(elements, dtype=np.float16)

    dim_term = (np.arange(HEAD_DIM, dtype=np.float32) % 37.0 - 18.0) * 0.0007
    row_term = (np.arange(ATTENTION_WINDOW, dtype=np.float32) % 251.0) * 0.00001
    expected_mean = np.empty((T_LAT, HEAD_DIM), dtype=np.float32)

    for frame in range(T_LAT):
        values = row_term[:, None] + dim_term[None, :] + np.float32(0.03125 * frame)
        values_f16 = values.astype(np.float16)
        start = frame * ATTENTION_WINDOW * HEAD_DIM
        end = start + ATTENTION_WINDOW * HEAD_DIM
        v[start:end] = values_f16.reshape(-1)
        expected_mean[frame, :] = values_f16.astype(np.float32).mean(axis=0)
    return q, k, v, expected_mean


def get_data_on_device(ocl_queue_arg):
    """Upload Q/K/V to GPU and return (kernel args tuple, expected float32 reference)."""
    q, k, v, expected_mean = generate_test_data()
    ctx = ocl_queue_arg.context
    mf = cl.mem_flags

    q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    k_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k)
    v_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v)
    bytes_per_tensor = q.nbytes
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=bytes_per_tensor)

    scale = np.float32(1.0 / math.sqrt(float(HEAD_DIM)))

    args = (
        q_buf, k_buf, v_buf, out_buf,
        np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0),
        np.uint32(SEQ_LEN), np.uint32(NUM_HEADS),
        np.uint32(HEAD_DIM), np.uint32(ATTENTION_WINDOW),
        scale,
    )
    return args, expected_mean


def _check_tolerance(result_f16, expected_f32):
    """
    Match pyopencl_runner.verify acceptance criterion: max absolute error < 0.0025.

    The kernel output represents (per-frame, per-attention-window) softmax-weighted
    averages of V. With Q=K=0, this reduces to V's mean over the attention window —
    constant per frame across the head_dim — so we broadcast expected per frame.
    """
    out2 = result_f16.reshape(T_LAT, ATTENTION_WINDOW, HEAD_DIM).astype(np.float32)
    expected = expected_f32[:, None, :]
    diff = np.abs(out2 - expected)
    max_abs = float(diff.max())
    assert max_abs < 0.0025, (
        f"verification failed: max_abs={max_abs:.6g} >= 0.0025"
    )


# ===================== Kernel initialization =====================

def initialize_attention_kernel(filename: str, queue: cl.CommandQueue):
    """Compile and return a callable that launches vision_attention_tiled_f16."""
    cl_src = Path(__file__).parent / filename
    src_text = cl_src.read_text()
    program = cl.Program(queue.context, src_text).build()
    knl = getattr(program, KERNEL_NAME)

    # Dispatch matches pyopencl_runner.launch:
    #   global: (32, num_heads*16, ceil(seq_len / query_tile))
    #   local:  (32, 16, 1)
    global_size = (
        REDUCE_LANES,
        NUM_HEADS * KEY_TILE,
        (SEQ_LEN + QUERY_TILE - 1) // QUERY_TILE,
    )
    local_size = (REDUCE_LANES, KEY_TILE, 1)

    def run(*args):
        return knl(queue, global_size, local_size, *args)

    return run


@pytest.fixture(scope="session")
def kernel(use_reference, ocl_queue):
    """Returns a callable kernel that runs vision_attention_tiled_f16 on the GPU."""
    filename = "vision_attention_tiled_reference.cl" if use_reference else "vision_attention_tiled.cl"
    return initialize_attention_kernel(filename, ocl_queue)


# ===================== Test class =====================

class TestVisionAttentionOCL(CustomTest):
    """Custom task for SeedVR2 vision attention (vision_attention_tiled_f16) using pyopencl."""

    def build(self, gpu_arch) -> list[str]:
        """Validate compilation of vision_attention_tiled.cl at build time."""
        device = cl.get_platforms()[0].get_devices()[0]
        ctx = cl.Context([device])
        build_queue = cl.CommandQueue(ctx)
        initialize_attention_kernel("vision_attention_tiled.cl", build_queue)

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        """Compare GPU kernel output against the analytical f32 reference (mean of V)."""
        args, expected = get_data_on_device(ocl_queue)

        kernel(*args)

        out_buf = args[3]
        result = np.empty(SEQ_LEN * NUM_HEADS * HEAD_DIM, dtype=np.float16)
        cl.enqueue_copy(ocl_queue, result, out_buf)

        _check_tolerance(result, expected)

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        """Compare GPU kernel output against vision_attention_tiled_reference.cl output."""
        args, _ = get_data_on_device(ocl_queue)

        kernel(*args)

        out_buf = args[3]
        result = np.empty(SEQ_LEN * NUM_HEADS * HEAD_DIM, dtype=np.float16)
        cl.enqueue_copy(ocl_queue, result, out_buf)

        # Run reference OCL kernel into the same output buffer
        reference_fn = initialize_attention_kernel("vision_attention_tiled_reference.cl", ocl_queue)
        reference_fn(*args)
        ref = np.empty(SEQ_LEN * NUM_HEADS * HEAD_DIM, dtype=np.float16)
        cl.enqueue_copy(ocl_queue, ref, out_buf)

        # Same fp16 tolerance band as the analytical reference path.
        gpu = result.astype(np.float32)
        cpu = ref.astype(np.float32)
        max_abs = float(np.max(np.abs(gpu - cpu)))
        assert max_abs < 0.0025, (
            f"reference mismatch: max_abs={max_abs:.6g} >= 0.0025"
        )

    @pytest.mark.performance
    def test_benchmark(self, kernel, ocl_queue, measure_runtime):
        """Benchmark: time the GPU kernel dispatch only — no host<->device copies."""
        args, _ = get_data_on_device(ocl_queue)
        measure_runtime(
            kernel,
            args=args,
            sync_fn=ocl_queue.finish,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestVisionAttentionOCL()
    task.build(CustomTest.get_machine_gpu_arch())
