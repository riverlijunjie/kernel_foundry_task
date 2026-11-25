import argparse
import math
import time
from pathlib import Path

import numpy as np
import pyopencl as cl


def truthy(value: str) -> bool:
    return value in ("1", "true", "TRUE", "on", "ON", "yes", "YES")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch vision_attention_tiled_f16 with PyOpenCL."
    )
    parser.add_argument("--kernel", default="kernels/vision_attention_tiled_original.cl")
    parser.add_argument("--kernel-name", default="vision_attention_tiled_f16")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--query-tile", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--verify", default="1")
    parser.add_argument("--platform", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def list_gpu_devices():
    devices = []
    for platform_index, platform in enumerate(cl.get_platforms()):
        for device_index, device in enumerate(platform.get_devices(device_type=cl.device_type.GPU)):
            devices.append((platform_index, device_index, platform, device))
    if not devices:
        raise RuntimeError("no OpenCL GPU devices found")
    return devices


def select_device(platform_index: int, device_index: int):
    platforms = cl.get_platforms()
    if platform_index < 0 or platform_index >= len(platforms):
        raise RuntimeError(f"invalid platform index {platform_index}")
    gpus = platforms[platform_index].get_devices(device_type=cl.device_type.GPU)
    if not gpus:
        raise RuntimeError(f"platform {platform_index} has no GPU devices")
    if device_index < 0 or device_index >= len(gpus):
        raise RuntimeError(f"invalid device index {device_index} for platform {platform_index}")
    return platforms[platform_index], gpus[device_index]


def make_workload(width: int, height: int, frames: int) -> dict:
    if width <= 0 or height <= 0 or frames <= 0:
        raise RuntimeError("width, height, and frames must be positive")
    if width % 8 != 0 or height % 8 != 0:
        raise RuntimeError("SeedVR2 VAE decode dimensions must be divisible by 8")
    temporal_downsample = 4
    spatial_downsample = 8
    t_lat = (frames - 1) // temporal_downsample + 1
    h_lat = height // spatial_downsample
    w_lat = width // spatial_downsample
    head_dim = 512
    num_heads = 1
    seq_len = t_lat * h_lat * w_lat
    attention_window = h_lat * w_lat
    elements = seq_len * num_heads * head_dim
    bytes_per_tensor = elements * np.dtype(np.float16).itemsize
    return {
        "output_width": width,
        "output_height": height,
        "output_frames": frames,
        "t_lat": t_lat,
        "h_lat": h_lat,
        "w_lat": w_lat,
        "seq_len": seq_len,
        "attention_window": attention_window,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "elements": elements,
        "bytes_per_tensor": bytes_per_tensor,
    }


def print_workload(w: dict) -> None:
    mib = w["bytes_per_tensor"] / (1024.0 * 1024.0)
    print("SeedVR2 VAE decoder mid-attention workload")
    print(f"  target output: {w['output_width']}x{w['output_height']} x {w['output_frames']} frames")
    print(f"  latent decode grid: t_lat={w['t_lat']} h_lat={w['h_lat']} w_lat={w['w_lat']}")
    print("  Q/K/V dtype: f16")
    print(f"  Q/K/V shape: [{w['seq_len']}, {w['num_heads']}, {w['head_dim']}]")
    print(f"  output shape: [{w['seq_len']}, {w['num_heads'] * w['head_dim']}]")
    print(f"  attention_window: {w['attention_window']}")
    print(f"  elements per tensor: {w['elements']}")
    print(f"  bytes per tensor: {w['bytes_per_tensor']} ({mib:.2f} MiB)")
    print(f"  Q+K+V+Output bytes: {4 * w['bytes_per_tensor']} ({4.0 * mib:.2f} MiB)")


def make_inputs(w: dict):
    elements = w["elements"]
    q = np.zeros(elements, dtype=np.float16)
    k = np.zeros(elements, dtype=np.float16)
    v = np.empty(elements, dtype=np.float16)

    dim_term = (np.arange(w["head_dim"], dtype=np.float32) % 37.0 - 18.0) * 0.0007
    row_term = (np.arange(w["attention_window"], dtype=np.float32) % 251.0) * 0.00001
    expected_mean = np.empty((w["t_lat"], w["head_dim"]), dtype=np.float32)

    for frame in range(w["t_lat"]):
        values = row_term[:, None] + dim_term[None, :] + np.float32(0.03125 * frame)
        values_f16 = values.astype(np.float16)
        start = frame * w["attention_window"] * w["head_dim"]
        end = start + w["attention_window"] * w["head_dim"]
        v[start:end] = values_f16.reshape(-1)
        expected_mean[frame, :] = values_f16.astype(np.float32).mean(axis=0)
    return q, k, v, expected_mean


def build_kernel(ctx, kernel_path: Path, kernel_name: str):
    source = kernel_path.read_text(encoding="utf-8")
    program = cl.Program(ctx, source).build()
    return getattr(program, kernel_name)


def launch(queue, kernel, buffers, w: dict, query_tile: int):
    q_buf, k_buf, v_buf, out_buf = buffers
    scale = np.float32(1.0 / math.sqrt(float(w["head_dim"])))
    global_size = (
        np.int64(32),
        np.int64(w["num_heads"] * 16),
        np.int64((w["seq_len"] + query_tile - 1) // query_tile),
    )
    local_size = (32, 16, 1)
    event = kernel(
        queue,
        global_size,
        local_size,
        q_buf,
        k_buf,
        v_buf,
        out_buf,
        np.uint64(0),
        np.uint64(0),
        np.uint64(0),
        np.uint64(0),
        np.uint32(w["seq_len"]),
        np.uint32(w["num_heads"]),
        np.uint32(w["head_dim"]),
        np.uint32(w["attention_window"]),
        scale,
    )
    event.wait()
    return (event.profile.end - event.profile.start) * 1.0e-6


def verify(queue, out_buf, w: dict, expected_mean) -> None:
    out = np.empty(w["elements"], dtype=np.float16)
    cl.enqueue_copy(queue, out, out_buf, is_blocking=True)
    out2 = out.reshape(w["t_lat"], w["attention_window"], w["head_dim"]).astype(np.float32)
    expected = expected_mean[:, None, :]
    diff = np.abs(out2 - expected)
    max_index = int(np.argmax(diff))
    max_abs = float(diff.reshape(-1)[max_index])
    print(f"verify: max_abs={max_abs:.6g} at element {max_index}")
    if max_abs >= 0.0025:
        raise RuntimeError("verification failed: max_abs >= 0.0025")


def main() -> int:
    args = parse_args()
    devices = list_gpu_devices()
    for platform_index, device_index, platform, device in devices:
        print(
            f"[platform={platform_index} device={device_index}] "
            f"{device.name} | vendor={device.vendor} | version={device.version}"
        )
    if args.list_devices:
        return 0

    if args.query_tile <= 0:
        raise RuntimeError("--query-tile must be positive")

    platform, device = select_device(args.platform, args.device)
    print(f"selected: platform={args.platform} {platform.name}, device={args.device} {device.name}")

    w = make_workload(args.width, args.height, args.frames)
    print_workload(w)
    print(f"kernel file: {args.kernel}")
    print(f"kernel name: {args.kernel_name}")
    print(f"query_tile: {args.query_tile}")
    global_size = (32, w["num_heads"] * 16, (w["seq_len"] + args.query_tile - 1) // args.query_tile)
    print(f"global: {global_size}")
    print("local:  (32, 16, 1)")

    t0 = time.perf_counter()
    q, k, v, expected_mean = make_inputs(w)
    print(f"host input generation: {(time.perf_counter() - t0) * 1000.0:.3f} ms")

    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    mf = cl.mem_flags
    q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    k_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k)
    v_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, w["bytes_per_tensor"])
    kernel = build_kernel(ctx, Path(args.kernel), args.kernel_name)

    print(f"warmup launches: {args.warmup}")
    for i in range(args.warmup):
        ms = launch(queue, kernel, (q_buf, k_buf, v_buf, out_buf), w, args.query_tile)
        print(f"  warmup[{i}] {ms:.3f} ms")

    print(f"timed launches: {args.iters}")
    times = []
    for i in range(args.iters):
        ms = launch(queue, kernel, (q_buf, k_buf, v_buf, out_buf), w, args.query_tile)
        times.append(ms)
        print(f"  iter[{i}] {ms:.3f} ms")
    if times:
        print(f"kernel_time_ms avg={sum(times) / len(times):.3f} best={min(times):.3f}")

    if truthy(args.verify):
        verify(queue, out_buf, w, expected_mean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
