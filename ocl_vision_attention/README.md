# Vision Attention Kernel Package

Standalone OpenCL package for optimizing Orbix `kVisionAttentionTiledKernelSource`.

## Workload

This package defaults to the SeedVR2 VAE **decoder** mid-attention workload for:

```powershell
ORBIX_SEEDVR2_ENCODE_SPLIT=4
ORBIX_SEEDVR2_DECODE_SPLIT=1
C:\data\code\Orbix_framework_code\orbix\build\src\app\RelWithDebInfo\orbix-seedvr2.exe --model-dir C:\data\models\Huggingface\SeedVR2-3B\orbix --input-dir C:\data\code\Orbix_framework_code\test_480 --input-frames 5 --output-dir C:\data\code\Orbix_framework_code\test_480_1456_f5 --res 1456 --seed 666
```

For the stated 1920x1088, 5-frame output:

- VAE spatial downsample: 8
- VAE temporal downsample: 4
- latent decode grid: `t_lat=2`, `h_lat=136`, `w_lat=240`
- `seq_len = t_lat * h_lat * w_lat = 65280`
- `attention_window = h_lat * w_lat = 32640`
- `num_heads = 1`
- `head_dim = 512`
- Q/K/V dtype: `f16`
- Q/K/V shape: `[65280, 1, 512]`
- output shape: `[65280, 512]`

The runner prints these values at startup.


## Run Without Build, Using PyOpenCL

Install dependency once:

```powershell
python -m pip install pyopencl
```

Then launch the same kernel and workload directly:

```powershell
cd C:\data\code\Orbix_framework_code\vision_attention_kernel
python pyopencl_runner.py --iters 1 --warmup 1
```

Use another kernel source with the same ABI:

```powershell
python pyopencl_runner.py --kernel kernels\my_optimized.cl --kernel-name vision_attention_tiled_f16
```

Use another kernel source with the same signature:

```powershell
.\build\RelWithDebInfo\vision_attention_runner.exe --kernel kernels\my_optimized.cl --kernel-name vision_attention_tiled_f16
```

Useful options:

```text
--kernel <path>        OpenCL source file
--kernel-name <name>   Kernel symbol
--width <pixels>       Default 1920
--height <pixels>      Default 1088
--frames <count>       Default 5
--query-tile <rows>    Query rows per work-group for global Z. Default 1
--warmup <n>           Default 1
--iters <n>            Default 1
--verify <0|1>         Default 1
--list-devices         Print OpenCL GPU devices
```

## Data Pattern

The host uses zero Q/K and deterministic f16 V. This preserves the real kernel
shape, dtype, NDRange, and loop trip counts while making correctness cheap to
check: attention is uniform within each frame window, so each output row should
match that frame's V mean. The runner verifies the full output tensor by default.

For optimized kernels that keep the same argument ABI but map multiple query
rows per work-group, pass the matching `--query-tile N` so the host uses
`global_z = ceil(seq_len / N)`.
