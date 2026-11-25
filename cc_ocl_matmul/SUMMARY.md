# cc_ocl_matmul — FP16 GEMM optimization on Intel Arc B580

Optimized [matmul_opt.cl](matmul_opt.cl) for `C[M,N] = A[M,K] · B[K,N]` (fp16 in/out, f32 accumulator) at the fixed shape **M=2048, K=2560, N=2048**, starting from a one-WI-per-output naive scalar kernel.

The optimization was done from scratch using only this directory: a [bench.py](bench.py) harness was written first, then `matmul_opt.cl` was rewritten in successive trials guided by measured time-on-device.

## Results

Hardware: Intel Arc B580 (Battlemage, Xe2-LPG; 20 Xe-cores, 116 TFLOPS f16 XMX peak). Roofline at this shape:

```
2 * M * K * N  =  2 * 2048 * 2560 * 2048  =  21.47 GFLOPs
ideal time     =  21.47e9 / 116e12        =  0.185 ms     (100 % roofline)
70 % target    =  0.185 / 0.70            =  0.264 ms
```

| | Runtime (ms, min) | Speedup vs naive | XMX util |
|---|---:|---:|---:|
| Initial naive scalar (one WI per `C[i,j]`)            | 33.84 | 1.0×  | 0.5 % |
| Final v7 (DPAS + 2D block IO + SWIZZLE=4)             | **0.249** | **136×** | **74.5 %** |

Min over 100 timed launches after 30 warmups ([bench.py](bench.py) `python bench.py matmul_opt.cl 30 100`). Correctness verified against numpy fp32 reference (`max_abs = 0.089, max_rel = 0.027`, within `rtol=2e-2 / atol=2e-2`).

The 70 % roofline target is met with ~4 percentage points of margin.

## Trial log

Each trial mutated one knob from the previous best and was benched with the same harness. Numbers shown are the min of a 5-warmup / 20-trial run unless noted.

| # | Change                                   | Per-SG tile | WG (SGs) | WG-tile  | min ms | util  | decision |
|--:|:-----------------------------------------|:-----------:|:--------:|:--------:|-------:|:-----:|:---------|
| 0 | naive scalar baseline                     | —           | —        | —        | 33.84  | 0.5 % | baseline |
| 1 | DPAS + 2D block reads + VNNI B + 32×32 SG | 32×32       | 16 (2×8) | 64×256   | 0.321  | 57.6 % | new best |
| 2 | WG_N 8 → 16 (wider B-panel reuse)         | 32×32       | 32 (2×16)| 64×512   | 0.299  | 61.9 % | new best |
| 3 | WG_M=4 WG_N=8 (square-ish 32 SGs)         | 32×32       | 32 (4×8) | 128×256  | 0.294  | 62.9 % | new best |
| 4 | WG_M=8 WG_N=4 (M-tall 32 SGs)             | 32×32       | 32 (8×4) | 256×128  | 0.327  | 56.6 % | reject  |
| 5 | WG_M=4 WG_N=16 (1024 WIs)                 | 32×32       | 64       | 128×512  | 0.314  | 59.0 % | reject  |
| 6 | Per-SG tile 32×64 (16 float8 accumulators)| 32×64       | 16 (2×8) | 64×512   | 1.840  | 10.1 % | reject (GRF spill) |
| 7 | + SWIZZLE=2 M-major block dispatch         | 32×32       | 32 (4×8) | 128×256  | 0.257  | 72.1 % | new best |
| 7'| SWIZZLE 2 → 4                              | 32×32       | 32 (4×8) | 128×256  | 0.256  | 72.3 % | new best |
| 7"| SWIZZLE 4 → 8                              | 32×32       | 32 (4×8) | 128×256  | 0.306  | 60.4 % | reject  |
| † | + cooperative B 2D-block-prefetch (3 variants tried) | — | — | — | 0.49–0.59 | 31–38 % | reject  |
| † | K-step 32 → 64 (deeper unroll)            | 32×32       | 32 (4×8) | 128×256  | 0.643  | 28.8 % | reject (GRF pressure) |

Final: trial 7' → after stretching warmup to 30/100, sustained min **0.249 ms / 74.5 %**.

## What worked, and why

### 1. Replace the naive loop with DPAS + 2D block reads (33.8 ms → 0.32 ms, ~105×)

The naive kernel does one fp16 multiply-add per work-item per K-step from scattered global loads — every load is a uncoalesced 2-byte fetch and every MAC is a scalar fp16 op going through ALU lanes, never the systolic array. The fix is the standard B580 GEMM idiom:

- **DPAS systolic** — `intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)` does an 8×16 outer product per call (8 M-rows × 16 N-cols × 16 K-elements in one instruction), which is the only path that hits the 116 TFLOPS f16 peak. The accumulator stays in f32 the whole K-loop and is downcast once at store time.
- **2D block load on A** — `intel_sub_group_2d_block_read_16b_32r16x2c` pulls a 32-row × 32-col fp16 strip in one transaction, with the sub-group cooperatively splitting the 16-byte-wide row across its 16 lanes.
- **Transform-load on B** — `intel_sub_group_2d_block_read_transform_16b_16r16x2c` returns B already crosspacked (VNNI: two consecutive K-elements packed into one 32-bit lane), which is exactly the layout DPAS expects. No software repack is needed.
- **2D block store on C** — `intel_sub_group_2d_block_write_16b_8r16x1c` writes each 8×16 fp16 acc tile in one transaction.

Per SG: 8 `float8` accumulators (4 M-rows × 2 N-cols of 8×16 DPAS blocks) → a 32 (M) × 32 (N) C-tile. K is consumed in steps of 32 (two `k=16` DPAS rounds per step).

### 2. WG shape: 4 × 8 sub-groups, WG-tile 128 × 256 (0.32 → 0.29 ms)

Holding 32 SGs/WG (the sweet spot — tried 16, 32, 64; 16 leaves XMX issue slots empty, 64 hits per-SG GRF pressure), the only knob is M-vs-N split. Tried 2×16 (64×512, very wide N), 4×8 (128×256, square-ish), 8×4 (256×128, very tall M).

The 4×8 / 128×256 variant won by a small margin, and matters more in combination with swizzle (next step) — both M and N panels of the WG-tile fit comfortably in L2, and the dispatch grid (16, 8) gives enough parallelism for swizzle to actually find spatial locality.

### 3. SWIZZLE=4 M-major block walk (0.29 → 0.25 ms — the big win)

The default OpenCL dispatch order walks the (`grid_m`=16, `grid_n`=8) grid linearly along `gid_m` first. With WG-tile = 128×256 and 32 active WGs per "wave", consecutive WGs in flight share neither A (different `gid_m` → different M-rows) nor B (each WG only spans one N-strip for a tiny window). The B panel — 256 cols × 2560 K = 1.25 MB per WG — has to be re-streamed from L3 for every WG.

The swizzle re-orders the walk into M-major blocks of `SWIZZLE` consecutive `gid_m`'s sharing the same `gid_n`:

```
linear   = gid_n * grid_m + gid_m
block_id = linear / (SWIZZLE * grid_n)
block_off = linear % (SWIZZLE * grid_n)
gm = block_id * SWIZZLE + (block_off % SWIZZLE)
gn = block_off / SWIZZLE
```

Now within each 4-WG block the same `gn` (B-panel) stays resident: the 1.25 MB B-panel is fetched once and reused four times. With 20 Xe-cores executing ~32 WGs concurrently, this lets one B-panel feed eight of them rather than one, and the L2 hit rate on B jumps. Tried SWIZZLE = 2, 4, 8: 4 was best (8 starts to break A locality across the M-block; 2 leaves reuse on the table).

### 4. K-loop body kept tight (k=32 step, two k=16 DPAS rounds)

Tried unrolling K to 64 (cache two 32-cycle A-strips and four B-tiles per loop body) — the GRF footprint blew up and the kernel dropped to 0.64 ms. Tried widening the per-SG tile to 32×64 (16 `float8` accumulators) — same problem, 1.84 ms. The current footprint is right at the GRF/EU sweet spot: 8 `float8` accumulators (32 GRFs of acc) + one A strip (4 GRFs) + one pair of B tiles (4 GRFs) = ~40 GRFs — fits 8 threads/EU comfortably with no spill.

### 5. Things tried that did *not* help (and why they're not in the final kernel)

- **2D block prefetch on B** (multiple variants: warmup-only, all-SGs steady-state, sg_row-cooperative, K-distance 2 / 3). Every variant *regressed* the kernel by 1.5–2×. The driver's L2 streaming on the read-side already saturates the load slots; explicit prefetch instructions appear to compete for the same LSC issue ports rather than overlap with them. The 2D-block-prefetch builtin compiles fine on this driver — it just isn't a win when reads alone already keep the bandwidth pipe full and L2 reuse is good (which the swizzle delivers).
- **Wider per-SG tile (32×64)** — register pressure blow-up; spill cost outweighs the lower memory traffic.
- **K-step = 64** — same.
- **WG shape 8×4 / 4×16** — either M-tall (small grid_n hurts swizzle reuse) or too many SGs/WG (per-SG GRF count drops, more cross-SG scheduling cost).

## Final kernel architecture

```
WG (one work-group)  ─────────── 128 (M) × 256 (N) C-tile
├── 32 sub-groups arranged 4 × 8
│   each sub-group → 32 (M) × 32 (N) C-tile (8 float8 accs)
│
└── K-loop, step 32:
    ├── A : 1× intel_sub_group_2d_block_read_16b_32r16x2c       (32×32 half)
    ├── B : 2× intel_sub_group_2d_block_read_transform_16b_16r16x2c
    │       (VNNI-packed; covers k=16 each)
    └── 2× DPAS rounds × 8 mads = 16 calls of intel_sub_group_f16_f16_matrix_mad_k16

WG dispatch swizzle: M-major blocks of 4, then walk N
```

GRF estimate per SG: 8 × float8 accumulators (32 GRFs) + 64 ushort A-strip (4 GRFs) + 32 uint B-data (4 GRFs) ≈ 40 GRFs, well under 128.

## Files in this directory

- [matmul_opt.cl](matmul_opt.cl) — final v7 kernel
- [bench.py](bench.py) — standalone OpenCL bench harness; verifies vs numpy fp32 and reports min/avg ms + roofline util. Usage: `python bench.py matmul_opt.cl [warmup=5] [trials=20]`
- `SUMMARY.md` — this file

## Reproducing

```bash
source /home/openvino-ci-74/miniforge3/etc/profile.d/conda.sh
conda activate kernel_intel
cd /mnt/river/kernel_foundry/workspace/cc_ocl_matmul
python bench.py matmul_opt.cl 30 100
```

Expected output (Intel Arc B580):

```
GWS=(8192, 8)  LWS=(512, 1)
correctness: OK  max_abs=0.0887  max_rel=0.0274
min=0.249 ms  avg=0.318 ms  util=74.5%  (86.4 TFLOPS)
```
