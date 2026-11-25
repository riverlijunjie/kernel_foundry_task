---
name: cm-gemm
description: Guidance for implementing, benchmarking, and optimizing Intel C-for-Metal (CM) GEMM kernels on Xe2/Xe3 GPUs (BMG, PTL). Use when writing or improving a CM matrix-matrix multiplication kernel, choosing tiling/SLM staging, tuning register/occupancy trade-offs, or diagnosing why a CM GEMM kernel is memory- or compute-bound.
metadata:
  programming-languages: cm CM
  tags: gemm cm optimization xe2 xe3 bmg slm dpas
---

# CM GEMM Kernel Optimization

Use this skill when writing or improving a CM (C for Metal) kernel that computes
`C = A x B` (optionally `+ D`), tuning its work-group/SLM dataflow, or diagnosing
whether it is memory- or compute-bound. This is a knowledge and methodology
guide, not a repository map: locate the target `.cm` file, host harness, and
reference implementation in the active workspace before starting.

## 0. Check the obvious thing first

A correct-but-naive CM GEMM often dispatches **one thread per output row with a
work-group size of 1** (`GWS=(M,)`, `LWS=(1,)`). That means every thread
independently re-reads the *entire* `B` matrix from global memory with zero
reuse across threads, and each thread accumulates in a scalar/serial loop with
no vectorization. Before reaching for advanced tricks, check whether this is
the starting point — if so, the first and largest wins are almost always, in
order:

1. **Work-group tiling**: give the work-group a shared block of `B` (and/or
   `A`) staged through SLM so multiple threads reuse the same load.
2. **Vectorized loads/stores**: replace scalar element-at-a-time loads with
   wider CM vector loads (`vector<float, N>`, `cm_ptr_load<T,N>`) and unroll the
   inner `k` loop.
3. **Register blocking**: let each thread compute more than one output
   element (a small `row x col` tile) so one loaded operand is reused across
   several FMAs instead of one.

Only after these structural changes are in place does it make sense to tune
finer knobs (prefetch distance, SLM double buffering, `dpas`/XMX usage for
lower-precision operands, etc.).

## 1. Fast bottleneck lookup

| Symptom | Likely cause | First experiment |
|---|---|---|
| Removing the inner-loop load of `B` (replace with a constant) makes it much faster | `B` load latency/traffic dominates | add SLM staging or vectorize the load |
| Removing the `A` load makes it much faster | `A` load latency/traffic dominates | check `A` reuse across the work-group, layout/stride |
| Neither ablation changes much | compute/accumulate path or dispatch overhead dominates | check vectorization, unrolling, and `LWS`/occupancy |
| Performance collapses after enlarging a tile | register spill or occupancy loss | check compiler spill diagnostics, reduce tile or GRF ask |
| Correct at small `M,N,K` but wrong at larger/odd sizes | tail handling, non-divisible tile | test non-multiple-of-tile shapes explicitly |
| Sequential before/after runs disagree wildly | GPU clock/power state, not the code change | measure interleaved, same-process, min-of-N |

Optimization order: **measure -> classify the bottleneck (compute vs memory vs
dispatch) -> add work-group/SLM reuse -> vectorize -> tune registers/occupancy
-> consider `dpas`/XMX if operand precision allows**. Do not jump straight to
`dpas` shape or GRF size before the measurements show that they are the
limiting resource.

## 2. Non-negotiable workflow

1. Read the target `.cm` kernel, its host harness (how it's JIT-compiled/dispatched, e.g. via `clBuildProgram(-cmc)`), and the reference implementation before editing.
2. Establish a correctness baseline (compare against a NumPy/host reference) before changing anything for performance.
3. Build/compare kernel variants from the same source in one process when possible (e.g. via `-D` macros), not separate one-off runs.
4. Benchmark candidates interleaved (round-robin), with the same warmup policy; report mean/median/min separately, and use min-of-N for kernel-level A/B comparisons.
5. Inspect compiler output for `Spill memory used = ...`. Treat a spill as a rejection unless a controlled measurement proves it's still a net win.
6. Record shape (M,K,N), tile/macro configuration, register count, SLM usage, GPU clock, latency, achieved TFLOPS/bandwidth, and correctness status for every candidate.
7. Re-test every adopted change across the representative shapes the task actually benchmarks — do not generalize a win measured on one shape.
8. Update kernel header comments (and any repo memory) with measured wins and rejected ideas so future iterations don't repeat dead ends.

## 3. Hardware facts (Intel Xe2/Xe3 GPUs)

### BMG / Arc B580 / Xe2 (the usual `gpu_arch: bmg` target)

- 160 XVE (vector engines) at ~2.9 GHz nominal
- 8 threads per XVE
- 128 KB SLM per Xe-core
- ~456 GB/s GDDR6 memory bandwidth
- ~18 MB L2
- nominal fp16 peak ~116 TFLOPS (dpas), much lower for scalar fp32
- **Hard thermal/power throttling under sustained runs** (see §4.2) — do not
  trust a single before/after process run.

### PTL / Arc B390 / Xe3 iGPU

- 96 XVE at ~2.4 GHz, 10 threads per XVE
- 128 KB SLM per Xe-core
- shared LPDDR bandwidth ~136 GB/s (much lower than BMG's discrete memory)
- nominal fp16 peak ~59 TFLOPS
- more register-constrained per thread than BMG; tiles that fit on BMG can spill on PTL

Never assume the same tile/blocking configuration is optimal on both
architectures — always re-tune (or at minimum re-measure) per `gpu_arch`.

## 4. Roofline and measurement discipline

### 4.1 Roofline

```
t_compute = FLOPs / peak_compute
t_memory  = bytes_moved / peak_bandwidth
t_roofline = max(t_compute, t_memory)
```
For `C[M,N] = A[M,K] x B[K,N]`: `FLOPs = 2 * M * N * K`. State clearly whether
a reported "% of roofline" uses nominal peak vs. a measured dpas-only ceiling,
and whether it's based on mean, median, or min latency.

### 4.2 BMG clock drift — do not trust sequential before/after runs

BMG throttles hard under sustained load; the same binary/shape measured
repeatedly in one run can show e.g. 1.7 ms -> 3.2 ms -> 4.7 ms -> 5.2 ms purely
from clock ramping down (2900 MHz -> ~1150 MHz). Consequences:

- Two separate process launches (baseline, then optimized) are **not** valid
  A/B evidence by themselves — a "win" can just be a lucky clock state.
  Interleave variants in the same process/session when comparing.
- If reporting a roofline percentage, also report the GPU clock observed
  during the timed window, or normalize explicitly.

### 4.3 PTL sustained-power effects

Mean latency can grow with iteration count while min latency stays close to
stable; don't interpret a change in mean latency without also checking min.

## 5. CM syntax/API pitfalls

- CM has no OpenCL sub-group lanes: one CM thread computes a full vector's
  worth of output, not one lane of a subgroup. Don't port an OpenCL subgroup
  dimension literally.
- Use `uint`, not `size_t`, for pointer/index arithmetic.
- `cm_ptr_load<T,N>`/`cm_ptr_store<T,N>` need 4- or 8-byte element types; for
  `half`/`ushort`/`char` load as `uint32` and reinterpret with
  `.format<half>()` / `.format<ushort>()` / `.format<char>()`.
- Callee output parameters need `vector_ref<T,N>`/`matrix_ref<T,R,C>`, not
  plain C++ references to CM vectors.
- A single CM vector object is limited to well under 8 KB; larger tiles need
  arrays of smaller vectors (which can themselves spill — check register
  usage after doing this).
- After any SLM write, pair `cm_slm_fence(CM_LOCAL_BARRIER)` with
  `cm_barrier()` before the corresponding SLM read. A bare `cm_barrier()`
  without the fence has produced real, nondeterministic reduction races.
- `cm_ptr_store`/`cm_ptr_load` have a maximum element count (commonly 64) and
  must be a legal vector size; a "merged" wide store spanning several tiles
  can silently stop compiling once the tile count grows. Guard with a
  compile-time size check and fall back to per-tile stores above the limit.
- `cm_dpas<...>` (XMX) requires low-precision operands (fp16/bf16/int8, not
  plain fp32); only reach for it if the task's numerics tolerate that
  precision, and confirm `CM_HAS_DPASW`/architecture support before assuming
  a dpas variant is available on the target part.

## 6. Register/occupancy and dataflow principles

- Reuse via **blocking**: whichever operand (`A` row(s) or `B` column(s)) is
  held in registers and reused across multiple FMAs/dpas issues gets "free"
  reuse; whichever operand is re-loaded from SLM/global memory for every
  reuse gets "paid for" reuse. When choosing between two blocking factors
  that reduce the same operand's traffic by the same amount, prefer the one
  that keeps values in registers over the one that relies on multiple threads
  hitting the same address in cache.
- **Order matters for latency hiding, not just register count**: issuing all
  of one operand's loads back-to-back before the first compute instruction
  (instead of interleaving one load per compute step) can let their latency
  overlap with compute even though it costs more live registers. If enabling
  this only wins on the shapes where that operand's latency was exposed,
  that's confirmation, not a red flag — but re-measure after any staging
  change, since it can change which operand is now the bottleneck.
- **SLM ring sizing**: a producer/consumer SLM ring needs at least
  `depth + 1` slots for a full-barrier protocol and `depth + 2` for a split
  barrier; getting this wrong is a silent correctness bug, not just a
  performance one.
- Do not increase GRF allocation (e.g. requesting a bigger register file)
  just to fit a larger tile — it typically halves resident threads per XVE,
  which can lose more from reduced occupancy/latency-hiding than it gains
  from the larger tile. Measure before adopting.
- A spill (`Spill memory used = ...` in the compiler log) is a strong signal
  to shrink the tile/blocking factor rather than push through it, unless a
  controlled measurement proves the spilling variant still wins.

## 7. Bottleneck diagnosis checklist

When a shape is slower than expected:

1. Confirm the shape tuple order (`M, K, N`, not accidentally swapped).
2. Check divisibility: does `M`/`N`/`K` divide evenly into the chosen tile/work-group size, or does the kernel need tail handling?
3. Print the actual dispatch (`GWS`, `LWS`) and tile macros used; verify they're internally consistent.
4. Build "ablation" variants that skip loading `A` (use a constant), skip loading `B` (use a constant), or skip the inner accumulate, to see which one the runtime is most sensitive to.
5. Check for compiler spill warnings.
6. Check GPU clock/power state (see §4.2/4.3) before concluding a config change helped or hurt.
7. Only then consider a new tiling/dataflow change.

## 8. Correctness requirements

- Validate against a reference (NumPy or the task's reference kernel) for every candidate, not just the final one.
- Test tail/non-tile-multiple shapes explicitly, not only the "nice" benchmark shapes.
- After any SLM protocol change, run the same case many times — a nondeterministic SLM race often passes once and fails intermittently.
- Require correctness to pass before trusting any performance number for that variant.

## 9. Completion standard

A CM GEMM optimization pass is complete only when:

- the kernel is synchronized with its host dispatch (tile macros match `GWS`/`LWS` computation)
- correctness passes on every benchmarked shape, including tails
- no unintended compiler spill exists (or the spill was measured and still wins)
- performance was measured interleaved against the previous/baseline version, not via separate sequential runs
- the roofline bound, peak assumption, and observed GPU clock are stated explicitly
- rejected ideas and their measured impact are noted (in kernel comments or task notes) so they aren't re-tried blindly next iteration

Keep explanations conversational when summarizing findings. For complex tiling/dataflow trade-offs, use a concrete numeric example (e.g. bytes moved per FMA) rather than only qualitative language.
