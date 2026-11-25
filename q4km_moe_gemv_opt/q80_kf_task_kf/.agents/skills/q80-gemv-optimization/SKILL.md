---
name: q80-gemv-optimization
description: Optimize Intel OpenCL GEMV kernels, especially Q8_0 GGUF subgroup kernels with SG-transposed weights. Use when improving memory-bound matrix-vector multiplication performance while preserving the existing quantized weight layout and task tests.
metadata:
  programming-languages: opencl
  tags: gemv q8_0 gguf subgroup intel optimization
---

# Q8_0 GEMV optimization

## Non-negotiable constraints

- Do not create commits, push to a remote repository, or submit a pull request.
- Preserve the existing Q8_0 GGUF weight format and SG-transposed memory layout.
- Preserve the kernel entry-point ABI, compile-time parameters, output semantics, and numerical behavior.
- Modify only the region between `[EVOLVE_START]` and `[EVOLVE_END]` in the evolve kernel.
- Do not optimize only one shape; validate every configured K/N shape.

## Understand the operation first

GEMV computes a vector from a matrix and an input vector, commonly written as `y = A * x + b`. This task uses a quantized Q8_0 representation: each block contains an FP16 scale and signed INT8 values. Before changing the kernel, inspect:

1. The Q8_0 block size and byte layout.
2. The SG-transposed `[pqs_T | pd_T]` layout and how it maps to subgroup lanes.
3. The relationship between `K_SIZE`, `N_SIZE`, and `KSPLIT`.
4. The subgroup-size requirement and the work-group mapping.
5. The host-side compile options and all test shapes.

## Intel GPU optimization guidance

GEMV is generally memory-bound. Prioritize reducing memory traffic and hiding memory latency over adding arithmetic.

- Check the target GPU architecture and actual subgroup capabilities before selecting a work-group shape.
- Use subgroup block reads or vectorized loads when they match the physical layout.
- Keep loads coalesced and avoid redundant reads of scales, weights, or input elements.
- Consider local/shared memory only when reuse offsets its synchronization and occupancy cost.
- Use one subgroup to process a natural block of 16 or 32 elements when that improves access regularity.
- Minimize divergent branches, especially in tail handling and quantized unpacking.
- Unroll short fixed loops carefully to increase instruction-level parallelism without causing register pressure or spills.
- Use FMA where it is supported and numerically safe.
- For INT8/Q8_0 data, keep sign extension, scale conversion, and accumulation efficient.
- Consider double buffering only when profiling shows load latency is limiting execution.
- Avoid changing layouts as a workaround; the layout is part of the task contract.

## Correctness and benchmarking workflow

1. Read the reference kernel, evolve kernel, and `task.py` before editing.
2. Make one focused optimization at a time.
3. Run the task correctness tests for all configured shapes.
4. Run the KernelFoundry `build_and_test` workflow and compare against the reference.
5. Inspect compiler errors, numerical mismatches, runtime, and profiling data.
6. Keep a change only when it remains correct and improves the relevant benchmark, not merely a single noisy sample.
7. Record before/after timings, speedup, and—when bandwidth data is available—the achieved fraction of the memory roofline.

The current task covers K=2048 with N=512, 4096, and 8192. Treat all three as acceptance cases.

## Test expectations

Keep correctness and performance tests separate. Test multiple matrix sizes and relevant data types where the task supports them. Compare the optimized output with the reference output using the task's established tolerance. Do not weaken tolerances or remove shapes to make an optimization pass.

## Reporting

For each accepted optimization, summarize:

- What changed in memory access, subgroup mapping, arithmetic, or synchronization.
- Why the change should help this Intel GPU and layout.
- Correctness results for every shape.
- Reference versus optimized timings and speedup.
- Any tradeoff in registers, occupancy, compilation, or portability.

Keep explanations concise and technical. Use the build/test tools to validate claims rather than relying only on static reasoning.
