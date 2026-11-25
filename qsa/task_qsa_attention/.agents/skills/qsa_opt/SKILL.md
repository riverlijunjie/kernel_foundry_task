---
name: qsa-q3-attention-optimization
description: 'Optimize, extend, or debug the QSA (Query Sparse Attention) Q3 stage on Intel GPU (CM/Xe2 dpas) — the causal sparse-attention kernel that reads real K/V for the top-k selected blocks. Use when: tuning qsa_sparse_attention_gqa_dpas.cm (or its scalar/union/split siblings); reasoning about causal masking of selected KV blocks; deciding whether a proposed dpas/prefetch/tiling/worker-split change to this kernel is worth trying; benchmarking Q3 on Xe2 GPUs. Fully self-contained — no other files need to be read to apply this skill.'
---

# QSA Q3 (Sparse Attention) Optimization

## 1. What Q3 does and why causal masking needs no explicit mask matrix

Q3 is the final stage of QSA: given a query token and a list of selected compressed-KV
block IDs (produced by an earlier scoring/top-k stage, out of scope here), it runs real
causal attention against the **real** K/V cache (not the block summaries used for
selection) restricted to those blocks, plus a small causal "tail".

QSA never builds an explicit `-inf` mask matrix for Q3. Causality is enforced
**structurally** — non-causal KV is simply never selected or loaded — via one invariant:

```
n_complete(abs_pos) = (abs_pos + 1) / R      // R = compression ratio
```

A compressed block is "complete" (i.e. eligible for selection at all) only once **every**
one of its `R` raw tokens is causally visible, i.e. `(block_id+1)*R <= abs_pos+1`. The
selection stage upstream is responsible for only ever choosing block IDs `< n_complete`,
so by the time Q3 receives `selected_blocks`/`selected_counts`, every selected block is
already guaranteed fully causal — **no per-element masking is needed inside Q3's main
loop over selected blocks.**

The one thing Q3 itself must still handle explicitly is the **causal tail**: the current,
not-yet-compressed block, covering raw positions `[tail_begin, abs_pos]` where
`tail_begin = (abs_pos+1)/R*R`. This range is never part of `selected_blocks` (it hasn't
been compressed into a summary yet) but is always causally visible, so Q3 must always
attend to it unconditionally, with an *exact* row count
(`tail_rows = abs_pos + 1 - tail_begin`) — never reading past `abs_pos`.

**Off-by-one gotcha**: `n_complete = (abs_pos+1)/R`, NOT `abs_pos/R`. Using the latter
silently drops or leaks exactly one token at every block boundary. This is the single
most common correctness bug when modifying Q3's block-loop bounds or tail computation.

`QSA_IS_CAUSAL` must be true for every dpas Q3 variant (`static_assert(QSA_IS_CAUSAL, ...)`
in the kernel; non-causal QSA is unsupported end to end). Do not add a non-causal code
path without re-deriving the tail/selection contract above — it was designed causal-only.

## 2. Current architecture: `qsa_sparse_attention_gqa_dpas.cm` (prefill, "Route B")

One work-group = **one query token × one KV head**; it shares K/V across all
`HEADS = HQ/HKV` (e.g. 12) query heads that GQA-replicate onto that KV head. No
cross-token union of selected blocks is formed (unlike the older union-based sibling
kernel, see §5.1).

- **DPAS shapes**: QK is `M8 KV-rows / N16 heads (HEADS padded up to 16) / K16 dims`;
  PV is `M8 heads / N16 dims / K16 KV`.
- **Worker split**: `GQA_W=4` workers each own a 64-element slice of `Dh`/`Dv` (head
  dimension), so a 256-wide head fits without spilling one thread's accumulator. Only the
  16x16 float score tile (`St`) needs to cross workers, via a double-buffered SLM
  reduction after each QK step (so at most one barrier of lag between workers).
- **Step granularity**: 16 KV rows are consumed per loop iteration
  (`GQA_BLOCKS = 16/R` compressed blocks per step). A `contiguous` fast path detects when
  a step's 16 rows are one physically contiguous, page-local range and issues one wide 2D
  block load instead of per-sub-block gathers.
- **Loop shape**: `steps = ceil(selected_count / GQA_BLOCKS)` selected-block steps,
  followed by exactly one causal-tail step (see §1) when `tail_rows != 0`. Every worker in
  the WG executes the identical number of steps (uniform control flow, no divergence).
- **Two independent, currently opt-in (default-off in the JIT) toggles**:
  - `QSA_GQA_HEAD_MAJOR`: swaps which NDRange axis (token vs. KV-head) is the
    fast-varying one, so adjacent work-groups are more likely to share a KV head. Pure
    scheduling change — same addresses, same arithmetic order, same results.
  - `QSA_GQA_VECTOR_META`: batches the per-step selected-block-ID and physical-page-ID
    reads into masked vector gathers (one wide load per step) instead of one scalar load
    per slot.
- **Capability constraints** (must hold for this kernel to apply; otherwise fall back to
  the union-based dpas kernel, then the scalar kernel): GPU arch >= Xe2 (needed for
  512-bit GRF / `QSA_REG_N==16` dpas); `HQ % HKV == 0` and `1 <= n_rep <= 16` (GQA group
  must fit the N16 tile); `Dh, Dv` both multiples of `GQA_W*32 = 128` and `<= 256`
  (worker-slice alignment + verified accumulator range); compression ratio in
  `{2,4,8,16}`; page size divisible by the compression ratio; causal only.

## 3. Validated optimizations — safe to enable or build on

- **`QSA_GQA_HEAD_MAJOR=1`**: measured ~1.35x/1.23x on 64K/128K-token long-context
  prefill (pure scheduling change, zero risk to correctness). Should be the default
  whenever this kernel is used for long-context prefill.
- **`QSA_GQA_VECTOR_META=1`**: small gain alone, multiplies with head-major when combined
  (masked vector metadata gather removes per-slot scalar load latency chains). Also
  requires zero algorithm change.
- **Head-dimension worker split (`GQA_W`)**: the only reason `Dh/Dv=256` works at all
  without GRF spill — a single thread cannot hold a 256-wide accumulator plus K/V tiles.
  Do not remove or collapse this even if it looks like unnecessary complexity for smaller
  head sizes; it's what makes the kernel scale to the largest supported head size.
- **32-half (64-byte) row cap on wide 2D block loads**: this is a hard hardware limit of
  the LSC 2D block-load message, not a tunable — loading wider rows compiles and runs but
  silently returns corrupted data. Always chunk K/V column loads at 32 half-elements.
- **(head, token) packing instead of pure-token packing on the DPAS N axis** (used by the
  older union-based sibling kernel `qsa_sparse_attention_dpas.cm`, relevant if you ever
  revisit multi-token tiling): filling the systolic N axis with independent query tokens
  forces the kernel to compute the *union* of each token's independent top-k selection,
  which can amplify useful work by up to ~15x at realistic selection ratios. Filling N
  with (head, token) pairs so that all heads of one token share its single selection
  avoids this; it's the reason Route B (per-token, all-GQA-heads) was adopted as the
  default over per-4-token tiling for this configuration.

## 4. Rejected optimizations — do not re-attempt without new hardware evidence

All of these were implemented, benchmarked with rigorous A/B (alternating baseline vs.
variant in the same process/device state, >=10 runs, min taken), and reverted:

- **Prefetching the next step's K/V with `cm_prefetch`**: **2.3-2.4x regression**. This
  kernel is bound by LSC *message-issue rate*, not by memory latency — prefetch adds more
  messages competing for the same issue slots instead of hiding latency. This also
  regressed on the older union-based sibling kernel, so treat it as a property of this
  whole kernel family, not one specific variant.
- **Demand-load cached/uncached/streaming memory hints**: no stable improvement measured.
- **`GQA_W=8`** (tried at two different GRF-file-size configurations): slower than
  `GQA_W=4` — doubling the worker count doubles the SLM cross-worker exchange and barrier
  synchronization cost, which outweighs the benefit of finer-grained per-worker loads.
- **`step=32`** (32 KV rows per loop iteration instead of 16, meant to amortize the
  softmax-update and barrier cost over more rows): roughly flat at `W4`, worse at `W8` —
  the extra live registers/SLM pressure cancel out the amortization gain.
- **Micro-optimizations that "reduce scalar work" without reducing memory traffic**
  (caching k-way merge state in a live vector register instead of re-reading it, hoisting
  page-table address computation out of head-dimension loops, branch-free zero-padding of
  empty tile slots): all measured as *exactly zero* time difference and, together, added
  GRF spill. **Lesson**: this kernel family is data-movement/message-bound, not
  ALU/branch-bound — cutting scalar instruction count does nothing on its own and can
  actively regress a previously-fitting kernel into spill.

## 5. Open directions with quantified rationale (not yet implemented)

### 5.1 Systolic (DPAS N-axis) utilization gap
`HEADS` (the GQA group size, e.g. 12) is padded to the DPAS `N16` tile, so only 75% of
the systolic array's N axis carries useful work at a typical GQA ratio. The alternative
("Route A": pack 4 query tokens x their own head groups into one tile, sharing one KV
read across them, as the older union-based kernel does) trades that padding waste against
having to compute the *union* of each token's independent selected-block set. Which route
is actually cheaper is governed by the cross-query selected-block overlap ratio `U/K`
(union size over per-query selection size): Route B (this kernel) wins when `U/K > 4/3`;
below that, Route A's shared KV read can win despite recomputing a union. **There is
currently no automatic routing between the two kernels based on measured or estimated
overlap** — a host-side heuristic (e.g. picking Route A for short/dense prefill where
selections are near-identical across adjacent tokens, Route B for long, sparse
selections) is an unimplemented, quantifiable opportunity.

### 5.2 Whole-loop dense-contiguous fast path
The existing `contiguous` check only coalesces *one* 16-row step into a wide load. When a
token's entire selection is the dense prefix `0..n_complete` (i.e. every complete block
was selected, which happens whenever the causal history is shorter than the top-k
budget), the *entire* KV range for that token is trivially contiguous across all steps,
not just within one. A dedicated fast path — detected on the host side from
`selected_count == n_complete` — could skip the per-step `selected_blocks`/page-ID gather
entirely and issue a small number of much wider 2D block loads spanning multiple steps at
once. This is a concrete extension of the "reduce redundant work, don't just widen a
single load" principle that has not yet been implemented.

### 5.3 Long-context KV locality decay
With a fixed top-k budget and growing causal history, selected blocks become
increasingly scattered across physical pages as context length grows, so the `contiguous`
fast path (§5.2 notwithstanding) fires less often, and per-step metadata/page-address
computation becomes a proportionally larger share of the work. This is a documented, but
not yet counter-instrumented, open problem: before attempting a tiered/hierarchical KV
selection scheme to address it, first capture hardware cache/TLB/message-stall/occupancy
counters to confirm where cycles are actually going — do not assume a specific speedup
purely from a FLOP/byte roofline gap, since roofline models do not capture integer
address computation, SLM traffic, or barrier/sync overhead in this kernel.

### 5.4 Capability-gate range is a validated range, not an architectural ceiling
`max(Dh, Dv) <= 256` and `compress_ratio in {2,4,8,16}` reflect what has been verified,
not a hardware limit. Extending either (e.g. to support a future model with head
dimension > 256) requires re-verifying accumulator register pressure and GRF spill at the
new size before relaxing the guard — do not just widen the `static_assert`/predicate
without re-measuring.

## 6. Testing and benchmarking recipe (Q3-specific)

- **A/B protocol**: gate any new variant behind an explicit compile-time flag so one
  binary can alternate baseline vs. optimized in the same process and device state;
  always report the **min over >= 10 runs**, never a single-shot time — first-run
  overhead alone can inflate measured latency by 2-3x.
- **Correctness tolerance**: compare against an FP32/FP64 numeric reference with an
  absolute tolerance around `2e-2` to `6e-2` (FP16-stored intermediate state, e.g.
  summaries/scores, carries ~1e-3 relative noise — much larger than the ~1e-5 error of
  the FP16 dpas math itself). Near any top-k selection boundary, also require an absolute
  margin of cleanly-separated tokens (roughly 8), never a bare percentage threshold —
  ties at the boundary are common and are a property of the selection stage, not a bug in
  Q3 itself.
- **Test geometry must actually exercise sparsity**: construct tests where the selected
  count is strictly less than `n_complete` (i.e. `top-k < history/R`). If the budget is
  large enough that every complete block is always selected, the selected-block loop
  becomes a no-op dense scan and any bug in the block-loop/metadata-gather logic will not
  be caught by the test.
- **Boundary cases to always cover**: zero selected blocks, a single selected block,
  selected count exactly at a step boundary (multiple of `GQA_BLOCKS`), empty causal tail
  (`tail_rows == 0`), non-power-of-two/odd head sizes if the capability gate allows them,
  and a selected block whose 16-row step straddles two physical pages (must NOT take the
  `contiguous` fast path).
- **Profiling approach**: prefer **ablation** (temporarily delete a section of the
  kernel, rebuild, re-time the delta) over reading an instruction/opcode histogram from
  the compiled assembly — this kernel family is dominated by memory-message issue rate,
  and opcode counts (e.g. `add`/`mov`/`shl`) can dominate the histogram while contributing
  negligible actual runtime. When checking for register spill, look for explicit
  `spill size N` (real GRF spill, costly) vs. `spill flag store/load` (predicate-register
  spill only, cheap) in the compiler's shader dump — do not treat all "spill" mentions as
  equally severe.

## 7. Relevant external validation (self-contained summary, no need to read other repos)

- Another sparse-attention implementation independently validated the same "one query
  token, one KV head, all GQA-replicated heads packed into one wide tile, no cross-token
  union" design as this kernel's Route B — supporting evidence that this is a sound
  general strategy for GQA sparse attention on wide-SIMD/systolic hardware, not just an
  artifact of this codebase's specific shapes.
- Not every sparse-attention codebase has actually enabled true selected-KV sparse
  attention in production: some still fall back to a dense attention pass with an
  explicit `-inf` mask that merely zeroes out non-selected positions after the fact. That
  approach does not achieve the O(selected) memory-traffic reduction that this kernel's
  direct paged-KV gather does, even if a generic flash-attention backend can skip
  fully-masked tiles internally — it is not equivalent to guaranteeing sparse access
  complexity, and is not a pattern to imitate here.
