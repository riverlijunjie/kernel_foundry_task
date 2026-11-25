生成的文件
OpenCL kernel 文件
gemv_q4k_shuffle.cl
包含三个 Q4_K kernel：

gemv_q4k — 普通 GEMV（OPG=16，与 SYCL 版对应）
group_up_gate_q4k — MoE up+gate 融合 GEMV + SiLU 激活（OPG=8/BS=1）
down_merge_q4k — MoE down 投影加权求和（OPG=4）
gemv_q5k_shuffle.cl
包含两个 Q5_K kernel：

gemv_q5k — 普通 GEMV，正确处理 5-bit 高位（qh 的列主序 bit packing）
down_merge_q5k — MoE down 投影（OPG=4）
gemv_q6k_shuffle.cl
包含三个 Q6_K kernel：

gemv_q6k — 普通 GEMV（OPG=16/BS=2，含 scale-fusion 优化）
sparse_gemv_q6k — 稀疏 GEMV，只计算 indexes 指定的输出行，scatter 写回
down_merge_q6k — MoE down 投影（OPG=4）
权重 layout 说明（Shuffled 格式）
所有 kernel 基于 Shuffle 后的分离布局：

Q4_K: pqs[N*K/256*128] + psl[N*K/256*16]（scales/mins/d/dmin 全部打包进 16 字节）
Q5_K: 在 Q4_K 基础上多出 pqh[N*K/256*32]（高位 bit 的列主序压缩）
Q6_K: pql + pqh + ps(int8 scales) + pd(fp16)，每 block 210 字节
Python 测试文件
test_gemv_shuffle_kernels.py

包含：

纯 Python shuffle 函数——将标准 block_q4/5/6_K 重排为 Shuffled 格式，与 C++ Shuffle 函数逻辑完全一致
纯 Python dequantize 函数——从 Shuffled 格式还原 float32 权重矩阵，用作参考值
正确性测试——与 NumPy 矩阵向量乘参考结果比对（rtol=1e-2）
性能测试——K=2048 规格的 warmup+bench，报告延迟和 GOPS
覆盖场景：gemv_q4k、gemv_q5k、gemv_q6k、sparse_gemv_q6k、down_merge_q4k

运行方式：

pip install pyopencl numpy
python workspace/q4km_moe/test_gemv_shuffle_kernels.py [--device 0] [--verbose]





Q6K_K12288_N4096/Q4K_K12288_N4096/Q4K_K4096_N12288/Q4K_K4096_N4096/Q4K_K4096_N4096/Q6K_K4096_N1024/Q4K_K4096_N1024              OK      0.829       49.9     9.7%     121.4      9.7%   memory       41.4
                OK      0.159      178.7    34.9%     633.8     34.9%   memory       28.4
                OK      0.145      195.2    38.1%     692.4     38.1%   memory       28.4
  Q4K_K4096_N4096               OK      0.069      137.2    26.8%     486.1     26.8%   memory        9.5
                 OK      0.301       11.5     2.2%      27.9      2.2%   memory        3.5
  















测试用例（7个）
TEST_CASES = [
    Q6K  K=4096   N=151936   # vocab head (~487 MB weight)
    Q6K  K=12288  N=4096
    Q4K  K=12288  N=4096
    Q4K  K=4096   N=12288
    Q4K  K=4096   N=4096
    Q6K  K=4096   N=1024
    Q4K  K=4096   N=1024
]
L3 Cache Flush（参考 bench.py）
内联 cache_flush OpenCL kernel，分配 --flush_mib（默认 256 MiB）的 float 缓冲
每次计时迭代前执行 2 次 sweep，确保权重数据每次都从 DRAM 重新加载
避免小权重（如 Q4K K=4096 N=1024，~2 MB）滞留 L3 cache 导致测试偏快
计时（CL Profiling Events）
使用 CL_PROFILING_COMMAND_START/END，精度为设备时钟（纳秒级），排除 host 调度延迟
默认 500 次计时 + 10 次 warmup，报告 mean/median/min/max
Roofline 分析
AI（算术强度）= 2NK / weight_bytes（Q4K≈3.56 FLOP/B，Q6K≈2.44 FLOP/B，均为强 memory bound）
BW% = 实际带宽 / peak_bw × 100
Roof% = 实际 GFLOPS / roofline_GFLOPS × 100
正确性验证（高效）
只对前 --verify_rows（默认 64）行生成正确的 GGUF block → shuffle → dequant → 与 kernel 输出比对
其余行填随机字节（GPU 全量读取以保证 benchmark 准确，不影响计时）
运行方式

python test_gemv_shuffle_kernels.py \
    --device 0 \
    --iters 500 --warmup 10 \
    --flush_mib 256 \
    --peak_bw 512 --peak_fp32 9800 \
    --verify_rows 64
================================================================================
MoE SG kernel optimization (moe_gemv_q4k/q5k/q6k_sg.cl)
================================================================================
Device: Intel Arc B580 (Xe2, 160 XVE, 456 GB/s peak).  Harness:
test_moe_gemv_sg_kernels.py (unmodified), --iters 300 --warmup 20, L3 flushed
before every timed iteration.  Originals kept as moe_gemv_q*_sg.cl.baseline.

1. Diagnosis: these kernels were NOT bandwidth bound
--------------------------------------------------------------------------------
A pure-read kernel with the *identical* dispatch and the *identical* sub-group
block reads (_micro_bw.py) sustains ~410 GB/s, i.e. 0.023 ms for the 9.4 MB
gate_up working set - while the original kernel needed 0.077-0.104 ms.  ISA dumps
(ocloc + IGC_ShaderDumpEnable) showed why: ~5.4-8 SIMD16 instructions per single
4/5/6-bit weight, plus register spilling in group_up_gate_q4k_sg, plus a
redundant 256-load/256-add activation sum per block per matrix.  So the work was
de-quantisation ALU, not DRAM.

2. What was changed (numerics and memory layout are unchanged)
--------------------------------------------------------------------------------
a) Packed pair-word de-quantisation.  Two weights that sit 16 bits apart in a qs
   word are isolated with ONE and (0x000F000F); their 5th bit (Q5_K) / upper 2
   bits (Q6_K) are 16 bits apart in the qh word too, so one shift + one masked or
   completes BOTH weights.  Each pair then converts with a single
   "mov (16) dst:f src<2;1,0>:uw" (+ shr for the high half) instead of
   shr+and+convert per weight.  Byte-wise extraction was also tried and is worse:
   Xe has no direct byte->float mov, so convert_float(uchar) costs three movs.
   An alternative fp32-bit-pattern trick (as_float(0x3F800000|v<<19) == 1+v/16,
   with the bias absorbed by the existing -min*isum term) is kept as ACC_MODE 2;
   it measures the same, and a packed-fp16 variant was rejected (IGC re-packs
   every half, and fp16 accumulation failed the 1e-2 tolerance).
b) Vector accumulators.  The 32 mads of a sub-block used to form ONE serial
   dependency chain; they now feed 8 independent float4 lanes (vlo/vhi), and the
   activations are read from SLM as float4 (one uniform-address d32x4 per 4
   weights).
c) Activation sums for free.  The 256 activations arrive in ONE
   intel_sub_group_block_read8 (512 B) in which component j / lane l holds the
   fp16 pair (32j+2l, 32j+2l+1), i.e. component j IS sub-block j.  Q4_K/Q5_K get
   isum[j] from a single sub-group reduction; Q6_K needs the two half-sums
   separately (different int8 scales), and lanes 0..7 / 8..15 are exactly those
   halves, so three shuffle-xor butterfly steps + two broadcasts produce both.
   This replaces 256 SLM loads + 256 adds per block per matrix.
d) Up/gate split across sub-groups (SPLIT_UG).  The first half of a work-group's
   sub-groups reduce the UP matrix, the second half the GATE matrix.  Live weight
   registers are halved (the q4k kernel no longer spills), the in-flight loads
   per matrix double, and the two halves meet in SLM at the work-group barrier
   that already existed.
e) One block_read4 for the whole 256-byte psl scale record (4 sub-group shuffles
   recover the per-lane 16-bit fields) instead of 6 narrow block reads; Q6_K
   folds its -32 offset into a single per-block scalar to free 16 registers
   (holding them made IGC split the six wide reads into two batches, ~20%);
   down_merge distributes work in contiguous chunks (DM_BLOCKED) and fetches the
   routing table with one block read + shuffles instead of a dependent scalar
   load per iteration.

3. Result (full harness, mean ms, all 6 kernels PASS correctness)
--------------------------------------------------------------------------------
  kernel                              baseline   optimized   speedup   BW util
  group_up_gate_q4k_sg [256,2048,512]   0.050       0.036      1.39x   41% -> 57%
  group_up_gate_q5k_sg [256,2048,512]   0.076       0.047      1.62x   33% -> 54%
  group_up_gate_q6k_sg [256,2048,512]   0.104       0.060      1.73x   29% -> 51%
  down_merge_q4k_sg    [256,512,2048]   0.037       0.020      1.85x   28% -> 52%
  down_merge_q5k_sg    [256,512,2048]   0.052       0.024      2.17x   25% -> 53%
  down_merge_q6k_sg    [256,512,2048]   0.053       0.036      1.47x   28% -> 42%
  total                                 0.372       0.223      1.67x
Correctness is verified for KSPLIT = 1, 2, 4, 8, 16 (6/6 each) and no kernel
spills registers at any of those settings.

4. Notes / remaining headroom
--------------------------------------------------------------------------------
* Measured with the strict in-process A/B tool (_ab.py, which alternates the two
  binaries over the same buffers and is immune to the ~30% run-to-run variance of
  separate processes), gate_up is 1.48-1.77x faster and down_merge q4k/q5k
  1.02-1.18x, but down_merge_q6k comes out 0.84-0.90x, i.e. that one kernel is
  slower than the baseline under back-to-back conditions even though it executes
  ~20% fewer instructions - an IGC scheduling artefact that neither reordering,
  register-pressure reduction, scalar-vs-vector SLM loads nor the work
  distribution could recover.  It is the only regression; keep
  moe_gemv_q6k_sg.cl.baseline if down_merge_q6k dominates a given workload.
* gate_up now runs at 80-85% of the 410 GB/s that the pure-read kernel of the
  same shape achieves, so it is close to the practical ceiling; down_merge sits
  at 60-70% (its loads-only floor is 0.0176 ms for q6k).
* KSPLIT: the kernels are now bandwidth bound, so LESS K-splitting is better than
  the harness heuristic's choice of 8.  With MOE_KSPLIT=2 (gate_up) / 4
  (down_merge) another 7-10% is available (_tune.py prints the full table); the
  harness's choose_ksplit() was intentionally left untouched.
* Tools added: _micro_bw.py (roofline/ceiling probe), _tune.py (in-process KSPLIT
  and build-flag sweep), _ab.py (baseline vs optimized A/B in one process).
