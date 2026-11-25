# -*- coding: utf-8 -*-
"""
================================================================================
PageAttention CM kernel 优化总结 — PA_PERF=1 (seq_len=2558) @ PTL 4Xe machine-2
================================================================================

目标命令:
    cd ~/river/aboutSHW && source venv/bin/activate \
      && cd opencl/tests/pageatten && PA_PERF=1 python3 test_pa.py

配置: seq_len=2558, num_heads=32, num_kv_heads=8 (GQA), head_size=128,
      block_sz=256, blocks_per_trunk=16 (trunk_sz=4096),
      sparse_block_sz ∈ {256, 1}, density ∈ {1.0, 0.33},
      KV-cache ∈ {NONE(FP16), BY_TOKEN(INT8), BY_CHANNEL(INT8)}。
硬件: Intel PTL 4Xe iGPU, 32 EU, XMX FP16 峰值 ≈ 20 TFLOPS, 128KB SLM。

优化方法: dev_kf_distill_master_opt (MAP-Elites/遗传式搜索 + ablation 定位瓶颈,
          correctness-gate + 鲁棒 interleaved A/B 度量, 共享 iGPU 抗噪)。

================================================================================
1. 优化结果 (已提交 commit d94dce3, 分支 river/opt_ptl4xe_HD_72, 已在真实 test_pa.py 验证)
================================================================================

改动 (cm_pa_xe2.hpp, 共 +12 行): 把 INT8/U8 路径的 OPTIMIZED_SPARSE_PIPELINE 关闭,
让 INT8 走 legacy 连续流水线 (FP16 仍走 optimized)。

根因: 块粒度 OPTIMIZED_SPARSE_PIPELINE 在每个 256-token 块开始处都要重新预热 SLM 流水线
(2 次 staged load + slm_fence + sbarrier) 并复位 ring。seq=2558/trunk=4096 时有 ~10 个活跃块,
重复的 prime/barrier 开销在 INT8(U8) 路径上反而比 legacy 单次 prime 慢; 而 legacy 路径本身已经
通过 skip_compute/skip_load 支持块粒度稀疏跳过。

性能对比 (sparse_block_sz=256, density=1.0, avg ms, machine-2):
    KV 模式        baseline    优化后     加速      util(基线->优化)
    NONE  (FP16)    4.95        4.95      1.00x     54.1% -> 54.1%  (未改动)
    BYTOKEN(INT8)   7.39        6.15      1.20x     36.3% -> 43.6%
    BYCHAN (INT8)   7.46        6.40      1.17x     35.9% -> 41.9%

其它配置: density=0.33 中性/微正 (INT8 BYTOKEN 鲁棒 A/B +7.7%); sparse_block_sz=1 与 FP16 路径
完全不受影响; 全部 3 种 KV 模式在 density 1.0/0.33 下正确性通过 (check_close)。

================================================================================
2. 距离 roofline 还有多远
================================================================================

FLOP (test 全方阵约定 2*nh*S^2*d) = 53.6 GFLOP。XMX FP16 峰值 = 20 TFLOPS。
算术强度 AI ≈ 1135 FLOP/byte (最小 DRAM 占用 ~47MB) —— 远在 roofline 拐点右侧,
即 **compute / on-chip(SLM+barrier) bound, 不是 DRAM 带宽 bound**。

    路径               实测      达到的算力     占 XMX 峰值     距峰值
    INT8 BYTOKEN 优化   6.15ms    8.72 TFLOPS    43.6%          2.29x
    INT8 BYCHAN  优化   6.40ms    8.38 TFLOPS    41.9%          2.39x
    FP16 (同等注意力)   4.95ms   10.82 TFLOPS    54.1%          1.85x   <- 现实可达的近端 floor
    --------------------------------------------------------------------
    XMX FP16 峰值       —        20.0  TFLOPS   100%           1.0x    (理论上界, 不可达)

两层差距:
  (a) INT8 优化后 -> FP16 floor: 6.15 vs 4.95 ms, 还差 ~24% (~10pp util)。
      这是 INT8 特有的 dequant + u8 经 SLM 中转的开销 (FP16 直接 global->reg->DPAS, 无 SLM 中转)。
  (b) FP16 floor -> XMX 峰值: 54% util。注意力是 softmax 串联的两段 GEMM (QK 后必须做
      online-softmax 再做 PV), DPAS 之间被 transpose / exp / 归约打断, 且 PV 的 P 来自 softmax
      无法 int8 化; 54% 已是该 iGPU 上成熟 flash-attention 风格 kernel 的良好水平。

================================================================================
3. 为什么不能继续提升 (已逐一实测排除, 全部 correctness-gate + 鲁棒 A/B)
================================================================================

[1] 源码级微优化 —— 全部中性或变慢:
      dequant 融合 fma 0.97x, 向量化 dequant 0.93x, tail-zero guard 0.99x,
      K-loader unroll 1.00x, linear-softmax 0.96x。
      原因: dequant 行循环编译器已充分向量化; ablation 显示 dequant 运算本身只占 ~4.8%。

[2] 编译 flag (去 -g2/-mdump_asm, -O3, GRF auto) —— 0.997x, 对 codegen 无影响。

[3] barrier/ring 结构:
      - 去掉 per-step 冗余 cm_fence: 0.98x (变慢, fence 是必需的)。
      - 加深 SLM ring 4->8 槽 + prefetch+4: 0.98x (4 槽已足够隐藏延迟, 8 槽反增 SLM/占用开销)。

[4] 结构性重写 —— 均失败:
      - "dequant 直出寄存器"(FP16 式, 不过 SLM): 0.36x (慢 2.8x)。每条 lane 重复 dequant
        16 次, 16x 冗余致命。SLM 共享 (一次 dequant 给 16 lane) 才是正确结构。
      - int8 QK DPAS (U8xU8, 2x 收缩宽度) + 非对称量化 4 项修正: 数值接近 (max err 0.13,
        过不了 1% 严格容差) 且 **慢 33x** —— 每步 16x16=256 次标量修正循环在 GPU 上是灾难,
        把 int8 DPAS 的收益完全吞掉。需全向量化修正才可能, 但容差仍是硬伤。

[5] 真正的瓶颈 = u8 经 SLM 中转 (write -> barrier -> read), ablation: no-SLM-write +43%
      (该数被 DCE 高估)。唯一对症的 lever 是 "u8 存 SLM"(K/V 字节减半, 读后再 dequant), 但:
      - split-WG 生产者 (8 lane K / 8 lane V) 不能只改一半: 单独去掉 K 或 V 写各自 regress 到
        0.85x (流水线失衡)。必须 K+V 同时改。
      - **该 INT8 kernel 已 GRF 饱和 (GRF count 256/256, 且有 flag spill)**。在 consumer 端加
        dequant 会溢出寄存器 (spill) -> 反而变慢。这与 machine-1 上 int8-QK/direct 撞到的是同一堵墙。
      - 因此 u8-SLM 改造是多小时、高风险、净收益不确定的工程, 在寄存器预算下大概率得不偿失。

结论:
  本次提交的 **+17~20% (INT8 稀疏 density=1.0)** 是 "受限、低风险、保正确性" 改动下的现实上限,
  已落地并在真实 PA_PERF=1 test_pa.py 上验证, FP16 与 dense 路径零回归。
  要再向 FP16 floor (~再 24%) 逼近, 需要一次 "感知 256-GRF 预算" 的 u8-in-SLM 重设计
  (K/V 一起字节减半 + 读后 dequant, 严格管理寄存器), 且最好有 unitrace 硬件计数器佐证
  (本环境不可用)。在没有明确收益证据前不建议投入。

================================================================================
附: 复现实验脚本与完整记录
    harness: /mnt/river/kernel_foundry_master/kf_search/ (rsh2.py, orchestrator2.py,
             perf_eval.py, sparse_ab.py, gen_sparse.py 等)
    机器 machine-2 构建前置: 必须 export CM_FE_DIR=/home/intel/river/
                            (否则 CM 编译器找不到 libclangFEWrapper.so, clBuildProgram 失败)
================================================================================
"""
