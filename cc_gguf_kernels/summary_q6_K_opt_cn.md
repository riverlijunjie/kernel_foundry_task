# Q6_K `fc_gguf_dp4a` 内核优化总结

> 目标：优化 `workspace/cc_gguf_kernels` 下的 `fc_gguf_dp4a` 内核，在 **Q6_K 格式、K=2048、N=248320、M=1**（LM head / 词表投影 GEMV）场景下的性能。
> 方法：基于 MAP-Elites 的进化式搜索（`dev_kf_distill_master_opt` skill）。
> 硬件：远程 machine-3（Windows PTL 12Xe，Intel Arc B390 / Panther Lake Xe3，96 CU，实测 DRAM 峰值 ~390 GB/s）。
> 测试命令：`python3 ./harness/bench.py --config ./configs/q6k_vocab.json --out ./results/test_results.json`

---

## 最终结果

| 版本 | 中位耗时 | 达成带宽 | 加速比 | 正确性 |
|---|---|---|---|---|
| 基线 `fc_gguf_dp4a`（NROW=4） | 6.49 ms | 64 GB/s | 1.00x | PASS |
| **优化版（SLM staging + 64B 合并加载）** | **4.09 ms** | **102 GB/s** | **1.58x** | PASS（rel_l2=2e-4, cos=1） |

最终内核：`kernels/fc_gguf_dp4a.cl`（带 `[EVOLVE_START]/[EVOLVE_END]` 标记），备份于 `kernels/_variants/fc_gguf_dp4a_FINAL_slm_uint16.cl`。

---

## 问题分析

- Q6_K：`block_elem=256`，`block_bytes=210`（存储步长，非 16/64 对齐）。K=2048 → 每行 8 个 block。
- `SG_SIZE=16`（硬性要求），`NROW=4`（来自 `ov_jit.dp4a_jit` 默认值）。
- 权重总量 = 248320 × 8 × 210 ≈ **397.8 MiB**，远大于 L2（16 MiB），每次推理都必须完整地从 DRAM 读一遍。
- **原始内核瓶颈**：相邻 lane 读取的 block 在内存中相隔 210 字节（strided），访问不合并（uncoalesced），受"访存事务数"限制，而非带宽。

### 关键发现：真实带宽天花板
- 配置文件里的 `peak_bw=110 GB/s` 是**错误的**。用一个干净的流式读取探针（`harness/bw_probe.py`）实测：
  - 连续合并读 398 MiB → **403 GB/s**；读 2 GiB → 386 GB/s。**真实 DRAM 峰值 ~390 GB/s**。
- 因此基线的 64 GB/s 仅为可达带宽的 ~16%，说明**不是带宽瓶颈，而是访存模式（strided/uncoalesced）瓶颈**，存在巨大优化空间。

---

## 测量方法论（踩坑记录，非常重要）

- **必须**在**全新进程、单个 case、warmup≥40** 的条件下测量每个变体。
- 若在同一进程里连续跑多个 case，会得到**虚假的**逐 case ~2x 加速（GPU 升频 + 缓存预热假象，甚至出现 970 GB/s > 物理峰值）。
- 冷启动首跑会因 GPU 未升频而偏慢；同样条件下基线稳定在 ~6.5 ms（standalone/reverse-sweep/single 均为 6.49~6.58）。

---

## 尝试过的方法与结果（进化搜索历程）

| # | 方法 | 结果 | 结论 |
|---|---|---|---|
| V1 | **16-lane 半 block 映射**：`blocks_per_row=8 < SG=16` 时让全部 16 lane 都工作（lane%8=block，lane/8=half） | 6.69 ms | **无提升**。说明不是 subgroup 内 MLP/占用率不足；确认是访存模式瓶颈 |
| V2 | **SLM staging（uint4 / 16B）**：把连续的 NROW 行权重区域合并加载进 local memory，barrier 后从 SLM 解码 | 5.05 ms（NROW=4）| 合并访存生效，64→82 GB/s，**1.28x** |
| V2b | V2 + **NROW 扫描** | NROW=1:4.23 / **NROW=2:4.19(最佳)** / 4:5.05 / 8:9.33 | 小 NROW → SLM 占用小 → 占用率高 → 更快；NROW=8 的 SLM 拖垮占用率 |
| V2c | V2 + **16-lane SLM 解码** | 4.19 ms（与 8-lane 相同）| **解码不是瓶颈** |
| V3 | **双缓冲软件流水线**：解码当前行时，把下一行的 DRAM 读预取到寄存器（in-flight），2 个 SLM buffer 乒乓 | 4.09 ms（NROW=4/8）| 仅微弱提升 |
| V4 | **64 字节 uint16 合并加载**（整块 tile，NROW=4 使 tile 基址 64 对齐）+ 16-lane 解码 | 4.11 ms | 加载宽度是关键杠杆（见下），但被解码阶段串行掩盖 |
| — | uint8（32B）/ NROW=2、去循环展开降寄存器压力 | 4.13 / 4.10 ms | 均收敛到 ~4.1 ms |

### 关键杠杆：加载宽度（load width）
用 tiled-staging 探针（模拟内核访存结构，无解码）实测：
- 16 字节 `uint4` 加载 → **103 GB/s**
- 64 字节 `uint16` 加载 → **174 GB/s**（1.7x！）

因此最终内核采用 **64 字节 uint16 合并加载**（NROW 为 4 的倍数时 tile 基址 64 对齐，N=248320 可被 4 整除，105 个 uint16 正好对齐）。

---

## 性能墙分析（为什么停在 ~4.1 ms）

- **诊断**：把解码量减到 1/4，耗时**仍为 4.1 ms** → 说明**解码计算不是瓶颈，拷贝（DRAM→SLM）才是**。
- 但探针里同样的拷贝能到 173 GB/s（2.4 ms），内核里的拷贝却只有 ~100 GB/s。
- **根因**：解码代码的**寄存器压力**（8 个 int 累加器等）降低了内核占用率（occupancy），从而拖慢了拷贝阶段的 DRAM 读。去循环展开也无法解除。
- 拷贝与解码在单个 kernel 内**串行**，跨 subgroup 的重叠又受占用率限制，无法充分利用 DRAM。
- 若拆成两个 kernel 解耦拷贝/解码，会带来 **2× DRAM 流量**，反而更慢。
- 结论：**~4.1 ms / ~100 GB/s 是当前单 kernel 结构在该硬件上的实际上限。**

---

## 最终内核结构（SLM staging）

```
每个 subgroup 负责 NROW=4 个输出行：
  1. 全部 16 lane 用 64B uint16 合并加载，把连续的 NROW 行权重区域
     （NROW*8*210 字节，64 对齐）搬入 __local SLM
  2. barrier
  3. 全部 16 lane 从 SLM 解码（lane%8=block，lane/8=128 元素的 half），
     dp4a（SWAR 解包 + dot_acc_sat_4x8packed_us_int）
  4. sub_group_reduce_add 归约后写出
```
- 相比原始版本：把 strided/uncoalesced 的 DRAM 读，转为**合并、64 字节对齐**的 DRAM 读 + 快速的 SLM 解码。
- 对 OV GGUF 格式保持 drop-in（NROW=4 默认，无需改权重布局）。

---

## 复现方式（远程驱动）

- 驱动脚本：`workspace/cc_gguf_kernels/remote_run.py`（基于 paramiko，因远程无 sshpass）。
- 远程带 pyopencl 的 Python：`C:\Users\Local_Admin\river\py310\Scripts\python.exe`；远程目录 `D:\river\gguf_kernels`。
- 常用命令：
  - `python3 remote_run.py setup`：上传 harness/kernels/configs
  - `python3 remote_run.py bench q6k_vocab.json`：上传内核+config，远程运行，回传结果
  - `python3 remote_run.py gpu`：查看远程 OpenCL 设备
- 带宽探针：`harness/bw_probe.py`（`flat` 流式、`tiled`/`tiled16` 分块 staging，用于定位带宽/结构天花板）。

---

## 后续可探索方向

- 二 kernel 方案中，若第一趟能把权重**重排（repack）**为解码友好的对齐布局，并将其作为**一次性摊销**成本（LM head 每 token 复用），则第二趟解码可接近流式带宽——但当前 harness 只计时 dp4a kernel，无法体现摊销收益。
- 通过 Intel 编译器的寄存器分配/大 GRF 模式提升占用率（当前 OpenCL 路径不易控制）。
- 用 `intel_sub_group_block_read` 做硬件合并读，但仍需 SLM/shuffle 重排以满足 Q6_K 的 block 内字节布局。
