# test_15k_perf_comparison.py Kernel 优化总结

> 平台：PTL 4Xe machine-2（`intel@10.239.152.94`，Panther Lake iGPU，Xe2，32 EU，2450MHz，128KB SLM，DPAS）
> 日期：2026-06-30　方法：`dev_kf_distill_master_opt` skill 的 MAP-Elites 遗传搜索
> 目标 kernel：`pa_small_q.cm` + `pa_small_q_finalization.cm`（15K 上下文 spec-decode flash-decode）

---

## 一、任务背景

`test_15k_perf_comparison.py` 在 15K past_len、32 Q头 / 8 KV头、head_size=128、block_sz=256 下，
对比三个 flash-decode kernel：

| 场景 | kernel | q_len |
|---|---|---|
| 标准解码 | `pa_single_token.cm` | 1 |
| **投机解码（优化目标）** | **`pa_small_q.cm` + `pa_small_q_finalization.cm`** | 3–16 |
| 基线对比 | `pa_multi_token.cm` | 3–16 |

`pa_small_q` 是真正的优化对象：每个 work-item（`lws=[1,1,1]`）处理一个
(tile, head-chunk, kv-partition)，DPAS 的 `RepeatCount = Q_head_chunk_size × TILE_Q ≤ 8`。

---

## 二、关键前提：识别并绕开"固定主机重传"测量陷阱

### 现象
原测试的计时函数 `_time_kernel` 在循环里反复调用 `run_once()` 卡墙钟，而 `_enqueue_once`
**每次迭代都重新把整个 KV cache 从 numpy 上传到 GPU**：

```python
t_k = cl.tensor(key_cache.contiguous().detach().numpy())   # ← 每次迭代重建（~63MB fp16 / ~31MB int8）
t_v = cl.tensor(value_cache.contiguous().detach().numpy())
...
out.copy_(torch.from_numpy(t_out_final.numpy()))           # ← 每次读回
```

这块 host→device memcpy 与 q 无关、恒定约 **3–6ms**，和 kernel 执行混在一起被计时：

```
测到的墙钟时间 = 固定重传(恒定 ~3-6ms) + kernel 真实执行时间
```

### 后果
q 越小 kernel 占比越小，优化省下的零点几毫秒被恒定重传稀释，**墙钟几乎看不出变化**
（这正是直接跑 pytest 时 q≤8 看似无提升的原因）。

### 正确测法
device tensor 只上传一次，循环里只 enqueue kernel，用 `cl.finish()` 返回的
**GPU profiling 事件时长**计时。harness 自带 `test_pa_small_q._run_perf` 正是这样实现的
（预分配轮转 buffer + profiling 事件）。

### 缓存假命中验证（HOT vs COLD）
担心"省掉重传会让 KV 常驻 L2 而虚高"——实测排除：

| 形状 | HOT（单 buffer 复用，最大化命中） | COLD（60 buffer 轮转，L2 全冷） | cold/hot |
|---|---|---|---|
| q16 fp16 | 2.857ms | 2.920ms | 1.022 |
| q16 int8 | 3.083ms | 3.119ms | 1.012 |
| q8  int8 | 1.614ms | 1.668ms | 1.033 |
| q4  fp16 | 1.093ms | 1.163ms | 1.064 |

两者仅差 1.2–6.4% → 31–63MB 工作集 ≫ iGPU L2，无法常驻，去掉重传不会让 kernel 偏快。
`_run_perf` 用的就是更保守的 cold 轮转路径，测的是真实 DRAM 带宽下的 kernel 时间。

---

## 三、Roofline 与瓶颈定位（profiler 步骤）

- **实测峰值 DRAM 带宽 = 233 GB/s**（256MB CM block-copy 探针）。
- `pa_small_q` 优化后纯 kernel 仅 ~22 GB/s（**约 10% of peak**）。
- 根因（ASM 证据）：`cm_pa_small_q.asm` 寄存器用满 **r255（256/256 GRF）+ flag spills**
  → 每个 EU 只能驻留 1 个线程，occupancy ~12%。这是与 PA_OMNI INT8 kernel 相同的**硬性 GRF 占用墙**。
- 每个 q-token（tile）独立流式读取整个 KV → 真实 DRAM 流量约 16×，接近 roofline 但**含 16× 冗余 KV 读取**。

阶梯判定：*low occupancy → 提升 parallelism_opt / 调整 launch 尺寸*；compute 改写会撞 GRF 墙。

---

## 四、MAP-Elites 搜索：杠杆与结果

搜索沿优化阶梯（correctness-gated，即先过 SDPA 对比 gate 再比速度）评估了如下杠杆：

| 杠杆 | 效果 | 结论 |
|---|---|---|
| **`reduce_split_step` 8→64** | finalize kernel 1.45ms→0.17ms（**8.5×**） | ✅ 通用安全增益，已设为默认 |
| **`tile_q` 2→8**（KV 复用，q 打包） | small_q q16 3.60→2.87ms | ✅ q≥8 用 8，q<8 用 q |
| `k_partition_block_num`>1 | NONFINITE（kernel 多块/分区路径有 bug + rS tile 溢出 GRF） | ❌ correctness gate 拒绝 |
| fused-reduce kernel 改写（exp 只算一次） | 0.997×（中性） | ❌ rss=64 后 reduce 已不在关键路径，撞 GRF 墙 |

**结论**：kernel 源码轴撞 GRF 墙，真实增益在 **config 轴**（tile_q + reduce_split_step）。

---

## 五、最终结果（严格 interleaved min-of-5 A/B，纯 GPU kernel 时间）

基线 = `reduce_split_step=8` + `tile_q=2`（测试原默认）；优化 = `reduce_split_step=64` + 最优 tile_q。

| q_len | fp16 基线→优化 | fp16 加速 | int8 基线→优化 | int8 加速 |
|---|---|---|---|---|
| **16** | 5.05→3.08ms | **1.64×** | 5.54→3.27ms | **1.69×** |
| 8 | 2.22→1.62ms | 1.37× | 2.46→1.79ms | 1.38× |
| 4 | 1.19→1.10ms | 1.08× | 1.40→1.15ms | 1.22× |
| 3 | 1.13→0.89ms | 1.27× | 1.23→0.78ms | 1.58× |

> 注：测试端到端**墙钟**数字（q16 仅 1.05–1.12×）被那块 ~3–6ms 固定主机重传稀释——
> 真实 kernel 加速是上表的 1.37–1.69×（q16/q8）。

### 距离 roofline 的差距
优化后 q16 纯 kernel ~3.28ms / ~10.7 GB/s，仍只有峰值 233 GB/s 的 ~10%。
进一步逼近 roofline 需要 register-budget 重构（multi-thread WG / SLM 暂存以降低单线程 GRF），
高风险、收益不确定，未实施。当前 1.37–1.69× 是低风险 contained 改动的实际上限。

---

## 六、改动清单

### 1. 日志清理（只保留性能数据）
- `test_15k_perf_comparison.py`：删除 `[x/13] Testing...` 进度行、`====`×100 banner、`KEY FINDINGS` 块。
- `test_pa_small_q.py` / `test_pa_decoding.py`：编译参数移除 `-mCM_printregusage`
  （消除 `: N registers` 输出），保留 `-mdump_asm` 作为 ASM 寄存器 profiler 反馈。

### 2. 优化落地
- `test_pa_small_q.py`：`reduce_split_step` 默认 **8→64**（可用环境变量 `KF_RSS` 覆盖；
  head_size=128 仍可整除，A/B 验证正确、无回退）。
- `test_15k_perf_comparison.py`：tile_q sweep 扩展为 `{1,2,4,8}`。

### 3. 准确 kernel 时间打印
- `PerfResult` 新增 `kernel_ms` / `kernel_bw_gbs` 字段（纯 GPU 事件时间，无主机重传）。
- `_benchmark_small_q` 复用 harness 的 `_run_perf` 填充该字段。
- 输出格式：`... | wall=8.027ms min=6.265ms | kernel=3.275ms (10.7 GB/s)`，
  底部 best-per-q 用纯 kernel 时间计算 kernel-only 加速。

### 4. 搜索工具（位于 `kf_search/`）
- `eval15k.py`：kernel-only、correctness-gated 评测驱动（复用 `_run_perf` + SDPA gate）。
- `ab15k.py`：interleaved min-of-N A/B（基线 vs 优化 config，抗 iGPU 争用噪声）。
- `orch15k.py`：kernel 文件 swap + 远程评测编排。
- `v15k/`：基线与变体 kernel 快照（含 fused-reduce 变体）。

---

## 七、复现命令

```bash
# 远程（machine-2，pageatten 目录）：
cd ~/river/aboutSHW && source venv/bin/activate && export CM_FE_DIR=/home/intel/river/
cd opencl/tests/pageatten

# 跑优化后的对比（同时打印 wall 与 kernel 两套时间）：
RUN_PA_PERF=1 python3 -m pytest -s test_15k_perf_comparison.py -k "summary_cmpr0 or summary_cmpr1"

# 纯 kernel 时间评测（KF_RSS / KF_QLENS / KF_TILEQS / KF_CMPR 可调）：
KF_EVAL_MODE=both KF_QLENS=16,8,4 KF_TILEQS=8 KF_CMPR=0,1 python3 eval15k.py
```
