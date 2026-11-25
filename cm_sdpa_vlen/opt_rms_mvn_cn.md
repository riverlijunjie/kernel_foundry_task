# RMS 归一化 kernel 优化总结（PTL 4Xe，MAP-Elites）

> 用 `dev_kf_distill_master_opt` skill 的 MAP-Elites 进化算法，在远端 **PTL 4Xe machine-1**
> 上优化 cmd-3 对应的 kernel `rms_gpu_bfyx_opt.cl`（Qwen3-Omni-4B C6 Thinker RMSNorm）。
> 结果：**gmean ~1.70x**，正确性全部通过（真实 cmd-3 `accuracy: all good`）。

---

## 1. 目标与对象

- **测试命令 cmd-3**：`opencl/tests/omni_norm/test_omni_rms.py`
- **被优化 kernel**：`rms_gpu_bfyx_opt.cl`（OpenVINO intel_gpu 的 verbatim RMSNorm kernel）
- **三个 shape（"ov" 配置，即 OV 生产实际派发的动态 JIT 配置）**：

| case | 角色 | shape | D | rows | LWS | items/WI |
|---|---|---|---:|---:|---:|---:|
| hidden | input+post_attn LN | 2556×2560 | 2560 | 2556 | 512 | 5 |
| q_norm | GQA query norm (32 head) | 2556·32×128 | 128 | 81792 | 16 | 8 |
| k_norm | GQA key norm (8 head) | 2556·8×128 | 128 | 20448 | 16 | 8 |

---

## 2. 方法（MAP-Elites 进化搜索）

严格按 skill 的方法执行：

- **基因组**：一份 kernel 源码；**行为描述符**由 `optimization_classifier.classify(code,"opencl")`
  从代码静态映射到 `(memory_opt, compute_opt, parallelism_opt)` 阶梯 cell（0–3 级）。
- **适应度 = 正确性门控 + 加速比**：三个 case 全部 `np.allclose` 通过才给 `perf_score=5`，再加
  `gmean(baseline_med / candidate_med)`；任何不正确的 kernel 一律低于任何正确的 kernel。
- **evaluate（远端 GPU）**：候选 kernel → scp 到 machine-1 → 跑 `_mapelites_eval.py`（复用
  test_omni_rms 的 rotated-buffer 计量）→ 解析每 case 的中位数 + 正确性 → 回填 archive。
- **generate（变异算子）**：由本 agent 充当 LLM，依据阶梯 cell + 上一版的 profiler/带宽反馈
  产生下一版 kernel。
- **计量口径**：设备端硬件时间戳（`CL_PROFILING_COMMAND_START/END`），并用**多缓冲轮换**破坏
  L2/SLC 缓存复用（与本仓库更新后的 test_omni_rms 一致），取 100+ 次中位数。

### 关键连接设置
- 用 `pexpect` 一次性安装 SSH 公钥到 machine-1（密码来自 remote_machine.list，经环境变量传入、
  不落盘、不打印），之后全部免密 ssh/scp。

---

## 3. 基线（cache-defeating 计量）

| case | baseline 中位数 | 有效带宽 |
|---|---:|---:|
| hidden | 730.8 µs | ~36 GB/s |
| q_norm | 1305.1 µs | ~32 GB/s |
| k_norm | 326.1 µs | ~32 GB/s |

**关键观察**：三个 case 的有效带宽仅 ~32–36 GB/s，远低于该平台 DRAM 峰值（~110+ GB/s）。
说明 kernel **不是带宽饱和，而是低效**——存在大量优化空间。

---

## 4. 根因分析

1. **合并访存的 block read/write 路径在 OV 配置下形同死代码**：
   - q/k 的 `LWS==SUB_GROUP_SIZE(16)`，读写路径的判据 `workers_per_data > SUB_GROUP_SIZE` 为假 → 走标量；
   - `USE_BLOCK_WRITE` 宏因 C 运算符优先级 bug（`& 0xF == 0` 解析为 `& (0xF==0)` = `&0` = 0）**恒为 0**，
     写路径永远走标量；
   - hidden 的 `items_num=5 < JIT SUBGROUP_BLOCK_SIZE=8`，block 循环迭代 0 次 → 也走标量。
   - 结果：三个 case 实际都在做**逐元素半精度标量访存**（每指令仅 2 字节/lane）。
2. **多子组归约用对数树 + 每层 barrier**：hidden（32 子组）要 5 层 barrier；q/k（单子组）却仍付出
   SLM 往返 + barrier 的纯开销（单子组下 `sub_group_reduce_add` 已完成整行求和）。
3. `native_powr(x,2)` / `powr(sqrt,-1)` 的写法（后被证实编译器已折叠，非瓶颈）。

---

## 5. 搜索轨迹（每步一个变异，全部正确，cell 均为 (2,2,2)）

| 变体 | 核心改动 | gmean 加速 | 结论 |
|---|---|---:|---|
| v1_sq | `x*x` 代替 `native_powr(x,2)`；`native_rsqrt` | 1.008x | 计算**不是**瓶颈 |
| v2_blockrw | 强制 `USE_BLOCK_WRITE=1` + 判据放宽为 `>=` → q/k 向量化 | **1.471x** | q/k 各 ~1.8x，hidden 不变 |
| v3_cascade | load/store **级联** block8→4→2→scalar，任意 items 都能向量化 | 1.501x | hidden 737→620 |
| v4_noslm | 单子组行(q/k)完全跳过 SLM 与 barrier | 1.588x | q/k → 699/177 µs |
| v5_fastreduce | 对数树归约改为 2 barrier 一次性归约 | 1.692x | hidden 631→526 |
| **v6_1barrier** | 每个子组冗余归约 → **仅 1 个 barrier** | **1.697x** | hidden 526→~511（最终采用） |

---

## 6. 最终结果

**150 次中位数确认 + 真实 cmd-3 端到端验证（`accuracy: all good`）：**

| case | baseline | 优化后 (ov) | 对 baseline | 对 OV C6 参考 |
|---|---:|---:|---:|---:|
| hidden | 730.8 µs | **511 µs** | 1.43x | 722→511 (1.41x) |
| q_norm | 1305.1 µs | **705 µs** | 1.85x | 1267→705 (1.80x) |
| k_norm | 326.1 µs | **177 µs** | 1.84x | 319→177 (1.80x) |
| **gmean** | | | **~1.70x** | **~1.66x** |

**有效带宽**：优化后 q_norm/k_norm 达 **~114 GB/s（≈平台峰值，已最优）**；hidden 约 51 GB/s，
受限于 `LWS=512` 的 workgroup barrier + 归约串行（load→归约→store 必须串行，RMS 语义决定），是
剩余的性能上限所在。

---

## 7. 最终 kernel 的 4 项优化（v6）

1. **向量化 load/store 级联**（block8 → block4 → block2 → scalar）：让合并访存的 sub-group
   block 路径对**任意 items_num** 都能触发，而不只是 JIT block size 的整数倍。
2. **平方用乘法、归一化用 `native_rsqrt`**（去掉 `native_powr`）。
3. **单子组行（q/k）完全跳过 SLM 和所有 barrier**（`sub_group_reduce_add` 已足够）。
4. **多子组行（hidden）用 1 个 barrier 完成跨子组归约**（每个子组冗余归约 32 个 partial），
   替代原来的多层 barrier 对数树。

---

## 8. 交付物与复现

- **优化 kernel（已安装为激活版本，本地 + 远端）**：
  `opencl/tests/omni_norm/rms_gpu_bfyx_opt.cl`
- **原始 verbatim 基线（已保留）**：`rms_gpu_bfyx_opt.cl.baseline`（本地 + 远端）
- **搜索工具 / 全部变体 / archive**：`opencl/tests/omni_norm/_mapelites/`
  - `mapelites.py`（`baseline` / `add <kernel> <label>` / `show`）
  - `variants/v1_sq.cl … v6_1barrier.cl`、`archive.json`
  - 远端评测器 `_mapelites_eval.py`

**复现验证**：
```bash
cd /home/intel/river/aboutSHW && source venv/bin/activate \
  && cd opencl/tests/omni_norm && python3 test_omni_rms.py
```

**回退到原始 kernel**：
```bash
cp rms_gpu_bfyx_opt.cl.baseline rms_gpu_bfyx_opt.cl
```

---

## 9. MVN / cmd-4（`mvn_gpu_bfyx_opt.cl`）

用**完全相同的方法与工具链**优化了 cmd-4 对应的 MVN（LayerNorm）kernel。

### 基线与根因
| case | shape | D | LWS | baseline | 有效带宽 |
|---|---|---:|---:|---:|---:|
| vit | 6864×1152 | 1152 | 256 | 691.6 µs | 45.7 GB/s |
| merger | 1716×4608 | 4608 | 1024 | 869.0 µs | 36.4 GB/s |

- baseline 把 input **从全局内存读了 3 遍**（均值、方差、归一化各一遍），且标量访存；
- 用 `work_group_reduce_add`×2 + `work_group_broadcast`，在**大 workgroup**（vit 256 / merger 1024）上归约，开销大。

### 搜索轨迹
| 变体 | 核心改动 | gmean 加速 | 结论 |
|---|---|---:|---|
| v1_cache | 一次性读入寄存器（3 读→1 读） | 1.069x | 3 遍读**不是**主瓶颈 |
| v2_sg_welford | 加 `REQD_SUB_GROUP_SIZE(16)` + 手写 SLM 归约 | **0.666x** | 退化 |
| **v3_cache_sumsq** | 缓存 + 单遍 sum/sum²（`E[x²]-E[x]²`）+ 保留内建归约 + 去掉 broadcast | **1.165x** | 最优，采用 |
| v4_blockread | 加 `REQD_SUB_GROUP_SIZE(16)` + block read | 0.595x | 退化 |

**关键结论**：MVN 上**不要**强加 `REQD_SUB_GROUP_SIZE` 或手写归约——原生 kernel 让编译器自选
SIMD 宽度并对内建 `work_group_reduce_add` 做了针对性优化，强约束反而变慢；瓶颈是**动态 shape 下
大 workgroup 的归约**，访存不是可达瓶颈（`static` 变体在编译期定 LWS 后可达 73–92 GB/s，佐证该结论，
但那不是 OV 生产的动态派发配置）。

### 最终结果（真实 cmd-4 `accuracy: all good`）
| case | baseline | 优化后 (ov) | 对 baseline | 对 OV C6 参考 |
|---|---:|---:|---:|---:|
| vit | 691.6 µs | **592 µs** | 1.17x | 833→592 (1.41x) |
| merger | 869.0 µs | **755 µs** | 1.15x | 910→755 (1.21x) |
| **gmean** | | | **~1.16x** | |

### v3 的 4 项优化
1. input **只从全局读一次**（缓存进寄存器）；
2. **单遍**同时算 sum 与 sum²，方差用 `E[x²]-E[x]²`，省掉第二遍数据；
3. 去掉 `work_group_broadcast`（每个 WI 从两个归约标量自行推出 mean/inv）；
4. 保留内建 `work_group_reduce_add`，**不加任何 sub-group 约束**。

### 交付物
- 优化 kernel（已安装为激活版，本地+远端）：`opencl/tests/omni_norm/mvn_gpu_bfyx_opt.cl`
- 原始基线：`mvn_gpu_bfyx_opt.cl.baseline`
- 工具/变体/archive：`_mapelites/mapelites_mvn.py`、`variants_mvn/`、`archive_mvn.json`、
  远端 `_mapelites_eval_mvn.py`

**回退**：`cp mvn_gpu_bfyx_opt.cl.baseline mvn_gpu_bfyx_opt.cl`

---

## 10. 两个 kernel 总览

| kernel | cmd | gmean 加速 | 主要手段 |
|---|---|---:|---|
| RMS (`rms_gpu_bfyx_opt.cl`) | cmd-3 | **~1.70x** | 向量化 block 访存级联 + 精简 barrier 归约 |
| MVN (`mvn_gpu_bfyx_opt.cl`) | cmd-4 | **~1.16x** | 单次读入 + 单遍 sum/sum² + 去 broadcast |

> 有趣对比：RMS 的大收益来自**向量化访存**（原 block 路径失效）；MVN 的向量化/手写归约反而退化，
> 收益来自**减少读遍数与同步**——同一套 MAP-Elites 方法在两个 kernel 上收敛到不同的最优结构，
> 正体现了「正确性门控 + 按代码阶梯搜索」的价值。
