"""Offline, stdlib-only audit of the 28 saved long-context QSA measurements.

Print a reproducible Markdown report (default) or full-precision JSON. Never
imports the GPU harness, NumPy, qsa_work or clops; never modifies input files.
The I/O models are references, NOT guaranteed DRAM bounds or speedup forecasts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
SIZES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
PHASES = ("prefill", "decode")
CACHES = ("cold96MiB", "no-flush")
STAGES = ("Q0", "Q1-project", "Q1-prepare", "Q2-score", "Q2-topk", "Q3")
PEAK_FLOPS = 118.784e12
PEAK_BW = 456e9


def require(condition, message):
    """Do not disable audit checks with python -O."""
    if not condition:
        raise ValueError(message)


def near(actual, expected, label, *, abs_tol=1e-9):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=abs_tol),
            f"{label}: {actual!r} != {expected!r}")


def read_records(path):
    """Ignore compiler noise and NULs, but fail on malformed recognized JSON."""
    records = {}
    tags = {"ENV", "PLAN", "WORK", "BENCH", "PASS", "DONE", "HISTORY"}
    for number, line in enumerate(path.read_text(encoding="utf-8").replace("\0", "").splitlines(), 1):
        tag, _, payload = line.strip().partition(" ")
        if tag in tags:
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{number}: invalid {tag} JSON") from exc
            require(isinstance(record, dict), f"{path.name}:{number}: expected object")
            records.setdefault(tag, []).append(record)
    for tag in ("ENV", "PLAN", "WORK", "PASS", "DONE"):
        require(len(records.get(tag, [])) == 1, f"{path.name}: expected one {tag}")
    require(len(records.get("BENCH", [])) == 2, f"{path.name}: expected two BENCH records")
    return records


def stats(values):
    require(bool(values) and all(math.isfinite(x) and x >= 0 for x in values),
            "empty, negative or nonfinite timing samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    lo, hi = math.floor(position), math.ceil(position)
    return {"mean_ms": statistics.fmean(values), "median_ms": statistics.median(values),
            "p95_ms": ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)}


def compare_stats(actual, saved, label):
    require(actual.keys() == saved.keys(), f"{label}: statistics schema drift")
    for key, value in actual.items():
        near(value, saved[key], f"{label}.{key}")


def category(label):
    if label in STAGES[:3]:
        return label
    if label in ("Q2-dpas", "Q2-partition16"):
        return "Q2-score"
    if label in ("Q2-scalar", "Q2-WG8", "Q2-WG16"):
        return "Q2-topk"
    if label in ("Q3-scalar", "Q3-gqa-W4", "Q3-split-P16", "Q3-reduce"):
        return "Q3"
    raise ValueError(f"unsupported event label: {label}")


def audit_timing(bench, plan):
    labels = bench["event_labels"]
    require(labels == plan["event_labels"], "BENCH/PLAN labels differ")
    require(bench["warmup"] == 100 and bench["samples"] == 20, "unexpected sampling protocol")
    events, hosts = bench["event_samples_ms"], bench["host_samples_ms"]
    require(len(events) == len(hosts) == 20, "missing samples")
    require(all(len(row) == len(labels) for row in events), "ragged event samples")
    # Validate individual events too, not just the potentially finite slice sums.
    stats([x for row in events for x in row])
    totals = stats([math.fsum(row) for row in events])
    host_stats = stats(hosts)
    compare_stats(totals, bench["gpu_sum"], "gpu_sum")
    compare_stats(host_stats, bench["host_elapsed"], "host_elapsed")
    require(set(labels) == set(bench["stages"]), "stage schema mismatch")
    by_label = {}
    for label in dict.fromkeys(labels):
        indices = [i for i, name in enumerate(labels) if name == label]
        by_label[label] = stats([math.fsum(row[i] for i in indices) for row in events])
        compare_stats(by_label[label], bench["stages"][label], label)
    grouped = {}
    for stage in STAGES:
        indices = [i for i, name in enumerate(labels) if category(name) == stage]
        require(bool(indices), f"missing stage {stage}")
        grouped[stage] = stats([math.fsum(row[i] for i in indices) for row in events])
    compare_stats(grouped["Q3"], bench["q3_total"], "q3_total")
    near(math.fsum(s["mean_ms"] for s in grouped.values()), totals["mean_ms"], "stage sum")
    require(bench["post_benchmark_all_payloads_exact"], "post benchmark validation failed")
    return dict(gpu=totals, host=host_stats, stages=grouped, events=by_label)


def work_counts(past, q):
    """Useful causal pairs; not the padded DPAS instruction count."""
    nc = [(p + 1) // 4 for p in range(past, past + q)]
    selected = [min(n, 512) for n in nc]
    tails = [(p + 1) % 4 for p in range(past, past + q)]
    attended = [4 * k + t for k, t in zip(selected, tails)]
    return dict(n_complete_sum=sum(nc), sparse_candidates_sum=sum(n for n in nc if n > 512),
                max_complete=max(nc), dense_rows=sum(n <= 512 for n in nc),
                sparse_rows=sum(n > 512 for n in nc),
                summaries_produced=(past + q) // 4 - past // 4,
                selected_blocks_sum=sum(selected), tail_positions_sum=sum(tails),
                attended_positions_sum=sum(attended), projection_flops=2 * q * 2560 * 640)


def audit_plan(plan, work, bench):
    expected_config = dict(heads_num=24, kv_heads_num=2, k_head_size=256, v_head_size=256,
        hidden_size=2560, indexer_heads=4, indexer_kv_heads=1, indexer_head_dim=128,
        indexer_rotary_dim=32, compress_ratio=4, budget=2048, block_topk=512,
        pa_block_size=16, has_output_gate=True, is_causal=True)
    require(all(plan["config"].get(k) == v for k, v in expected_config.items()),
            "unsupported geometry: analysis is specific to these B1/r4/K512 measurements")
    phase, size, past, q = (plan[k] for k in ("phase", "size", "past", "q"))
    require(phase in PHASES and size in SIZES, "unsupported benchmark shape")
    require((past, q) == ((0, size) if phase == "prefill" else (size, 1)), "not the B1 sweep")
    for key in ("phase", "size", "past", "q"):
        require(work[key] == bench[key] == plan[key], f"shape mismatch: {key}")
    expected_work = work_counts(past, q)
    for key, value in expected_work.items():
        require(work[key] == value, f"WORK.{key}: {work[key]} != {value}")
    sp = plan["score_plan"]
    require(sp["num_tokens"] == q, "score num_tokens mismatch")
    mb = max(8, ((work["max_complete"] + 7) // 8) * 8)
    rows = min(q, ((128 << 20) - 256) // (4 * mb))
    require(sp["budget_bytes"] == 128 << 20 and sp["guard_elements"] == 64,
            "unexpected scratch budget/guard")
    require(sp["scratch_rows"] == rows and sp["max_blocks"] == mb, "scratch geometry mismatch")
    require(sp["scratch_bytes"] == 4 * rows * mb, "scratch payload mismatch")
    require(sp["allocation_bytes"] == sp["scratch_bytes"] + 256, "scratch guard mismatch")
    chunks, tiles = [], []
    for begin in range(0, q, rows):
        end, offset = min(q, begin + rows), len(tiles)
        tiles.extend([[0, t] for t in range(begin, end, 16)])
        chunks.append(dict(row_begin=begin, row_end=end, tile_offset=offset, tile_count=len(tiles) - offset))
    require(chunks == sp["chunks"] and tiles == sp["tile_map"], "chunk/tile coverage mismatch")
    for field, expected in dict(score_scratch_bytes=sp["allocation_bytes"],
            score_payload_bytes=sp["scratch_bytes"], score_rows=rows, max_blocks=mb,
            score_chunks=len(chunks), projection_scratch_bytes=q * 640 * 4,
            split_scratch_bytes=396288 if phase == "decode" and size >= 2048 else 0).items():
        require(bench[field] == expected, f"BENCH.{field} mismatch")
    score = "Q2-dpas" if phase == "prefill" else "Q2-partition16"
    topk = ("Q2-WG8" if work["sparse_rows"] == 0 else "Q2-scalar") if phase == "prefill" else "Q2-WG16"
    q3 = ["Q3-gqa-W4"] if phase == "prefill" else (["Q3-scalar"] if size == 1024 else ["Q3-split-P16", "Q3-reduce"])
    require(plan["event_labels"] == list(STAGES[:3]) + [score, topk] * len(chunks) + q3,
            "unexpected dispatch policy/order")


def roofline(flops, byte_count):
    compute, memory = flops / PEAK_FLOPS * 1000, byte_count / PEAK_BW * 1000
    return dict(useful_dot_flops=flops, reference_bytes=byte_count, compute_ms=compute,
                memory_ms=memory, theory_ms=max(compute, memory))


def compare(model, measured):
    theory = model["theory_ms"]
    return dict(model, measured_ms=measured, gap_times=measured / theory if theory else None,
                eff_percent=100 * theory / measured)


def model_work(plan, w, bench):
    q, past, phase = plan["q"], plan["past"], plan["phase"]
    candidates, sparse = w["sparse_candidates_sum"], w["sparse_rows"]
    ns, selected, attended = (w[k] for k in ("summaries_produced", "selected_blocks_sum", "attended_positions_sum"))
    f1, f2, f3 = w["projection_flops"], 1024 * candidates, 24576 * attended
    # These exact constants follow Fixture/Pipeline, not input allocation sizes.
    rope_positions = set(range(past + 3, past + q + 3)) | set(range(past, past + 4 * ns, 4))
    norm_rope_bytes = 256 * (1 + bool(ns)) + 64 * len(rope_positions)
    history_kv = 2048 * 4 * selected if phase == "decode" else 0
    history_summary = 256 * (past // 4) if candidates else 0
    external = dict(projection_weight=3276800, mixed=5120 * q, new_main_kv_input=2048 * q,
        query_and_gate=24576 * q, output=12288 * q, main_cache_write=2048 * q,
        raw_cache_write=256 * q, summary_cache_write=256 * ns,
        norm_and_rotary=norm_rope_bytes, history_main_kv=history_kv,
        history_summary=history_summary)
    # Current K/V source is already an external input: do NOT read it externally
    # again for Q3. Cache writes are retained only as an explicit cold-I/O convention.
    global_model = roofline(f1 + f2 + f3, sum(external.values()))
    global_model["external_components_bytes"] = external
    global_model["cold_history_plus_weight_bytes"] = 3276800 + history_kv + history_summary

    # Prefill selection IDs were not saved. The dense prefix and each query's own
    # incomplete tail prove this SUBSET of the union, not its actual cardinality.
    if phase == "prefill":
        mandatory = set(range(min(q, 2051))) | {p for p in range(q) if (p + 1) % 4}
        unique_min, unique_max = len(mandatory), q
    else:
        unique_min = unique_max = attended  # one query, distinct blocks + disjoint tail

    # Audit actual DPAS score stores (M8 aligned to tile max, including causal
    # overstore), separately from useful candidate bytes (4*C). Dense rows never
    # publish scores, but mixed DPAS tiles load/compute their columns.
    stores, loaded_q, executed_score_flops = 0, 0, 0
    if phase == "prefill":
        for chunk in plan["score_plan"]["chunks"]:
            for t0 in range(chunk["row_begin"], chunk["row_end"], 16):
                end = min(t0 + 16, chunk["row_end"])
                maximum = end // 4
                if maximum <= 512:
                    continue
                width = ((maximum + 7) // 8) * 8
                stores += 4 * width * sum((t + 1) // 4 > 512 for t in range(t0, end))
                loaded_q += end - t0
                executed_score_flops += 1024 * 16 * width
    else:
        stores, loaded_q = 4 * candidates, sparse
        executed_score_flops = f2  # useful scalar dot convention, not ISA counting
    max_summary = w["max_complete"] if candidates else 0
    selection_out = 4 * q * (512 + 1)  # includes -1 padding, not just selected IDs
    selection_in = 4 * (selected + q)
    stage_models = {
        "Q0": roofline(0, 4096 * q),
        "Q1-project": roofline(f1, 3276800 + 5120 * q + 2560 * q),
        "Q1-prepare": roofline(0, 2560 * q + 1024 * q + 256 * q + 256 * ns + norm_rope_bytes),
        "Q2-score": roofline(f2, 1024 * loaded_q + 256 * max_summary + stores),
        "Q2-topk": roofline(0, 4 * candidates + selection_out),
    }
    q3_base = 2048 * unique_min + 24576 * q + 12288 * q + selection_in
    split = bench["split_scratch_bytes"]
    # All P16 segments are nonempty at these measured shapes. Each partial
    # tensor counted once at each endpoint, regardless of cache hits/rereads.
    if split:
        producer = roofline(f3, 2048 * unique_min + 12288 * q + selection_in + split)
        reducer = roofline(0, split + 12288 * q + 12288 * q)
        stage_models["Q3"] = roofline(f3, q3_base + 2 * split)
        stage_models["Q3"]["theory_ms"] = producer["theory_ms"] + reducer["theory_ms"]
        stage_models["Q3"]["substage_models"] = {"Q3-split-P16": producer, "Q3-reduce": reducer}
    else:
        stage_models["Q3"] = roofline(f3, q3_base)
    topk_coop = "Q2-scalar" not in plan["event_labels"]
    # Scalar emit stops at the largest selected ID: K <= reads <= n.
    # Without saved IDs the exact fifth-pass length cannot be reconstructed.
    read_min = 6 * candidates if topk_coop else 4 * candidates + 512 * sparse
    read_max = 6 * candidates if topk_coop else 5 * candidates
    details = dict(q3_unique_positions_min=unique_min, q3_unique_positions_max=unique_max,
        q3_unique_positions_exact=phase == "decode" or unique_min == unique_max,
        useful_score_bytes=4 * candidates, score_store_bytes=stores,
        score_loaded_query_rows=loaded_q, score_padded_dot_flops=executed_score_flops,
        topk_logical_score_read_bytes_min=4 * read_min, topk_logical_score_read_bytes_max=4 * read_max,
        score_allocation_bytes=bench["score_scratch_bytes"], split_payload_bytes=split,
        stage_reference_sum_ms=math.fsum(m["theory_ms"] for m in stage_models.values()),
        q3_no_interquery_reuse_kv_bytes=2048 * attended)
    return global_model, stage_models, details


def audit_sources(records, root):
    """Bind equations to the exact recorded Python/kernel/header/config sources."""
    sources = records["ENV"][0]["sources"]
    required = {"cm_kernel/benchmark_long_context.py", "cm_kernel/test_qsa_pipeline.py",
                "cm_kernel/qsa_harness.py", "cm_kernel/qsa_score_chunked.py"}
    require(required <= sources.keys(), "missing source provenance")
    for name, digest in sources.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "source path outside workspace")
        require(path.is_file(), f"missing recorded source {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"source drift: {name}")
    tree = ast.parse((root / "cm_kernel/qsa_harness.py").read_text())
    peaks = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in ("PEAK_FLOPS", "PEAK_BW")}
    require(peaks == dict(PEAK_FLOPS=PEAK_FLOPS, PEAK_BW=PEAK_BW), "harness peak assumptions changed")
    return sources


def audit_performance_report(path, rows):
    """Check the 28 existing stage-mean rows against raw samples, to 6 decimals."""
    lookup = {(r["phase"], r["size"], r["cache"]): r for r in rows}
    cache, seen = None, set()
    for line in path.read_text().splitlines():
        if line.startswith("## 4."):
            cache = "cold96MiB"
        elif line.startswith("## 5."):
            cache = "no-flush"
        elif line.startswith("## "):
            cache = None
        cells = line.strip("|").split("|")
        if cache and len(cells) == 10 and cells[0] in PHASES:
            key = cells[0], int(cells[1]), cache
            require(key not in seen, f"duplicate report row {key}")
            row = lookup[key]
            expected = [row["stages"][s]["measured_ms"] for s in STAGES]
            expected += [row["global"]["measured_ms"], row["timing"]["host"]["mean_ms"]]
            for actual, value in zip(cells[2:], expected):
                near(float(actual), value, f"performance report {key}", abs_tol=.5001e-6)
            seen.add(key)
    require(seen == lookup.keys(), "performance report does not contain all 28 mean rows")


def analyze(log_dir, source_root, performance_report):
    expected = {f"{phase}_{size}.log" for phase in PHASES for size in SIZES}
    require({p.name for p in log_dir.glob("*.log")} == expected, "expected exactly 14 shape logs")
    rows, hashes, source_hashes = [], {}, None
    event_count = 0
    for phase in PHASES:
        for size in SIZES:
            path = log_dir / f"{phase}_{size}.log"
            rec = read_records(path)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            current_sources = audit_sources(rec, source_root)
            if source_hashes is not None:
                require(current_sources == source_hashes, "source snapshot changed between shapes")
            source_hashes = current_sources
            plan, work = rec["PLAN"][0], rec["WORK"][0]
            require((plan["phase"], plan["size"]) == (phase, size), "filename/shape mismatch")
            require(len(rec.get("HISTORY", [])) == (phase == "decode"), "history record mismatch")
            if phase == "decode":
                require(rec["HISTORY"][0]["included_in_timing"] is False, "history was timed")
            require(rec["PASS"][0]["all_guards_pass"] and
                    rec["PASS"][0]["own_score_selection_different_rows"] == 0, "validation failed")
            require([b["cache"] for b in rec["BENCH"]] == list(CACHES), "cache order/coverage mismatch")
            for bench in rec["BENCH"]:
                audit_plan(plan, work, bench)
                timing = audit_timing(bench, plan)
                global_model, stages, details = model_work(plan, work, bench)
                rows.append(dict(phase=phase, size=size, cache=bench["cache"], work=work,
                    timing=timing, global_model_name="whole_graph_cold_external_io_reference",
                    global_compute_only_ms=global_model["compute_ms"],
                    global_=None,  # replaced below: 'global' is a Python keyword
                    stages={s: compare(m, timing["stages"][s]["mean_ms"]) for s, m in stages.items()},
                    details=details, event_labels=bench["event_labels"]))
                rows[-1].pop("global_")
                rows[-1]["global"] = compare(global_model, timing["gpu"]["mean_ms"])
                event_count += sum(map(len, bench["event_samples_ms"]))
    require(len(rows) == 28 and event_count == 6960, "incomplete event coverage")
    audit_performance_report(performance_report, rows)
    return dict(peak_flops=PEAK_FLOPS, peak_bandwidth=PEAK_BW, log_sha256=hashes,
                source_sha256=source_hashes, rows=rows,
                audit=dict(logs=14, benchmarks=28, samples=560, events=event_count,
                           performance_mean_rows=28, all_saved_statistics_recomputed=True))


INTRO = r"""# B580 长上下文 QSA：可复现 roofline gap

## 1. 范围与解释

分析 `long_context_logs/{prefill,decode}_*.log` 的全部 **28 组实际测量**：
七尺寸 × 两阶段 × cold96MiB/no-flush，每组 100 warmup、20 samples。
实测列来自 `BENCH.event_samples_ms`：每个 sample 先求所有 event 的和，再取算术平均；
Q2 合并全部 chunk，Q3 包含 split producer/reducer。不是 min、host 时间或首尾事件跨度。
HISTORY、flush、校验、上传/读回、JIT 和 event 间空隙不计入 GPU sum。

**gap = 实测 / 理论参考；eff % = 100 × 理论参考 / 实测。**
这里 eff 只是与指定参考的接近程度，**不是测得的硬件利用率**；gap 不是可实现加速比。
未测量 SIMD、整数、SFU、SLM 或消息吞吐峰值，不为它们编造吞吐数字。

## 2. useful dot 工作量与硬件假设

使用 `qsa_harness.py` 中 P=118.784e12 FP16 XMX FLOP/s、B=456e9 byte/s；
十进制 TFLOP/GB/MB，时间统一 ms；FMA=2 FLOP，FP16=2 B、FP32/int32=4 B。
这是理想 XMX 容量参考；decode score/scalar/split 的 FP32 SIMD 实现不具有此 XMX 吞吐。

令 q 为新 query 数，p 为 past，n(t)=floor((p+t+1)/4)，k(t)=min(n(t),512)，
a(t)=4k(t)+(p+t+1)%4，C=sum_{n(t)>512} n(t)，A=sum a(t)，S 为新完整 summary 数。

- Q1 projection：F1=2q×2560×640；FP32 projection scratch 是 q×640×4 B。
- Q2 score：F2=2×4×128×C=1024C，**排除 dense bypass**。
  `qsa_score_tile_dpas.cm` 在全 dense tile 提前返回；混合 tile 仍执行 dense/padded 列，
  但不发布 dense scores。额外 DPAS 工作不算 useful FLOPs。
- Q3：F3=2×24×(256+256)×A=24576A，涵盖 useful QK+PV。
  不含 online-softmax correction、exp、gate、RMSNorm、RoPE、pooling、reduce、地址计算等。
  GQA kernel N16 含 12 个有效 heads，M/K 补齐也不算 useful 工作。
- Q0、Q1 post、Q2 top-k 的 **useful dot** FLOPs=0，不代表它们没有运算或成本。
- 64K decode：512 个不重复完整块×4+1 tail=**2049**；其中历史 KV=2048 个，
  当前 tail 的外部源已由 Q0 输入计入，绝不能改成读取全部 past=65536。

## 3. 两类参考，不能混用

### 3.1 全图计算下界 + 冷外部 I/O 参考（主表）

T_compute=1000(F1+F2+F3)/P 是该 useful-dot 算法在所设峰值下的理想计算下界。
它忽略中间结果、缓存和非点积成本；即使输入在缓存中也保留此计算约束。

主表理论参考 T_global=max(T_compute,1000×B_ext/B)。B_ext 是如下**冷外部张量一次读写约定**：

|外部张量|字节|
|---|---:|
|projection weight|3,276,800|
|mixed input|5,120q|
|当前 main K/V input（独立于 mixed/query）|2,048q|
|query+gate input|24,576q|
|最终 attention output|12,288q|
|可观察 main cache 更新|2,048q|
|可观察 raw cache 更新|256q|
|新 summary cache 更新|256S|
|Q/K norm weights|256×(1+[S>0])|
|RoPE table|64×不同访问位置数|
|decode 历史 main K/V|2,048×4×sum k(t)，prefill 为 0|
|decode 历史 summary|256×floor(p/4)，仅 sparse score 时；prefill 为 0|

RoPE 位置是 query 的 `{p+t+3}` 与新 summary 的 block-start 集合之并；
本次 prefill 为 q+1 个位置，decode 为 1 个。decode past 均 r 对齐，没有旧 partial raw 读取。
prefill 当前 K/V 已作为外部输入读一次，不为 Q3 再计外部读取；新 summary 同理。
iq、projected、scores、selected/count 和 split partial 是内部中间量，不计入 B_ext。
小型 page table、position/token/tile maps、标量元数据、cache-line 舍入及写分配流量省略。

**不是完整管线的严格 DRAM 下界**：输出/cache store 计为外部写是冷 I/O 约定，
event 完成不保证脏缓存已写回 DRAM；输入可能驻留缓存，特别是 no-flush。
它是忽略中间 spill、跨阶段重读、launch、SFU/整数/SLM 成本的乐观常规参考，
不是已测 DRAM 流量。no-flush 使用同一参考仅为对照，并非宣称每次都读权重/历史 DRAM。
JSON 另列 `cold_history_plus_weight_bytes`，不含当前新 cache 写，帮助核对冷 decode 主输入。

### 3.2 逐阶段理想独立内存参考（分阶段表）

T_stage=sum_i max(1000Fi/P,1000Bi/B)。各阶段的逻辑输入只计一次，
中间量在 producer 写、consumer 读各计一次；同阶段内假设理想 GQA/跨 query/cache reuse。
它**不是**无阶段间 flush 的链式执行严格 DRAM 下界，也不与 T_global 相加。

令 W 为 score 实际 store 字节数（从 chunk/tile 边界重建），L 为 active score tiles
实际加载的 query 行数，M 为 sparse scorer 用到的不同 summary 数；全 dense 时三者为 0。
I=4(sum k(t)+q) 为 Q3 读取有效 selected IDs/count，O=4q(512+1) 含 finalizer 写入的 -1 padding。
R 为上表 norm+RoPE 字节数，U 为下面定义的唯一 KV 位置数保守参考。

|阶段|Bi（byte）|
|---|---|
|Q0|4096q：输入 K/V + cache stores|
|Q1 projection|3276800+5120q+2560q|
|Q1 post|2560q+1024q+256q+256S+R；fresh pooling 保持 SLM FP32，不重读 fresh raw cache|
|Q2 score|1024L+256M+W；不是 score allocation|
|Q2 top-k|4C+O；只计不同 score 一次的缓存理想参考，**不是 radix 总读取量**|
|Q3 非 split|2048U+24576q+12288q+I|
|Q3 split producer|2048U+12288q+I+Z|
|Q3 split reducer|Z+12288q+12288q|

Z=q×24×16×(256+2)×4=396288q 是 partial acc/ml **payload**；
Q3 split 理论为 producer 与 reducer 各自 max 的和，所有本次分区非空。
ml 的两次源码读取按一次逻辑 tensor 读取参考处理，不声称每次访问 DRAM。
guard 是校验用分配，不是 timed kernel payload 流量。

decode 的 U=a(0) 精确；prefill 没有保存全部 selected IDs，**无法恢复实际跨 query union**。
仅使用可证明的子集：dense prefix `[0,min(q,2051))` 与每个 query 自身的 incomplete tail
位置 `{t:(t+1)%4 !=0}` 的并集；U=q（q≤2051），否则本次 r 对齐 q 有 U=3q/4+512。
实际 union 在 [U,q] 内；不是假定全 context 必被选中，也不是旧 qsa_work.q3 的 max-position。
此处 U 仍过于乐观，且不表示 kernel 能把所有共享 KV 永久保留在缓存中。
本次所有 prefill Q3 即使用 U=q，仍由 useful compute 项主导，不改变表中理论时间。

### 3.3 radix 与 scratch 的源码核对

`include/qsa_common.hpp::radix_select_topk`：四次完整 n 扫描，加一次到最大 selected ID
为止的 emit；没有 selected IDs 快照，因此仅知每 sparse row 总读取在 [4n+512,5n]。
`qsa_topk_cooperative.cm`：四次 histogram + 一次 greater/equal count + 一次 emit，固定 6n。
dense 行不读 scores，但仍写 selected/count。字节区间见后表，都是**源码逻辑读取**，
不能直接除以 456 GB/s 当作严格 DRAM 下界：重复访问可在缓存中命中。
无整数/SFU 吞吐假设，不能仅从 top-k gap 判定是 DRAM、整数、依赖链或 SLM 限制。

DPAS scorer 对每 active tile 的 sparse 行写 ceil8(max n_tile) 个 float，包括 causal/padding；
脚本同时输出 useful score bytes=4C 和实际 store bytes。分配只描述同时存活容量，
33 个 chunk 的反复写入不是“只写一次 allocation”，也不是固定读写完整 4 GiB。
"""


def table(headers, body):
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    lines += ["|" + "|".join(map(str, row)) + "|" for row in body]
    return "\n".join(lines)


def ratio_cells(model):
    return [f'{model["theory_ms"]:.6f}', f'{model["measured_ms"]:.6f}',
            "—" if model["gap_times"] is None else f'{model["gap_times"]:.2f}',
            f'{model["eff_percent"]:.2f}']


def render_report(data):
    rows, parts = data["rows"], [INTRO.rstrip()]
    parts.append("## 4. 全图结果：全部 28 组\n\n理论与计算下界均为 ms；eff 是参考比值，不是硬件利用率。")
    for cache in CACHES:
        for phase in PHASES:
            subset = [r for r in rows if (r["phase"], r["cache"]) == (phase, cache)]
            parts.append(f"### {phase} / {cache}")
            parts.append(table(["size", "计算下界 ms", "外部 bytes", "理论参考 ms", "实测 ms", "gap ×", "eff %"],
                [[r["size"], f'{r["global_compute_only_ms"]:.6f}', r["global"]["reference_bytes"],
                  *ratio_cells(r["global"])] for r in subset]))
    parts.append("## 5. 逐阶段结果（全部 cold96MiB）\n\n理论是第 3.2 节的独立内存参考；全 dense Q2-score 理论=0、gap 不定义（—），实测空 dispatch 仍保留。Q1 post/top-k 只提供一次内存参考，不提供运算吞吐下界。")
    for stage in STAGES:
        parts.append(f"### {stage}")
        parts.append(table(["phase", "size", "useful GFLOP", "参考 bytes", "理论参考 ms", "实测 ms", "gap ×", "eff %"],
            [[r["phase"], r["size"], f'{r["stages"][stage]["useful_dot_flops"] / 1e9:.6f}',
              r["stages"][stage]["reference_bytes"], *ratio_cells(r["stages"][stage])]
             for r in rows if r["cache"] == "cold96MiB"]))
    parts.append("## 6. 缓存协议对照与阶段参考和\n\n差值=cold−no-flush；顺序固定 cold 在前，微小差值不作因果归因。逐阶段参考和与全图参考是不同模型。")
    cold = [r for r in rows if r["cache"] == "cold96MiB"]
    warm = {(r["phase"], r["size"]): r for r in rows if r["cache"] == "no-flush"}
    parts.append(table(["phase", "size", "cold ms", "no-flush ms", "差值 ms", "cold/no-flush ×", "逐阶段参考和 ms"],
        [[r["phase"], r["size"], f'{r["global"]["measured_ms"]:.6f}',
          f'{warm[r["phase"], r["size"]]["global"]["measured_ms"]:.6f}',
          f'{r["global"]["measured_ms"] - warm[r["phase"], r["size"]]["global"]["measured_ms"]:.6f}',
          f'{r["global"]["measured_ms"] / warm[r["phase"], r["size"]]["global"]["measured_ms"]:.3f}',
          f'{r["details"]["stage_reference_sum_ms"]:.6f}'] for r in cold]))
    parts.append("## 7. 工作量/流量审计\n\n以下 store/read 是源码逻辑字节，不是测量的 DRAM 流量；U 区间不是实测 union。")
    parts.append(table(["phase", "size", "A attended sum", "U min..max", "useful score B", "score store B", "top-k read B min..max", "score allocation B"],
        [[r["phase"], r["size"], r["work"]["attended_positions_sum"],
          f'{r["details"]["q3_unique_positions_min"]}..{r["details"]["q3_unique_positions_max"]}',
          r["details"]["useful_score_bytes"], r["details"]["score_store_bytes"],
          f'{r["details"]["topk_logical_score_read_bytes_min"]}..{r["details"]["topk_logical_score_read_bytes_max"]}',
          r["details"]["score_allocation_bytes"]] for r in cold]))
    parts.append("## 8. 重点与限制")
    for phase in PHASES:
        r = next(r for r in cold if r["phase"] == phase and r["size"] == 65536)
        m, topk = r["global"], r["stages"]["Q2-topk"]
        parts.append(f'- 64K {phase} cold：全图参考 **{m["theory_ms"]:.6f} ms**，实测 **{m["measured_ms"]:.6f} ms**，'
                     f'gap **{m["gap_times"]:.2f}×**，eff **{m["eff_percent"]:.2f}%**；'
                     f'Q2 top-k 实测占 GPU sum **{100 * topk["measured_ms"] / m["measured_ms"]:.2f}%**。')
    parts.append("- 1024→2048 decode 有 scalar→split-P16 路径切换，不是单调长度曲线。prefill ≥4K 使用 scalar finalizer；decode 为 WG16。\n"
                 "- Q3 无跨 query KV reuse 时，仅 KV 逻辑读取就是 2048A B；64K prefill 约 270.78 GB，除以外存峰值得约 593.82 ms，甚至超过实测 Q3。这只能作缓存敏感性反例，不能作下界。\n"
                 "- 本文不能证明任何阶段的实际 DRAM 带宽、XMX 占用率、SFU 利用率或未来优化收益。执行的 padded DPAS、SIMD、online-softmax、gate、地址/依赖链、SLM/barrier、重复消息和实际 cache misses 都需要单独测量。\n"
                 "- 本次没有重新运行 GPU benchmark，没有完整模型/plugin/token/s/质量结论；保留全部原始样本，不扣历史以外的 timed events。")
    parts.append("## 9. 离线复现与审计\n\n在 `cm_kernel/` 下运行 `python3 analyze_long_context_roofline.py` 输出本文；"
                 "`--format json` 输出全精度 28 组数据、所有六阶段（含 no-flush）和源码/log SHA-256；"
                 "`--check-report LONG_CONTEXT_ROOFLINE_CN.md` 检查本文逐字一致。"
                 "`python3 -m unittest test_analyze_long_context_roofline -v` 运行 CPU-only 回归。\n\n"
                 "可用 `--logs`、`--source-root`、`--performance-report` 指定归档路径；"
                 "默认均相对脚本路径，不依赖工作目录。无第三方依赖、GPU、网络或环境变量需求。"
                 "脚本不写文件；源文件或日志/schema/WORK/事件/统计/已发布 mean 行不一致时非零退出。")
    a = data["audit"]
    parts.append(f'验证覆盖：{a["logs"]} logs、{a["benchmarks"]} BENCH、{a["samples"]} slices、'
                 f'{a["events"]} events、原性能报告 {a["performance_mean_rows"]} 行六阶段 mean。'
                 "全部 saved mean/median/线性 p95 由原始样本重算，所有 ENV.sources 哈希与本地归档源码匹配。")
    parts.append(table(["原始 log", "SHA-256"], [[k, v] for k, v in data["log_sha256"].items()]))
    parts.append("关联：[实测性能报告](LONG_CONTEXT_PERFORMANCE_CN.md) · [README](README.md) · [分析脚本](analyze_long_context_roofline.py)")
    return "\n\n".join(parts) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=HERE / "long_context_logs")
    parser.add_argument("--source-root", type=Path, default=HERE.parent)
    parser.add_argument("--performance-report", type=Path, default=HERE / "LONG_CONTEXT_PERFORMANCE_CN.md")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--check-report", type=Path)
    args = parser.parse_args(argv)
    data = analyze(args.logs, args.source_root, args.performance_report)
    report = render_report(data)
    if args.check_report:
        require(args.check_report.read_text() == report, "generated report differs from saved report")
        print("PASS: saved roofline report matches all 28 analyzed measurements", file=sys.stderr)
    print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) if args.format == "json" else report, end="\n" if args.format == "json" else "")


if __name__ == "__main__":
    main()