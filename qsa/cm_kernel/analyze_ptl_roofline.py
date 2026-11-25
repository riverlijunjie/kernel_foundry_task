"""Read-only roofline audit for the PTL Arc B390 iGPU long-context QSA sweep.

Reads `{log_dir}/{phase}_{size}.log` for the 8 requested sizes
(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072) x 2 phases
x up to 2 cache modes each, and renders `PTL_LONG_CONTEXT_ROOFLINE_CN.md`.

Reuses analyze_long_context_roofline.py's pure ``model_work``/``roofline``/
``stats``/``compare``/``require``/``near``/``compare_stats`` functions
UNCHANGED for the FLOP/byte "useful dot" accounting -- same methodology as the
existing BMG reports, not a new ad hoc formula. Only the two module-level
hardware peak constants are monkeypatched to the PTL numbers the user gave
directly (58e12 FP16 FLOP/s, 110e9 byte/s); the BMG numbers (118.784e12/456e9)
are never used here.

Deliberately does NOT call ``A.audit_plan``/``A.audit_timing``: those enforce
the fixed B580 dispatch table (default WG8/scalar finalizers only, geometry
asserts, and ``A.SIZES`` capped at 65536) and would reject the fast/fast-wg16
finalizer labels and the new 131072 shape used on this run. This script
performs its own minimal, generic label bucketing (``category``) that
additionally recognizes the fast/fast-wg8/fast-wg16 Q2 finalizer labels, and
its own timing/consistency audit (``audit_timing``) that does not require the
fixed warmup=100/samples=20 protocol (65536/131072 may use fewer, and the
actual counts used are recorded per-row from the log itself, never assumed).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import analyze_long_context_roofline as A

HERE = Path(__file__).resolve().parent
LOGS = HERE / "ptl_long_context_logs"
REPORT = HERE / "PTL_LONG_CONTEXT_ROOFLINE_CN.md"

# This sweep includes all eight requested sizes.
SIZES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
PHASES = ("prefill", "decode")
CACHES = ("cold96MiB", "no-flush")
STAGES = A.STAGES  # ("Q0", "Q1-project", "Q1-prepare", "Q2-score", "Q2-topk", "Q3")

# PTL Arc B390 iGPU peaks, given directly by the user in remote_machine.txt /
# the task instructions. Do NOT substitute the BMG numbers (118.784e12/456e9).
PEAK_FLOPS = 58e12
PEAK_BW = 110e9

# Monkeypatch analyze_long_context_roofline's module-level peak constants so
# its unmodified model_work()/roofline() compute against the PTL numbers.
A.PEAK_FLOPS = PEAK_FLOPS
A.PEAK_BW = PEAK_BW


def category(label):
    """Same 6-stage bucketing as A.category, extended for fast finalizer labels."""
    if label in STAGES[:3]:
        return label
    if label in ("Q2-dpas", "Q2-partition16"):
        return "Q2-score"
    if label in ("Q2-scalar", "Q2-WG8", "Q2-WG16", "Q2-fast", "Q2-fast-WG8", "Q2-fast-WG16", "Q2-fast-WG16-cached"):
        return "Q2-topk"
    if label in ("Q3-scalar", "Q3-gqa-W4", "Q3-gqa-W4-head-major-vector-meta", "Q3-split-P16", "Q3-reduce"):
        return "Q3"
    raise ValueError(f"unsupported event label: {label}")


def finalizer_used(plan):
    """Recover the --topk-finalizer choice actually dispatched, from labels alone."""
    topk_label = plan["event_labels"][4]
    return {"Q2-scalar": "legacy(scalar)", "Q2-WG8": "legacy(cooperative-WG8)",
            "Q2-WG16": "legacy(cooperative-WG16)", "Q2-fast": "fast",
            "Q2-fast-WG8": "fast-wg8", "Q2-fast-WG16": "fast-wg16",
            "Q2-fast-WG16-cached": "fast-wg16-cached"}[topk_label]


def audit_timing(bench, plan):
    """Recompute mean/median/p95 from raw event samples and cross-check the
    saved summary statistics, WITHOUT assuming a fixed warmup/samples count.
    """
    labels = bench["event_labels"]
    A.require(labels == plan["event_labels"], "BENCH/PLAN labels differ")
    events, hosts = bench["event_samples_ms"], bench["host_samples_ms"]
    n = bench["samples"]
    A.require(len(events) == len(hosts) == n, "missing samples")
    A.require(all(len(row) == len(labels) for row in events), "ragged event samples")
    A.stats([x for row in events for x in row])
    totals = A.stats([math.fsum(row) for row in events])
    host_stats = A.stats(hosts)
    A.compare_stats(totals, bench["gpu_sum"], "gpu_sum")
    A.compare_stats(host_stats, bench["host_elapsed"], "host_elapsed")
    A.require(set(labels) == set(bench["stages"]), "stage schema mismatch")
    by_label = {}
    for label in dict.fromkeys(labels):
        idx = [i for i, name in enumerate(labels) if name == label]
        by_label[label] = A.stats([math.fsum(row[i] for i in idx) for row in events])
        A.compare_stats(by_label[label], bench["stages"][label], label)
    grouped = {}
    for stage in STAGES:
        idx = [i for i, name in enumerate(labels) if category(name) == stage]
        A.require(bool(idx), f"missing stage {stage}")
        grouped[stage] = A.stats([math.fsum(row[i] for i in idx) for row in events])
    q3_idx = [i for i, name in enumerate(labels) if name.startswith("Q3-")]
    q3_stats = A.stats([math.fsum(row[i] for i in q3_idx) for row in events])
    A.compare_stats(q3_stats, bench["q3_total"], "q3_total")
    A.near(math.fsum(s["mean_ms"] for s in grouped.values()), totals["mean_ms"], "stage sum")
    A.require(bench["post_benchmark_all_payloads_exact"], "post benchmark validation failed")
    return dict(gpu=totals, host=host_stats, stages=grouped, events=by_label,
                samples=n, warmup=bench["warmup"])


def self_check():
    """Quick, auditable sanity check of the peak-patch and roofline split."""
    A.require(A.PEAK_FLOPS == PEAK_FLOPS and A.PEAK_BW == PEAK_BW, "PTL peak patch failed")
    A.require(A.roofline(PEAK_FLOPS, 0)["theory_ms"] == 1000, "1 FLOP-second sanity failed")
    A.require(A.roofline(0, PEAK_BW)["theory_ms"] == 1000, "1 byte-second sanity failed")
    m = A.roofline(2 * PEAK_FLOPS, 0)
    A.require(m["compute_ms"] == 2000 and m["memory_ms"] == 0 and m["theory_ms"] == 2000,
              "compute/memory split sanity failed")
    m = A.roofline(0, 3 * PEAK_BW)
    A.require(m["compute_ms"] == 0 and m["memory_ms"] == 3000 and m["theory_ms"] == 3000,
              "compute/memory split sanity failed")
    A.require(A.roofline(PEAK_FLOPS, 2 * PEAK_BW)["theory_ms"] == 2000, "max() selection sanity failed")
    return True


def analyze(log_dir=LOGS):
    self_check()
    expected = {f"{phase}_{size}.log" for phase in PHASES for size in SIZES}
    present = {p.name for p in log_dir.glob("*.log")}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    rows, skipped = [], []
    for phase in PHASES:
        for size in SIZES:
            name = f"{phase}_{size}.log"
            path = log_dir / name
            if not path.exists():
                skipped.append(dict(phase=phase, size=size, reason="log file missing"))
                continue
            rec = A.read_records(path)
            plan, work = rec["PLAN"][0], rec["WORK"][0]
            A.require((plan["phase"], plan["size"]) == (phase, size), f"{name}: filename/shape mismatch")
            passed = rec["PASS"][0]
            if not (passed["all_guards_pass"] and passed["own_score_selection_different_rows"] == 0):
                skipped.append(dict(phase=phase, size=size, reason=f"validation FAILED: {passed}"))
                continue
            history = rec.get("HISTORY", [None])[0]
            for bench in rec["BENCH"]:
                timing = audit_timing(bench, plan)
                global_model, stages, details = A.model_work(plan, work, bench)
                row = dict(phase=phase, size=size, cache=bench["cache"],
                    finalizer=finalizer_used(plan), warmup=timing["warmup"], samples=timing["samples"],
                    work=work, event_labels=bench["event_labels"], timing=timing,
                    stages={s: A.compare(m, timing["stages"][s]["mean_ms"]) for s, m in stages.items()},
                    details=details, pass_evidence=passed, history=history)
                row["global"] = A.compare(global_model, timing["gpu"]["mean_ms"])
                rows.append(row)
    return dict(peak_flops=PEAK_FLOPS, peak_bandwidth=PEAK_BW, sizes=list(SIZES),
                rows=rows, missing_logs=missing, extra_logs=extra, skipped=skipped,
                log_count=len(present & expected))


def table(headers, body):
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    lines += ["|" + "|".join(map(str, row)) + "|" for row in body]
    return "\n".join(lines)


def shape_kernel_table(row):
    """One (phase, size, cache) shape rendered as one line per Q0..Q3 kernel stage
    (measured ms, compute-bound ms, memory-bound ms, roofline ms = max(compute, memory)),
    plus a trailing TOTAL(Q0-Q3) line summing measured and roofline ms and their ratio.
    Standard Qwen3.8-Flash-Next geometry only (this repo's config.json); only phase and
    size vary between shapes.
    """
    body = []
    total_measured = total_roofline = 0.0
    for stage in STAGES:
        s = row["stages"][stage]
        total_measured += s["measured_ms"]
        total_roofline += s["theory_ms"]
        body.append([row["size"], stage, f'{s["measured_ms"]:.6f}', f'{s["compute_ms"]:.6f}',
                     f'{s["memory_ms"]:.6f}', f'{s["theory_ms"]:.6f}',
                     "\u2014" if s["gap_times"] is None else f'{s["gap_times"]:.2f}'])
    ratio = total_measured / total_roofline if total_roofline else float("nan")
    body.append([row["size"], "TOTAL(Q0-Q3)", f'{total_measured:.6f}', "\u2014", "\u2014",
                 f'{total_roofline:.6f}', f'{ratio:.2f}'])
    header = (f"### {row['phase']} / size={row['size']} / {row['cache']} / "
              f"finalizer={row['finalizer']}")
    return header + "\n\n" + table(["size", "kernel", "\u5b9e\u6d4b ms", "compute ms", "memory ms",
                                    "roofline ms", "gap\u00d7"], body)


INTRO = f"""# PTL Arc B390 iGPU 长上下文 QSA：roofline 参考（新测量，非 BMG 对照）

## 1. 范围

硬件：远程 PTL Windows 机器（`remote_machine.txt`），Intel Arc(TM) B390 iGPU，
Xe3，96 EUs @2400MHz，128KiB SLM，[DPAS][sbarrier]，与 BMG Arc B580 是**不同架构、
不同峰值档位**的独立一次运行；本报告**不**把绝对数值当作与历史 B580 report
受控 A/B 的对照结论，只在明确标注"separate run"处做描述性引用。

8 个尺寸 × 2 阶段（prefill: past=0,q=size；decode: past=size,q=1,batch=1），
包含本次要求的 32768 与 131072（128K）。
Pipeline：Q1 DPAS projection、dense-bypass、按形状选择的 fast/fast-wg16 topk
finalizer、prefill GQA Q3、decode scalar/split-P16 Q3 —— 与本仓库历史一致的
"optimized" 变体，本次未修改任何 kernel。

峰值：P=58e12 FP16 FLOP/s，B=110e9 byte/s，用户直接给出，不替换为 BMG 的
118.784e12/456e9，也未发明其他 PTL 数字。

## 2. 方法：与 BMG report 相同的 useful-dot 会计，仅替换峰值

FLOP/byte 公式直接复用 `analyze_long_context_roofline.py::model_work`（未修改，
只 monkeypatch 其模块级 `PEAK_FLOPS`/`PEAK_BW` 为上面两个 PTL 数字），因此与
`LONG_CONTEXT_ROOFLINE_CN.md` 第 2–3 节公式完全一致，此处不重复推导。
`Q2-fast`/`Q2-fast-WG8`/`Q2-fast-WG16` 是本文件新增的通用 label 归类
（`category()`），旧 `A.audit_plan`/`A.audit_timing` 的固定 B580 dispatch 表
（仅 WG8/scalar，`A.SIZES` 上限 65536）未被调用，因为它会拒绝 fast 系列 label
与 131072。gap×=measured_mean/Tref；eff%=100×Tref/measured_mean；两者都只是与
指定参考的接近程度，不是硬件利用率，也不是可实现加速比。

每个 stage 显式给出 t_compute_ms=1000×FLOPs/P 与 t_memory_ms=1000×Bytes/B 两者，
再取 roofline_ms=max(t_compute_ms,t_memory_ms)（下表 "compute ms"/"memory ms"/"ref ms"
三列），不是只报告 max 后的单一数字。
"""


def render_report(data):
    parts = [INTRO.rstrip()]
    parts.append("## 3. 实际使用的 warmup/samples 与 finalizer（每个 shape 显式记录，不假设标准协议）")
    seen = set()
    rows_by_shape = []
    for r in data["rows"]:
        key = (r["phase"], r["size"])
        if key in seen:
            continue
        seen.add(key)
        rows_by_shape.append(r)
    parts.append(table(["phase", "size", "finalizer", "warmup", "samples", "caches run"],
        [[r["phase"], r["size"], r["finalizer"], r["warmup"], r["samples"],
          ", ".join(b["cache"] for b in data["rows"] if (b["phase"], b["size"]) == (r["phase"], r["size"]))]
         for r in rows_by_shape]))

    parts.append("## 4. 校验证据（每个 shape 的 own-score/attention 误差幅度）")
    parts.append(table(["phase", "size", "all_guards_pass", "own_score_selection_different_rows",
                        "numpy_score_max_abs", "attention_max_abs", "dense_rows", "sparse_rows"],
        [[r["phase"], r["size"], r["pass_evidence"]["all_guards_pass"],
          r["pass_evidence"]["own_score_selection_different_rows"],
          f'{r["pass_evidence"]["numpy_score_max_abs"]:.6g}',
          f'{r["pass_evidence"]["attention_max_abs"]:.6g}',
          r["pass_evidence"]["dense_rows"], r["pass_evidence"]["sparse_rows"]] for r in rows_by_shape]))

    for cache in CACHES:
        subset = [r for r in data["rows"] if r["cache"] == cache]
        if not subset:
            parts.append(f"## {cache}: 未运行（本轮跳过，见第 8 节）")
            continue
        parts.append(f"## 5.{'1' if cache == 'cold96MiB' else '2'} {cache}：全图结果 + roofline")
        for phase in PHASES:
            rows = [r for r in subset if r["phase"] == phase]
            if not rows:
                continue
            parts.append(f"### {phase} / {cache}")
            parts.append(table(["size", "compute ms", "memory ms", "ref ms", "实测 mean ms",
                                "median ms", "p95 ms", "gap×", "eff%"],
                [[r["size"], f'{r["global"]["compute_ms"]:.6f}', f'{r["global"]["memory_ms"]:.6f}',
                  f'{r["global"]["theory_ms"]:.6f}', f'{r["timing"]["gpu"]["mean_ms"]:.6f}',
                  f'{r["timing"]["gpu"]["median_ms"]:.6f}', f'{r["timing"]["gpu"]["p95_ms"]:.6f}',
                  f'{r["global"]["gap_times"]:.2f}', f'{r["global"]["eff_percent"]:.2f}'] for r in rows]))

    parts.append("## 6. 逐阶段结果（cold96MiB，compute/memory/ref 三列都显式给出）")
    cold = [r for r in data["rows"] if r["cache"] == "cold96MiB"]
    for stage in STAGES:
        parts.append(f"### {stage}")
        parts.append(table(["phase", "size", "compute ms", "memory ms", "ref ms",
                            "实测 mean ms", "gap×", "eff%"],
            [[r["phase"], r["size"], f'{r["stages"][stage]["compute_ms"]:.6f}',
              f'{r["stages"][stage]["memory_ms"]:.6f}', f'{r["stages"][stage]["theory_ms"]:.6f}',
              f'{r["stages"][stage]["measured_ms"]:.6f}',
              "—" if r["stages"][stage]["gap_times"] is None else f'{r["stages"][stage]["gap_times"]:.2f}',
              f'{r["stages"][stage]["eff_percent"]:.2f}'] for r in cold]))

    parts.append("## 7. cold vs no-flush（仅两者都跑到的 shape）")
    warm = {(r["phase"], r["size"]): r for r in data["rows"] if r["cache"] == "no-flush"}
    have_both = [r for r in cold if (r["phase"], r["size"]) in warm]
    if have_both:
        parts.append(table(["phase", "size", "cold mean ms", "no-flush mean ms", "cold/no-flush ×"],
            [[r["phase"], r["size"], f'{r["timing"]["gpu"]["mean_ms"]:.6f}',
              f'{warm[r["phase"], r["size"]]["timing"]["gpu"]["mean_ms"]:.6f}',
              f'{r["timing"]["gpu"]["mean_ms"] / warm[r["phase"], r["size"]]["timing"]["gpu"]["mean_ms"]:.3f}']
             for r in have_both]))
    else:
        parts.append("(no shape completed both cache modes)")

    parts.append("## 8. 跳过/缺失的 shape")
    if data["skipped"] or data["missing_logs"]:
        for s in data["skipped"]:
            parts.append(f'- {s["phase"]} {s["size"]}: {s["reason"]}')
        for name in data["missing_logs"]:
            if not any(name == f'{s["phase"]}_{s["size"]}.log' for s in data["skipped"]):
                parts.append(f'- {name}: log not present in {LOGS.name}/')
    else:
        parts.append("无：全部 7×2=14 个 (phase, size) 组合均有至少一个 cache 协议的通过结果。")

    parts.append("## 9. 限制")
    parts.append("- eff%/gap× 只是与本文件 FLOP/byte 参考模型的比值，不是硬件 SIMD/整数/SLM/消息吞吐利用率。\n"
                 "- 128K（131072）此前从未在任何机器测过；没有历史数字可比，本节数字是这次运行的首次记录。\n"
                 "- PTL 与 BMG 是不同代次/不同功耗档 iGPU vs dGPU；本报告不做跨机器 A/B 结论。\n"
                 "- prefill 长上下文的 Q3 union 参考沿用 BMG report 相同的可证明子集 U（见 model_work），\n"
                 "  不是实测 union，对 65536/131072 同样成立。\n"
                 "- 若 65536/131072 使用了低于 warmup=100/samples=20 的协议，已在第 3 节显式记录，\n"
                 "  不代表全部 shape 使用同一协议。\n"
                 "- prefill 131072 的正式结果使用显式 `--topk-finalizer fast-wg16`（与 65536 一致）。\n"
                 "  一次未显式指定 finalizer 的重复运行（回退到默认 legacy scalar 路径）得到\n"
                 "  cold total 6592ms（gap 41.66x）、Q2-topk 单项 3800ms，均明显更差；该次运行的\n"
                 "  原始日志归档在 `ptl_long_context_logs/prefill_131072_legacyscalar_default.log`，\n"
                 "  未计入本文件任何表格，仅作为 finalizer 选择对长上下文性能影响巨大的佐证保留。")

    parts.append("## 10. 每个 shape 的逐 kernel 明细 + 总计（cold96MiB；标准 Qwen3.8-Flash-Next 参数，"
                 "只需 phase+size 即可定位一个 shape）")
    for r in sorted((r for r in data["rows"] if r["cache"] == "cold96MiB"), key=lambda r: (r["phase"], r["size"])):
        parts.append(shape_kernel_table(r))
    return "\n\n".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", type=Path, default=LOGS)
    ap.add_argument("--json-out", type=Path, help="optional path to dump full analysis as JSON")
    ap.add_argument("--report", type=Path, default=REPORT, help="output markdown report path")
    ap.add_argument("--phase", choices=PHASES,
                    help="paired with --size: print one shape's kernel/roofline table to stdout")
    ap.add_argument("--size", type=int, choices=SIZES, help="paired with --phase")
    ap.add_argument("--cache", choices=CACHES, default="cold96MiB",
                    help="cache mode selected for the --phase/--size/--print-shapes console table(s)")
    ap.add_argument("--print-shapes", action="store_true",
                    help="print every (phase, size) shape's kernel/roofline table to stdout in one call "
                         "(same content as report section 10), plus a grand total across all shapes")
    args = ap.parse_args()
    if bool(args.phase) != bool(args.size):
        ap.error("--phase and --size must be given together")
    data = analyze(args.log_dir)
    args.report.write_text(render_report(data), encoding="utf-8")
    print(f"wrote {args.report} ({len(data['rows'])} rows, {len(data['skipped'])} skipped, "
          f"{len(data['missing_logs'])} missing logs)")
    if args.json_out:
        args.json_out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    if args.phase and args.size:
        match = [r for r in data["rows"] if r["phase"] == args.phase and r["size"] == args.size
                 and r["cache"] == args.cache]
        if not match:
            raise SystemExit(f"no row for phase={args.phase} size={args.size} cache={args.cache}")
        print()
        print(shape_kernel_table(match[0]))
    if args.print_shapes:
        subset = [r for r in data["rows"] if r["cache"] == args.cache]
        grand_measured = grand_roofline = 0.0
        for r in sorted(subset, key=lambda r: (r["phase"], r["size"])):
            print()
            print(shape_kernel_table(r))
            grand_measured += sum(r["stages"][s]["measured_ms"] for s in STAGES)
            grand_roofline += sum(r["stages"][s]["theory_ms"] for s in STAGES)
        print()
        grand_ratio = grand_measured / grand_roofline if grand_roofline else float("nan")
        print(f"=== GRAND TOTAL across {len(subset)} shapes ({args.cache}) ===")
        print(f"measured_sum_ms={grand_measured:.6f}  roofline_sum_ms={grand_roofline:.6f}  "
              f"gap={grand_ratio:.2f}")


if __name__ == "__main__":
    main()
