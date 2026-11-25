"""CPU-only roofline for the fresh 20260910T1614Z sweep; inputs are read-only.

Reuse the original analyzer's pure model_work/roofline functions unchanged.
Never call its legacy dispatch audit on fast labels, and never publish its
legacy radix read-count diagnostics for fast finalizers. Measured-source hashes
cover only ENV.sources; this later analysis is separately identified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import analyze_long_context_roofline as A

HERE = Path(__file__).resolve().parent
LOGS = HERE / "long_context_fast_topk_logs_rerun_20260910T1614Z"
REPORT = HERE / "LONG_CONTEXT_FAST_TOPK_ROOFLINE_CN.md"
SNAPSHOT_SHA256 = "37aee9b88d9f5d0464796c6eea1f9db0de2807d22aeb740b318154d1c74cf139"
GLOBAL_MODEL = "whole_graph_cold_external_io_reference"
STAGE_MODEL = "independent_stage_unique_io_optimistic_reference"


def load_measurement_auditor():
    spec = importlib.util.spec_from_file_location("fresh_measurement_audit", LOGS / "audit_measurements.py")
    A.require(spec is not None and spec.loader is not None, "cannot load measurement auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_measurement_auditor()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_snapshot(path, sources):
    """Validate archived bytes without extracting, rewriting or extending them."""
    A.require(sha256(path) == SNAPSHOT_SHA256, "source snapshot archive hash mismatch")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        A.require(len(members) == len(sources) and {m.name for m in members} == set(sources),
                  "snapshot member coverage mismatch")
        for member in members:
            A.require(member.isfile(), "snapshot non-file member")
            stream = archive.extractfile(member)
            A.require(stream is not None, "snapshot member unavailable")
            with stream:
                A.require(hashlib.sha256(stream.read()).hexdigest() == sources[member.name],
                          "snapshot source hash mismatch: " + member.name)


def audit_capsule(path, records, bench, log_name, digest):
    expected = dict(records, BENCH=[bench])
    A.require(json.loads(path.read_text()) == dict(schema="qsa-fast-topk-raw-protocol-v1",
              source_log=log_name, source_log_sha256=digest, records=expected),
              "protocol capsule mismatch: " + path.name)


def audited_measurements(log_dir=LOGS, performance_report=HERE / "LONG_CONTEXT_FAST_TOPK_PERFORMANCE_CN.md"):
    # M checks only recorded source names, not a new glob. New analysis files
    # are deliberately NOT added to the immutable measured-source manifest.
    data = M.audit(log_dir)
    A.require(data == json.loads((log_dir / "measurements.json").read_text()),
              "measurements.json differs from raw re-audit")
    A.require(M.tables(data) in performance_report.read_text(), "new performance report tables drift")
    audit_snapshot(log_dir / "source_snapshot.tar.gz", data["source_sha256"])
    expected = {f"{p}_{n}_{c}.json" for p in A.PHASES for n in A.SIZES for c in A.CACHES}
    paths = list((log_dir / "protocol_json").glob("*.json"))
    A.require({p.name for p in paths} == expected, "protocol capsule coverage mismatch")
    for phase in A.PHASES:
        for size in A.SIZES:
            name = f"{phase}_{size}.log"
            records = A.read_records(log_dir / name)
            for bench in records["BENCH"]:
                audit_capsule(log_dir / "protocol_json" / f"{phase}_{size}_{bench['cache']}.json",
                              records, bench, name, data["log_sha256"][name])
    return data


def reference_models(row):
    """No dispatch relabelling; discard the old, inapplicable read diagnostics."""
    global_model, stages, details = A.model_work(row["plan"], row["work"], row["bench"])
    details = {k: v for k, v in details.items() if not k.startswith("topk_logical_score_read_bytes")}
    details["topk_logical_score_read_bytes"] = None if row["work"]["sparse_rows"] else 0
    details["topk_read_model"] = "unknown_actual_scans; unique_score_once_reference_only"
    return global_model, stages, details


def historical_comparison_inputs():
    """Old timings are confined to the explicitly separate-run comparison.

    Recompute raw statistics and verify the published old mean table. Its full
    historical source validation remains audit_long_context_history.py's job.
    """
    rows, hashes = [], {}
    for phase in A.PHASES:
        for size in A.SIZES:
            name = f"{phase}_{size}.log"
            path = HERE / "long_context_logs" / name
            records = A.read_records(path)
            hashes[name] = sha256(path)
            plan, work = records["PLAN"][0], records["WORK"][0]
            for bench in records["BENCH"]:
                A.audit_plan(plan, work, bench)
                timing = A.audit_timing(bench, plan)
                model, _, _ = A.model_work(plan, work, bench)
                rows.append(dict(phase=phase, size=size, cache=bench["cache"], timing=timing,
                    stages={s: dict(measured_ms=timing["stages"][s]["mean_ms"]) for s in A.STAGES},
                    **{"global": A.compare(model, timing["gpu"]["mean_ms"])}))
    A.audit_performance_report(HERE / "LONG_CONTEXT_PERFORMANCE_CN.md", rows)
    return {(r["phase"], r["size"]): r for r in rows if r["cache"] == "cold96MiB"}, hashes


def analyze(log_dir=LOGS):
    measured = audited_measurements(log_dir)
    old, old_hashes = historical_comparison_inputs()
    rows = []
    for r in measured["rows"]:
        global_model, stages, details = reference_models(r)
        previous = old[r["phase"], r["size"]]
        # Exact float/integer equality against original WORK/PLAN, not rounded
        # reference milliseconds transcribed from Markdown.
        A.require(all(previous["global"][k] == v for k, v in global_model.items()),
                  "original global reference changed")
        rows.append(dict(phase=r["phase"], size=r["size"], cache=r["cache"],
            dispatch=r["dispatch"], work=r["work"], event_labels=r["bench"]["event_labels"],
            timing=dict(gpu=r["gpu"], host=r["host"], stages=r["stages"]),
            global_model_name=GLOBAL_MODEL, stage_model_name=STAGE_MODEL,
            stages={s: A.compare(m, r["stages"][s]["mean_ms"]) for s, m in stages.items()},
            details=details, **{"global": A.compare(global_model, r["gpu"]["mean_ms"])}))
    comparisons = []
    for r in rows:
        if r["cache"] != "cold96MiB":
            continue
        previous = old[r["phase"], r["size"]]
        before, after = previous["global"]["measured_ms"], r["global"]["measured_ms"]
        topk_old = previous["stages"]["Q2-topk"]["measured_ms"]
        topk_new = r["stages"]["Q2-topk"]["measured_ms"]
        comparisons.append(dict(phase=r["phase"], size=r["size"], old_ms=before, new_ms=after,
            old_over_new=before / after, reduction_percent=100 * (1 - after / before),
            old_gap=previous["global"]["gap_times"], new_gap=r["global"]["gap_times"],
            old_topk_ms=topk_old, new_topk_ms=topk_new, topk_old_over_new=topk_old / topk_new))
    return dict(schema="qsa-fast-topk-roofline-v1", peak_flops=A.PEAK_FLOPS, peak_bandwidth=A.PEAK_BW,
        audit=dict(measured["audit"], protocol_capsules=28, snapshot_sources_verified=67,
                   measurements_json_exact=True, original_global_references_exact=14),
        measured_source_sha256=measured["source_sha256"], log_sha256=measured["log_sha256"],
        artifact_sha256={name: sha256(log_dir / name) for name in ("measurements.json", "source_snapshot.tar.gz")},
        protocol_sha256={p.name: sha256(p) for p in sorted((log_dir / "protocol_json").glob("*.json"))},
        analysis_source_sha256={"analyze_fast_topk_roofline.py": sha256(Path(__file__)),
            "audit_measurements.py": sha256(LOGS / "audit_measurements.py")},
        historical_comparison_log_sha256=old_hashes, rows=rows, separate_run_comparison=comparisons)


INTRO = """# Fast top-k 后 QSA：新测量 roofline 与历史对照

## 1. 范围与协议

仅将 `long_context_fast_topk_logs_rerun_20260910T1614Z/` 的新测量用于当前结果：
2026-09-10 16:14:30–16:18:38 UTC，B580，14 shapes × cold96MiB/no-flush，
每组100 warmup/20 samples，合计560 slices、6960 events。未重新运行GPU。
[新性能报告](LONG_CONTEXT_FAST_TOPK_PERFORMANCE_CN.md) 保留完整六阶段/host mean、median、p95与验证细节。
输入为14份原log；`measurements.json` 与28份 `protocol_json/` 逐项核对，不是额外样本。

每个sample先求所有GPU event duration之和，再计算mean/median/线性p95；不相加stage分位数。
Q2合并全部chunk，Q3合并split+reduce；排除flush、HISTORY、JIT、校验、读回与event间隙。
顺序为prefill升序→decode升序，每形状cold→no-flush；两协议各自100次不flush预热。
cold正式样本前flush96MiB，不在阶段之间flush；不是A/B/C v2的cache-matched warmup。
no-flush不保证驻留cache，保留全部样本与cold波动，不用其他run或最小值替换。

显式策略（默认dispatcher未改变）：prefill1K/2K=`legacy` WG8；4K/8K/16K/32K=`fast` LWS1 CAP256；
64K=`fast-wg16`。decode1K/2K=`legacy` WG16；≥4K=`fast-wg16`。
Q1 DPAS；prefill score DPAS、decode partition16。Q3 prefill GQA-W4；decode1K scalar、≥2K split-P16+reduce。
因此decode1K→2K有既有Q3路径切换，不能解释成同一kernel的长度曲线。

## 2. 原模型保持不变

直接调用 [原分析器](analyze_long_context_roofline.py) 的纯函数 `model_work/roofline`，
完整公式及省略项见 [原roofline第2–3节](LONG_CONTEXT_ROOFLINE_CN.md)。不调用旧dispatch审计处理fast标签。
对14形状逐项验证新旧global模型全精度完全相同；不用已四舍五入的理论ms重新计算。

P=118.784 TFLOP/s、B=456 GB/s（十进制），FP16=2B、FP32/int32=4B、FMA=2 FLOP。
设备报告最高2900MHz，不是本轮实际运行频率；未记录逐sample频率/功率，不推测DVFS或cache原因。
XMX峰值为理想容量，decode SIMD/scalar不具有此吞吐。

令n(t)=floor((past+t+1)/4)，k(t)=min(n(t),512)，C=sum[n(t)>512]n(t)，
A=sum(4k(t)+(past+t+1)%4)，S为新summary数：F=2q×2560×640+1024C+24576A。
只计useful dot，不含padding、softmax/exp、gate、RMSNorm、RoPE、整数选择或地址/同步成本。
Q0/Q1-post/top-k的dot FLOP=0不表示没有计算。

主模型标签 `whole_graph_cold_external_io_reference`：Tcompute=1000F/P；
Tref=max(Tcompute,1000Bext/B)。Bext=3276800+5120q+2048q+24576q+12288q+
2048q+256q+256S+R+Hkv+Hsummary；各项分别为weight、mixed、当前KV输入、query/gate、output、
main/raw/summary cache写、norm/RoPE与历史输入。
R=256(1+[S>0])+64×|{past+t+3}∪{past+4j:0≤j<S}|；
decode Hkv=2048×4×sum k，Hsummary=256floor(past/4)仅C>0；prefill两项均0。
当前KV不为Q3重复计入外部读；sparse decode为2049 attended位置，不是past+1。
中间scratch/selection不计全图外部量。冷外部写是约定，不保证event结束已回写DRAM。

**gap×=measured_mean/Tref；ref_as_percent（eff）=100Tref/measured_mean。不是硬件利用率或可达加速比。**
no-flush使用同一外部参考仅便于对照；不是每次都从DRAM读权重/历史的声明。

逐阶段模型标签 `independent_stage_unique_io_optimistic_reference`：复用原第3.2节，
producer写/consumer读中间张量各一次，理想cache/GQA/跨query复用；不是严格DRAM下界，不与全图相加。
特别Q2-topk只用4C+4q(512+1)的一次unique-score读/selection写参考。
fast/fast-wg16的实际整数扫描、candidate overflow/fallback、SLM/barrier和消息成本未计量；
**丢弃旧model_work的4n/6n radix读取诊断，fast实际读取字节标未知，绝不把一次参考称为实际单遍算法。**
没有整数/SFU/SLM硬件峰值，不报告这些阶段的硬件利用率。
Q3 prefill唯一KV位置仅为原模型可证明子集U（≤2K为q，否则3q/4+512），不是实测union；
decode U精确。split payload Z=396288B，producer/reducer各自max后相加。
score store按真实chunk/tile重建；allocation/guard不是DRAM流量。
"""


def render_report(data):
    parts = [INTRO.rstrip(), "## 3. 全部新测量：cold与no-flush\n\n时间单位ms。理论/ratio用全精度计算，仅显示时舍入。"]
    rows = data["rows"]
    for cache in A.CACHES:
        parts.append(f"### {cache}")
        parts.append(A.table(["phase", "size", "compute-only ms", "ref ms", "mean ms", "median ms", "p95 ms", "gap×", "ref_as_percent %"],
            [[r["phase"], r["size"], f'{r["global"]["compute_ms"]:.6f}', f'{r["global"]["theory_ms"]:.6f}',
              *[f'{r["timing"]["gpu"][k]:.6f}' for k in ("mean_ms", "median_ms", "p95_ms")],
              f'{r["global"]["gap_times"]:.2f}', f'{r["global"]["eff_percent"]:.2f}']
             for r in rows if r["cache"] == cache]))
    parts.append("## 4. 六阶段诊断（cold）\n\n每格为 measured mean ms / optimistic ref ms / gap×；不是硬件利用率。dense Q2-score参考0，gap为—，空dispatch仍计时。JSON包含两协议所有六阶段mean/median/p95、F/B、compute/memory参考和ratio。")
    def stage_cell(m):
        gap = "—" if m["gap_times"] is None else f'{m["gap_times"]:.2f}'
        return f'{m["measured_ms"]:.6f}/{m["theory_ms"]:.6f}/{gap}'
    cold = [r for r in rows if r["cache"] == "cold96MiB"]
    parts.append(A.table(["phase", "size", *A.STAGES],
        [[r["phase"], r["size"], *[stage_cell(r["stages"][s]) for s in A.STAGES]] for r in cold]))
    parts.append("## 5. 历史cold→新cold：SEPARATE RUN，非受控A/B\n\n旧数值来自 [原性能报告](LONG_CONTEXT_PERFORMANCE_CN.md) 第4节对应raw events（全精度重算并验证表格）。\n旧prefill≥4K为scalar finalizer、decode为旧WG16。不同run、不同策略，时钟/cache状态未受控且无实际频率遥测；\n同名cold协议也不能消除跨运行差异。旧/新比值仅为描述性对照，不能把全部变化因果归给top-k。")
    comparisons = data["separate_run_comparison"]
    parts.append(A.table(["phase", "size", "old ms", "new ms", "old/new×", "降低%", "old gap×", "new gap×", "old topk ms", "new topk ms", "topk old/new×"],
        [[c["phase"], c["size"], f'{c["old_ms"]:.6f}', f'{c["new_ms"]:.6f}', f'{c["old_over_new"]:.3f}',
          f'{c["reduction_percent"]:.2f}', f'{c["old_gap"]:.2f}', f'{c["new_gap"]:.2f}',
          f'{c["old_topk_ms"]:.6f}', f'{c["new_topk_ms"]:.6f}', f'{c["topk_old_over_new"]:.2f}'] for c in comparisons]))
    parts.append("## 6. 结论与波动")
    for phase in A.PHASES:
        r = next(r for r in cold if r["phase"] == phase and r["size"] == 65536)
        c = next(c for c in comparisons if c["phase"] == phase and c["size"] == 65536)
        m = r["global"]
        parts.append(f'- 64K {phase}：新cold {m["measured_ms"]:.6f} ms，参考 {m["theory_ms"]:.6f} ms，'
                     f'gap {m["gap_times"]:.2f}×，ref {m["eff_percent"]:.2f}%；跨run总时间比 {c["old_over_new"]:.3f}×，'
                     f'top-k时间比 {c["topk_old_over_new"]:.2f}×。新Q3占 {100*r["stages"]["Q3"]["measured_ms"]/m["measured_ms"]:.2f}%，'
                     f'top-k占 {100*r["stages"]["Q2-topk"]["measured_ms"]/m["measured_ms"]:.2f}%。')
    parts.append("- prefill长序列top-k时间大幅下降，64K主要成本转向Q3；没有据此预测尚未测量的进一步加速。\n"
                 "- 新cold decode16K **0.179940**、64K **0.240304** ms高于先前受控A/B/C中的fast-WG16 **0.105866/0.156919** ms。"
                 "后者仅作 [不同实验的背景链接](TOPK_LONG_OPTIMIZATION_CN.md#9-完整-qsa-slice-同输入-abc最终协议-v2)，不进入本轮结果或ratio。"
                 "本轮对应no-flush为 **0.088321/0.135106** ms；保留新cold全部20样本，不能据此发明DVFS、cache或fallback原因。\n"
                 "- decode2K也保留跨run变慢的结果；1K/2K仍为legacy，不声称短形状收益。无跨协议顺序平衡，20样本p95不是长期尾延迟保证。\n"
                 "- 全行own-score selection通过；64K prefill NumPy Q3仅104行、score98行，full output有限性覆盖402653184元素。"
                 "不是全量NumPy attention、模型质量或完整模型token/s测量。")
    parts.append("## 7. 离线复现与来源隔离\n\n"
                 "在 `cm_kernel/` 执行 `python3 analyze_fast_topk_roofline.py --check-report LONG_CONTEXT_FAST_TOPK_ROOFLINE_CN.md` "
                 "逐字核对本文；`--format json`输出全精度结果及来源哈希，`--logs`可指定本轮证据副本。"
                 "`python3 -m unittest test_analyze_fast_topk_roofline -v`运行新CPU回归。无GPU、NumPy或新环境变量要求。\n\n"
                 "同时运行 `python3 long_context_fast_topk_logs_rerun_20260910T1614Z/audit_measurements.py --format tables "
                 "--check-report LONG_CONTEXT_FAST_TOPK_PERFORMANCE_CN.md`；原12项回归通过 "
                 "`python3 audit_long_context_history.py --test`在临时目录恢复精确历史源码，不覆盖工作区。\n\n"
                 "验证14 logs / 28 BENCH / 560 slices / 6960 events；全部saved GPU/stage/host mean、median、p95重算；"
                 "`measurements.json`全结构相等、28 protocol capsules与原log相等、67份测量源码与ENV/归档原始字节一致。"
                 "旧报告28行mean亦复核，新旧14个global模型全精度相等。\n\n"
                 "`ENV.sources`只绑定测量时67个文件；新分析器/测试不冒充当时已存在，也不扩写manifest。"
                 "JSON中 `measured_source_sha256` 与 `analysis_source_sha256`分开；后者为后处理来源，不是新测量。"
                 "归档tar仅只读校验，不提取覆盖文件，不改历史analyzer/report/log或任何原始证据。")
    parts.append(A.table(["本轮证据", "SHA256"], [*data["artifact_sha256"].items(), *data["log_sha256"].items()]))
    parts.append("关联：[新性能报告](LONG_CONTEXT_FAST_TOPK_PERFORMANCE_CN.md) · [原模型](LONG_CONTEXT_ROOFLINE_CN.md) · "
                 "[新分析器](analyze_fast_topk_roofline.py) · [CPU测试](test_analyze_fast_topk_roofline.py) · [README](README.md)")
    return "\n\n".join(parts) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=LOGS)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--check-report", type=Path)
    args = parser.parse_args(argv)
    data = analyze(args.logs)
    report = render_report(data)
    if args.check_report:
        A.require(args.check_report.read_text() == report, "generated report differs from saved report")
        print("PASS: fresh roofline report matches 28 measurements, 560 samples, 6960 events", file=sys.stderr)
    print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) if args.format == "json" else report,
          end="\n" if args.format == "json" else "")


if __name__ == "__main__":
    main()