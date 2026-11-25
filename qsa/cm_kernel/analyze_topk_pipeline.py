"""CPU-only audit of integrated top-k A/B evidence (not a roofline model).

Recompute every mean/median/p95 and match SAMPLE to BENCH raw events, check
alternating order, before/last-sample/after payload hashes, same-score chunks,
real decode history and immutable non-finalizer sources. Prints JSON + tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def stats(a):
    a = sorted(a)
    index = (len(a) - 1) * .95
    lo = int(index)
    p95 = a[lo] + (a[min(lo + 1, len(a) - 1)] - a[lo]) * (index - lo)
    return dict(mean_ms=statistics.fmean(a), median_ms=statistics.median(a), p95_ms=p95)


def check_stats(actual, values):
    assert values and all(math.isfinite(x) and x >= 0 for x in values)
    for key, want in stats(values).items():
        assert abs(actual[key] - want) < 1e-9, (key, actual[key], want)


def audit(path, *, check_sources=False):
    tags = {tag: [] for tag in ("ENV", "PLAN", "PASS", "SAME_SCORE_CHUNK", "AB_PASS",
                               "HISTORY", "SAMPLE", "BENCH", "DONE")}
    for line in path.read_text().splitlines():
        tag, _, text = line.partition(" ")
        if tag in tags:
            tags[tag].append(json.loads(text))
    for tag in ("ENV", "PLAN", "PASS", "DONE"):
        assert len(tags[tag]) == 1, (path, tag)
    plan, reference = tags["PLAN"][0], tags["PASS"][0]
    assert plan["protocol"] == "v2: cache-matched warmups; no readbacks/reporting between timed samples"
    variants = list(plan["event_labels"])
    assert variants == ["legacy", "fast", "fast-wg16"]
    assert reference["own_score_selection_different_rows"] == 0
    assert reference["metadata_exact_rows"] == plan["q"]
    assert reference["all_guards_pass"] and reference["replay_all_payloads_exact"]
    assert plan["score_plan"]["allocation_bytes"] <= 128 * 1024 * 1024
    assert len(tags["SAME_SCORE_CHUNK"]) == len(plan["score_plan"]["chunks"])
    for chunk, checked in zip(plan["score_plan"]["chunks"], tags["SAME_SCORE_CHUNK"]):
        assert (chunk["row_begin"], chunk["row_end"]) == (checked["begin"], checked["end"])
        assert checked["variants"] == variants and checked["all_selected_counts_exact"]
    if plan["phase"] == "decode":
        history = tags["HISTORY"][0]
        assert history["tokens"] == plan["past"] and not history["included_in_timing"]
        assert history["validation"]["projection_rows"] == plan["past"]
        assert "actual optimized Q0 + DPAS Q1" in history["source"]
    payload = tags["AB_PASS"][0]["payloads"]
    for checked in tags["AB_PASS"]:
        assert checked["payloads"] == payload
        assert checked["all_payload_bits_exact"] and checked["all_guards"]
        assert checked["selected_count_exact_rows"] == plan["q"]
    caches = list(dict.fromkeys(r["cache"] for r in tags["BENCH"]))
    assert len(tags["AB_PASS"]) == len(variants) * (1 + len(caches)) + len(caches)
    events = 0
    for cache in caches:
        ordered = [r for r in tags["SAMPLE"] if r["cache"] == cache]
        assert [(r["index"], r["variant"]) for r in ordered] == [
            (i, mode) for i in range(20) for mode in (variants if i % 2 == 0 else variants[::-1])]
    for r in tags["BENCH"]:
        assert r["warmup"] == 100 and r["samples"] == 20 and r["all_payload_bits_exact"]
        assert r["event_labels"] == plan["event_labels"][r["variant"]]
        labels, samples = r["event_labels"], r["event_samples_ms"]
        assert len(samples) == 20 and all(len(e) == len(labels) for e in samples)
        assert all(math.isfinite(v) and v >= 0 for e in samples for v in e)
        actual = [s for s in tags["SAMPLE"] if (s["cache"], s["variant"]) == (r["cache"], r["variant"])]
        assert [s["event_samples_ms"] for s in actual] == samples
        assert [s["host_elapsed_ms"] for s in actual] == r["host_samples_ms"]
        check_stats(r["gpu_sum"], [sum(e) for e in samples])
        check_stats(r["host_elapsed"], r["host_samples_ms"])
        for name, measured in r["stages"].items():
            check_stats(measured, [sum(v for v, label in zip(e, labels) if label == name) for e in samples])
        for key, prefix in (("q2_score", "Q2-dpas" if r["phase"] == "prefill" else "Q2-partition"),
                            ("q2_topk", "Q2-topk-"), ("q3_total", "Q3-")):
            check_stats(r[key], [sum(v for v, label in zip(e, labels) if label.startswith(prefix)) for e in samples])
        for when in ("before-timing", "after-" + r["cache"]):
            assert len([p for p in tags["AB_PASS"] if (p["variant"], p["when"]) == (r["variant"], when)]) == 1
        if r["variant"] == "legacy":  # 20 alternating rounds end with legacy
            assert len([p for p in tags["AB_PASS"] if (p["variant"], p["when"]) ==
                        ("legacy", "last-measured-" + r["cache"])]) == 1
        events += sum(map(len, samples))
    root = Path(__file__).resolve().parent.parent
    if check_sources:
        for name, sha in tags["ENV"][0]["sources"].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    return tags, events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-sources", action="store_true", help="compare recorded source hashes to this checkout")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    rows, manifest, event_count, checks = [], {}, 0, []
    for phase, sizes in (("prefill", (8192, 16384, 32768, 65536)), ("decode", (16384, 32768, 65536))):
        for size in sizes:
            path = root / "topk_long_logs" / f"pipeline_ab_{phase}{size}.log"
            tags, events = audit(path, check_sources=args.check_sources)
            rows.extend(tags["BENCH"])
            event_count += events
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            checks.append(dict(phase=phase, size=size, validation=tags["PASS"][0], done=tags["DONE"][0]))
    print("AUDIT " + json.dumps(dict(files=manifest, bench_rows=len(rows), slices=len(rows) * 20,
        events=event_count, check_sources=args.check_sources, validation=checks)))
    print("\n|phase/size|cache|variant|GPU mean|topk mean|score mean|host mean|GPU median|GPU p95|")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f'|{r["phase"]}/{r["size"]}|{r["cache"]}|{r["variant"]}|' + "|".join(
            f'{r[key][stat]:.6f}' for key, stat in (("gpu_sum", "mean_ms"), ("q2_topk", "mean_ms"),
            ("q2_score", "mean_ms"), ("host_elapsed", "mean_ms"), ("gpu_sum", "median_ms"), ("gpu_sum", "p95_ms"))) + "|")
    print("\n|phase/size|chosen|GPU old→new|GPU speedup|topk old→new|topk speedup|")
    print("|---|---|---:|---:|---:|---:|")
    for old in rows:
        if old["variant"] != "legacy" or old["cache"] != "cold96MiB":
            continue
        chosen = "fast-wg16" if old["phase"] == "decode" or old["size"] == 65536 else "fast"
        new = next(r for r in rows if (r["phase"], r["size"], r["cache"], r["variant"]) ==
                   (old["phase"], old["size"], "cold96MiB", chosen))
        a, b, x, y = [r[k]["mean_ms"] for r, k in ((old, "gpu_sum"), (new, "gpu_sum"), (old, "q2_topk"), (new, "q2_topk"))]
        print(f'|{old["phase"]}/{old["size"]}|{chosen}|{a:.6f}→{b:.6f}|{a/b:.3f}×|{x:.6f}→{y:.6f}|{x/y:.3f}×|')


if __name__ == "__main__":
    main()