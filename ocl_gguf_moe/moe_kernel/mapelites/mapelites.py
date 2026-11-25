#!/usr/bin/env python3
"""MAP-Elites archive driver for the MoE 3-GEMM SwiGLU kernel (faithful to
dev_kf_distill_master_opt): correctness-gated speedup fitness, deterministic code->cell
descriptor via the skill's optimization_classifier, one elite per cell.

Cell = (memory_opt, compute_opt, parallelism_opt), each 0..3, classified from the EVOLVE region
of the candidate .cl (the DP4A decode/GEMV block between EVOLVE_START/EVOLVE_END).

Fitness (EvalResult.compute_performance_score):
    perf_score = 5 if all cases correct else {0 extract,1 build,2 runtime,3 shape,4 value}
    fitness    = perf_score + gmean(baseline_median / cand_median)   (only when perf_score==5)

Usage:
    python3 mapelites.py baseline                 # measure the pristine baseline kernel
    python3 mapelites.py add <cand.cl> <label>    # evaluate a candidate, insert into archive
    python3 mapelites.py show                      # print the archive grid + best
"""
import sys, os, json, math, shutil, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                                  # moe_kernel/
KERNEL = os.path.join(PKG, "kernels", "moe_3gemm_swiglu_mlp.cl")
BASELINE = os.path.join(PKG, "kernels", "moe_3gemm_swiglu_mlp.baseline.cl")
ARCHIVE = os.path.join(HERE, "archive.json")
VARIANTS = os.path.join(HERE, "variants")
CFG = "qwen36.json"

# --- import the skill's deterministic classifier ---
CLS_PATH = "/mnt/river/kernel_foundry_master/skills/dev_kf_distill_master_opt/scripts/optimization_classifier.py"
_spec = importlib.util.spec_from_file_location("optimization_classifier", CLS_PATH)
_cls = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cls)

import remote_run as rr


def evolve_region(code):
    a = code.find("// [EVOLVE" + "_START]")
    b = code.find("// [EVOLVE" + "_END]")
    return code[a:b] if (a >= 0 and b > a) else code


def classify(code):
    return tuple(_cls.classify(evolve_region(code), backend="opencl", include_esimd=False))


def parse_report(report):
    """-> dict case_key -> {median_ms, passed}, and all_passed bool."""
    cases = {}
    for r in report.get("records", []):
        p = r["params"]
        key = "%s/%s H%d I%d E%d" % (r["kernel"], r["format"], p["H"], p["I"], p["E"])
        cases[key] = {"median_ms": r["timing"]["median_ms"],
                      "passed": bool(r["correctness"].get("passed"))}
    all_pass = len(cases) > 0 and all(v["passed"] for v in cases.values())
    return cases, all_pass


def load_archive():
    if os.path.exists(ARCHIVE):
        with open(ARCHIVE) as f:
            return json.load(f)
    return {"baseline": None, "cells": {}}


def save_archive(a):
    with open(ARCHIVE, "w") as f:
        json.dump(a, f, indent=2)


def measure(label):
    """Run the CURRENT kernels/*.cl on the remote, return (cases, all_pass, rc, out, err)."""
    rc, out, err, report = rr.bench(CFG)
    if report is None:
        print(out[-2000:]); print("STDERR:", err[-2000:])
        return None, False, rc, out, err
    cases, all_pass = parse_report(report)
    return cases, all_pass, rc, out, err


def fitness(cases, all_pass, base_cases):
    if not all_pass:
        return 4.0, None            # value/correctness gate not passed -> perf_score 4
    ratios = []
    for k, v in cases.items():
        if k in base_cases and v["median_ms"] > 0:
            ratios.append(base_cases[k]["median_ms"] / v["median_ms"])
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else 0.0
    return 5.0 + gm, gm


def cmd_baseline():
    shutil.copyfile(BASELINE, KERNEL)
    print("Deployed pristine baseline. Measuring on remote (Arc B390)...")
    cases, all_pass, rc, out, err = measure("baseline")
    if cases is None:
        print("BASELINE MEASURE FAILED (rc=%d)" % rc); return
    a = load_archive()
    a["baseline"] = {"cases": cases, "all_pass": all_pass}
    save_archive(a)
    for k, v in cases.items():
        print("  %-34s %.4f ms  %s" % (k, v["median_ms"], "PASS" if v["passed"] else "FAIL"))
    print("baseline all_pass:", all_pass)


def cmd_add(cand_path, label):
    a = load_archive()
    if not a.get("baseline"):
        print("Run `baseline` first."); return
    base_cases = a["baseline"]["cases"]
    os.makedirs(VARIANTS, exist_ok=True)
    saved = os.path.join(VARIANTS, label + ".cl")
    if os.path.abspath(cand_path) != os.path.abspath(saved):
        shutil.copyfile(cand_path, saved)
    if os.path.abspath(cand_path) != os.path.abspath(KERNEL):
        shutil.copyfile(cand_path, KERNEL)                   # deploy candidate
    code = open(cand_path).read()
    coords = classify(code)
    print("[%s] cell=%s  measuring on remote..." % (label, coords))
    cases, all_pass, rc, out, err = measure(label)
    if cases is None:
        print("MEASURE FAILED rc=%d (compile/runtime error). perf_score<5." % rc)
        # still record as a failed attempt (not an elite)
        return
    fit, gm = fitness(cases, all_pass, base_cases)
    print("  correctness all_pass=%s  gmean_speedup=%s  fitness=%.4f"
          % (all_pass, ("%.4f" % gm) if gm is not None else "n/a", fit))
    for k, v in cases.items():
        b = base_cases.get(k, {}).get("median_ms", 0)
        sp = (b / v["median_ms"]) if v["median_ms"] else 0
        print("    %-34s %.4f ms  (%.3fx)  %s" % (k, v["median_ms"], sp, "PASS" if v["passed"] else "FAIL"))
    key = "%d,%d,%d" % coords
    cells = a["cells"]
    cur = cells.get(key)
    if all_pass and (cur is None or fit > cur["fitness"]):
        cells[key] = {"label": label, "coords": list(coords), "fitness": fit,
                      "gmean_speedup": gm, "cases": cases, "variant": saved}
        save_archive(a)
        print("  -> NEW ELITE in cell %s (fitness %.4f)" % (key, fit))
    else:
        if cur is not None:
            print("  -> cell %s keeps %s (fitness %.4f >= %.4f)" % (key, cur["label"], cur["fitness"], fit))
        else:
            print("  -> not inserted (failed correctness gate)")


def cmd_show():
    a = load_archive()
    if a.get("baseline"):
        print("BASELINE:")
        for k, v in a["baseline"]["cases"].items():
            print("  %-34s %.4f ms" % (k, v["median_ms"]))
    print("\nARCHIVE cells (memory,compute,parallelism):")
    best = None
    for key, c in sorted(a.get("cells", {}).items()):
        print("  [%s] %-22s fitness=%.4f  speedup=%.4f" %
              (key, c["label"], c["fitness"], c.get("gmean_speedup") or 0))
        if best is None or c["fitness"] > best["fitness"]:
            best = c
    if best:
        print("\nBEST: %s cell=%s speedup=%.4fx" %
              (best["label"], best["coords"], best.get("gmean_speedup") or 0))


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    if cmd == "baseline":
        cmd_baseline()
    elif cmd == "add":
        cmd_add(sys.argv[2], sys.argv[3])
    elif cmd == "show":
        cmd_show()
    else:
        print("unknown cmd", cmd)


if __name__ == "__main__":
    main()
