"""CPU-only fresh-data, exact-reference and tamper-rejection regressions."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_fast_topk_roofline as F


class FastRooflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = F.analyze()

    def row(self, phase, size, cache="cold96MiB"):
        return next(r for r in self.data["rows"] if (r["phase"], r["size"], r["cache"]) == (phase, size, cache))

    def test_complete_new_coverage_and_provenance(self):
        a = self.data["audit"]
        self.assertEqual((a["logs"], a["benchmarks"], a["slices"], a["events"]), (14, 28, 560, 6960))
        self.assertEqual((len(self.data["protocol_sha256"]), len(self.data["measured_source_sha256"])), (28, 67))
        self.assertEqual(self.data["artifact_sha256"]["source_snapshot.tar.gz"], F.SNAPSHOT_SHA256)
        self.assertNotIn("cm_kernel/analyze_fast_topk_roofline.py", self.data["measured_source_sha256"])
        self.assertIn("analyze_fast_topk_roofline.py", self.data["analysis_source_sha256"])

    def test_exact_original_reference_all_14_not_rounded_inputs(self):
        # Independent integer WORK from LONG_CONTEXT_PERFORMANCE_CN.md §9;
        # exact external bytes from the original reference, not rounded ms.
        candidates = (0, 0, 1570816, 7861248, 33025024, 133684224, 536329216)
        attended = (524800, 2098176, 6295552, 14690304, 31479808, 65058816, 132216832)
        external = {
            "prefill": (50856512, 98435648, 193593920, 383910464, 764543552, 1525809728, 3048342080),
            "decode": (5420608, 7517760, 7779904, 8042048, 8566336, 9614912, 11712064),
        }
        published = {
            "prefill": (.136828, .490602, 1.429063, 3.333129, 7.249735, 15.516840, 33.786624),
            "decode": (.011887, .016486, .017061, .017636, .018786, .021085, .025684),
        }
        for phase in F.A.PHASES:
            for i, size in enumerate(F.A.SIZES):
                q = size if phase == "prefill" else 1
                c = candidates[i] if phase == "prefill" else (size // 4 if size > 2048 else 0)
                a = attended[i] if phase == "prefill" else (1025 if size == 1024 else 2049)
                flops = 3276800 * q + 1024 * c + 24576 * a
                compute = flops / 118.784e12 * 1000
                expected = max(compute, external[phase][i] / 456e9 * 1000)
                original = F.A.read_records(F.HERE / "long_context_logs" / f"{phase}_{size}.log")
                old_model, _, _ = F.A.model_work(original["PLAN"][0], original["WORK"][0], original["BENCH"][0])
                for cache in F.A.CACHES:
                    with self.subTest(phase=phase, size=size, cache=cache):
                        model = self.row(phase, size, cache)["global"]
                        self.assertEqual(model["useful_dot_flops"], flops)
                        self.assertEqual(model["reference_bytes"], external[phase][i])
                        self.assertEqual(model["compute_ms"], compute)
                        self.assertEqual(model["theory_ms"], expected)
                        self.assertEqual({k: model[k] for k in old_model}, old_model)
                        self.assertEqual(round(expected, 6), published[phase][i])
                        # 64K prefill happens to be exactly representable at
                        # the published precision; all other shapes are not.
                        if (phase, size) != ("prefill", 65536):
                            self.assertNotEqual(expected, published[phase][i])

    def test_fast_scan_counts_are_unknown_not_legacy_radix(self):
        for row in self.data["rows"]:
            d, w = row["details"], row["work"]
            self.assertNotIn("topk_logical_score_read_bytes_min", d)
            self.assertNotIn("topk_logical_score_read_bytes_max", d)
            self.assertEqual(d["topk_logical_score_read_bytes"], None if w["sparse_rows"] else 0)
            self.assertEqual(row["stages"]["Q2-topk"]["reference_bytes"],
                             4 * w["sparse_candidates_sum"] + 4 * (row["size"] if row["phase"] == "prefill" else 1) * 513)
            self.assertEqual(row["global_model_name"], F.GLOBAL_MODEL)

    def test_dispatch_q3_discontinuity_and_scratch(self):
        for phase in F.A.PHASES:
            for size in F.A.SIZES:
                row = self.row(phase, size)
                self.assertEqual(row["dispatch"], F.M.policy(phase, size))
        self.assertIn("Q3-scalar", self.row("decode", 1024)["event_labels"])
        self.assertIn("Q3-split-P16", self.row("decode", 2048)["event_labels"])
        d = self.row("prefill", 65536)["details"]
        self.assertEqual(d["score_allocation_bytes"], 134152448)
        self.assertGreater(d["score_store_bytes"], 10 * d["score_allocation_bytes"])
        self.assertEqual(len(self.row("prefill", 65536)["event_labels"]), 70)

    def test_decode_2049_positions_and_split_reference(self):
        row = self.row("decode", 65536)
        self.assertEqual(row["work"]["attended_positions_sum"], 2049)
        self.assertEqual(row["global"]["external_components_bytes"]["history_main_kv"], 4194304)
        self.assertEqual(row["details"]["split_payload_bytes"], 396288)
        q3 = row["stages"]["Q3"]
        self.assertEqual(q3["theory_ms"], sum(m["theory_ms"] for m in q3["substage_models"].values()))

    def test_all_new_raw_samples_and_ratios_preserved(self):
        for row in self.data["rows"]:
            records = F.A.read_records(F.LOGS / f'{row["phase"]}_{row["size"]}.log')
            bench = next(b for b in records["BENCH"] if b["cache"] == row["cache"])
            self.assertEqual(row["timing"]["gpu"], F.A.stats([math.fsum(e) for e in bench["event_samples_ms"]]))
            m = row["global"]
            self.assertEqual(m["gap_times"], m["measured_ms"] / m["theory_ms"])
            self.assertAlmostEqual(m["eff_percent"] * m["gap_times"], 100)
        for size, expected in ((16384, .179940), (65536, .240304)):
            self.assertEqual(round(self.row("decode", size)["global"]["measured_ms"], 6), expected)
        row = self.row("decode", 16384)
        self.assertNotAlmostEqual(row["timing"]["gpu"]["p95_ms"],
                                 sum(s["p95_ms"] for s in row["timing"]["stages"].values()), places=6)

    def test_separate_run_comparison_includes_regression(self):
        comparison = self.data["separate_run_comparison"]
        self.assertEqual(len(comparison), 14)
        for c in comparison:
            self.assertEqual(c["new_ms"], self.row(c["phase"], c["size"])["global"]["measured_ms"])
            self.assertEqual(c["old_over_new"], c["old_ms"] / c["new_ms"])
        short = next(c for c in comparison if (c["phase"], c["size"]) == ("decode", 2048))
        self.assertLess(short["reduction_percent"], 0)

    def test_raw_tampering_rejected(self):
        original = F.A.read_records(F.LOGS / "decode_65536.log")
        mutations = (
            lambda r: r["BENCH"][0]["event_samples_ms"][0].__setitem__(0, -1),
            lambda r: r["BENCH"][0]["event_samples_ms"][0].__setitem__(0, float("nan")),
            lambda r: r["PLAN"][0]["event_labels"].__setitem__(4, "Q2-WG16"),
            lambda r: r["ENV"][0]["sources"].__setitem__("cm_kernel/benchmark_long_context.py", "0" * 64),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = deepcopy(original)
                mutation(bad)
                with self.assertRaises(ValueError):
                    F.M.audit_record(bad, "decode", 65536)

    def test_capsule_tampering_rejected(self):
        name = "decode_65536.log"
        original = F.A.read_records(F.LOGS / name)
        capsule = F.LOGS / "protocol_json/decode_65536_cold96MiB.json"
        content = json.loads(capsule.read_text())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "bad.json"
            for key, value in (("source_log_sha256", "0" * 64), ("source_log", "prefill_65536.log"), ("records", {})):
                target.write_text(json.dumps(dict(content, **{key: value})))
                with self.assertRaisesRegex(ValueError, "capsule mismatch"):
                    F.audit_capsule(target, original, original["BENCH"][0], name, self.data["log_sha256"][name])

    def test_measurement_summary_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "measurements.json").write_text("{}")
            with patch.object(F.M, "audit", return_value={"schema": "actual"}):
                with self.assertRaisesRegex(ValueError, "measurements.json differs"):
                    F.audited_measurements(root)

    def test_archive_and_member_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.tar.gz"
            archive.write_bytes(b"not the archived source bytes")
            with self.assertRaisesRegex(ValueError, "archive hash mismatch"):
                F.audit_snapshot(archive, self.data["measured_source_sha256"])
        wrong = dict(self.data["measured_source_sha256"])
        wrong[next(iter(wrong))] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            F.audit_snapshot(F.LOGS / "source_snapshot.tar.gz", wrong)

    def test_json_and_exact_saved_report(self):
        self.assertEqual(len(json.loads(json.dumps(self.data, allow_nan=False))["rows"]), 28)
        report = F.render_report(self.data)
        self.assertEqual(report, F.render_report(deepcopy(self.data)))
        self.assertEqual(report, F.REPORT.read_text())
        self.assertIn("SEPARATE RUN", report)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.md"
            path.write_text(report.replace("358.769638", "358.000000"))
            with patch.object(F, "analyze", return_value=self.data):
                with self.assertRaisesRegex(ValueError, "report differs"):
                    F.main(["--check-report", str(path)])


if __name__ == "__main__":
    unittest.main()