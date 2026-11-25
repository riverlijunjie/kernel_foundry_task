"""CPU-only stdlib regression tests; no CM, GPU or NumPy imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import analyze_long_context_roofline as A


class RooflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = A.analyze(A.HERE / "long_context_logs", A.HERE.parent,
                             A.HERE / "LONG_CONTEXT_PERFORMANCE_CN.md")

    def row(self, phase, size, cache="cold96MiB"):
        return next(r for r in self.data["rows"]
                    if (r["phase"], r["size"], r["cache"]) == (phase, size, cache))

    def test_full_coverage_and_units(self):
        self.assertEqual(self.data["audit"]["events"], 6960)
        self.assertEqual(len(self.data["rows"]), 28)
        self.assertEqual(len(self.data["log_sha256"]), 14)
        self.assertEqual(A.roofline(A.PEAK_FLOPS, 0)["theory_ms"], 1000)
        self.assertEqual(A.roofline(0, A.PEAK_BW)["theory_ms"], 1000)
        for row in self.data["rows"]:
            model = row["global"]
            self.assertAlmostEqual(model["gap_times"] * model["eff_percent"], 100)
            self.assertEqual(model["reference_bytes"], sum(model["external_components_bytes"].values()))
            self.assertEqual(model["useful_dot_flops"],
                             sum(s["useful_dot_flops"] for s in row["stages"].values()))

    def test_linear_percentile_not_stage_percentiles(self):
        self.assertAlmostEqual(A.stats(list(range(20)))["p95_ms"], 18.05)
        row = self.row("decode", 16384)
        summed = sum(s["p95_ms"] for s in row["timing"]["stages"].values())
        self.assertNotAlmostEqual(summed, row["timing"]["gpu"]["p95_ms"], places=6)

    def test_causal_dense_boundary(self):
        last = A.work_counts(2050, 1)
        first = A.work_counts(2051, 1)
        self.assertEqual((last["max_complete"], last["sparse_rows"], last["attended_positions_sum"]),
                         (512, 0, 2051))
        self.assertEqual((first["max_complete"], first["sparse_rows"], first["attended_positions_sum"]),
                         (513, 1, 2048))
        self.assertEqual(A.work_counts(0, 65536)["dense_rows"], 2051)

    def test_dense_bypass_has_no_score_work_or_reads(self):
        for phase in A.PHASES:
            for size in (1024, 2048):
                row = self.row(phase, size)
                score = row["stages"]["Q2-score"]
                self.assertEqual(score["useful_dot_flops"], 0)
                self.assertEqual(score["reference_bytes"], 0)
                self.assertIsNone(score["gap_times"])
                self.assertGreater(score["measured_ms"], 0)
                self.assertEqual(row["details"]["topk_logical_score_read_bytes_max"], 0)
                self.assertGreater(row["stages"]["Q2-topk"]["reference_bytes"], 0)

    def test_decode_unique_kv_not_context(self):
        row = self.row("decode", 65536)
        self.assertEqual(row["work"]["attended_positions_sum"], 2049)
        self.assertEqual(row["details"]["q3_unique_positions_min"], 2049)
        self.assertTrue(row["details"]["q3_unique_positions_exact"])
        external = row["global"]["external_components_bytes"]
        self.assertEqual(external["history_main_kv"], 2048 * 2048)
        self.assertEqual(external["history_summary"], 16384 * 256)
        self.assertEqual(external["new_main_kv_input"], 2048)
        self.assertEqual(external["norm_and_rotary"], 256 + 64)
        self.assertEqual(row["global"]["cold_history_plus_weight_bytes"], 11665408)

    def test_prefill_union_is_range_and_does_not_change_compute_dominance(self):
        for size in A.SIZES:
            row = self.row("prefill", size)
            d, q3 = row["details"], row["stages"]["Q3"]
            self.assertEqual(d["q3_unique_positions_max"], size)
            self.assertEqual(d["q3_unique_positions_min"], size if size <= 2048 else 3 * size // 4 + 512)
            extra = 2048 * (size - d["q3_unique_positions_min"])
            self.assertEqual(A.roofline(q3["useful_dot_flops"], q3["reference_bytes"] + extra)["theory_ms"],
                             q3["theory_ms"])
            self.assertEqual(row["global"]["external_components_bytes"]["history_main_kv"], 0)

    def test_scratch_is_not_traffic_and_radix_is_not_one_scan(self):
        pf, dec = self.row("prefill", 65536), self.row("decode", 65536)
        details = pf["details"]
        self.assertEqual(details["score_allocation_bytes"], 134152448)
        self.assertGreater(details["score_store_bytes"], details["useful_score_bytes"])
        self.assertGreater(details["score_store_bytes"], 10 * details["score_allocation_bytes"])
        self.assertLess(details["topk_logical_score_read_bytes_min"], details["topk_logical_score_read_bytes_max"])
        self.assertEqual(dec["details"]["topk_logical_score_read_bytes_min"], 6 * 16384 * 4)
        self.assertEqual(dec["details"]["topk_logical_score_read_bytes_max"], 6 * 16384 * 4)
        self.assertEqual(dec["details"]["split_payload_bytes"], 396288)
        substages = dec["stages"]["Q3"]["substage_models"]
        self.assertAlmostEqual(dec["stages"]["Q3"]["theory_ms"],
                               sum(m["theory_ms"] for m in substages.values()))

    def test_preserved_cache_protocol_and_means(self):
        pf, dec = self.row("prefill", 65536), self.row("decode", 65536)
        self.assertAlmostEqual(pf["global"]["measured_ms"], 839.17570975)
        self.assertAlmostEqual(dec["global"]["measured_ms"], .59183525)
        self.assertGreater(self.row("prefill", 32768, "no-flush")["global"]["measured_ms"],
                           self.row("prefill", 32768)["global"]["measured_ms"])
        for phase in A.PHASES:
            for size in A.SIZES:
                self.assertEqual(self.row(phase, size)["global"]["theory_ms"],
                                 self.row(phase, size, "no-flush")["global"]["theory_ms"])

    def test_raw_sample_corruption_rejected(self):
        records = A.read_records(A.HERE / "long_context_logs/decode_65536.log")
        plan = records["PLAN"][0]
        for bad_value in (float("nan"), -1, .5):
            bench = copy.deepcopy(records["BENCH"][0])
            bench["event_samples_ms"][0][0] = bad_value
            with self.assertRaises(ValueError):
                A.audit_timing(bench, plan)
        bench = copy.deepcopy(records["BENCH"][0])
        bench["event_samples_ms"][0].pop()
        with self.assertRaisesRegex(ValueError, "ragged"):
            A.audit_timing(bench, plan)

    def test_work_and_plan_corruption_rejected(self):
        rec = A.read_records(A.HERE / "long_context_logs/prefill_65536.log")
        work = dict(rec["WORK"][0], sparse_candidates_sum=0)
        with self.assertRaisesRegex(ValueError, "WORK.sparse_candidates_sum"):
            A.audit_plan(rec["PLAN"][0], work, rec["BENCH"][0])
        plan = copy.deepcopy(rec["PLAN"][0])
        plan["score_plan"]["chunks"][-1]["row_end"] -= 1
        with self.assertRaisesRegex(ValueError, "coverage"):
            A.audit_plan(plan, rec["WORK"][0], rec["BENCH"][0])

    def test_parser_noise_nul_and_malformed_records(self):
        text = "compiler noise\0\n" + "\n".join(
            f"{tag} {{}}" for tag in ("ENV", "PLAN", "WORK", "PASS", "DONE", "BENCH", "BENCH"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.log"
            path.write_text(text)
            self.assertEqual(len(A.read_records(path)["BENCH"]), 2)
            path.write_text(text + "\nBENCH not-json")
            with self.assertRaisesRegex(ValueError, "invalid BENCH JSON"):
                A.read_records(path)
            path.write_text(text + "\nWORK {}")
            with self.assertRaisesRegex(ValueError, "expected one WORK"):
                A.read_records(path)

    def test_json_and_report_are_deterministic(self):
        encoded = json.dumps(self.data, allow_nan=False)
        self.assertEqual(len(json.loads(encoded)["rows"]), 28)
        report = A.render_report(self.data)
        self.assertEqual(report, A.render_report(copy.deepcopy(self.data)))
        saved = A.HERE / "LONG_CONTEXT_ROOFLINE_CN.md"
        if saved.exists():
            self.assertEqual(saved.read_text(), report)


if __name__ == "__main__":
    unittest.main()