"""Offline real-log audit and detection of damaged timing/validation evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from analyze_topk_pipeline import audit, check_stats, stats


class AuditTests(unittest.TestCase):
    def test_statistics(self):
        got = stats([3, 1, 2, 4])
        self.assertEqual((got["mean_ms"], got["median_ms"]), (2.5, 2.5))
        self.assertAlmostEqual(got["p95_ms"], 3.85, places=14)
        check_stats(stats([1]), [1])

    def test_all_seven_logs(self):
        files = sorted(Path(__file__).with_name("topk_long_logs").glob("pipeline_ab_*.log"))
        self.assertEqual(len(files), 7)
        total = 0
        for path in files:
            _, count = audit(path)
            total += count
        self.assertGreater(total, 0)

    def test_reject_tampered_evidence(self):
        source = Path(__file__).with_name("topk_long_logs") / "pipeline_ab_decode16384.log"
        for tag, mutate in (
            ("BENCH", lambda r: r["gpu_sum"].update(mean_ms=12345)),
            ("SAMPLE", lambda r: r.update(variant="fast")),
            ("AB_PASS", lambda r: r.update(all_guards=False)),
            ("SAME_SCORE_CHUNK", lambda r: r.update(all_selected_counts_exact=False)),
            ("HISTORY", lambda r: r.update(included_in_timing=True)),
        ):
            lines = source.read_text().splitlines()
            i = next(i for i, line in enumerate(lines) if line.startswith(tag + " "))
            data = json.loads(lines[i].partition(" ")[2])
            mutate(data)
            lines[i] = tag + " " + json.dumps(data)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "tampered.log"
                path.write_text("\n".join(lines))
                with self.assertRaises(AssertionError, msg=tag):
                    audit(path)


if __name__ == "__main__":
    unittest.main()