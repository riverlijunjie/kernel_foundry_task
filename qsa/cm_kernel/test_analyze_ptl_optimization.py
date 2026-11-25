"""CPU regression for PTL same-session A/B log completeness and stage sums."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from analyze_ptl_optimization import MEASURED_SOURCES, check_sources, load_log, stage


class AnalysisTests(unittest.TestCase):
    def test_measured_source_hashes_and_windows_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {}
            for name in MEASURED_SOURCES:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(name.encode())
                sources[('cm_kernel/' + name).replace('/', '\\')] = hashlib.sha256(p.read_bytes()).hexdigest()
            log = root / 'log.txt'
            log.write_text('ENV ' + json.dumps({'sources': sources}))
            check_sources(log, root)
            (root / MEASURED_SOURCES[0]).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                check_sources(log, root)
            log.write_text('DONE {}')
            with self.assertRaises(ValueError):
                check_sources(log, root)

    def test_stage_filter(self):
        v = {'stages': {k: {'mean_ms': n} for k, n in (
            ('Q0', 10), ('Q2-partition16', 9), ('Q2-dpas', 8),
            ('Q2-fast-WG16-cached', 2), ('Q3-split-P16', 3), ('Q3-reduce', 1))}}
        self.assertEqual(stage(v, 'q3'), 4)
        self.assertEqual(stage(v, 'topk'), 2)
        with self.assertRaises(ValueError):
            stage(v, 'unknown')
        with self.assertRaises(ValueError):
            stage({'stages': {}}, 'topk')

    def test_complete_logs_required(self):
        lines = ['ENV {}', 'AB_PASS ' + json.dumps({'all_topk_rows_exact': True})]
        for cache in ('cold96MiB', 'no-flush'):
            lines.append('BENCH ' + json.dumps({'cache': cache, 'post_timing_ab_exact': True}))
        lines.append('DONE {}')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'log.txt'
            p.write_text('\n'.join(lines))
            self.assertEqual(set(load_log(p)), {'cold96MiB', 'no-flush'})
            for malformed in (lines[:-1], lines[:2] + lines[3:], lines[:1] + lines[2:],
                              lines + [lines[2]], [s.replace('true', 'false') for s in lines]):
                p.write_text('\n'.join(malformed))
                with self.assertRaises(ValueError):
                    load_log(p)


if __name__ == '__main__':
    unittest.main()
