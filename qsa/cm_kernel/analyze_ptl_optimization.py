"""Summarize complete same-session PTL optimization A/B logs without GPU access."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

MEASURED_SOURCES = ('kernels/qsa_sparse_attention_gqa_dpas.cm',
                    'kernels/qsa_topk_radix_wg.cm', 'qsa_attention_gqa.py', 'qsa_score_chunked.py')


def check_sources(path, root):
    env = next((json.loads(line[4:]) for line in path.read_text(encoding='utf-8').splitlines()
                if line.startswith('ENV ')), None)
    if env is None:
        raise ValueError('missing ENV: ' + str(path))
    hashes = {name.replace('\\', '/'): value for name, value in env['sources'].items()}
    for name in MEASURED_SOURCES:
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if hashes.get('cm_kernel/' + name) != actual:
            raise ValueError(f'measured source mismatch: {path.name}: {name}')


def load_log(path):
    records = {}
    done, passed = False, False
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('DONE '):
            done = True
        elif line.startswith('AB_PASS '):
            passed = json.loads(line[8:]).get('all_topk_rows_exact') is True
        elif line.startswith('BENCH '):
            d = json.loads(line[6:])
            if d['cache'] in records:
                raise ValueError(f'duplicate cache in {path}')
            if not d['post_timing_ab_exact']:
                raise ValueError(f'post-timing validation failed: {path}')
            records[d['cache']] = d
    if not done or not passed or set(records) != {'cold96MiB', 'no-flush'}:
        raise ValueError(f'incomplete or unvalidated log: {path}')
    return records


def stage(v, kind):
    if kind == 'q3':
        values = [s['mean_ms'] for name, s in v['stages'].items() if name.startswith('Q3-')]
    elif kind == 'topk':
        values = [s['mean_ms'] for name, s in v['stages'].items()
                  if name.startswith('Q2-') and 'partition' not in name and 'dpas' not in name]
    else:
        raise ValueError('unknown stage: ' + kind)
    if not values:
        raise ValueError('missing ' + kind)
    return sum(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--logs', type=Path, default=Path(__file__).parent / 'ptl_optimization_logs')
    ap.add_argument('--output', type=Path, default=Path(__file__).parent / 'PTL_Q3_TOPK_AB_RESULTS.md')
    ap.add_argument('--check-sources', action='store_true',
                    help='verify current Q3/top-k kernels and wrappers against measured ENV hashes')
    args = ap.parse_args()
    lines = ['# PTL Q3 / top-k 同输入 A/B 测量', '',
        '下表单位 ms；加速比为 baseline / optimized。独立生成，不覆盖历史 roofline 报告。', '',
        'Prefill baseline 使用 fast-wg16 + 原 GQA；decode baseline 使用 legacy WG16 + split-P16。',
        'Optimized 使用 fast-wg16-cached；prefill 另启用 head-major/vector-meta。', '',
        '| 阶段 | 长度 | cache | warmup/samples | Q3 old→new | Q3 speedup | top-k old→new | top-k speedup | Total old→new | Total speedup |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    manifests = []
    for phase in ('prefill', 'decode'):
        for size in (65536, 131072):
            path = args.logs / f'ptl_ab_{phase}_{size}.log'
            records = load_log(path)
            if args.check_sources:
                check_sources(path, Path(__file__).parent)
            manifests.append((path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
            for cache in ('cold96MiB', 'no-flush'):
                d = records[cache]
                if (d['phase'], d['size']) != (phase, size):
                    raise ValueError('shape mismatch: ' + str(path))
                a, b = d['variants']['baseline'], d['variants']['optimized']
                q0, q1, t0, t1 = stage(a, 'q3'), stage(b, 'q3'), stage(a, 'topk'), stage(b, 'topk')
                v0, v1 = a['total']['mean_ms'], b['total']['mean_ms']
                lines.append(f'| {phase} | {size} | {cache} | {d["warmup"]}/{d["samples"]} | '
                    f'{q0:.6f}→{q1:.6f} | {q0/q1:.3f}× | {t0:.6f}→{t1:.6f} | {t0/t1:.3f}× | '
                    f'{v0:.6f}→{v1:.6f} | {v0/v1:.3f}× |')
    lines.extend(['', '## 测量日志 SHA256', ''])
    lines.extend(f'- `{name}`: `{digest}`' for name, digest in manifests)
    args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {args.output}: 8 complete A/B rows')


if __name__ == '__main__':
    main()
