#!/usr/bin/env python3
"""of3t-twoside: the trunk section scored with both sides on the SAME boundary (D241).

The scorer is `perf/of3t_trajectory/agreement.py`, imported and not reimplemented -- it already
applies A14 on the float64 gradient for every pair, scores a tensor absent from an arm as zero
instead of dropping its mass, and keeps r and cos beside rel_l2 (D35). A fourth summary-only
variant of that arithmetic is what this campaign keeps being bitten by.

R133: every reading below names its reference in its own label. 'ours against float64' and
'ours against upstream's bf16' are different numbers and the orchestrator already conflated them
once.

    trunkscore.py --f64 grads_f64_043.pt --arm NAME=path.pt ... --pair ARM:REF ... --out O.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'perf' / 'of3t_trajectory'))
import agreement  # noqa: E402  the scorer, not a copy of it

PRE = 'pairformer_stack.'


def load(path, names=None, unwrap='grads'):
    d = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(d, dict) and unwrap in d and isinstance(d[unwrap], dict):
        d = d[unwrap]
    if names is None:
        names = sorted(k for k in d if k.startswith(PRE))
        return {n: d[n].to(torch.float64).reshape(-1) for n in names
                if d[n] is not None}, names
    return {n: (d[n].to(torch.float64).reshape(-1) if d.get(n) is not None else None)
            for n in names}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--f64', required=True, help='the model float64 reference, grads_f64_043.pt')
    ap.add_argument('--arm', action='append', default=[], metavar='NAME=PATH')
    ap.add_argument('--pair', action='append', default=[], metavar='ARM:REF')
    ap.add_argument('--mass-floor', type=float, default=1e-6,
                    help='for the secondary per-tensor report: the share of the trunk squared '
                         'gradient mass a tensor must carry for its relative error to mean '
                         'anything')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()

    print(f'loading float64 reference {a.f64}', flush=True)
    f64, names = load(a.f64, None)
    trunk_sq = sum(float(torch.linalg.vector_norm(v)) ** 2 for v in f64.values())
    print(f'  {len(names)} trunk tensors, squared norm {trunk_sq!r}', flush=True)
    f64 = {n: f64.get(n) for n in names}

    arms = {'FLOAT64': f64}
    paths = {'FLOAT64': a.f64}
    for spec in a.arm:
        nm, p = spec.split('=', 1)
        print(f'loading arm {nm} {p}', flush=True)
        arms[nm] = load(p, names)
        paths[nm] = p
        got = sum(1 for v in arms[nm].values() if v is not None)
        print(f'  {got}/{len(names)} present', flush=True)

    sections = ['pairformer_stack']
    out = {
        'what': __doc__.strip().splitlines()[0],
        'host': os.uname().nodename,
        'row': 'of3t-twoside', 'defect': 'D241',
        'device_involved': False,
        'scope': 'pairformer_stack only',
        'n_trunk_tensors': len(names),
        'trunk_squared_gradient_norm_float64': trunk_sq,
        'inputs': {k: {'path': str(v), 'bytes': os.path.getsize(v),
                       'sha256': agreement.sha256(v)} for k, v in paths.items()},
        'readings': {},
    }
    for spec in a.pair:
        arm, ref = spec.split(':', 1)
        label = f'{arm}_vs_{ref}'
        print(f'scoring {label}', flush=True)
        rows = agreement.pair_rows(arms[ref], arms[arm], names, f64, sections)
        s = agreement.stat(rows, label)
        # the secondary per-tensor view, restricted to tensors that carry mass. A14's floor is
        # absolute; this one is relative to the trunk, which is the thing the clause weighs by.
        keep = [r for r in rows
                if r['rel_l2'] is not None and r['mass_sq'] / trunk_sq >= a.mass_floor]
        w = max(keep, key=lambda r: r['rel_l2'], default=None)
        s['worst_over_mass_floor'] = {
            'mass_floor_share_of_trunk': a.mass_floor, 'n_tensors_above_floor': len(keep),
            'worst_tensor': w['param'] if w else None,
            'worst_rel_l2': w['rel_l2'] if w else None,
            'worst_share_of_trunk_mass': (w['mass_sq'] / trunk_sq) if w else None,
        }
        s['top5_by_error_mass'] = [
            {'param': r['param'], 'rel_l2': r['rel_l2'],
             'pct_of_trunk_error_mass': 100.0 * (r['diff_norm'] or 0.0) ** 2
             / sum((q['diff_norm'] or 0.0) ** 2 for q in rows),
             'pct_of_trunk_reference_mass': 100.0 * r['ref_norm'] ** 2
             / sum(q['ref_norm'] ** 2 for q in rows)}
            for r in sorted(rows, key=lambda r: -((r['diff_norm'] or 0.0) ** 2))[:5]]
        s['reference_named'] = ref
        s['arm_named'] = arm
        out['readings'][label] = s

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    for k, s in out['readings'].items():
        print(f"{k:34s} rel {s['mass_weighted_rel_l2']!r} r {s['mass_weighted_norm_ratio']!r} "
              f"cos {s['mass_weighted_cos']!r}")
        print(f"{'':34s} worst {s['worst_tensor']} {s['worst_rel_l2']!r} | over mass floor "
              f"{s['worst_over_mass_floor']['worst_tensor']} "
              f"{s['worst_over_mass_floor']['worst_rel_l2']!r}")
    print('wrote ' + str(a.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
