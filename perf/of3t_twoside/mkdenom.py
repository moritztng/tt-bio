#!/usr/bin/env python3
"""of3t-twoside: the TWO-SIDED bf16 denominator (D241).

`of3t_refprec/pinned_p175/arm4_bf16_autocast/grads_f64.pt` is a full-model bf16 autocast run.
Its trunk was driven by whatever bf16 boundary and bf16 cotangent its own forward and backward
produced, while our arm is handed the reference's float64 pair. PROTOCOL A34 wants both sides on
the same boundary, so this writes arm4 with its 2,736 `pairformer_stack.*` tensors replaced by
the injected bf16auto arm's and NOTHING else touched.

Every other section is byte-carried from arm4, which is the point: the clause moves only by the
trunk, so the re-scored headline is attributable to the substitution and not to five other
changes at once.

    mkdenom.py --base arm4/grads_f64.pt --trunk inj_bf16.pt --out twosided_bf16.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import torch

PRE = 'pairformer_stack.'


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, type=Path, help='the full-model bf16 autocast run')
    ap.add_argument('--base-sha256', required=True)
    ap.add_argument('--trunk', required=True, type=Path, help='the injected bf16auto trunk arm')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--report', required=True, type=Path)
    a = ap.parse_args()

    got = sha256(a.base)
    if got != a.base_sha256:
        raise SystemExit(f'STOP: base is {got} and the pin says {a.base_sha256}')

    base = torch.load(a.base, map_location='cpu', weights_only=False)
    wrapped = isinstance(base, dict) and 'grads' in base and isinstance(base['grads'], dict)
    g = base['grads'] if wrapped else base

    t = torch.load(a.trunk, map_location='cpu', weights_only=False)
    tg = t['grads'] if isinstance(t, dict) and 'grads' in t else t

    base_trunk = sorted(k for k in g if k.startswith(PRE))
    inj_trunk = sorted(k for k in tg if k.startswith(PRE))
    if base_trunk != inj_trunk:
        only_b = sorted(set(base_trunk) - set(inj_trunk))[:4]
        only_i = sorted(set(inj_trunk) - set(base_trunk))[:4]
        raise SystemExit(f'STOP: the trunk key sets differ. {len(base_trunk)} in the base, '
                         f'{len(inj_trunk)} in the arm. base-only {only_b} arm-only {only_i}. '
                         f'A substitution over a different key set is a different experiment.')

    moved, shape_bad = 0, []
    for k in base_trunk:
        if tg[k] is None:
            raise SystemExit(f'STOP: the injected arm has no gradient at {k!r}')
        if tuple(g[k].shape) != tuple(tg[k].shape):
            shape_bad.append((k, list(g[k].shape), list(tg[k].shape)))
            continue
        g[k] = tg[k].to(g[k].dtype).clone()
        moved += 1
    if shape_bad:
        raise SystemExit(f'STOP: {len(shape_bad)} shape mismatches, e.g. {shape_bad[:2]}')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, a.out)
    rep = {
        'what': __doc__.strip().splitlines()[0],
        'host': socket.gethostname(), 'row': 'of3t-twoside', 'defect': 'D241',
        'device_involved': False,
        'base': {'path': str(a.base), 'sha256': got, 'n_tensors': len(g)},
        'trunk_arm': {'path': str(a.trunk), 'sha256': sha256(a.trunk)},
        'substituted': {'n': moved, 'prefix': PRE,
                        'n_untouched': len(g) - moved},
        'wrapped_under_grads_key': wrapped,
        'out': {'path': str(a.out), 'sha256': sha256(a.out),
                'bytes': os.path.getsize(a.out)},
    }
    a.report.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
