#!/usr/bin/env python3
"""Replay archived reference seed spread; no folds, timing or new acceptance bar."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'perf/other512'))
from cif_rmsd import kabsch_rmsd, read_atoms


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_atoms(keys, xyz, size):
    if (not keys or any(len(k) != 4 for k in keys) or len(set(keys)) != len(keys)
            or xyz.shape != (len(keys), 3) or not np.isfinite(xyz).all()):
        raise ValueError('missing/duplicate atom identity or invalid coordinates')
    ca = [k for k in keys if k[2] == 'CA']
    if len({k[0] for k in keys}) != 1 or len(ca) != size:
        raise ValueError('expected one complete fixture chain')
    if sorted(int(k[1]) for k in ca) != list(range(1, size + 1)):
        raise ValueError('CA sequence identities do not match the fixture')


def replay(root=ROOT):
    archive = root / 'perf/k10_anchor'
    score_path = archive / 'out/score.json'
    recorded = json.loads(score_path.read_text())
    result = {'scope': 'CPU replay of archived fp32 upstream reference CIFs, not current TT accuracy or a float64 model reference.',
              'timing': 'No device or performance measurement.',
              'metric': 'Maximum independently superposed domain all-atom Kabsch RMSD, split after residue 298; one domain at 298 aa.',
              'score_path': str(score_path.relative_to(root)), 'score_sha256': sha(score_path),
              'arithmetic': 'float64 coordinate superposition with the existing CIF scorer',
              'sizes': {}}
    for size in (298, 512):
        inputs, atoms = [], {}
        for seed in range(4):
            path = archive / 'gpucif' / f'{size}_gpuref-s{seed}' / f'cdk2x2_{size}.cif'
            keys, xyz = read_atoms(path)
            validate_atoms(keys, xyz, size)
            if atoms and keys != atoms[0][0]:
                raise ValueError('reference seeds have different atom identities')
            atoms[seed] = keys, xyz
            inputs.append({'seed': seed, 'path': str(path.relative_to(root)), 'sha256': sha(path)})
        seq = np.array([int(k[1]) for k in atoms[0][0]])
        selections = {'domain1_all_atom_A': seq <= 298}
        if size == 512:
            selections['domain2_all_atom_A'] = seq > 298
        pairs = []
        for a, b in combinations(range(4), 2):
            tag = f'gpuref: s{a} vs s{b}'
            expected = recorded['sizes'][str(size)]['seed_floor'][tag]
            measured = {name: kabsch_rmsd(atoms[a][1][sel], atoms[b][1][sel])
                        for name, sel in selections.items()}
            if any(abs(value - expected[name]) > 0.0000051 for name, value in measured.items()):
                raise ValueError(f'archived scorer mismatch at {size} {tag}')
            pairs.append({'seeds': [a, b], **measured, 'worst_domain_A': max(measured.values())})
        spread = [p['worst_domain_A'] for p in pairs]
        result['sizes'][str(size)] = {
            'inputs': inputs, 'seed_pairs': pairs, 'pairs': len(pairs),
            'mean_A': float(np.mean(spread)), 'min_A': min(spread), 'max_A': max(spread),
            'matches_archived_five_decimal_scores': True,
        }
    result['limits'] = ('A seed floor differs from same-seed A/A repeatability. These old reference folds do not replace current paired baseline/stack/A/A controls. A shared numeric seed across different sampler implementations does not guarantee shared noise. Float64 mathematical transform controls are a separate requirement. No acceptance threshold is inferred from these samples.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        result = replay()
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f'invalid reference evidence: {error}\n')
    text = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
