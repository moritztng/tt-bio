#!/usr/bin/env python3
"""Per-domain Kabsch RMSD between the CIFs a fold A/B wrote, for the chimeric cdk2x2 fixtures.

`perf/other512/cif_rmsd.py` superposes ALL atoms at once. On the cdk2x2_N family that number is
unreadable for any non-bit-exact arm: the fixture is CDK2 concatenated to truncated copies of
itself with no real inter-domain interface, so the softest degree of freedom is the hinge between
pseudo-domains and it saturates all-atom RMSD at ~8 A whatever caused the change (memory
`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`). The diagnosis that exposed it
superposed each pseudo-domain separately; this script is that, as a tool.

cdk2x2_768 is the 298 aa CDK2 sequence twice plus a 172 aa truncation, so the domain boundaries
are the repeat length. Pass `--repeat` for another member of the family.

    cif_domain_rmsd.py <root-of-arm-dirs> [--size 768] [--repeat 298]

Reports, per arm pair: all-atom RMSD (the saturating number), then each domain superposed on
itself. A bit-level perturbation keeps every domain near 0 and moves only the hinge; a real
accuracy change moves the domains too.
"""
import argparse, itertools, json, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "other512"))
from cif_rmsd import read_atoms, kabsch_rmsd, arm_of   # same parser, same superposition


def seq_ids(keys):
    """label_seq_id per atom, as int, from the key tuple cif_rmsd.read_atoms builds."""
    # keys are (asym, seq, atom, comp) in that column order when all four are present
    out = []
    for k in keys:
        for f in k:
            if f.isdigit():
                out.append(int(f))
                break
        else:
            raise SystemExit(f"no numeric label_seq_id in key {k}")
    return np.asarray(out)


def domains(n_res, repeat):
    b = list(range(1, n_res + 1, repeat)) + [n_res + 1]
    return [(b[i], b[i + 1] - 1) for i in range(len(b) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--size", default=None)
    ap.add_argument("--repeat", type=int, default=298)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    data = {}
    for d in sorted(a.root.iterdir()):
        if not d.is_dir():
            continue
        size, arm = arm_of(d.name)
        if a.size and size != a.size:
            continue
        cifs = sorted(d.glob("*.cif"))
        if cifs:
            k, x = read_atoms(cifs[0])
            data[d.name] = (arm, k, x)
    if len(data) < 2:
        raise SystemExit("need at least two folds on disk")

    ref = sorted(data)[0]
    for name, (_, k, _) in data.items():
        if k != data[ref][1]:
            raise SystemExit(f"atom identity differs in {name}")
    sid = seq_ids(data[ref][1])
    doms = domains(int(sid.max()), a.repeat)
    print(f"  {len(data)} folds, {len(sid)} atoms, {int(sid.max())} residues, "
          f"domains {doms}\n")

    rows = []
    for x, y in itertools.combinations(sorted(data), 2):
        P, Q = data[x][2], data[y][2]
        kind = "A/A" if data[x][0] == data[y][0] else "A/B"
        per = []
        for lo, hi in doms:
            m = (sid >= lo) & (sid <= hi)
            per.append(kabsch_rmsd(P[m], Q[m]))
        row = {"kind": kind, "a": x, "b": y, "all_atom": kabsch_rmsd(P, Q),
               "max_abs_coord_delta": float(np.abs(P - Q).max()),
               "per_domain": [{"lo": lo, "hi": hi, "rmsd": r}
                              for (lo, hi), r in zip(doms, per)]}
        rows.append(row)
        print(f"  {kind}  {x:24s} vs {y:24s}  all-atom {row['all_atom']:9.6f} A   "
              + "  ".join(f"d{i+1} {r:9.6f}" for i, r in enumerate(per)))
    if a.out:
        a.out.write_text(json.dumps({"repeat": a.repeat, "domains": doms, "rows": rows}, indent=1))


main()
