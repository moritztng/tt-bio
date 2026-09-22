"""Where a rung's clashes and breaks actually sit: the fixture, the motif, or the design.

check_structure returns one number per file. At these sizes that number cannot say whether the
geometry it is unhappy about is something the model produced or something it was handed, and
the difference is the whole verdict: the motif atoms are COPIED from the input structure, so a
contact between two of them is a property of the deposited coordinates and no size result.

    python3 perf/rfd3cov/split.py <cif> <motif_residues> [--target t.cif --contig C]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLASH = 2.0          # check_structure's heavy-atom contact threshold
CA_CA_BREAK = 5.0    # its backbone-gap threshold


def atoms(path: pathlib.Path):
    """(chain, seq, name, element, xyz) per atom. Column order read, never assumed."""
    cols, rows = [], []
    for line in path.read_text().splitlines():
        st = line.strip()
        if st.startswith("_atom_site."):
            cols.append(st.split(".", 1)[1].split()[0])
        elif st.startswith(("ATOM", "HETATM")):
            f = st.split()
            if len(f) == len(cols):
                rows.append(f)
    def col(*n):
        for x in n:
            if x in cols:
                return cols.index(x)
        raise KeyError(n)
    ch, sq, an = col("label_asym_id", "auth_asym_id"), col("label_seq_id", "auth_seq_id"), \
        col("label_atom_id", "auth_atom_id")
    el = col("type_symbol")
    x, y, z = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")
    out = []
    for r in rows:
        if r[el].strip('"').upper() == "H":
            continue
        out.append((r[ch], int(r[sq]), r[an].strip('"'),
                    (float(r[x]), float(r[y]), float(r[z]))))
    return out


def residue_order(a):
    """Residues in file order, deduplicated. The contig's token order, which is how the motif
    and the designed tail are told apart."""
    seen, order = set(), []
    for ch, sq, _, _ in a:
        if (ch, sq) not in seen:
            seen.add((ch, sq))
            order.append((ch, sq))
    return order


def clash_split(a, motif_res: set) -> dict:
    """Every heavy-atom pair under 2.0 A, bucketed by which side of the motif line it is on.

    Pairs inside one residue and between sequence neighbours are excluded, the same exclusion
    check_structure makes -- a bonded pair is not a clash.
    """
    cell = {}
    for i, (ch, sq, an, p) in enumerate(a):
        k = (int(p[0] // CLASH), int(p[1] // CLASH), int(p[2] // CLASH))
        cell.setdefault(k, []).append(i)
    counts = {"motif-motif": 0, "motif-design": 0, "design-design": 0}
    worst = {k: 99.0 for k in counts}
    for (cx, cy, cz), idx in cell.items():
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    near += cell.get((cx + dx, cy + dy, cz + dz), [])
        for i in idx:
            for j in near:
                if j <= i:
                    continue
                ci, si, _, pi = a[i]
                cj, sj, _, pj = a[j]
                if ci == cj and abs(si - sj) <= 1:
                    continue
                d = math.dist(pi, pj)
                if d >= CLASH:
                    continue
                mi, mj = (ci, si) in motif_res, (cj, sj) in motif_res
                key = "motif-motif" if (mi and mj) else (
                    "design-design" if not (mi or mj) else "motif-design")
                counts[key] += 1
                worst[key] = min(worst[key], d)
    return {"counts": counts,
            "worst": {k: (round(v, 3) if v < 99 else None) for k, v in worst.items()},
            "heavy_atoms": len(a)}


def backbone(a, label: str) -> list[dict]:
    """CA-CA continuity per chain, so a gap can be attributed to a chain rather than a file."""
    ca = [(ch, sq, p) for ch, sq, an, p in a if an == "CA"]
    out = []
    for c in sorted({x[0] for x in ca}):
        t = [x for x in ca if x[0] == c]
        st = [math.dist(t[i][2], t[i + 1][2]) for i in range(len(t) - 1)]
        if not st:
            continue
        gaps = [(t[i][1], round(st[i], 2)) for i in range(len(st)) if st[i] > CA_CA_BREAK]
        out.append({"what": label, "chain": c, "n_res": len(t), "gaps": len(gaps),
                    "gap_at": gaps[:8], "step_median": round(statistics.median(st), 3)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cif")
    ap.add_argument("motif", type=int, help="motif residues, i.e. the head of the token order")
    ap.add_argument("--target", help="the input structure, scored the same way as a control")
    ap.add_argument("--target-chain", action="append", default=[],
                    help="CHAIN:LO-HI of the target the contig selected, e.g. B:1-448")
    a = ap.parse_args()

    at = atoms(pathlib.Path(a.cif))
    order = residue_order(at)
    motif_res = set(order[:a.motif])
    rep = {"cif": a.cif, "residues": len(order), "motif_residues": a.motif,
           "designed_residues": len(order) - a.motif,
           "clashes": clash_split(at, motif_res),
           "backbone": backbone(at, "output")}
    # The designed tail on its own, which is where this model's recorded caveat lives.
    des = [r for r in order[a.motif:]]
    ca = {(ch, sq): p for ch, sq, an, p in at if an == "CA"}
    tail = [ca[r] for r in des if r in ca]
    st = [math.dist(tail[i], tail[i + 1]) for i in range(len(tail) - 1)]
    rep["designed"] = {"n": len(tail), "breaks": sum(1 for v in st if v > CA_CA_BREAK),
                       "worst_step": round(max(st), 3),
                       "step_median": round(statistics.median(st), 3)}
    if a.target:
        t = atoms(pathlib.Path(a.target))
        keep = set()
        for spec in a.target_chain:
            c, rng = spec.split(":")
            lo, hi = (int(v) for v in rng.split("-"))
            keep |= {(c, i) for i in range(lo, hi + 1)}
        if keep:
            t = [x for x in t if (x[0], x[1]) in keep]
        rep["target_control"] = {
            "heavy_atoms": len(t),
            "backbone": backbone(t, "input target"),
            # Same clash rule, whole-input: this is the floor the deposited coordinates set.
            "clashes": clash_split(t, {(x[0], x[1]) for x in t})["counts"]["motif-motif"]}
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
