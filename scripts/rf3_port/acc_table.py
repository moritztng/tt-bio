#!/usr/bin/env python3
"""The ragged pad across RF3's accuracy ladder: absolute Angstrom first, ratio only as a qualifier.

Per-seed X is printed alongside the mean because ubq_76 has a two-basin seed (E3 of
state/rf3-fast-arm-accuracy.md): the reference itself lands 0.88 A from its own seed-0 sibling at
seed 4, so an arm whose device happens to follow it across reads a flatteringly low X there and a
correspondingly inflated D, which then flatters X/floor from both ends. The four non-bifurcating
seeds are reported as their own column for that reason.
"""
import argparse, json
from pathlib import Path


def row(path):
    d = json.loads(Path(path).read_text())
    m = d["metrics"]["kabsch_rmsd"]
    xs = m.get("X_per_seed") or []
    if isinstance(xs, dict):
        xs = [xs[k] for k in sorted(xs, key=lambda s: str(s))]
    xs = [x if isinstance(x, (int, float)) else x.get("rmsd", x.get("x")) for x in xs]
    return dict(tag=Path(path).stem, X=m["cross"]["mean"], Xstd=m["cross"]["std"],
                R=m["ref_floor"]["mean"], Rstd=m["ref_floor"]["std"],
                D=m["dev_floor"]["mean"], Dstd=m["dev_floor"]["std"],
                floor=m["floor_mean"], ratio=m["cross_over_floor"],
                inside=m["within_noise_floor"], per_seed=xs,
                pad=str(d.get("flags", {}).get("SDPA_RAGGED_PAD",
                       d.get("flags", {}).get("TT_BIO_SDPA_RAGGED_PAD", "?"))),
                arm=d["arm"], fixture=d["fixture"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--drop-seeds", default="", help="comma list of seed indices to also report without")
    a = ap.parse_args()
    drop = {int(x) for x in a.drop_seeds.split(",") if x != ""}
    rows = [row(p) for p in a.paths]
    hdr = (f"{'run':22s} {'fix':4s} {'X (A)':>8s} {'R':>7s} {'D':>7s} {'floor':>7s} "
           f"{'X/floor':>8s} {'inside':>7s}")
    if drop:
        hdr += f" {'X w/o ' + ','.join(map(str, sorted(drop))):>12s}"
    print(hdr)
    for r in rows:
        line = (f"{r['tag'][:22]:22s} {r['arm']:4s} {r['X']:8.4f} {r['R']:7.4f} {r['D']:7.4f} "
                f"{r['floor']:7.4f} {r['ratio']:8.4f} {str(r['inside']):>7s}")
        if drop:
            keep = [x for i, x in enumerate(r["per_seed"]) if i not in drop and x is not None]
            line += f" {sum(keep) / len(keep):12.4f}" if keep else f" {'-':>12s}"
        print(line)
    print()
    for r in rows:
        ps = ", ".join("-" if x is None else f"{x:.4f}" for x in r["per_seed"])
        print(f"  {r['tag'][:22]:22s} per-seed X: {ps}")


if __name__ == "__main__":
    main()
