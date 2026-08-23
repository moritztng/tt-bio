"""`af2_easy`'s fourth criterion: bound-unbound RMSD < 3.5.

The other three are confidence scalars one trunk pass produces. This one is a comparison between
two predictions of the same binder, so it is a join rather than a model run: `filter_tolerance.py
--dump-ca` writes each stage's last-recycle CA cloud, and this superposes the binder-alone
prediction onto the same binder inside the complex prediction of the same arm.

Transcribed from `pxdbench/metrics/Kalign.py:205-235`, and the details are the number:

* every CA of the monomer prediction against every CA of the binder chain of the complex
  prediction, paired by index. No weights, no per-residue mask, no alignment on a subset.
* differing CA counts is an error, not a resolvable pairing. Upstream's `Binder_` variant returns
  None where its sibling `align_and_calculate_rmsd` falls back to common keys.
* the threshold applies to the value rounded to two decimals (`tools/af2/main_af2_monomer.py:164`),
  the same convention `af2_confidence`'s docstring records for the three confidence scalars. It
  only matters within 0.005 of the bar, and it is one line.

`--upstream` runs pxdbench's own implementation on CA-only PDBs written from the same two clouds
and requires agreement at two decimals. On the first design it was tried on the two agreed to 6e-5,
which is what makes it worth keeping as a check instead of a one-off: a coordinate metric with a
sign convention and a reflection branch has more than one way to be quietly wrong.

    PYTHONPATH=. python3 scripts/af2_port/bound_unbound_rmsd.py \\
        --population parity_artifacts/designpop_pxd196/population.jsonl \\
        --ca-dir .af2ig_p14/ca_device --arm device --upstream --out rmsd_device.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RMSD_BAR = 3.5          # pxdbench/pxd_configs/eval.py:84-89
UPSTREAM_SRC = "/home/ttuser/pxdbench_src"
# the CA-only PDB carries three decimals, so a cross-implementation comparison that goes through
# one cannot be tighter than half of the last place it can represent
UPSTREAM_TOL = 0.005


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """Unweighted index-paired CA superposition, `Kalign.py:50-92` including the reflection fix."""
    if mobile.shape != target.shape:
        raise ValueError("CA counts differ: %s vs %s -- upstream's Binder_ variant returns None "
                         "here rather than matching residues up" % (mobile.shape, target.shape))
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("expected (N, 3) CA clouds, got %s" % (mobile.shape,))
    cm, ct = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - cm).T @ (target - ct))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = vt.T @ u.T
    d = mobile - ((target - ct) @ rot + cm)
    return float(np.sqrt((d * d).sum() / len(mobile)))


def accepts(rmsd: float | None) -> bool:
    """Upstream thresholds the rounded value, so 3.494 accepts and 3.504 rejects."""
    return rmsd is not None and round(rmsd, 2) < RMSD_BAR


def write_ca_pdb(path: str | Path, chains: list[tuple[str, np.ndarray]]) -> None:
    """The minimum a PDB parser needs to give back the same CA cloud, one ALA per residue."""
    lines, serial = [], 1
    for chain_id, coords in chains:
        for i, (x, y, z) in enumerate(coords, start=1):
            lines.append("ATOM  %5d  CA  ALA %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C"
                         % (serial, chain_id, i, x, y, z))
            serial += 1
        lines.append("TER")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def upstream_rmsd(monomer: np.ndarray, target: np.ndarray, binder: np.ndarray,
                  work: Path) -> float:
    """pxdbench's own `Binder_align_and_calculate_rmsd`, through its own PDB reader."""
    import sys
    if UPSTREAM_SRC not in sys.path:
        sys.path.insert(0, UPSTREAM_SRC)
    from pxdbench.metrics.Kalign import Binder_align_and_calculate_rmsd
    work.mkdir(parents=True, exist_ok=True)
    mono, cplx = work / "mono_ca.pdb", work / "cplx_ca.pdb"
    write_ca_pdb(mono, [("A", monomer)])
    write_ca_pdb(cplx, [("A", target), ("B", binder)])
    got = Binder_align_and_calculate_rmsd(str(mono), str(cplx), "B")
    if got is None:
        raise ValueError("upstream returned None: it refuses a CA-count mismatch")
    return float(got)


def population_ids(path: Path) -> list[tuple[str, int]]:
    """(id, binder_len) per unique population row, in file order."""
    out, seen = [], set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["pdb"], row["seq"])
        if key in seen:
            continue
        seen.add(key)
        out.append((row["id"], int(row.get("binder_len", len(row["seq"])))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True)
    ap.add_argument("--ca-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--upstream", action="store_true",
                    help="cross-check every row against pxdbench's own implementation and fail if "
                         "the two disagree at two decimals")
    ap.add_argument("--work", default="/tmp/af2ig_rmsd")
    args = ap.parse_args()

    ca_dir = Path(args.ca_dir)
    rows, missing, checked = [], [], 0
    for rid, binder_len in population_ids(Path(args.population)):
        cpath, mpath = ca_dir / ("%s.complex_ca.npy" % rid), ca_dir / ("%s.monomer_ca.npy" % rid)
        if not (cpath.exists() and mpath.exists()):
            missing.append(rid)
            continue
        complex_ca, monomer_ca = np.load(cpath), np.load(mpath)
        bound = complex_ca[-binder_len:]
        rmsd = kabsch_rmsd(monomer_ca, bound)
        row = {"id": rid, "arm": args.arm, "binder_len": binder_len,
               "bound_unbound_rmsd": round(rmsd, 6),
               "bound_unbound_rmsd_rounded2": round(rmsd, 2),
               "passes_rmsd": accepts(rmsd)}
        if args.upstream:
            try:
                theirs = upstream_rmsd(monomer_ca, complex_ca[:-binder_len], bound,
                                       Path(args.work) / args.arm / rid)
            except ImportError as exc:
                print("upstream cross-check SKIPPED, not checked: %s" % exc, flush=True)
                args.upstream = False
            else:
                row["bound_unbound_rmsd_upstream"] = round(theirs, 6)
                row["upstream_diff"] = round(abs(rmsd - theirs), 6)
                assert round(theirs, 2) == round(rmsd, 2) and abs(rmsd - theirs) < UPSTREAM_TOL, \
                    "%s: ours %.6f vs upstream %.6f" % (rid, rmsd, theirs)
                checked += 1
        rows.append(row)
        print(json.dumps(row), flush=True)

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"arm": args.arm, "scored": len(rows), "missing_ca": missing,
                      "upstream_checked": checked,
                      "accepted": sum(1 for r in rows if r["passes_rmsd"]),
                      "max_rmsd": max((r["bound_unbound_rmsd"] for r in rows), default=None)}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
