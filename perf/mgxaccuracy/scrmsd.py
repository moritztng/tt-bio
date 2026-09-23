#!/usr/bin/env python3
"""scRMSD from the refolds on disk, without waiting for the analysis step.

`scripts/boltzgen_designability.py` reads `aggregate_metrics_analyze.csv`, which BoltzGen
writes at step 5 of 6. The number in it is already determined at step 4: `design_folding`
refolds each designed sequence in isolation into
`intermediate_designs_inverse_folded/refold_design_cif/`, and analysis only Kabsch-aligns
that refold to the designed backbone. So a run that has finished refolding already holds its
designability, and the step after it is not free -- on the CPU reference one 512 design cost
2:01:29 in the design step alone.

Two refold directories sit side by side and they are not interchangeable:
`refold_design_cif/` is step 4, the design refolded ALONE (80 residues here), and it is what
`designfolding-bb_rmsd` measures; `refold_cif/` is step 3, the whole complex refolded (592
residues here), which feeds `bb_rmsd_design`. Reading the wrong one silently answers a
different question.

This reads the metric out at step 4. It is only usable because it is checked against the
pipeline's own column on a run that did reach analysis:

    python3 perf/mgxaccuracy/scrmsd.py OUT_DIR --check      # reproduce the CSV, report the gap
    python3 perf/mgxaccuracy/scrmsd.py OUT_DIR --json       # score a run that stopped at step 4

`--check` is the part that matters: a recomputation nobody compared against the instrument it
replaces is a second opinion, not a reading.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contact import atoms  # noqa: E402

# BoltzGen's `bb_rmsd` is over the backbone, and which atoms "the backbone" means is the one
# free choice in reproducing it. All three candidates are computed and --check names the one
# that matches, so the answer is measured rather than assumed.
BACKBONES = {"N,CA,C,O": ("N", "CA", "C", "O"), "N,CA,C": ("N", "CA", "C"), "CA": ("CA",)}

METRICS = {
    "designfolding": ("refold_design_cif", "designfolding-bb_rmsd",
                      "design refolded alone (step 4)"),
    "complex": ("refold_cif", "bb_rmsd_design", "whole complex refolded (step 3)"),
}


def kabsch_rmsd(p, q):
    """RMSD after the optimal rigid superposition of p onto q (both (n,3))."""
    import numpy as np
    p = p - p.mean(0)
    q = q - q.mean(0)
    v, _, wt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(v @ wt))
    r = v @ np.diag([1.0, 1.0, d]) @ wt
    return float(np.sqrt((((p @ r) - q) ** 2).sum(1).mean()))


def backbone(path, names, chain=None):
    """(residue key, xyz) rows for the named backbone atoms, in file order."""
    import numpy as np
    meta, xyz = atoms(path)
    keep = [i for i, (c, s, a) in enumerate(meta)
            if a in names and (chain is None or c == chain)]
    return [(meta[i][0], meta[i][1], meta[i][2]) for i in keep], np.asarray(xyz)[keep]


def designed_chain(design_cif, n_res):
    """The binder chain in the complex: the one with `n_res` residues.

    The refold holds the binder alone, so its residue count identifies the chain without
    needing the pipeline's own bookkeeping. A tie is refused rather than guessed."""
    meta, _ = atoms(design_cif)
    per: dict = {}
    for c, s, a in meta:
        if a == "CA":
            per[c] = per.get(c, 0) + 1
    hits = [c for c, n in per.items() if n == n_res]
    if len(hits) != 1:
        raise SystemExit(f"{design_cif.name}: {len(hits)} chains with {n_res} residues "
                         f"(chains {per}) — cannot identify the binder")
    return hits[0]


def score_dir(out: pathlib.Path, metric: str = "designfolding") -> list[dict]:
    sub, _, what = METRICS[metric]
    refolds = sorted((out / "intermediate_designs_inverse_folded" / sub).glob("*.cif"))
    if not refolds:
        raise SystemExit(f"no {sub}/*.cif under {out} — {what} has not run")
    rows = []
    for rf in refolds:
        design = out / "intermediate_designs" / rf.name
        if not design.is_file():
            raise SystemExit(f"{rf.name}: no matching design in intermediate_designs/")
        row = {"id": rf.stem}
        for label, names in BACKBONES.items():
            rmeta, rxyz = backbone(rf, names)
            n_res = len({(c, s) for c, s, _ in rmeta})
            ch = designed_chain(design, n_res)
            dmeta, dxyz = backbone(design, names, chain=ch)
            if len(rxyz) != len(dxyz):
                row[label] = None
                row.setdefault("skipped", []).append(
                    f"{label}: {len(rxyz)} refold vs {len(dxyz)} design atoms")
                continue
            row[label] = kabsch_rmsd(rxyz, dxyz)
            row["n_res"], row["chain"] = n_res, ch
        rows.append(row)
    return rows


def check(out: pathlib.Path, rows: list[dict], metric: str) -> int:
    """Compare every backbone definition against the pipeline's own column."""
    import csv
    hits = sorted(out.rglob("aggregate_metrics_*.csv"))
    if not hits:
        print(f"[scrmsd] no aggregate_metrics_*.csv under {out} — nothing to check against",
              file=sys.stderr)
        return 1
    col = METRICS[metric][1]
    ref = {}
    for r in csv.DictReader(open(min(hits, key=lambda p: len(p.parts)))):
        v = r.get(col)
        if v not in (None, ""):
            ref[r["id"]] = float(v)
    print(f"\n{'id':<14}{col[:10]:>10}" + "".join(f"{k:>12}" for k in BACKBONES))
    worst = {k: 0.0 for k in BACKBONES}
    for row in rows:
        if row["id"] not in ref:
            continue
        line = f"{row['id']:<14}{ref[row['id']]:>10.5f}"
        for k in BACKBONES:
            v = row.get(k)
            line += f"{v:>12.5f}" if v is not None else f"{'-':>12}"
            if v is not None:
                worst[k] = max(worst[k], abs(v - ref[row["id"]]))
        print(line)
    print(f"\n{'':<14}{'max |gap|':>10}" + "".join(f"{worst[k]:>12.5f}" for k in BACKBONES))
    best = min(worst, key=lambda k: worst[k])
    ok = worst[best] <= 0.01
    print(f"\nclosest backbone definition: {best}  max gap {worst[best]:.5f} A  "
          f"-> {'REPRODUCES the pipeline column' if ok else 'DOES NOT reproduce it'}")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--check", action="store_true",
                    help="compare against aggregate_metrics_*.csv and exit non-zero on a gap")
    ap.add_argument("--backbone", default="N,CA,C,O", choices=sorted(BACKBONES),
                    help="atom set for the reported scrmsd (--json)")
    ap.add_argument("--metric", default="designfolding", choices=sorted(METRICS),
                    help="designfolding = the scRMSD bar; complex = the step-3 whole-complex fit")
    ap.add_argument("--json", action="store_true", help="emit one jsonl-ready record")
    # The record has to land in report.py's cell key, which is (side, target, size, offset) --
    # pooling two targets of one size was a real defect here, so these are not optional
    # decoration. A row missing `target` is kept separate rather than merged into anything.
    ap.add_argument("--model", default="boltzgen")
    ap.add_argument("--target", default="", help="the target file this run designed against")
    ap.add_argument("--target-res", type=int, default=None, help="residues of the target crop")
    ap.add_argument("--side", default="device",
                    help="which side of the comparison, e.g. device or upstream-cpu")
    args = ap.parse_args()

    out = args.out_dir.expanduser()
    rows = score_dir(out, args.metric)
    if args.check:
        return check(out, rows, args.metric)
    vals = [r[args.backbone] for r in rows if r.get(args.backbone) is not None]
    if not vals:
        raise SystemExit("no design scored")
    import statistics as st
    rec = {"model": args.model, "target": args.target, "target_res": args.target_res,
           "side": args.side, "out_dir": str(out),
           "source": f"{METRICS[args.metric][0]} on disk, not analysis",
           "metric": METRICS[args.metric][1], "backbone": args.backbone,
           "n": len(vals), "scrmsd": vals, "median": st.median(vals),
           "min": min(vals), "max": max(vals),
           "pass_strict": sum(v <= 2.0 for v in vals) / len(vals),
           "pass_permissive": sum(v <= 4.0 for v in vals) / len(vals)}
    print(json.dumps(rec) if args.json else
          "\n".join(f"{r['id']:<14}{r[args.backbone]:>10.3f}" for r in rows)
          + f"\nn={rec['n']} median {rec['median']:.3f} A  "
            f"<=2A {rec['pass_strict']*100:.0f}%  <=4A {rec['pass_permissive']*100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
