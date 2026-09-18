#!/usr/bin/env python3
"""Reduce one or more capture_ab runs to per-arm fold seconds, Mcycles and structure.

Seconds are converted at the clock each fold was actually measured at, which the run records
per label; a run whose during-clock is not a single pinned value is refused rather than quoted.
The A-to-A term is the adjacent-A spread of the same session, and it is the bar an arm has to
clear before its delta means anything.
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path


def med(xs):
    return statistics.median(xs) if xs else None


def summarise(run: Path):
    r = json.loads((run / "result.json").read_text())
    rows = [x for x in r["rows"] if x["label"] != "cold" and x.get("valid")]
    clocks = {x["clock"]["min_MHz"] for x in rows} | {x["clock"]["max_MHz"] for x in rows}
    if len(clocks) != 1:
        raise SystemExit(f"{run}: clock not pinned during folds: {sorted(clocks)}")
    mhz = clocks.pop()
    arms: dict[str, list] = {}
    for x in rows:
        arms.setdefault(x["arm"], []).append(x)
    a = sorted(x["elapsed_s"] for x in arms.get("A", []))
    aa = max(a) - min(a) if len(a) > 1 else None
    base = med(a)
    out = dict(run=str(run), size=r["size"], node=r["node"], MHz=mhz, completed=r["completed"],
               errors=r["errors"], reps=r["reps"], A_spread_s=aa, arms={})
    for name, xs in sorted(arms.items()):
        s = sorted(x["elapsed_s"] for x in xs)
        d = med(s) - base
        out["arms"][name] = dict(
            n=len(s), median_s=round(med(s), 4), min_s=round(min(s), 4), max_s=round(max(s), 4),
            delta_s=round(d, 4), delta_Mcycles=round(-d * mhz, 1),
            injected=max(sum(x["injected"].values()) for x in xs),
            refused=max(sum(x["refused"].values()) for x in xs),
            cifs=sorted({x["cif_sha256"][:12] for x in xs}),
            rmsd_vs_A0_A=[round((x.get("structure_vs_A0") or {}).get("max_domain_all_atom_A", float("nan")), 4) for x in xs],
            plddt=[round(float(x["plddt"]), 4) for x in xs])
    return out


if __name__ == "__main__":
    print(json.dumps([summarise(Path(p)) for p in sys.argv[1:]], indent=1))
