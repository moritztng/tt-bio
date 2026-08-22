#!/usr/bin/env python3
"""The frontier table for RF3's triangle-attention arms, read off the run JSONs.

Nothing here recomputes a number. It reads what `accuracy_cell.py` wrote per arm per rung
and what `page512_tt.py` wrote per arm at 512 aa, and lays them out so the four things that
have to be compared side by side actually are:

  X in absolute Angstrom, which is the primary comparison, because the floor MOVES between
  arms -- R is arm-independent by construction but D is not, and floor = max(mean R, mean D),
  so an arm with a noisier D buys itself a larger floor and a flattering X/floor;

  that arm's own R, D, floor and X/floor beside it, so the reader can see which moved;

  the route counters, so an arm that silently did not engage cannot be reported as a result;

  both perf readings, whole fold and device-only, because RF3 is the row where they are far
  apart (3.56x and 9.4x for the shipped arm) and "4x ballpark" is only true on one of them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import ROUTE, SCOPE  # noqa: E402

# The page's own cells, quoted not re-measured (perf-page-matched-batch-protocol-recurrence).
PAGE = json.loads((ROOT / "site/data/perf-512aa.json").read_text())
RF3 = next(m for m in PAGE["models"] if m["id"] == "rf3")
H200 = RF3["cells"]["h200"]["s_per_fold"]
H200_DEV = RF3["cells"]["h200"]["split"]["device_s"]
TT_HOST = RF3["cells"]["p150a"]["split"]["host_s"]
A0_CELL = RF3["cells"]["p150a"]["s_per_fold"]


def readings(s_fold: float) -> tuple[float, float]:
    """Whole fold and device-only, against the H200, the page's own subtraction."""
    return s_fold / H200, (s_fold - TT_HOST) / H200_DEV


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def find(R: Path, arm: str, rung: str):
    """a0's landed rung keeps its published name; every other file is <arm>_<rung>.json."""
    for name in (f"{arm}_{rung}.json", f"accuracy_{rung}.json" if arm == "a0" else None):
        if name and (R / name).exists():
            return load(R / name)
    return None


def row(d: dict, metric: str) -> dict:
    v = d["metrics"][metric]
    return {"X": v["cross"]["mean"], "R": v["ref_floor"]["mean"], "D": v["dev_floor"]["mean"],
            "floor": v["floor_mean"], "ratio": v["cross_over_floor"],
            "within": v["within_noise_floor"], "per_seed": v["X_per_seed"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "perf/rf3/results"))
    ap.add_argument("--rungs", default="ubq76,7roa117,cdk2_128,cdk2_298")
    ap.add_argument("--arms", default="a0,a1,a2,a3")
    ap.add_argument("--perf", default="", help="comma-separated arm=path to page512_tt json")
    a = ap.parse_args()
    R = Path(a.results)
    rungs = a.rungs.split(",")
    arms = a.arms.split(",")

    perf = {}
    for spec in filter(None, a.perf.split(",")):
        k, _, p = spec.partition("=")
        j = load(Path(p))
        if j:
            perf[k] = j["median_s"]

    print(f"H200 cell {H200} s/fold, device-only {H200_DEV} s; TT host featurisation "
          f"{TT_HOST} s (upper bound). Shipped p150a cell {A0_CELL} s.\n")

    for metric, label in (("kabsch_rmsd", "CA-RMSD"), ("allatom_rmsd", "all-atom RMSD")):
        print(f"### {label}, absolute Angstrom (X), with each arm's own floor\n")
        print("| rung | arm | X (A) | R (A) | D (A) | floor (A) | X/floor | within floor |")
        print("|---|---|--:|--:|--:|--:|--:|---|")
        for rung in rungs:
            for arm in arms:
                d = find(R, arm, rung)
                if d is None:
                    continue
                r = row(d, metric)
                print(f"| {rung} | {arm} | {r['X']:.4f} | {r['R']:.4f} | {r['D']:.4f} | "
                      f"{r['floor']:.4f} | {r['ratio']:.3f} | "
                      f"{'yes' if r['within'] else 'NO'} |")
        print()

    print("### per-seed X (CA), because a bifurcating seed moves the mean without moving "
          "the arm\n")
    for rung in rungs:
        for arm in arms:
            d = find(R, arm, rung)
            if d is None:
                continue
            r = row(d, "kabsch_rmsd")
            print(f"{rung:10s} {arm}  " + "  ".join(
                f"s{k}={v:.4f}" for k, v in r["per_seed"].items()))
    print()

    print("### route counters, so an arm that did not engage cannot be reported as one\n")
    print("| rung | arm | fp32_softmax calls | fused served | declined | too_short |")
    print("|---|---|--:|--:|--:|--:|")
    for rung in rungs:
        for arm in arms:
            d = find(R, arm, rung)
            if d is None or "flags" not in d:
                continue
            f = d["flags"]
            t = f["triatt_fused_hifi_stats"]
            print(f"| {rung} | {arm} | {f['fp32_softmax_stats']['calls']} | {t['served']} | "
                  f"{t['declined']} | {t['too_short']} |")
    print()

    if perf:
        print("### 512 aa, both readings\n")
        print("| arm | s/fold | vs shipped | whole fold vs H200 | device-only vs H200 | scope |")
        print("|---|--:|--:|--:|--:|---|")
        for arm in arms:
            s = perf.get(arm)
            if s is None:
                continue
            w, dv = readings(s)
            print(f"| {arm} | {s:.3f} | {A0_CELL / s:.2f}x | {w:.3f}x | {dv:.3f}x | "
                  f"{SCOPE[arm]} |")
        print()

    print("### routes\n")
    for arm in arms:
        print(f"* **{arm}** -- {ROUTE[arm]}  ({SCOPE[arm]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
