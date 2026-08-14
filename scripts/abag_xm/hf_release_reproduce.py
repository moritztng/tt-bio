#!/usr/bin/env python3
"""Reproduce the dataset card's headline table from the PUBLISHED Hub data, using the
PUBLISHED analysis code rather than a reimplementation of it.

    python3 hf_release_reproduce.py --analysis /path/to/abag-scaling/analysis

`core.rank_order` resolves selector ties AGAINST the selector. That is not a detail:
esmfold2's pLDDT selector is quantised to 4 dp (~181 distinct values per 512, a top tie on
20 of 161 scorable targets), and breaking those ties the other way moves delivered@512 from
0.2852 to 0.2878. The card states the convention for exactly this reason.
"""
import argparse, sys
import numpy as np
from datasets import load_dataset

REPO = "Tenstorrent/abag-xm"
# (targets, random, oracle@16, oracle@512, delivered@16, delivered@512) as the card prints them
CARD = {
    "boltz2":       (161, 0.2246, 0.3370, 0.4480, 0.2449, 0.2514),
    "opendde-abag": (160, 0.4991, 0.5613, 0.6211, 0.5047, 0.4978),
    "protenix-v2":  (161, 0.3009, 0.4164, 0.5673, 0.3201, 0.3191),
    "esmfold2":     (161, 0.2512, 0.3387, 0.4431, 0.2795, 0.2852),
}
TOL = 5e-4  # the card prints 4 dp, so this is the rounding half-width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True,
                    help="path to abag-scaling/analysis (for core.py)")
    args = ap.parse_args()
    sys.path.insert(0, args.analysis)
    import core

    df = load_dataset(REPO, split="train").to_pandas()
    print(f"{'model':<14}{'targets':>8}{'random':>9}{'orc@16':>9}{'orc@512':>9}"
          f"{'del@16':>9}{'del@512':>9}   verdict")
    bad, worst = [], 0.0
    for model, card in CARD.items():
        sub = df[(df["model"] == model) & df["dockq"].notna()]
        rand, oc, dc = [], [], []
        for _, g in sub.groupby("target"):
            g = g.sort_values("rank")
            rand.append(g["dockq"].mean())
            oc.append(core.oracle_curves(g))
            dc.append(core.selector_curves(g, "selector"))
        oc, dc = np.array(oc), np.array(dc)
        got = (len(rand), float(np.mean(rand)), float(oc[:, 15].mean()),
               float(oc[:, -1].mean()), float(dc[:, 15].mean()), float(dc[:, -1].mean()))
        dmax = max(abs(g - c) for g, c in zip(got[1:], card[1:]))
        worst = max(worst, dmax)
        ok = got[0] == card[0] and dmax < TOL
        if not ok:
            bad.append(model)
        print(f"{model:<14}{got[0]:>8}{got[1]:>9.4f}{got[2]:>9.4f}{got[3]:>9.4f}"
              f"{got[4]:>9.4f}{got[5]:>9.4f}   {'OK' if ok else 'MISMATCH'}  max|d|={dmax:.2e}")

    print(f"\nworst deviation from the card: {worst:.2e}")
    print("ALL FOUR MODELS REPRODUCE" if not bad else f"MISMATCH: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
