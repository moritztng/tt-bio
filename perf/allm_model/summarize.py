#!/usr/bin/env python3
"""The block table for one blockcensus.py capture.

Two denominators, both printed. The TAPED fold is the wall the blocks sum to, because the
tape's syncs are inside it. The UNTAPED folds bracketing it are what the fold actually costs,
and their spread is the A/A floor any share here has to clear.
"""
import json
import sys
from pathlib import Path


def main():
    d = json.load(open(sys.argv[1]))
    folds = d["folds"]
    un = [f["fold_s"] for f in folds if f["tag"] in ("A", "C")]
    aa = (max(un) - min(un)) / min(un) * 100 if len(un) > 1 else float("nan")
    clk = [f["clock"].get("aiclk_median") for f in folds if f["clock"].get("aiclk_n")]
    print(f"{d['model']}  size={d['size']}  card={d['card']}  host={d['host']}  "
          f"git={d['git_head'][:9]}")
    print(f"  AICLK held {d['aiclk_held_mhz']} MHz, during-sampled median per fold: {clk}")
    print(f"  untaped folds {un}  ->  A/A floor {aa:.2f} %")
    for f in folds:
        if "blocks" not in f:
            continue
        tot = f["fold_s"]
        rows = sorted(f["blocks"].items(), key=lambda kv: -kv[1]["self_s"])
        print(f"\n  [{f['tag']}] depth={f['depth']} taped fold {tot:.4f} s "
              f"(tape charge {100 * (tot - min(un)) / min(un):+.2f} % on the untaped best)")
        s = 0.0
        for k, v in rows:
            if v["self_s"] < 0.002 * tot:
                continue
            print(f"    {v['self_s']:8.4f} s  {100 * v['self_s'] / tot:6.2f} %  "
                  f"n={v['n']:6d}  incl={v['incl_s']:8.4f}  {k}")
        s = sum(v["self_s"] for v in f["blocks"].values())
        print(f"    {'-' * 60}\n    {s:8.4f} s  {100 * s / tot:6.2f} %  taped blocks, self")
        print(f"    {tot - s:8.4f} s  {100 * (tot - s) / tot:6.2f} %  residual: host work "
              f"(featurisation, structure write) plus device work no block wraps")


if __name__ == "__main__":
    main()
