#!/usr/bin/env python3
"""Per-phase distinct-value census of a BindCraft 2 *_losses.csv, with the file's own resolution.

The order matters and is the point. A constant metric across varied inputs is evidence about the
mechanism ONLY if the input variation is large enough to show at the recording precision;
otherwise it is a fact about the file format. This row was opened on "pTM is 0.59 for all fifteen
mutate rounds", which is what a WORKING loop writes: the file carries two decimals, so its
rounding step is 0.005, and a single-residue move is ~0.001.

So this prints the resolution first, then the census, then the negative control -- the longest
constant run that occurs in a phase the trajectory PASSED.

    python3 phase_census.py <losses.csv> [...]
"""
import csv
import sys
from collections import OrderedDict

METRICS = ("iptm", "ptm", "plddt_loss")


def decimals(v):
    return len(v.split(".")[1]) if v and "." in v else 0


def census(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"{path}: empty")
        return
    cols = {m: next((k for k in rows[0] if k.endswith("." + m)), None) for m in METRICS}

    dp = max((decimals(v) for r in rows for k, v in r.items()
              if k not in ("phase", "round")), default=0)
    step = 0.5 * 10 ** -dp if dp else 0.0
    print(f"\n{path}")
    print(f"  resolution: {dp} decimals -> spacing {10**-dp:g}, rounding step {step:g}")
    print("  a single-residue move is ~0.001, so constancy below this step says nothing")

    by = OrderedDict()
    for r in rows:
        by.setdefault(r["phase"], []).append(r)

    hdr = "  " + f"{'phase':10s}{'rounds':>7s}"
    for m in METRICS:
        hdr += f"{m + ' dist':>17s}{m + ' range':>18s}{'longest flat':>14s}"
    print(hdr)
    for ph, rs in by.items():
        line = "  " + f"{ph:10s}{len(rs):7d}"
        for m in METRICS:
            c = cols[m]
            vals = [r[c] for r in rs] if c else []
            vals = [v for v in vals if v != ""]
            if not vals:
                line += f"{'-':>17s}{'-':>18s}{'-':>14s}"
                continue
            run = best = 1
            for a, b in zip(vals, vals[1:]):
                run = run + 1 if a == b else 1
                best = max(best, run)
            f = [float(v) for v in vals]
            line += (f"{len(set(vals)):>17d}"
                     f"{f'{min(f):.3f}-{max(f):.3f}':>18s}"
                     f"{best:>14d}")
        print(line)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        census(p)
