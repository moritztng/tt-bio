#!/usr/bin/env python3
"""Per-phase iptm/ptm band for a BindCraft 2 trajectory, from its own losses.csv.

BindCraft 2 writes one row per round to
`1_Trajectories/<design>/<design>_losses.csv` with a `phase` column, so the band a
stage really traversed is on disk even though the stage line prints only the final
value. The mutate band is the one the campaign compares across arms: `run_mutation_polish`
ranks by `interface_iptm`, so a collapsed iptm makes those 15 rounds an unguided walk, and
you cannot see a collapse from the stage line alone.

The stage line is a best-of-N over the stage, not the round it prints beside: the pc
reference finished screen at iptm 0.22 and its stage line reads 0.87, which is screen's
maximum over 50 rounds. `mutate` is the exception, its line being the final round. So
`final` is printed here next to the band, and two arms' stage lines are not a paired
comparison.

`plddt_loss` is not `1 - pLDDT` (0.97-0.99 against stage lines of 0.93-0.95) and the
mutate rows carry no loss components at all, so only iptm and ptm are read here.

    python3 perf/bcx_shipped/band.py <project-dir> [<project-dir> ...]
"""
import csv
import sys
from collections import OrderedDict
from pathlib import Path


def bands(losses: Path):
    rows = list(csv.DictReader(losses.open()))
    if not rows:
        return None, {}
    iptm = next((k for k in rows[0] if k.endswith(".iptm")), None)
    ptm = next((k for k in rows[0] if k.endswith(".ptm")), None)
    phases = OrderedDict()
    for r in rows:
        phases.setdefault(r.get("phase", "?"), []).append(r)
    out = OrderedDict()
    for p, rs in phases.items():
        out[p] = ([float(r[iptm]) for r in rs], [float(r[ptm]) for r in rs])
    return len(rows), out


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    for proj in argv[1:]:
        for d in sorted(Path(proj).glob("1_Trajectories/*/")):
            losses = next(iter(d.glob("*_losses.csv")), None)
            if losses is None:
                continue
            n, out = bands(losses)
            print(f"\n{d.name}   {n} rounds   {losses}")
            for p, (ip, pt) in out.items():
                print(f"  {p:8s} n={len(ip):3d}  iptm {min(ip):.2f}-{max(ip):.2f} "
                      f"final {ip[-1]:.4f}   ptm {min(pt):.2f}-{max(pt):.2f} final {pt[-1]:.4f}")
            for p, (ip, pt) in out.items():
                if p in ("mutate", "mutation", "polish"):
                    print(f"  {p} iptm  " + "  ".join(f"{v:.2f}" for v in ip))
                    print(f"  {p} ptm   " + "  ".join(f"{v:.2f}" for v in pt))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
