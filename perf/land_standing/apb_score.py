#!/usr/bin/env python3
"""Score the apb_seedfloor folds: is the lever's 1.254 A a structural move or a re-ranking?

Three questions, three tables.

1. Per sample, RMSD against the experimental structure. The writer emits prot.cif for the
   rank-0 (confidence-selected) sample and prot_model_1..4 for the rest, so sample index is
   confidence rank and prot.cif is what a user receives. The off:s0 row is a control on the
   instrument: it must read 1.658 / 2.929 / 1.652 / 1.700 / 3.620, the release gate's own
   openfold3 table.

2. Sample-to-sample CA RMSD between the off and on arms AT THE SAME SEED. If the lever only
   changes the confidence head's ordering, each on-sample has an off-sample it is ~0 A from
   and the assignment is a permutation. If the lever moves the structures, no such pairing
   exists.

3. The seed floor, measured here rather than borrowed: CA RMSD between the delivered
   structures of the off arm at four seeds, and the spread of their RMSD-vs-native. The
   lever's effect is quoted against that, not against a floor taken from another model.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/land-standing")
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "perf/of3t_rankunify"))

from ca_rmsd import ca_rmsd  # noqa: E402

OUT = WT / "perf/land_standing/out/apb_seedfloor"
NATIVE = WT / "examples/ground_truth_structures/prot.cif"
RUNS = json.loads((OUT / "runs.json").read_text())


def samples(run_dir):
    """The five CIFs of one fold, keyed by confidence rank. Rank 0 is the delivered structure."""
    d = Path(run_dir) / "openfold3_results_prot" / "structures"
    out = {0: d / "prot.cif"}
    for i in range(1, 5):
        out[i] = d / ("prot_model_%d.cif" % i)
    missing = [str(p) for p in out.values() if not p.is_file()]
    if missing:
        sys.exit("missing CIFs: %s" % missing)
    return out


def main():
    names = list(RUNS)
    cif = {n: samples(RUNS[n]) for n in names}
    native = {n: {i: ca_rmsd(str(p), str(NATIVE)) for i, p in cif[n].items()} for n in names}

    print("=" * 78)
    print("1. RMSD vs 7ROA by confidence rank. Rank 0 is the structure the user receives.")
    print("=" * 78)
    print("  %-8s %8s %8s %8s %8s %8s   delivered" % ("run", "rank0", "rank1", "rank2", "rank3", "rank4"))
    for n in names:
        v = native[n]
        print("  %-8s %8.3f %8.3f %8.3f %8.3f %8.3f   %8.3f"
              % (n, v[0], v[1], v[2], v[3], v[4], v[0]))
    print("  control: off:s0 must read 1.658 2.929 1.652 1.700 3.620 (the gate's own table)")

    print()
    print("=" * 78)
    print("2. off vs on at the same seed: does every on-sample have a near-zero off twin?")
    print("=" * 78)
    for seed in ("0", "1"):
        a, b = "off:s" + seed, "on:s" + seed
        if a not in names or b not in names:
            continue
        print("  seed %s  (rows: off rank -> nearest on rank)" % seed)
        pairs = []
        for i in range(5):
            dist = {j: ca_rmsd(str(cif[a][i]), str(cif[b][j])) for j in range(5)}
            twin = min(dist, key=dist.get)
            pairs.append((i, twin, dist[twin]))
            second = sorted(dist.values())[1]
            print("    off rank %d -> on rank %d   %8.4f A   (next nearest %.4f A)"
                  % (i, twin, dist[twin], second))
        mapped = [t for _, t, _ in pairs]
        print("    permutation: %s   worst twin distance %.4f A"
              % (len(set(mapped)) == len(mapped), max(d for _, _, d in pairs)))
        print("    delivered off %.3f A vs delivered on %.3f A, the two delivered structures "
              "being %.4f A apart"
              % (native[a][0], native[b][0], ca_rmsd(str(cif[a][0]), str(cif[b][0]))))

    print()
    print("=" * 78)
    print("3. Seed floor, lever off, measured in this session")
    print("=" * 78)
    off = sorted(n for n in names if n.startswith("off:"))
    floor = []
    for x, y in itertools.combinations(off, 2):
        v = ca_rmsd(str(cif[x][0]), str(cif[y][0]))
        floor.append(v)
        print("    %-8s vs %-8s  %8.4f A" % (x, y, v))
    natives = [native[n][0] for n in off]
    print("  seed floor (delivered structure, seed to seed): n=%d  %.4f..%.4f A  mean %.4f A"
          % (len(floor), min(floor), max(floor), sum(floor) / len(floor)))
    print("  delivered RMSD vs 7ROA across seeds, lever off: %.3f..%.3f A"
          % (min(natives), max(natives)))

    json.dump({"native": {n: {str(k): v for k, v in native[n].items()} for n in names},
               "seed_floor": floor}, open(OUT / "score.json", "w"), indent=2)
    print("\nwrote %s" % (OUT / "score.json"))


if __name__ == "__main__":
    main()
