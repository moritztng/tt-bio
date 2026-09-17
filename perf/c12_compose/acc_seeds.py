#!/usr/bin/env python3
"""Split the stack's accuracy cost between its two levers, on plDDT, across seeds.

The campaign read silu's 512 aa plDDT deficit as -0.016657, "5.82x that fixture's plDDT seed
floor", and concluded silu carried 94 % of the stack's accuracy cost. Both halves of that come from
n=1: the deficit is one seed's draw and the floor is one seed PAIR. This scores every arm at five
seeds against the base's own across-seed scatter, which is what "floor" has to mean when the
quantity is a per-seed draw.

plDDT is the right metric on cdk2x2_512 and RMSD is not: the fixture's unconstrained hinge
saturates whole-molecule RMSD, and the base-against-base seed floor here runs 5.35 to 21.91 A
across the ten seed pairs, so no 0.4 A lever is resolvable against it. plDDT carries no frame, so
it cannot be confounded by which basin the trajectory chose.

    acc_seeds.py perf/c12_compose/out/acc512_seeds.json
"""
import json
import statistics as st
import sys
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def stat(v):
    m = st.mean(v)
    sd = st.stdev(v)
    se = sd / len(v) ** 0.5
    t = T95[len(v) - 1]
    return m, sd, t * se, abs(m) > t * se


def main() -> int:
    d = json.loads(Path(sys.argv[1]).read_text())
    seeds, arms = d["seeds"], d["arms"]
    p = {(f["seed"], f["arm"]): f["plddt"]
         for k, f in d["folds"].items() if not k.endswith("_aa")}

    base = [p[(s, "base")] for s in seeds]
    bm, bsd, bci, _ = stat(base)
    print(f"size {d['size']} aa, {len(seeds)} seeds, clock {d['clock_forced_mhz']} MHz forced\n")
    print("plDDT of the SHIPPED DEFAULT across seeds -- this is the floor any arm is read against")
    print("  " + "  ".join(f"s{s} {p[(s, 'base')]:.6f}" for s in seeds))
    print(f"  mean {bm:.6f}  sd {bsd:.6f}  95% CI +/-{bci:.6f}\n")

    print("per-arm plDDT delta against base AT THE SAME SEED")
    print(f"{'arm':>7} " + " ".join(f"{'s%d' % s:>10}" for s in seeds)
          + f" {'mean':>10} {'sd':>9} {'95% CI':>10}  resolved")
    for a in arms:
        if a == "base":
            continue
        v = [p[(s, a)] - p[(s, "base")] for s in seeds]
        m, sd, ci, res = stat(v)
        print(f"{a:>7} " + " ".join(f"{x:>+10.6f}" for x in v)
              + f" {m:>+10.6f} {sd:>9.6f} {ci:>+10.6f}  {'YES' if res else 'no'}")

    print("\nthe n=1 readings this replaces")
    s0 = seeds[0]
    for a in arms:
        if a == "base":
            continue
        print(f"  {a:>6} at seed {s0} alone: {p[(s0, a)] - p[(s0, 'base')]:+.6f}")
    print(f"  seed floor from the single pair s{seeds[0]}/s{seeds[1]}: "
          f"{p[(seeds[1], 'base')] - p[(seeds[0], 'base')]:+.6f}")
    print(f"  the base's ACTUAL across-seed sd: {bsd:.6f}  "
          f"({bsd / abs(p[(seeds[1], 'base')] - p[(seeds[0], 'base')]):.2f}x that single pair)")

    fl = [v["all_atom_A"] for k, v in d["all_pairs"].items()
          if k.count("base") == 2 and "_aa" not in k]
    stack = [v["all_atom_A"] for v in d["stack_paired_per_seed"]]
    print(f"\nwhole-molecule RMSD on this fixture, reported and NOT used as the bar")
    print(f"  base-vs-base seed floor over {len(fl)} pairs: "
          f"{min(fl):.4f} to {max(fl):.4f} A, mean {st.mean(fl):.4f}")
    print(f"  stack paired per seed:                   "
          f"{min(stack):.4f} to {max(stack):.4f} A, mean {st.mean(stack):.4f}")
    print(f"  stack mean / floor mean: {st.mean(stack) / st.mean(fl):.3f}x")
    aa = d["aa_control"]
    print(f"  A/A control: {aa['all_atom_A']:.4f} A, digests identical={aa['digests_identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
