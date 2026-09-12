#!/usr/bin/env python3
"""Every number this row quotes, printed from the committed session JSON and nothing else.

One rule: no figure in the state doc may be typed by hand. Two rows of this campaign had to
withdraw a ratio because it divided by a denominator from a different session, so the denominator
here is always the `base` arm of the SAME interleaved session.

    python3 perf/b2z2_bothshards/closing.py perf/b2z2_bothshards/b2z2_union_mesh2.json
"""
import json
import statistics as st
import sys
from pathlib import Path


def med(xs):
    return st.median(xs) if xs else None


def main(path):
    d = json.loads(Path(path).read_text())
    reps = d["reps"]
    arms = list(dict.fromkeys(r["arm"] for r in reps))
    fold = {a: [r["fold_s"] for r in reps if r["arm"] == a] for a in arms}
    trunk = {a: [r["stages"]["trunk_s"] for r in reps
                 if r["arm"] == a and r["stages"].get("trunk_s")] for a in arms}
    samp = {a: [r["stages"]["sampler_s"] for r in reps
                if r["arm"] == a and r["stages"].get("sampler_s")] for a in arms}

    print(f"session {d['tag']}  commit {d.get('commit')}  mesh 1x{d['mesh']}  "
          f"trace={d['trace']}  benchlocked={d.get('benchlocked')}  card {d.get('card')}")
    print(f"loadavg {d.get('loadavg_start')} -> {d.get('loadavg_end')}")
    print()

    # PARITY FIRST. A ratio from an arm that wrote a different structure is not a result.
    digests = sorted({c for r in reps for c in r["cif"]})
    print("PARITY")
    for a in arms:
        ds = sorted({c for r in reps if r["arm"] == a for c in r["cif"]})
        rung = sorted({tuple(sorted(r["rungs"].items())) for r in reps if r["arm"] == a})
        nr = max(sum(v or 0 for _k, v in x) for x in rung) if rung else 0
        sh = [r["atom_shard"] for r in reps if r["arm"] == a]
        eng = sorted({s["sharded"] for s in sh}) if sh and "sharded" in sh[0] else ["?"]
        print(f"  {a:6s} n={len(fold[a]):2d}  cif={ds}  config_rungs_max={nr}  "
              f"atom_shard_calls={eng}")
    print(f"  ALL ARMS BIT-IDENTICAL: {len(digests) == 1}  ({digests})")
    print()

    if "base" not in arms:
        return 0

    # The A/A floor: the base arm against itself. The reps alternate, so the odd and the even
    # occurrences of `base` are two independent estimates of the same quantity taken over the same
    # interval. Their ratio is what this session can resolve; anything smaller is not a measurement.
    b = fold["base"]
    aa = max(med(b[0::2]) / med(b[1::2]), med(b[1::2]) / med(b[0::2])) if len(b) >= 4 else None
    print("A/A FLOOR")
    print(f"  base folds {[round(x, 3) for x in b]}")
    print(f"  A/A = {aa:.5f}x" if aa else "  A/A = (needs >= 4 base reps)")
    print(f"  spread = {100 * (max(b) - min(b)) / med(b):.2f} % of the median")
    print()

    print("RATIOS, each against the base arm of THIS session")
    print(f"  {'arm':6s} {'fold s':>9s} {'fold x':>9s} {'trunk s':>9s} {'trunk x':>9s} "
          f"{'sampler s':>10s} {'sampler x':>10s}")
    r = {}
    for a in arms:
        fx = med(fold["base"]) / med(fold[a])
        tx = med(trunk["base"]) / med(trunk[a]) if trunk[a] and trunk["base"] else float("nan")
        sx = med(samp["base"]) / med(samp[a]) if samp[a] and samp["base"] else float("nan")
        r[a] = fx
        print(f"  {a:6s} {med(fold[a]):9.4f} {fx:9.5f} {med(trunk[a] or [float('nan')]):9.4f} "
              f"{tx:9.5f} {med(samp[a] or [float('nan')]):10.4f} {sx:10.5f}")
    print()

    if {"trunk", "atom", "both"} <= set(arms):
        prod = r["trunk"] * r["atom"]
        disc = r["both"] / prod
        print("ADDITIVITY")
        print(f"  product of the halves  {r['trunk']:.5f} x {r['atom']:.5f} = {prod:.5f}x")
        print(f"  measured union                            {r['both']:.5f}x")
        print(f"  ADDITIVITY-DISCOUNT    {100 * (disc - 1):+.3f} %"
              f"  (union / product = {disc:.5f})")
        if aa:
            fires = disc < 1 / aa
            print(f"  A/A floor is {100 * (aa - 1):.3f} %, so the pre-registered falsifier "
                  f"{'FIRES' if fires else 'does NOT fire'}: the two shards "
                  f"{'contend' if fires else 'do not detectably contend'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "perf/b2z2_bothshards/b2z2_union_mesh2.json"))
