#!/usr/bin/env python3
"""Read a fold_ab.py A/B artifact and report the numbers a landing decision needs.

Three things the harness's own summary does not do: a permutation test over the block medians
(the block is the independent unit, not the fold -- each block is its own process, so folds inside
one share a program cache and a warmup), the per-fold clock and load beside every median, and the
structure digests, which say whether the base arm reproduced the shipped default bit for bit.
"""
import itertools
import json
import statistics
import sys
from collections import defaultdict


def main(path: str) -> int:
    d = json.load(open(path))
    per_arm = defaultdict(list)          # arm -> [block median]
    folds = defaultdict(list)            # arm -> [fold_s]
    clocks, loads, digests, nodes, foreign = [], [], defaultdict(set), set(), set()
    for b in d["blocks"]:
        r = b.get("result") or {}
        fs = [f["fold_s"] for f in r.get("folds", [])]
        if not fs:
            print(f"block {b['arm']}/{b['block']}: NO FOLDS (rc={b.get('returncode')})")
            continue
        per_arm[b["arm"]].append(statistics.median(fs))
        folds[b["arm"]] += fs
        for f in r["folds"]:
            c = f.get("clock", {})
            if c.get("aiclk_n"):
                clocks += [c["aiclk_min"], c["aiclk_max"]]
            loads.append(f.get("loadavg1"))
            nodes.add(f.get("clock_node"))
            digests[b["arm"]].add(f.get("cif_sha256"))
            for h in (f.get("foreign_device_holders") or []) + (f.get("foreign_device_holders_end") or []):
                foreign.add((h["pid"], h["cwd"]))

    base, on = per_arm["base"], per_arm["on"]
    mb, mo = statistics.median(folds["base"]), statistics.median(folds["on"])
    print(f"base block medians {[round(x, 3) for x in base]}")
    print(f"on   block medians {[round(x, 3) for x in on]}")
    print(f"base {mb:.3f}s (n={len(folds['base'])})  on {mo:.3f}s (n={len(folds['on'])})  "
          f"delta {mb - mo:+.4f}s  ratio {mb / mo:.5f}")

    # A/A floor: the same comparison the A/B makes, but base against base. Split the base blocks
    # every way into two halves and take the widest median-of-halves ratio.
    def floor(xs):
        worst = 1.0
        for k in range(1, len(xs)):
            for left in itertools.combinations(range(len(xs)), k):
                a = [xs[i] for i in left]
                b_ = [xs[i] for i in range(len(xs)) if i not in left]
                r = statistics.median(a) / statistics.median(b_)
                worst = max(worst, r, 1 / r)
        return worst
    fb, fo = floor(base), floor(on)
    print(f"A/A floor  base {fb:.5f}  on {fo:.5f}   effect/floor {(mb / mo - 1) / (max(fb, fo) - 1):.2f}x")

    # Permutation over the block medians: how often does a random relabelling separate the arms
    # this far? Two-sided on the difference of medians.
    pool = base + on
    obs = abs(statistics.median(base) - statistics.median(on))
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), len(base)):
        a = [pool[i] for i in idx]
        b_ = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1
        hits += abs(statistics.median(a) - statistics.median(b_)) >= obs - 1e-12
    print(f"permutation over block medians: p = {hits}/{tot} = {hits / tot:.4f}")

    sep = max(folds["on"]) < min(folds["base"])
    print(f"complete separation of every fold: {sep}  "
          f"(base {min(folds['base']):.3f}-{max(folds['base']):.3f}, "
          f"on {min(folds['on']):.3f}-{max(folds['on']):.3f})")
    print(f"clock during folds: {min(clocks)}-{max(clocks)} MHz on node(s) {sorted(nodes)}")
    print(f"loadavg1 during folds: {min(loads):.2f}-{max(loads):.2f}")
    print(f"foreign device holders seen during folds: {sorted(foreign) or 'none'}")
    for arm in ("base", "on"):
        print(f"{arm} digests: {sorted(digests[arm])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
