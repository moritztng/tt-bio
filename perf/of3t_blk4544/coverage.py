#!/usr/bin/env python3
"""How much of a taped backward does a pin selector actually reach?

A pin that reaches 6.88 % of the backward nodes cannot refute "the carrier is a taped verb", and
no output comparison says which of those two a given arm was. `all` reads like a saturation test
and is not one, so its coverage is measured off a sidecar's own backward census rather than
argued from the selector source (D196: a selector written against the wrong object is silently
dead, and the same mechanism makes a selector look broader than it is).

    coverage.py <SIDE_*.json> [--out COVERAGE.json]

The sidecar's `backward` map is `"<caller> <shape>" -> firings` over one whole taped backward,
all 48 blocks, so the shares below are per backward and not per block.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))   # the worktree root, for tt_bio
import pinvjp as P                                              # noqa: E402


def parse(k):
    i = k.find(" ")
    return k[:i], ast.literal_eval(k[i + 1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("side")
    ap.add_argument("--out")
    a = ap.parse_args()
    bw = json.load(open(a.side))["backward"]

    names = sorted(P.SELECT)
    tot = 0
    hits = {n: 0 for n in names}
    refable = 0
    rows = []
    for k, n in bw.items():
        caller, shp = parse(k)
        tot += n
        hr = caller in P._REF
        refable += n if hr else 0
        sel = []
        for nm in names:
            try:
                ok = bool(P.SELECT[nm](caller, shp, None))
            except Exception:                                    # noqa: BLE001
                ok = False
            if ok:
                hits[nm] += n
                sel.append(nm)
        rows.append({"caller": caller, "shape": shp, "firings": n,
                     "has_exact_vjp": hr, "selected_by": sel})
    rows.sort(key=lambda r: -r["firings"])

    out = {"side": a.side,
           "backward_firings_total": tot,
           "distinct_nodes": len(bw),
           "firings_with_an_exact_vjp": refable,
           "pct_with_an_exact_vjp": round(100.0 * refable / tot, 4),
           "coverage": {n: {"firings": hits[n], "pct": round(100.0 * hits[n] / tot, 4)}
                        for n in names},
           "unreached_but_refable": [r for r in rows
                                     if r["has_exact_vjp"] and not r["selected_by"]],
           "nodes": rows}
    print("total backward firings %d over %d distinct nodes; %d (%.2f %%) have an exact VJP"
          % (tot, len(bw), refable, 100.0 * refable / tot))
    for n in names:
        print("  %-14s %6d  %6.2f %%" % (n, hits[n], 100.0 * hits[n] / tot))
    print("refable but unreached by `all`:")
    for r in rows:
        if r["has_exact_vjp"] and "all" not in r["selected_by"]:
            print("  %6d  %s %s" % (r["firings"], r["caller"], r["shape"]))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
