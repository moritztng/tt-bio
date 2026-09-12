#!/usr/bin/env python3
"""The chain histogram of the step's 339 BinaryNg programs, and what each chain costs.

Joins `site_map.py`'s recorded call list (names + source line + tensor identity) to
`b2z2-step-program-fusion`'s `site_cost_wh_c2.json` (a us figure per top-level ttnn call) BY
POSITION. The join is only legitimate if the two orderings are the same call, so it is checked:
the recorder does not wrap `ttnn.Tensor.__getitem__`, so those 30 entries are dropped from the
graph sequence first and then the two sequences must agree element for element or this exits.

A CHAIN is a maximal run of BinaryNg calls where the result of one is an operand of the next and
of nothing else. The distance between the two in dispatch order is reported, because a chain whose
links are 40 programs apart is not the same fusion problem as one whose links are adjacent.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BINARY = {"ttnn.add", "ttnn.add_", "ttnn.multiply", "ttnn.multiply_",
          "ttnn.subtract", "ttnn.subtract_"}
NO_DISPATCH = {"ttnn.deallocate"}


def tensors(entry):
    """serials of every tensor operand of a recorded call"""
    out = []
    for x in entry["in"]:
        if isinstance(x, dict) and "t" in x:
            out.append(x["t"])
    for v in entry["kw"].values():
        if isinstance(v, dict) and "t" in v:
            out.append(v["t"])
    return out


def out_serial(entry):
    o = entry["out"]
    if isinstance(o, dict) and "t" in o:
        return o["t"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=Path, required=True)
    ap.add_argument("--cost", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    sites = json.load(open(a.sites))
    calls = sites["calls"]
    cost = json.load(open(a.cost))

    # --- the join, and its check -------------------------------------------------------------
    graph = [r for r in cost["table"] if r["ttnn"] != "ttnn.Tensor.__getitem__"]
    if len(graph) != len(calls):
        raise SystemExit(f"length mismatch after dropping getitem: {len(graph)} vs {len(calls)}")
    bad = [(i, g["ttnn"], c["name"]) for i, (g, c) in enumerate(zip(graph, calls))
           if g["ttnn"] != c["name"]]
    if bad:
        raise SystemExit(f"sequence mismatch at {bad[:5]} ({len(bad)} total)")
    for g, c in zip(graph, calls):
        c["us"] = g.get("us")
        c["code"] = g.get("code")

    # --- dispatching calls only --------------------------------------------------------------
    disp = [c for c in calls if c["name"] not in NO_DISPATCH]
    for k, c in enumerate(disp):
        c["d"] = k
    bins = [c for c in disp if c["name"] in BINARY]
    print(f"{len(calls)} recorded calls, {len(disp)} dispatching, {len(bins)} BinaryNg")

    # --- who consumes what -------------------------------------------------------------------
    consumers = defaultdict(list)
    for c in disp:
        for t in tensors(c):
            consumers[t].append(c["d"])

    # --- chain links -------------------------------------------------------------------------
    by_d = {c["d"]: c for c in disp}
    nxt = {}
    for c in bins:
        o = out_serial(c)
        if o is None:
            continue
        later = [d for d in consumers[o] if d > c["d"]]
        if len(later) != 1:
            continue                       # no consumer, or more than one -> not a private chain
        succ = by_d[later[0]]
        if succ["name"] in BINARY:
            nxt[c["d"]] = (succ["d"], later[0] - c["d"])

    # --- maximal runs ------------------------------------------------------------------------
    heads = {c["d"] for c in bins} - {v[0] for v in nxt.values()}
    chains = []
    for h in sorted(heads):
        run, d = [h], h
        while d in nxt:
            d = nxt[d][0]
            run.append(d)
        chains.append(run)
    assert sum(len(c) for c in chains) == len(bins), (sum(len(c) for c in chains), len(bins))

    hist = Counter(len(c) for c in chains)
    print("\nCHAIN-LENGTH HISTOGRAM")
    print(f"{'len':>4} {'chains':>7} {'programs':>9} {'kernel ms':>10} {'us/prog':>8}")
    for L in sorted(hist):
        progs = [by_d[d] for c in chains if len(c) == L for d in c]
        ms = sum(p["us"] for p in progs) / 1e3
        print(f"{L:>4} {hist[L]:>7} {len(progs):>9} {ms:>10.4f} {ms*1e3/len(progs):>8.2f}")

    # --- what a chain looks like, by source site ---------------------------------------------
    def sig(d):
        c = by_d[d]
        act = [k for k in c["kw"] if "activations" in k or k == "activation"]
        actv = ",".join(str(c["kw"][k]) for k in act)
        return f"{c['name']}@{c['site'][0] if c['site'] else '?'}" + (f"[{actv}]" if actv else "")

    print("\nCHAIN SHAPES (source sites), by frequency")
    shapes = Counter(" -> ".join(sig(d) for d in c) for c in chains)
    tot_us = defaultdict(float)
    gaps = defaultdict(list)
    for c in chains:
        key = " -> ".join(sig(d) for d in c)
        tot_us[key] += sum(by_d[d]["us"] for d in c)
        gaps[key].extend(nxt[d][1] for d in c[:-1])
    for key, n in shapes.most_common():
        L = key.count("->") + 1
        g = gaps[key]
        gs = f" gap {min(g)}-{max(g)}" if g else ""
        print(f"  {n:4d}x len{L}  {tot_us[key]/1e3:7.4f} ms  "
              f"{tot_us[key]/n/L:6.2f} us/prog{gs}\n        {key}")

    # --- shapes of the operands, per chain shape ---------------------------------------------
    print("\nOPERAND SHAPES per chain shape")
    seen = set()
    for c in chains:
        key = " -> ".join(sig(d) for d in c)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {key}")
        for d in c:
            e = by_d[d]
            ins = [f"{x.get('shape')}/{str(x.get('dtype','')).split('.')[-1]}"
                   for x in e["in"] if "t" in x]
            print(f"      {e['name']:20s} {e['us']:7.2f} us  in={ins}  kw={e['kw']}")

    if a.out:
        a.out.write_text(json.dumps({
            "n_binary": len(bins), "histogram": {str(k): v for k, v in sorted(hist.items())},
            "chain_shapes": {k: {"n": v, "ms": round(tot_us[k] / 1e3, 4),
                                 "len": k.count("->") + 1} for k, v in shapes.items()},
            "chains": [[{"d": d, "site": by_d[d]["site"][0], "name": by_d[d]["name"],
                         "us": by_d[d]["us"]} for d in c] for c in chains],
        }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
