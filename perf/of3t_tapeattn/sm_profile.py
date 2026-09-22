#!/usr/bin/env python3
"""Where along the backward does each verb inject, and does that profile move with width?

The cotangent ladder says the 384-over-64 growth is created in the first three blocks the
backward touches and then decays over the remaining forty-five. The per-verb injection table
says softmax is the only verb above the elementwise bf16 floor and the only one that grows with
width. Those two readings are only the same story if softmax's injection is concentrated at the
top of the backward. This reads that off the census nodes already captured, by ordinal, with no
device run.

`sm_ord` is the number of single-track softmax backwards already fired, so ordinal 0 is the
deepest block (47) and ordinal 47 is block 0.

    sm_profile.py --side64 SIDE_C64.json --side384 SIDE_C384.json --out SMPROFILE.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def load(path):
    return json.load(open(path))["nodes"]


def profile(nodes, verb, key="own"):
    """rel_l2 of each firing of `verb`, bucketed by softmax ordinal."""
    by_ord = defaultdict(lambda: {"num": 0.0, "den": 0.0, "n": 0, "cos_min": 1.0})
    for rec in nodes:
        if rec["verb"] != verb:
            continue
        o = rec["sm_ord"]
        for par in rec["parents"]:
            t = par.get(key)
            if not t or t.get("ref_norm") in (None, 0.0):
                continue
            b = by_ord[o]
            b["num"] += (t["rel_l2"] * t["ref_norm"]) ** 2
            b["den"] += t["ref_norm"] ** 2
            b["n"] += 1
            if t.get("cos") is not None:
                b["cos_min"] = min(b["cos_min"], t["cos"])
    out = {}
    for o, b in sorted(by_ord.items()):
        if b["den"]:
            out[o] = {"rel_l2": (b["num"] / b["den"]) ** 0.5, "n": b["n"],
                      "cos_min": b["cos_min"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side64", required=True)
    ap.add_argument("--side384", required=True)
    ap.add_argument("--verb", action="append", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n64, n384 = load(a.side64), load(a.side384)
    verbs = a.verb or sorted({r["verb"] for r in n64} & {r["verb"] for r in n384})

    body = {"what": __doc__.strip().splitlines()[0],
            "ordinal": "sm_ord = single-track softmax backwards already fired; 0 is block 47, "
                       "47 is block 0",
            "verbs": {}}
    for v in verbs:
        p64, p384 = profile(n64, v), profile(n384, v)
        rows = {}
        for o in sorted(set(p64) & set(p384)):
            r64, r384 = p64[o]["rel_l2"], p384[o]["rel_l2"]
            rows[str(o)] = {"rel_l2_64": r64, "rel_l2_384": r384,
                            "ratio": (r384 / r64) if r64 else None,
                            "cos_min_64": p64[o]["cos_min"],
                            "cos_min_384": p384[o]["cos_min"],
                            "firings": p64[o]["n"]}
        if rows:
            body["verbs"][v] = rows
    json.dump(body, open(a.out, "w"), indent=1)

    for v, rows in body["verbs"].items():
        ks = sorted(rows, key=int)
        head = [rows[k]["ratio"] for k in ks[:3] if rows[k]["ratio"]]
        tail = [rows[k]["ratio"] for k in ks[6:] if rows[k]["ratio"]]
        print("%-20s ordinals %2d  ratio@0-2 %s  median ratio from ordinal 6 %.4f"
              % (v, len(ks), " ".join("%.4f" % x for x in head),
                 sorted(tail)[len(tail) // 2] if tail else float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
