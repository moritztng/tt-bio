#!/usr/bin/env python3
"""of3t-sigma step 1b: locate draw 3's trunk excess by a difference of absolute errors.

Per draw k, on the scored set: e_k = ours - f64, b_k = bf16 - f64, g_k = f64.
The trunk forward and the distogram loss do not depend on the draw, so what they contribute to
a trunk gradient (and to its error) is common to every draw; e_3 - e_1 is carried only by the
draw-dependent losses (diffusion through z/s, confidence heads through pred_xyz).

Control first: the distogram head's own parameters see only draw-independent paths, so
g_3 == g_1 and e_3 == e_1 there if both sides are deterministic.

    carrier.py --base 1 --draw 3 [--draw 2 --draw 4] --out CARRIER.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_fullstep64"))
from score import head_of, section_of  # noqa: E402

D = Path("/home/ttuser/of3t_sigma")
ap = argparse.ArgumentParser()
ap.add_argument("--base", type=int, default=1)
ap.add_argument("--draw", type=int, action="append", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def n2(ts):
    return sum(float((t * t).sum()) for t in ts)


def load(k):
    d = torch.load(D / f"scored_s{k}.pt", weights_only=False)
    return d["f64"], d["ours"], d["bf16"]


def groups(keys):
    g = {"trunk": [k for k in keys if head_of(k) == "trunk"],
         "distogram_head": [k for k in keys if head_of(k) == "distogram"]}
    for k in g["trunk"]:
        g.setdefault("trunk:" + section_of(k), []).append(k)
    return g


g1, o1, b1 = load(a.base)
rep = {"base": a.base, "pairs": {}}
for k in a.draw:
    gk, ok, bk = load(k)
    keys = sorted(set(gk) & set(g1))
    rows = {}
    for name, ks in groups(keys).items():
        e1 = [o1[x] - g1[x] for x in ks]; ek = [ok[x] - gk[x] for x in ks]
        c1 = [b1[x] - g1[x] for x in ks]; ck = [bk[x] - gk[x] for x in ks]
        r = {"n": len(ks),
             "g_k": n2(gk[x] for x in ks) ** .5, "g_base": n2(g1[x] for x in ks) ** .5,
             "g_k_minus_g_base": n2(gk[x] - g1[x] for x in ks) ** .5,
             "ours_e_k": n2(ek) ** .5, "ours_e_base": n2(e1) ** .5,
             "ours_e_k_minus_e_base": n2(a_ - b_ for a_, b_ in zip(ek, e1)) ** .5,
             "bf16_e_k": n2(ck) ** .5, "bf16_e_base": n2(c1) ** .5,
             "bf16_e_k_minus_e_base": n2(a_ - b_ for a_, b_ in zip(ck, c1)) ** .5}
        for s in ("ours", "bf16"):
            r[s + "_rel_k"] = r[s + "_e_k"] / r["g_k"] if r["g_k"] else None
            r[s + "_diff_share"] = (r[s + "_e_k_minus_e_base"] / r[s + "_e_k"]) if r[s + "_e_k"] else None
        rows[name] = r
    rep["pairs"][f"s{k}-s{a.base}"] = rows
Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
for p, rows in rep["pairs"].items():
    print(f"== {p}")
    print(f"{'group':34s} {'|g_k|':>8s} {'|dg|':>8s} | {'e_k':>7s} {'e_b':>7s} {'e_k-e_b':>7s} | {'bf e_k':>7s} {'bf e_b':>7s} {'bf dif':>7s}")
    for n, r in rows.items():
        print(f"{n:34s} {r['g_k']:8.4f} {r['g_k_minus_g_base']:8.4f} | {r['ours_e_k']:7.4f} {r['ours_e_base']:7.4f} "
              f"{r['ours_e_k_minus_e_base']:7.4f} | {r['bf16_e_k']:7.4f} {r['bf16_e_base']:7.4f} {r['bf16_e_k_minus_e_base']:7.4f}")
