#!/usr/bin/env python3
"""Does R123's logit/value split survive ASSEMBLY? Re-analysis only, no card, no new run.

R123 split the triangle attention by position -- logit path (`linear_q`, `linear_k`, `linear_z`,
`layer_norm`) against value path (`linear_v`, `linear_o`, `linear_g`) -- and found the value path
clean at ~0.025 on both attentions while the logit path ran 4.2x-4.6x dirtier, with
`fp32_softmax` moving only the logit side. It measured the sub-modules ALONE and said so: silent
about the assembled block, which is the scope D8 actually fails at.

The assembled tables were already on disk. `instrument_a_bundle_*.json` carries `per_parameter`
with the same leaf names, so the identical grouping applies with no measurement at all. Reported
for every assembled arm in the branch: block 0 at crop 384 and 64, blocks 23 and 47 at crop 64,
and the `transpose_bias` off arm beside each, because the flag is the only lever known to move
this split and an arm without its control is not a reading.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

D = "perf/of3t_gradients"
BAR = 5.0e-02
LOGIT = ("linear_q", "linear_k", "linear_z", "layer_norm")
VALUE = ("linear_v", "linear_o", "linear_g")
ARMS = [
    ("block0 crop384 shipped", "block0_r0"),
    ("block0 crop384 nofp32 ", "block0_r0_nofp32"),
    ("block0 crop64  shipped", "block0_r0_crop64"),
    ("block0 crop64  tb-off ", "block0_r0_crop64_tboff"),
    ("block23 crop64 shipped", "block23_r0_crop64_tbshipped"),
    ("block23 crop64 tb-off ", "block23_r0_crop64_tboff"),
    ("block47 crop64 shipped", "block47_r0_crop64_tbshipped"),
    ("block47 crop64 tb-off ", "block47_r0_crop64_tboff"),
    ("stack0-47 n384 shipped", "stack0_47_n384"),
    ("stack0-47 n384 tb-off ", "stack0_47_n384_tboff"),
]


def path_group(key: str) -> str:
    leaf = key.split(".")[-2] if key.endswith((".weight", ".bias")) else key.split(".")[-1]
    if any(leaf.startswith(g) for g in LOGIT):
        return "logit"
    if any(leaf.startswith(g) for g in VALUE):
        return "value"
    return "other"


def module_of(key: str) -> str | None:
    for m in ("tri_att_end", "tri_att_start"):
        if f".{m}." in key or key.startswith(m):
            return m
    return None


def summarise(rows):
    v = sorted(r["rel_l2"] for r in rows)
    if not v:
        return None
    w = max(rows, key=lambda r: r["rel_l2"])
    return {"n": len(v), "median": float(np.median(v)), "worst": float(v[-1]),
            "worst_tensor": w["their_tensor"], "over_bar": int(sum(1 for x in v if x > BAR))}


def main() -> int:
    out = {"instrument": "R123's logit/value split applied to the ASSEMBLED-block tables",
           "bar": BAR, "grouping": {"logit": LOGIT, "value": VALUE},
           "note": "re-analysis of instrument_a_bundle_*.json per_parameter; no new measurement",
           "caveat": "`linear_z` produces the pair bias ADDED TO THE LOGITS and is counted on "
                     "the logit side, R123's own convention; moving it softens the ratios "
                     "without changing their direction",
           "arms": {}}
    print(f"{'arm':24s} {'module':14s} {'logit':>8s} {'value':>8s} {'ratio':>7s}  n(l/v)")
    for label, tag in ARMS:
        p = os.path.join(D, f"instrument_a_bundle_{tag}.json")
        if not os.path.isfile(p):
            print(f"{label:24s} MISSING {p}")
            continue
        rep = json.load(open(p))
        rows = [r for r in rep["per_parameter"] if r["ref_norm"] >= 1e-12]
        arm = {"tag": tag, "crop": rep.get("crop"), "tokens": rep.get("probe", {}).get("tokens"),
               "config": rep.get("shipped_config"), "modules": {}}
        for m in ("tri_att_start", "tri_att_end"):
            sel = [r for r in rows if module_of(r["key"]) == m]
            lg = summarise([r for r in sel if path_group(r["key"]) == "logit"])
            vl = summarise([r for r in sel if path_group(r["key"]) == "value"])
            if not (lg and vl):
                continue
            ratio = lg["median"] / vl["median"]
            arm["modules"][m] = {"logit": lg, "value": vl, "ratio_logit_over_value": ratio,
                                 "all": summarise(sel)}
            print(f"{label:24s} {m:14s} {lg['median']:8.4f} {vl['median']:8.4f} "
                  f"{ratio:7.2f}  {lg['n']}/{vl['n']}")
        out["arms"][label.strip()] = arm
    os.makedirs("perf/of3t_reopen", exist_ok=True)
    dst = "perf/of3t_reopen/logit_assembled.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
