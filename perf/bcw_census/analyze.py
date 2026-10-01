#!/usr/bin/env python3
"""Fold attrib.json rungs into the census table: item, exponent of N, bytes per rung.

    analyze.py out/bh_*/attrib.json out/wh_*/attrib.json > census.json

attrib.py counts a dim as N when it equals the axis, which miscounts a channel dim that happens
to equal N (512 channels at N=512 read as N^3). Here the exponent comes from the shape's
leading dims only: the last dim of a rank>=3 tensor is a channel. Items are bucketed by the
model sub-module that made them, read off the site attrib.py stamped.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

SUB = [  # (site substring, sub-module)
    ("_in_proj_matmul", "trimul"), ("_channel_move", "trimul"), (":_multiply", "trimul"),
    ("_attend_pair", "triatt"), (":attend", "triatt"), ("_attend_heads", "triatt"),
    ("gate_and_project", "triatt"), ("_pair_bias_from_z", "msa_row_bias"),
    ("_pair_proj_linear", "pair_proj(trimul out, triatt bias/out)"),
    ("_pair_transpose_impl", "triatt"), ("triangle_attention", "triatt"),
    ("af2.py:_rows", "pair_transition"), ("_sum_rows", "opm"),
    ("_split_heads", "msa_track"), ("batched_matmul", "msa_track"),
    ("_fp32_softmax", "msa_row_bias"),
]


CHANNEL_SUBS = ("trimul", "pair_transition")


def exponent(shape, n, sub=""):
    """N-dims of the shape. The last dim of a rank>=3 tensor is a channel unless it is N, and
    at N=512 a trimul or transition hidden of 512 channels is still a channel."""
    lead = list(shape[:-1]) if len(shape) >= 3 else list(shape)
    e = sum(1 for d in lead if d == n) + 2 * sum(1 for d in lead if d == n * n)
    if len(shape) >= 3 and shape[-1] == n and not (e >= 2 and sub in CHANNEL_SUBS):
        e += 1
    return e


def submodule(g):
    site, verb = g.get("site") or "", g.get("verb") or ""
    if verb == "checkpoint":
        return "pin:" + site.split(":")[-1]
    if verb == "triangle_attention":
        return "triatt"
    for key, name in SUB:
        if key in site:
            return name
    if site == "dispatch.py:call" and verb in ("relu", "_taped_linear"):
        return "pair_transition"
    if verb == "_k_rne_add":
        return "residual"
    return site or "-"


def fold(path):
    d = json.loads(pathlib.Path(path).read_text())
    n = d["axis"]
    items = collections.Counter()
    exps = {}
    for g in d["groups"]:
        role = g["role"]
        e = exponent(g["shape"], n, submodule(g))
        if role.startswith("raw"):
            item = "weights+consts" if e == 0 else f"raw_N{e}"
        elif role.startswith("grad"):
            item = "grads"
        elif role == "leaf_input":
            item = "leaf_inputs"
        else:
            sm = submodule(g)
            item = sm if sm.startswith("pin:") else f"block:{sm}"
            if role == "outer_tape" and not sm.startswith("pin:"):
                item = f"outer:{sm}"
        items[item] += g["bytes"]
        exps.setdefault(item, collections.Counter())[e] += g["bytes"]
    listed_groups = sum(items.values())
    items["unlisted(C++ held + page rounding)"] = d["walked_peak_used_bytes"] - d["listed_bytes"]
    items["below top-200 groups"] = d["listed_bytes"] - listed_groups
    return {"axis": n, "peak": d["walked_peak_used_bytes"], "frontier": d["frontier_peak_bytes"],
            "total": d["device_total_bytes"], "largest_free": d["largest_free_bytes_at_walk"],
            "walk_at": d["walk_at_retire_call"], "refusal": d.get("refusal"),
            "items": dict(items.most_common()),
            "exp": {k: v.most_common(1)[0][0] for k, v in exps.items()}}


def main():
    rows = {p: fold(p) for p in sys.argv[1:]}
    json.dump(rows, sys.stdout, indent=1)
    print(file=sys.stderr)
    for p, r in rows.items():
        print(f"== {p}  axis {r['axis']}  peak {r['peak'] / 1e9:.3f} GB  of {r['total'] / 1e9:.3f}"
              f"  largest free {r['largest_free'] / 1e6:.0f} MB  at retire {r['walk_at']}",
              file=sys.stderr)
        pair = r["axis"] ** 2 * 256
        for k, v in r["items"].items():
            print(f"   {k:40s} N^{r['exp'].get(k, '?')}  {v / 1e9:7.3f} GB  "
                  f"{v / pair:6.1f} pair-eq", file=sys.stderr)


if __name__ == "__main__":
    main()
