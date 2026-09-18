#!/usr/bin/env python3
"""Every launch key the fold issues, and for each one the exact reason it was or was not priced.

`c10-fold-census` priced 70 keys and refused 52. The brief asks WHY each of the 52 was refused,
because a key that is unmeasurable for a structural reason is a different finding from one the
arm builder simply had no branch for. That question is answerable from source, not from a
device: the refusal happens in three named places and this file reproduces all three on the
fold's own captures.

  1. `true_floor.LAUNCH_SKIP`      host metadata, or a python wrapper whose device child is
                                   already counted. Charging it would be double counting.
  2. `true_floor.LAUNCH_ARM`       no entry => `launch_arm()` returns None => the op never
                                   reaches `fold_shapes.json` at all. It is not in the 52 and
                                   was not listed as refused either. `ttnn.generic_op` is the
                                   whole of this bucket and it is the fold's second largest op.
  3. `c10_fold_census.census._operands`  returns None for any class it has no branch for. This
                                   is the 52. The reason is uniform and it is NOT structural:
                                   the builder had no branch, the shapes were there all along.

Bytes come from `roof_arb/corrected_traffic.counts`, the same call `true_floor` makes, which
dedupes on BUFFER ADDRESS. Operand params come from the capture's own `tensor` nodes, so an
arm built from this file carries the fold's own layout, memory config and dtype instead of a
default. Nothing here opens a device.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
sys.path.insert(0, str(PERF / "roof_true"))
sys.path.insert(0, str(PERF / "b2x_difflayer"))
import true_floor as TF                                                       # noqa: E402
import itemize as IT                                                          # noqa: E402

# the arms `c10_fold_census/census.py:_operands` has a branch for. Everything else it refuses.
CENSUS_PRICED = {"linear", "matmul", "multiply_", "add_", "multiply", "add",
                 "layer_norm", "layer_norm_w"}

SHAPE_RE = re.compile(r"Shape\(\[([0-9,\s]+)\]\)")
MEMLAYOUT_RE = re.compile(r"memory_layout=TensorMemoryLayout::(\w+)")
SHARD_RE = re.compile(r"shard_spec=(?!std::nullopt)")


class Args:
    roofs = "shape_roofs_qb2c3_shipped.json"
    eltwise_rate = None
    uncovered_lo = None
    uncovered_hi = None


def _shape(s):
    m = SHAPE_RE.search(s or "")
    return tuple(int(x) for x in m.group(1).split(",")) if m else None


def tensor_params(nodes, owner):
    """op index -> the capture's own params for every distinct buffer reaching it.

    `exec_flops.operands` keeps only (address, shape); an arm needs the layout, the memory
    config and the dtype too, or it measures a different op from the one the fold ran.
    """
    ins = defaultdict(dict)
    for n in nodes:
        p = n.get("params") or {}
        if n["node_type"] != "tensor" or not p.get("shape"):
            continue
        sh = _shape(p["shape"])
        if not sh:
            continue
        mc = p.get("memory_config") or ""
        rec = {"address": p.get("address"), "shape": list(sh),
               "dtype": (p.get("dtype") or "").replace("DataType::", ""),
               "layout": (p.get("layout") or "").replace("Layout::", ""),
               "buffer": p.get("buffer_type_value"),
               "mem_layout": (MEMLAYOUT_RE.search(mc).group(1) if MEMLAYOUT_RE.search(mc)
                              else None),
               "sharded": bool(SHARD_RE.search(mc)),
               "size": p.get("size")}
        for c in (n.get("connections") or []):
            o = owner.get(c)
            if o is not None:
                ins[o].setdefault((rec["address"], tuple(sh)), rec)
    return ins


def classify(name, arm, key):
    """The refusal bucket, named after the line of source that produces it."""
    if name in TF.LAUNCH_SKIP:
        return "skip", "LAUNCH_SKIP: host metadata or a wrapper whose device child is counted"
    if arm is None:
        return "noarm", ("no LAUNCH_ARM entry: dropped before fold_shapes.json, so the census "
                         "could not price it OR list it as refused")
    if key is None:
        return "nokey", "no launch shape: no output tensor in the capture and not in-place"
    if arm in CENSUS_PRICED:
        return "priced", "priced by c10-fold-census"
    return "refused", "census._operands has no branch for this class (shapes were available)"


def classify_census(name, arm, key, K, shape_rank):
    """`classify` plus the census's own two extra matmul refusals, so the split reproduces its
    70/52 exactly. Without these it reads 71/51: `matmul|1x768x512|KNone` has no recorded K, so
    the census refused it for a reason that IS structural (no K, no arithmetic) while every
    other refusal was just a missing branch."""
    bucket, why = classify(name, arm, key)
    if bucket == "priced" and arm in ("linear", "matmul"):
        if not K:
            return "refused", "census: no recorded K, arithmetic not derivable (structural)"
        if shape_rank < 2:
            return "refused", "census: output rank below 2, not a matmul key (structural)"
    return bucket, why


def collect(perf):
    R = TF.setup(perf, Args())
    R["eltwise_rate"] = R["lo_rate"]
    EF, SU = R["EF"], R["SU"]
    by = R["by"]
    keys = {}
    per_op = defaultdict(lambda: defaultdict(float))
    totals = defaultdict(float)
    for sig in TF.TOP:
        calls = by[sig]["calls"]
        nodes = EF.nodes_of(SU.cap_path(sig))
        J = TF.Join(R, sig)
        _o2, owner = IT.top_level_spans(nodes)
        tp = tensor_params(nodes, owner)
        for i in range(len(J.ops)):
            name = J.ops[i]["name"]
            ins, outs = J.ins[i], J.outs[i]
            arm = TF.launch_arm(name, ins)
            rows = TF.op_shape_rows(EF, name, ins, outs)
            key = TF.launch_key(arm, name, ins, outs, rows)
            kid = key or ("%s|%s|noshape" % (arm or "noarm", name))
            rank = len(kid.split("|")[1].split("x")) if key else 0
            bucket, why = classify_census(name, arm, key,
                                          rows[0][2] if rows else None, rank)
            e = keys.setdefault(kid, {
                "key": kid, "op": name, "arm": arm, "bucket": bucket, "why": why,
                "calls": 0.0, "B": 0.0, "sites": [], "n_sites": 0,
                "K": rows[0][2] if rows else None})
            e["calls"] += calls
            e["B"] += calls * J.bytes[i]
            site = {"sig": sig.split("|")[0], "op_index": i, "calls": calls,
                    "B_per_call": J.bytes[i],
                    "outs": [list(o) for o in outs],
                    "ins": sorted(tp.get(i, {}).values(), key=lambda r: -(r["size"] or 0))}
            if len(e["sites"]) < 6:
                e["sites"].append(site)
            e["n_sites"] += 1
            per_op[name]["calls"] += calls
            per_op[name]["B"] += calls * J.bytes[i]
            totals["calls"] += calls
            totals["B"] += calls * J.bytes[i]
    return R, keys, per_op, totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", type=Path, default=PERF)
    ap.add_argument("--out", type=Path, default=HERE / "keys_512.json")
    a = ap.parse_args()
    _R, keys, per_op, totals = collect(a.perf.resolve())
    buckets = defaultdict(lambda: {"keys": 0, "calls": 0.0, "B": 0.0})
    for e in keys.values():
        b = buckets[e["bucket"]]
        b["keys"] += 1
        b["calls"] += e["calls"]
        b["B"] += e["B"]
    print("%-8s %6s %10s %10s" % ("bucket", "keys", "calls", "TB"))
    for b in ["priced", "refused", "noarm", "nokey", "skip"]:
        if b not in buckets:
            continue
        v = buckets[b]
        print("%-8s %6d %10d %10.4f" % (b, v["keys"], v["calls"], v["B"] / 1e12))
    print("TOTAL    %6d %10d %10.4f" % (len(keys), totals["calls"], totals["B"] / 1e12))
    print()
    print("the block this row owes a price for -- refused + noarm, by key:")
    blk = sorted((e for e in keys.values() if e["bucket"] in ("refused", "noarm")),
                 key=lambda e: -e["B"])
    print("%-48s %9s %10s %9s %6s" % ("key", "calls", "GB", "bucket", "sites"))
    for e in blk:
        print("%-48s %9d %10.3f %9s %6d"
              % (e["key"][:48], e["calls"], e["B"] / 1e9, e["bucket"], e["n_sites"]))
    out = {"buckets": {k: dict(v) for k, v in buckets.items()},
           "totals": dict(totals),
           "n_keys": len(keys),
           "keys": keys,
           "by_op": {k: dict(v) for k, v in per_op.items()}}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
