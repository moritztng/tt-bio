#!/usr/bin/env python3
"""Top op signatures of one census fold with shapes, ms/call and achieved GB/s (inputs + outputs once).
usage: agg_sig.py RUN_DIR/<arm> [N]"""
import json, sys, collections
import numpy as np
R = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625, "UINT32": 4, "INT32": 4, "UINT16": 2}
sigs, outs = {}, {}
for l in open(f"{R}/sig_full.jsonl"):
    e = json.loads(l)
    if "op" in e: sigs[e["sig"]] = e
    elif "out" in e: outs[e["sig"]] = e["out"]


def tensors(x):
    if isinstance(x, dict):
        if "T" in x and isinstance(x["T"], dict) and "padded" in x["T"]: yield x["T"]
        else:
            for v in x.values(): yield from tensors(v)
    elif isinstance(x, list):
        for v in x: yield from tensors(v)


def nbytes(ts):
    return sum(np.prod(t["padded"]) * BYTES.get(t["dtype"], 2) for t in ts)


calls = np.array([json.loads(l) for l in open(f"{R}/ops_full.jsonl")], dtype=np.int64)
calls = calls[np.argsort(calls[:, 1], kind="stable")]
P = np.array([[x if x is not None else -1 for x in json.loads(l)] for l in open(f"{R}/progs_full.jsonl")], dtype=np.float64)
P[:, 0] = np.floor(P[:, 0] / 1024); P[:, 3] = np.where(P[:, 3] > 1e9, 0, P[:, 3])
idx = np.searchsorted(calls[:, 1], P[:, 0], side="right") - 1
ok = (idx >= 0) & (P[:, 0] < calls[np.clip(idx, 0, None), 2])
P, idx = P[ok], idx[ok]
t = collections.Counter(); c = collections.Counter()
for s, k in zip(calls[idx, 0], P[:, 3]): t[s] += k
for s in calls[:, 0]: c[s] += 1
print("%8s %6s %8s %7s  %-16s %-40s %s" % ("s", "calls", "ms/call", "GB/s", "op", "site", "shapes in -> out"))
for s, k in t.most_common(N):
    e = sigs[s]; ins = list(tensors([e["args"], e["kw"]])); o = list(tensors(outs.get(s, [])))
    ms = k / 1e6 / c[s]; gbs = (nbytes(ins) + nbytes(o)) / (ms * 1e6) if ms else 0
    shp = " ".join(f"{t['dtype'][:4]}{t['padded']}" for t in ins[:3]) + " -> " + " ".join(f"{t['dtype'][:4]}{t['padded']}" for t in o[:1])
    print("%8.3f %6d %8.3f %7.0f  %-16s %-40s %s" % (k / 1e9, c[s], ms, gbs, e["op"].replace("ttnn.", ""), (e["site"] or [""])[0], shp))
