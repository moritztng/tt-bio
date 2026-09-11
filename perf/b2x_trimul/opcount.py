#!/usr/bin/env python3
"""Op count, bytes and time as three numbers per arm, plus the corrected per-op table.

The first pass's byte capture reported n_ops = 4 for a whole trimul, which is not a count of
anything real: `itemize.top_level_spans` only recognises a node whose `node_type` is
`function_start` and whose name starts with `ttnn.` at stack depth 0, and this ttnn build does not
emit those for most ops. So before any per-op table is published, dump what the capture actually
contains and count ops from the thing that is there.

Op count is taken two ways so neither is trusted alone:
  graph   distinct top-level operations in the ttnn.graph capture
  hook    a counter wrapped around ttnn's own dispatch, i.e. what the host actually issues
"""
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))
sys.path.insert(0, str(ROOT / "perf" / "b2x_trimul"))

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.trimul_tail as F1
from itemize import itemize
from real_traffic import counts
from trimul_ab import weights, make_inputs


def node_hist(g):
    h = Counter()
    names = Counter()
    for n in g:
        t = n.get("node_type")
        h[t] += 1
        p = n.get("params") or {}
        nm = str(p.get("name", ""))
        if nm:
            names[nm] += 1
    return h, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--dump-first", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ck = T.trunk_compute_kernel_config(compute_kernel_config())
    mods = {e: T.TriangleMultiplication(e, weights(), ck) for e in (False, True)}
    z, m = make_inputs(dev, a.n, "ones")
    res = {"n": a.n, "host": "qb2", "card": 2,
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    ARMS = [("A", False, None, False), ("B", True, None, False),
            ("C64", True, 64, False), ("D", True, None, True)]

    def set_arm(mask, r, f1):
        T.set_trimul_mask_after_move(mask)
        T.set_trimul_inproj_rowblock(r is not None, r)
        F1.set_f1_cz128(f1)

    out = {}
    for ending in (False, True):
        cell = {}
        for nm, mask, r, f1 in ARMS:
            set_arm(mask, r, f1)
            ttnn.deallocate(mods[ending](z, m))
            ttnn.synchronize_device(dev)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            o = mods[ending](z, m)
            ttnn.synchronize_device(dev)
            g = ttnn.graph.end_graph_capture()
            ttnn.deallocate(o)
            h, names = node_hist(g)
            c = counts({"sig": "trimul", "nodes": g})
            cell[nm] = {
                "real_MB": round(c["real_MB"], 3),
                "real_Z": round(c["real_MB"] / 67.108864, 3),
                "node_types": dict(h),
                "n_named_ops": sum(names.values()),
                "ops": dict(sorted(names.items(), key=lambda kv: -kv[1])),
            }
            if a.dump_first and nm == "A" and not ending:
                res["sample_nodes"] = g[:40]
            set_arm(False, None, False)
        out["ending" if ending else "starting"] = cell
        for nm in [x[0] for x in ARMS]:
            v = cell[nm]
            print("%-9s %-4s  Z=%7.3f  ops=%3d  %s" % (
                "ending" if ending else "starting", nm, v["real_Z"], v["n_named_ops"],
                json.dumps(v["ops"])[:150]), flush=True)
    res["arms"] = out
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
