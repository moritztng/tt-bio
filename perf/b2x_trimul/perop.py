#!/usr/bin/env python3
"""The corrected per-op byte table for the trimul, both ownership rules, plus op counts.

The orchestrator asked one specific question before any Z-delta is trusted: is `ttnn.chunk` 8 Z of
real DRAM traffic or free metadata over an existing buffer? The published block attribution has it
as the single fattest op in a pairformer block, and a channel-axis chunk is exactly the shape of
thing the tensor-id counter over-charged. This answers it per op, with buffer-address dedupe.

Also reports device-op counts per arm, because at the campaign's measured 22.0 us/op an arm that
removes ops may be paying off through op count and not through bytes.
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))
sys.path.insert(0, str(ROOT / "perf" / "b2x_trimul"))

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.trimul_tail as F1
import itemize as IT
from real_traffic import counts
from trimul_ab import weights, make_inputs

Z = 67.108864


def devops(g):
    """Device programs enqueued, from the capture. Every name appears once at function_start and
    once at function_end, so the raw histogram is exactly twice the call count."""
    c = Counter()
    for n in g:
        if n.get("node_type") == "function_start":
            nm = str((n.get("params") or {}).get("name", ""))
            if nm.endswith("DeviceOperation"):
                c[nm] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ck = T.trunk_compute_kernel_config(compute_kernel_config())
    mods = {e: T.TriangleMultiplication(e, weights(), ck) for e in (False, True)}
    z, m = make_inputs(dev, a.n, "ones")
    res = {"n": a.n, "host": "qb2", "card": 2, "Z_MB": Z,
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arms": {}}

    ARMS = [("A", False, None, False), ("B", True, None, False),
            ("C64", True, 64, False), ("D", True, None, True)]

    for ending in (False, True):
        key = "ending" if ending else "starting"
        res["arms"][key] = {}
        for nm, mask, r, f1 in ARMS:
            T.set_trimul_mask_after_move(mask)
            T.set_trimul_inproj_rowblock(r is not None, r)
            F1.set_f1_cz128(f1)
            ttnn.deallocate(mods[ending](z, m))
            ttnn.synchronize_device(dev)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            o = mods[ending](z, m)
            ttnn.synchronize_device(dev)
            g = ttnn.graph.end_graph_capture()
            ttnn.deallocate(o)
            call = {"sig": "trimul", "nodes": g}
            row = {}
            for rule in ("stack", "range"):
                IT_rule = rule
                orig = IT.top_level_spans
                IT.top_level_spans = lambda nodes, _r=IT_rule: orig(nodes, _r)
                try:
                    c = counts(call)
                finally:
                    IT.top_level_spans = orig
                row[rule] = {
                    "real_Z": round(c["real_MB"] / Z, 3),
                    "n_ops_attributed": sum(1 for t, *_ in c["per_op"] if t),
                    "per_op_Z": [[round(t / Z, 4), n, round(w / Z, 4), round(rd / Z, 4)]
                                 for t, n, w, rd in c["per_op"] if t / Z >= 0.005],
                }
            d = devops(g)
            row["devops"] = {k: v for k, v in sorted(d.items(), key=lambda kv: -kv[1])}
            row["n_devops"] = sum(d.values())
            res["arms"][key][nm] = row
            T.set_trimul_mask_after_move(False)
            T.set_trimul_inproj_rowblock(False, None)
            F1.set_f1_cz128(False)
            print("%-9s %-4s  stackZ=%7.3f  rangeZ=%7.3f  devops=%3d" % (
                key, nm, row["stack"]["real_Z"], row["range"]["real_Z"], row["n_devops"]),
                flush=True)
            if key == "starting":
                for t, n, w, rd in row["range"]["per_op_Z"][:12]:
                    print("      %7.3f Z  %-38s w %6.3f r %6.3f" % (t, n, w, rd), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
