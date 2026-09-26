#!/usr/bin/env python3
"""Quote-ready summary of one round arm: the timed rounds only, with their clock and load.

Drops the leading 0.002 s boundary artifact (two `sequence_gradients` entries 2 ms apart before
any work) and the compile round that follows it, which is what ">= 2 compiling rounds" means
here. Everything else is reported as a median with min/max, never as a mean.
"""
import json
import statistics as st
import sys

d = json.load(open(sys.argv[1]))
rows, stamp, clk = d["rounds"], d["stamp"], None
timed = [r for r in rows if r["wall"] > 1.0][1:]          # drop artifact, drop compile round


def q(name, xs, nd=3):
    xs = [x for x in xs if x is not None]
    return {name: {"med": round(st.median(xs), nd), "min": round(min(xs), nd),
                   "max": round(max(xs), nd), "n": len(xs)}}


dev = [r["taped_s"] + r["bwd_s"] + r["primal_s"] for r in timed]
share = [(t + 0.0) / w for t, w in zip(dev, (r["wall"] for r in timed))]
host = [w - t for w, t in zip((r["wall"] for r in timed), dev)]
ceil = [w / h for w, h in zip((r["wall"] for r in timed), host)]

out = {
    "tree": {"commit": stamp["commit"], "tt_bio_file": stamp.get("tt_bio_file"),
             "exact": stamp.get("exact"), "shipped_pool": stamp.get("shipped_pool"),
             "binder_pinned": stamp.get("binder_pinned"),
             "state_shape": stamp.get("state_shape") or d.get("state_shape"),
             "length_bucket_size": stamp["length_bucket_size"],
             "host": stamp["host"], "card": stamp["card"], "pci": stamp["pci"],
             "seed": stamp["seed"], "rounds_requested": stamp["rounds_requested"],
             "stopped": stamp.get("stopped"), "wall_seconds": stamp.get("wall_seconds"),
             "device_calls": stamp.get("device_calls"), "host_folds": stamp.get("host_folds"),
             "exact_softmax_stats": stamp.get("exact_softmax_stats"),
             "exact_layer_norm_stats": stamp.get("exact_layer_norm_stats")},
    "timed_rounds": [r["round"] for r in timed],
}
for name, xs, nd in (("round_wall_s", [r["wall"] for r in timed], 3),
                     ("sequence_gradients_s", [r["sg"] for r in timed], 3),
                     ("device_s", dev, 3),
                     ("taped_fwd_s", [r["taped_s"] for r in timed], 3),
                     ("backward_s", [r["bwd_s"] for r in timed], 3),
                     ("host_s", host, 3),
                     ("device_share", share, 4),
                     ("ceiling_if_device_free", ceil, 3),
                     ("aiclk_med", [r["aiclk_med"] for r in timed], 1),
                     ("aiclk_min", [r["aiclk_min"] for r in timed], 1),
                     ("load1", [r["load1"] for r in timed], 1)):
    out.update(q(name, xs, nd))
out["spread_max_over_min"] = round(max(r["wall"] for r in timed)
                                   / min(r["wall"] for r in timed), 3)
print(json.dumps(out, indent=1))
