#!/usr/bin/env python3
"""Pool the subtraction control's arms and price the multimer template embedding.

`analyze.py` reports one arm at a time. This groups the arms by the `template_const` stamp,
pools their per-round walls, and prints the delta with the AICLK and the loadavg each arm
actually ran at, because the round is 63 % host and a host number without a load is not a
measurement on this workload.
"""
import json
import statistics as st
import sys


def rounds_of(path):
    d = json.load(open(path))
    ev, stamp, clk = d["events"], d["stamp"], d["aiclk"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    out = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        sg = [e for e in inside if e["phase"] == "sequence_gradients"]
        if not sg:
            continue
        w0, w1 = sg[0]["t0"], sg[0]["t1"]
        dev = sum(e["dt"] for e in inside
                  if e["kind"] == "device" and w0 <= e["t0"] and e["t1"] <= w1)
        sample = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        aiclk = sorted(c for c, _ in sample)
        out.append({"src": path, "round": starts[i]["round"], "wall": t1 - t0,
                    "sg": sg[0]["dt"], "device": dev, "host": sg[0]["dt"] - dev,
                    "const": bool(stamp.get("template_const")),
                    "fired": stamp.get("template_patch_fired"),
                    "aiclk": aiclk[len(aiclk) // 2] if aiclk else None,
                    "load1": sum(l for _, l in sample) / len(sample) if sample else None})
    return out


def summarize(rows, label):
    med = lambda k: round(st.median([r[k] for r in rows]), 3)
    return {"arm": label, "rounds": len(rows), "wall": med("wall"), "sg": med("sg"),
            "device": med("device"), "host": med("host"),
            "aiclk_min": min(r["aiclk"] for r in rows),
            "aiclk_med": st.median([r["aiclk"] for r in rows]),
            "load1": round(st.median([r["load1"] for r in rows]), 2)}


def main(paths):
    rows = [r for p in paths for r in rounds_of(p)]
    # Round 1 carries BindCraft 2's jit compile and analyze.py reports it apart; so does this.
    rows = [r for r in rows if r["round"] > 1]
    model = summarize([r for r in rows if not r["const"]], "template embedding")
    const = summarize([r for r in rows if r["const"]], "forced to a constant")
    delta = {"round_s": round(model["wall"] - const["wall"], 3),
             "host_s": round(model["host"] - const["host"], 3),
             "device_s": round(const["device"] - model["device"], 3),
             "x": round(model["wall"] / const["wall"], 3)}
    delta["host_removed_minus_device_added"] = round(delta["host_s"] - delta["device_s"], 3)
    doc = {"arms": [model, const], "delta": delta,
           "fired": sorted({r["fired"] for r in rows if r["const"]}),
           "sources": paths}
    print(json.dumps(doc, indent=1))
    return doc


if __name__ == "__main__":
    main(sys.argv[1:])
