#!/usr/bin/env python3
"""The interleaved `recompute` A/B, round by round: wall, device column and DRAM high-water.

Round 1 carries the jit compile and the first trunk load and is dropped. A round that loaded a
checkpoint for the first time is flagged, because that upload lands inside its first device call
and belongs to neither arm.
"""
import collections
import json
import statistics as st
import sys

GB = 1 << 30


def med(xs):
    return round(st.median(xs), 3) if xs else None


def main(path, out=None):
    d = json.load(open(path))
    ev, stamp = d["events"], d["stamp"]
    clk = d.get("aiclk") or []
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    arms = {e["round"]: e["phase"] for e in ev if e["kind"] == "arm"}

    rows = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t1") is not None and t0 - 1e-6 <= e["t0"]
                  and e["t1"] <= t1 + 1e-6]
        dev = [e for e in inside if e["kind"] == "device"]
        mem = [e for e in inside if e["kind"] == "mem" and e.get("dram_bytes")]
        sg = sum(e["dt"] for e in inside if e["phase"] == "sequence_gradients")
        col = sum(e["dt"] for e in dev)
        samples = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        aiclk = sorted(c for c, _ in samples)
        r = starts[i]["round"]
        rows.append({
            "round": r, "arm": arms.get(r, "?"), "wall": round(t1 - t0, 3),
            "sg": round(sg, 3), "device_col": round(col, 3),
            "host_in_sg": round(sg - col, 3),
            "taped": round(sum(e["dt"] for e in dev if e["phase"] == "taped"), 3),
            "bwd": round(sum(e["dt"] for e in dev if e["phase"] == "backward"), 3),
            "dram_fwd_gb": round(max((e["dram_bytes"] for e in mem
                                      if e["phase"] == "taped"), default=0) / GB, 3),
            "dram_max_gb": round(max((e["dram_bytes"] for e in mem), default=0) / GB, 3),
            "aiclk_med": aiclk[len(aiclk) // 2] if aiclk else None,
            "load1": round(sum(l for _, l in samples) / len(samples), 1) if samples else None,
        })

    body = [r for r in rows if r["round"] > 1]
    by = collections.defaultdict(list)
    for r in body:
        by[r["arm"]].append(r)
    keys = ("wall", "sg", "device_col", "host_in_sg", "taped", "bwd", "dram_fwd_gb",
            "dram_max_gb", "aiclk_med", "load1")
    summary = {}
    for arm, rs in by.items():
        summary[arm] = {"n": len(rs), **{k: med([x[k] for x in rs]) for k in keys}}
        summary[arm]["wall_min"] = min(x["wall"] for x in rs)
        summary[arm]["wall_max"] = max(x["wall"] for x in rs)
    if "ckpt" in summary and "nockpt" in summary:
        a, b = summary["ckpt"], summary["nockpt"]
        summary["delta"] = {k: round(b[k] - a[k], 3) for k in
                            ("wall", "device_col", "host_in_sg", "taped", "bwd",
                             "dram_fwd_gb", "dram_max_gb")}
        summary["speedup"] = {"round": round(a["wall"] / b["wall"], 3),
                              "device_col": round(a["device_col"] / b["device_col"], 3)}
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(f"round {r['round']:2d} {r['arm']:7s} wall {r['wall']:7.3f} dev {r['device_col']:7.3f} "
              f"host {r['host_in_sg']:7.3f} taped {r['taped']:6.3f} bwd {r['bwd']:6.3f} "
              f"dram_fwd {r['dram_fwd_gb']:6.3f} GB max {r['dram_max_gb']:6.3f} "
              f"clk {r['aiclk_med']} load {r['load1']}")
    if out:
        json.dump({"stamp": stamp, "summary": summary, "rounds": rows},
                  open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
