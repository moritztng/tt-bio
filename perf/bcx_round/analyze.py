#!/usr/bin/env python3
"""Turn the round meter's event log into the round, its attribution and its distribution.

A round is bounded by two consecutive entries into `sequence_gradients`, so the last round
collected has no closing boundary and is dropped: N requested rounds give N-1 measured
walls. Round 1 carries BindCraft 2's jit compile and is reported apart from the rest.
"""
import json
import statistics as st
import sys


def med(xs):
    return round(st.median(xs), 3) if xs else None


def main(path, out=None):
    d = json.load(open(path))
    ev, stamp = d["events"], d["stamp"]
    clk = d["aiclk"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    # The stop stamp closes the last round; without it the last collected round has no
    # closing boundary and N requested rounds give N-1 walls.
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    rows = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t0") is not None and e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        sg = [e for e in inside if e["phase"] == "sequence_gradients"]
        sg_t = sum(e["dt"] for e in sg)
        sg_win = (sg[0]["t0"], sg[0]["t1"]) if sg else (t0, t0)
        dev = {p: [e for e in inside if e["kind"] == "device" and e["phase"] == p]
               for p in ("taped", "backward", "primal")}
        dev_in_sg = sum(e["dt"] for p in dev for e in dev[p]
                        if sg_win[0] <= e["t0"] and e["t1"] <= sg_win[1])
        opt = sum(e["dt"] for e in inside if e["kind"] == "optimizer")
        pred_out = sum(e["dt"] for e in inside if e["phase"] == "predict"
                       and not (sg_win[0] <= e["t0"] <= sg_win[1]))
        samples = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        aiclk = sorted(c for c, _ in samples)
        rows.append({
            "round": starts[i]["round"], "wall": round(t1 - t0, 3),
            "sg": round(sg_t, 3),
            "taped_n": len(dev["taped"]), "taped_s": round(sum(e["dt"] for e in dev["taped"]), 3),
            "bwd_n": len(dev["backward"]), "bwd_s": round(sum(e["dt"] for e in dev["backward"]), 3),
            "primal_n": len(dev["primal"]), "primal_s": round(sum(e["dt"] for e in dev["primal"]), 3),
            "host_in_sg": round(sg_t - dev_in_sg, 3),
            "opt": round(opt, 3), "predict_outside_sg": round(pred_out, 3),
            "residual": round(t1 - t0 - sg_t - opt - pred_out, 3),
            "aiclk_med": aiclk[len(aiclk) // 2] if aiclk else None,
            "aiclk_min": aiclk[0] if aiclk else None,
            "n_clk": len(aiclk),
            "load1": round(sum(l for _, l in samples) / len(samples), 1) if samples else None,
        })
    body = rows[1:] or rows
    summary = {
        "rounds_measured": len(rows), "round1_wall": rows[0]["wall"] if rows else None,
        "wall_med": med([r["wall"] for r in body]),
        "wall_min": min(r["wall"] for r in body), "wall_max": max(r["wall"] for r in body),
        "sg_med": med([r["sg"] for r in body]),
        "taped_s_med": med([r["taped_s"] for r in body]),
        "taped_n_med": med([r["taped_n"] for r in body]),
        "bwd_s_med": med([r["bwd_s"] for r in body]),
        "primal_s_med": med([r["primal_s"] for r in body]),
        "host_in_sg_med": med([r["host_in_sg"] for r in body]),
        "opt_med": med([r["opt"] for r in body]),
        "predict_outside_sg_med": med([r["predict_outside_sg"] for r in body]),
        "residual_med": med([r["residual"] for r in body]),
        "aiclk_med": med([r["aiclk_med"] for r in body]),
        "load1_med": med([r["load1"] for r in body]),
    }
    per_call = {}
    for p in ("taped", "backward", "primal"):
        xs = [e["dt"] for e in ev if e["kind"] == "device" and e["phase"] == p]
        per_call[p] = {"n": len(xs), "med": med(xs),
                       "min": round(min(xs), 3) if xs else None,
                       "max": round(max(xs), 3) if xs else None}
    res = {"stamp": stamp, "summary": summary, "per_call": per_call, "rounds": rows,
           "setup": [e for e in ev if e["kind"] == "setup"],
           "stages": [e for e in ev if e["kind"] in ("stage", "stage_start")]}
    print(json.dumps({"summary": summary, "per_call": per_call}, indent=1))
    for r in rows:
        print(" ".join(f"{k}={r[k]}" for k in
                       ("round", "wall", "sg", "taped_n", "taped_s", "bwd_s", "primal_n",
                        "primal_s", "host_in_sg", "opt", "predict_outside_sg", "residual",
                        "aiclk_med", "load1")))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
