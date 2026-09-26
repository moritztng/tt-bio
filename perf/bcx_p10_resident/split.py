#!/usr/bin/env python3
"""Split a round into host, Evoformer-on-card and extra-MSA-on-card, per arm.

`perf/bcx_round/analyze.py` already gives the round wall, `sequence_gradients` and the host
remainder; it sums every device event, so `host_in_sg` is right on both arms of this row
without it knowing the extra-MSA swap exists. What it does not do is say which of the two
device-side stacks the device seconds went to, and that is the number this row is graded on:

    host_removed - device_added

Reads one or more `round_events.json`, groups by arm (the `extra_msa_on_device` stamp) and
prints the per-arm medians with the AICLK and loadavg each arm actually ran at.
"""
import json
import statistics as st
import sys

PHASES = ("taped", "backward", "primal")


def med(xs):
    return round(st.median(xs), 3) if xs else None


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
        dev = [e for e in inside if e["kind"] == "device" and w0 <= e["t0"] and e["t1"] <= w1]
        # An event written before this patch carries no `module`; it can only be the Evoformer.
        by = {}
        for m in ("evoformer", "extra_msa"):
            for p in PHASES:
                by[m + "_" + p] = round(sum(e["dt"] for e in dev
                                            if e.get("module", "evoformer") == m
                                            and e["phase"] == p), 3)
        sample = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        aiclk = sorted(c for c, _ in sample)
        row = {"src": path, "round": starts[i]["round"], "wall": round(t1 - t0, 3),
               "sg": round(sg[0]["dt"], 3),
               "device": round(sum(e["dt"] for e in dev), 3),
               "host_in_sg": round(sg[0]["dt"] - sum(e["dt"] for e in dev), 3),
               "arm": "device" if stamp.get("extra_msa_on_device") else "jax",
               "aiclk_med": aiclk[len(aiclk) // 2] if aiclk else None,
               "load1": round(sum(l for _, l in sample) / len(sample), 1) if sample else None,
               **by}
        row["evoformer"] = round(sum(row["evoformer_" + p] for p in PHASES), 3)
        row["extra_msa"] = round(sum(row["extra_msa_" + p] for p in PHASES), 3)
        out.append(row)
    # Round 1 carries BindCraft 2's jit compile and is never a measurement.
    return out[1:] or out


KEYS = ("wall", "sg", "host_in_sg", "device", "evoformer", "extra_msa",
        "evoformer_taped", "evoformer_backward", "extra_msa_taped", "extra_msa_backward",
        "aiclk_med", "load1")


def main(paths, out=None):
    rows = [r for p in paths for r in rounds_of(p)]
    arms = {}
    for a in ("jax", "device"):
        got = [r for r in rows if r["arm"] == a]
        if got:
            arms[a] = {"n": len(got), **{k: med([r[k] for r in got]) for k in KEYS}}
    res = {"arms": arms, "rounds": rows}
    if "jax" in arms and "device" in arms:
        A, B = arms["jax"], arms["device"]
        res["delta"] = {
            "host_removed": round(A["host_in_sg"] - B["host_in_sg"], 3),
            "device_added": round(B["device"] - A["device"], 3),
            "net": round((A["host_in_sg"] - B["host_in_sg"]) - (B["device"] - A["device"]), 3),
            "round_before": A["wall"], "round_after": B["wall"],
            "speedup": round(A["wall"] / B["wall"], 3) if B["wall"] else None,
        }
    print(json.dumps({k: res[k] for k in res if k != "rounds"}, indent=1))
    for r in rows:
        print(" ".join(k + "=" + str(r[k]) for k in ("arm", "round", "wall", "host_in_sg",
                                                     "evoformer", "extra_msa",
                                                     "aiclk_med", "load1")))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--out=")]
    o = [a[6:] for a in sys.argv[1:] if a.startswith("--out=")]
    main(args, o[0] if o else None)
