#!/usr/bin/env python3
"""bcx-p10-devgap: the mask-bias A/B, arm against arm, on the rounds that are comparable.

`round_ab.py` alternates the arm at the round boundary in one process, so the comparison is
paired: every `on` round sits between two `off` rounds and a drift in the box lands on both.
Three rounds are dropped and the reasons are different:

    round 1         jit compile plus the lazy trunk load, ~104 s against ~13 s
    verify rounds   the `on` arm rebuilds every hit and copies both tensors to the host
    the last        `round_start` is the only boundary, so the last round has no closing one

Reach is the first thing printed. `build` against `hit` says whether the cache served; an `on`
round with 0 hits is a round where the lever never reached, and its seconds mean nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_devgap.gap import MODULES, PHASES          # noqa: E402

KEYS = ("wall", "device", "host_in_sg", "evoformer.s", "extra_msa.s", "template.s")


def rounds(path):
    d = json.load(open(path))
    ev, clk = d["events"], d["aiclk"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    out = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        sg = [e for e in inside if e["phase"] == "sequence_gradients"]
        dev = [e for e in inside if e["kind"] == "device"]
        r = {"round": starts[i]["round"], "arm": starts[i].get("arm"),
             "verify": bool(starts[i].get("verify")), "wall": t1 - t0,
             "sg": sum(e["dt"] for e in sg)}
        for m in MODULES:
            r[f"{m}.s"] = sum(e["dt"] for e in dev if e.get("module") == m)
        r["device"] = sum(r[f"{m}.s"] for m in MODULES)
        r["host_in_sg"] = r["sg"] - sum(
            e["dt"] for e in dev
            if sg and sg[0]["t0"] <= e["t0"] and e["t1"] <= sg[0]["t1"])
        s = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        r["aiclk_min"] = min((c for c, _ in s), default=None)
        r["aiclk_med"] = st.median(sorted(c for c, _ in s)) if s else None
        r["load1"] = sum(l for _, l in s) / len(s) if s else None
        # reach the round consumed, from the counters taken at its two boundaries
        b, a = starts[i].get("reach_before") or {}, starts[i].get("reach_after") or {}
        r["built"] = a.get("build", 0) - b.get("build", 0)
        r["hits"] = a.get("hit", 0) - b.get("hit", 0)
        out.append(r)
    return d["stamp"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stamp, rs = rounds(args.events)

    print("stamp:", json.dumps({k: stamp.get(k) for k in
                                ("host", "card", "commit", "triatt_hifi", "binder_pinned",
                                 "arms", "verify_rounds", "reach_end")}))
    hdr = "%5s %5s %7s %8s %8s %8s %8s %8s %7s %7s %6s %6s"
    print(hdr % ("round", "arm", "verify", "wall", "device", "evo", "extra", "tmpl",
                 "aiclk", "load1", "built", "hits"))
    for r in rs:
        print("%5d %5s %7s %8.3f %8.3f %8.3f %8.3f %8.3f %7s %7.1f %6d %6d"
              % (r["round"], r["arm"], r["verify"], r["wall"], r["device"], r["evoformer.s"],
                 r["extra_msa.s"], r["template.s"], r["aiclk_min"], r["load1"],
                 r["built"], r["hits"]))

    timed = [r for r in rs if r["round"] > 1 and not r["verify"]]
    arms = sorted({r["arm"] for r in timed})
    med = {a: {k: st.median([r[k] for r in timed if r["arm"] == a]) for k in KEYS}
           for a in arms}
    n = {a: sum(1 for r in timed if r["arm"] == a) for a in arms}
    print("\ntimed rounds: " + ", ".join(f"{a} n={n[a]}" for a in arms))
    print("%-14s" % "metric" + "".join("%10s" % a for a in arms) + "%10s" % "x(off/on)")
    for k in KEYS:
        line = "%-14s" % k + "".join("%10.3f" % med[a][k] for a in arms)
        if len(arms) == 2 and med[arms[1]][k]:
            line += "%10.4f" % (med[arms[0]][k] / med[arms[1]][k])
        print(line)
    clk = [r["aiclk_min"] for r in timed if r["aiclk_min"]]
    print("AICLK over every timed round: min %s, median of per-round minima %s"
          % (min(clk), st.median(sorted(clk))))
    print("load1 median %.1f  (per arm: %s)"
          % (st.median([r["load1"] for r in timed]),
             ", ".join("%s %.1f" % (a, st.median([r["load1"] for r in timed if r["arm"] == a]))
                       for a in arms)))
    v = stamp.get("reach_end", {})
    print("verify: ok %d, MISMATCH %d" % (v.get("verify_ok", 0), v.get("verify_bad", 0)))

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"stamp": stamp, "rounds": rs, "timed_n": n, "median": med}, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
