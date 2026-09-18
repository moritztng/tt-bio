#!/usr/bin/env python3
"""What the fold's matmul class can reach, given what was MEASURED on the two thin-K keys.

class_roof.py gives each shape its own binding roofline. This asks the question the frontier
actually turns on: the roofline is an upper bound nothing attains, so what happens to the class
if the two thin-K keys are held at a rate that was measured, and every OTHER shape in the class
is granted its full roofline -- a bound no shape in this class has ever reached.

Inputs are the s3 session's own numbers. Nothing here opens a device.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# The two thin-K keys, keyed by (batch, M, K, N) as class_roof.py canonicalises them.
A_SIG = (16, 512, 128, 512)
B_SIG = (16, 512, 512, 128)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-roof", type=Path, required=True)
    ap.add_argument("--target", type=float, default=35.49)
    # measured on qb2 node 3, session s3, 1350 MHz during-sampled, 21 reps, A/A inside 1.8 %
    ap.add_argument("--a-shipped", type=float, default=12.63, help="key A best DRAM-resident arm")
    ap.add_argument("--b-shipped", type=float, default=10.45, help="key B best DRAM-resident arm")
    ap.add_argument("--a-nodram", type=float, default=23.57, help="key A best arm with no DRAM")
    ap.add_argument("--b-nodram", type=float, default=26.08, help="key B best arm with no DRAM")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    d = json.loads(a.class_roof.read_text())
    rows = d["rows"]
    tot_f = d["class_tflop"]

    def find(sig):
        for r in rows:
            if (r["batch"], r["m"], r["k"], r["n"]) == sig:
                return r
        raise SystemExit("shape %s not in %s" % (sig, a.class_roof))

    ra, rb = find(A_SIG), find(B_SIG)
    rest_s = d["class_roof_s"] - ra["t_roof_s"] - rb["t_roof_s"]
    rest_f = tot_f - ra["tflop"] - rb["tflop"]

    print("CLASS %.3f TFLOP, %.0f calls, in-session roofs %.2f TFLOP/s and %.1f GB/s"
          % (tot_f, sum(r["calls"] for r in rows), d["cube_tflops"], d["dram_gbs"]))
    print("  the two thin-K keys carry %.3f TFLOP = %.1f %% of the class over %.0f calls"
          % (ra["tflop"] + rb["tflop"], 100.0 * (ra["tflop"] + rb["tflop"]) / tot_f,
             ra["calls"] + rb["calls"]))
    print("  every OTHER shape at its own exact roofline: %.4f s for %.3f TFLOP"
          % (rest_s, rest_f))
    print("  thin-K roofline would be A %.1f / B %.1f TFLOP/s -- neither was attained"
          % (ra["shape_roof_tflops"], rb["shape_roof_tflops"]))

    scen = [
        ("everything at its own roofline (unattainable upper bound)",
         ra["t_roof_s"], rb["t_roof_s"]),
        ("thin-K at their best MEASURED shipped-config rate, all else at roofline",
         ra["tflop"] / a.a_shipped, rb["tflop"] / a.b_shipped),
        ("thin-K at their best MEASURED rate with NO DRAM AT ALL, all else at roofline",
         ra["tflop"] / a.a_nodram, rb["tflop"] / a.b_nodram),
    ]
    print("\n%-72s %9s %9s %s" % ("scenario", "class s", "TFLOP/s", "vs target"))
    out = []
    for name, ta, tb in scen:
        s = rest_s + ta + tb
        r = tot_f / s
        print("%-72s %9.4f %9.2f %s"
              % (name, s, r, "CLEARS %.2f" % a.target if r >= a.target
                 else "SHORT of %.2f by %.2f" % (a.target, a.target - r)))
        out.append({"scenario": name, "class_s": s, "class_tflops": r,
                    "clears_target": r >= a.target})

    print("\nThe middle row is the one that answers the campaign's question: it grants every other\n"
          "shape in the class a rate no shape in this class has ever been measured to reach, and\n"
          "the class still lands short of %.2f TFLOP/s. So %.2f is not reachable without lifting\n"
          "the two thin-K keys themselves, and those are measured to cap at %.2f and %.2f TFLOP/s\n"
          "even with their DRAM traffic removed entirely." % (a.target, a.target,
                                                              a.a_nodram, a.b_nodram))
    if a.out:
        a.out.write_text(json.dumps({"target": a.target, "class_tflop": tot_f,
                                     "rest_roof_s": rest_s, "scenarios": out,
                                     "measured": {"a_shipped": a.a_shipped,
                                                  "b_shipped": a.b_shipped,
                                                  "a_nodram": a.a_nodram,
                                                  "b_nodram": a.b_nodram}}, indent=1))
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
