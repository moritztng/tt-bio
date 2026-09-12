#!/usr/bin/env python3
"""The byte axis as measured: block ratio and measured bytes per site, against an interleaved base.

Reads `site_screen.py` output. Every site arm is paired with the base arms measured either side of
it, which is the only defence against `_L1_OUT_RUNG` rewriting the baseline mid-run (it is a
module-level dict that only grows, and one arm's L1 refusal demotes the ladder for every arm after
it -- wave 1 watched its own incumbent move 5 % that way).
"""
import json
import statistics as st
import sys
from pathlib import Path


def main(path):
    d = json.load(open(path))
    arms = d["arms"]
    bases = [(a["order"], a["block_median_s"] * 1e3, a["graph"]["real_MB"])
             for a in arms if a["arm"] == "base" and "error" not in a]
    bt = [b[1] for b in bases]
    env = d["env"]
    print(f"{env['host']} card {env['card']} grid {env['grid'][0]}x{env['grid'][1]} "
          f"{env['arch']} seq {env['seq']} ttnn {env.get('ttnn')} reps {env['reps']}")
    print(f"base x{len(bt)}: median {st.median(bt):.2f} ms, spread "
          f"{(max(bt)-min(bt))/st.median(bt)*100:.3f} %, A/A floor {max(bt)/min(bt):.5f}x, "
          f"{bases[0][2]:.1f} MB")
    print(f"\n{'arm':14s} {'block_ms':>9s} {'ratio':>8s} {'MB':>9s} {'dbytes':>8s} "
          f"{'realiz':>7s}  note")
    for a in arms:
        if a["arm"] == "base":
            continue
        if "error" in a:
            print(f"{a['arm']:14s} {'THREW':>9s} {'':>8s} {'':>9s} {'':>8s} {'':>7s}  "
                  f"{a['error'].splitlines()[-1][:68]}")
            continue
        # the two bases nearest this arm in run order, so a mid-run baseline shift is visible
        near = sorted(bases, key=lambda b: abs(b[0] - a["order"]))[:2]
        b_ms = st.mean([n[1] for n in near])
        b_mb = st.mean([n[2] for n in near])
        ms, mb = a["block_median_s"] * 1e3, a["graph"]["real_MB"]
        db = (mb - b_mb) / b_mb
        r = b_ms / ms
        realiz = ((b_ms - ms) / b_ms) / -db if db else float("nan")
        gates = {k: v for k, v in (a.get("gates") or {}).items() if v.get("declined")}
        note = " ".join(f"{k.split('.')[-1]}:{v['declined']}decl" for k, v in gates.items())
        print(f"{a['arm']:14s} {ms:9.2f} {r:8.4f}x {mb:9.1f} {db*100:+7.2f}% {realiz:7.3f}  {note[:60]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "perf/b2z2_byte_axis/results/screen2_512_whglx_c1.json")
