#!/usr/bin/env python3
"""Line the two arms up phase by phase. The row's number comes out of here."""
import json, sys, statistics as st
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"
def load(tag):
    d = json.load(open(OUT / f"step_{tag}_384.json"))
    steady = d["reps"][1:]
    return d, steady
rows = {}
for tag in sys.argv[1:] or ["OLD", "NEW"]:
    d, steady = load(tag)
    rows[tag] = (d, steady)
    clk = d["env"].get("aiclk_during", {}).get("0", {})
    print(f"\n=== {tag}  commit {d['env']['commit'][:9]}  avail {d.get('avail_at_start_gib')} GiB "
          f"loadavg {d['env']['loadavg_start'][0]:.2f}->{d['env']['loadavg_end'][0]:.2f}")
    print(f"    AICLK during: min {clk.get('min')} median {clk.get('median')} max {clk.get('max')} "
          f"n={clk.get('n')}")
    for k in ("trunk_s", "diffusion_s", "losses_s", "backward_s", "optimizer_s", "step_s"):
        v = [r[k] for r in steady if r.get(k) is not None]
        if v:
            print(f"    {k:14s} median {st.median(v):8.3f}  reps {['%.3f'%x for x in v]}")
if len(rows) == 2:
    (do, so), (dn, sn) = rows["OLD"], rows["NEW"]
    print("\n=== OLD -> NEW (steady medians)")
    for k in ("losses_s", "backward_s", "optimizer_s", "step_s"):
        a = st.median([r[k] for r in so if r.get(k) is not None])
        b = st.median([r[k] for r in sn if r.get(k) is not None])
        print(f"    {k:14s} {a:8.3f} -> {b:8.3f}   delta {a-b:+7.3f} s   {a/b:6.4f}x")
