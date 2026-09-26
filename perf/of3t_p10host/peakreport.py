#!/usr/bin/env python3
"""Peak host resident set and the host arithmetic it buys, arm by arm. Card-free."""
import json, statistics as st, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"
rows = {}
for tag in sys.argv[1:]:
    f = OUT / f"peak_{tag}.json"
    if not f.exists():
        print(f"{tag}: missing"); continue
    d = json.load(open(f))
    clk = d.get("env", {}).get("aiclk_during", {}).get("0", {})
    hq = d.get("arm", {}).get("host_quiet_pre", {})
    print(f"\n=== {tag}  commit {d['env']['commit'][:9]}  "
          f"AICLK {clk.get('min')}/{clk.get('median')}/{clk.get('max')} n={clk.get('n')}  "
          f"host_quiet {hq.get('green')} loadavg {hq.get('loadavg',[None])[0]}")
    peak = d.get("peak_rss_gib_run") or max(
        r["peak_rss_gib"] for r in d["reps"] if r.get("peak_rss_gib"))
    print(f"    PEAK RSS, steady (VmHWM): {peak} GiB")
    for r in d["reps"]:
        rss = r.get("rss_gib", {})
        av = r.get("mem_available_gib", {})
        print(f"    rep {r['rep']}{' (cold)' if r['cold'] else ''}: peak {r.get('peak_rss_gib')} GiB"
              f"  step {r.get('step_s')}  losses {r.get('losses_s')}  opt {r.get('optimizer_s')}")
        print("       RSS   " + "  ".join(f"{k}={v}" for k, v in rss.items()))
        print("       avail " + "  ".join(f"{k}={v}" for k, v in av.items()))
    steady = d["reps"][1:] or d["reps"]
    rows[tag] = {
        "peak": d.get("peak_rss_gib_run") or max(
            r["peak_rss_gib"] for r in d["reps"] if r.get("peak_rss_gib")),
        "rep_peak": st.median([r["peak_rss_gib"] for r in steady if r.get("peak_rss_gib")]),
        "losses": st.median([r["losses_s"] for r in steady]),
        "opt": st.median([r["optimizer_s"] for r in steady]),
        "step": st.median([r["step_s"] for r in steady if r.get("step_s")]),
        "avail_at_loss": st.median([r["mem_available_gib"]["after_diffusion"] for r in steady]),
    }
if len(rows) == 2 and "old" in rows and "new" in rows:
    o, n = rows["old"], rows["new"]
    print("\n=== OLD -> NEW (steady medians)")
    for k, unit in (("peak", "GiB"), ("rep_peak", "GiB"), ("avail_at_loss", "GiB"),
                    ("losses", "s"), ("opt", "s"), ("step", "s")):
        print(f"    {k:14s} {o[k]:8.3f} -> {n[k]:8.3f} {unit}   delta {n[k]-o[k]:+7.3f}")
