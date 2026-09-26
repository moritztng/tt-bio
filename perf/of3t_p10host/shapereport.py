#!/usr/bin/env python3
"""Per-root and per-replicate loss cost, with the memory each root ran in. Card-free.

Answers the two things the re-brief asks for: whether the 2-to-4 sample discontinuity in
`losses_s` is arithmetic or host paging, and what one more diffusion replicate costs -- which
is the number a chunked 48-sample step is built out of.
"""
import json, statistics as st, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"

for tag in sys.argv[1:]:
    f = OUT / f"{tag}.json"
    if not f.exists():
        print(f"{tag}: missing"); continue
    d = json.load(open(f))
    clk = d.get("env", {}).get("aiclk_during", {}).get("0", {})
    print(f"\n=== {tag}  commit {d['env']['commit'][:9]}  "
          f"AICLK {clk.get('min')}/{clk.get('median')}/{clk.get('max')} n={clk.get('n')}  "
          f"avail_at_start {d.get('avail_at_start_gib')} GiB")
    hq = d.get("arm", {})
    print(f"    host_quiet pre {hq.get('host_quiet_pre', {}).get('green')} "
          f"post {hq.get('host_quiet_post', {}).get('green')}  "
          f"loadavg {hq.get('host_quiet_pre', {}).get('loadavg', [None])[0]}")
    for i, r in enumerate(d["reps"]):
        l = r.get("losses", {})
        mt = l.get("mem_trace_gib", [])
        print(f"    rep {i}: losses_s {r.get('losses_s')}  shape {l.get('shape')}  "
              f"step_s {r.get('step_s')}")
        print(f"      per_root_s {l.get('per_root_s')}")
        if l.get("once_terms_s") is not None:
            print(f"      once_terms_s {l['once_terms_s']}   "
                  f"per_replicate_s {l.get('per_replicate_s')}")
        print(f"      RSS {l.get('rss_gib_at_entry')} -> {l.get('rss_gib_at_exit')} GiB   "
              f"MemAvailable {l.get('mem_available_gib_at_entry')} -> "
              f"{l.get('mem_available_gib_at_exit')} GiB")
        if mt:
            print("      per-root avail GiB " + " ".join(f"{m['avail']:.2f}" for m in mt))

    steady = d["reps"][1:] or d["reps"]
    l = steady[-1].get("losses", {})
    if l.get("per_replicate_s") is not None:
        for n in (4, 8, 16, 48):
            print(f"    projected at {n:2d} replicates: "
                  f"{l['once_terms_s'] + (n - 1) * l['per_replicate_s']:.3f} s")
