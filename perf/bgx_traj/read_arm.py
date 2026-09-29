#!/usr/bin/env python3
"""What one repro arm actually did: the count it ran, the rounds it finished, the host it held.

`round_events.json` is flushed at every round boundary, so this reads an arm that died as well
as one that finished. The host high-water is `VmHWM` sampled at each boundary, which is the
figure `auto` is pricing when it chooses a count.
"""
import json
import sys

GB = 2**30

for path in sys.argv[1:]:
    d = json.load(open(path))
    st = d["stamp"]
    ev = d["events"]
    rounds = [e for e in ev if e["kind"] == "round_start"]
    per_slot = {}
    for e in rounds:
        per_slot[e.get("slot")] = per_slot.get(e.get("slot"), 0) + 1
    reach = [e["reach"] for e in rounds if e.get("reach")]
    hwm = max((r.get("vmhwm", 0) for r in reach), default=0)
    rss = max((r.get("vmrss", 0) for r in reach), default=0)
    avail = min((r.get("mem_available", 0) for r in reach if r.get("mem_available")), default=0)
    clk = sorted(x[1] for x in d.get("aiclk", []))
    gate = st.get("gate", {})
    print(json.dumps({
        "file": path, "trajectories": st.get("trajectories"),
        "interleave": st.get("interleave"), "binder": st.get("binder"),
        "wall_s": st.get("wall_seconds"), "rounds_per_slot": per_slot,
        "stopped": st.get("stopped"),
        "host_hwm_gb": round(hwm / GB, 3), "host_rss_gb": round(rss / GB, 3),
        "mem_available_min_gb": round(avail / GB, 1),
        "aiclk_median": clk[len(clk) // 2] if clk else None,
        "aiclk_min": clk[0] if clk else None,
        "aiclk_under_1200": sum(1 for c in clk if c < 1200),
        "gate_held_s": gate.get("held_s"), "gate_waited_s": gate.get("waited_s"),
    }))
