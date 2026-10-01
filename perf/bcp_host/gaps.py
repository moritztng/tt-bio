#!/usr/bin/env python3
"""Host gaps between on-card seams per warm N=1 round, labelled by the seams either side.

    gaps.py out/<arm> ...     (rounds 1 and 2 dropped, medians over the rest)
"""
import collections, json, pathlib, statistics as st, sys

for p in sys.argv[1:]:
    d = json.loads((pathlib.Path(p) / "round_events.json").read_text()); ev = d["events"]
    b = sorted(e["t0"] for e in ev if e["kind"] == "round_start") +         [e["t0"] for e in ev if e["kind"] == "round_stop"][:1]
    gaps, seams = collections.defaultdict(list), collections.defaultdict(list)
    for i in range(2, len(b) - 1):
        t0, t1 = b[i], b[i + 1]
        dev = sorted((e for e in ev if e["kind"] == "device" and t0 - 1e-6 <= e["t0"] and e["t1"] <= t1 + 1e-6),
                     key=lambda e: e["t0"])
        prev, per, sp = ("round start", t0), collections.Counter(), collections.Counter()
        for e in dev:
            name = e["module"] + "/" + e["phase"]
            per[prev[0] + " -> " + name] += e["t0"] - prev[1]; sp[name] += e["dt"]
            prev = (name, e["t1"])
        per[prev[0] + " -> round end"] += t1 - prev[1]
        for k, v in per.items(): gaps[k].append(v)
        for k, v in sp.items(): seams[k].append(v)
    print(f"== {pathlib.Path(p).name}  host gaps (median s)")
    for k, v in sorted(gaps.items(), key=lambda x: -st.median(x[1])):
        print(f"  {st.median(v):7.3f}  {k}")
    print(f"  {sum(st.median(v) for v in gaps.values()):7.3f}  total")
    print("  seams:", ", ".join(f"{k} {st.median(v):.3f}" for k, v in sorted(seams.items())))
