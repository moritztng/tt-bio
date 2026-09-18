#!/usr/bin/env python3
"""Score the program-cache A/B against its own A/A floor.

`bare` and `bare2` are the same arm under two labels, so their separation is this session's
repeatability including any position bias left after the per-rep rotation. A `keepcache` delta is
worth reporting only against that number, which is the field c12-host-decomp could not fill: it
had one clean rep per arm at 1350 MHz, so 0.0524 s had nothing to be scored against.
"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text())
valid = [x for x in r["rows"] if x.get("valid")]
arms = {}
for x in valid:
    arms.setdefault(x["arm"], []).append(x)

out = {"node": r["node"], "size": r["size"], "clocks": r["clocks"], "reps": r["reps"],
       "ttnn": r.get("ttnn_version"), "driver": r["before"]["module_srcversion"],
       "containment_unit": r["before"].get("containment_unit"),
       "port": r["before"].get("port", {}).get("port_COMMAND"),
       "errors": r["errors"], "completed": r["completed"], "arms": {}}

for a, rows in sorted(arms.items()):
    w = [x["elapsed_s"] for x in rows]
    out["arms"][a] = {
        "n": len(w), "median_s": round(st.median(w), 4), "mean_s": round(st.mean(w), 4),
        "min_s": round(min(w), 4), "max_s": round(max(w), 4),
        "spread_s": round(max(w) - min(w), 4),
        "walls_s": [round(x, 4) for x in w],
        "MHz": sorted({x["clock"]["max_MHz"] for x in rows} |
                      {x["clock"]["min_MHz"] for x in rows}),
        "W_range": [min(x["clock"]["min_W"] for x in rows),
                    max(x["clock"]["max_W"] for x in rows)],
        "cache_entries_after": sorted({x.get("program_cache_entries") for x in rows}),
        "cifs": sorted({v for x in rows for v in x["cif"].values()}),
        "plddt": sorted({x["plddt"] for x in rows}),
    }

b, b2, k = (out["arms"].get(n) for n in ("bare", "bare2", "keepcache"))
if b and b2:
    aa = round(abs(b["median_s"] - b2["median_s"]), 4)
    pooled = [x["elapsed_s"] for x in arms["bare"] + arms["bare2"]]
    out["AA_floor_s"] = aa
    out["AA_note"] = ("bare vs bare2, the same arm under two labels, medians differenced. "
                      "This is the smallest delta this session can resolve.")
    out["bare_pooled_median_s"] = round(st.median(pooled), 4)
    out["bare_pooled_spread_s"] = round(max(pooled) - min(pooled), 4)
    if k:
        d = round(out["bare_pooled_median_s"] - k["median_s"], 4)
        out["keepcache_delta_s"] = d
        out["keepcache_ratio"] = round(k["median_s"] / out["bare_pooled_median_s"], 5)
        out["resolved"] = abs(d) > aa
        out["verdict_arith"] = (
            f"delta {d:+.4f} s against an A/A floor of {aa:.4f} s: "
            + ("RESOLVED" if abs(d) > aa else "UNRESOLVED, |delta| <= floor"))
        e = [x for x in b["cache_entries_after"] if isinstance(x, int)]
        if e:
            out["programs_rebuilt"] = e
            out["per_program_s"] = [round(d / n, 6) for n in e]
            out["per_program_note"] = (
                "delta divided by the distinct programs a bare fold leaves cached. A rebuild "
                "rebuilds exactly that many, so this is the per-program host cost and it has to "
                "be plausible program assembly, not 10 ms.")

digests = {v for a in out["arms"].values() for v in a["cifs"]}
plddts = {v for a in out["arms"].values() for v in a["plddt"]}
out["bit_exact_across_all_arms"] = len(digests) == 1 and len(plddts) == 1
out["digests"] = sorted(digests)
out["plddts"] = sorted(plddts)
print(json.dumps(out, indent=2))
