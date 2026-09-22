#!/usr/bin/env python3
"""Score the narrow-q re-take against its own pre-registered falsifier.

Pass 17 pre-registered a 768 aa negative control: TT_BIO_TRIATT_NARROW_Q_FALLBACK cannot
reach the fallback list at a length the production q_chunk divides, so that arm MUST read
1.00x. If it moves, the instrument is measuring something other than the lever.

The A/B harness (perf/xmsoftmax/fold_ab_flip.py) labels arms by the flag it FLIPS, so with
--off-value 1 the arm named "off" is the one with the lever ENABLED and "on" is shipped
defaults. This script reads that convention out of the JSON rather than out of the log
prose, because the log prints the arm names and not their meaning.

Each leg also carries the AICLK and loadavg sampled DURING it, so a per-leg clock
difference can be separated from a lever effect. tt-smi reports AICLK as a right-aligned
string, hence the strip/int.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "perf/land_standing/out"


def samples(jl):
    rows = []
    for line in jl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        clk = r.get("aiclk", {}).get("0")
        rows.append((r["t"],
                     int(str(clk).strip()) if clk is not None else None,
                     r["loadavg"][0]))
    return rows


def during(rows, t0, t1):
    win = [(c, l) for (t, c, l) in rows if t0 <= t <= t1 and c is not None]
    if not win:
        return None
    clks = [c for c, _ in win]
    loads = [l for _, l in win]
    return {"n": len(win), "clk_med": statistics.median(clks), "clk_min": min(clks),
            "clk_max": max(clks),
            "pct_ge_1200": 100.0 * sum(c >= 1200 for c in clks) / len(clks),
            "load_med": statistics.median(loads)}


def main():
    rows = samples(OUT / "narrowq_retake_contention.jsonl")
    for name in ("narrowq_retake_896_qb2c0.json", "narrowq_retake_768_qb2c0.json"):
        p = OUT / name
        if not p.exists():
            print("MISSING %s" % p)
            continue
        d = json.loads(p.read_text())
        lever_arm = "off" if "=1" in d["off_arm"] else "on"
        shipped_arm = "on" if lever_arm == "off" else "off"
        for cell in d["cells"]:
            print("\n=== %s @ %s aa  (%s) ===" % (cell["model"], cell["rung"], name))
            print("  arm naming: %r -> arm \"off\"; %r -> arm \"on\""
                  % (d["off_arm"], d["on_arm"]))
            print("  so the LEVER-ENABLED arm is %r" % lever_arm)
            print("  %-12s%8s%9s%9s%9s%7s" % ("leg", "s", "clk_med", "clk_min", "%>=1200", "load"))
            for f in cell["folds"]:
                st = during(rows, f["t_start"], f["t_end"])
                lab = "%s%s%s" % (f["arm"], f["rep"], "*" if f["arm"] == lever_arm else " ")
                if st:
                    print("  %-12s%8.1f%9.0f%9.0f%8.0f%%%7.2f"
                          % (lab, f["runtime_s"], st["clk_med"], st["clk_min"],
                             st["pct_ge_1200"], st["load_med"]))
                else:
                    print("  %-12s%8.1f%35s" % (lab, f["runtime_s"], "no samples"))
            lev = statistics.median(cell[lever_arm])
            shp = statistics.median(cell[shipped_arm])
            eff = 100.0 * (shp - lev) / shp
            aa = cell["aa_spread_pct"]
            print("  lever-enabled median %.2f s   shipped median %.2f s" % (lev, shp))
            print("  effect %+.3f%% (%+.3f s)   A/A floor %.3f%%   effect/floor %.2fx"
                  % (eff, shp - lev, aa, abs(eff) / aa))
    print("\n* = lever enabled. The harness order is fixed off-then-on, so the")
    print("starred leg ran FIRST in every pair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
