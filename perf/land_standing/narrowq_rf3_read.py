#!/usr/bin/env python3
"""What the narrow-q fallback is worth on rf3 at 896 aa, and why the cell is still refused.

Pass 9 priced this lever on **boltz2** at 896 aa and read nothing: 43.5 / 43.5, under 0.1 s. That
was the wrong model. The run-time census already on disk says so without spending a fold --
`perf/sizegate/baseline/census_*_896_p300c.json`, TRIATT_PERSISTENT_MASK:

    boltz2   896 aa   served 560    declined 560    pm_over_l1 559
    rf3      896 aa   served 0      declined 2177   fill_preconditions 1088, pm_over_l1 1087

boltz2 loses the two wide candidates to L1 and still serves its production config, so the fallback
is never reached and the flag cannot do anything. rf3 is the one cell in the whole seven-model,
six-rung p300c baseline where `fill_preconditions` declines EVERY call -- the exact pathology the
lever was written for, and the same shape the 16.7 s Galaxy Wormhole reading came from
(`pm_over_l1` 1087, 0 of 1088). So the lever's target on Blackhole is rf3 at 896 aa, and the nine
passes that called it inert had measured a model it cannot reach.

This script reads the fold A/B that follows from that, and it exists rather than a paragraph
because the cell needs partitioning before it means anything: a co-tenant landed in rep 3 and the
harness's own A/A floor is a max-min spread over all three reps, which prices that co-tenant as
noise in both arms. Partitioning on the benchlock ceiling is the honest cut, and it is a cut the
reader can check.
"""
import json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AB = ROOT / "perf/land_standing/out/narrowq_rf3_896_qb2c1.json"
JL = ROOT / "perf/land_standing/out/narrowq_rf3_contention.jsonl"
CEILING = 2.00  # benchlock's loadavg1 ceiling


def main() -> int:
    cell = json.loads(AB.read_text())["cells"][0]
    recs = [json.loads(l) for l in JL.read_text().splitlines() if l.strip()]
    rows = []
    for f in cell["folds"]:
        ins = [r for r in recs if f["t_start"] <= r["t"] <= f["t_end"]]
        la = [r["loadavg"][0] for r in ins]
        clk = [int(c) for c in (r["aiclk"].get("1") for r in ins
                                if isinstance(r["aiclk"], dict)) if c]
        rows.append({"arm": f["arm"], "rep": f["rep"], "t": f["runtime_s"],
                     "load_max": max(la), "clk_peak": max(clk) if clk else None,
                     "quiet": max(la) <= CEILING})
    print("arm  rep  runtime_s  load1 max  AICLK peak  quiet")
    for r in rows:
        print("  %-3s  %d   %8.1f   %8.2f   %9s  %s"
              % (r["arm"], r["rep"], r["t"], r["load_max"], r["clk_peak"],
                 "yes" if r["quiet"] else "NO"))

    # The paired delta is the design's point: each rep runs both arms back to back, so a drift
    # that walks over the cell hits both members of a pair almost equally.
    print("\npaired, lever-on minus shipped, one line per rep:")
    for rep in sorted({r["rep"] for r in rows}):
        on = next(r for r in rows if r["rep"] == rep and r["arm"] == "off")   # 'off' sets the flag
        sh = next(r for r in rows if r["rep"] == rep and r["arm"] == "on")    # 'on' = shipped
        print("  rep%d  %+.1f s%s" % (rep, on["t"] - sh["t"],
                                      "" if on["quiet"] and sh["quiet"] else "   (contended)"))

    quiet = [r for r in rows if r["quiet"]]
    reps = sorted({r["rep"] for r in quiet}
                  & {r["rep"] for r in quiet if r["arm"] == "off"}
                  & {r["rep"] for r in quiet if r["arm"] == "on"})
    lev = [r["t"] for r in quiet if r["arm"] == "off" and r["rep"] in reps]
    shp = [r["t"] for r in quiet if r["arm"] == "on" and r["rep"] in reps]
    if len(lev) < 2 or len(shp) < 2:
        print("\nnot enough quiet reps to floor the cell")
        return 1
    aa = 100.0 * (max(lev) - min(lev)) / st.median(lev)
    ab = 100.0 * (st.median(shp) - st.median(lev)) / st.median(lev)
    print("\nquiet reps only (%s):" % ", ".join("rep%d" % r for r in reps))
    print("  lever on   %s   median %.2f s" % (lev, st.median(lev)))
    print("  shipped    %s   median %.2f s" % (shp, st.median(shp)))
    print("  A/A floor %+.3f%%   A/B %+.3f%%   %.4fx   %+.2f s   effect/floor %.1fx"
          % (aa, ab, st.median(shp) / st.median(lev), st.median(shp) - st.median(lev),
             abs(ab) / aa if aa else float("inf")))
    print("  separation: max lever-on %.1f %s min shipped %.1f"
          % (max(lev), "<" if max(lev) < min(shp) else ">=", min(shp)))

    # And the reason this is still not a reportable number.
    print("\nREFUSED, and the refusal is the sampler, not the card. `clock_during.py` needs the "
          "longest\nrun at or above 1200 MHz to CONTAIN the timed fold; at a 3 s cadence it "
          "proves 104 s inside a\n104.6 s fold and 94 s inside a 94.8 s one. Every shortfall is "
          "under one sampling interval and\nevery leg peaks at 1350 MHz, so nothing here reads as "
          "a throttle -- but a cell the instrument\nrefuses does not become a shipped second "
          "because its direction is plausible. Re-take at a 1 s\ncadence with four reps on a box "
          "that stays under the ceiling for the whole cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
