#!/usr/bin/env python3
"""Score the 1 s-cadence narrow-q retake, cutting legs on the benchlock loadavg ceiling.

The 4-rep cell was invaded mid-run by an of3t host-side f64 gradient job (856 %CPU, 14 threads)
and the harness refused the whole cell on its own A/A floor: +32.135 % against a +11.149 % A/B.
That refusal is correct and stands. This scores the sub-cell that stayed under the ceiling, and
reports the cut it made rather than the cut that flatters the lever: a fixed 2.00 loadavg applied
to each leg's OWN [t_start, t_end], fixed before the timings were read.

A cut cell is only reportable if it still carries an A/A floor, so a rep counts only when BOTH of
its interleaved arms survive.
"""
import json, statistics, sys

ART = "perf/land_standing/out/narrowq_rf3_retake1s_896.json"
CON = "perf/land_standing/out/narrowq_rf3_retake1s_contention.jsonl"
CEIL = 2.00

art = json.load(open(ART))
samples = [json.loads(l) for l in open(CON) if l.strip()]


def load_in(t0, t1):
    v = [s["loadavg"][0] for s in samples if t0 <= s["t"] <= t1]
    return (min(v), statistics.median(v), max(v), len(v)) if v else None


cell = art["cells"][0]
print("%s %s aa   off_arm=%s   on_arm=%s" % (cell["model"], cell["rung"], art["off_arm"], art["on_arm"]))
print("ceiling %.2f loadavg1, applied to each leg's own [t_start, t_end]\n" % CEIL)
print("%-14s %10s %9s %7s %7s  %s" % ("leg", "runtime_s", "load min", "med", "max", "verdict"))

legs = []
for f in cell["folds"]:
    lo = load_in(f["t_start"], f["t_end"])
    ok = lo is not None and lo[2] <= CEIL
    legs.append((f, lo, ok))
    print("%-14s %10.1f %9.2f %7.2f %7.2f  %s" % (
        "%s rep%d" % (f["arm"], f["rep"]), f["runtime_s"], lo[0], lo[1], lo[2],
        "KEEP" if ok else "CUT (over ceiling)"))

keep = [(f, lo) for f, lo, ok in legs if ok]
by_rep = {}
for f, _ in keep:
    by_rep.setdefault(f["rep"], {})[f["arm"]] = f["runtime_s"]
paired = {r: d for r, d in by_rep.items() if "off" in d and "on" in d}
print("\nkept legs: %d/%d   complete interleaved reps: %s" % (len(keep), len(legs), sorted(paired)))

off = [paired[r]["off"] for r in sorted(paired)]
on = [paired[r]["on"] for r in sorted(paired)]
if len(off) < 2:
    print("\nREFUSED: fewer than two complete reps survive the cut, so there is no A/A floor.")
    sys.exit(3)

omed, nmed = statistics.median(off), statistics.median(on)
aa_off = (max(off) - min(off)) / min(off) * 100
aa_on = (max(on) - min(on)) / min(on) * 100
aa = max(aa_off, aa_on)
ab = (nmed - omed) / omed * 100
print("\n  lever ON  (flag=1): %s  median %.2f s" % (off, omed))
print("  shipped   (flag=0): %s  median %.2f s" % (on, nmed))
print("  A/A floor  %+.3f %%  (worst within-arm spread; off %+.3f %%, on %+.3f %%)" % (aa, aa_off, aa_on))
print("  A/B        %+.3f %%   speedup %.4fx   delta %+.4f s" % (ab, nmed / omed, nmed - omed))
print("  effect / floor  %.2fx" % (ab / aa))
print("  separation: max lever-on %.1f vs min shipped %.1f -> %s" % (
    max(off), min(on), "NO OVERLAP" if max(off) < min(on) else "OVERLAP"))
ok = ab > aa and max(off) < min(on)
print("\nVERDICT: " + ("PASS -- A/B exceeds its own A/A floor and the arms do not overlap"
                       if ok else "REFUSED -- inside the floor or the arms overlap"))
