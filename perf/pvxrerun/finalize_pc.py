"""Turn the benchlocked pc run into the state doc's numeric sections.

Everything here is computed from the run's own artifacts (run.err, aiclk.jsonl, loadavg.log)
plus the previous pass's benchlocked qb1 672, which is the only outside number and is quoted
with its provenance every time it is used.
"""
import json, re, statistics, subprocess, sys
from pathlib import Path

RUN = Path("perf/pvxrerun/quietpc")
WEIGHT = {672: 29, 704: 32, 736: 32, 768: 24}
QB1_672 = [157.662, 157.862, 158.187]          # pvx-custchart, benchlocked, stock p150a, loadavg 1.06-2.23
QB1_704, QB1_768 = 174.478, 183.542            # same arm
RATIO3 = 1.195                                  # pvx-custchart-rerun's matched-load 736/672

out = subprocess.run([sys.executable, "perf/pvxrerun/mean_pc.py", str(RUN / "run.err")],
                     capture_output=True, text=True).stdout
print(out)

rows = []
for l in out.splitlines():
    m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(yes|no)\s+([\d.]+)\s+(\S+)\s+(\S+)", l)
    if m:
        rows.append(dict(job=int(m.group(1)), bucket=int(m.group(4)),
                         cold=m.group(5) == "yes", s=float(m.group(6)),
                         clk=m.group(7), load=m.group(8)))
warm = {}
for r in rows:
    if not r["cold"]:
        warm.setdefault(r["bucket"], []).append(r)

print("\n" + "=" * 78)
if set(warm) != set(WEIGHT):
    print(f"INCOMPLETE: warm buckets {sorted(warm)}; need {sorted(WEIGHT)}")
    sys.exit(0)

per = {b: statistics.median(x["s"] for x in v) for b, v in warm.items()}
mean = sum(WEIGHT[b] * per[b] for b in WEIGHT) / sum(WEIGHT)

# the in-run 736/672 ratio: both buckets, one process, one card
ratio = per[736] / per[672]
# the cross-calibrator: this run's 672 against the previous pass's benchlocked qb1 672
qb1 = statistics.median(QB1_672)
factor = per[672] / qb1
corrected = {b: per[b] / factor for b in per}
cmean = sum(WEIGHT[b] * corrected[b] for b in WEIGHT) / sum(WEIGHT)

print(f"in-run 736/672 ratio (load-invariant, one process): {ratio:.4f}")
print(f"  against pvx-custchart-rerun's matched-load ratio3 {RATIO3}: "
      f"{100 * (ratio - RATIO3) / RATIO3:+.1f} %")
print(f"\npc/qb1 factor at 672: {per[672]:.1f} / {qb1:.3f} = {factor:.4f}")
print("load-corrected to the benchlocked qb1 frame:")
for b in sorted(corrected):
    extra = ""
    if b == 704: extra = f"   (qb1 benchlocked measured {QB1_704})"
    if b == 768: extra = f"   (qb1 benchlocked measured {QB1_768})"
    print(f"  bucket {b}: {corrected[b]:8.3f} s{extra}")
print(f"\nPLAIN MEAN, pc as measured:        {mean:.3f} s")
print(f"PLAIN MEAN, corrected to qb1 frame: {cmean:.3f} s")
print(f"736 quiet, cross-calibrated:        {corrected[736]:.3f} s   "
      f"(brief's bracket 185-199 s: {'INSIDE' if 185 <= corrected[736] <= 199 else 'OUTSIDE'})")
print(f"736 via ratio3 x qb1 672:           {RATIO3 * qb1:.3f} s")

# does the correction reproduce the two buckets qb1 also measured? that is its own test
for b, q in ((704, QB1_704), (768, QB1_768)):
    print(f"CONTROL bucket {b}: corrected {corrected[b]:.3f} vs qb1 benchlocked {q}  "
          f"-> {100 * (corrected[b] - q) / q:+.2f} %")
