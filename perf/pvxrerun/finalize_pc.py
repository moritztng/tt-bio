"""Turn the benchlocked pc run into the state doc's numeric sections.

Everything is computed from the run's own artifacts (run.err, aiclk.jsonl, loadavg.log) plus the
previous pass's benchlocked qb1 buckets, which are the only outside numbers and carry their
provenance wherever they are used.

The correction is deliberately over-determined. qb1 measured 672, 704 and 768 benchlocked but
never 736, which is the whole reason this row exists. So the pc-to-qb1 factor is taken as the
MEDIAN of the per-bucket ratios over whichever of those three landed, and every one of them is
then re-predicted from that factor and its residual printed. A factor fitted to one bucket
explains that bucket by construction; a factor that reproduces three independent buckets it was
not individually fitted to has been tested. 736 is the only extrapolation and it is labelled as
one.
"""
import re, statistics, subprocess, sys
from pathlib import Path

RUN = Path("perf/pvxrerun/quietpc")
WEIGHT = {672: 29, 704: 32, 736: 32, 768: 24}
# pvx-custchart, benchlocked on a stock p150a (qb1), loadavg 1.06-2.23 sampled during the fold
QB1 = {672: statistics.median([157.662, 157.862, 158.187]), 704: 174.478, 768: 183.542}
RATIO3 = 1.195          # pvx-custchart-rerun's matched-load 736/672, its best prior estimate
BRACKET = (185.0, 199.0)

out = subprocess.run([sys.executable, "perf/pvxrerun/mean_pc.py", str(RUN / "run.err")],
                     capture_output=True, text=True).stdout
print(out)

rows = []
for l in out.splitlines():
    m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(yes|no)\s+([\d.]+)", l)
    if m:
        rows.append(dict(job=int(m.group(1)), bucket=int(m.group(4)),
                         cold=m.group(5) == "yes", s=float(m.group(6))))
warm = {}
for r in rows:
    if not r["cold"]:
        warm.setdefault(r["bucket"], []).append(r["s"])
per = {b: statistics.median(v) for b, v in warm.items()}

print("=" * 78)
print(f"warm buckets measured: {sorted(per)}")
if not per:
    print("nothing warm yet"); sys.exit(0)

if 736 in per and 672 in per:
    r = per[736] / per[672]
    print(f"\nIN-RUN 736/672 RATIO (both buckets, one process, one card, load-invariant): "
          f"{r:.4f}\n  against pvx-custchart-rerun's matched-load ratio3 {RATIO3}: "
          f"{100 * (r - RATIO3) / RATIO3:+.1f} %")

cross = {b: per[b] / QB1[b] for b in QB1 if b in per}
if not cross:
    print("\nno bucket measured on BOTH hosts yet -- no correction possible"); sys.exit(0)
factor = statistics.median(cross.values())
print(f"\npc-to-qb1 FACTOR = median of {len(cross)} per-bucket ratios "
      + ", ".join(f"{b}:{v:.4f}" for b, v in sorted(cross.items())) + f"  ->  {factor:.4f}")
print(f"  (a factor of {factor:.3f} says this pc run ran {100*(factor-1):+.1f} % against the "
      f"previous pass's benchlocked qb1 folds)")

print("\nRESIDUALS -- each cross-measured bucket re-predicted from the pooled factor:")
worst = 0.0
for b in sorted(cross):
    pred, obs = per[b] / factor, QB1[b]
    d = 100 * (pred - obs) / obs
    worst = max(worst, abs(d))
    print(f"  bucket {b}: corrected {pred:8.3f} s   qb1 benchlocked {obs:8.3f} s   {d:+6.2f} %")
print(f"  worst residual {worst:.2f} %  -- the correction's own error bar")

corrected = {b: per[b] / factor for b in per}
print("\nCORRECTED TO THE BENCHLOCKED qb1 FRAME:")
for b in sorted(corrected):
    tag = "  <- EXTRAPOLATION, qb1 never folded 736" if b == 736 else ""
    print(f"  bucket {b}: {corrected[b]:8.3f} s{tag}")

if 736 in corrected:
    c = corrected[736]
    lo, hi = c * (1 - worst / 100), c * (1 + worst / 100)
    print(f"\n736 QUIET, cross-calibrated: {c:.3f} s  (+-{worst:.2f} % -> {lo:.1f}-{hi:.1f} s)")
    print(f"  brief's published bracket {BRACKET[0]}-{BRACKET[1]} s: "
          f"{'INSIDE' if BRACKET[0] <= c <= BRACKET[1] else 'OUTSIDE'}")
    print(f"  prior estimate, ratio3 x qb1 672 = {RATIO3 * QB1[672]:.3f} s")
    print(f"  floor from the previous pass (736 > 768 at every load, both board classes): "
          f">= {QB1[768]} s -> {'holds' if c >= QB1[768] else 'VIOLATED'}")

if set(per) == set(WEIGHT):
    m = sum(WEIGHT[b] * per[b] for b in WEIGHT) / sum(WEIGHT.values())
    print(f"\nPLAIN MEAN over the 117 integer binder lengths (29/32/32/24)")
    print(f"  pc as measured, on a loaded box: {m:.3f} s")
# In the CORRECTED frame every bucket qb1 folded benchlocked can supply itself, so the mean
# only strictly needs 736 from pc -- which is the one bucket qb1 never folded and the reason
# this row exists. A bucket taken from qb1 is labelled, never silently substituted.
frame, src = dict(corrected), {b: "pc, corrected" for b in corrected}
for b in WEIGHT:
    if b not in frame and b in QB1:
        frame[b], src[b] = QB1[b], "qb1 benchlocked, used directly"
if set(frame) == set(WEIGHT):
    print("\nMEAN INPUTS:")
    for b in sorted(frame):
        print(f"  bucket {b}: {frame[b]:8.3f} s  ({src[b]})")
    cm = sum(WEIGHT[b] * frame[b] for b in WEIGHT) / sum(WEIGHT.values())
    lo, hi = cm * (1 - worst / 100), cm * (1 + worst / 100)
    print(f"  corrected to the benchlocked frame: {cm:.3f} s  ({lo:.1f}-{hi:.1f} s)")
    print(f"  against the customer's 208 s: {208 - cm:+.1f} s")
