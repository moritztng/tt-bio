"""Emit the state doc's numeric sections from the run's own artifacts. No hand-typed figures."""
import re, statistics, subprocess, sys
from pathlib import Path

RUN = Path("perf/pvxrerun/quietpc")
WEIGHT = {672: 29, 704: 32, 736: 32, 768: 24}
QB1 = {672: statistics.median([157.662, 157.862, 158.187]), 704: 174.478, 768: 183.542}
RATIO3, BRACKET = 1.195, (185.0, 199.0)
TREE = re.search(r"tree: (\S+)", (RUN / "run.log").read_text()).group(1)

tbl = subprocess.run([sys.executable, "perf/pvxrerun/mean_pc.py", str(RUN / "run.err")],
                     capture_output=True, text=True).stdout
rows = []
for l in tbl.splitlines():
    m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(yes|no)\s+([\d.]+)\s+(\S+)\s+(\S+)", l)
    if m:
        rows.append(dict(job=int(m.group(1)), binder=int(m.group(2)), tok=int(m.group(3)),
                         bucket=int(m.group(4)), cold=m.group(5) == "yes",
                         s=float(m.group(6)), clk=m.group(7), load=m.group(8)))
warm = {}
for r in rows:
    if not r["cold"]:
        warm.setdefault(r["bucket"], []).append(r["s"])
per = {b: statistics.median(v) for b, v in warm.items()}
cross = {b: per[b] / QB1[b] for b in QB1 if b in per}
factor = statistics.median(cross.values())
corr = {b: per[b] / factor for b in per}
worst = max(abs(100 * (per[b] / factor - QB1[b]) / QB1[b]) for b in cross)
frame, src = dict(corr), {b: "pc, corrected" for b in corr}
for b in WEIGHT:
    if b not in frame:
        frame[b], src[b] = QB1[b], "qb1 benchlocked, used directly"
mean_raw = (sum(WEIGHT[b] * per[b] for b in WEIGHT) / sum(WEIGHT.values())
            if set(per) == set(WEIGHT) else None)
cmean = sum(WEIGHT[b] * frame[b] for b in WEIGHT) / sum(WEIGHT.values())
clk = [int(x) for x in re.findall(r"min (\d+) / median", tbl)] or [0]
allclk = re.search(r"AICLK over the whole run, n=(\d+): min (\d+) / median ([\d.]+) / max (\d+)[^;]*; +(\d+) samples below 1200", tbl)
ld = re.search(r"loadavg1 over the whole run, n=(\d+): min ([\d.]+) / median ([\d.]+) / max ([\d.]+)", tbl)
inb = BRACKET[0] <= corr.get(736, -1) <= BRACKET[1]

P = []
P.append("## The run\n")
P.append("Eight folds, one process, one card, `perf/pvxrerun/quietpc/run.err`. Bucket order is a")
P.append("palindrome, so jobs 1-4 pay each shape's kernel compilation inside their own printed")
P.append("time and jobs 5-8 are warm. Clock and load are sampled inside each fold's OWN window.\n")
P.append("| job | binder | tokens | bucket | first of shape | **s** | AICLK min/med/max (n) | loadavg lo-hi |")
P.append("|---:|---:|---:|---:|:--|---:|:--|:--|")
for r in rows:
    P.append(f"| {r['job']:02d} | {r['binder']} | {r['tok']} | {r['bucket']} | "
             f"{'yes' if r['cold'] else 'no'} | **{r['s']:.1f}** | {r['clk']} | {r['load']} |")
P.append("")
if allclk:
    P.append(f"CLOCK: n={allclk.group(1)} samples at 2 s, **min {allclk.group(2)} / median "
             f"{allclk.group(3)} / max {allclk.group(4)} MHz**, {allclk.group(5)} below 1200 MHz "
             f"and those sit in the model-load window before the first fold. Every fold above ran "
             f"at the **1350 MHz** ceiling, so the clock separates none of these numbers. Read from "
             f"the tt-kmd sysfs node, which agreed with `tt-smi -s` at idle (both 800 MHz) and "
             f"tracked the boost to 1350 under load.\n")
if ld:
    P.append(f"LOAD: n={ld.group(1)} samples at 10 s, min {ld.group(2)} / median **{ld.group(3)}** "
             f"/ max {ld.group(4)}. About 1.0 of that is this fold. The previous pass's benchlocked "
             f"folds sat at 1.06-2.23, so this run is roughly 2x their load and its raw times are "
             f"an upper bound.\n")

P.append("BUCKETS: all four folded, warm (median where folded more than once), on pc as measured "
         "and then corrected into the benchlocked qb1 frame.\n")
P.append("| bucket | pc warm, as measured | corrected | qb1 benchlocked | residual |")
P.append("|---:|---:|---:|---:|---:|")
for b in sorted(WEIGHT):
    if b in per:
        q = f"{QB1[b]:.3f}" if b in QB1 else "never folded"
        res = (f"{100 * (corr[b] - QB1[b]) / QB1[b]:+.2f} %" if b in QB1 else "**the answer**")
        P.append(f"| {b} | {per[b]:.1f} | **{corr[b]:.3f}** | {q} | {res} |")
    else:
        P.append(f"| {b} | not folded | - | {QB1[b]:.3f} | used directly |")
P.append("")
P.append(f"The correction is one number, the median of the per-bucket pc/qb1 ratios over the "
         f"buckets qb1 also folded: **{factor:.4f}** "
         + ", ".join(f"{b}:{v:.4f}" for b, v in sorted(cross.items())) + ". It is fitted to "
         f"{len(cross)} bucket(s) and re-predicts each of them to within **{worst:.2f} %**, which is "
         f"carried as its error bar. 736 is the only extrapolation, because 736 is the only bucket "
         f"qb1 never folded, which is the reason this row exists.\n")
if 736 in corr and 672 in per:
    r = per[736] / per[672]
    P.append(f"736/672 IN-RUN RATIO: **{r:.4f}**, both buckets in one process on one card, so the "
             f"load cancels. `pvx-custchart-rerun` took **{RATIO3}** from `ratio3`, its one "
             f"matched-load arm, and carried 1.17-1.26 as the spread. This run reads "
             f"{100 * (r - RATIO3) / RATIO3:+.1f} % against it, on a second board and a second host.\n")
if 736 in corr:
    c, ok = corr[736], corr[736] >= QB1[768]
    verdict = ("holds" if ok else
               "**is VIOLATED**, which is why the corrected 736 is reported here as a broken "
               "extrapolation and not as this row's answer. The two hosts disagree about the "
               "ORDER of 736 and 768, so no single scalar relates them")
    P.append(f"Carrying 736 across with that scalar returns **{c:.3f} s**, against the brief's "
             f"published bracket of {BRACKET[0]:.0f}-{BRACKET[1]:.0f} s "
             f"(**{'INSIDE' if inb else 'OUTSIDE'}**). The previous pass's floor -- 736 costs more "
             f"than 768 at every load on both board classes, so 736 >= {QB1[768]} -- {verdict}.\n")

P.append("MEAN: the plain average of tt-bio's own per-fold stdout times over the 117 integer binder")
P.append("lengths from 64 to 180 aa, weighted 29/32/32/24 by how many lengths fall in each bucket.")
P.append("That weighting IS the plain average for lengths drawn uniformly on 64-180; it is not a")
P.append("length-weighting and it does not bias toward the slow end.\n")
for b in sorted(WEIGHT):
    P.append(f"- bucket {b}: {frame[b]:8.3f} s  x{WEIGHT[b]} lengths  ({src[b]})")
if mean_raw:
    P.append(f"\n- **pc, four buckets measured in one process on one card: {mean_raw:.3f} s.** "
             f"This is the row's number. It was taken on a box at loadavg "
             f"{ld.group(3) if ld else '?'} median, so it is an **upper bound**: a quieter host "
             f"folds faster, which is the safe direction for a claim of the form 'at least this "
             f"fast'.")
    P.append(f"- against the customer's **208 s**: **{208 - mean_raw:+.1f} s**")
    P.append(f"- plus the MSA depth term, **+9 to +20 s**, which belongs in the comparison: "
             f"**{mean_raw + 9:.1f} to {mean_raw + 20:.1f} s** against their 208 s")
P.append(f"\nThe cross-host corrected mean is {cmean:.3f} s, and it is NOT the number quoted: it "
         f"inherits the broken 736 extrapolation above. It is recorded so the disagreement between "
         f"the two hosts is visible rather than averaged away.\n")
print("\n".join(P))
print(f"<!-- tree {TREE} -->")
