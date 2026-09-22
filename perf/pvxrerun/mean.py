"""Read tt-bio's own stdout fold-time lines out of a run log and take the customer's statistic.

The customer said their 208 s is "the reported fold per fold time straight from tt bio's
stdout line" and "just the average" over 100 binders from 64 to 180 aa. One fold time per
token bucket, weighted by how many integer binder lengths land in that bucket, IS that plain
average for binder lengths drawn uniformly on 64-180.
"""
import json, re, statistics, sys
from pathlib import Path

# integer binder lengths per bucket over 64..180 aa, 117 lengths in total
WEIGHT = {672: 29, 704: 32, 736: 32, 768: 24}

log = Path(sys.argv[1] if len(sys.argv) > 1 else "perf/pvxrerun/run2/run.log")
rows = []
for line in log.read_text(errors="replace").splitlines():
    m = re.search(r"(\d\d)_b(\d+)_tok(\d+)_bk(\d+)\s+[^\s]*\s*[-—]+\s*([\d.]+)s\s*$", line)
    if m:
        rows.append({"job": int(m.group(1)), "binder": int(m.group(2)),
                     "tokens": int(m.group(3)), "bucket": int(m.group(4)),
                     "seconds": float(m.group(5))})
if not rows:
    print("no completed fold lines in", log); sys.exit(1)

print(f"{len(rows)} completed folds")
for r in rows:
    print(f"  job {r[\"job\"]:02d}  binder {r[\"binder\"]:3d} aa  {r[\"tokens\"]} tok  "
          f"bucket {r[\"bucket\"]}  {r[\"seconds\"]:.1f} s")

by_bucket, warm = {}, {}
for r in rows:
    by_bucket.setdefault(r["bucket"], []).append(r["seconds"])
    if r["job"] > 1:  # job 1 pays the run's kernel compilation inside its own printed time
        warm.setdefault(r["bucket"], []).append(r["seconds"])

def stat(d, label):
    if set(d) != set(WEIGHT):
        print(f"\n{label}: buckets {sorted(d)} -- incomplete, no mean taken")
        return None
    per = {b: statistics.median(v) for b, v in d.items()}
    mean = sum(WEIGHT[b] * per[b] for b in WEIGHT) / sum(WEIGHT.values())
    # the plain median over the same 117 binder lengths
    lengths = sorted(x for b in WEIGHT for x in [per[b]] * WEIGHT[b])
    print(f"\n{label}")
    for b in sorted(per):
        print(f"  bucket {b}: {per[b]:.3f} s  x{WEIGHT[b]} lengths  (n={len(d[b])}: "
              + ", ".join(f"{x:.1f}" for x in d[b]) + ")")
    print(f"  PLAIN MEAN over 117 binder lengths: {mean:.3f} s")
    print(f"  median: {statistics.median(lengths):.3f} s   "
          f"min bucket {min(per.values()):.3f}  max bucket {max(per.values()):.3f}  "
          f"spread {max(per.values()) - min(per.values()):.3f} s")
    return mean

stat(by_bucket, "ALL FOLDS (job 1 included, as a customer running one process would see it)")
stat(warm, "WARM (job 1 dropped)")

for b, v in sorted(by_bucket.items()):
    if len(v) > 1:
        print(f"\nA/A floor at bucket {b}: " + ", ".join(f"{x:.1f}" for x in v)
              + f"  -> spread {max(v) - min(v):.3f} s, {100 * (max(v) - min(v)) / statistics.median(v):.2f} %")

clk = log.parent / "aiclk.jsonl"
if clk.exists():
    s = [json.loads(l) for l in clk.read_text().splitlines() if l.strip()]
    a = [r["aiclk"] for r in s]
    print(f"\nAICLK during the run, n={len(a)}: min {min(a)} / median {statistics.median(a)} / "
          f"max {max(a)} MHz;  power {min(r[\"power\"] for r in s):.0f}-"
          f"{max(r[\"power\"] for r in s):.0f} W")
