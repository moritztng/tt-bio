"""The customer's statistic off a four-bucket run, with each fold's own clock and load beside it.

Differs from mean.py in two ways, both needed by the benchlocked pc run:

* warm is "not the first fold of THIS shape", found by first occurrence, not "job > 1".
  The pc run folds the buckets in a palindrome, so jobs 1-4 are the cold ones, not job 1.
* every fold gets the AICLK and the loadavg sampled inside its OWN window, so no fold-time
  is quoted without the clock it was measured at.
"""
import json, re, statistics, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

WEIGHT = {672: 29, 704: 32, 736: 32, 768: 24}   # integer binder lengths, 64..180 aa, 117 total

log = Path(sys.argv[1])
run = log.parent
text = log.read_text(errors="replace")

# "13:47:00  [pc:tt0] 01_b78_tok658_bk672" starts a job; the same stem with a time ends it.
start_re = re.compile(r"^(\d\d):(\d\d):(\d\d)\s+\S+\s+(\d\d)_b(\d+)_tok(\d+)_bk(\d+)\s*$")
done_re = re.compile(r"(\d\d):(\d\d):(\d\d).*?(\d\d)_b(\d+)_tok(\d+)_bk(\d+)\s+\S*\s*[-—]+\s*([\d.]+)s\s*$")

day = datetime.strptime(re.search(r"START (\S+)", (run / "run.log").read_text()).group(1),
                        "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
# run.err stamps are host-LOCAL (CEST here); loadavg.log is written with `date -u`, so it is
# already UTC. Mixing the two unconverted shifts every fold window by the UTC offset and every
# per-fold clock and load reads empty. OFFSET is taken from the run's own two clocks: run.log's
# START is UTC, and run.err's first line is the same moment in local time.
_first_local = re.search(r"^(\d\d):(\d\d):(\d\d)", text, re.M)
OFFSET = timedelta(hours=round((
    int(_first_local.group(1)) * 3600 + int(_first_local.group(2)) * 60
    + int(_first_local.group(3)) - (day.hour * 3600 + day.minute * 60 + day.second)
) / 3600.0))

def stamp(h, m, s, local=True):
    t = day.replace(hour=int(h), minute=int(m), second=int(s))
    if t < day - timedelta(hours=12):       # the run crossed midnight
        t += timedelta(days=1)
    return t - OFFSET if local else t

starts = {}
for line in text.splitlines():
    m = start_re.match(line.strip())
    if m:
        starts[int(m.group(4))] = stamp(*m.groups()[:3])

rows = []
for line in text.splitlines():
    m = done_re.search(line.strip())
    if m:
        job = int(m.group(4))
        end = stamp(*m.groups()[:3])
        rows.append({"job": job, "binder": int(m.group(5)), "tokens": int(m.group(6)),
                     "bucket": int(m.group(7)), "seconds": float(m.group(8)),
                     "end": end, "start": end - timedelta(seconds=float(m.group(8)))})
if not rows:
    print("no completed fold lines in", log); sys.exit(1)
rows.sort(key=lambda r: r["job"])

# clock + load inside each fold's own window. aiclk.jsonl is unix time; loadavg.log is local HH:MM:SS.
clk = [json.loads(l) for l in (run / "aiclk.jsonl").read_text().splitlines() if l.strip()] \
    if (run / "aiclk.jsonl").exists() else []
load = []
if (run / "loadavg.log").exists():
    for l in (run / "loadavg.log").read_text().splitlines():
        p = l.split()
        if len(p) >= 2:
            h, m, s = p[0].split(":")
            load.append((stamp(h, m, s, local=False), float(p[1])))

def window(r, series, key):
    lo, hi = r["start"].timestamp(), r["end"].timestamp()
    if key == "clk":
        return [x["aiclk"] for x in series if lo <= x["t"] <= hi]
    return [v for t, v in series if lo <= t.timestamp() <= hi]

seen, cold, warm = set(), {}, {}
for r in rows:
    r["cold"] = r["bucket"] not in seen
    seen.add(r["bucket"])
    (cold if r["cold"] else warm).setdefault(r["bucket"], []).append(r["seconds"])
    a, l = window(r, clk, "clk"), window(r, load, "load")
    r["aiclk"] = (min(a), statistics.median(a), max(a), len(a)) if a else None
    r["load"] = (min(l), max(l), len(l)) if l else None

print(f"{len(rows)} completed folds   tree "
      + re.search(r"tree: (\S+)", (run / "run.log").read_text()).group(1)[:9])
print(f"{'job':>3} {'binder':>6} {'tok':>5} {'bucket':>6} {'first?':>6} {'s':>8}"
      f" {'AICLK min/med/max (n)':>26} {'loadavg lo-hi (n)':>20}")
for r in rows:
    a = f"{r['aiclk'][0]}/{r['aiclk'][1]:.0f}/{r['aiclk'][2]} ({r['aiclk'][3]})" if r["aiclk"] else "-"
    l = f"{r['load'][0]:.2f}-{r['load'][1]:.2f} ({r['load'][2]})" if r["load"] else "-"
    print(f"{r['job']:>3} {r['binder']:>6} {r['tokens']:>5} {r['bucket']:>6}"
          f" {'yes' if r['cold'] else 'no':>6} {r['seconds']:>8.1f} {a:>26} {l:>20}")

def stat(d, label):
    if set(d) != set(WEIGHT):
        print(f"\n{label}: buckets {sorted(d)} -- incomplete, no mean taken")
        return None
    per = {b: statistics.median(v) for b, v in d.items()}
    mean = sum(WEIGHT[b] * per[b] for b in WEIGHT) / sum(WEIGHT.values())
    lengths = sorted(x for b in WEIGHT for x in [per[b]] * WEIGHT[b])
    print(f"\n{label}")
    for b in sorted(per):
        print(f"  bucket {b}: {per[b]:8.3f} s  x{WEIGHT[b]:>2} lengths  (n={len(d[b])}: "
              + ", ".join(f"{x:.1f}" for x in d[b]) + ")")
    print(f"  PLAIN MEAN over 117 binder lengths: {mean:.3f} s")
    print(f"  median {statistics.median(lengths):.3f} s   spread "
          f"{max(per.values()) - min(per.values()):.3f} s")
    return mean

stat(warm, "WARM (first fold of each shape dropped)")
stat(cold, "FIRST-OF-SHAPE only")
allf = {}
for r in rows:
    allf.setdefault(r["bucket"], []).append(r["seconds"])
stat(allf, "ALL FOLDS")

for b, v in sorted(allf.items()):
    if len(v) > 1:
        print(f"\nbucket {b} cold vs warm: " + ", ".join(f"{x:.1f}" for x in v)
              + f"  -> compilation {max(v) - min(v):.1f} s")

if clk:
    a = [x["aiclk"] for x in clk]
    print(f"\nAICLK over the whole run, n={len(a)}: min {min(a)} / median "
          f"{statistics.median(a):.0f} / max {max(a)} MHz;  "
          f"{sum(1 for x in a if x < 1200)} samples below 1200 MHz;  "
          f"power {min(x['power'] for x in clk):.0f}-{max(x['power'] for x in clk):.0f} W, "
          f"peak temp {max(x['temp'] for x in clk):.1f} C")
if load:
    lv = [v for _, v in load]
    print(f"loadavg1 over the whole run, n={len(lv)}: min {min(lv):.2f} / median "
          f"{statistics.median(lv):.2f} / max {max(lv):.2f}")
