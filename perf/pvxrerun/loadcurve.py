"""Pair every completed fold with the host loadavg measured DURING it.

A fold line carries its own duration, so its window is [end - seconds, end]. The loadavg
sampler writes one line per 10 s with a UTC HH:MM:SS stamp, so the samples inside that
window are the load the fold actually ran under.
"""
import re, statistics
from pathlib import Path

R = Path("perf/pvxrerun")

def secs(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s

rows = []
for tag in ("run2", "run3", "armNEW", "armOLD", "runq"):
    err, lg = R / tag / "run.err", R / tag / "loadavg.log"
    if not err.exists():
        continue
    samples = []
    if lg.exists():
        for line in lg.read_text().splitlines():
            p = line.split()
            # run2 logged loadavg without a timestamp, so it carries no window information
            # and its folds correctly come out with n=0 rather than a load taken from
            # whichever sample happened to be adjacent.
            if len(p) >= 2 and ":" in p[0]:
                samples.append((secs(p[0]), float(p[1])))
    seen = set()
    for line in err.read_text(errors="replace").splitlines():
        m = re.search(r"^(\d\d:\d\d:\d\d).*?(\d\d)_b(\d+)_tok(\d+)_bk(\d+)\s+\S*\s*[-\u2014]+\s*([\d.]+)s\s*$", line)
        if not m:
            continue
        key = (tag, m.group(2))
        if key in seen:
            continue
        seen.add(key)
        end, job, bk, sec = secs(m.group(1)), int(m.group(2)), int(m.group(5)), float(m.group(6))
        win = [v for t, v in samples if end - sec <= t <= end]
        med = statistics.median(win) if win else None
        rows.append((tag, job, bk, sec, len(win), med, min(win) if win else None, max(win) if win else None))

print("run      job bucket        s    n  load_med    min    max")
for t, j, bk, s, n, med, lo, hi in rows:
    f = lambda x: "--" if x is None else ("%.2f" % x)
    print("%-8s %3d %6d %8.1f %4d %9s %6s %6s" % (t, j, bk, s, n, f(med), f(lo), f(hi)))
