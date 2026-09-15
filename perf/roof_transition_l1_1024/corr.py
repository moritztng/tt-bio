"""For every 1024/768 aa fold this session: did it wedge, and how long until qb2 next rebooted?

The wedge was read as an engine defect for two passes. qb2 rebooted twelve times in the two hours
this ladder ran, from a hardware watchdog, so the competing explanation is that the "wedge" is the
leading edge of the host failure rather than anything the fold computes. That is a testable
correlation and this is the test: a wedge that is an engine defect has no reason to sit a few
minutes in front of a reboot, and a completion has no reason to avoid one.

Start time comes from the run's own JSON when it wrote one, and from the log mtime otherwise --
a wedged run is killed before it can write a verdict, so the log is all it leaves.
"""
import glob
import json
import os
import time
from pathlib import Path

OUT = Path("/home/ttuser/.coworker/wt/roof-transition-l1-1024-overflow-fix/"
           "perf/roof_transition_l1_1024/out")

boots = []
for ln in Path("/home/ttuser/qbcard/cardtel.tsv").read_text().splitlines():
    if ln.startswith("#BOOT"):
        boots.append(float(ln.split("\t")[1]))
boots.sort()

rows = []
for lg in sorted(OUT.glob("*.log")):
    stem = lg.with_suffix("").name
    js = OUT / (stem + ".json")
    size = arm = None
    verdict = "WEDGED"
    start = os.path.getmtime(lg)
    if js.exists():
        try:
            d = json.loads(js.read_text())
            size, arm = d.get("size"), d.get("arm")
            verdict = d.get("verdict", "WEDGED")
            if d.get("started_utc"):
                start = time.mktime(time.strptime(d["started_utc"], "%Y-%m-%dT%H:%M:%SZ")) \
                    - time.timezone
        except Exception:
            pass
    if size not in (768, 1024):
        continue
    nxt = [b for b in boots if b > start]
    dt = (nxt[0] - start) / 60.0 if nxt else None
    rows.append((stem, size, arm, verdict, start, dt))

rows.sort(key=lambda r: r[4])
print("%-30s %5s %-5s %-7s %-9s %s" % ("run", "size", "arm", "verdict", "started", "min to next boot"))
for stem, size, arm, verdict, start, dt in rows:
    print("%-30s %5s %-5s %-7s %-9s %s" % (
        stem, size, arm, verdict, time.strftime("%H:%M:%S", time.gmtime(start)),
        "n/a" if dt is None else "%.1f" % dt))

for want in ("WEDGED", "DONE"):
    sel = [r[5] for r in rows if r[3] == want and r[5] is not None]
    if sel:
        print("\n%-7s n=%d  median %.1f min to next boot  (min %.1f, max %.1f)"
              % (want, len(sel), sorted(sel)[len(sel) // 2], min(sel), max(sel)))
