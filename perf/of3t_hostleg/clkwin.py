#!/usr/bin/env python3
"""Per-fold AICLK, over the FOLD's own window rather than the whole process.

`foldab.sh` samples the card's sysfs `tt_aiclk` at 1 Hz for the whole run, and a mean over that
window is diluted by the import and featurization phases, where the card is idle at 800 MHz: the
same 13 s fold read mean 1350 in one run and mean 1103 in another purely because the process
around it was longer. tt_bio's own per-target line gives the fold's end time and duration, so the
window is reconstructed from it and the clock is read DURING the timed work, which is the only
reading this fleet accepts.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/tmp/of3t/of3t-hostleg/fold")
rows = {}
for d in sorted(ROOT.glob("*")):
    log, clk = d / "fold.log", d / "aiclk.tsv"
    if not (log.is_file() and clk.is_file()):
        continue
    m = re.search(r"(\d\d):(\d\d):(\d\d)\s+\[[^\]]*\]\s+\S+\s+ubq\s+—\s+([\d.]+)s", log.read_text())
    if not m:
        continue
    hh, mm, ss, secs = int(m[1]), int(m[2]), int(m[3]), float(m[4])
    samples = [(int(a), int(b)) for a, b in
               (ln.split("\t") for ln in clk.read_text().splitlines() if "\t" in ln and ln.split("\t")[1])]
    if not samples:
        continue
    # the log stamps local time; anchor the day on the sampler's own epochs
    day = datetime.fromtimestamp(samples[0][0], timezone.utc).astimezone()
    end = day.replace(hour=hh, minute=mm, second=ss, microsecond=0).timestamp()
    start = end - secs
    win = [c for t, c in samples if start - 1 <= t <= end + 1]
    rows[d.name] = {
        "fold_s": secs,
        "aiclk_n": len(win),
        "aiclk_mean": round(sum(win) / len(win), 1) if win else None,
        "aiclk_min": min(win) if win else None,
        "aiclk_max": max(win) if win else None,
        "whole_run_mean": round(sum(c for _, c in samples) / len(samples), 1),
    }
    print(d.name, json.dumps(rows[d.name]), flush=True)

arms = {}
for k, v in rows.items():
    arm = k.rsplit("_", 1)[0]
    arms.setdefault(arm, []).append(v["fold_s"])
summary = {a: {"n": len(v), "folds_s": sorted(v),
               "median_s": sorted(v)[len(v) // 2] if len(v) % 2 else
               round((sorted(v)[len(v) // 2 - 1] + sorted(v)[len(v) // 2]) / 2, 3),
               "min_s": min(v), "max_s": max(v), "spread_s": round(max(v) - min(v), 3)}
           for a, v in sorted(arms.items())}
out = {
    "instrument": "of3t-hostleg clkwin.py -- per-fold wall clock and the AICLK sampled DURING "
                  "the fold, board class p150a Blackhole, qb1 (tt-quietbox) card 1",
    "per_run": rows,
    "per_arm": summary,
}
p = Path(__file__).with_name("FOLD_CLOCK.json")
p.write_text(json.dumps(out, indent=1) + "\n")
print("\n" + json.dumps(summary, indent=1))
print("->", p)
