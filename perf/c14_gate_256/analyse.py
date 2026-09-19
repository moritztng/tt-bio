#!/usr/bin/env python3
"""Join every fold in a cell to the AICLK samples taken DURING that fold, and print the table.

`predict` logs one line per fold, "HH:MM:SS ... OK <id> - <runtime>s", at the moment the fold
finishes, and results.json carries the same runtime_s the release gate records. So each fold's
window is [end - runtime_s, end] and the ~500 Hz sysfs sampler in run_cell.sh covers it. A fold
is 4-14 s here, well inside a governor step, which is why the clock is read per fold and not
per cell.

    python3 analyse.py --dir out --work work
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics as st
from pathlib import Path

LINE = re.compile(r"^(\d\d):(\d\d):(\d\d)\s+\S.*?\s(\S+)\s+[—-]\s+([0-9.]+)s\s*$")


def folds_from_log(log: Path, day: dt.date):
    """[(id, runtime_s, end_epoch)] in the order the worker folded them."""
    out = []
    for raw in log.read_text(errors="replace").splitlines():
        m = LINE.match(raw.strip())
        if not m:
            continue
        h, mi, s, fid, rt = m.groups()
        end = dt.datetime.combine(day, dt.time(int(h), int(mi), int(s)),
                                  tzinfo=dt.timezone.utc).timestamp()
        out.append((fid, float(rt), end))
    return out


def clock_window(samples, t0, t1):
    v = [m for t, m in samples if t0 <= t <= t1]
    if not v:
        return None
    q = sorted(v)
    return {"n": len(q), "min": q[0], "median": st.median(q), "max": q[-1],
            "mean": round(st.fmean(q), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out"))
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    clocks = {}
    for p in sorted(a.dir.glob("aiclk_*.jsonl")):
        cell = p.stem[len("aiclk_"):]
        rows = []
        for line in p.read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "MHz" in r:
                rows.append((r["t"], r["MHz"]))
        clocks[cell] = rows

    cells = {}
    for wd in sorted(a.work.iterdir()):
        log = wd / "predict.log"
        if not log.exists():
            continue
        tag = wd.name                      # e.g. q256p1
        cell = re.sub(r"p\d+$", "", tag)
        day = dt.datetime.utcfromtimestamp(log.stat().st_mtime).date()
        folds = folds_from_log(log, day)
        if not folds:
            continue
        seq = []
        for i, (fid, rt, end) in enumerate(folds):
            clk = clock_window(clocks.get(cell, []), end - rt, end)
            seq.append({"order": i, "id": fid, "runtime_s": rt, "clk": clk})
        cells.setdefault(cell, []).append({"proc": tag, "folds": seq})

    print("%-7s %-9s %-34s %s" % ("cell", "process", "runtime_s by fold order",
                                  "AICLK MHz during each fold (min/med/max)"))
    for cell in sorted(cells):
        for pr in sorted(cells[cell], key=lambda d: d["proc"]):
            rts = " ".join("%5.1f" % f["runtime_s"] for f in pr["folds"])
            clk = " ".join("-" if not f["clk"] else
                           "%d/%d/%d" % (f["clk"]["min"], f["clk"]["median"], f["clk"]["max"])
                           for f in pr["folds"])
            print("%-7s %-9s %-34s %s" % (cell, pr["proc"], rts, clk))

    print()
    print("per cell: fold0 across processes, then folds 1.. pooled")
    for cell in sorted(cells):
        f0 = [pr["folds"][0]["runtime_s"] for pr in cells[cell] if pr["folds"]]
        rest = [f["runtime_s"] for pr in cells[cell] for f in pr["folds"][1:]]
        proc_med = [st.median([f["runtime_s"] for f in pr["folds"][1:]])
                    for pr in cells[cell] if len(pr["folds"]) > 1]
        print("  %-6s fold0=%s  later=%s  per-process median(later)=%s"
              % (cell, f0, rest, proc_med))

    if a.json:
        a.json.write_text(json.dumps(cells, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
