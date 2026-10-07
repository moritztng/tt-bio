#!/usr/bin/env python3
"""Tables for state/pfm-gpu.md from a session's results dir.

    python parse.py results/A [results/B ...]

Per box x arm x input: cold (r0) fwd_s, warm r1..r3 mean and range, and the SM clock / power / throttle
reasons nvidia-smi saw while that arm ran. Cost per fold at Modal's per-second rate and at the box's raw
vast $/h (from results/<box>/instance + the rate passed in the RATE dict)."""
import csv, datetime as dt, re, statistics as st, sys
from pathlib import Path

MODAL = {"A100-40": 0.000583, "A100-80": 0.000694}  # $/s, modal.com/pricing (A100 40GB / 80GB)
PH = re.compile(r"PHASE item=(\S+)_r(\d+) .*?fwd_s=([\d.]+)")

def ts(s): return dt.datetime.strptime(s.strip()[:23], "%Y/%m/%d %H:%M:%S.%f")
def iso(p): return dt.datetime.strptime(p.read_text().strip()[:23], "%Y-%m-%dT%H:%M:%S.%f")

for box in map(Path, sys.argv[1:]):
    smi = [r for r in csv.reader(open(box / "smi.csv")) if len(r) >= 9 and r[0][:2] == "20"]
    card = "A100-80" if "80" in (box / "nvidia-smi-q.txt").read_text().split("Product Name")[1][:40] else "A100-40"
    print(f"## {box.name} ({card})")
    print("| arm | input | cold r0 s | warm r1-r3 mean s | range s | median SM MHz | median W | reasons | Modal $/fold |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in sorted(box.glob("time_*")):
        t0, t1 = iso(arm / "t_start"), iso(arm / "t_end")
        run = [r for r in smi if t0 <= ts(r[0]) <= t1 and int(r[6].split()[0]) > 50]
        clk = st.median(int(r[1].split()[0]) for r in run) if run else 0
        pw = st.median(float(r[3].split()[0]) for r in run) if run else 0
        rs = sorted({r[7].strip() for r in run})
        folds = {}
        for m in PH.finditer((arm / "stdout.log").read_text()):
            folds.setdefault(m[1], {})[int(m[2])] = float(m[3])
        for name, f in folds.items():
            warm = [f[i] for i in (1, 2, 3) if i in f]
            w = st.mean(warm) if warm else float("nan")
            print(f"| {arm.name[5:]} | {name} | {f.get(0, float('nan')):.2f} | {w:.2f} | "
                  f"{min(warm):.2f}-{max(warm):.2f} | {clk:.0f} | {pw:.0f} | {' '.join(rs)} | {w * MODAL[card]:.4f} |")
