#!/usr/bin/env python3
"""Summarise an l1_aba.sh directory: per-arm seam medians, AICLK median over each arm's window, and the
census's bytes split by where each tensor lives (DRAM vs L1).   summ.py <dir>"""
import json, re, statistics as st, sys
from pathlib import Path

D = Path(sys.argv[1])
clk = [tuple(map(int, l.split())) for l in (D / "aiclk.tsv").read_text().split("\n") if l.strip()]
win = {}
for l in (D / "card").read_text().splitlines():
    m = re.match(r"(\S+) (start|rc=\d+ end) (\d+) load (\S+)", l)
    if m:
        win.setdefault(m[1], {})["s" if m[2] == "start" else "e"] = int(m[3])
        win[m[1]]["load_" + ("s" if m[2] == "start" else "e")] = float(m[4])
ITEM = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625}
SKIP = {"deallocate", "reshape", "allocate_tensor_on_device", "empty"}
for arm in sorted(win, key=lambda a: win[a]["s"]):
    f = D / arm / "devprof.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    w = win[arm]
    c = [v for t, v in clk if w["s"] <= t <= w.get("e", 1e12)]
    line = f"{arm:8s} aiclk_med {st.median(c) if c else '-'} min {min(c) if c else '-'} load {w.get('load_s')}-{w.get('load_e')}"
    if d.get("iteration_s") and not arm.endswith("_cen"):
        it = d["iteration_s"]
        s = d["summary"]
        line += (f" iter med {st.median(it):.3f} [{min(it):.3f}-{max(it):.3f}] n={len(it)}"
                 f" evo.taped {s['evo.taped']['median_s']:.3f} evo.bwd {s['evo.backward']['median_s']:.3f}")
    cen = d.get("census")
    if cen:
        by = {"DRAM": 0.0, "L1": 0.0}
        for o in cen["ops"]:
            if o["name"] in SKIP:
                continue
            for m in re.finditer(r"\[([\d, ]+)\]:(\w+):(DRAM|L1)", o["sig"] + " " + o["out"]):
                n = 1
                for x in m[1].split(","):
                    n *= int(x)
                by[m[3]] += n * ITEM.get(m[2], 2) * o["calls"]
        line += f" census GB DRAM {by['DRAM']/1e9:.1f} L1 {by['L1']/1e9:.1f}"
    print(line)
