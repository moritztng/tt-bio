#!/usr/bin/env python3
"""D43(2) bisect grade: bb_ca |ours-ref| per arm vs the D37 floored allowance (1.5 x |up_bf16-ref|)."""
import json, glob, os, sys
OUT = os.path.expanduser("~/r4t/perf/bis/out")
ALLOW = {"00": 2.006e-03, "07": 4.701e-03}   # perf/r4t/results/grade_P.txt (wk/r4t-accuracy 32d10948e)
def rec(arm, b):
    try:
        for line in open(f"{OUT}/{arm}_{b}.log"):
            if line.startswith("{") and "master_digest" in line:
                return json.loads(line)
    except FileNotFoundError:
        return None
arms = []
for line in open(f"{OUT}/run.log"):
    if line.startswith("RC_"):
        a = line.split()[0][3:].rsplit("_", 1)[0]
        if a not in arms: arms.append(a)
for a in arms:
    row = [f"{a:20s}"]; ok = True
    for b in ("00", "07"):
        r = rec(a, b)
        if not r: row.append(f"b{b} missing"); ok = None; continue
        d = abs(r["terms"]["bb_ca"] - r["upstream_terms"]["bb_ca"])
        row.append(f"b{b} {d:.3e} ({d/ALLOW[b]:.2f}x) {r['master_digest']} aiclk{r['aiclk']}")
        ok = ok and d <= ALLOW[b]
    print(" ".join(row), "PASS" if ok else ("FAIL" if ok is False else "?"))
