"""Mid-session anchor gate: does 512 aa still read the numbers of record?

Runs the moment the 512 aa process finishes and before a single fold is spent on another rung,
because a failed anchor voids the whole session and there is no point paying for the rest of the
ladder. Deliberately independent of reduce.py -- it reads result.json directly -- so the anchor has
two readers rather than one.
"""
from __future__ import annotations
import json, math, statistics, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from sizefit import work_two_clock

a = json.loads((HERE / "prediction.json").read_text())["anchor"]
run = json.loads((Path(sys.argv[1]) / "512" / "result.json").read_text())
rows = [r for r in run["rows"] if r.get("valid") and r["label"] != "cold"]
cells = {}
for f in (1350, 800):
    t = sorted(r["elapsed_s"] for r in rows if r["clock_MHz"] == f)
    if len(t) < 4:
        print(f"ANCHOR GATE: FAIL -- only {len(t)} accepted folds at {f} MHz"); sys.exit(2)
    sd = statistics.stdev(t)
    cells[f] = (statistics.median(t), 1.2533 * sd / math.sqrt(len(t)), len(t), sd)

w = work_two_clock(1350, cells[1350][0], cells[1350][1], 800, cells[800][0], cells[800][1])
tol = 2 * math.hypot(w["se_W_Mcycles"], a["W_stated_error_Mcycles"])
checks = {
    "t_1350_s": (cells[1350][0], a["t_1350_allowed_s"][0] <= cells[1350][0] <= a["t_1350_allowed_s"][1], a["t_1350_allowed_s"]),
    "t_800_s": (cells[800][0], a["t_800_allowed_s"][0] <= cells[800][0] <= a["t_800_allowed_s"][1], a["t_800_allowed_s"]),
    "W_Mcycles": (w["W_Mcycles"], abs(w["W_Mcycles"] - a["W_Mcycles"]) <= tol, [a["W_Mcycles"] - tol, a["W_Mcycles"] + tol]),
}
for k, (v, ok, allowed) in checks.items():
    print(f"  {'OK  ' if ok else 'FAIL'} {k} = {v:.4f}  allowed [{allowed[0]:.4f}, {allowed[1]:.4f}]")
print(f"  info F_s = {w['F_s']:.4f} +- {w['se_F_s']:.4f} against the {a['F_s']} +- {a['F_stated_error_s']} s of record")
print(f"  info folds/sd: 1350 MHz n={cells[1350][2]} sd={cells[1350][3]:.4f} s; 800 MHz n={cells[800][2]} sd={cells[800][3]:.4f} s")
ok = all(c[1] for c in checks.values())
print("ANCHOR GATE: " + ("PASS -- the ladder may continue" if ok else
                         "FAIL -- STOP, nothing in this session may be quoted"))
sys.exit(0 if ok else 2)
