"""Read chain1.sh: per arm the evo seam medians (card s/it) and the in-run AICLK, then each ablation
against the mean of the base arms. usage: read1.py OUT_DIR"""
import json
import statistics
import sys
from pathlib import Path

D = Path(sys.argv[1])
clk = [line.split("\t") for line in (D / "aiclk.tsv").read_text().splitlines()]
clk = [(int(t), int(a) if a.strip().isdigit() else None, float(ld)) for t, a, ld in clk]
win = {}
for line in (D / "arms").read_text().splitlines():
    w = line.split()
    if len(w) > 2 and w[1] in ("start", "end"):
        win.setdefault(w[0], {})[w[1]] = int(w[2])

rows = {}
for name, se in sorted(win.items()):
    f = D / name / "devprof.json"
    if not f.exists() or "end" not in se:
        continue
    s = json.loads(f.read_text())["summary"]
    seam = {k: v["median_s"] for k, v in s.items()}
    ck = [a for t, a, _ in clk if se["start"] <= t <= se["end"] and a]
    ld = [x for t, _, x in clk if se["start"] <= t <= se["end"]]
    rows[name] = {"evo": seam.get("evo.taped", 0) + seam.get("evo.backward", 0),
                  "evo.taped": seam.get("evo.taped"), "evo.backward": seam.get("evo.backward"),
                  "aiclk_med": statistics.median(ck) if ck else None,
                  "under_1200": sum(a < 1200 for a in ck), "n_clk": len(ck),
                  "load_med": statistics.median(ld) if ld else None}
base = [r for n, r in rows.items() if n.startswith("base")]
out = {"arms": rows}
if base:
    ref = {k: statistics.mean(r[k] for r in base) for k in ("evo", "evo.taped", "evo.backward")}
    out["base_mean"] = ref
    out["base_spread_evo"] = [min(r["evo"] for r in base), max(r["evo"] for r in base)]
    delta = {}
    for n, r in rows.items():
        if not n.startswith("base"):
            delta.setdefault(n.rsplit("_", 1)[0], []).append({k: r[k] - ref[k] for k in ref})
    out["delta_vs_base"] = delta
print(json.dumps(out, indent=1))
