"""Medians per arm of an abab.sh run: warm s/design pooled over rounds, the paired delta to `base` per round,
the CIF digest each arm produced, and the AICLK the run's own chip read while it ran."""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
warm, dig, rnd = defaultdict(list), defaultdict(set), defaultdict(dict)
for f in sorted(out.glob("*.r*.json")):
    arm, r = f.name[:-5].rsplit(".r", 1)
    d = json.loads(f.read_text())
    w = [x["fold_s"] for x in d["folds"] if x["tag"].startswith("warm")]
    warm[arm] += w
    rnd[arm][r] = st.median(w) if w else None
    dig[arm] |= {x.get("cif_sha256") for x in d["folds"] if x.get("cif_sha256")}
base = rnd.get("base", {})
for arm in sorted(warm):
    pd = [rnd[arm][r] - base[r] for r in rnd[arm] if r in base and rnd[arm][r] and base[r]]
    print(f"{arm:30s} n={len(warm[arm])} median {st.median(warm[arm]):.3f} s  paired d {[round(x, 3) for x in pd]}"
          f"  digests {sorted(d[:12] for d in dig[arm])}")
nodes = {n.split("/")[-1] for n in (out / "nodes.txt").read_text().split()} if (out / "nodes.txt").exists() else set()
rows = [l.split("\t") for l in (out / "aiclk.tsv").read_text().splitlines()] if (out / "aiclk.tsv").exists() else []
clk = [int(r[2]) for r in rows if len(r) == 4 and r[2].isdigit() and (not nodes or r[1] in nodes)]
if clk:
    print(f"AICLK (opened nodes) median {st.median(clk)} MHz, min {min(clk)}, n {len(clk)}; device nodes opened {nodes}")
