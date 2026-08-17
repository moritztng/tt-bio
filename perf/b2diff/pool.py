import json, sys, statistics as st
from collections import defaultdict
by = defaultdict(list); shas = defaultdict(set)
for f in sys.argv[1:]:
    for r in json.load(open(f))["runs"]:
        if r.get("error"): continue
        by[r["arm"]].append(r["fold_s"]); shas[r["arm"]] |= set(r["cif_sha256"].values())
on = by["on"]; base = st.mean(on); sd = st.stdev(on)
print(f"on n={len(on)} mean={base:.3f} sd={sd:.4f} range={max(on)-min(on):.3f} ({100*(max(on)-min(on))/base:.2f}%)")
print(f"{'arm':9s} {'n':>2s} {'mean s':>8s} {'delta s':>8s} {'ratio':>8s} {'sd(diff)':>9s} {'sigma':>6s}  cif")
for arm in ["on", "l7", "l6", "l7l6", "s6", "l7l6s6"]:
    v = by.get(arm)
    if not v: continue
    m = st.mean(v); d = m - base
    se = (sd**2/len(v) + sd**2/len(on)) ** 0.5
    print(f"{arm:9s} {len(v):2d} {m:8.3f} {d:+8.3f} {base/m:8.4f} {se:9.3f} {abs(d)/se:6.1f}  {sorted(shas[arm])}")
print("\nDERIVED, page baseline 24.822 s carried through the measured ratio:")
for arm in ["l7l6", "l7l6s6"]:
    m = st.mean(by[arm]); print(f"  {arm:8s} 24.822 / {base/m:.4f} = {24.822*m/base:.3f} s")
