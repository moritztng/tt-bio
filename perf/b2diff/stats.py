import json, sys, statistics as st
from collections import defaultdict
d = json.load(open(sys.argv[1]))
skip_first = "--skipfirst" in sys.argv
by = defaultdict(list); shas = defaultdict(set); order = []
for i, r in enumerate(d["runs"]):
    if r.get("error"): print("ERROR", r["arm"], r["error"][:80]); continue
    order.append((r["arm"], r["fold_s"]))
    if skip_first and i == 0: continue
    by[r["arm"]].append(r["fold_s"])
    shas[r["arm"]] |= set(r["cif_sha256"].values())
print("order:", ", ".join(f"{a}={s}" for a, s in order))
base = st.mean(by["on"])
print(f"\n{'arm':9s} {'n':>2s} {'mean s':>8s} {'min':>7s} {'max':>7s} {'spread':>7s} {'delta s':>8s} {'ratio':>8s}  cif")
for arm in ["on", "l7", "l6", "l7l6", "s6", "l7l6s6"]:
    v = by.get(arm)
    if not v: continue
    m = st.mean(v)
    print(f"{arm:9s} {len(v):2d} {m:8.3f} {min(v):7.3f} {max(v):7.3f} {max(v)-min(v):7.3f} "
          f"{m-base:+8.3f} {base/m:8.4f}  {sorted(shas[arm])}")
print(f"\nA/A noise floor (on arms): range {max(by['on'])-min(by['on']):.3f} s "
      f"= {100*(max(by['on'])-min(by['on']))/base:.2f} % of {base:.3f} s")
# region walls if present
w = [r for r in d["runs"] if r.get("walls_ms")]
if w and any(r["walls_ms"] for r in w):
    keys = sorted({k for r in w for k in r["walls_ms"]})
    print()
    rw = defaultdict(lambda: defaultdict(list))
    for r in w:
        for k, v in r["walls_ms"].items():
            rw[k][r["arm"]].append(v)
    for k in keys:
        cells = []
        for arm in ["on", "l7", "l6", "l7l6", "s6", "l7l6s6"]:
            v = rw[k].get(arm)
            cells.append(f"{arm}={st.mean(v):8.1f}" if v else f"{arm}=       -")
        b = st.mean(rw[k]["on"])
        cells.append("| on-spread=%.1f" % (max(rw[k]['on'])-min(rw[k]['on'])))
        print(f"{k:46s} " + "  ".join(cells))
    # the L7 region difference
    print("\nstage:DiffusionTransformer minus its two layer regions, per arm:")
    for r in w:
        dt = r["walls_ms"].get("stage:DiffusionTransformer")
        tok = r["walls_ms"].get("block:DiffusionTransformerLayer|token")
        at = r["walls_ms"].get("block:DiffusionTransformerLayer|atom")
        if dt: print(f"  {r['arm']:9s} #{r['ix']}  {dt - tok - at:8.1f} ms")
