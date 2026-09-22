#!/usr/bin/env python3
"""Merge the p2 sweep JSONs into one blob and re-fit every arm.

Separate processes were needed only because two points OOMed uncheckpointed and had to be
retaken; the points themselves are comparable because every run used the same card, the same
seed, the same dims and the same steps, and each carries its own clock window.
"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from perf.hallgrad.blocksweep import analyze

out = sys.argv[1]
srcs = sys.argv[2:]
base = json.load(open(srcs[0]))
base["merged_from"] = [str(s) for s in srcs]
seen = {(p["n"], p["blocks"], p["block_transition"], p["checkpoint"]) for p in base["points"]}
roof = dict(base.get("matmul_roof", {}))
for s in srcs[1:]:
    b = json.load(open(s))
    for p in b["points"]:
        k = (p["n"], p["blocks"], p["block_transition"], p["checkpoint"])
        if k in seen and any(q.get("ok") for q in base["points"]
                             if (q["n"], q["blocks"], q["block_transition"],
                                 q["checkpoint"]) == k):
            continue
        base["points"].append(p)
        seen.add(k)
    for n, r in b.get("matmul_roof", {}).items():
        roof.setdefault(n, r)
    base.setdefault("clock_samples", []).extend(b.get("clock_samples", []))
base["matmul_roof"] = roof
base["points"].sort(key=lambda p: (p["checkpoint"], p["block_transition"], p["n"], p["blocks"]))
analyze(base)
json.dump(base, open(out, "w"), indent=1)

print(f"{'arm':<28} {'a (s)':>9} {'b (s)':>9} {'maxrel':>9} {'r2':>10} "
      f"{'b_lo':>9} {'b_hi':>9} {'drift':>7}  K           linear")
for k, f in base["fits"].items():
    s = f["step"]
    print(f"{k:<28} {s['a']:>9.5f} {s['b']:>9.5f} {s['maxrel']:>9.2e} {s['r2']:>10.7f} "
          f"{s.get('b_lo', float('nan')):>9.5f} {s.get('b_hi', float('nan')):>9.5f} "
          f"{s.get('slope_drift', float('nan')):>7.4f}  {[int(x) for x in s['K']]}  "
          f"{s['linear']}")
print()
print("arithmetic:", json.dumps(base["arithmetic"], indent=1))
print("discriminator 1:", json.dumps(base["discriminator_1_scaling"], indent=1))
print("discriminator 2:", json.dumps(base["discriminator_2_roof"], indent=1))
