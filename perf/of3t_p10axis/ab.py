#!/usr/bin/env python3
"""The fixed/unfixed A/B, reduced: what each arm declared, trained and cost."""
import json, sys, re, collections
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"
tag = sys.argv[1] if len(sys.argv) > 1 else "ab"

for name in ("unfixed", "fixed"):
    p = OUT / f"arm_{name}_{tag}.json"
    if not p.exists():
        print(f"{name}: no artifact"); continue
    d = json.loads(p.read_text())
    reps, st = d.get("reps", []), d.get("steady") or {}
    c = d.get("grad_census", [])
    ends = [i for i in range(len(c)) if i + 1 == len(c) or c[i+1]["with_grad"] < c[i]["with_grad"]]
    print(f"\n=== {name} ===  declared {d['params']['declared']}  "
          f"duplicate_handles_dropped {d['params']['duplicate_handles_dropped']}")
    print(f"  cold {d.get('cold_s')}  steady median {st.get('median_s')}  "
          f"values {st.get('values_s')}  AICLK {d.get('env',{}).get('aiclk_line','')[:60]}")
    for r in reps:
        print(f"  rep{r['rep']:<2} step {r['step_s']:>7}  trunk {r['trunk_s']:>5}  "
              f"diff {r['diffusion_s']:>5}  loss {r['losses_s']:>5}  "
              f"bwd {r['backward_s']:>6}  adam {r['optimizer_s']:>5}  {r['params_with_grad']}")
    for k, i in enumerate(ends):
        miss = c[i]["missing"]
        g = collections.Counter(re.sub(r"\.(\d+)\.", ".N.", n).split(".")[1] for n in miss)
        print(f"  census rep{k}: {c[i]['with_grad']}/{c[i]['declared']} trained, "
              f"{c[i]['without_grad']} not — {dict(g.most_common(5))}")

a = OUT / f"arm_unfixed_{tag}.json"; b = OUT / f"arm_fixed_{tag}.json"
if a.exists() and b.exists():
    da, db = json.loads(a.read_text()), json.loads(b.read_text())
    sa = (da.get("steady") or {}).get("median_s"); sb = (db.get("steady") or {}).get("median_s")
    ra = [r for r in da.get("reps", []) if not r["cold"]]
    rb = [r for r in db.get("reps", []) if not r["cold"]]
    if sa and sb:
        print(f"\nSTEADY  unfixed {sa} s -> fixed {sb} s  ({sb/sa:.4f}x)")
        ba = sorted(r["backward_s"] for r in ra)[len(ra)//2]
        bb = sorted(r["backward_s"] for r in rb)[len(rb)//2]
        print(f"BACKWARD unfixed {ba} s -> fixed {bb} s  ({bb/ba:.4f}x)")
        print(f"TRAINED  unfixed {ra[0]['params_with_grad']} -> fixed {rb[0]['params_with_grad']}")
