#!/usr/bin/env python3
"""DISCOUNT factors: bootstrap CI on the geometric mean of R per screen kind.

The discount a brief should apply is 1/geomean(R) when R<1 (the screen is optimistic) -- but the
honest quotable number is the geometric mean itself with its 95 % CI and the sign-flip rate beside
it, because a point factor on a screen that flips sign 1 time in 4 is not a discount, it is a coin.
Rows whose prediction was 'this is dead' (near-zero denominator), one-sided floors, and the
work-removal cheat are excluded, as in stats.py.
"""
import json, math, random, statistics as st
rows=[r for r in json.load(open("transfer_table.json")) if r.get("scored") and r["outcome"]=="HELD"]
allsc=[r for r in json.load(open("transfer_table.json")) if r.get("scored")]
random.seed(0)
KIND={"replay":"replay gap","roof":"roofline headroom","count":"source-derived count/rate model",
      "op_ab":"op/block A/B extrapolated","fold_tx":"measured fold ratio transferred",
      "novel":"not-yet-existing mechanism"}
print(f"{'screen kind':34s} {'n':>3s} {'geomean R':>9s} {'95% CI':>18s} {'x-scatter':>9s} {'flip/null':>9s}")
out={}
for k,name in KIND.items():
    sel=[r["Rv"] for r in rows if r["kind"]==k]
    tot=[r for r in allsc if r["kind"]==k]
    bad=sum(1 for r in tot if r["outcome"]!="HELD")
    if not sel:
        print(f"{name:34s} {len(tot):3d} {'--':>9s} {'--':>18s} {'--':>9s} {bad}/{len(tot)}")
        out[k]=dict(n=len(tot),geo=None,ci=None,flip=f"{bad}/{len(tot)}"); continue
    lg=[math.log10(x) for x in sel]
    g=10**st.fmean(lg)
    bs=sorted(10**st.fmean(random.choices(lg,k=len(lg))) for _ in range(20000))
    lo,hi=bs[int(.025*20000)],bs[int(.975*20000)]
    sc=10**(st.stdev(lg) if len(lg)>1 else 0.0)
    print(f"{name:34s} {len(tot):3d} {g:9.3f} [{lo:7.3f}, {hi:7.3f}] {sc:9.2f}x {bad}/{len(tot)}")
    out[k]=dict(n=len(tot),n_held=len(sel),geo=round(g,3),ci=[round(lo,3),round(hi,3)],
                scatter=round(sc,2),flip=f"{bad}/{len(tot)}")
json.dump(out,open("discount.json","w"),indent=1)
print("\nfold-measured subset only:")
for k,name in KIND.items():
    sel=[r["Rv"] for r in rows if r["kind"]==k and r["level"].startswith("fold")]
    tot=[r for r in allsc if r["kind"]==k and r["level"].startswith("fold")]
    bad=sum(1 for r in tot if r["outcome"]!="HELD")
    if len(sel)<2:
        print(f"  {name:32s} n={len(tot)}  (too few to bootstrap)  flip/null {bad}/{len(tot)}"); continue
    lg=[math.log10(x) for x in sel]; g=10**st.fmean(lg)
    bs=sorted(10**st.fmean(random.choices(lg,k=len(lg))) for _ in range(20000))
    print(f"  {name:32s} n={len(tot)}  geomean {g:.3f}  95% CI [{bs[500]:.3f}, {bs[19500]:.3f}]"
          f"  scatter {10**st.stdev(lg):.2f}x  flip/null {bad}/{len(tot)}")
