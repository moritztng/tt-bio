#!/usr/bin/env python3
"""Distribution of measured/predicted per screen kind, from transfer_table.json."""
import json, statistics as st, math
rows=json.load(open("transfer_table.json"))

# rows whose PREDICTION was "this is dead / inside the floor": R is undefined (near-zero denominator).
# Scored on the DECISION, not the magnitude.
KILL={("b2z2-step-matmul-group","N-stacking the s-projections"),
      ("b2z2-trunk-bh-leg","qkvg alone on BH (P2)"),
      ("b2z2-bh-compose-landed","DST-resident on BH")}
# one-sided predictions (">= x"): a floor, not a point. Excluded from the R distribution.
ONESIDED={("b2z2-everything-union-wh","confidence stage (P6)"),
          ("b2z2-everything-union-wh","conditioning stage (P6)")}
# work-removal cheat under the standing no-cheat rule: recorded, never scored.
CHEAT={("b2z-sampler-steps","200 -> 50 sampling steps on BH*")}

KINDNAME={"replay":"replay gap","roof":"roofline headroom","count":"source-derived count/rate model",
          "op_ab":"op- or block-level A/B, extrapolated","fold_tx":"measured fold ratio transferred",
          "novel":"not-yet-existing mechanism"}

for r in rows:
    k=(r["slug"],r["lever"])
    r["kill"]=k in KILL; r["onesided"]=k in ONESIDED; r["cheat"]=k in CHEAT
    p,m=r["pred"],r["meas"]
    # outcome: HELD | NULLED | FLIPPED  (NULLED = predicted a lever, measured zero / inside floor)
    if abs(m)<1e-9: r["outcome"]="NULLED"
    elif (p>0)!=(m>0): r["outcome"]="FLIPPED"
    else: r["outcome"]="HELD"
    r["scored"] = not (r["kill"] or r["onesided"] or r["cheat"])

def summarise(sel,label):
    sel=[r for r in sel if r["scored"]]
    if not sel: return
    Rs=sorted(r["Rv"] for r in sel)
    pos=[r["Rv"] for r in sel if r["outcome"]=="HELD"]
    nf=sum(1 for r in sel if r["outcome"]=="FLIPPED")
    nn=sum(1 for r in sel if r["outcome"]=="NULLED")
    band=[r for r in sel if r["inband"] is not None]
    nib=sum(1 for r in band if r["inband"])
    med=st.median(Rs)
    print(f"\n{label}   n={len(sel)}")
    print(f"  median R (all)        {med:7.3f}")
    if pos:
        lp=sorted(math.log10(x) for x in pos)
        print(f"  median R (sign held)  {st.median(pos):7.3f}   n={len(pos)}"
              f"   p10..p90 {10**lp[max(0,int(.1*(len(lp)-1)))]:.3f} .. {10**lp[int(.9*(len(lp)-1))]:.3f}")
        print(f"  geometric mean R      {10**st.fmean(lp):7.3f}   log10 sd {st.stdev(lp) if len(lp)>1 else 0:.3f}")
        print(f"  |log10 R| median      {st.median(abs(x) for x in lp):7.3f}"
              f"  -> typical miss {10**st.median(abs(x) for x in lp):.2f}x")
    print(f"  sign flips            {nf}/{len(sel)} = {100*nf/len(sel):.0f} %")
    print(f"  nulled (lever absent) {nn}/{len(sel)} = {100*nn/len(sel):.0f} %")
    print(f"  wrong sign or nulled  {nf+nn}/{len(sel)} = {100*(nf+nn)/len(sel):.0f} %")
    if band: print(f"  landed inside its own pre-registered band {nib}/{len(band)} = {100*nib/len(band):.0f} %")

print("="*78); print("ALL ROWS, BY SCREEN KIND"); print("="*78)
for k in ("replay","roof","count","op_ab","fold_tx","novel"):
    summarise([r for r in rows if r["kind"]==k], f"[{k}] {KINDNAME[k]}")
summarise(rows,"[ALL KINDS POOLED]")

print("\n"+"="*78); print("FOLD-MEASURED SUBSET ONLY (the screen-to-FOLD transfer question)"); print("="*78)
fold=[r for r in rows if r["level"].startswith("fold")]
for k in ("replay","roof","count","op_ab","fold_tx","novel"):
    summarise([r for r in fold if r["kind"]==k], f"[{k}] {KINDNAME[k]}")
summarise(fold,"[ALL KINDS POOLED, fold-measured]")

print("\n"+"="*78); print("DECISION ACCURACY (did the pre-registered verdict hold, ignoring magnitude?)"); print("="*78)
for k in ("replay","roof","count","op_ab","fold_tx","novel"):
    sel=[r for r in rows if r["kind"]==k and not r["cheat"]]
    if not sel: continue
    # a decision is correct if the row's own band/kill contained the answer, else wrong
    ok=sum(1 for r in sel if (r["kill"] and r["outcome"] in ("NULLED","HELD","FLIPPED") and abs(r["meas"])<=abs(r["pred"])*3 or r["kill"] and r["outcome"]!="HELD")
           or (r["inband"] is True))
    print(f"  [{k:7s}] in-band or kill-confirmed: {ok}/{len(sel)}")
print("\nKILL / FLOOR PREDICTIONS (scored on the decision, R undefined):")
for r in rows:
    if r["kill"]: print(f"  {r['slug']:32s} {r['lever'][:38]:38s} pred={r['pred']:+.4f} meas={r['meas']:+.4f} -> {r['outcome']}")
print("\nONE-SIDED (floor) PREDICTIONS, excluded from R:")
for r in rows:
    if r["onesided"]: print(f"  {r['slug']:32s} {r['lever'][:38]:38s} pred>={r['pred']:.3f} meas={r['meas']:.4f}")
print("\nEXCLUDED AS A WORK-REMOVAL CHEAT:")
for r in rows:
    if r["cheat"]: print(f"  {r['slug']:32s} {r['lever'][:44]}")
json.dump(rows, open("transfer_table.json","w"), indent=1)
