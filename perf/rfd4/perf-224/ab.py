"""Paired width A/B from a run.sh outdir: per-design traced trunk call at width a vs b, by round, with the clock.
  python3 perf/rfd4/perf-224/ab.py <outdir>
"""
import json,glob,sys,statistics as st
R=sys.argv[1]
pairs=[(224,5,10),(192,7,14),(256,4,8),(384,1,3),(512,1,2)]
for n,a,b in pairs:
    ra=[];rb=[];clk=[]
    for fa in sorted(glob.glob(f"{R}/pass-{n}-b{a}*.json")):
        fb=fa.replace(f"-b{a}",f"-b{b}")
        try: A=json.load(open(fa));B=json.load(open(fb))
        except Exception as e: print(n,"missing",fb); continue
        ra.append(A["call_traced_ms"]["median"]/a); rb.append(B["call_traced_ms"]["median"]/b)
        for X in (A,B): c=X["aiclk"]["0"]; clk+= [c["min"],c["median"]]
        print(f"  {n} {fa.split('/')[-1]}: A {ra[-1]:.2f} B {rb[-1]:.2f} ms/design-call  gain {ra[-1]/rb[-1]:.4f}  eq {A['traced_equals_eager']},{B['traced_equals_eager']}  clk A {A['aiclk']['0']} B {B['aiclk']['0']}")
    if ra:
        g=[x/y for x,y in zip(ra,rb)]
        print(f"{n} b{a} vs b{b}: rounds {len(g)}  gain median {st.median(g):.4f} min {min(g):.4f} max {max(g):.4f}  A {st.median(ra):.2f} B {st.median(rb):.2f}  clk min {min(clk)}")
