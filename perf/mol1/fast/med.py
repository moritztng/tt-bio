# Median warm s/fold, AICLK floor and first-warm pLDDT/iptm per <arm>-<cell> dir of a mol1-fast screen: python3 med.py <screen dir>
import json,sys,glob,statistics,os
for d in sorted(glob.glob(sys.argv[1]+"/*-*/")):
    rows=[]
    for f in glob.glob(d+"**/runs.jsonl",recursive=True):
        for l in open(f):
            try: rows.append(json.loads(l))
            except: pass
    w=[r for r in rows if r.get("kind")=="warm"]
    c=[r for r in rows if r.get("kind")=="cold" and r.get("phase")=="compile"]
    if not w and not c: print(os.path.basename(d.rstrip('/')),"no rows"); continue
    ws=[r["wall_s"] for r in w]
    clk=[r.get("aiclk_min_mhz") for r in w]
    pl=[(p.get("plddt"),p.get("iptm"),p.get("coords_sha256","")[:8]) for r in w[:1] for p in r.get("per_chip",[])]
    print(os.path.basename(d.rstrip('/')), "warm", [round(x,1) for x in ws], "med", round(statistics.median(ws),1) if ws else None, "clkmin", clk, "cold", [round(r["wall_s"]) for r in c], pl)
