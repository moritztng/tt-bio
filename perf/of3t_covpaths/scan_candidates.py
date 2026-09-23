import json, collections
d=json.load(open("/home/ttuser/of3t_up/datasets/training_cache_with_templates.json"))
sd=d["structure_data"]
pn=[]; outres=[]
for sid,e in sd.items():
    ch=e["chains"]; t=collections.Counter(c["molecule_type"] for c in ch.values())
    res=e.get("resolution"); ntpl=sum(1 for c in ch.values() if c.get("template_ids"))
    nuc=t.get("DNA",0)+t.get("RNA",0)
    inrange = isinstance(res,(int,float)) and res==res and 0.1<=res<=4.0
    rec=(sid,len(ch),res,ntpl,dict(t))
    if nuc>0 and t.get("PROTEIN",0)>0 and inrange and ntpl>0 and len(ch)<=6: pn.append(rec)
    if t.get("PROTEIN",0)>0 and nuc==0 and ntpl>0 and len(ch)<=3 and (res is None or (isinstance(res,(int,float)) and res==res and res>4.0)): outres.append(rec)
pn.sort(key=lambda r:(r[1],-r[3])); outres.sort(key=lambda r:(r[1],-r[3]))
print("protein+nucleic, in-range res, templated, <=6 chains:", len(pn))
for r in pn[:20]: print("  ",r)
print("protein-only, OUT-of-range res, templated, <=3 chains:", len(outres))
for r in outres[:10]: print("  ",r)
numeric=[r for r in outres if isinstance(r[2],(int,float)) and r[2]==r[2]]
print("  ... of which numeric >4.0:", len(numeric)); 
for r in numeric[:10]: print("   N",r)
