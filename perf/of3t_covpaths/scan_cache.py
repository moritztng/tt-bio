import json, collections, sys
p="/home/ttuser/of3t_up/datasets/training_cache_with_templates.json"
d=json.load(open(p))
sd=d["structure_data"]
print("n_structures", len(sd))
mt=collections.Counter()
cand_nuc=[]; cand_res=[]; both=[]
for sid,e in sd.items():
    ch=e["chains"]
    types=collections.Counter(c["molecule_type"] for c in ch.values())
    mt.update(types)
    res=e.get("resolution")
    ntpl=sum(1 for c in ch.values() if c.get("template_ids"))
    nnuc=types.get("DNA",0)+types.get("RNA",0)
    outres = (res is None) or (isinstance(res,float) and res!=res) or (res is not None and res==res and (res<0.1 or res>4.0))
    rec=(sid,len(ch),res,ntpl,dict(types))
    if nnuc>0: cand_nuc.append(rec)
    if outres: cand_res.append(rec)
    if nnuc>0 and outres: both.append(rec)
print("molecule_type counts", mt)
print("n nucleotide-bearing", len(cand_nuc))
print("n out-of-resolution", len(cand_res))
print("n both", len(both))
json.dump({"nuc":cand_nuc[:4000],"res":cand_res[:4000],"both":both[:4000]},open("/tmp/of3t/of3t-covpaths/scan_cache.json","w"))
def show(name, L, k=15):
    L=sorted(L, key=lambda r:(r[1], -r[3]))
    print("==",name)
    for r in L[:k]: print("  ",r)
show("nucleotide smallest", cand_nuc)
show("out-of-res smallest", cand_res)
show("both smallest", both)
