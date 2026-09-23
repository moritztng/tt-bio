import hashlib, json, os, socket, torch
def sha(p, c=1<<22):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(c), b''): h.update(b)
    return h.hexdigest()
P="/home/ttuser/of3t_modelframe"; N="/home/ttuser/of3t_frameself"
out={"what":"qb1 (published, of3t-modelframe) against qb2 (of3t-frameself --selftest re-run): "
            "is the captured boundary/cotangent pair reproducible across hosts?",
     "host":socket.gethostname(),"row":"of3t-frameself","defect":"D242","device_involved":False,
     "why_no_aiclk":"CPU only, no Tenstorrent device is opened","pairs":{}}
for tag,(a,b) in {
  "boundary":(f"{P}/boundary_model_n384.pt", f"{N}/boundary_selftest_n384.pt"),
  "cotangent":(f"{P}/cot_model_n384.pt",     f"{N}/cot_selftest_n384.pt")}.items():
    da=torch.load(a,map_location="cpu",weights_only=False)
    db=torch.load(b,map_location="cpu",weights_only=False)
    rec={"qb1":{"path":a,"sha256":sha(a),"bytes":os.path.getsize(a),"keys":sorted(map(str,da.keys()))},
         "qb2":{"path":b,"sha256":sha(b),"bytes":os.path.getsize(b),"keys":sorted(map(str,db.keys()))},
         "file_sha256_identical": sha(a)==sha(b), "tensors":{}}
    def flat(d):
        o={}
        for k,v in d.items():
            if torch.is_tensor(v): o[str(k)]=v
            elif isinstance(v,(tuple,list)):
                for i,t in enumerate(v):
                    if torch.is_tensor(t): o[f"{k}[{i}]"]=t
            elif isinstance(v,dict):
                for kk,t in v.items():
                    if torch.is_tensor(t): o[f"{k}.{kk}"]=t
        return o
    fa,fb=flat(da),flat(db)
    for k in sorted(set(fa)|set(fb)):
        if k not in fa or k not in fb:
            rec["tensors"][k]={"present_both":False}; continue
        x=fa[k].to(torch.float64).reshape(-1); y=fb[k].to(torch.float64).reshape(-1)
        nx=float(torch.linalg.vector_norm(x))
        d=float(torch.linalg.vector_norm(x-y)) if x.shape==y.shape else None
        rec["tensors"][k]={"present_both":True,"shape":list(fa[k].shape),
            "bit_identical": bool(x.shape==y.shape and torch.equal(x,y)),
            "norm_qb1":nx,"norm_qb2":float(torch.linalg.vector_norm(y)),
            "rel_l2": (d/nx if (d is not None and nx>0) else d)}
    out["pairs"][tag]=rec
print(json.dumps(out,indent=1))
open(f"{N}/PAIR_REPRO.json","w").write(json.dumps(out,indent=1))
