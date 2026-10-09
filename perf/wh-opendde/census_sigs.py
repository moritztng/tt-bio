"""Per-signature device time of the ops a census fold ran at one call site (census_regs.py splits by module).

usage: census_sigs.py RUN_DIR SITE_SUBSTRING [TOP]   e.g. census_sigs.py cs1/normal tenstorrent.py:11401
"""
import json, sys, numpy as np
from pathlib import Path
RUN=Path(sys.argv[1]); PAT=sys.argv[2]; FOLD="full"
sigs={}
for l in open(RUN/f"sig_{FOLD}.jsonl"):
    e=json.loads(l)
    if "op" in e: sigs[e["sig"]]=e
calls=np.array([json.loads(l) for l in open(RUN/f"ops_{FOLD}.jsonl")],dtype=np.int64)
calls=calls[np.argsort(calls[:,1],kind="stable")]
P=np.array([[x if x is not None else -1 for x in json.loads(l)][:5] for l in open(RUN/f"progs_{FOLD}.jsonl")],dtype=np.float64)
rid=np.floor(P[:,0]/1024); idx=np.searchsorted(calls[:,1],rid,side="right")-1
ok=(idx>=0)&(rid<calls[np.clip(idx,0,None),2]); k=P[ok,3]/1e9; so=calls[idx[ok],0]
per=np.bincount(so,weights=k,minlength=max(sigs)+1); n=np.bincount(calls[:,0],minlength=max(sigs)+1)
rows=[(per[s],s) for s in sigs if any(PAT in x for x in sigs[s]["site"]) and per[s]>0]
for t,s in sorted(rows,reverse=True)[:int(sys.argv[3]) if len(sys.argv)>3 else 25]:
    e=sigs[s]
    a=[ (x.get("T",{}).get("shape"),x.get("T",{}).get("dtype"),x.get("T",{}).get("mem","")[:60]) if isinstance(x,dict) and "T" in x else str(x)[:80] for x in e["args"]]
    kw={k:str(v)[:120] for k,v in e.get("kwargs",{}).items()} if "kwargs" in e else ""
    print(f"{t:7.2f}s {n[s]:7d} {1e6*t/max(n[s],1):8.1f}us {e['op']} {e['reg']} {a} {kw}")
