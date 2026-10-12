#!/usr/bin/env python3
"""Device kernel seconds of one census fold by region and by op+call site. usage: agg.py RUN_DIR/<arm>"""
import json,sys,collections
import numpy as np
R=sys.argv[1]
sigs={}
for l in open(f"{R}/sig_full.jsonl"):
    e=json.loads(l)
    if "op" in e: sigs[e["sig"]]=e
calls=np.array([json.loads(l) for l in open(f"{R}/ops_full.jsonl")],dtype=np.int64)
calls=calls[np.argsort(calls[:,1],kind="stable")]
P=np.array([[x if x is not None else -1 for x in json.loads(l)] for l in open(f"{R}/progs_full.jsonl")],dtype=np.float64)
P[:,0]=np.floor(P[:,0]/1024)
K=3
P[:,K]=np.where(P[:,K]>1e9,0,P[:,K])
idx=np.searchsorted(calls[:,1],P[:,0],side="right")-1
ok=(idx>=0)&(P[:,0]<calls[np.clip(idx,0,None),2])
print("progs",len(P),"unattr",(~ok).sum(), "unattr kernel s",P[~ok,K].sum()/1e9)
P=P[ok]; idx=idx[ok]; sg=calls[idx,0]; cyc=calls[idx,3]; st=calls[idx,4]
tot=P[:,K].sum()/1e9
print("total kernel s %.2f"%tot)
by=collections.Counter(); byop=collections.Counter(); bysite=collections.Counter(); cnt=collections.Counter()
for s,k in zip(sg,P[:,K]):
    e=sigs[s]; r=e["reg"]; by[r]+=k
    byop[(r.split("/")[0],e["op"])]+=k
    bysite[(e["op"],e["site"][-1] if e["site"] else "", r)]+=k; cnt[(e["op"],e["site"][-1] if e["site"] else "", r)]+=1
print("== by reg")
for r,k in by.most_common(25): print("%8.3f  %s"%(k/1e9,r))
print("== by op+site")
for r,k in bysite.most_common(45): print("%8.3f %6d  %s"%(k/1e9,cnt[r],r))
