# 16-rank DP arm grade: median over ranks of each step phase (steps 3+), and the spread of per-rank medians. Edit the arm list per run.
import json,glob,statistics as st,os
base=os.path.expanduser("~/fleet/r4t/perf/glxab")
K=["step_s","fwd_s","bwd_s","fold_bwd_s","opt_s","rebuild_s"]
for arm in ["C3","B3","B4","C4"]:
    v={k:[] for k in K}; per=[]
    for f in glob.glob(f"{base}/w16_{arm}/rank*.log"):
        rows=[]
        for l in open(f):
            try: j=json.loads(l)
            except Exception: continue
            if isinstance(j,dict) and "step_s" in j and j.get("step",0)>=3: rows.append(j)
        for r in rows:
            for k in K: v[k].append(r.get(k,0) or 0)
        per.append(st.median(r["step_s"] for r in rows))
    print(arm," ".join("%s %.2f"%(k[:-2],st.median(v[k])) for k in K), "rank-median spread %.2f-%.2f"%(min(per),max(per)))
