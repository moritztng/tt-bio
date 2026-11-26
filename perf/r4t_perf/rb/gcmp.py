import numpy as np, collections
a=np.load("/home/ttuser/r4t/perf/rb/g0.npz"); b=np.load("/home/ttuser/r4t/perf/rb/g1.npz")
agg=collections.defaultdict(list)
for k in a.files:
    x=a[k].astype(np.float64); y=b[k].astype(np.float64)
    r=np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30); c=float((x*y).sum()/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30))
    role=k.split("__")[-2]+"."+k.split("__")[-1]
    agg[role].append((r,c,np.linalg.norm(x),np.linalg.norm(y)))
for role,v in sorted(agg.items(), key=lambda kv:-max(t[0] for t in kv[1]))[:16]:
    print(f"{role:32s} n={len(v)} rel max {max(t[0] for t in v):.3e} med {sorted(t[0] for t in v)[len(v)//2]:.3e} cos min {min(t[1] for t in v):.6f} |g0|/|g1| {v[0][2]:.3e}/{v[0][3]:.3e}")
