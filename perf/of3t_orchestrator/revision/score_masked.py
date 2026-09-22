import json, torch
L=lambda p: torch.load(p,map_location="cpu",weights_only=False)
D="/tmp/of3t/revarm/"
b=L(D+"trunk_entry_043.pt")
tm=b["single_mask"][0].bool()                 # [384]
pm=(tm[:,None]&tm[None,:])                    # [56x56 real block]
a43,a50,a50off,abf=(L(D+f) for f in ("out_043_fp32.pt","out_050_fp32.pt","out_050_tboff.pt","out_043_bf16.pt"))
def sel(t,k):
    t=t[0]
    return t[tm] if k=="s" else t[pm]
def dist(e,t):
    e,t=e.double(),t.double(); d=e-t
    ne,nt,nd=float(e.norm()),float(t.norm()),float(d.norm())
    cos=float((d.flatten()@t.flatten())/(nd*nt)) if nd>0 else 0.0
    return {"rel_vs_t":nd/nt,"norm_ratio":ne/nt,"cos":cos}
out={"scope":"REAL TOKENS ONLY (56 of 384); padded rows excluded from numerator and denominator",
     "n_real_tokens":int(tm.sum())}
out["CONTROL"]={k:dist(sel(a50off[k],k),sel(a43[k],k)) for k in ("s","z")}
out["REVISION_050_vs_043"]={k:dist(sel(a50[k],k),sel(a43[k],k)) for k in ("s","z")}
out["UPSTREAM_OWN_BF16"]={k:dist(sel(abf[k],k),sel(a43[k],k)) for k in ("s","z")}
out["R_in_bf16_units"]={k:out["REVISION_050_vs_043"][k]["rel_vs_t"]/out["UPSTREAM_OWN_BF16"][k]["rel_vs_t"] for k in ("s","z")}
print(json.dumps(out,indent=1)); json.dump(out,open(D+"score_masked.json","w"),indent=1)
