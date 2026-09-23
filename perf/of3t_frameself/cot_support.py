"""Where does the captured cotangent live: real tokens or pads?

The self-test settled that the 768 trunk tensors reachable ONLY through cot_s reproduce
grads_f64_043.pt exactly, so cot_s is right and the defect is cot_z. This asks the cheapest
structural question about cot_z: the real backward's dL/dz_in carries 100 % of its mass on the
56 real token rows, so a cotangent with mass on the 328 pad rows cannot be the one that flowed.
"""
import json, socket, torch
torch.set_num_threads(4)
B=torch.load("/home/ttuser/of3t_modelframe/boundary_model_n384.pt",map_location="cpu",weights_only=False)
C=torch.load("/home/ttuser/of3t_modelframe/cot_model_n384.pt",map_location="cpu",weights_only=False)
cot_s,cot_z=C["cot"]
sm=B["single_mask"][0].bool(); pm=B["pair_mask"][0].bool()
n_real=int(sm.sum()); print("real tokens",n_real,"of",sm.numel())
def sq(t): return float(torch.dot(t.reshape(-1).double(),t.reshape(-1).double()))
def rep(name,t,mask):
    tt=t[0].double(); tot=sq(tt)
    on=sq(tt[mask]); off=tot-on
    nz=int((tt!=0).sum()); nz_on=int((tt[mask]!=0).sum())
    return {"name":name,"shape":list(t.shape),"squared_norm":tot,"norm":tot**0.5,
            "masked_in_squared_norm":on,"masked_out_squared_norm":off,
            "share_masked_in":(on/tot if tot else None),
            "nonzero":nz,"nonzero_masked_in":nz_on,"nonzero_masked_out":nz-nz_on,
            "numel":tt.numel(),"max_abs":float(tt.abs().max()),
            "max_abs_masked_out":(float(tt[~mask].abs().max()) if (~mask).any() else 0.0)}
out={"what":__doc__.strip().splitlines()[0],"host":socket.gethostname(),"row":"of3t-frameself",
     "defect":"D242","device_involved":False,
     "why_no_aiclk":"CPU only, no Tenstorrent device is opened",
     "boundary":"/home/ttuser/of3t_modelframe/boundary_model_n384.pt",
     "cotangent":"/home/ttuser/of3t_modelframe/cot_model_n384.pt",
     "real_tokens":n_real,"tokens":int(sm.numel()),
     "tensors":[rep("cot_s",cot_s,sm), rep("cot_z",cot_z,pm),
                rep("s_in",B["s_in"],sm), rep("z_in",B["z_in"],pm)]}
# is cot_s orthogonal to s_out?  a LayerNorm consumer forces <x, dL/dx> = 0 exactly.
out["pair_mask_true"]=int(pm.sum()); out["pair_mask_numel"]=int(pm.numel())
# distinct magnitudes in cot_s: a near-uniform cotangent is a different animal from a real one
v=cot_s[0].double().reshape(-1); nzv=v[v!=0]
out["cot_s_nonzero_value_stats"]={"n":int(nzv.numel()),
    "n_distinct_abs":int(torch.unique(nzv.abs()).numel()),
    "min_abs":float(nzv.abs().min()) if nzv.numel() else None,
    "max_abs":float(nzv.abs().max()) if nzv.numel() else None,
    "mean":float(nzv.mean()) if nzv.numel() else None}
print(json.dumps(out,indent=1))
open("/home/ttuser/of3t_frameself/COT_SUPPORT.json","w").write(json.dumps(out,indent=1))
