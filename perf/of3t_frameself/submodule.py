"""of3t-frameself: inside ONE block, which sub-module's gradient diverges?

`--blockprobe` settled that the reference is right: `torch.autograd.grad` on block 47's
parameters reproduces `grads_f64_043.pt` at 3.04e-15, and 16 of that block's 57 tensors are
BIT-IDENTICAL between the real backward and the injected replay. Same block object, same input,
same output cotangent, same weights, and 41 of 57 disagree. This splits those 57 by sub-module
so the divergence is attached to an op rather than to "the pair stack".
"""
import json, re, socket, torch
torch.set_num_threads(4)
REF="/home/ttuser/of3t-campaign-refs/bundle_min_043/grads_f64_043.pt"
ARM="/home/ttuser/of3t_twoside/ctrl_f64.pt"
def load(p):
    d=torch.load(p,map_location="cpu",weights_only=False,mmap=True)
    return d["grads"] if isinstance(d,dict) and "grads" in d else d
ref,arm=load(REF),load(ARM)
out={"what":__doc__.strip().splitlines()[0],"host":socket.gethostname(),"row":"of3t-frameself",
     "defect":"D242","device_involved":False,
     "why_no_aiclk":"CPU only, no Tenstorrent device is opened",
     "reference":{"path":REF,"is":"the full-model float64 backward, batch_step003, num_recycles 0",
                  "injected":False},
     "arm":{"path":ARM,"is":"of3t-twoside's injected float64 replay","injected":True},
     "blocks":{}}
for blk in (47,46,24,0):
    pre=f"pairformer_stack.blocks.{blk}."
    names=sorted(n for n in ref if n.startswith(pre))
    groups={}
    for n in names:
        sub=n[len(pre):].split(".")[0]
        if sub in ("pair_stack",):
            sub="pair_stack."+n[len(pre)+len("pair_stack."):].split(".")[0]
        groups.setdefault(sub,[]).append(n)
    rows={}
    for sub,ns in sorted(groups.items()):
        dot=a2=r2=e2=0.0; nbit=0; worst=-1.0; wn=None
        for n in ns:
            r=ref[n].to(torch.float64).reshape(-1); m=arm[n].to(torch.float64).reshape(-1)
            rn2=float(torch.dot(r,r)); r2+=rn2; a2+=float(torch.dot(m,m))
            dot+=float(torch.dot(m,r)); e=float(torch.dot(m-r,m-r)); e2+=e
            if torch.equal(m,r): nbit+=1
            rel=(e/rn2)**0.5 if rn2>0 else (0.0 if e==0 else float("inf"))
            if rel>worst: worst,wn=rel,n
        rows[sub]={"n_tensors":len(ns),"n_bit_identical":nbit,
                   "rel_l2_as_is":(e2/r2)**0.5 if r2 else None,
                   "norm_ratio_arm_over_ref":(a2/r2)**0.5 if r2 else None,
                   "cos":dot/(a2*r2)**0.5 if a2>0 and r2>0 else None,
                   "ref_squared_norm":r2,"worst_rel_l2":worst,"worst_tensor":wn}
    out["blocks"][str(blk)]=rows
print(json.dumps(out["blocks"],indent=1))
open("/home/ttuser/of3t_frameself/SUBMODULE.json","w").write(json.dumps(out,indent=1))
