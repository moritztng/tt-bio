import json, sys, torch
sys.path.insert(0, "perf/of3t_fullstep64")
import score as S
R="/home/ttuser/of3t_denoise/ref384/"
ref={k:v.to(torch.float64) for k,v in torch.load(R+"grads_f64.pt",weights_only=False).items() if v is not None and bool(v.any())}
bf={k:v.to(torch.float64) for k,v in torch.load(R+"grads_bf16.pt",weights_only=False).items() if v is not None}
bij=json.load(open("perf/of3t_go384/BIJECTION_GO384.json"))
shapes=json.load(open("perf/of3t_go384/DEVICE_SHAPES_GO384.json"))["shapes"]
arm,_=S.to_upstream(S.load_device("/home/ttuser/of3t_go384/grad_GO384.pt",shapes),bij)
rows=[]
for k in ref:
  sec=S.section_of(k)
  if sec not in ("aux_heads.pairformer_embedding","aux_heads.experimentally_resolved"): continue
  m=float((ref[k]**2).sum()); a=arm.get(k, torch.zeros_like(ref[k])); b=bf.get(k, torch.zeros_like(ref[k]))
  e=float(((a-ref[k])**2).sum()); eb=float(((b-ref[k])**2).sum())
  r=(float((a*a).sum())/m)**.5
  rows.append((e,sec,k,(e/m)**.5,(eb/m)**.5,r))
for sec in ("aux_heads.pairformer_embedding","aux_heads.experimentally_resolved"):
  rs=[x for x in rows if x[1]==sec]; tot=sum(x[0] for x in rs)
  print(sec, len(rs), "tensors")
  for e,_,k,rel,brel,r in sorted(rs,reverse=True)[:8]: print("  %-70s errshare %.3f rel %.3g bf16 %.3g |a|/|ref| %.3g"%(k,e/tot,rel,brel,r))
import statistics as st
pe=[k for k in ref if S.section_of(k)=="aux_heads.pairformer_embedding"]
rat=[];cs=[]
for k in pe:
  a=arm.get(k, torch.zeros_like(ref[k])).flatten(); r=ref[k].flatten()
  rat.append(float(a.norm()/r.norm())); cs.append(float(a@r/(a.norm()*r.norm()+1e-300)))
q=lambda v: [round(x,3) for x in (min(v),st.quantiles(v,n=10)[0],st.median(v),st.quantiles(v,n=10)[-1],max(v))]
print("PE ratio min/p10/med/p90/max", q(rat)); print("PE cos", q(cs))
A=torch.cat([arm[k].flatten() for k in pe]); Rr=torch.cat([ref[k].flatten() for k in pe])
s=float(A@Rr/(Rr@Rr)); print("best scalar", s, "rel after rescale", float((A/s-Rr).norm()/Rr.norm()))
