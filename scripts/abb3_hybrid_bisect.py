"""Bisect every host-side IPA sub-step of the hybrid vs the reference internals."""
import os, pickle, sys, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights
from tt_bio.abodybuilder3 import abb3_compute_kernel_config, StructureModuleTT, _from_torch, _to_torch
from tt_bio._vendor.abodybuilder3.openfold.utils.rigid_utils import Rigid, Rotation
from tt_bio._vendor.abodybuilder3.openfold.utils.tensor_utils import flatten_final_dims, permute_final_dims

def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a-a.mean())*(b-b.mean())).sum()/((a-a.mean()).norm()*(b-b.mean()).norm()))

cache = "/tmp/abb3_cache"
I = pickle.load(open(os.path.join(cache, "abb3_ipa_internals.pkl"), "rb"))
sd = torch.load(ensure_abb3_weights(cache), map_location="cpu", weights_only=True)
ck = abb3_compute_kernel_config()
m = StructureModuleTT(sd, ck, ABB3_CONFIG)
ipa = m.ipa[0]
H, C = ipa.no_heads, ipa.c_hidden
Pq, Pv = ipa.no_qk_points, ipa.no_v_points
r = Rigid(Rotation(rot_mats=I["rot_mats"], quats=None), I["trans"])
mask = I["mask"]

s_tt = _from_torch(I["s"].contiguous().float()); z_tt = _from_torch(I["z"].contiguous().float())
proj = ipa(s_tt, z_tt, None, None, None)
q = _to_torch(proj["q"]).reshape(1, -1, H, C)
kv = _to_torch(proj["kv"]).reshape(1, -1, H, 2*C); k, v = torch.split(kv, C, dim=-1)
qp = _to_torch(proj["qp"]).view(1, -1, 3, H*Pq).permute(0,1,3,2)
kvp = _to_torch(proj["kvp"]).view(1, -1, 3, H*(Pq+Pv)).permute(0,1,3,2)
b = _to_torch(proj["b"])
print("q  PCC=%.4f" % pcc(q, I["q"]))
print("k  PCC=%.4f" % pcc(k, I["k"]))
print("v  PCC=%.4f" % pcc(v, I["v"]))
print("b  PCC=%.4f" % pcc(b, I["b"]))
q_pts = r[..., None].apply(qp).view(1, -1, H, Pq, 3)
kv_pts = r[..., None].apply(kvp).view(1, -1, H, Pq+Pv, 3)
k_pts, v_pts = torch.split(kv_pts, [Pq, Pv], dim=-2)
print("q_pts PCC=%.4f" % pcc(q_pts, I["q_pts"]))
print("k_pts PCC=%.4f" % pcc(k_pts, I["k_pts"]))
print("v_pts PCC=%.4f" % pcc(v_pts, I["v_pts"]))
a = torch.matmul(permute_final_dims(q, (1,0,2)), permute_final_dims(k, (1,2,0)))
a = a * math.sqrt(1.0/(3*C))
a = a + math.sqrt(1.0/3) * permute_final_dims(b, (2,0,1))
print("a_scalar PCC=%.4f" % pcc(a, I["a_scalar"]))
pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
pt_att = pt_att ** 2
pt_att = sum(torch.unbind(pt_att, dim=-1))
hw = torch.nn.functional.softplus(ipa.weights["head_weights"]).view(*((1,)*len(pt_att.shape[:-2])+(-1,1)))
hw = hw * math.sqrt(1.0/(3*(Pq*9.0/2)))
pt_att = pt_att * hw
pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
square_mask = ipa.inf * (square_mask - 1)
pt_att = permute_final_dims(pt_att, (2,0,1))
print("pt_att PCC=%.4f" % pcc(pt_att, I["pt_att"]))
a_full = a + pt_att + square_mask.unsqueeze(-3)
print("a_full PCC=%.4f" % pcc(a_full, I["a"]))
attn = torch.softmax(a_full, dim=-1)
print("attn PCC=%.4f" % pcc(attn, I["attn"]))
o = torch.matmul(attn, v.transpose(-2,-3).to(attn.dtype)).transpose(-2,-3)
o = flatten_final_dims(o, 2)
print("o PCC=%.4f" % pcc(o, I["o"]))
o_pt = torch.sum((attn[...,None,:,:,None] * permute_final_dims(v_pts,(1,3,0,2))[...,None,:,:]), dim=-2)
o_pt = permute_final_dims(o_pt, (2,0,3,1))
o_pt = r[...,None,None].invert_apply(o_pt)
print("o_pt PCC=%.4f" % pcc(o_pt, I["o_pt"]))
o_pt_norm = flatten_final_dims(torch.sqrt(torch.sum(o_pt**2, dim=-1)+ipa.eps), 2)
print("o_pt_norm PCC=%.4f" % pcc(o_pt_norm, I["o_pt_norm"]))
o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
o_pair = torch.matmul(attn.transpose(-2,-3), I["z"].to(attn.dtype))
o_pair = flatten_final_dims(o_pair, 2)
print("o_pair PCC=%.4f" % pcc(o_pair, I["o_pair"]))
cat = torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1)
print("cat PCC=%.4f" % pcc(cat, I["cat"]))
