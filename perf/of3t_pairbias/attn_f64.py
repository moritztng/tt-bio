"""Which value of `scale_pair_bias` the OpenFold3 trunk should carry, settled in float64.

Two attentions live under one `PairformerLayer` and one flag used to feed both. This isolates
them, drives each from the real golden's block-0 inputs and the real of3-p2-155k weights, and
scores them against a transcription of OpenFold3's own formula written out in float64 on host
(AF3 Alg 24 for the token track, Alg 14 for the triangle track) -- not against a second call
into the code under test.

float64 is load-bearing: the quantity under test is a sqrt(24) = 4.899 factor on an additive
bias, and a bf16 reference carries its own 4e-3 and could not resolve it.

The decisive column is `apb_vs_scaled`. Besides the true reference this also builds a SECOND
float64 reference in which the token pair bias is deliberately multiplied by 1/sqrt(24), i.e.
exactly the formula the shipped flag makes the kernel compute. If the shipped arm is a
convention error rather than a precision one, it must sit far from the true reference and right
on top of the deliberately-wrong one. That is a positive identification, not just a gap.

    TT_VISIBLE_DEVICES=N python3 perf/of3t_pairbias/attn_f64.py
"""
import argparse
import json
import math
import os
import pickle
import sys

import torch
import torch.nn.functional as F
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.openfold3_weights import _sub, remap_pairformer_block  # noqa: E402
from tt_bio.tenstorrent import PairformerLayer, accurate_softmax_site, get_device  # noqa: E402
import tt_bio  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="perf/of3t_pairbias/attn_f64.json")
args = ap.parse_args()

assert os.path.realpath(tt_bio.__file__).startswith(os.path.realpath(os.getcwd())), \
    f"imported tt_bio from {tt_bio.__file__}, not this checkout"

F64 = torch.float64


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                weights_only=False)
if hasattr(sd, "state_dict"):
    sd = sd.state_dict()
blk = _sub(sd, "pairformer_stack.blocks.0")
pd = remap_pairformer_block(blk)
W = lambda k: blk[k].to(F64)

g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"]["pairformer_block0"]
s_in, z_in = (t.to(F64) for t in g["in"][:2])
N, C_S = s_in.shape[-2], s_in.shape[-1]
C_Z = z_in.shape[-1]
APB_H = blk["attn_pair_bias.linear_z.weight"].shape[0]
APB_D = C_S // APB_H
TRI_H = blk["pair_stack.tri_att_start.linear_z.weight"].shape[0]
TRI_D = C_Z // TRI_H


def ref_apb(a, z, bias_factor=1.0):
    """AF3 Alg 24 AttentionPairBias in float64. `a` is the pre-normalised single.

    `bias_factor` scales the pair bias. 1.0 is the architecture; 1/sqrt(d) is what the shipped
    flag makes the kernel compute, and is here so the shipped arm can be identified rather than
    only faulted."""
    zn = F.layer_norm(z, (C_Z,), W("attn_pair_bias.layer_norm_z.weight"),
                      W("attn_pair_bias.layer_norm_z.bias"))
    bias = F.linear(zn, W("attn_pair_bias.linear_z.weight")).permute(2, 0, 1) * bias_factor
    q = F.linear(a, W("attn_pair_bias.mha.linear_q.weight"), W("attn_pair_bias.mha.linear_q.bias"))
    k = F.linear(a, W("attn_pair_bias.mha.linear_k.weight"))
    v = F.linear(a, W("attn_pair_bias.mha.linear_v.weight"))
    sh = lambda t: t.view(N, APB_H, APB_D).permute(1, 0, 2)
    q, k, v = sh(q) / math.sqrt(APB_D), sh(k), sh(v)
    o = torch.einsum("hqk,hkd->hqd", F.softmax(torch.einsum("hqd,hkd->hqk", q, k) + bias, dim=-1), v)
    o = o.permute(1, 0, 2).reshape(N, APB_H * APB_D)
    gate = torch.sigmoid(F.linear(a, W("attn_pair_bias.mha.linear_g.weight")))
    return F.linear(o * gate, W("attn_pair_bias.mha.linear_o.weight"))


def ref_tri_start(z, bias_factor=1.0):
    """AF3 Alg 14 starting-node TriangleAttention in float64. Returns the update."""
    p = "pair_stack.tri_att_start."
    zn = F.layer_norm(z, (C_Z,), W(p + "layer_norm.weight"), W(p + "layer_norm.bias"))
    sh = lambda t: t.view(N, N, TRI_H, TRI_D).permute(2, 0, 1, 3)
    q = sh(F.linear(zn, W(p + "mha.linear_q.weight"))) / math.sqrt(TRI_D)
    k = sh(F.linear(zn, W(p + "mha.linear_k.weight")))
    v = sh(F.linear(zn, W(p + "mha.linear_v.weight")))
    b = F.linear(zn, W(p + "linear_z.weight")).permute(2, 0, 1) * bias_factor
    att = F.softmax(torch.einsum("hijd,hikd->hijk", q, k) + b[:, None, :, :], dim=-1)
    o = torch.einsum("hijk,hikd->hijd", att, v)
    gate = torch.sigmoid(F.linear(zn, W(p + "mha.linear_g.weight")))
    o = o.permute(1, 2, 0, 3).reshape(N, N, C_Z) * gate
    return F.linear(o, W(p + "mha.linear_o.weight"))


a_ref = F.layer_norm(s_in, (C_S,), W("attn_pair_bias.layer_norm_a.weight"),
                     W("attn_pair_bias.layer_norm_a.bias"))
apb_ref = ref_apb(a_ref, z_in)
apb_ref_scaled = ref_apb(a_ref, z_in, bias_factor=1.0 / math.sqrt(APB_D))
tri_ref = ref_tri_start(z_in)
tri_ref_scaled = ref_tri_start(z_in, bias_factor=1.0 / math.sqrt(TRI_D))
print(f"block0 N={N} c_s={C_S} c_z={C_Z} | apb heads={APB_H} d={APB_D} 1/sqrt(d)="
      f"{APB_D ** -0.5:.4f} | tri heads={TRI_H} d={TRI_D} 1/sqrt(d)={TRI_D ** -0.5:.4f}")
print(f"f64 refs: apb {apb_ref.norm():.6e} (0.204x-bias twin {apb_ref_scaled.norm():.6e})  "
      f"tri {tri_ref.norm():.6e}")
# The two references must actually differ, or every number below is 0/0.
print(f"f64 control: true ref vs 0.204x-bias twin  apb {rel_l2(apb_ref_scaled, apb_ref):.4e}  "
      f"tri {rel_l2(tri_ref_scaled, tri_ref):.4e}")

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
up = lambda x: ttnn.from_torch(x.float().reshape(1, *x.shape), layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)
dn = lambda t, shape: torch.Tensor(ttnn.to_torch(t)).double().reshape(*shape)

ARMS = (("shipped", False, None), ("fixed", True, False), ("both_true", True, True))
rows = []
for name, spb, tri in ARMS:
    layer = PairformerLayer(TRI_D, TRI_H, APB_D, APB_H, True, pd, ckc,
                            scale_pair_bias=spb, tri_att_scale_pair_bias=tri,
                            fp32_softmax=True,
                            accurate_softmax=accurate_softmax_site("openfold3.trunk"))
    apb_d = dn(layer.attention_pair_bias(up(a_ref), up(z_in)), (N, C_S))
    tri_d = dn(layer.triangle_attention_start(up(z_in)), (N, N, C_Z))
    row = dict(arm=name, scale_pair_bias=spb, tri_att_scale_pair_bias=tri,
               apb_vs_ref=rel_l2(apb_d, apb_ref), apb_pcc=pcc(apb_d, apb_ref),
               apb_vs_scaled=rel_l2(apb_d, apb_ref_scaled),
               tri_vs_ref=rel_l2(tri_d, tri_ref), tri_pcc=pcc(tri_d, tri_ref),
               tri_vs_scaled=rel_l2(tri_d, tri_ref_scaled))
    rows.append(row)
    print(f"  {name:9s} spb={str(spb):5s} tri={str(tri):5s} | APB vs ref {row['apb_vs_ref']:.4e} "
          f"PCC {row['apb_pcc']:.6f}  vs 0.204x-twin {row['apb_vs_scaled']:.4e} | TRI vs ref "
          f"{row['tri_vs_ref']:.4e} PCC {row['tri_pcc']:.6f}  vs twin {row['tri_vs_scaled']:.4e}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(dict(tokens=N, apb_heads=APB_H, apb_head_dim=APB_D, tri_heads=TRI_H,
               tri_head_dim=TRI_D, card=os.environ.get("TT_VISIBLE_DEVICES"),
               f64_control=dict(apb=rel_l2(apb_ref_scaled, apb_ref),
                                tri=rel_l2(tri_ref_scaled, tri_ref)),
               arms=rows), open(args.out, "w"), indent=1)
print("wrote", args.out)
