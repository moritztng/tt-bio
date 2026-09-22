"""Each attention on its own, both flag values, against a float64 reference of OF3's formula.

The block-level and stack-level goldens move both tracks at once and read a state, so they
cannot say which of the two attentions a flag is right for. This isolates them: one
`TriangleAttention(ending=False)` and one `AttentionPairBias`, driven by the real golden's
block-0 inputs and the real of3-p2-155k weights, each compared against the SAME arithmetic
written out in float64 on host from the raw checkpoint tensors.

float64 matters here and is not ceremony: the thing under test is a sqrt(d) = 4.899 factor
on the additive bias, and a bf16 reference would carry its own 4e-3 and could not resolve a
disagreement smaller than that. The reference is a transcription of OpenFold3's documented
formula (AF3 Alg 14 for the triangle track, Alg 24 for the token track), not a second call
into the code under test.

    python3 perf/of3t_pairbias/attn_f64.py
"""
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

OUT = "perf/of3t_pairbias/attn_f64.json"
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


def ref_apb(a, z):
    """AF3 Alg 24 AttentionPairBias, float64. `a` is the pre-normalised single."""
    zn = F.layer_norm(z, (C_Z,), W("attn_pair_bias.layer_norm_z.weight"),
                      W("attn_pair_bias.layer_norm_z.bias"))
    bias = F.linear(zn, W("attn_pair_bias.linear_z.weight")).permute(2, 0, 1)  # [H, N, N]
    q = F.linear(a, W("attn_pair_bias.mha.linear_q.weight"), W("attn_pair_bias.mha.linear_q.bias"))
    k = F.linear(a, W("attn_pair_bias.mha.linear_k.weight"))
    v = F.linear(a, W("attn_pair_bias.mha.linear_v.weight"))
    sh = lambda t: t.view(N, APB_H, APB_D).permute(1, 0, 2)
    q, k, v = sh(q) / math.sqrt(APB_D), sh(k), sh(v)
    o = torch.einsum("hqk,hkd->hqd", F.softmax(torch.einsum("hqd,hkd->hqk", q, k) + bias, dim=-1), v)
    o = o.permute(1, 0, 2).reshape(N, APB_H * APB_D)
    gate = torch.sigmoid(F.linear(a, W("attn_pair_bias.mha.linear_g.weight")))
    return F.linear(o * gate, W("attn_pair_bias.mha.linear_o.weight"))


def ref_tri_start(z):
    """AF3 Alg 14 starting-node TriangleAttention, float64. Returns the update."""
    p = "pair_stack.tri_att_start."
    zn = F.layer_norm(z, (C_Z,), W(p + "layer_norm.weight"), W(p + "layer_norm.bias"))
    sh = lambda t: t.view(N, N, TRI_H, TRI_D).permute(2, 0, 1, 3)          # [h, i, j, d]
    q = sh(F.linear(zn, W(p + "mha.linear_q.weight"))) / math.sqrt(TRI_D)
    k = sh(F.linear(zn, W(p + "mha.linear_k.weight")))
    v = sh(F.linear(zn, W(p + "mha.linear_v.weight")))
    b = F.linear(zn, W(p + "linear_z.weight")).permute(2, 0, 1)            # [h, j, k]
    att = F.softmax(torch.einsum("hijd,hikd->hijk", q, k) + b[:, None, :, :], dim=-1)
    o = torch.einsum("hijk,hikd->hijd", att, v)                            # [h, i, j, d]
    gate = torch.sigmoid(F.linear(zn, W(p + "mha.linear_g.weight")))       # [i, j, c_z]
    o = o.permute(1, 2, 0, 3).reshape(N, N, C_Z) * gate
    return F.linear(o, W(p + "mha.linear_o.weight"))


a_ref = F.layer_norm(s_in, (C_S,), W("attn_pair_bias.layer_norm_a.weight"),
                     W("attn_pair_bias.layer_norm_a.bias"))
apb_ref = ref_apb(a_ref, z_in)
tri_ref = ref_tri_start(z_in)
print(f"block0 N={N} c_s={C_S} c_z={C_Z} | apb heads={APB_H} d={APB_D} sqrt(d)={APB_D ** 0.5:.4f}"
      f" | tri heads={TRI_H} d={TRI_D} sqrt(d)={TRI_D ** 0.5:.4f}")
print(f"f64 reference norms: apb update {apb_ref.norm():.6e}  tri update {tri_ref.norm():.6e}")

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
up = lambda x: ttnn.from_torch(x.float().reshape(1, *x.shape), layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)
dn = lambda t, shape: torch.Tensor(ttnn.to_torch(t)).double().reshape(*shape)

rows = []
for flag in (False, True):
    layer = PairformerLayer(TRI_D, TRI_H, APB_D, APB_H, True, pd, ckc,
                            scale_pair_bias=flag, tri_att_scale_pair_bias=flag,
                            fp32_softmax=True,
                            accurate_softmax=accurate_softmax_site("openfold3.trunk"))
    apb_d = dn(layer.attention_pair_bias(up(a_ref), up(z_in)), (N, C_S))
    tri_d = dn(layer.triangle_attention_start(up(z_in)), (N, N, C_Z))
    row = dict(flag=flag,
               apb_rel_l2=rel_l2(apb_d, apb_ref), apb_pcc=pcc(apb_d, apb_ref),
               apb_ratio=float(apb_d.norm() / apb_ref.norm()),
               tri_rel_l2=rel_l2(tri_d, tri_ref), tri_pcc=pcc(tri_d, tri_ref),
               tri_ratio=float(tri_d.norm() / tri_ref.norm()))
    rows.append(row)
    print(f"  scale_pair_bias={str(flag):5s}  APB relL2 {row['apb_rel_l2']:.4e} PCC {row['apb_pcc']:.6f}"
          f" |x|/|ref| {row['apb_ratio']:.4f}   TRI relL2 {row['tri_rel_l2']:.4e}"
          f" PCC {row['tri_pcc']:.6f} |x|/|ref| {row['tri_ratio']:.4f}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(tokens=N, apb_heads=APB_H, apb_head_dim=APB_D, tri_heads=TRI_H,
               tri_head_dim=TRI_D, arms=rows), open(OUT, "w"), indent=1)
print("wrote", OUT)
