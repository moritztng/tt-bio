"""Which pair-bias convention is closer to the reference, per attention, on OF3 trunk block 0.

The single `scale_pair_bias` flag fed both `TriangleAttention` (z-track) and
`AttentionPairBias` (s-track), so flipping it moved both at once and any one-flag A/B is a
net of two changes. With the flag split, this runs the full 2x2 and reads the two tracks
separately, against the REAL reference golden `pairformer_block0` (of3-p2-155k weights,
upstream capture in ~/of3_ref_out.pkl) -- not against a host reimplementation of our own
kernel, which would only prove we agree with ourselves.

Read the s track on the UPDATE, not the state: s carries a residual far larger than one
block adds, so a relative L2 on s mostly reports the residual both arms share.

    python3 perf/of3t_pairbias/block0_arms.py
"""
import json
import os
import pickle
import sys

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from perf.of3t_confidence.forward_vs_float64 import pcc, rel_l2  # noqa: E402
from tt_bio.openfold3_weights import _sub, remap_pairformer_block  # noqa: E402
from tt_bio.tenstorrent import PairformerLayer, accurate_softmax_site, get_device  # noqa: E402

OUT = "perf/of3t_pairbias/block0_arms.json"

sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                weights_only=False)
if hasattr(sd, "state_dict"):
    sd = sd.state_dict()
blk_sd = _sub(sd, "pairformer_stack.blocks.0")
pd = remap_pairformer_block(blk_sd)
g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"]["pairformer_block0"]
s_in, z_in = (t.float() for t in g["in"][:2])
s_out, z_out = (t.float() for t in g["out"][:2])
N, C_S = s_in.shape[-2], s_in.shape[-1]
n_heads = blk_sd["attn_pair_bias.linear_z.weight"].shape[0]
head_dim = blk_sd["attn_pair_bias.mha.linear_q.weight"].shape[0] // n_heads
tn = blk_sd["pair_stack.tri_att_start.linear_z.weight"].shape[0]
td = blk_sd["pair_stack.tri_att_start.mha.linear_q.weight"].shape[0] // tn
hdr = (f"trunk block0: N={N} c_s={C_S} apb heads={n_heads} head_dim={head_dim} "
       f"1/sqrt(d)={head_dim ** -0.5:.4f} | tri heads={tn} head_dim={td}")
print(hdr)

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)


def up(x):
    x = x.float()
    return ttnn.from_torch(x.reshape(1, *x.shape[-2:]) if x.dim() == 2 else x.reshape(1, *x.shape[-3:]),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


rows = []
for apb in (False, True):
    for tri in (False, True):
        layer = PairformerLayer(td, tn, head_dim, n_heads, True, pd, ckc,
                                scale_pair_bias=apb, tri_att_scale_pair_bias=tri,
                                fp32_softmax=True,
                                accurate_softmax=accurate_softmax_site("openfold3.trunk"))
        s_d, z_d = layer(up(s_in), up(z_in))
        s_g = torch.Tensor(ttnn.to_torch(s_d)).float().reshape(N, C_S)
        z_g = torch.Tensor(ttnn.to_torch(z_d)).float().reshape(N, N, -1)
        row = dict(apb_scale=apb, tri_scale=tri,
                   s_update_rel_l2=rel_l2(s_g - s_in, s_out - s_in),
                   s_rel_l2=rel_l2(s_g, s_out), s_pcc=pcc(s_g, s_out),
                   z_rel_l2=rel_l2(z_g, z_out), z_pcc=pcc(z_g, z_out))
        rows.append(row)
        print(f"  apb={str(apb):5s} tri={str(tri):5s}  s UPDATE relL2 {row['s_update_rel_l2']:.4e}"
              f"  s PCC {row['s_pcc']:.6f}  z relL2 {row['z_rel_l2']:.4e}  z PCC {row['z_pcc']:.6f}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(header=hdr, arms=rows), open(OUT, "w"), indent=1)
print("wrote", OUT)
