"""The 48-block OF3 trunk Pairformer, one flag combination per run, against the real golden.

`pairformer_stack_real` is the upstream OpenFold3 capture at the production 76-token length
with the production of3-p2-155k weights: the same s and z in, the same s and z out. One
block cannot separate the two tracks well (block 0's z update is a small part of the state
it lands in); 48 can, because whatever each convention does wrong is applied 48 times.

One arm per process: four 48-block Pairformers do not fit one device context, and reusing a
process would price arm 2 against arm 1's cached weights.

    python3 perf/of3t_pairbias/stack_arms.py --apb 1 --tri 0
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from perf.of3t_confidence.forward_vs_float64 import pcc, rel_l2  # noqa: E402
from tt_bio.openfold3_trunk import _N_PAIRFORMER_BLOCKS, _PF_DIMS  # noqa: E402
from tt_bio.openfold3_weights import remap_pairformer_stack  # noqa: E402
from tt_bio.tenstorrent import Pairformer, accurate_softmax_site, get_device  # noqa: E402

OUT = "perf/of3t_pairbias/stack_arms.json"

ap = argparse.ArgumentParser()
ap.add_argument("--apb", type=int, required=True, help="scale_pair_bias for AttentionPairBias")
ap.add_argument("--tri", type=int, required=True, help="tri_att_scale_pair_bias")
ap.add_argument("--key", default="pairformer_stack_real")
a = ap.parse_args()
apb, tri = bool(a.apb), bool(a.tri)

sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                weights_only=False)
if hasattr(sd, "state_dict"):
    sd = sd.state_dict()
pf_sd = remap_pairformer_stack(sd, prefix="pairformer_stack")
g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"][a.key]
s_in, z_in = (t.float() for t in g["in"][:2])
s_out, z_out = (t.float() for t in g["out"][:2])
N, C_S = s_in.shape[-2], s_in.shape[-1]

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
up = lambda x: ttnn.from_torch(x.float().reshape(1, *x.shape), layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)
pf = Pairformer(_N_PAIRFORMER_BLOCKS, *_PF_DIMS, True, pf_sd, ckc,
                scale_pair_bias=apb, tri_att_scale_pair_bias=tri, fp32_softmax=True,
                accurate_softmax=accurate_softmax_site("openfold3.trunk"))
t0 = time.perf_counter()
s_d, z_d = pf(up(s_in), up(z_in))
s_g = torch.Tensor(ttnn.to_torch(s_d)).float().reshape(N, C_S)
z_g = torch.Tensor(ttnn.to_torch(z_d)).float().reshape(N, N, -1)
row = dict(key=a.key, blocks=_N_PAIRFORMER_BLOCKS, tokens=N, apb_scale=apb, tri_scale=tri,
           s_update_rel_l2=rel_l2(s_g - s_in, s_out - s_in), s_rel_l2=rel_l2(s_g, s_out),
           s_pcc=pcc(s_g, s_out),
           z_update_rel_l2=rel_l2(z_g - z_in, z_out - z_in), z_rel_l2=rel_l2(z_g, z_out),
           z_pcc=pcc(z_g, z_out), seconds=time.perf_counter() - t0)
print(f"{a.key} N={N} blocks={_N_PAIRFORMER_BLOCKS} apb={apb} tri={tri}")
print(f"  s UPDATE relL2 {row['s_update_rel_l2']:.4e}  s PCC {row['s_pcc']:.6f}")
print(f"  z UPDATE relL2 {row['z_update_rel_l2']:.4e}  z PCC {row['z_pcc']:.6f}  {row['seconds']:.1f}s")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
all_rows = json.load(open(OUT)) if os.path.exists(OUT) else []
all_rows = [r for r in all_rows
            if (r["key"], r["apb_scale"], r["tri_scale"]) != (a.key, apb, tri)] + [row]
json.dump(all_rows, open(OUT, "w"), indent=1)
print("wrote", OUT)
