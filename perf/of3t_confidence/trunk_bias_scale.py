"""Does the shipped OF3 TRUNK carry the same pair-bias scaling error?

The trunk builds its 48-block Pairformer with `scale_pair_bias=False` -- the identical
construction the confidence head used. The token-level AttentionPairBias computes
(q@k^T + z) * head_dim**-0.5 inline (tenstorrent.py, the non-atom-level branch), so the
bias has to arrive pre-baked by sqrt(head_dim) to come out unscaled, which is exactly what
scale_pair_bias=True does. If that reading is right, the trunk's s-track has been adding
its pair bias at 1/sqrt(head_dim) of the reference value for every one of its 48 blocks.

Gated against the REAL reference golden `pairformer_block0` (of3_golden.py, real
of3-p2-155k weights), not against a host reimplementation: block 0's own input in, block
0's own output out, one arm per flag value.
"""
import os, pickle, sys, torch, ttnn
sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/perf/of3t_confidence"))
from forward_vs_float64 import rel_l2, pcc
from tt_bio.tenstorrent import PairformerLayer, get_device, accurate_softmax_site
from tt_bio.openfold3_weights import remap_pairformer_block, _sub

sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                weights_only=False)
if hasattr(sd, "state_dict"): sd = sd.state_dict()
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
print(f"trunk block0: N={N} c_s={C_S} apb heads={n_heads} head_dim={head_dim} "
      f"-> 1/sqrt(d)={head_dim**-0.5:.4f}")
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
up = lambda x: ttnn.from_torch(x.float().reshape(1, *x.shape[-2:]) if x.dim() == 2 else x.float().reshape(1, *x.shape[-3:]),
                               layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
for flag in (False, True):
    layer = PairformerLayer(td, tn, head_dim, n_heads, True, pd, ckc,
                            scale_pair_bias=flag, fp32_softmax=True,
                            accurate_softmax=accurate_softmax_site("openfold3.trunk"))
    s_d, z_d = layer(up(s_in), up(z_in))
    s_g = torch.Tensor(ttnn.to_torch(s_d)).float().reshape(N, C_S)
    z_g = torch.Tensor(ttnn.to_torch(z_d)).float().reshape(N, N, -1)
    # The UPDATE, not the state. s carries a residual far larger than one block adds, so a
    # relative L2 on s reads the residual both arms share -- the same defect PROTOCOL SS7a
    # corrects for in the trajectory, applied one block at a time.
    print(f"  scale_pair_bias={str(flag):5s}  s relL2 {rel_l2(s_g, s_out):.4e} PCC {pcc(s_g, s_out):.6f}"
          f"   UPDATE relL2 {rel_l2(s_g - s_in, s_out - s_in):.4e}"
          f"   z relL2 {rel_l2(z_g, z_out):.4e}")
