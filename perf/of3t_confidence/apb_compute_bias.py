import os, pickle, sys, torch, torch.nn.functional as F, ttnn
sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/perf/of3t_confidence"))
from forward_vs_float64 import ca_walk, rel_l2
from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead
_C_Z = 128
sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu", weights_only=False)
if hasattr(sd, "state_dict"): sd = sd.state_dict()
aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"]
si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
N = si_trunk.shape[0]
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
head = OF3ConfidenceHead(aux, dev, ckc)
apb = head.pf.blocks[0].attention_pair_bias
print(f"head_dim={apb.head_dim} n_heads={apb.n_heads} _bias_scale={apb._bias_scale} "
      f"fp32_softmax={apb.fp32_softmax} accurate={apb.accurate_softmax} "
      f"compute_pair_bias={apb.compute_pair_bias} concat_heads={apb._concat_heads}")
print(f"1/sqrt(head_dim)={apb.head_dim**-0.5:.6f}")
z = zij_trunk
zd = ttnn.from_torch(z.unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
got = torch.Tensor(ttnn.to_torch(apb.compute_bias(zd))).float().reshape(apb.n_heads, N, N)
bw = lambda n: head._w["pairformer_embedding.pairformer_stack.blocks.0.attn_pair_bias." + n].float()
zn = F.layer_norm(z, (_C_Z,), bw("layer_norm_z.weight"), bw("layer_norm_z.bias"))
ref = F.linear(zn, bw("linear_z.weight")).permute(2, 0, 1)
print(f"compute_bias vs reference           relL2 {rel_l2(got, ref):.4e}")
print(f"compute_bias vs reference/sqrt(24)  relL2 {rel_l2(got, ref*apb.head_dim**-0.5):.4e}")
print(f"ratio device/reference (median)     {float((got/ref).median()):.6f}")
