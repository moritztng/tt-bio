"""Is the device AttentionPairBias adding the pair bias at the wrong scale?

PROTOCOL SS3c names this exact failure on PTX's SDPA: a kernel that adds the mask BEFORE
the scale computes softmax((qk + b) * scale), which is not what the reference does. Sweep
the bias coefficient c in softmax(qk/sqrt(24) + c*bias) and see whether one value explains
the device output across all four blocks. A consistent c != 1 is a formula error; no
consistent c is precision.
"""
import os, pickle, sys, torch, torch.nn.functional as F, ttnn
sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/perf/of3t_confidence"))
from forward_vs_float64 import ca_walk, rel_l2
from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead

_C_S, _C_Z, _H, _HD = 384, 128, 16, 24
sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                weights_only=False)
if hasattr(sd, "state_dict"): sd = sd.state_dict()
aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"]
si_input = g["input_embedder_real"]["out"][0].float()
si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
N = si_trunk.shape[0]
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
head = OF3ConfidenceHead(aux, dev, ckc)
steps = []
orig = OF3ConfidenceHead._host_s_block
OF3ConfidenceHead._host_s_block = lambda self, s, z, i: (
    steps.append((s.clone(), z.clone())) or orig(self, s, z, i))
head.forward(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, repr_x_pred=ca_walk(N),
             max_atom_per_token_mask=torch.ones(N * 23), use_zij_trunk_embedding=True, s_path="host")
OF3ConfidenceHead._host_s_block = orig
head._dtype = torch.float32
up = lambda x: ttnn.from_torch(x.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)
CAND = {"1 (reference)": 1.0, "1/sqrt(24)": _HD ** -0.5, "sqrt(24)": _HD ** 0.5,
        "1/sqrt(32)": 32 ** -0.5, "sqrt(32)": 32 ** 0.5, "0 (no bias)": 0.0}
print(f"{'blk':>3s} " + " ".join(f"{k:>14s}" for k in CAND) + f" {'|z|max':>9s}")
for i, (s_in, z_in) in enumerate(steps):
    blk = head.pf.blocks[i]
    bw = lambda n: head._bw(i, "attn_pair_bias." + n)
    a = F.layer_norm(s_in, (_C_S,), bw("layer_norm_a.weight"), bw("layer_norm_a.bias"))
    zn = F.layer_norm(z_in, (_C_Z,), bw("layer_norm_z.weight"), bw("layer_norm_z.bias"))
    bias = F.linear(zn, bw("linear_z.weight")).permute(2, 0, 1)
    q = F.linear(a, bw("mha.linear_q.weight"), bw("mha.linear_q.bias"))
    k = F.linear(a, bw("mha.linear_k.weight")); v = F.linear(a, bw("mha.linear_v.weight"))
    qh = q.view(N, _H, _HD).permute(1, 0, 2) * (_HD ** -0.5)
    kh = k.view(N, _H, _HD).permute(1, 0, 2); vh = v.view(N, _H, _HD).permute(1, 0, 2)
    gate = torch.sigmoid(F.linear(a, bw("mha.linear_g.weight")))
    got = torch.Tensor(ttnn.to_torch(blk.attention_pair_bias(up(a), up(z_in)))).float().reshape(N, _C_S)
    row = []
    for c in CAND.values():
        sc = torch.einsum("hqd,hkd->hqk", qh, kh) + c * bias
        o = torch.einsum("hqk,hkd->hqd", F.softmax(sc, -1), vh).permute(1, 0, 2).reshape(N, _H * _HD)
        row.append(rel_l2(got, F.linear(o * gate, bw("mha.linear_o.weight"))))
    print(f"{i:3d} " + " ".join(f"{r:14.3e}" for r in row) + f" {float(z_in.abs().max()):9.4g}")
