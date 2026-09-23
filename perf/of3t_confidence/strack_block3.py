"""Why block 3 and not the others. Magnitudes of what each block is handed and what it adds."""
import os, pickle, sys, torch, torch.nn.functional as F, ttnn
sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/perf/of3t_confidence"))
from forward_vs_float64 import ca_walk
from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead

CK = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
sd = torch.load(CK, map_location="cpu", weights_only=False)
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
def spy(self, s, z, i):
    out = orig(self, s, z, i); steps.append((s.clone(), z.clone(), out.clone())); return out
OF3ConfidenceHead._host_s_block = spy
head.forward(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, repr_x_pred=ca_walk(N),
             max_atom_per_token_mask=torch.ones(N * 23), use_zij_trunk_embedding=True, s_path="host")
OF3ConfidenceHead._host_s_block = orig
print(f"{'blk':>3s} {'|s_in|max':>11s} {'s_in rowstd':>12s} {'|update|max':>12s} "
      f"{'upd/s_in':>9s} {'|z|max':>10s} {'|Wq|max':>9s} {'|Wo|max':>9s}")
for i, (s_in, z_in, s_out) in enumerate(steps):
    u = s_out - s_in
    bw = lambda n: aux[f"pairformer_embedding.pairformer_stack.blocks.{i}.{n}"].float()
    print(f"{i:3d} {float(s_in.abs().max()):11.4g} {float(s_in.std(-1).mean()):12.4g} "
          f"{float(u.abs().max()):12.4g} {float(u.norm()/s_in.norm()):9.3e} "
          f"{float(z_in.abs().max()):10.4g} "
          f"{float(bw('attn_pair_bias.mha.linear_q.weight').abs().max()):9.3g} "
          f"{float(bw('attn_pair_bias.mha.linear_o.weight').abs().max()):9.3g}")
# where inside block 3 does it part? re-run block 3 on device with an fp32 input s.
blk = head.pf.blocks[3]
s_in, z_in, s_out = steps[3]
def up(x, dt=ttnn.bfloat16):
    return ttnn.from_torch(x.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
rel = lambda a, b: float((a - b).norm() / b.norm())
for i, (a, b, c) in enumerate(steps):
    sd_, _ = head.pf.blocks[i](up(a), up(b))
    got = torch.Tensor(ttnn.to_torch(sd_)).float().reshape(N, 384)
    u_h, u_d = c - a, got - a
    print(f"blk{i}: update relL2 device-vs-host {rel(u_d, u_h):.3e}   "
          f"||update||/||s|| {float(u_h.norm()/a.norm()):.3e}")
