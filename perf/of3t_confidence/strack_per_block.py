"""Per-block: is the device s-track a different FORMULA or just bf16 accumulating?

Each device block is fed the HOST block's exact input (s_i, z_i) and its output compared
against the host block's output. No accumulation, so whatever each row shows is that one
block's own arithmetic. An error at the bf16 floor (~2e-03 on LN(s), measured in
strack_precision_arms.py) is precision; anything well above it is a formula difference.
"""
import os, pickle, sys, torch, torch.nn.functional as F, ttnn
sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/perf/of3t_confidence"))
from forward_vs_float64 import ca_walk, rel_l2
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

steps = []          # (s_in, z_in, s_out) per block, all host fp32
orig = OF3ConfidenceHead._host_s_block
def spy(self, s, z, i):
    out = orig(self, s, z, i)
    steps.append((s.clone(), z.clone(), out.clone()))
    return out
OF3ConfidenceHead._host_s_block = spy
head.forward(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, repr_x_pred=ca_walk(N),
             max_atom_per_token_mask=torch.ones(N * 23), use_zij_trunk_embedding=True,
             s_path="host")
OF3ConfidenceHead._host_s_block = orig

up = lambda x: ttnn.from_torch(x.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)
print(f"{'block':>6s} {'s_out relL2':>13s} {'LN(s_out) relL2':>17s}  (bf16 floor on LN(s) ~2.4e-03)")
for i, (s_in, z_in, s_out) in enumerate(steps):
    blk = head.pf.blocks[i]
    s_d, _ = blk(up(s_in), up(z_in))
    got = torch.Tensor(ttnn.to_torch(s_d)).float().reshape(N, 384)
    print(f"{i:6d} {rel_l2(got, s_out):13.3e} "
          f"{rel_l2(F.layer_norm(got, (384,)), F.layer_norm(s_out, (384,))):17.3e}")
