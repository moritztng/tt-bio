"""Is the pLDDT error in the s-track or in the device head? Recompute the head on host
from the DEVICE s and compare. If the host head on device-s reproduces the device
number, the s-track owns it; if it does not, the head does."""
import os, pickle, torch, torch.nn.functional as F, ttnn, sys

CK = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
G = os.path.expanduser("~/of3_ref_out.pkl")
sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/perf/of3t_confidence"))
from forward_vs_float64 import ca_walk, rel_l2

from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead

sd = torch.load(CK, map_location="cpu", weights_only=False)
if hasattr(sd, "state_dict"): sd = sd.state_dict()
aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
g = pickle.load(open(G, "rb"))["intermediates"]
si_input = g["input_embedder_real"]["out"][0].float()
si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
N = si_trunk.shape[0]
kw = dict(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
          repr_x_pred=ca_walk(N), max_atom_per_token_mask=torch.ones(N * 23),
          use_zij_trunk_embedding=True)
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
head = OF3ConfidenceHead(aux, dev, ckc)
h = head.forward(**kw, s_path="host")
d = head.forward(**kw, s_path="device")
sh, sd_ = h["si_conf"].float(), d["si_conf"].float()
print(f"si_conf            relL2 {rel_l2(sd_, sh):.3e}")
lnh, lnd = F.layer_norm(sh, (384,)), F.layer_norm(sd_, (384,))
print(f"LN(si_conf)        relL2 {rel_l2(lnd, lnh):.3e}   <- what the atom heads read")
for name in ("plddt", "experimentally_resolved"):
    w = aux[f"{name}.layer_norm.weight"].float()
    b = aux.get(f"{name}.layer_norm.bias")
    b = b.float() if b is not None else 0.0
    L = aux[f"{name}.linear.weight"].float()
    oh_, od_ = F.linear(lnh * w + b, L), F.linear(lnd * w + b, L)
    key = "plddt_logits" if name == "plddt" else "experimentally_resolved_logits"
    print(f"{name:24s} host-head on device-s relL2 {rel_l2(od_, oh_):.3e}   "
          f"| device-head relL2 {rel_l2(d[key].float(), h[key].float()):.3e}")
