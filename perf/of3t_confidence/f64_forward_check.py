"""Leg 1 of PROTOCOL SS3c: the float64 reference against the forward it differentiates."""
import os, pickle, sys, torch
sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/perf/of3t_confidence"))
import ttnn
from forward_vs_float64 import ca_walk, rel_l2, pcc
import f64_reference as Rf
from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead

sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                weights_only=False)
if hasattr(sd, "state_dict"): sd = sd.state_dict()
aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"]
si_input = g["input_embedder_real"]["out"][0].float()
si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
N = si_trunk.shape[0]
repr_x, mask = ca_walk(N), torch.ones(N * 23)
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
head = OF3ConfidenceHead(aux, dev, ckc)
blocks = Rf.build_blocks(aux)
ref = Rf.forward(head, blocks, si_input, si_trunk, zij_trunk, repr_x, mask)
dv = head.forward(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
                  repr_x_pred=repr_x, max_atom_per_token_mask=mask,
                  use_zij_trunk_embedding=True, s_path="device")
ho = head.forward(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
                  repr_x_pred=repr_x, max_atom_per_token_mask=mask,
                  use_zij_trunk_embedding=True, s_path="host")
print(f"{'tensor':34s} {'device vs f64':>16s} {'host vs f64':>16s}")
for k in ["plddt_logits", "experimentally_resolved_logits", "pae_logits", "pde_logits",
          "distogram_logits", "si_conf", "zij_conf"]:
    r = ref[k].float()
    print(f"  {k:32s} {rel_l2(dv[k].float(), r):16.3e} {rel_l2(ho[k].float(), r):16.3e}")
