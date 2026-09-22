"""Is the pair-track parameter-gradient miss ours, or is it bf16's?

The device is one bf16 implementation of this backward. This is a SECOND one that shares no
code with it: the float64 reference, arithmetic untouched, with only the two activations
that cross a block boundary rounded to bf16 and back. Everything else -- every reduction,
every accumulation, the whole backward -- stays float64, so what the arm measures is the
REPRESENTATION of the pair and single tracks and nothing else.

If that alone reproduces the device's numbers, the 5.0e-02 bar is meeting bf16's floor on
these tensors and the port is not the reason. If it does not, the device has a defect and
this says so. The shipped OF3 pairformer test set the precedent for the shape of this
control: it established by the same route that no bf16 implementation, device or CPU, can
clear a 0.97 raw-stack-z gate on this checkpoint.
"""
import os, pickle, sys, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward_vs_float64 import ca_walk
import f64_reference as Rf
from grad_device import rel_l2, _BLK

HEADS = ["plddt_logits", "experimentally_resolved_logits", "pae_logits",
         "pde_logits", "distogram_logits"]
S = "/tmp/of3t/of3t-confidence"


def run(round_to):
    sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                    weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"]
    si_input = g["input_embedder_real"]["out"][0].float()
    si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
    N = si_trunk.shape[0]
    repr_x, mask = ca_walk(N), torch.ones(N * 23)
    seed = torch.load(f"{S}/seed.pt")["seed"]
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    head = OF3ConfidenceHead.__new__(OF3ConfidenceHead)
    bins = torch.linspace(3.25, 50.75, 39, dtype=torch.float32)
    head._squared_bins = bins ** 2
    head._upper = torch.cat([head._squared_bins[1:], head._squared_bins.new_tensor([1e8])])
    blocks = Rf.build_blocks(aux)
    for m in blocks:
        for p_ in m.parameters():
            p_.requires_grad_(True)
    for k in list(aux):
        if ".pair_stack." not in k:
            aux[k] = aux[k].double().requires_grad_(True)
    head._w = aux
    st = si_trunk.double().requires_grad_(True)
    zt = zij_trunk.double().requires_grad_(True)
    out = Rf.forward(head, blocks, si_input, st, zt, repr_x, mask, round_to=round_to)
    loss = sum((out[k] * seed[k]).sum() for k in HEADS)
    loss.backward()
    gr = {k: v.grad.clone() for k, v in aux.items()
          if isinstance(v, torch.Tensor) and v.grad is not None}
    for i, m in enumerate(blocks):
        for n, p_ in m.named_parameters():
            if p_.grad is not None:
                gr[f"{_BLK % i}pair_stack.{n}"] = p_.grad.clone()
    return gr, st.grad.clone(), zt.grad.clone()


gb, stb, ztb = run(torch.bfloat16)
rg, rb = torch.load(f"{S}/ref_aux_grad.pt"), torch.load(f"{S}/ref_blk_grad.pt")
ra = torch.load(f"{S}/ref_act_grad.pt")
ref = {k: v for k, v in rg.items() if v is not None}
for i, d in enumerate(rb):
    for n, v in d.items():
        ref[f"{_BLK % i}pair_stack.{n}"] = v
pair, rest = [], []
med = sorted(v.norm().item() for v in ref.values())
med = med[len(med) // 2]
for k, v in gb.items():
    if k not in ref or tuple(v.shape) != tuple(ref[k].shape) or ref[k].norm() < 1e-8 * med:
        continue
    (pair if ".pair_stack." in k else rest).append((k, rel_l2(v, ref[k])))
f = lambda v: (f"max {max(r for _, r in v):.3e} median "
               f"{sorted(r for _, r in v)[len(v)//2]:.3e} over-bar "
               f"{sum(1 for _, r in v if r > 5e-2)}/{len(v)}") if v else "none"
print("Control: float64 arithmetic, bf16 block-boundary activations, same seed and reference")
print(f"  pair-track (z) parameters, {len(pair):3d}: {f(pair)}")
print(f"  everything else,           {len(rest):3d}: {f(rest)}")
print(f"  si_trunk  relL2 {rel_l2(stb, ra['si_trunk']):.4e}")
print(f"  zij_trunk relL2 {rel_l2(ztb, ra['zij_trunk']):.4e}")
w = max(pair + rest, key=lambda x: x[1])
print(f"  worst {w[0]} = {w[1]:.4e}")
