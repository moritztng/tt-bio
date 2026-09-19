"""PROTOCOL SS3d: the confidence head's gradients, per parameter, against float64.

The reference is `f64_reference.py`, whose forward is already validated against the device
forward it differentiates (f64_forward_check.py) and whose gradient is validated here
against its own float64 central finite differences before anything is compared to it.

The loss is a fixed inner product <seed, logits> summed over the five heads, with the SAME
seed on both stacks. A loss rather than a single output because SS3d is a per-parameter
claim and one output reaches only part of the tree; a fixed seed rather than a real label
because the quantity under test is d(loss)/d(parameter), and a seed that excites every
output channel is the probe LEDGER K28 says to use. Which real loss terms fire is a
separate question and COVERAGE answers it.

Bijection, PROTOCOL SS3a. Every device weight is a transform of one of THEIR tensors and
the comparison is made in THEIR space, so the transform is inverted rather than applied:

  * a Linear uploads ``W.t()``            -> grad_W   = grad_dev.t()
  * the fused qkv uploads cat([q,k,v]) reshaped to heads, zero-padded 24->32, flattened,
    then transposed                       -> unpad and split, exactly
  * the pair-bias projection uploads ``W.t() * sqrt(head_dim)``
                                          -> grad_W   = grad_dev.t() * sqrt(head_dim)
  * a LayerNorm gain/bias uploads as-is   -> grad      = grad_dev
  * TriangleMultiplication's p_in/g_in are already fused in OF3's own checkpoint, so they
    are split on dim 0 and each half reported separately (SS3a: a fused comparison lets
    one half's agreement mask the other half's error).
"""
import math, os, pickle, sys, torch, ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward_vs_float64 import ca_walk
import f64_reference as Rf
from tt_bio import autograd as ag
from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead

HEADS = ["plddt_logits", "experimentally_resolved_logits", "pae_logits",
         "pde_logits", "distogram_logits"]
_BLK = "pairformer_embedding.pairformer_stack.blocks.%d."


def rel_l2(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def seeds_for(shapes, gen):
    return {k: torch.randn(s, generator=gen, dtype=torch.float64) for k, s in shapes.items()}


def main():
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
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    head = OF3ConfidenceHead(aux, dev, ckc)
    blocks = Rf.build_blocks(aux)

    # ---- reference: autograd, and its finite-difference validation ------------------
    gen = torch.Generator().manual_seed(11)
    st = si_trunk.double().requires_grad_(True)
    zt = zij_trunk.double().requires_grad_(True)
    for m in blocks:
        for p in m.parameters():
            p.requires_grad_(True)
    # Every parameter the REFERENCE actually reads. The z-track lives in the reference
    # blocks (loaded from the remap), the s-track and the heads are read out of `aux` by
    # the host block and the head code, so `.pair_stack.` is the one prefix to skip here
    # or a tensor would carry two leaves and neither would see the whole gradient.
    for k in list(aux):
        if ".pair_stack." not in k:
            aux[k] = aux[k].double().requires_grad_(True)
    head._w = aux

    def ref_loss(seed):
        out = Rf.forward(head, blocks, si_input, st, zt, repr_x, mask)
        return sum((out[k] * seed[k]).sum() for k in HEADS), out

    out0 = Rf.forward(head, blocks, si_input, st, zt, repr_x, mask)
    seed = {k: torch.randn(out0[k].shape, generator=gen, dtype=torch.float64) for k in HEADS}
    loss, _ = ref_loss(seed)
    loss.backward()
    print(f"reference loss {loss.item():.6f}")

    # SS3c: central finite differences through the reference's own forward, along sign(g)
    # (K28 -- a peaked gradient makes a raw direction read the tolerance, not the model).
    print("\nfinite-difference validation of the reference (float64, central, d=sign(g)):")
    fd_probes = [
        ("blocks.0.pair_stack.tri_mul_out.p_in.weight", blocks[0].tri_mul_out.p_in.weight),
        ("blocks.3.pair_stack.tri_att_end.mha.linear_q.weight",
         blocks[3].tri_att_end.mha.linear_q.weight),
        ("blocks.3.attn_pair_bias.mha.linear_q.weight",
         aux[_BLK % 3 + "attn_pair_bias.mha.linear_q.weight"]),
        ("blocks.1.single_transition.swiglu.linear_a.weight",
         aux[_BLK % 1 + "single_transition.swiglu.linear_a.weight"]),
        ("plddt.linear.weight", aux["plddt.linear.weight"]),
        ("pae.linear.weight", aux["pae.linear.weight"]),
        ("pairformer_embedding.linear_i.weight",
         aux["pairformer_embedding.linear_i.weight"]),
        ("pairformer_embedding.linear_distance.weight",
         aux["pairformer_embedding.linear_distance.weight"]),
        ("si_trunk (activation, the trunk connection)", st),
        ("zij_trunk (activation, the trunk connection)", zt),
    ]
    for name, leaf in fd_probes:
        d = torch.sign(leaf.grad)
        analytic = float((leaf.grad * d).sum())
        fd = []
        for eps in (1e-4, 1e-5, 1e-6):
            with torch.no_grad():
                leaf += eps * d
            lp, _ = ref_loss(seed)
            lp = lp.item()
            with torch.no_grad():
                leaf -= 2 * eps * d
            lm, _ = ref_loss(seed)
            lm = lm.item()
            with torch.no_grad():
                leaf += eps * d
            fd.append((lp - lm) / (2 * eps))
        best = min(fd, key=lambda f: abs(f - analytic))
        print(f"  {name:44s} analytic {analytic:+.8e}  fd {best:+.8e}  "
              f"rel {abs(best - analytic) / abs(analytic):.3e}")
    torch.save({"seed": seed}, "/tmp/of3t/of3t-confidence/seed.pt")
    torch.save({k: (v.grad.clone() if v.grad is not None else None)
                for k, v in aux.items() if isinstance(v, torch.Tensor) and v.requires_grad},
               "/tmp/of3t/of3t-confidence/ref_aux_grad.pt")
    torch.save([{n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
                for m in blocks], "/tmp/of3t/of3t-confidence/ref_blk_grad.pt")
    torch.save({"si_trunk": st.grad.clone(), "zij_trunk": zt.grad.clone()},
               "/tmp/of3t/of3t-confidence/ref_act_grad.pt")
    print("\nreference gradients saved")


if __name__ == "__main__":
    main()
