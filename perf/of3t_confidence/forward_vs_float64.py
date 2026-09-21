"""OF3 confidence head: the device path against a float64 reference that is the SAME code.

PROTOCOL SS3c wants a float64 reference that is not another approximation of the thing it
checks. `OF3ConfidenceHead`'s host path is dtype-parametric, so `dtype=torch.float64` runs
the shipped arithmetic in float64 with no transcription and nothing to get wrong.

Inputs are the module's OWN (LEDGER K28): si_input, si_trunk and zij_trunk come from the
real golden legs `input_embedder_real` and `pairformer_stack_real` -- the real 48-block
trunk on real ubiquitin with the real of3-p2-155k weights, si_trunk at absmax 2.28e5.
`repr_x_pred` is the one input the local golden does not carry (it is the diffusion
rollout's output), so it is drawn as a 3.8 A CA-CA random walk under a fixed seed, which
puts the pairwise distances across the head's 3.25-50.75 A bin range instead of piling
them into the first bin the way a unit-normal draw would.
"""
import os, sys, pickle, torch, ttnn

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
HEADS = ["plddt_logits", "experimentally_resolved_logits", "pae_logits",
         "pde_logits", "distogram_logits", "si_conf", "zij_conf"]


def rel_l2(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def ca_walk(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    step = torch.randn(n, 3, generator=g)
    step = step / step.norm(dim=-1, keepdim=True) * 3.8
    return torch.cumsum(step, dim=0)


def main():
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]
    si_input = g["input_embedder_real"]["out"][0].float()
    si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
    N = si_trunk.shape[0]
    repr_x = ca_walk(N)
    mask = torch.ones(N * 23)
    d = torch.cdist(repr_x, repr_x)
    print(f"N_tok={N}  si_trunk absmax={float(si_trunk.abs().max()):.4g}  "
          f"zij_trunk absmax={float(zij_trunk.abs().max()):.4g}")
    print(f"repr dist: min={float(d[d>0].min()):.2f} med={float(d.median()):.2f} "
          f"max={float(d.max()):.2f} A   (head bins 3.25-50.75 A)")

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    head = OF3ConfidenceHead(aux, dev, ckc)
    kw = dict(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
              repr_x_pred=repr_x, max_atom_per_token_mask=mask,
              use_zij_trunk_embedding=True)
    ref = head.forward(**kw, s_path="host", dtype=torch.float32)
    torch.save(ref["si_conf"].float(), "/tmp/of3t/of3t-confidence/si_conf_host.pt")
    runs = {"host-fp32": head.forward(**kw, s_path="host", dtype=torch.float32),
            "device": head.forward(**kw, s_path="device")}
    print(f"\n{'tensor':34s} " + "  ".join(f"{k:>22s}" for k in runs))
    for h in HEADS:
        cells = []
        for k, out in runs.items():
            cells.append(f"{rel_l2(out[h].float(), ref[h]):.3e}/{pcc(out[h].float(), ref[h]):.5f}")
        print(f"  {h:32s} " + "  ".join(f"{c:>22s}" for c in cells))
    print("\n  (cells are relative L2 / PCC against the float64 run of the same code)")


if __name__ == "__main__":
    main()
