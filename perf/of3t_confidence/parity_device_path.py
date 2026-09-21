"""Device-path parity for the OF3 confidence heads, against the real reference golden.

Runs the SAME golden the shipped test gates on (tests/test_openfold3_confidence.py),
once per s-path, and prints PCC per head so the host->device move is a measurement
rather than an assertion.
"""
import os, sys, torch, ttnn
sys.path.insert(0, os.path.expanduser("~/.coworker/wt/of3t-confidence/tests"))
import of3_golden

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
HEADS = ["plddt_logits", "experimentally_resolved_logits", "pae_logits",
         "pde_logits", "distogram_logits", "si_conf", "zij_conf"]


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def main():
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    g = of3_golden.intermediates(_GOLD)["confidence_heads_real"]
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    head = OF3ConfidenceHead(aux, dev, ckc)
    print(f"N_tok={g['si_trunk'].shape[0]}  si_trunk absmax={float(g['si_trunk'].abs().max()):.1f}")
    for path in ("host", "device"):
        out = head.forward(
            si_input=g["si_input"].float(), si_trunk=g["si_trunk"].float(),
            zij_trunk=g["zij_trunk"].float(), repr_x_pred=g["repr_x_pred"].float(),
            max_atom_per_token_mask=g["max_atom_per_token_mask"].float(),
            use_zij_trunk_embedding=g["use_zij_trunk_embedding"], s_path=path)
        print(f"\n--- s_path={path} ---")
        for h in HEADS:
            print(f"  {h:34s} PCC={pcc(out[h].float(), g[h].float()):.6f} "
                  f"shape={tuple(out[h].shape)}")


if __name__ == "__main__":
    main()
